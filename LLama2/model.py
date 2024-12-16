# fmt: off

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32

    n_heads: int = 32  # number of heads for the query
    n_kv_heads: Optional[int] = None  # number of heads for the key and value
    # because of grouped query attention, we don't need to have the sane number of heads for the query and key-value
    vocab_size: int = -1  # this will be set by the tokenizer
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # parameters for KV cache
    max_batch_size: int = 32
    max_seq_length: int = 2048
    
    # GPU parameters
    device: str = None


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device:str, theta: float=10000.0):
    assert head_dim % 2 == 0, "As mentioned in the paper, the dimension of the embedding must be even"
    
    # build theta parameters: theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ..., dim / 2], shape = (head_dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    
    # build m (position parameter), shape = (seq_len)
    m = torch.arange(seq_len, device=device)
    
    # Multiply each theta by each position using the outer product, shape = (seq_len, head_dim / 2)
    freqs = torch.outer(m, theta).float()
    
    # Compute complex number in the polar form: c = R * exp(i * m * tehta) where R = 1, shape = (seq_len, head_dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    
    return freqs_complex


def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # (batch_size, seq_len, n_heads, head_size) -> (batch_size, seq_len, n_heads, head_size // 2)
    # group into blocks of every two dimension across head_size
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    
    # Change shape of freqs_complex
    # (seq_len, head_dim // 2) -> (1, seq_len, 1, head_dim // 2)
    freqs_complex = freqs_complex.unsqueeze(-2).unsqueeze(0)
    
    # Multiply the complex vectors with broadcasting
    x_rotated = x_complex * freqs_complex
    
    # Convert to normal vectors but in block of 2
    # (batch_size, seq_len, n_heads, head_dim // 2) -> (batch_size, seq_len, n_heads, head_dim // 2, 2)
    x_out = torch.view_as_real(x_rotated)
    
    # Convert to the original format
    x_out = x_out.reshape(*x.size())
    
    return x_out.type_as(x).to(device)


class RMSNorm(nn.Module):
    """Paper: https://arxiv.org/pdf/1910.07467"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        
        # gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))
        
    def _norm(self, x: torch.Tensor):
        # x ~ (batch_size, seq_len, dim)
        # rsqrt: calculate 1 / sqrt(x) ~ (batch_size, seq_len, 1)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor):
        return (self._norm(x.float()) * self.weight).type_as(x)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        return (
            x.unsqueeze(3)
            .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
            .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
        )

class SelfAttention(nn.Module):
    def __init__(self, args:ModelArgs):
        super().__init__()
        
        # The number of heads for the Key and Value
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        
        # The number of heads for the Query
        self.n_heads_q = args.n_heads
        
        # Hoow many times each Key-Value head should be repeated to match the head of Query
        self.n_rep = self.n_heads_q // self.n_kv_heads
        
        # The dimension of each head
        self.head_dim = args.dim // args.n_heads
        
        self.wq = nn.Linear(args.dim, self.n_heads_q * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        
        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_length, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_length, self.n_kv_heads, self.head_dim))
        
    def forward(self, x: torch.Tensor, start_pos:int, freqs_complex: torch.Tensor):
        # (batch_size, seq_len=1, dim)
        batch_size, seq_len, dim = x.shape
        
        # Apply the W^Q, W^K, W^V to the query, key, value
        # xq ~ (batch_size, 1, n_heads_q * head_dim)
        # xk, xv ~ (batch_size, 1, n_kv_heads * head_dim)
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        # Split the heads
        # xq: (batch_size, 1, n_heads_q * head_dim) -> (batch_size, 1, n_heads_q, head_dim)
        # xk, xv: (batch_size, 1, n_kv_heads * head_dim) -> (batch_size, 1, n_kv_heads, head_dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        # Apply rotary embedding to Query and Key
        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)
        
        # Add KV to cache
        self.cache_k[:batch_size, start_pos:start_pos+seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos+seq_len] = xv
        
        # Retrieve all the cached Key and Value up to current_position (start_pos+seq_len)
        # K, V ~ (batch_size, seq_len_kv, n_kv_heads, head_dim)
        keys = self.cache_k[:batch_size, 0:start_pos+seq_len]
        values = self.cache_v[:batch_size, 0:start_pos+seq_len]
        
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)
        
        # Move n_heads dimension before the seq_len dimension because each head 
        #   watch all the sequence but the corresponding head embedding of each token
        # (batch_size, seq_len, n_heads, head_dim) -> (batch_size, n_heads, seq_len, head_dim)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # QK^T / sqrt(d_k)
        # (batch_size, n_heads, seq_len, head_dim) @ (batch_size, n_heads, head_dim, seq_len_kv) 
        #   -> (batch_size, n_heads, seq_len, seq_len_kv)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # softmax(QK^T / sqrt(d_k)) ~ (batch_size, n_heads, seq_len, seq_len_kv)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        
        # (batch_size, n_heads, seq_len, seq_len_kv) @ (batch_size, n_heads, seq_len_kv, head_dim)
        #   -> (batch_size, n_heads, seq_len, head_dim)
        output = torch.matmul(scores, values)
        
        # Move n_heads dimension and bring back seq_len
        # (batch_size, n_heads, seq_len, head_dim) -> (batch_size, seq_len, n_heads, head_dim)
        output = output.transpose(1, 2).contiguous()
        
        # Concat all the head to the dim
        # (batch_size, seq_len, n_heads, head_dim) -> (batch_size, seq_len, dim)
        output = output.view(batch_size, seq_len, -1)
        
        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(self, args:ModelArgs) -> None:
        super().__init__()
        
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        
        # Round the hidden_dim to the nearest multiple of the multiple_of parameters
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
        
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)
    
    def forward(self, x: torch.Tensor):
        """FFN_SwiGLU(x, W, V, W2) = (Swish(xW) \times xV) W2"""
        swish = F.silu(self.w1(x))
        x_V = self.w3(x)
        x = swish * x_V
        x = self.w2(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        
        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)
        
        # pre-attention norm
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        
        # pre-ffn norm
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        
    def forward(self, x:torch.Tensor, start_pos:int, freqs_compex: torch.Tensor):
        # x ~ (batch_size, seq_len, dim)
        # skip connection + pre-attention norm + attention
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_compex)
        
        out = h + self.feed_forward(self.ffn_norm(h))
        
        return out


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        assert args.vocab_size != -1

        self.args = args
        self.n_layers = args.n_layers
        self.vocab_size = args.vocab_size
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)
        
        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))
        
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)


        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_length * 2, device=self.args.device)


    def forward(self, tokens: torch.Tensor, start_pos: int):
        # for KV cached, just compute the QKV of new token
        
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token at a time can be processed"
        
        # Going through token embedding layer to
        h = self.tok_embeddings(tokens)
        
        # Retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]
        
        # Consecutively apply all the encoder layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output
