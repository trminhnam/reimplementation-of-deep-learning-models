import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.d_model = d_model

        # create a matrix of shape (max_len, d_model) filled with zeros
        pe = torch.zeros(max_len, d_model)
        pe.require_grad = False
        pe = self.get_positional_encoding(pe)

        self.register_buffer("pe", pe)

    def get_positional_encoding(self, pe: torch.Tensor):
        position = torch.arange(0, pe.size(0)).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / self.d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def forward(self, x: torch.Tensor):
        return x + self.pe[: x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, is_causal=False):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model should be divisible by n_heads"

        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_model // n_heads
        self.is_causal = is_causal

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask=None):
        # Shape of query, key, value: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = query.size()

        # Linear projections to get the Q, K, V matrices
        q: torch.Tensor = self.W_q(query)
        k: torch.Tensor = self.W_k(key)
        v: torch.Tensor = self.W_v(value)

        # Split the d_model dimension of Q, K and V into n_heads
        q = q.reshape(batch_size, seq_len, self.n_heads, self.d_head)
        k = k.reshape(batch_size, seq_len, self.n_heads, self.d_head)
        v = v.reshape(batch_size, seq_len, self.n_heads, self.d_head)

        # Transpose to get the shape: (batch_size, n_heads, seq_len, d_head)
        # Because we use the two last dimensions as the matrix multiplication dimensions
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute the dot product attention
        k_T = k.transpose(-1, -2)
        e = torch.matmul(q, k_T) / (self.d_head**2)

        # Handle padding mask
        if mask is not None:
            padding_mask = self.get_padding_mask(mask, e.device)
            e = e + padding_mask

        # If this is a decoder (cross attention), add causal mask to avoid seeing the future
        if self.is_causal:
            causal_mask = self.get_causal_mask(seq_len, e.device)
            e = e + causal_mask

        # Compute the attention weights in the shape (batch_size, n_heads, seq_len, seq_len)
        a = F.softmax(e, dim=-1)

        # Compute the context vector
        c = torch.matmul(a, v)

        # Convert the context vector back to the shape (batch_size, seq_len, n_heads, d_head),
        #   then reshape to (batch_size, seq_len, d_model)
        c = c.transpose(1, 2)
        c = c.reshape(batch_size, seq_len, self.d_model)

        # Linear projection to get the output
        o = self.W_o(c)

        return o

    def get_causal_mask(self, seq_len: int, device: str):
        # Return a causal mask in the shape (1, 1, seq_len, seq_len)
        # where the upper triangular part is 0 and the lower triangular part is -inf
        trill = torch.tril(torch.ones(seq_len, seq_len))
        causal_mask = torch.zeros(seq_len, seq_len).float()
        causal_mask = causal_mask.masked_fill(trill == 0, float("-inf"))
        return causal_mask.unsqueeze(0).unsqueeze(0).to(device)

    def get_padding_mask(self, mask: torch.Tensor, device: str):
        # Padding mask is in the shape (batch_size, seq_len), where 1 indicates normal token and 0 indicates the padding token
        # Return a mask in the shape (batch_size, n_heads, seq_len, seq_len) where normal token is 0 and padding token is -inf
        mask = mask.unsqueeze(1).unsqueeze(1)
        padding_mask = torch.zeros_like(mask).float()
        padding_mask = padding_mask.masked_fill(mask.long() == 0, float("-inf"))
        return padding_mask.to(device)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor):
        return self.linear2(F.relu(self.linear1(x)))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout) -> None:
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, is_decoder=False)
        self.self_attn_norm = nn.LayerNorm(d_model)

        self.feedforward = FeedForward(d_model=d_model, d_ff=d_ff)
        self.ff_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # Calculate self-attention
        self_attn_output = self.self_attn(x, x, x, mask)
        self_attn_output = self.dropout(self_attn_output)
        self_attn_output = self.self_attn_norm(self_attn_output + x)

        # Feedforward
        ff_output = self.feedforward(self_attn_output)
        ff_output = self.dropout(ff_output)
        ff_output = self.ff_norm(ff_output + self_attn_output)

        return ff_output


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout) -> None:
        super().__init__()

        self.masked_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, is_causal=True)
        self.masked_attn_norm = nn.LayerNorm(d_model)

        self.cross_attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads, is_causal=False)
        self.cross_attn_norm = nn.LayerNorm(d_model)

        self.feedforward = FeedForward(d_model=d_model, d_ff=d_ff)
        self.ff_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        encoder_mask: torch.Tensor = None,
        decoder_mask: torch.Tensor = None,
    ):
        # Calculate query with mask-attention
        masked_attn_output = self.masked_attn(query, query, query, decoder_mask)
        masked_attn_output = self.dropout(masked_attn_output)
        masked_attn_output = self.masked_attn_norm(masked_attn_output + query)

        # Calculate query with cross-attention
        cross_attn_output = self.cross_attention(masked_attn_output, key, value, encoder_mask)
        cross_attn_output = self.dropout(cross_attn_output)
        cross_attn_output = self.cross_attn_norm(cross_attn_output + masked_attn_output)

        # Feedforward
        ff_output = self.feedforward(cross_attn_output)
        ff_output = self.dropout(ff_output)
        ff_output = self.ff_norm(ff_output + cross_attn_output)

        return ff_output


class GPTLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout) -> None:
        super().__init__()
        self.masked_attn = MultiHeadAttention(d_model=d_model, n_heads=n_heads, is_causal=True)
        self.masked_attn_norm = nn.LayerNorm(d_model)

        self.feedforward = FeedForward(d_model=d_model, d_ff=d_ff)
        self.ff_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, input_mask: torch.Tensor = None):
        # Calculate self-attention
        self_attn_output = self.masked_attn(x, x, x, input_mask)
        self_attn_output = self.dropout(self_attn_output)
        self_attn_output = self.masked_attn_norm(self_attn_output + x)

        # Feedforward
        ff_output = self.feedforward(self_attn_output)
        ff_output = self.dropout(ff_output)
        ff_output = self.ff_norm(ff_output + self_attn_output)

        return ff_output


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, n_layers, n_heads, d_ff, dropout):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList(
            [GPTLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.lm_head = nn.Linear(d_model, vocab_size)

        # Weight tying between the token_embedding weights and the language model head
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, x: torch.Tensor, y: torch.Tensor = None, mask: torch.Tensor = None):
        x = self.token_embedding(x)
        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.lm_head(x)

        if y is not None:
            y = y.view(-1)
            x = x.view(-1, x.size(-1))
            loss = F.cross_entropy(x, y)
            return (loss, x)
        else:
            return x

    @torch.no_grad()
    def generate(self, x: torch.Tensor, max_new_tokens: int, mask: torch.Tensor = None):
        for _ in range(max_new_tokens):
            logits = self.forward(x, mask)
            probs = F.softmax(logits[:, -1], dim=-1)
            new_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, new_token], dim=-1)
        return x


if __name__ == "__main__":
    # Test the MultiHeadAttention module
    batch_size, seq_len, d_model = 2, 4, 8
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)
    mask = torch.tensor([[1, 0, 1, 0], [1, 1, 1, 1]])

    mha = MultiHeadAttention(d_model=d_model, n_heads=2, is_decoder=True)
    output = mha(query, key, value, mask)
    print(output.shape)
