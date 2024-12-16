import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(
        self, 
        embedding_size, 
        nhead, 
    ):
        """
            :param embedding_size: dimension of word embedding (d_model in paper)
            :param nhead: number of heads in multi-head attentions (in papers, nhead=8)
        """
        super(SelfAttention, self).__init__()
        
        self.embedding_size = embedding_size
        self.nhead = nhead
        self.d_model = embedding_size // nhead
        
        assert (self.d_model * self.nhead == self.embedding_size), "Embedding size must be div by n_heads"
        
        self.values = nn.Linear(
            in_features=self.d_model,
            out_features=self.d_model,
            bias=False
        )
        self.keys = nn.Linear(
            in_features=self.d_model,
            out_features=self.d_model,
            bias=False
        )
        self.queries = nn.Linear(
            in_features=self.d_model,
            out_features=self.d_model,
            bias=False
        )
        
        self.fc_out = nn.Linear(
            in_features=self.nhead * self.d_model,
            out_features=self.embedding_size
        )
        
    def forward(self, values, keys, queries, mask):
        N = queries.shape[0] # batch_size
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.nhead, self.d_model)
        keys = keys.reshape(N, key_len, self.nhead, self.d_model)
        queries = queries.reshape(N, query_len, self.nhead, self.d_model)
        
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, n_heads, d_model)
        # keys shape: (N, key_len, n_heads, d_model)
        # energy shape: (N, n_heads, query_len, key_len)
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
            # when the enery is approxminus inf, 
            # the value of energy go through softmax layer will be approx 0    
        
        attention = torch.softmax(energy / (self.embedding_size ** 0.5), dim=3)
        # attention shape: (N, nhead, query_len, key_len)
        
        # attention shape: (N, nhead, query_len, key_len)
        # values shape: (N, value_len, nhead, d_model)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])
        # output shape: (N, query_len, nhead, d_model)
        
        # concat all attention head
        out = out.reshape(N, query_len, self.nhead * self.d_model)
        # out shape: (N, query_len, embedding_size)
        
        out = self.fc_out(out)
        return out
        
class TransformerBlock(nn.Module):
    def __init__(
        self, 
        embedding_size, 
        nhead, 
        dropout, 
        forward_expansion
    ):
        super(TransformerBlock, self).__init__()
        
        self.attention = SelfAttention(
            embedding_size=embedding_size,
            nhead=nhead
        )
        self.norm1 = nn.LayerNorm(
            normalized_shape=embedding_size
        )
        self.norm2 = nn.LayerNorm(
            normalized_shape=embedding_size
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(
                in_features=embedding_size,
                out_features=forward_expansion*embedding_size,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=forward_expansion*embedding_size,
                out_features=embedding_size
            )
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        
        x = self.norm1(attention + query)
        x = self.dropout(x)
        
        forward = self.feed_forward(x)
        
        out = self.norm2(forward + x)
        out = self.dropout(out)
        
        return out
    
class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embedding_size,
        num_layers,
        nhead,
        device,
        forward_expansion,
        dropout,
        max_length
    ):
        super(Encoder, self).__init__()
        self.embedding_size = embedding_size
        self.device = device
        self.word_embedding = nn.Embedding(
            num_embeddings=src_vocab_size, 
            embedding_dim=self.embedding_size
        )
        
        self.position_embedding = nn.Embedding(
            num_embeddings=max_length, 
            embedding_dim=self.embedding_size
        )
        
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    self.embedding_size,
                    nhead=nhead,
                    dropout=dropout,
                    forward_expansion=forward_expansion
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        
        x = self.word_embedding(x) + self.position_embedding(positions)
        out = self.dropout(x)
        
        for layer in self.layers:
            out = layer(out, out, out, mask)
            
        return out
    
class DecoderBlock(nn.Module):
    def __init__(
        self, 
        embedding_size, 
        nhead, 
        forward_expansion, 
        dropout, 
        device
    ):
        """
        attn -> layer_norm -> transformer block
        """
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(
            embedding_size=embedding_size,
            nhead=nhead,
        )
        self.norm = nn.LayerNorm(embedding_size)
        self.transformer_block = TransformerBlock(
            embedding_size=embedding_size,
            nhead=nhead,
            dropout=dropout,
            forward_expansion=forward_expansion
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_value, enc_key, src_mask, tgt_mask):
        """
        src_mask and tgt_mask are to hide padding values
        value and key come from the encoder output
        """
        # extract query from the input of decoder
        attention = self.attention(
            values=x,
            keys=x,
            queries=x,
            mask=tgt_mask
        )
        dec_query = self.norm(attention + x)
        dec_query = self.dropout(dec_query)
        
        # pass extracted query of decoder input, value and key from encoder
        # to the transformer block
        out = self.transformer_block(
            value=enc_value, 
            key=enc_key, 
            query=dec_query, 
            mask=src_mask
        )
        return out
    
class Decoder(nn.Module):
    def __init__(
        self,
        tgt_vocab_size,
        embedding_size,
        num_layers,
        nhead,
        forward_expansion,
        dropout,
        device,
        max_length,
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(
            num_embeddings=tgt_vocab_size,
            embedding_dim=embedding_size
        )
        self.position_embedding = nn.Embedding(
            num_embeddings=max_length,
            embedding_dim=embedding_size
        )
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    embedding_size=embedding_size,
                    nhead=nhead,
                    forward_expansion=forward_expansion,
                    dropout=dropout,
                    device=device
                )
                for _ in range (num_layers)
            ]
        )
        self.fc_out = nn.Linear(
            in_features=embedding_size,
            out_features=tgt_vocab_size
        )
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x, enc_out, src_mask, tgt_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.word_embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(
                x=x,
                enc_value=enc_out,
                enc_key=enc_out,
                src_mask=src_mask,
                tgt_mask=tgt_mask
            )
            
        out = self.fc_out(x)
        return out
        
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        src_pad_idx,
        tgt_pad_idx,
        embedding_size=512,
        num_layers=6,
        forward_expansion=4,
        nhead=8,
        dropout=0,
        device="cuda",
        max_length=128
    ):
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(
            src_vocab_size=src_vocab_size,
            embedding_size=embedding_size,
            num_layers=num_layers,
            nhead=nhead,
            device=device,
            forward_expansion=forward_expansion,
            dropout=dropout,
            max_length=max_length
        )
        
        self.decoder = Decoder(
            tgt_vocab_size=tgt_vocab_size,
            embedding_size=embedding_size,
            num_layers=num_layers,
            nhead=nhead,
            forward_expansion=forward_expansion,
            dropout=dropout,
            device=device,
            max_length=max_length
        )
        
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # src_mask.shape = (N, 1, 1, src_len)
        return src_mask.to(self.device)
    
    def make_tgt_mask(self, tgt):
        N, tgt_len = tgt.shape
        tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len)).expand(
            N, 1, tgt_len, tgt_len
        )
        return tgt_mask.to(self.device)
    
    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(tgt, enc_src, src_mask, tgt_mask)
        
        return out
    
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(
        device
    )
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device).to(
        device
    )
    out = model(x, trg[:, :-1])
    print(out.shape)
    
    transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
    src = torch.rand((10, 32, 512))
    tgt = torch.rand((20, 32, 512))
    out = transformer_model(src, tgt)
    print(out.shape)