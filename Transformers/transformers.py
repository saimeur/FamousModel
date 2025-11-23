import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
            ), f"Embed size cannot be divided by the number of heads. embed_size : {embed_size}, heads : {heads}"

        self.queries = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.values = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, query, key, value, mask=None):
        assert query.shape[2] == self.embed_size
        assert query.shape[2] == key.shape[2]
        assert query.shape[2] == value.shape[2]

        N = query.shape[0]

        query_len, key_len, value_len = query.shape[1], key.shape[1], value.shape[1]
        
        query = self.queries(query)  # (N, query_len, embed_size)
        value = self.values(value)  # (N, value_len, embed_size)
        key = self.keys(key)  # (N, key_len, embed_size)

        # Split the embedding into self.heads different pieces
        query = query.reshape(N, query_len, self.heads, self.head_dim)
        value = value.reshape(N, value_len, self.heads, self.head_dim)
        key = key.reshape(N, key_len, self.heads, self.head_dim)

        energy = torch.einsum('nqhd, nkhd -> nhqk', query, key) / torch.sqrt(self.embed_size)
        # query shape : (N, query_len, heads, head_dim)
        # key shape : (N, key_len, heads, head_dim)
        # value shape : (N, value_len, heads, head_dim)

        if mask is not None:
            energy.masked_fill(mask == 0, float("-1e20"))

        score = torch.softmax(energy, dim=-1)
        # score shape : (N, head, query_len, key_len)

        output = torch.einsum('nhqk, nkhd -> nqhd', score, value)
        # output shape : (N, query_len, heads, head_dim)
        
        head = score.reshape(N, query_len, self.embed_size)

        output = self.fc_out(head)

        return output
    

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()

        self.layer_norm1 = nn.LayerNorm(embed_size)
        self.layer_norm2 = nn.LayerNorm(embed_size)

        self.attention = SelfAttention(embed_size, heads)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size*forward_expansion),
            nn.ReLU(),
            nn.Linear(embed_size*forward_expansion, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        attention = self.attention(query, key, value, mask)

        x = self.dropout(self.layer_norm1(query + attention))
        forward = self.feed_forward(x)
        out = self.dropout(self.layer_norm2(forward + x))
        return out
    

class Encoder(nn.Module):
    def __init__(self,
                 vocab_dim, 
                 num_layer,
                 embed_size, 
                 heads, 
                 dropout, 
                 forward_expansion,
                 max_length,
                 device):
        
        super(Encoder, self).__init__()
        self.device = device
        self.embed_size = embed_size

        self.words_embedding = nn.Embedding(vocab_dim, embed_size)
        self.positional_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout,
                    forward_expansion
                )
                for _ in range(num_layer)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        N, seq_len = x.shape[0], x.shape[1]
        positions = torch.arange(seq_len, device=self.device).expand(N, seq_len)    # (N, seq_len)
        out = self.dropout(
            self.words_embedding(x) + self.positional_embedding(positions)
        )

        for layer in self.layers:
            out = layer(query=out, key=out, value=out, mask=mask)

        return out
    

class Decoder(nn.Module):
    def __init__(
            self,
            vocab_dim,
            num_layer,
            max_length,
            embed_size,
            heads,
            dropout,
            forward_expansion,
            device
            ):
        super(Decoder, self).__init__()
        self.max_length = max_length
        self.device = device

        self.words_embedding = nn.Embedding(vocab_dim, embed_size)
        self.positionnal_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout,
                    forward_expansion
                )
                for _ in range(num_layer)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        N, seq_len = x.shape[0], x.shape[1]
        positions = torch.range(seq_len, device=self.device).expand(N, seq_len)
        out = self.dropout(
            self.words_embedding(x) + self.positionnal_embedding(positions)
        )
        