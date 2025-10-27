import torch
import torch.nn as nn
import math


class AbsolutePositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(AbsolutePositionalEncoding, self).__init__()
        self.embed_size = embed_size

        # Create position embeddings
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() *
                           (-math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, embed_size)
        Returns:
            x + positional_encoding: (batch_size, seq_len, embed_size)
        """
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(RotaryPositionalEncoding, self).__init__()
        self.embed_size = embed_size

        # Precompute the rotary matrix
        self.register_buffer('inv_freq', 1. / (10000 ** (torch.arange(0, embed_size, 2).float() / embed_size)))

        # Position indices
        self.register_buffer('position', torch.arange(max_len).view(-1, 1))

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, embed_size) or for queries/keys (batch_size, heads, seq_len, head_dim)
        Returns:
            rotated x
        """
        if x.dim() == 3:  # (batch_size, seq_len, embed_size)
            batch_size, seq_len, embed_size = x.shape
            x = x.view(batch_size, seq_len, -1, 2)  # split into pairs
        else:  # (batch_size, heads, seq_len, head_dim) where head_dim = embed_size // heads, but wait
            # In multi-head, embed_size is not directly used. For RoPE, it's applied per head on query/key
            # But in standard RoPE for transformers, it's applied to embed_size.
            # For simplicity, assuming x is (batch_size, seq_len, embed_size)
            batch_size, seq_len, embed_size = x.shape
            x = x.view(batch_size, seq_len, -1, 2)

        # Compute rotary embeddings
        sincos = torch.sin(self.position[:seq_len] * self.inv_freq).cos()
        sincos = torch.cat([sincos, sincos], dim=-1).view(seq_len, -1, 2)  # (seq_len, embed_size//2, 2)

        # For each position
        x_complex = torch.view_as_complex(x.float())
        sincos_complex = torch.view_as_complex(sincos.float())
        out = torch.view_as_real(x_complex * sincos_complex)

        return out.view(batch_size, seq_len, embed_size)


class TokenAndPositionalEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, max_len=5000, pos_type='absolute'):
        super(TokenAndPositionalEmbedding, self).__init__()
        self.embed_size = embed_size
        self.token_embedding = nn.Embedding(vocab_size, embed_size)

        if pos_type == 'absolute':
            self.positional_encoding = AbsolutePositionalEncoding(embed_size, max_len)
        elif pos_type == 'rotary':
            self.positional_encoding = RotaryPositionalEncoding(embed_size, max_len)
        else:
            raise ValueError("pos_type must be 'absolute' or 'rotary'")

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len) - token indices
        Returns:
            embedded: (batch_size, seq_len, embed_size)
        """
        token_embed = self.token_embedding(x)
        return self.positional_encoding(token_embed)
