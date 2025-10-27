import torch
import torch.nn as nn
import torch.nn.functional as F
from scaled_dot_product_attention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

        self.attention = ScaledDotProductAttention()

    def forward(self, values, keys, queries, mask=None):
        """
        Forward pass for multi-head attention.

        Args:
            values: (batch_size, seq_len, embed_size)
            keys: (batch_size, seq_len, embed_size)
            queries: (batch_size, seq_len, embed_size)
            mask: Optional mask

        Returns:
            output: (batch_size, seq_len, embed_size)
        """
        N = queries.shape[0]

        # Split embedding into self.heads different pieces
        values_len, keys_len, queries_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Split into heads
        values = values.reshape(N, values_len, self.heads, self.head_dim)
        keys = keys.reshape(N, keys_len, self.heads, self.head_dim)
        queries = queries.reshape(N, queries_len, self.heads, self.head_dim)

        # Transpose to (N, heads, seq_len, head_dim)
        values = values.transpose(1, 2)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)

        # Apply linear transformations after splitting
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Calculate attention using scaled dot-product attention
        out, self.attention_weights = self.attention(queries, keys, values, mask)

        # Concatenate heads and apply final linear layer
        # out: (N, heads, seq_len, head_dim) -> (N, seq_len, heads, head_dim)
        out = out.transpose(1, 2).contiguous()
        # (N, seq_len, heads * head_dim)
        out = out.view(N, queries_len, self.heads * self.head_dim)

        out = self.fc_out(out)

        return out
