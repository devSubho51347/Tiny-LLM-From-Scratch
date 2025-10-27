import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorflow import keras
from keras.models import Model, Sequential
from keras.layers import  Dense, Dropout, LayerNormalization


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, query, key, value, mask=None):
        """
        Compute scaled dot-product attention.

        Args:
            query: (batch_size, num_heads, seq_len, d_k)
            key: (batch_size, num_heads, seq_len, d_k)
            value: (batch_size, num_heads, seq_len, d_v)
            mask: Optional mask for attention

        Returns:
            attention_output: (batch_size, num_heads, seq_len, d_v)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        d_k = query.size(-1)

        # Compute attention scores: (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # Compute attention output
        attention_output = torch.matmul(attention_weights, value)

        return attention_output, attention_weights
