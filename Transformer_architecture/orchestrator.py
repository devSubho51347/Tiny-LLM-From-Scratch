import torch
import torch.nn as nn
from multi_head_attention import MultiHeadAttention
from positional_embeddings import TokenAndPositionalEmbedding


class TransformerBlock(nn.Module):
    def __init__(self, vocab_size, embed_size, heads, max_len=5000, pos_type='absolute'):
        super(TransformerBlock, self).__init__()
        self.embed_size = embed_size
        self.embedding = TokenAndPositionalEmbedding(vocab_size, embed_size, max_len, pos_type)
        self.multi_head_attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        # Simple feed-forward (can be expanded)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size)
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len) - input tokens
            mask: Optional attention mask
        Returns:
            output: (batch_size, seq_len, embed_size)
        """
        # Embed tokens with positional encoding
        x = self.embedding(x)  # (batch_size, seq_len, embed_size)

        # Multi-head self-attention
        attn_out = self.multi_head_attention(x, x, x, mask)  # self-attention
        x = self.norm1(x + attn_out)  # residual + norm

        # Feed-forward
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)  # residual + norm

        return x


class Orchestrator:
    def __init__(self, vocab_size, embed_size=512, heads=8, max_len=5000, pos_type='absolute'):
        """
        Initialize the Transformer orchestrator.

        Args:
            vocab_size: Size of vocabulary
            embed_size: Embedding dimension
            heads: Number of attention heads
            max_len: Maximum sequence length
            pos_type: 'absolute' or 'rotary' positional encoding
        """
        self.transformer_block = TransformerBlock(vocab_size, embed_size, heads, max_len, pos_type)
        self.pos_type = pos_type

    def forward(self, input_ids, mask=None):
        """
        Forward pass through the transformer.

        Args:
            input_ids: (batch_size, seq_len) - input token ids
            mask: Optional attention mask
        Returns:
            output: (batch_size, seq_len, embed_size)
        """
        return self.transformer_block(input_ids, mask)


if __name__ == "__main__":
    # Example usage
    vocab_size = 10000  # Example vocab size
    embed_size = 512
    heads = 8
    max_len = 100
    batch_size = 2
    seq_len = 50

    # Test with absolute positional encoding
    print("Testing with Absolute Positional Encoding:")
    orchestrator_abs = Orchestrator(vocab_size, embed_size, heads, max_len, pos_type='absolute')
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    output_abs = orchestrator_abs.forward(input_ids)
    print(f"Output shape (absolute): {output_abs.shape}")

    # Test with rotary positional encoding
    print("\nTesting with Rotary Positional Encoding:")
    orchestrator_rot = Orchestrator(vocab_size, embed_size, heads, max_len, pos_type='rotary')
    output_rot = orchestrator_rot.forward(input_ids)
    print(f"Output shape (rotary): {output_rot.shape}")
