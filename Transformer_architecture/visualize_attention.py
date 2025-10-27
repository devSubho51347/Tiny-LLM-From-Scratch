import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from multi_head_attention import MultiHeadAttention


def visualize_attention():
    # Set parameters
    embed_size = 12
    heads = 2
    seq_len = 10
    batch_size = 1

    # Create dummy input data
    # For self-attention, keys, queries, values are the same
    x = torch.randn(batch_size, seq_len, embed_size)

    # Initialize multi-head attention
    mha = MultiHeadAttention(embed_size=embed_size, heads=heads)

    # Forward pass
    output = mha(x, x, x)

    # Attention weights shape: (batch_size, heads, seq_len, seq_len)
    attention_weights = mha.attention_weights.squeeze(0)  # Remove batch dimension

    # Plot attention for each head
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for head in range(heads):
        ax = axes[head]
        attn_head = attention_weights[head].detach().numpy()

        im = ax.imshow(attn_head, cmap='viridis', aspect='auto')
        ax.set_title(f'Attention Head {head + 1}')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')

        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig('attention_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    visualize_attention()
