import torch
import torch.nn as nn
from einops import rearrange, repeat

class ViTBlock(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, num_layers, num_heads, mlp_dim, dropout):
        super(ViTBlock, self).__init__()
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim

        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=patch_size)
        self.norm1 = nn.LayerNorm(out_channels)
        self.attn = nn.MultiheadAttention(out_channels, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(out_channels)
        self.mlp = nn.Sequential(
            nn.Linear(out_channels, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, out_channels),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # Patch embedding
        x = x.flatten(2).transpose(1, 2)  # (B, C, H, W) -> (B, N, C)
        
        for _ in range(self.num_layers):
            x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
            x = x + self.mlp(self.norm2(x))
        
        return x
