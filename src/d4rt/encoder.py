"""
Vision Transformer Encoder for D4RT

Implements the encoder that processes the entire video sequence to create a global
latent representation capturing geometry, appearance, and temporal dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Tuple


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(self, img_size: int = 256, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input images
        Returns:
            (B, N, D) patch embeddings where N = (H*W) / (patch_size^2)
        """
        x = self.proj(x)  # (B, D, H/P, W/P)
        x = rearrange(x, 'b d h w -> b (h w) d')
        return x


class SpatialAttention(nn.Module):
    """Spatial (frame-local) self-attention"""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B*T, N, D) frame-local features
        Returns:
            (B*T, N, D) attended features
        """
        BT, N, D = x.shape

        qkv = self.qkv(x).reshape(BT, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (BT, H, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(BT, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GlobalAttention(nn.Module):
    """Global (across-frame) self-attention"""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, batch_size: int, num_frames: int) -> torch.Tensor:
        """
        Args:
            x: (B*T, N, D) features
            batch_size: B
            num_frames: T
        Returns:
            (B*T, N, D) globally attended features
        """
        BT, N, D = x.shape

        # Reshape to (B, T, N, D)
        x = rearrange(x, '(b t) n d -> b t n d', b=batch_size, t=num_frames)

        # Flatten spatial and temporal for global attention
        x = rearrange(x, 'b t n d -> b (t n) d')

        qkv = self.qkv(x).reshape(batch_size, num_frames * N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, H, T*N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch_size, num_frames * N, D)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Reshape back to (B*T, N, D)
        x = rearrange(x, 'b (t n) d -> (b t) n d', t=num_frames, n=N)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with optional spatial or global attention"""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attention_type: str = 'spatial'  # 'spatial' or 'global'
    ):
        super().__init__()
        self.attention_type = attention_type
        self.norm1 = nn.LayerNorm(dim)

        if attention_type == 'spatial':
            self.attn = SpatialAttention(dim, num_heads, qkv_bias, dropout)
        else:
            self.attn = GlobalAttention(dim, num_heads, qkv_bias, dropout)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, batch_size: int = 1, num_frames: int = 1) -> torch.Tensor:
        """
        Args:
            x: (B*T, N, D) input features
            batch_size: B (needed for global attention)
            num_frames: T (needed for global attention)
        """
        if self.attention_type == 'spatial':
            x = x + self.attn(self.norm1(x))
        else:
            x = x + self.attn(self.norm1(x), batch_size, num_frames)

        x = x + self.mlp(self.norm2(x))
        return x


class VideoEncoder(nn.Module):
    """
    Vision Transformer Encoder for D4RT

    Processes video sequences and encodes them into a global latent representation
    with alternating spatial and global self-attention layers.
    """

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.n_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        self.temporal_embed = nn.Parameter(torch.zeros(1, 100, embed_dim))  # Support up to 100 frames

        # Aspect ratio token
        self.aspect_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Transformer blocks with alternating spatial and global attention
        self.blocks = nn.ModuleList()
        for i in range(depth):
            attention_type = 'spatial' if i % 2 == 0 else 'global'
            self.blocks.append(
                TransformerBlock(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias,
                    dropout,
                    attention_type
                )
            )

        self.norm = nn.LayerNorm(embed_dim)

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.temporal_embed, std=0.02)
        nn.init.trunc_normal_(self.aspect_token, std=0.02)

    def forward(self, video: torch.Tensor, aspect_ratio: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            video: (B, T, C, H, W) input video
            aspect_ratio: (B,) aspect ratio of original video
        Returns:
            features: (B*T, N, D) encoded features
            aspect_token: (B, 1, D) aspect ratio token
        """
        B, T, C, H, W = video.shape

        # Reshape for batch processing: (B*T, C, H, W)
        video = rearrange(video, 'b t c h w -> (b t) c h w')

        # Patch embedding
        x = self.patch_embed(video)  # (B*T, N, D)

        # Add positional embeddings
        x = x + self.pos_embed

        # Add temporal embeddings
        temporal_embed = self.temporal_embed[:, :T, :]  # (1, T, D)
        temporal_embed = repeat(temporal_embed, '1 t d -> (b t) 1 d', b=B)
        temporal_embed = repeat(temporal_embed, 'bt 1 d -> bt n d', n=self.n_patches)
        x = x + temporal_embed

        # Process through transformer blocks
        for block in self.blocks:
            x = block(x, batch_size=B, num_frames=T)

        x = self.norm(x)

        # Prepare aspect ratio token
        aspect_token = repeat(self.aspect_token, '1 1 d -> b 1 d', b=B)
        if aspect_ratio is not None:
            # Encode aspect ratio into the token (simple linear encoding)
            aspect_encoding = aspect_ratio.view(B, 1, 1) * 0.1  # Scale factor
            aspect_token = aspect_token + aspect_encoding

        return x, aspect_token
