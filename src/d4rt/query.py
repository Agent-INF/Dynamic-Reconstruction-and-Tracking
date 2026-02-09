"""
Query Construction for D4RT

Implements the query mechanism with 5-tuple (u, v, t_src, t_tgt, t_cam) and
Fourier feature embeddings for spatial coordinates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class FourierEmbedding(nn.Module):
    """Fourier feature embedding for spatial coordinates"""

    def __init__(self, num_frequencies: int = 10, learnable: bool = False):
        super().__init__()
        self.num_frequencies = num_frequencies

        if learnable:
            self.frequencies = nn.Parameter(torch.randn(num_frequencies) * 10)
        else:
            # Fixed geometric progression of frequencies
            frequencies = 2.0 ** torch.arange(num_frequencies, dtype=torch.float32)
            self.register_buffer('frequencies', frequencies)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (B, N, 2) normalized coordinates in [-1, 1]
        Returns:
            (B, N, num_frequencies * 4) Fourier features
        """
        # coords: (B, N, 2)
        # frequencies: (num_frequencies,)

        # Expand dimensions for broadcasting
        coords = coords.unsqueeze(-2)  # (B, N, 1, 2)
        freqs = self.frequencies.view(1, 1, -1, 1)  # (1, 1, num_frequencies, 1)

        # Compute Fourier features
        angles = coords * freqs * np.pi  # (B, N, num_frequencies, 2)

        # Apply sin and cos
        sin_features = torch.sin(angles)  # (B, N, num_frequencies, 2)
        cos_features = torch.cos(angles)  # (B, N, num_frequencies, 2)

        # Concatenate sin and cos features
        features = torch.cat([sin_features, cos_features], dim=-1)  # (B, N, num_frequencies, 4)
        features = features.flatten(-2, -1)  # (B, N, num_frequencies * 4)

        return features


class QueryConstructor(nn.Module):
    """
    Query Construction for D4RT

    Constructs query tokens from 5-tuple: (u, v, t_src, t_tgt, t_cam)
    - (u, v): normalized 2D pixel coordinates
    - t_src: source frame timestamp
    - t_tgt: target timestamp for tracking
    - t_cam: reference camera coordinate frame
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_spatial_frequencies: int = 10,
        max_frames: int = 100,
        patch_size: int = 9,
        use_image_patches: bool = True,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.use_image_patches = use_image_patches

        # Fourier embedding for spatial coordinates (u, v)
        self.spatial_fourier = FourierEmbedding(num_spatial_frequencies, learnable=False)
        spatial_dim = num_spatial_frequencies * 4  # sin and cos for each freq and coord

        # Learned embeddings for temporal indices
        self.t_src_embed = nn.Embedding(max_frames, embed_dim // 4)
        self.t_tgt_embed = nn.Embedding(max_frames, embed_dim // 4)
        self.t_cam_embed = nn.Embedding(max_frames, embed_dim // 4)

        # Optional: image patch embedding
        if use_image_patches:
            patch_features = patch_size * patch_size * 3  # RGB patch
            self.patch_proj = nn.Linear(patch_features, embed_dim // 4)
        else:
            self.patch_proj = None

        # Project spatial Fourier features
        self.spatial_proj = nn.Linear(spatial_dim, embed_dim // 4)

        # Final projection to combine all query components
        if use_image_patches:
            query_input_dim = embed_dim  # 4 components * embed_dim//4
        else:
            query_input_dim = embed_dim  # Still 4 components but without patch

        self.query_proj = nn.Sequential(
            nn.Linear(query_input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def extract_image_patches(
        self,
        images: torch.Tensor,
        uv_coords: torch.Tensor,
        frame_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract local image patches around query points

        Args:
            images: (B, T, C, H, W) input video
            uv_coords: (B, N, 2) normalized coordinates in [-1, 1]
            frame_indices: (B, N) frame indices
        Returns:
            (B, N, patch_size^2 * C) flattened patches
        """
        B, T, C, H, W = images.shape
        N = uv_coords.shape[1]

        # Convert normalized coords to pixel coords
        grid = uv_coords.unsqueeze(1)  # (B, N, 2) -> (B, 1, N, 2)

        # Sample patches using grid_sample for each frame
        patches = []
        for b in range(B):
            frame_patches = []
            for n in range(N):
                t = frame_indices[b, n].item()
                if t >= T:
                    t = T - 1

                # Extract patch using grid_sample
                frame = images[b, t:t+1]  # (1, C, H, W)

                # Create a small grid around the query point
                half_size = self.patch_size // 2
                offsets = torch.linspace(-half_size, half_size, self.patch_size, device=images.device)
                offset_y, offset_x = torch.meshgrid(offsets, offsets, indexing='ij')
                offset_grid = torch.stack([offset_x, offset_y], dim=-1)  # (patch_size, patch_size, 2)

                # Scale offsets to normalized coordinates
                offset_grid = offset_grid / torch.tensor([W/2, H/2], device=images.device)

                # Add to query point
                query_grid = uv_coords[b, n:n+1, :] + offset_grid.view(-1, 2)  # (patch_size^2, 2)
                query_grid = query_grid.unsqueeze(0).unsqueeze(0)  # (1, 1, patch_size^2, 2)

                # Sample from frame
                patch = F.grid_sample(
                    frame,
                    query_grid,
                    mode='bilinear',
                    padding_mode='border',
                    align_corners=False
                )  # (1, C, 1, patch_size^2)

                patch = patch.squeeze(2).squeeze(0)  # (C, patch_size^2)
                frame_patches.append(patch)

            frame_patches = torch.stack(frame_patches, dim=0)  # (N, C, patch_size^2)
            patches.append(frame_patches)

        patches = torch.stack(patches, dim=0)  # (B, N, C, patch_size^2)
        patches = patches.permute(0, 1, 3, 2).flatten(-2, -1)  # (B, N, patch_size^2 * C)

        return patches

    def forward(
        self,
        uv_coords: torch.Tensor,
        t_src: torch.Tensor,
        t_tgt: torch.Tensor,
        t_cam: torch.Tensor,
        images: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Construct query tokens

        Args:
            uv_coords: (B, N, 2) normalized pixel coordinates in [-1, 1]
            t_src: (B, N) source frame indices
            t_tgt: (B, N) target frame indices
            t_cam: (B, N) camera reference frame indices
            images: (B, T, C, H, W) optional video for patch extraction
        Returns:
            (B, N, embed_dim) query tokens
        """
        B, N, _ = uv_coords.shape

        # 1. Spatial Fourier features
        spatial_features = self.spatial_fourier(uv_coords)  # (B, N, spatial_dim)
        spatial_features = self.spatial_proj(spatial_features)  # (B, N, embed_dim//4)

        # 2. Temporal embeddings
        t_src_features = self.t_src_embed(t_src)  # (B, N, embed_dim//4)
        t_tgt_features = self.t_tgt_embed(t_tgt)  # (B, N, embed_dim//4)
        t_cam_features = self.t_cam_embed(t_cam)  # (B, N, embed_dim//4)

        # 3. Optional image patch features
        if self.use_image_patches and images is not None:
            patch_features = self.extract_image_patches(images, uv_coords, t_src)  # (B, N, patch_size^2 * C)
            patch_features = self.patch_proj(patch_features)  # (B, N, embed_dim//4)

            # Combine all features
            query_features = torch.cat([
                spatial_features,
                t_src_features,
                t_tgt_features,
                t_cam_features
            ], dim=-1)  # (B, N, embed_dim)
        else:
            # Combine without patch features
            query_features = torch.cat([
                spatial_features,
                t_src_features,
                t_tgt_features,
                t_cam_features
            ], dim=-1)  # (B, N, embed_dim)

        # Final projection
        query_tokens = self.query_proj(query_features)  # (B, N, embed_dim)

        return query_tokens


def sample_random_queries(
    batch_size: int,
    num_queries: int,
    num_frames: int,
    device: torch.device,
    edge_sampling_ratio: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample random queries for training

    Args:
        batch_size: B
        num_queries: N (typically 2048)
        num_frames: T
        device: torch device
        edge_sampling_ratio: ratio of queries sampled near edges/motion boundaries
    Returns:
        uv_coords: (B, N, 2) in [-1, 1]
        t_src: (B, N) frame indices
        t_tgt: (B, N) frame indices
        t_cam: (B, N) frame indices
    """
    # Sample random pixel coordinates
    uv_coords = torch.rand(batch_size, num_queries, 2, device=device) * 2 - 1  # [-1, 1]

    # Sample temporal indices
    t_src = torch.randint(0, num_frames, (batch_size, num_queries), device=device)
    t_tgt = torch.randint(0, num_frames, (batch_size, num_queries), device=device)
    t_cam = torch.randint(0, num_frames, (batch_size, num_queries), device=device)

    # Note: In a full implementation, edge_sampling_ratio would be used to
    # focus some queries on edges and motion boundaries for better learning
    # This would require edge detection or optical flow preprocessing

    return uv_coords, t_src, t_tgt, t_cam
