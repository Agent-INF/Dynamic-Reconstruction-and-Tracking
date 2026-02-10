"""
Cross-Attention Decoder for D4RT

Implements the lightweight decoder that processes queries via cross-attention
to the encoded scene features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class CrossAttentionLayer(nn.Module):
    """Cross-attention layer for query-to-scene attention"""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Query projection
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)

        # Key and value projections (from encoded features)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: (B, N_queries, D) query tokens
            features: (B, N_features, D) encoded scene features
        Returns:
            (B, N_queries, D) attended query features
        """
        B, N_q, D = queries.shape
        N_f = features.shape[1]

        # Project queries
        q = self.q_proj(queries).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # q: (B, num_heads, N_q, head_dim)

        # Project keys and values from features
        kv = self.kv_proj(features).reshape(B, N_f, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # Each: (B, num_heads, N_f, head_dim)

        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N_q, N_f)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N_q, D)  # (B, N_q, D)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class DecoderBlock(nn.Module):
    """Decoder transformer block with cross-attention"""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()

        # Self-attention for queries
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        # Cross-attention to scene features
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = CrossAttentionLayer(dim, num_heads, qkv_bias, dropout)

        # MLP
        self.norm3 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, queries: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: (B, N_queries, D) query tokens
            features: (B, N_features, D) encoded scene features
        Returns:
            (B, N_queries, D) processed queries
        """
        # Self-attention among queries
        q_norm = self.norm1(queries)
        queries = queries + self.self_attn(q_norm, q_norm, q_norm, need_weights=False)[0]

        # Cross-attention to scene features
        queries = queries + self.cross_attn(self.norm2(queries), features)

        # MLP
        queries = queries + self.mlp(self.norm3(queries))

        return queries


class QueryDecoder(nn.Module):
    """
    Query Decoder for D4RT

    Processes queries via cross-attention to encoded scene features and
    produces 3D positions and auxiliary predictions.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        output_3d: bool = True,
        output_normals: bool = True,
        output_visibility: bool = True,
        output_motion: bool = True,
        output_confidence: bool = True,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.output_3d = output_3d
        self.output_normals = output_normals
        self.output_visibility = output_visibility
        self.output_motion = output_motion
        self.output_confidence = output_confidence

        # Decoder blocks
        self.blocks = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, mlp_ratio, qkv_bias, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Output heads
        if output_3d:
            self.pos_3d_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Linear(embed_dim // 2, 3)  # X, Y, Z
            )

        if output_normals:
            self.normal_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Linear(embed_dim // 2, 3)  # Normal vector
            )

        if output_visibility:
            self.visibility_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),
                nn.GELU(),
                nn.Linear(embed_dim // 4, 1)  # Binary visibility
            )

        if output_motion:
            self.motion_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Linear(embed_dim // 2, 2)  # 2D motion/displacement
            )

        if output_confidence:
            self.confidence_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),
                nn.GELU(),
                nn.Linear(embed_dim // 4, 1)  # Prediction confidence
            )

    def forward(
        self,
        queries: torch.Tensor,
        features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Process queries to produce 3D predictions

        Args:
            queries: (B, N_queries, D) query tokens
            features: (B, N_features, D) encoded scene features from encoder
        Returns:
            Dictionary containing:
                - pos_3d: (B, N, 3) 3D positions in camera coordinates
                - normals: (B, N, 3) surface normals (optional)
                - visibility: (B, N, 1) visibility scores (optional)
                - motion: (B, N, 2) 2D motion vectors (optional)
                - confidence: (B, N, 1) prediction confidence (optional)
        """
        # Process through decoder blocks
        x = queries
        for block in self.blocks:
            x = block(x, features)

        x = self.norm(x)

        # Generate outputs
        outputs = {}

        if self.output_3d:
            outputs['pos_3d'] = self.pos_3d_head(x)

        if self.output_normals:
            normals = self.normal_head(x)
            # Normalize to unit vectors
            outputs['normals'] = F.normalize(normals, dim=-1)

        if self.output_visibility:
            outputs['visibility'] = torch.sigmoid(self.visibility_head(x))

        if self.output_motion:
            outputs['motion'] = self.motion_head(x)

        if self.output_confidence:
            outputs['confidence'] = torch.sigmoid(self.confidence_head(x))

        return outputs

    def decode_depth(self, queries: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Convenience method to decode depth maps

        Args:
            queries: (B, N, D) query tokens
            features: (B, N_features, D) encoded features
        Returns:
            (B, N) depth values
        """
        outputs = self.forward(queries, features)
        pos_3d = outputs['pos_3d']  # (B, N, 3)
        # Depth is Z coordinate
        depth = pos_3d[:, :, 2]
        return depth

    def decode_camera_pose(
        self,
        queries: torch.Tensor,
        features: torch.Tensor,
        num_pose_queries: int = 8
    ) -> Dict[str, torch.Tensor]:
        """
        Decode camera pose from queries

        Args:
            queries: (B, num_pose_queries, D) special pose queries
            features: (B, N_features, D) encoded features
            num_pose_queries: number of queries for pose estimation
        Returns:
            Dictionary with:
                - rotation: (B, 3, 3) rotation matrix
                - translation: (B, 3) translation vector
        """
        outputs = self.forward(queries, features)

        # Average over pose queries to get camera parameters
        pos_3d = outputs['pos_3d']  # (B, num_pose_queries, 3)

        # Simple aggregation (in practice, would use more sophisticated method)
        translation = pos_3d.mean(dim=1)  # (B, 3)

        # For rotation, would need additional output head
        # Here we provide a placeholder identity rotation
        B = queries.shape[0]
        rotation = torch.eye(3, device=queries.device).unsqueeze(0).repeat(B, 1, 1)

        return {
            'rotation': rotation,
            'translation': translation
        }
