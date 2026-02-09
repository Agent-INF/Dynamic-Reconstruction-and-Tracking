"""
Main D4RT Model

Orchestrates encoder, query construction, and decoder for 4D scene reconstruction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import pytorch_lightning as pl

from .encoder import VideoEncoder
from .decoder import QueryDecoder
from .query import QueryConstructor
from .losses import D4RTLoss


class D4RTModel(pl.LightningModule):
    """
    D4RT: Efficiently Reconstructing Dynamic Scenes

    Unified transformer model for 4D reconstruction that jointly infers depth,
    spatio-temporal correspondence, and camera parameters from video.
    """

    def __init__(
        self,
        # Encoder params
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        encoder_depth: int = 12,
        encoder_num_heads: int = 12,
        # Decoder params
        decoder_depth: int = 6,
        decoder_num_heads: int = 8,
        # Query params
        num_spatial_frequencies: int = 10,
        max_frames: int = 100,
        patch_query_size: int = 9,
        use_image_patches: bool = True,
        # Loss params
        weight_3d: float = 1.0,
        weight_reproj: float = 0.5,
        weight_normal: float = 0.3,
        weight_visibility: float = 0.2,
        weight_motion: float = 0.3,
        weight_confidence: float = 0.1,
        # Training params
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.img_size = img_size
        self.embed_dim = embed_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps

        # Initialize components
        self.encoder = VideoEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
        )

        self.query_constructor = QueryConstructor(
            embed_dim=embed_dim,
            num_spatial_frequencies=num_spatial_frequencies,
            max_frames=max_frames,
            patch_size=patch_query_size,
            use_image_patches=use_image_patches,
        )

        self.decoder = QueryDecoder(
            embed_dim=embed_dim,
            num_layers=decoder_depth,
            num_heads=decoder_num_heads,
            output_3d=True,
            output_normals=True,
            output_visibility=True,
            output_motion=True,
            output_confidence=True,
        )

        # Loss function
        self.loss_fn = D4RTLoss(
            weight_3d=weight_3d,
            weight_reproj=weight_reproj,
            weight_normal=weight_normal,
            weight_visibility=weight_visibility,
            weight_motion=weight_motion,
            weight_confidence=weight_confidence,
        )

    def forward(
        self,
        video: torch.Tensor,
        uv_coords: torch.Tensor,
        t_src: torch.Tensor,
        t_tgt: torch.Tensor,
        t_cam: torch.Tensor,
        aspect_ratio: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through D4RT model

        Args:
            video: (B, T, C, H, W) input video
            uv_coords: (B, N, 2) normalized pixel coordinates
            t_src: (B, N) source frame indices
            t_tgt: (B, N) target frame indices
            t_cam: (B, N) camera reference frame indices
            aspect_ratio: (B,) aspect ratio of original video
        Returns:
            Dictionary with predictions
        """
        B, T, C, H, W = video.shape

        # Encode video
        features, aspect_token = self.encoder(video, aspect_ratio)  # (B*T, N_patches, D), (B, 1, D)

        # Construct queries
        queries = self.query_constructor(
            uv_coords=uv_coords,
            t_src=t_src,
            t_tgt=t_tgt,
            t_cam=t_cam,
            images=video,
        )  # (B, N_queries, D)

        # Decode queries
        # Need to aggregate features for cross-attention
        # Flatten features from all frames
        features_flat = features.reshape(B, T * features.shape[1], -1)  # (B, T*N_patches, D)

        predictions = self.decoder(queries, features_flat)

        return predictions

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step"""

        # Extract batch data
        video = batch['video']  # (B, T, C, H, W)
        uv_coords = batch['uv_coords']
        t_src = batch['t_src']
        t_tgt = batch['t_tgt']
        t_cam = batch['t_cam']
        target_3d = batch['target_3d']
        intrinsics = batch['intrinsics']

        # Forward pass
        predictions = self.forward(
            video=video,
            uv_coords=uv_coords,
            t_src=t_src,
            t_tgt=t_tgt,
            t_cam=t_cam,
        )

        # Prepare targets
        targets = {'pos_3d': target_3d}

        # Compute loss
        losses = self.loss_fn(predictions, targets, intrinsics)

        # Log losses
        self.log('train/loss', losses['total_loss'], prog_bar=True)
        for key, value in losses.items():
            if key != 'total_loss':
                self.log(f'train/{key}', value)

        return losses['total_loss']

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step"""

        # Extract batch data
        video = batch['video']
        uv_coords = batch['uv_coords']
        t_src = batch['t_src']
        t_tgt = batch['t_tgt']
        t_cam = batch['t_cam']
        target_3d = batch['target_3d']
        intrinsics = batch['intrinsics']

        # Forward pass
        predictions = self.forward(
            video=video,
            uv_coords=uv_coords,
            t_src=t_src,
            t_tgt=t_tgt,
            t_cam=t_cam,
        )

        # Prepare targets
        targets = {'pos_3d': target_3d}

        # Compute loss
        losses = self.loss_fn(predictions, targets, intrinsics)

        # Log losses
        self.log('val/loss', losses['total_loss'], prog_bar=True)
        for key, value in losses.items():
            if key != 'total_loss':
                self.log(f'val/{key}', value)

        return losses['total_loss']

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Cosine annealing with warmup
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / self.warmup_steps
            else:
                progress = (step - self.warmup_steps) / (self.trainer.max_steps - self.warmup_steps)
                return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }
        }

    def predict_depth(
        self,
        video: torch.Tensor,
        frame_idx: int = 0,
        num_points: int = 1024,
    ) -> torch.Tensor:
        """
        Predict depth map for a specific frame

        Args:
            video: (B, T, C, H, W) input video
            frame_idx: which frame to predict depth for
            num_points: number of query points
        Returns:
            (B, H, W) depth map
        """
        B, T, C, H, W = video.shape

        # Create grid of query points
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=video.device),
            torch.linspace(-1, 1, W, device=video.device),
            indexing='ij'
        )
        uv_coords = torch.stack([x, y], dim=-1).reshape(1, H * W, 2)  # (1, H*W, 2)
        uv_coords = uv_coords.repeat(B, 1, 1)

        # Query current frame
        t_src = torch.full((B, H * W), frame_idx, device=video.device, dtype=torch.long)
        t_tgt = t_src.clone()
        t_cam = t_src.clone()

        # Forward pass
        predictions = self.forward(
            video=video,
            uv_coords=uv_coords,
            t_src=t_src,
            t_tgt=t_tgt,
            t_cam=t_cam,
        )

        # Extract depth (Z coordinate)
        depth = predictions['pos_3d'][:, :, 2].reshape(B, H, W)

        return depth

    def track_points(
        self,
        video: torch.Tensor,
        points_2d: torch.Tensor,
        source_frame: int,
        target_frames: list,
    ) -> torch.Tensor:
        """
        Track 2D points across frames

        Args:
            video: (B, T, C, H, W) input video
            points_2d: (B, N, 2) 2D points to track in source frame
            source_frame: source frame index
            target_frames: list of target frame indices
        Returns:
            (B, len(target_frames), N, 3) tracked 3D positions
        """
        B, T, C, H, W = video.shape
        N = points_2d.shape[1]
        num_targets = len(target_frames)

        all_predictions = []

        for target_frame in target_frames:
            t_src = torch.full((B, N), source_frame, device=video.device, dtype=torch.long)
            t_tgt = torch.full((B, N), target_frame, device=video.device, dtype=torch.long)
            t_cam = torch.full((B, N), target_frame, device=video.device, dtype=torch.long)

            predictions = self.forward(
                video=video,
                uv_coords=points_2d,
                t_src=t_src,
                t_tgt=t_tgt,
                t_cam=t_cam,
            )

            all_predictions.append(predictions['pos_3d'])

        # Stack predictions
        tracked_points = torch.stack(all_predictions, dim=1)  # (B, len(target_frames), N, 3)

        return tracked_points
