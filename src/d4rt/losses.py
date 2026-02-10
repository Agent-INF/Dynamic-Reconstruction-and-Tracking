"""
Loss functions for D4RT training

Implements L1, reprojection, normal, visibility, motion, and confidence losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from .geometry import project_3d_to_2d


class D4RTLoss(nn.Module):
    """
    Combined loss function for D4RT training

    Includes:
    - L1 loss on 3D positions (log-scaled and normalized by mean depth)
    - 2D reprojection loss
    - Surface normal loss (cosine similarity)
    - Binary cross-entropy for visibility
    - Motion/displacement loss
    - Confidence penalty
    """

    def __init__(
        self,
        weight_3d: float = 1.0,
        weight_reproj: float = 0.5,
        weight_normal: float = 0.3,
        weight_visibility: float = 0.2,
        weight_motion: float = 0.3,
        weight_confidence: float = 0.1,
        depth_normalize: bool = True,
        log_scale: bool = True,
    ):
        super().__init__()

        self.weight_3d = weight_3d
        self.weight_reproj = weight_reproj
        self.weight_normal = weight_normal
        self.weight_visibility = weight_visibility
        self.weight_motion = weight_motion
        self.weight_confidence = weight_confidence

        self.depth_normalize = depth_normalize
        self.log_scale = log_scale

    def compute_3d_loss(
        self,
        pred_3d: torch.Tensor,
        target_3d: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute L1 loss on 3D positions

        Args:
            pred_3d: (B, N, 3) predicted 3D positions
            target_3d: (B, N, 3) ground truth 3D positions
            valid_mask: (B, N) mask for valid points
        Returns:
            scalar loss
        """
        # Compute L1 distance
        l1_loss = torch.abs(pred_3d - target_3d)  # (B, N, 3)

        # Normalize by mean depth
        if self.depth_normalize:
            mean_depth = target_3d[:, :, 2].mean(dim=1, keepdim=True) + 1e-8  # (B, 1)
            l1_loss = l1_loss / mean_depth.unsqueeze(-1)

        # Log-scale to down-weight distant errors
        if self.log_scale:
            l1_loss = torch.log1p(l1_loss)

        # Apply mask if provided
        if valid_mask is not None:
            l1_loss = l1_loss * valid_mask.unsqueeze(-1)
            loss = l1_loss.sum() / (valid_mask.sum() + 1e-8)
        else:
            loss = l1_loss.mean()

        return loss

    def compute_reprojection_loss(
        self,
        pred_3d: torch.Tensor,
        target_2d: torch.Tensor,
        intrinsics: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute 2D reprojection loss

        Args:
            pred_3d: (B, N, 3) predicted 3D positions
            target_2d: (B, N, 2) ground truth 2D positions
            intrinsics: (B, 3, 3) camera intrinsics
            valid_mask: (B, N) mask for valid points
        Returns:
            scalar loss
        """
        # Project 3D to 2D
        pred_2d = project_3d_to_2d(pred_3d, intrinsics)  # (B, N, 2)

        # Compute L1 loss
        reproj_loss = torch.abs(pred_2d - target_2d)  # (B, N, 2)

        # Apply mask if provided
        if valid_mask is not None:
            reproj_loss = reproj_loss * valid_mask.unsqueeze(-1)
            loss = reproj_loss.sum() / (valid_mask.sum() + 1e-8)
        else:
            loss = reproj_loss.mean()

        return loss

    def compute_normal_loss(
        self,
        pred_normals: torch.Tensor,
        target_normals: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute surface normal loss using cosine similarity

        Args:
            pred_normals: (B, N, 3) predicted normals
            target_normals: (B, N, 3) ground truth normals
            valid_mask: (B, N) mask for valid points
        Returns:
            scalar loss
        """
        # Normalize vectors
        pred_normals = F.normalize(pred_normals, dim=-1)
        target_normals = F.normalize(target_normals, dim=-1)

        # Compute cosine similarity
        cos_sim = (pred_normals * target_normals).sum(dim=-1)  # (B, N)

        # Loss is 1 - cosine similarity
        normal_loss = 1.0 - cos_sim

        # Apply mask if provided
        if valid_mask is not None:
            normal_loss = normal_loss * valid_mask
            loss = normal_loss.sum() / (valid_mask.sum() + 1e-8)
        else:
            loss = normal_loss.mean()

        return loss

    def compute_visibility_loss(
        self,
        pred_visibility: torch.Tensor,
        target_visibility: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute binary cross-entropy loss for visibility

        Args:
            pred_visibility: (B, N, 1) predicted visibility scores
            target_visibility: (B, N, 1) ground truth visibility (0 or 1)
            valid_mask: (B, N) mask for valid points
        Returns:
            scalar loss
        """
        # Binary cross-entropy
        vis_loss = F.binary_cross_entropy(
            pred_visibility.squeeze(-1),
            target_visibility.squeeze(-1),
            reduction='none'
        )  # (B, N)

        # Apply mask if provided
        if valid_mask is not None:
            vis_loss = vis_loss * valid_mask
            loss = vis_loss.sum() / (valid_mask.sum() + 1e-8)
        else:
            loss = vis_loss.mean()

        return loss

    def compute_motion_loss(
        self,
        pred_motion: torch.Tensor,
        target_motion: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute L1 loss on motion/displacement

        Args:
            pred_motion: (B, N, 2) predicted 2D motion
            target_motion: (B, N, 2) ground truth 2D motion
            valid_mask: (B, N) mask for valid points
        Returns:
            scalar loss
        """
        motion_loss = torch.abs(pred_motion - target_motion)  # (B, N, 2)

        # Apply mask if provided
        if valid_mask is not None:
            motion_loss = motion_loss * valid_mask.unsqueeze(-1)
            loss = motion_loss.sum() / (valid_mask.sum() + 1e-8)
        else:
            loss = motion_loss.mean()

        return loss

    def compute_confidence_penalty(
        self,
        confidence: torch.Tensor,
        errors: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute confidence penalty (encourages accurate uncertainty estimation)

        Args:
            confidence: (B, N, 1) predicted confidence scores
            errors: (B, N) prediction errors
            valid_mask: (B, N) mask for valid points
        Returns:
            scalar loss
        """
        # High confidence should correlate with low errors
        confidence_penalty = confidence.squeeze(-1) * errors

        # Apply mask if provided
        if valid_mask is not None:
            confidence_penalty = confidence_penalty * valid_mask
            loss = confidence_penalty.sum() / (valid_mask.sum() + 1e-8)
        else:
            loss = confidence_penalty.mean()

        return loss

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        intrinsics: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss

        Args:
            predictions: Dictionary containing predicted values
            targets: Dictionary containing ground truth values
            intrinsics: (B, 3, 3) camera intrinsics (needed for reprojection)
            valid_mask: (B, N) mask for valid points
        Returns:
            Dictionary with total loss and individual loss components
        """
        losses = {}
        total_loss = 0.0

        # 3D position loss
        if 'pos_3d' in predictions and 'pos_3d' in targets:
            loss_3d = self.compute_3d_loss(
                predictions['pos_3d'],
                targets['pos_3d'],
                valid_mask
            )
            losses['loss_3d'] = loss_3d
            total_loss += self.weight_3d * loss_3d

        # Reprojection loss
        if 'pos_3d' in predictions and 'pos_2d' in targets and intrinsics is not None:
            loss_reproj = self.compute_reprojection_loss(
                predictions['pos_3d'],
                targets['pos_2d'],
                intrinsics,
                valid_mask
            )
            losses['loss_reproj'] = loss_reproj
            total_loss += self.weight_reproj * loss_reproj

        # Normal loss
        if 'normals' in predictions and 'normals' in targets:
            loss_normal = self.compute_normal_loss(
                predictions['normals'],
                targets['normals'],
                valid_mask
            )
            losses['loss_normal'] = loss_normal
            total_loss += self.weight_normal * loss_normal

        # Visibility loss
        if 'visibility' in predictions and 'visibility' in targets:
            loss_vis = self.compute_visibility_loss(
                predictions['visibility'],
                targets['visibility'],
                valid_mask
            )
            losses['loss_visibility'] = loss_vis
            total_loss += self.weight_visibility * loss_vis

        # Motion loss
        if 'motion' in predictions and 'motion' in targets:
            loss_motion = self.compute_motion_loss(
                predictions['motion'],
                targets['motion'],
                valid_mask
            )
            losses['loss_motion'] = loss_motion
            total_loss += self.weight_motion * loss_motion

        # Confidence penalty
        if 'confidence' in predictions and 'pos_3d' in predictions and 'pos_3d' in targets:
            # Use 3D error as the error metric
            errors = torch.abs(predictions['pos_3d'] - targets['pos_3d']).sum(dim=-1)
            loss_conf = self.compute_confidence_penalty(
                predictions['confidence'],
                errors,
                valid_mask
            )
            losses['loss_confidence'] = loss_conf
            total_loss += self.weight_confidence * loss_conf

        losses['total_loss'] = total_loss
        return losses
