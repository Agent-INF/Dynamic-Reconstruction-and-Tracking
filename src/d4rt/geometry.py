"""
Geometry utilities for D4RT

Implements 3D transformations, projections, and geometric operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def normalize_coordinates(coords: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Normalize pixel coordinates to [-1, 1]

    Args:
        coords: (B, N, 2) pixel coordinates
        H, W: image height and width
    Returns:
        (B, N, 2) normalized coordinates
    """
    coords_norm = coords.clone()
    coords_norm[..., 0] = (coords[..., 0] / (W - 1)) * 2 - 1
    coords_norm[..., 1] = (coords[..., 1] / (H - 1)) * 2 - 1
    return coords_norm


def denormalize_coordinates(coords_norm: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Denormalize coordinates from [-1, 1] to pixel space

    Args:
        coords_norm: (B, N, 2) normalized coordinates
        H, W: image height and width
    Returns:
        (B, N, 2) pixel coordinates
    """
    coords = coords_norm.clone()
    coords[..., 0] = ((coords_norm[..., 0] + 1) / 2) * (W - 1)
    coords[..., 1] = ((coords_norm[..., 1] + 1) / 2) * (H - 1)
    return coords


def project_3d_to_2d(
    points_3d: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Project 3D points to 2D image coordinates

    Args:
        points_3d: (B, N, 3) 3D points in camera coordinates or world coordinates
        intrinsics: (B, 3, 3) camera intrinsic matrix
        extrinsics: (B, 4, 4) camera extrinsic matrix (optional, for world coords)
    Returns:
        (B, N, 2) 2D pixel coordinates
    """
    B, N, _ = points_3d.shape

    # Transform from world to camera coordinates if extrinsics provided
    if extrinsics is not None:
        # Convert to homogeneous coordinates
        points_homo = torch.cat([
            points_3d,
            torch.ones(B, N, 1, device=points_3d.device)
        ], dim=-1)  # (B, N, 4)

        # Apply extrinsics
        points_3d = torch.bmm(
            points_homo,
            extrinsics.transpose(-2, -1)
        )[:, :, :3]  # (B, N, 3)

    # Project to 2D
    points_2d_homo = torch.bmm(
        points_3d,
        intrinsics.transpose(-2, -1)
    )  # (B, N, 3)

    # Normalize by depth
    points_2d = points_2d_homo[:, :, :2] / (points_2d_homo[:, :, 2:3] + 1e-8)

    return points_2d


def unproject_2d_to_3d(
    points_2d: torch.Tensor,
    depth: torch.Tensor,
    intrinsics: torch.Tensor
) -> torch.Tensor:
    """
    Unproject 2D points with depth to 3D camera coordinates

    Args:
        points_2d: (B, N, 2) 2D pixel coordinates
        depth: (B, N) depth values
        intrinsics: (B, 3, 3) camera intrinsic matrix
    Returns:
        (B, N, 3) 3D points in camera coordinates
    """
    B, N, _ = points_2d.shape

    # Convert to homogeneous coordinates
    points_2d_homo = torch.cat([
        points_2d,
        torch.ones(B, N, 1, device=points_2d.device)
    ], dim=-1)  # (B, N, 3)

    # Compute inverse intrinsics
    intrinsics_inv = torch.inverse(intrinsics)  # (B, 3, 3)

    # Unproject
    points_3d = torch.bmm(
        points_2d_homo,
        intrinsics_inv.transpose(-2, -1)
    )  # (B, N, 3)

    # Scale by depth
    points_3d = points_3d * depth.unsqueeze(-1)

    return points_3d


def compute_camera_rays(
    uv_coords: torch.Tensor,
    intrinsics: torch.Tensor
) -> torch.Tensor:
    """
    Compute camera rays for given pixel coordinates

    Args:
        uv_coords: (B, N, 2) normalized or pixel coordinates
        intrinsics: (B, 3, 3) camera intrinsic matrix
    Returns:
        (B, N, 3) normalized ray directions
    """
    # Unproject at unit depth
    unit_depth = torch.ones(uv_coords.shape[0], uv_coords.shape[1], device=uv_coords.device)
    rays = unproject_2d_to_3d(uv_coords, unit_depth, intrinsics)

    # Normalize
    rays = F.normalize(rays, dim=-1)

    return rays


def transform_points_3d(
    points: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor
) -> torch.Tensor:
    """
    Transform 3D points with rotation and translation

    Args:
        points: (B, N, 3) 3D points
        rotation: (B, 3, 3) rotation matrix
        translation: (B, 3) translation vector
    Returns:
        (B, N, 3) transformed points
    """
    # Apply rotation
    points_rot = torch.bmm(points, rotation.transpose(-2, -1))

    # Apply translation
    points_transformed = points_rot + translation.unsqueeze(1)

    return points_transformed


def compute_surface_normals(
    depth_map: torch.Tensor,
    intrinsics: torch.Tensor
) -> torch.Tensor:
    """
    Compute surface normals from depth map

    Args:
        depth_map: (B, H, W) depth values
        intrinsics: (B, 3, 3) camera intrinsics
    Returns:
        (B, 3, H, W) surface normals
    """
    B, H, W = depth_map.shape

    # Create coordinate grid
    y, x = torch.meshgrid(
        torch.arange(H, device=depth_map.device),
        torch.arange(W, device=depth_map.device),
        indexing='ij'
    )
    coords_2d = torch.stack([x, y], dim=-1).float()  # (H, W, 2)
    coords_2d = coords_2d.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, H, W, 2)
    coords_2d = coords_2d.reshape(B, H * W, 2)

    # Unproject to 3D
    depth_flat = depth_map.reshape(B, H * W)
    points_3d = unproject_2d_to_3d(coords_2d, depth_flat, intrinsics)
    points_3d = points_3d.reshape(B, H, W, 3)

    # Compute gradients
    # dz/dx
    grad_x = points_3d[:, :, 1:, :] - points_3d[:, :, :-1, :]
    grad_x = F.pad(grad_x, (0, 0, 0, 1), mode='replicate')

    # dz/dy
    grad_y = points_3d[:, 1:, :, :] - points_3d[:, :-1, :, :]
    grad_y = F.pad(grad_y, (0, 0, 0, 0, 0, 1), mode='replicate')

    # Cross product for normals
    normals = torch.cross(grad_x, grad_y, dim=-1)  # (B, H, W, 3)

    # Normalize
    normals = F.normalize(normals, dim=-1)

    # Transpose to (B, 3, H, W)
    normals = normals.permute(0, 3, 1, 2)

    return normals


def compute_optical_flow(
    points_3d_t0: torch.Tensor,
    points_3d_t1: torch.Tensor,
    intrinsics: torch.Tensor
) -> torch.Tensor:
    """
    Compute optical flow from 3D point motion

    Args:
        points_3d_t0: (B, N, 3) 3D points at time t0
        points_3d_t1: (B, N, 3) 3D points at time t1
        intrinsics: (B, 3, 3) camera intrinsics
    Returns:
        (B, N, 2) 2D optical flow vectors
    """
    # Project both to 2D
    points_2d_t0 = project_3d_to_2d(points_3d_t0, intrinsics)
    points_2d_t1 = project_3d_to_2d(points_3d_t1, intrinsics)

    # Compute flow
    flow = points_2d_t1 - points_2d_t0

    return flow


def create_default_intrinsics(
    B: int,
    H: int,
    W: int,
    fov: float = 60.0,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Create default camera intrinsics

    Args:
        B: batch size
        H, W: image height and width
        fov: field of view in degrees
        device: torch device
    Returns:
        (B, 3, 3) intrinsic matrix
    """
    import math

    # Compute focal length from FOV
    focal = (W / 2.0) / math.tan(math.radians(fov / 2.0))

    intrinsics = torch.zeros(B, 3, 3, device=device)
    intrinsics[:, 0, 0] = focal  # fx
    intrinsics[:, 1, 1] = focal  # fy
    intrinsics[:, 0, 2] = W / 2.0  # cx
    intrinsics[:, 1, 2] = H / 2.0  # cy
    intrinsics[:, 2, 2] = 1.0

    return intrinsics


def rotation_matrix_from_vectors(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """
    Compute rotation matrix that rotates v1 to v2

    Args:
        v1: (B, 3) source vectors
        v2: (B, 3) target vectors
    Returns:
        (B, 3, 3) rotation matrices
    """
    B = v1.shape[0]

    # Normalize vectors
    v1 = F.normalize(v1, dim=-1)
    v2 = F.normalize(v2, dim=-1)

    # Compute rotation axis
    axis = torch.cross(v1, v2, dim=-1)
    axis_norm = torch.norm(axis, dim=-1, keepdim=True)

    # Handle parallel vectors
    axis = axis / (axis_norm + 1e-8)

    # Compute rotation angle
    cos_angle = (v1 * v2).sum(dim=-1, keepdim=True)
    sin_angle = axis_norm

    # Create skew-symmetric matrix
    K = torch.zeros(B, 3, 3, device=v1.device)
    K[:, 0, 1] = -axis[:, 2]
    K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2]
    K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]
    K[:, 2, 1] = axis[:, 0]

    # Rodrigues' rotation formula
    I = torch.eye(3, device=v1.device).unsqueeze(0).repeat(B, 1, 1)
    R = I + sin_angle.unsqueeze(-1) * K + (1 - cos_angle.unsqueeze(-1)) * torch.bmm(K, K)

    return R
