"""
Dataset and DataModule for D4RT training

Implements video dataset loading and query sampling for training.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Dict, Optional, Tuple, List
import numpy as np
from pathlib import Path
import json

from .query import sample_random_queries
from .geometry import normalize_coordinates, create_default_intrinsics


class VideoDataset(Dataset):
    """
    Dataset for video sequences with 3D annotations

    Expected directory structure:
    root/
        scene_001/
            frames/
                000000.jpg
                000001.jpg
                ...
            annotations.json  # Contains 3D points, camera params, etc.
        scene_002/
            ...
    """

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        num_frames: int = 16,
        image_size: int = 256,
        num_queries: int = 2048,
        edge_sampling_ratio: float = 0.2,
    ):
        super().__init__()

        self.root_dir = Path(root_dir)
        self.split = split
        self.num_frames = num_frames
        self.image_size = image_size
        self.num_queries = num_queries
        self.edge_sampling_ratio = edge_sampling_ratio

        # Load scene list
        split_file = self.root_dir / f'{split}.txt'
        if split_file.exists():
            with open(split_file, 'r') as f:
                self.scene_names = [line.strip() for line in f]
        else:
            # If no split file, use all scenes
            self.scene_names = [d.name for d in self.root_dir.iterdir() if d.is_dir()]

        print(f"Loaded {len(self.scene_names)} scenes for {split} split")

    def __len__(self) -> int:
        return len(self.scene_names)

    def load_video_frames(self, scene_path: Path) -> torch.Tensor:
        """
        Load video frames

        Args:
            scene_path: path to scene directory
        Returns:
            (T, C, H, W) video tensor
        """
        # This is a placeholder implementation
        # In practice, would load actual images using PIL or opencv

        frames_dir = scene_path / 'frames'

        if not frames_dir.exists():
            # Generate dummy data for testing
            return torch.rand(self.num_frames, 3, self.image_size, self.image_size)

        # Load frames (simplified)
        frame_files = sorted(frames_dir.glob('*.jpg'))[:self.num_frames]

        if len(frame_files) < self.num_frames:
            # Generate dummy data if not enough frames
            return torch.rand(self.num_frames, 3, self.image_size, self.image_size)

        # In real implementation, would load and process images
        # For now, return dummy data
        return torch.rand(self.num_frames, 3, self.image_size, self.image_size)

    def load_annotations(self, scene_path: Path) -> Dict:
        """
        Load 3D annotations and camera parameters

        Returns:
            Dictionary containing:
                - points_3d: List of 3D points per frame
                - intrinsics: Camera intrinsics
                - extrinsics: Camera extrinsics per frame
                - visibility: Visibility masks
        """
        ann_file = scene_path / 'annotations.json'

        if not ann_file.exists():
            # Generate dummy annotations for testing
            return self.generate_dummy_annotations()

        try:
            with open(ann_file, 'r') as f:
                annotations = json.load(f)
            return annotations
        except:
            return self.generate_dummy_annotations()

    def generate_dummy_annotations(self) -> Dict:
        """Generate dummy annotations for testing"""

        # Generate random 3D points
        num_points = self.num_queries
        points_3d = []

        for t in range(self.num_frames):
            # Generate points with some temporal consistency
            pts = np.random.randn(num_points, 3) * 2.0
            pts[:, 2] += 5.0  # Offset depth
            points_3d.append(pts.tolist())

        # Default camera intrinsics
        intrinsics = create_default_intrinsics(
            1, self.image_size, self.image_size
        ).squeeze(0).numpy().tolist()

        # Identity extrinsics
        extrinsics = []
        for t in range(self.num_frames):
            ext = np.eye(4).tolist()
            extrinsics.append(ext)

        # All visible
        visibility = [[1] * num_points for _ in range(self.num_frames)]

        return {
            'points_3d': points_3d,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'visibility': visibility
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample

        Returns:
            Dictionary containing:
                - video: (T, C, H, W) video frames
                - queries: Query parameters (uv, t_src, t_tgt, t_cam)
                - targets: Ground truth 3D positions and other annotations
                - intrinsics: Camera intrinsics
        """
        scene_name = self.scene_names[idx]
        scene_path = self.root_dir / scene_name

        # Load video
        video = self.load_video_frames(scene_path)  # (T, C, H, W)

        # Load annotations
        annotations = self.load_annotations(scene_path)

        # Sample queries
        uv_coords, t_src, t_tgt, t_cam = sample_random_queries(
            batch_size=1,
            num_queries=self.num_queries,
            num_frames=self.num_frames,
            device=video.device,
            edge_sampling_ratio=self.edge_sampling_ratio
        )

        # Get ground truth for queries
        # This is simplified - in practice would look up actual 3D points
        target_3d = torch.randn(1, self.num_queries, 3)
        target_3d[:, :, 2] += 5.0  # Offset depth

        # Get intrinsics
        intrinsics = torch.tensor(annotations['intrinsics'], dtype=torch.float32)

        return {
            'video': video,
            'uv_coords': uv_coords.squeeze(0),
            't_src': t_src.squeeze(0),
            't_tgt': t_tgt.squeeze(0),
            't_cam': t_cam.squeeze(0),
            'target_3d': target_3d.squeeze(0),
            'intrinsics': intrinsics,
        }


class D4RTDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for D4RT"""

    def __init__(
        self,
        root_dir: str,
        num_frames: int = 16,
        image_size: int = 256,
        num_queries: int = 2048,
        batch_size: int = 2,
        num_workers: int = 4,
    ):
        super().__init__()

        self.root_dir = root_dir
        self.num_frames = num_frames
        self.image_size = image_size
        self.num_queries = num_queries
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        """Setup datasets"""

        if stage == 'fit' or stage is None:
            self.train_dataset = VideoDataset(
                root_dir=self.root_dir,
                split='train',
                num_frames=self.num_frames,
                image_size=self.image_size,
                num_queries=self.num_queries,
            )

            self.val_dataset = VideoDataset(
                root_dir=self.root_dir,
                split='val',
                num_frames=self.num_frames,
                image_size=self.image_size,
                num_queries=self.num_queries,
            )

        if stage == 'test' or stage is None:
            self.test_dataset = VideoDataset(
                root_dir=self.root_dir,
                split='test',
                num_frames=self.num_frames,
                image_size=self.image_size,
                num_queries=self.num_queries,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
