"""
Inference script for D4RT model

Supports:
- Depth prediction
- Point tracking
- Camera pose estimation

Usage:
    python inference.py --checkpoint path/to/checkpoint.ckpt --video path/to/video.mp4 --task depth
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np
import cv2
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from d4rt.model import D4RTModel


def parse_args():
    parser = argparse.ArgumentParser(description='D4RT Inference')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--video', type=str, required=True,
                        help='Path to input video')
    parser.add_argument('--task', type=str, required=True,
                        choices=['depth', 'tracking', 'pose'],
                        help='Inference task')
    parser.add_argument('--output', type=str, default='output',
                        help='Output directory')
    parser.add_argument('--frame_idx', type=int, default=0,
                        help='Frame index for depth prediction')
    parser.add_argument('--num_frames', type=int, default=16,
                        help='Number of frames to process')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image size for processing')

    return parser.parse_args()


def load_video(video_path: str, num_frames: int = 16, img_size: int = 256) -> torch.Tensor:
    """
    Load video frames

    Args:
        video_path: path to video file
        num_frames: number of frames to load
        img_size: resize frames to this size
    Returns:
        (1, T, 3, H, W) video tensor
    """
    cap = cv2.VideoCapture(video_path)

    frames = []
    frame_count = 0

    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize
        frame = cv2.resize(frame, (img_size, img_size))

        # Normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0

        # Convert to tensor (H, W, C) -> (C, H, W)
        frame = torch.from_numpy(frame).permute(2, 0, 1)

        frames.append(frame)
        frame_count += 1

    cap.release()

    # Stack frames
    video = torch.stack(frames, dim=0)  # (T, C, H, W)
    video = video.unsqueeze(0)  # (1, T, C, H, W)

    return video


def save_depth_map(depth: torch.Tensor, output_path: str):
    """
    Save depth map as image

    Args:
        depth: (H, W) depth map
        output_path: path to save image
    """
    depth_np = depth.cpu().numpy()

    # Normalize to [0, 255]
    depth_min = depth_np.min()
    depth_max = depth_np.max()
    depth_norm = ((depth_np - depth_min) / (depth_max - depth_min + 1e-8) * 255).astype(np.uint8)

    # Apply colormap
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)

    # Save
    cv2.imwrite(output_path, depth_colored)
    print(f"Saved depth map to {output_path}")


def depth_prediction(model: D4RTModel, video: torch.Tensor, frame_idx: int, output_dir: Path):
    """Perform depth prediction"""

    print(f"Predicting depth for frame {frame_idx}...")

    with torch.no_grad():
        depth = model.predict_depth(video, frame_idx=frame_idx)

    # Save depth map
    depth_map = depth[0]  # (H, W)
    output_path = output_dir / f'depth_frame_{frame_idx:04d}.png'
    save_depth_map(depth_map, str(output_path))


def point_tracking(model: D4RTModel, video: torch.Tensor, output_dir: Path):
    """Perform point tracking"""

    print("Tracking points across frames...")

    B, T, C, H, W = video.shape

    # Sample some points to track from first frame
    num_points = 50
    points_2d = torch.rand(1, num_points, 2, device=video.device) * 2 - 1  # Random points in [-1, 1]

    # Track across all frames
    target_frames = list(range(T))

    with torch.no_grad():
        tracked_points = model.track_points(
            video=video,
            points_2d=points_2d,
            source_frame=0,
            target_frames=target_frames,
        )  # (1, T, N, 3)

    # Visualize tracking
    for t in range(T):
        frame = video[0, t].permute(1, 2, 0).cpu().numpy()  # (H, W, C)
        frame = (frame * 255).astype(np.uint8)

        # Get tracked points for this frame
        points_3d = tracked_points[0, t].cpu().numpy()  # (N, 3)

        # Project to 2D (simplified - just use X, Y)
        # In practice, would use proper projection with intrinsics
        points_2d_vis = ((points_3d[:, :2] + 1) / 2 * np.array([W, H])).astype(np.int32)

        # Draw points
        for pt in points_2d_vis:
            if 0 <= pt[0] < W and 0 <= pt[1] < H:
                cv2.circle(frame, tuple(pt), 3, (0, 255, 0), -1)

        # Save frame
        output_path = output_dir / f'tracking_frame_{t:04d}.png'
        cv2.imwrite(str(output_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    print(f"Saved tracking visualization to {output_dir}")


def camera_pose_estimation(model: D4RTModel, video: torch.Tensor, output_dir: Path):
    """Perform camera pose estimation"""

    print("Estimating camera poses...")

    # This is a placeholder - full implementation would require
    # specialized queries and processing
    print("Camera pose estimation is a placeholder in this implementation")

    # In the full paper implementation, this would:
    # 1. Use special pose queries
    # 2. Decode camera rotation and translation for each frame
    # 3. Save trajectory visualization


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = D4RTModel.load_from_checkpoint(args.checkpoint)
    model = model.to(device)
    model.eval()

    # Load video
    print(f"Loading video from {args.video}...")
    video = load_video(args.video, num_frames=args.num_frames, img_size=args.img_size)
    video = video.to(device)

    print(f"Video shape: {video.shape}")

    # Perform task
    if args.task == 'depth':
        depth_prediction(model, video, args.frame_idx, output_dir)

    elif args.task == 'tracking':
        point_tracking(model, video, output_dir)

    elif args.task == 'pose':
        camera_pose_estimation(model, video, output_dir)

    print("Inference completed!")


if __name__ == '__main__':
    main()
