"""
Example usage of D4RT for different tasks

This script demonstrates how to use the D4RT model for:
1. Depth prediction
2. Point tracking
3. Camera pose estimation
"""

import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from d4rt.model import D4RTModel
from d4rt.query import sample_random_queries


def example_depth_prediction():
    """Example: Predict depth map from video"""
    print("=" * 60)
    print("Example 1: Depth Prediction")
    print("=" * 60)

    # Create model
    model = D4RTModel(
        img_size=256,
        embed_dim=768,
        encoder_depth=12,
        decoder_depth=6,
    )
    model.eval()

    # Create dummy video (B, T, C, H, W)
    batch_size = 1
    num_frames = 16
    video = torch.rand(batch_size, num_frames, 3, 256, 256)

    # Predict depth for frame 0
    with torch.no_grad():
        depth_map = model.predict_depth(video, frame_idx=0)

    print(f"Input video shape: {video.shape}")
    print(f"Output depth map shape: {depth_map.shape}")
    print(f"Depth range: [{depth_map.min():.2f}, {depth_map.max():.2f}]")
    print()


def example_point_tracking():
    """Example: Track points across frames"""
    print("=" * 60)
    print("Example 2: Point Tracking")
    print("=" * 60)

    # Create model
    model = D4RTModel(
        img_size=256,
        embed_dim=768,
        encoder_depth=12,
        decoder_depth=6,
    )
    model.eval()

    # Create dummy video
    batch_size = 1
    num_frames = 16
    video = torch.rand(batch_size, num_frames, 3, 256, 256)

    # Define points to track (in normalized coordinates [-1, 1])
    num_points = 10
    points_2d = torch.rand(batch_size, num_points, 2) * 2 - 1

    # Track from frame 0 to all other frames
    source_frame = 0
    target_frames = list(range(num_frames))

    with torch.no_grad():
        tracked_points = model.track_points(
            video=video,
            points_2d=points_2d,
            source_frame=source_frame,
            target_frames=target_frames,
        )

    print(f"Input video shape: {video.shape}")
    print(f"Input points shape: {points_2d.shape}")
    print(f"Tracked points shape: {tracked_points.shape}")
    print(f"Tracking {num_points} points across {num_frames} frames")
    print()


def example_forward_pass():
    """Example: Direct forward pass with custom queries"""
    print("=" * 60)
    print("Example 3: Custom Query Forward Pass")
    print("=" * 60)

    # Create model
    model = D4RTModel(
        img_size=256,
        embed_dim=768,
        encoder_depth=12,
        decoder_depth=6,
    )
    model.eval()

    # Create dummy video
    batch_size = 2
    num_frames = 16
    video = torch.rand(batch_size, num_frames, 3, 256, 256)

    # Sample random queries
    num_queries = 2048
    uv_coords, t_src, t_tgt, t_cam = sample_random_queries(
        batch_size=batch_size,
        num_queries=num_queries,
        num_frames=num_frames,
        device=video.device,
    )

    # Forward pass
    with torch.no_grad():
        predictions = model.forward(
            video=video,
            uv_coords=uv_coords,
            t_src=t_src,
            t_tgt=t_tgt,
            t_cam=t_cam,
        )

    print(f"Input video shape: {video.shape}")
    print(f"Number of queries: {num_queries}")
    print(f"\nPredictions:")
    for key, value in predictions.items():
        print(f"  {key}: {value.shape}")
    print()


def example_model_size():
    """Example: Show model size and parameters"""
    print("=" * 60)
    print("Example 4: Model Size and Parameters")
    print("=" * 60)

    # Create model
    model = D4RTModel(
        img_size=256,
        embed_dim=768,
        encoder_depth=12,
        decoder_depth=6,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model: D4RT")
    print(f"Embedding dimension: 768")
    print(f"Encoder depth: 12 layers")
    print(f"Decoder depth: 6 layers")
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    print()


def main():
    print("\n" + "=" * 60)
    print("D4RT Example Usage")
    print("=" * 60 + "\n")

    # Run examples
    example_depth_prediction()
    example_point_tracking()
    example_forward_pass()
    example_model_size()

    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
