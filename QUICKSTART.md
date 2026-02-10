# Quick Start Guide for D4RT

This guide will help you get started with the D4RT implementation quickly.

## Installation

```bash
# Clone the repository
git clone https://github.com/Agent-INF/Dynamic-Reconstruction-and-Tracking.git
cd Dynamic-Reconstruction-and-Tracking

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Test

Run the example usage script to verify everything works:

```bash
python scripts/example_usage.py
```

Expected output:
```
============================================================
D4RT Example Usage
============================================================

============================================================
Example 1: Depth Prediction
============================================================
Input video shape: torch.Size([1, 16, 3, 256, 256])
Output depth map shape: torch.Size([1, 256, 256])
Depth range: [X.XX, X.XX]

============================================================
Example 2: Point Tracking
============================================================
Input video shape: torch.Size([1, 16, 3, 256, 256])
Input points shape: torch.Size([1, 10, 2])
Tracked points shape: torch.Size([1, 16, 10, 3])
Tracking 10 points across 16 frames

============================================================
Example 3: Custom Query Forward Pass
============================================================
Input video shape: torch.Size([2, 16, 3, 256, 256])
Number of queries: 2048

Predictions:
  pos_3d: torch.Size([2, 2048, 3])
  normals: torch.Size([2, 2048, 3])
  visibility: torch.Size([2, 2048, 1])
  motion: torch.Size([2, 2048, 2])
  confidence: torch.Size([2, 2048, 1])

============================================================
Example 4: Model Size and Parameters
============================================================
Model: D4RT
Embedding dimension: 768
Encoder depth: 12 layers
Decoder depth: 6 layers

Total parameters: XXX,XXX,XXX
Trainable parameters: XXX,XXX,XXX
Model size: XXX.XX MB (FP32)

============================================================
All examples completed successfully!
============================================================
```

## Training on Your Data

### 1. Prepare Your Dataset

Create a dataset directory with this structure:

```
my_dataset/
├── train.txt
├── val.txt
├── test.txt
├── scene_001/
│   ├── frames/
│   │   ├── 000000.jpg
│   │   ├── 000001.jpg
│   │   └── ...
│   └── annotations.json
├── scene_002/
│   └── ...
```

**annotations.json format:**
```json
{
  "points_3d": [
    [[x1, y1, z1], [x2, y2, z2], ...],  // frame 0
    [[x1, y1, z1], [x2, y2, z2], ...],  // frame 1
    ...
  ],
  "intrinsics": [
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
  ],
  "extrinsics": [
    [[r11, r12, r13, tx], [r21, r22, r23, ty], [r31, r32, r33, tz], [0, 0, 0, 1]],  // frame 0
    ...
  ],
  "visibility": [
    [1, 1, 0, 1, ...],  // frame 0
    ...
  ]
}
```

### 2. Start Training

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --data_dir my_dataset/ \
    --batch_size 2 \
    --num_workers 4 \
    --max_epochs 100 \
    --gpus 1 \
    --log_dir logs/
```

Monitor training with TensorBoard:
```bash
tensorboard --logdir logs/
```

### 3. Resume from Checkpoint

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --data_dir my_dataset/ \
    --checkpoint logs/checkpoints/last.ckpt \
    --batch_size 2 \
    --max_epochs 200 \
    --gpus 1
```

## Inference on Videos

### Depth Prediction

```bash
python scripts/inference.py \
    --checkpoint logs/checkpoints/best.ckpt \
    --video test_video.mp4 \
    --task depth \
    --frame_idx 0 \
    --output output/depth/
```

This will generate: `output/depth/depth_frame_0000.png`

### Point Tracking

```bash
python scripts/inference.py \
    --checkpoint logs/checkpoints/best.ckpt \
    --video test_video.mp4 \
    --task tracking \
    --num_frames 16 \
    --output output/tracking/
```

This will generate: `output/tracking/tracking_frame_XXXX.png` for each frame

### Camera Pose Estimation

```bash
python scripts/inference.py \
    --checkpoint logs/checkpoints/best.ckpt \
    --video test_video.mp4 \
    --task pose \
    --output output/pose/
```

## Using D4RT Programmatically

### Example 1: Simple Depth Prediction

```python
import torch
from d4rt import D4RTModel

# Load model
model = D4RTModel.load_from_checkpoint('checkpoint.ckpt')
model.eval()

# Prepare video (B, T, C, H, W)
video = torch.rand(1, 16, 3, 256, 256)

# Predict depth
with torch.no_grad():
    depth = model.predict_depth(video, frame_idx=0)

print(f"Depth shape: {depth.shape}")  # (1, 256, 256)
```

### Example 2: Track Points

```python
import torch
from d4rt import D4RTModel

# Load model
model = D4RTModel.load_from_checkpoint('checkpoint.ckpt')
model.eval()

# Prepare video
video = torch.rand(1, 16, 3, 256, 256)

# Define points to track (normalized coordinates)
points_2d = torch.tensor([
    [[0.0, 0.0], [0.5, 0.5], [-0.5, -0.5]]
])  # (1, 3, 2)

# Track from frame 0 to frames 1-15
with torch.no_grad():
    tracked = model.track_points(
        video=video,
        points_2d=points_2d,
        source_frame=0,
        target_frames=list(range(1, 16))
    )

print(f"Tracked points shape: {tracked.shape}")  # (1, 15, 3, 3)
```

### Example 3: Custom Forward Pass

```python
import torch
from d4rt import D4RTModel
from d4rt.query import sample_random_queries

# Load model
model = D4RTModel.load_from_checkpoint('checkpoint.ckpt')
model.eval()

# Prepare video
video = torch.rand(1, 16, 3, 256, 256)

# Sample queries
uv_coords, t_src, t_tgt, t_cam = sample_random_queries(
    batch_size=1,
    num_queries=1024,
    num_frames=16,
    device=video.device
)

# Forward pass
with torch.no_grad():
    predictions = model(
        video=video,
        uv_coords=uv_coords,
        t_src=t_src,
        t_tgt=t_tgt,
        t_cam=t_cam
    )

# Access predictions
pos_3d = predictions['pos_3d']        # (1, 1024, 3)
normals = predictions['normals']      # (1, 1024, 3)
visibility = predictions['visibility']  # (1, 1024, 1)
motion = predictions['motion']        # (1, 1024, 2)
confidence = predictions['confidence']  # (1, 1024, 1)
```

## Configuration

### Modify Model Architecture

Edit `configs/default.yaml`:

```yaml
model:
  embed_dim: 768      # Embedding dimension
  encoder_depth: 12   # Number of encoder layers
  decoder_depth: 6    # Number of decoder layers
  encoder_num_heads: 12  # Attention heads in encoder
  decoder_num_heads: 8   # Attention heads in decoder
```

### Adjust Training Settings

```yaml
model:
  learning_rate: 0.0001
  weight_decay: 0.01
  warmup_steps: 1000

training:
  batch_size: 2
  max_epochs: 100
  gpus: 1
```

### Tune Loss Weights

```yaml
model:
  weight_3d: 1.0
  weight_reproj: 0.5
  weight_normal: 0.3
  weight_visibility: 0.2
  weight_motion: 0.3
  weight_confidence: 0.1
```

## Common Issues and Solutions

### Out of Memory (OOM)

Reduce batch size or image size:
```yaml
data:
  image_size: 224  # Instead of 256
training:
  batch_size: 1    # Instead of 2
```

### Slow Training

Enable mixed precision (already enabled by default):
```python
trainer = pl.Trainer(
    precision='16-mixed',  # Use FP16
    ...
)
```

### Model Not Converging

Check loss weights and learning rate:
```yaml
model:
  learning_rate: 0.00005  # Reduce LR
  weight_3d: 2.0          # Increase 3D loss weight
```

## Next Steps

1. **Train on Real Data**: Prepare your dataset with proper annotations
2. **Fine-tune**: Start from a pre-trained checkpoint
3. **Evaluate**: Test on standard benchmarks
4. **Optimize**: Convert to ONNX or TensorRT for faster inference
5. **Extend**: Add new output heads or modify the architecture

## Getting Help

- Check `README.md` for detailed documentation
- See `IMPLEMENTATION.md` for technical details
- Run `python scripts/example_usage.py` for code examples
- Open an issue on GitHub for bugs or questions

## Citation

```bibtex
@article{d4rt2024,
  title={Efficiently Reconstructing Dynamic Scenes One D4RT at a Time},
  author={[Authors]},
  journal={arXiv preprint arXiv:2512.08924},
  year={2024}
}
```

Happy reconstructing! 🎯
