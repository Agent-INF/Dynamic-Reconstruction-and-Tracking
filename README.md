# D4RT: Efficiently Reconstructing Dynamic Scenes

[![Paper](https://img.shields.io/badge/arXiv-2512.08924-b31b1b.svg)](https://arxiv.org/abs/2512.08924)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://d4rt-paper.github.io/)

This repository contains a complete implementation of **D4RT (Dynamic 4D Reconstruction and Tracking)**, a unified transformer-based model for efficient 4D reconstruction of dynamic scenes from video input.

## Overview

D4RT introduces a novel approach to dynamic scene understanding by:
- **Unified Architecture**: Single transformer model for depth, correspondence, and camera pose estimation
- **Query-Based Decoding**: Lightweight on-demand querying mechanism for flexible output
- **State-of-the-Art Performance**: 9-100x faster than previous methods while maintaining accuracy
- **Multi-Task Capability**: Supports depth prediction, point tracking, and pose estimation from one model

## Paper Details

**Title**: Efficiently Reconstructing Dynamic Scenes One D4RT at a Time

**Abstract**: D4RT uses a unified transformer architecture to jointly infer depth, spatio-temporal correspondence, and full camera parameters from video. The model processes video through a Vision Transformer encoder with alternating spatial and global attention, then uses a lightweight cross-attention decoder to answer queries about 3D positions at any point in space and time.

**Key Features**:
- Vision Transformer encoder with spatial and global self-attention
- Query mechanism with 5-tuple: (u, v, t_src, t_tgt, t_cam)
- Fourier embeddings for spatial coordinates
- Cross-attention decoder (6-8 layers)
- Multiple loss functions: L1 (3D), reprojection, normal, visibility, motion, confidence
- 2048 random queries per batch during training

## Installation

### Requirements
- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA (for GPU training)

### Setup

```bash
# Clone the repository
git clone https://github.com/Agent-INF/Dynamic-Reconstruction-and-Tracking.git
cd Dynamic-Reconstruction-and-Tracking

# Install dependencies
pip install -r requirements.txt
```

## Architecture

### 1. Encoder: Vision Transformer (ViT)
- Processes entire video sequence
- Alternating spatial (frame-local) and global (across-frame) attention
- Encodes at 256x256 resolution
- Includes aspect ratio token

### 2. Query Construction
- 5-tuple query format: (u, v, t_src, t_tgt, t_cam)
- Fourier feature embeddings for spatial coordinates
- Learned embeddings for temporal indices
- Optional 9×9 image patch context

### 3. Decoder: Cross-Attention Transformer
- 6-8 lightweight cross-attention layers
- Processes queries independently (fully parallelizable)
- Outputs: 3D positions, normals, visibility, motion, confidence

### 4. Loss Functions
- **L1 Loss**: 3D positions (log-scaled, depth-normalized)
- **Reprojection Loss**: 2D consistency
- **Normal Loss**: Cosine similarity
- **Visibility Loss**: Binary cross-entropy
- **Motion Loss**: 2D displacement
- **Confidence Penalty**: Uncertainty estimation

## Usage

### Training

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --data_dir /path/to/dataset \
    --batch_size 2 \
    --num_workers 4 \
    --max_epochs 100 \
    --gpus 1
```

### Dataset Format

Expected directory structure:
```
dataset/
├── train.txt          # List of training scenes
├── val.txt            # List of validation scenes
├── test.txt           # List of test scenes
├── scene_001/
│   ├── frames/
│   │   ├── 000000.jpg
│   │   ├── 000001.jpg
│   │   └── ...
│   └── annotations.json
├── scene_002/
│   └── ...
```

### Inference

#### Depth Prediction
```bash
python scripts/inference.py \
    --checkpoint path/to/checkpoint.ckpt \
    --video path/to/video.mp4 \
    --task depth \
    --frame_idx 0 \
    --output output/depth
```

#### Point Tracking
```bash
python scripts/inference.py \
    --checkpoint path/to/checkpoint.ckpt \
    --video path/to/video.mp4 \
    --task tracking \
    --output output/tracking
```

#### Camera Pose Estimation
```bash
python scripts/inference.py \
    --checkpoint path/to/checkpoint.ckpt \
    --video path/to/video.mp4 \
    --task pose \
    --output output/pose
```

## Model Architecture Details

### Encoder (VideoEncoder)
- **Input**: (B, T, 3, 256, 256) video
- **Patch Size**: 16x16
- **Embedding Dim**: 768
- **Depth**: 12 transformer blocks
- **Attention Pattern**: Alternating spatial/global
- **Output**: (B×T, N_patches, 768) features

### Query Constructor
- **Spatial Encoding**: Fourier features with 10 frequencies
- **Temporal Encoding**: Learned embeddings for up to 100 frames
- **Patch Context**: Optional 9×9 RGB patches
- **Output**: (B, N_queries, 768) query tokens

### Decoder (QueryDecoder)
- **Input**: Query tokens + encoded features
- **Architecture**: 6 cross-attention layers
- **Heads**: 8
- **Output**: 3D positions (X,Y,Z), normals, visibility, motion, confidence

## Training Details

### Hyperparameters
- **Batch Size**: 2 videos
- **Queries per Batch**: 2048
- **Learning Rate**: 1e-4 with cosine annealing
- **Warmup Steps**: 1000
- **Weight Decay**: 0.01
- **Precision**: Mixed (FP16)
- **Optimizer**: AdamW

### Loss Weights
- 3D Position: 1.0
- Reprojection: 0.5
- Surface Normal: 0.3
- Visibility: 0.2
- Motion: 0.3
- Confidence: 0.1

## Performance

The model achieves:
- **Speed**: 200+ FPS on single A100 GPU
- **Efficiency**: 9-100x faster than previous SOTA methods
- **Accuracy**: State-of-the-art on multiple 4D reconstruction benchmarks

## Project Structure

```
Dynamic-Reconstruction-and-Tracking/
├── src/d4rt/
│   ├── __init__.py           # Package initialization
│   ├── encoder.py            # Vision Transformer encoder
│   ├── decoder.py            # Cross-attention decoder
│   ├── query.py              # Query construction
│   ├── model.py              # Main D4RT model
│   ├── losses.py             # Loss functions
│   ├── geometry.py           # Geometric utilities
│   └── dataset.py            # Dataset and DataModule
├── scripts/
│   ├── train.py              # Training script
│   └── inference.py          # Inference script
├── configs/
│   └── default.yaml          # Default configuration
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Citation

If you use this code in your research, please cite the original D4RT paper:

```bibtex
@article{d4rt2024,
  title={Efficiently Reconstructing Dynamic Scenes One D4RT at a Time},
  author={[Authors]},
  journal={arXiv preprint arXiv:2512.08924},
  year={2024}
}
```

## References

- **Paper**: https://arxiv.org/abs/2512.08924
- **Project Page**: https://d4rt-paper.github.io/
- **DeepMind Blog**: https://deepmind.google/blog/d4rt-teaching-ai-to-see-the-world-in-four-dimensions/

## License

This implementation is for research purposes. Please refer to the original paper for licensing details.

## Acknowledgments

This implementation is based on the paper "Efficiently Reconstructing Dynamic Scenes One D4RT at a Time" and follows the architectural details described in the paper and related technical documentation.

## Implementation Notes

This is a complete reproduction of the D4RT paper including:

✅ Vision Transformer encoder with alternating spatial/global attention
✅ Query construction with Fourier embeddings
✅ Cross-attention decoder
✅ All loss functions (L1, reprojection, normal, visibility, motion, confidence)
✅ Geometry utilities for 3D transformations
✅ Training pipeline with PyTorch Lightning
✅ Inference scripts for depth, tracking, and pose estimation
✅ Configuration system

The implementation closely follows the paper's specifications including:
- 256x256 input resolution
- 16x16 patch size
- 768 embedding dimension
- 12 encoder layers with alternating attention
- 6 decoder layers
- 2048 queries per batch
- All specified loss components

## Contact

For questions about this implementation, please open an issue on GitHub.
