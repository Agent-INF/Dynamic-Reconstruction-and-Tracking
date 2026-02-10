"""
Training script for D4RT model

Usage:
    python train.py --config configs/default.yaml --data_dir /path/to/data
"""

import argparse
import yaml
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

import sys
sys.path.append(str(Path(__file__).parent.parent))

from d4rt.model import D4RTModel
from d4rt.dataset import D4RTDataModule


def parse_args():
    parser = argparse.ArgumentParser(description='Train D4RT model')

    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory for logging')

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    else:
        # Return default config
        return {
            'model': {
                'img_size': 256,
                'patch_size': 16,
                'embed_dim': 768,
                'encoder_depth': 12,
                'encoder_num_heads': 12,
                'decoder_depth': 6,
                'decoder_num_heads': 8,
                'learning_rate': 1e-4,
                'weight_decay': 0.01,
            },
            'data': {
                'num_frames': 16,
                'num_queries': 2048,
                'image_size': 256,
            }
        }


def main():
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set random seed for reproducibility
    pl.seed_everything(42)

    # Create data module
    data_module = D4RTDataModule(
        root_dir=args.data_dir,
        num_frames=config['data'].get('num_frames', 16),
        image_size=config['data'].get('image_size', 256),
        num_queries=config['data'].get('num_queries', 2048),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Create model
    if args.checkpoint:
        model = D4RTModel.load_from_checkpoint(args.checkpoint)
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        model = D4RTModel(
            **config['model']
        )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(args.log_dir) / 'checkpoints',
        filename='d4rt-{epoch:02d}-{val/loss:.4f}',
        monitor='val/loss',
        mode='min',
        save_top_k=3,
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Setup logger
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name='d4rt',
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus if args.gpus > 0 else 1,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        gradient_clip_val=1.0,
        precision='16-mixed',  # Use mixed precision for efficiency
        log_every_n_steps=10,
    )

    # Train
    print("Starting training...")
    trainer.fit(model, data_module)

    print("Training completed!")
    print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")


if __name__ == '__main__':
    main()
