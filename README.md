# MR-to-CT Medical Image Synthesis using Conditional DDPM

This project implements a conditional Denoising Diffusion Probabilistic Model (DDPM) for MR-to-CT medical image synthesis using MONAI and Hugging Face Accelerate.

## Overview

The model performs modality transfer from MR (Magnetic Resonance) images to CT (Computed Tomography) images using a conditional diffusion model. The architecture uses:

- **Input**: 2 channels (CT noisy image xt concatenated with MR condition)
- **Output**: 1 channel (predicted noise epsilon)
- **Model**: DiffusionModelUNet from MONAI
- **Training/Inference**: Hugging Face Accelerate for mixed precision and distributed training

## Dataset Structure

```
dataset/
├── mr/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── test/
│       ├── image1.jpg
│       └── ...
└── ct/
    ├── train/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── test/
        ├── image1.jpg
        └── ...
```

**Note**: MR and CT images are paired by sorting filenames. The actual filenames don't need to match - they will be paired based on their sorted order. Make sure the number of images in corresponding MR and CT directories is the same.

## Installation

```bash
pip install -r requirements.txt
```

## Training

Basic training with mixed precision (FP16):
```bash
python train.py --dataset_dir /path/to/dataset --output_dir ./checkpoints
```

With online validation:
```bash
python train.py --dataset_dir /path/to/dataset --output_dir ./checkpoints --enable_validation
```

Resume training from checkpoint:
```bash
python train.py --dataset_dir /path/to/dataset --output_dir ./checkpoints --resume ./checkpoints/checkpoint_latest.pt
```

Disable mixed precision:
```bash
python train.py --dataset_dir /path/to/dataset --output_dir ./checkpoints --mixed_precision no
```

Multi-GPU training with Accelerate launcher:
```bash
accelerate launch train.py --dataset_dir /path/to/dataset --output_dir ./checkpoints
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset_dir` | (required) | Path to dataset directory |
| `--output_dir` | `./checkpoints` | Directory to save checkpoints |
| `--image_size` | 256 | Image size (square) |
| `--batch_size` | 4 | Batch size for training |
| `--num_epochs` | 100 | Number of training epochs |
| `--lr` | 2.5e-5 | Learning rate |
| `--num_workers` | 4 | Number of data loading workers |
| `--num_train_timesteps` | 1000 | Number of diffusion timesteps |
| `--resume` | None | Path to checkpoint to resume training |
| `--val_interval` | 5 | Validation interval (epochs) |
| `--enable_validation` | False | Enable online validation |
| `--save_interval` | 10 | Checkpoint save interval (epochs) |
| `--mixed_precision` | `fp16` | Mixed precision mode (`no`, `fp16`, `bf16`) |
| `--seed` | 42 | Random seed for reproducibility |

## Testing/Inference

Generate CT images from test MR images with mixed precision:
```bash
python test.py --checkpoint ./checkpoints/model_final.pt --dataset_dir /path/to/dataset --output_dir ./results
```

### Fast Sampling with DDIM

Use DDIM sampler with fewer steps for faster inference (e.g., 50 steps instead of 1000):
```bash
python test.py --checkpoint ./checkpoints/model_final.pt --dataset_dir /path/to/dataset --sampler ddim --num_inference_steps 50
```

DDIM with stochastic sampling (eta=1.0 makes it similar to DDPM):
```bash
python test.py --checkpoint ./checkpoints/model_final.pt --dataset_dir /path/to/dataset --sampler ddim --num_inference_steps 100 --ddim_eta 0.5
```

### Timestep Respacing

Both DDPM and DDIM support timestep respacing for accelerated sampling:
```bash
# DDPM with 250 steps (4x faster than default 1000)
python test.py --checkpoint ./checkpoints/model_final.pt --dataset_dir /path/to/dataset --sampler ddpm --num_inference_steps 250

# DDIM with 50 steps (20x faster than default 1000)
python test.py --checkpoint ./checkpoints/model_final.pt --dataset_dir /path/to/dataset --sampler ddim --num_inference_steps 50
```

Disable mixed precision:
```bash
python test.py --checkpoint ./checkpoints/model_final.pt --dataset_dir /path/to/dataset --output_dir ./results --mixed_precision no
```

### Testing Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | (required) | Path to model checkpoint |
| `--dataset_dir` | (required) | Path to dataset directory |
| `--output_dir` | `./results` | Directory to save generated CT images |
| `--image_size` | 256 | Image size (square) |
| `--batch_size` | 1 | Batch size for inference |
| `--num_inference_steps` | 1000 | Number of inference steps (use 50-250 for faster sampling) |
| `--num_workers` | 4 | Number of data loading workers |
| `--sampler` | `ddpm` | Sampling method: `ddpm` or `ddim` |
| `--ddim_eta` | 0.0 | DDIM eta parameter (0=deterministic, 1=stochastic like DDPM) |
| `--mixed_precision` | `fp16` | Mixed precision mode (`no`, `fp16`, `bf16`) |

## Data Preprocessing

- Images are loaded in grayscale (single channel)
- Images are resized to the specified image size (default: 256x256)
- Pixel values are normalized to [0, 1] range
- Generated images are de-normalized back to [0, 255] before saving

## Model Architecture

The model uses MONAI's `DiffusionModelUNet` with:
- Spatial dimensions: 2D
- Input channels: 2 (CT xt + MR condition)
- Output channels: 1 (epsilon)
- Channel configuration: (128, 256, 256)
- Attention levels: (False, True, True)
- Number of residual blocks: (1, 1, 1)
- Number of head channels: 256

## References

- [MONAI Generative Models](https://github.com/Project-MONAI/GenerativeModels)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Denoising Diffusion Implicit Models (DDIM)](https://arxiv.org/abs/2010.02502)
