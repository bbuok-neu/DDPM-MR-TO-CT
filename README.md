# MR-to-CT Medical Image Synthesis using Conditional DDPM

This project implements a conditional Denoising Diffusion Probabilistic Model (DDPM) for MR-to-CT medical image synthesis using MONAI.

## Overview

The model performs modality transfer from MR (Magnetic Resonance) images to CT (Computed Tomography) images using a conditional diffusion model. The architecture uses:

- **Input**: 2 channels (CT noisy image xt concatenated with MR condition)
- **Output**: 1 channel (predicted noise epsilon)
- **Model**: DiffusionModelUNet from MONAI

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

Basic training:
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

## Testing/Inference

Generate CT images from test MR images:
```bash
python test.py --checkpoint ./checkpoints/model_final.pt --dataset_dir /path/to/dataset --output_dir ./results
```

### Testing Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | (required) | Path to model checkpoint |
| `--dataset_dir` | (required) | Path to dataset directory |
| `--output_dir` | `./results` | Directory to save generated CT images |
| `--image_size` | 256 | Image size (square) |
| `--batch_size` | 1 | Batch size for inference |
| `--num_inference_steps` | 1000 | Number of inference steps |
| `--num_workers` | 4 | Number of data loading workers |

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
