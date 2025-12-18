#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MR-to-CT Conditional DDPM Training Script

This script trains a conditional DDPM model for MR-to-CT medical image synthesis.
The model takes 2-channel input (CT noisy image xt concatenated with MR condition)
and outputs 1-channel (predicted noise epsilon).

Uses Hugging Face Accelerate for mixed precision training and distributed training support.

Usage:
    python train.py --dataset_dir /path/to/dataset --output_dir /path/to/output

    # With Accelerate launcher for multi-GPU/distributed training:
    accelerate launch train.py --dataset_dir /path/to/dataset --output_dir /path/to/output

Dataset structure:
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

Note: MR and CT images are paired by sorting filenames. The filenames don't need to match.
"""

import os
import argparse
import time
import glob

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed

from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler


class MRCTDataset(Dataset):
    """
    Dataset for paired MR-CT images.
    
    Images are loaded in grayscale, normalized to [0, 1].
    MR and CT images are paired by sorting filenames.
    """
    
    def __init__(self, mr_dir, ct_dir, image_size=256):
        """
        Args:
            mr_dir: Directory containing MR images
            ct_dir: Directory containing CT images
            image_size: Size to resize images to (square)
        """
        self.image_size = image_size
        
        # Get sorted file lists
        mr_files = sorted(glob.glob(os.path.join(mr_dir, "*.jpg")))
        ct_files = sorted(glob.glob(os.path.join(ct_dir, "*.jpg")))
        
        # Also check for other common image formats
        if len(mr_files) == 0:
            mr_files = sorted(glob.glob(os.path.join(mr_dir, "*.png")))
        if len(ct_files) == 0:
            ct_files = sorted(glob.glob(os.path.join(ct_dir, "*.png")))
        
        # Verify equal counts
        assert len(mr_files) == len(ct_files), \
            f"Number of MR images ({len(mr_files)}) must equal number of CT images ({len(ct_files)})"
        
        assert len(mr_files) > 0, f"No images found in {mr_dir}"
        
        self.mr_files = mr_files
        self.ct_files = ct_files
        
        print(f"Loaded {len(self.mr_files)} paired MR-CT images")
    
    def __len__(self):
        return len(self.mr_files)
    
    def __getitem__(self, idx):
        # Load images in grayscale
        mr_img = Image.open(self.mr_files[idx]).convert('L')
        ct_img = Image.open(self.ct_files[idx]).convert('L')
        
        # Resize
        mr_img = mr_img.resize((self.image_size, self.image_size), Image.BILINEAR)
        ct_img = ct_img.resize((self.image_size, self.image_size), Image.BILINEAR)
        
        # Convert to numpy and normalize to [0, 1]
        mr_np = np.array(mr_img, dtype=np.float32) / 255.0
        ct_np = np.array(ct_img, dtype=np.float32) / 255.0
        
        # Add channel dimension: (H, W) -> (1, H, W)
        mr_tensor = torch.from_numpy(mr_np).unsqueeze(0)
        ct_tensor = torch.from_numpy(ct_np).unsqueeze(0)
        
        return {"mr": mr_tensor, "ct": ct_tensor}


def get_args():
    parser = argparse.ArgumentParser(description="Train MR-to-CT Conditional DDPM")
    
    # Data arguments
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to dataset directory containing mr/ and ct/ subdirectories")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Image size (square)")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2.5e-5,
                        help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    
    # DDPM arguments
    parser.add_argument("--num_train_timesteps", type=int, default=1000,
                        help="Number of diffusion timesteps")
    
    # Resume training
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    
    # Validation
    parser.add_argument("--val_interval", type=int, default=5,
                        help="Validation interval (epochs)")
    parser.add_argument("--enable_validation", action="store_true",
                        help="Enable online validation during training")
    
    # Checkpoint
    parser.add_argument("--save_interval", type=int, default=10,
                        help="Checkpoint save interval (epochs)")
    
    # Mixed precision
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16"],
                        help="Mixed precision training mode")
    
    # Random seed
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    return parser.parse_args()


def main():
    args = get_args()
    
    # Initialize Accelerator for mixed precision training
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=1,
    )
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory (only on main process)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    accelerator.wait_for_everyone()
    
    # Print device info
    accelerator.print(f"Using device: {accelerator.device}")
    accelerator.print(f"Mixed precision: {args.mixed_precision}")
    
    # Create datasets
    train_mr_dir = os.path.join(args.dataset_dir, "mr", "train")
    train_ct_dir = os.path.join(args.dataset_dir, "ct", "train")
    
    train_dataset = MRCTDataset(train_mr_dir, train_ct_dir, args.image_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create validation dataset if enabled
    val_loader = None
    if args.enable_validation:
        test_mr_dir = os.path.join(args.dataset_dir, "mr", "test")
        test_ct_dir = os.path.join(args.dataset_dir, "ct", "test")
        
        if os.path.exists(test_mr_dir) and os.path.exists(test_ct_dir):
            val_dataset = MRCTDataset(test_mr_dir, test_ct_dir, args.image_size)
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True
            )
            accelerator.print(f"Validation enabled with {len(val_dataset)} samples")
        else:
            accelerator.print("Warning: Test directory not found, validation disabled")
    
    # Create model
    # 2-channel input: CT noisy image (xt) + MR condition
    # 1-channel output: predicted noise (epsilon)
    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=2,  # CT xt + MR condition
        out_channels=1,  # epsilon
        channels=(128, 256, 256),
        attention_levels=(False, True, True),
        num_res_blocks=(1, 1, 1),
        num_head_channels=256,
    )
    
    # Create scheduler
    scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        if os.path.exists(args.resume):
            accelerator.print(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=accelerator.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            accelerator.print(f"Resumed from epoch {start_epoch}")
        else:
            accelerator.print(f"Warning: Checkpoint {args.resume} not found, starting from scratch")
    
    # Prepare model, optimizer, and dataloaders with Accelerator
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)
    
    # Training loop
    accelerator.print(f"Starting training from epoch {start_epoch}")
    total_start = time.time()
    
    epoch_loss_list = []
    val_loss_list = []
    
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        epoch_loss = 0.0
        
        progress_bar = tqdm(
            enumerate(train_loader), 
            total=len(train_loader), 
            ncols=80,
            disable=not accelerator.is_main_process
        )
        progress_bar.set_description(f"Epoch {epoch}")
        
        for step, batch in progress_bar:
            mr_images = batch["mr"]  # (B, 1, H, W)
            ct_images = batch["ct"]  # (B, 1, H, W)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Mixed precision is handled automatically by Accelerator
            # Generate random noise
            noise = torch.randn_like(ct_images)
            
            # Create random timesteps
            timesteps = torch.randint(
                0, args.num_train_timesteps,
                (ct_images.shape[0],),
                device=ct_images.device
            ).long()
            
            # Add noise to CT images (forward diffusion)
            noisy_ct = scheduler.add_noise(
                original_samples=ct_images,
                noise=noise,
                timesteps=timesteps
            )
            
            # Concatenate noisy CT with MR condition
            # Input: (B, 2, H, W) = [noisy_ct, mr]
            model_input = torch.cat([noisy_ct, mr_images], dim=1)
            
            # Get model prediction (predicted noise)
            noise_pred = model(model_input, timesteps)
            
            # Compute loss
            loss = F.mse_loss(noise_pred.float(), noise.float())
            
            # Backward pass with Accelerator
            accelerator.backward(loss)
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        epoch_loss_list.append(avg_epoch_loss)
        accelerator.print(f"Epoch {epoch} - Average Loss: {avg_epoch_loss:.6f}")
        
        # Validation
        if args.enable_validation and val_loader is not None and (epoch + 1) % args.val_interval == 0:
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for step, batch in enumerate(val_loader):
                    mr_images = batch["mr"]
                    ct_images = batch["ct"]
                    
                    noise = torch.randn_like(ct_images)
                    timesteps = torch.randint(
                        0, args.num_train_timesteps,
                        (ct_images.shape[0],),
                        device=ct_images.device
                    ).long()
                    
                    noisy_ct = scheduler.add_noise(
                        original_samples=ct_images,
                        noise=noise,
                        timesteps=timesteps
                    )
                    
                    model_input = torch.cat([noisy_ct, mr_images], dim=1)
                    noise_pred = model(model_input, timesteps)
                    loss = F.mse_loss(noise_pred.float(), noise.float())
                    
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_loss_list.append(avg_val_loss)
            accelerator.print(f"Epoch {epoch} - Validation Loss: {avg_val_loss:.6f}")
        
        # Save checkpoint (only on main process)
        if accelerator.is_main_process:
            if (epoch + 1) % args.save_interval == 0 or epoch == args.num_epochs - 1:
                # Unwrap model for saving
                unwrapped_model = accelerator.unwrap_model(model)
                
                checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt")
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": unwrapped_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_epoch_loss,
                    "args": vars(args)
                }
                
                torch.save(checkpoint, checkpoint_path)
                accelerator.print(f"Saved checkpoint: {checkpoint_path}")
                
                # Also save as latest
                latest_path = os.path.join(args.output_dir, "checkpoint_latest.pt")
                torch.save(checkpoint, latest_path)
        
        accelerator.wait_for_everyone()
    
    total_time = time.time() - total_start
    accelerator.print(f"Training completed in {total_time:.2f} seconds")
    
    # Save final model (only on main process)
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        final_path = os.path.join(args.output_dir, "model_final.pt")
        torch.save({
            "model_state_dict": unwrapped_model.state_dict(),
            "args": vars(args)
        }, final_path)
        accelerator.print(f"Saved final model: {final_path}")


if __name__ == "__main__":
    main()
