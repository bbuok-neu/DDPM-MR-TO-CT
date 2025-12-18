#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MR-to-CT Conditional DDPM Testing/Inference Script

This script performs inference using a trained conditional DDPM model
to generate CT images from MR images.

Uses Hugging Face Accelerate for mixed precision inference.
Supports DDPM and DDIM samplers with timestep respacing for faster sampling.

Usage:
    python test.py --checkpoint /path/to/checkpoint.pt --dataset_dir /path/to/dataset --output_dir /path/to/output

    # Fast sampling with DDIM (50 steps instead of 1000)
    python test.py --checkpoint /path/to/checkpoint.pt --dataset_dir /path/to/dataset --sampler ddim --num_inference_steps 50

The script will:
1. Load test MR images from dataset_dir/mr/test/
2. Generate corresponding CT images using the trained DDPM model
3. Save the generated CT images (de-normalized to [0, 255]) to output_dir
"""

import os
import argparse
import glob

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm

from accelerate import Accelerator

from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler, DDIMScheduler


class MRTestDataset(Dataset):
    """
    Dataset for MR test images (inference only).
    """
    
    def __init__(self, mr_dir, image_size=256):
        """
        Args:
            mr_dir: Directory containing MR images
            image_size: Size to resize images to (square)
        """
        self.image_size = image_size
        
        # Get sorted file lists
        mr_files = sorted(glob.glob(os.path.join(mr_dir, "*.jpg")))
        if len(mr_files) == 0:
            mr_files = sorted(glob.glob(os.path.join(mr_dir, "*.png")))
        
        assert len(mr_files) > 0, f"No images found in {mr_dir}"
        
        self.mr_files = mr_files
        print(f"Loaded {len(self.mr_files)} MR test images")
    
    def __len__(self):
        return len(self.mr_files)
    
    def __getitem__(self, idx):
        # Load image in grayscale
        mr_img = Image.open(self.mr_files[idx]).convert('L')
        
        # Resize
        mr_img = mr_img.resize((self.image_size, self.image_size), Image.BILINEAR)
        
        # Convert to numpy and normalize to [0, 1]
        mr_np = np.array(mr_img, dtype=np.float32) / 255.0
        
        # Add channel dimension: (H, W) -> (1, H, W)
        mr_tensor = torch.from_numpy(mr_np).unsqueeze(0)
        
        # Get filename for saving
        filename = os.path.basename(self.mr_files[idx])
        
        return {"mr": mr_tensor, "filename": filename}


def sample_ddpm(model, scheduler, mr_condition, accelerator, num_inference_steps=1000):
    """
    Sample CT image from MR condition using DDPM reverse process.
    
    Args:
        model: Trained DDPM model
        scheduler: DDPM scheduler
        mr_condition: MR image condition tensor (B, 1, H, W)
        accelerator: Accelerator instance for mixed precision
        num_inference_steps: Number of inference steps (timestep respacing)
    
    Returns:
        Generated CT image tensor (B, 1, H, W)
    """
    model.eval()
    
    batch_size = mr_condition.shape[0]
    height = mr_condition.shape[2]
    width = mr_condition.shape[3]
    
    # Start from random noise
    ct_sample = torch.randn((batch_size, 1, height, width), device=mr_condition.device)
    
    # Set timesteps (timestep respacing)
    scheduler.set_timesteps(num_inference_steps=num_inference_steps)
    
    # Reverse diffusion process
    with torch.no_grad():
        for t in tqdm(scheduler.timesteps, desc="DDPM Sampling", leave=False, disable=not accelerator.is_main_process):
            # Create timestep tensor
            timestep = torch.tensor([t] * batch_size, device=mr_condition.device).long()
            
            # Concatenate current sample with MR condition
            model_input = torch.cat([ct_sample, mr_condition], dim=1)
            
            # Get model prediction (predicted noise)
            noise_pred = model(model_input, timestep)
            
            # Perform one step of the reverse diffusion
            ct_sample, _ = scheduler.step(noise_pred, t, ct_sample)
    
    return ct_sample


def sample_ddim(model, scheduler, mr_condition, accelerator, num_inference_steps=50, eta=0.0):
    """
    Sample CT image from MR condition using DDIM reverse process.
    
    DDIM (Denoising Diffusion Implicit Models) allows for faster sampling
    with fewer steps while maintaining quality.
    
    Args:
        model: Trained DDPM model
        scheduler: DDIM scheduler
        mr_condition: MR image condition tensor (B, 1, H, W)
        accelerator: Accelerator instance for mixed precision
        num_inference_steps: Number of inference steps (can be much smaller than training steps)
        eta: DDIM eta parameter (0 = deterministic, 1 = DDPM-like stochastic)
    
    Returns:
        Generated CT image tensor (B, 1, H, W)
    """
    model.eval()
    
    batch_size = mr_condition.shape[0]
    height = mr_condition.shape[2]
    width = mr_condition.shape[3]
    
    # Start from random noise
    ct_sample = torch.randn((batch_size, 1, height, width), device=mr_condition.device)
    
    # Set timesteps (timestep respacing for accelerated sampling)
    scheduler.set_timesteps(num_inference_steps=num_inference_steps)
    
    # DDIM reverse diffusion process
    with torch.no_grad():
        for t in tqdm(scheduler.timesteps, desc="DDIM Sampling", leave=False, disable=not accelerator.is_main_process):
            # Create timestep tensor
            timestep = torch.tensor([t] * batch_size, device=mr_condition.device).long()
            
            # Concatenate current sample with MR condition
            model_input = torch.cat([ct_sample, mr_condition], dim=1)
            
            # Get model prediction (predicted noise)
            noise_pred = model(model_input, timestep)
            
            # Perform one step of the DDIM reverse diffusion
            ct_sample, _ = scheduler.step(noise_pred, t, ct_sample, eta=eta)
    
    return ct_sample


def sample(model, scheduler, mr_condition, accelerator, num_inference_steps=1000, 
           sampler_type="ddpm", eta=0.0):
    """
    Sample CT image from MR condition using specified sampler.
    
    Args:
        model: Trained DDPM model
        scheduler: Scheduler (DDPM or DDIM)
        mr_condition: MR image condition tensor (B, 1, H, W)
        accelerator: Accelerator instance for mixed precision
        num_inference_steps: Number of inference steps
        sampler_type: "ddpm" or "ddim"
        eta: DDIM eta parameter (only used for DDIM)
    
    Returns:
        Generated CT image tensor (B, 1, H, W)
    """
    if sampler_type == "ddim":
        return sample_ddim(model, scheduler, mr_condition, accelerator, 
                          num_inference_steps, eta)
    else:
        return sample_ddpm(model, scheduler, mr_condition, accelerator, 
                          num_inference_steps)


def denormalize_and_save(tensor, save_path):
    """
    De-normalize tensor from [0, 1] to [0, 255] and save as image.
    
    Args:
        tensor: Image tensor (1, H, W) in range [0, 1]
        save_path: Path to save the image
    """
    # Clamp to [0, 1] range
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy and de-normalize to [0, 255]
    img_np = (tensor.squeeze().cpu().numpy() * 255.0).astype(np.uint8)
    
    # Save as image
    img = Image.fromarray(img_np, mode='L')
    img.save(save_path)


def get_args():
    parser = argparse.ArgumentParser(description="Test MR-to-CT Conditional DDPM")
    
    # Model arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    
    # Data arguments
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to dataset directory containing mr/test/ subdirectory")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save generated CT images")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Image size (square)")
    
    # Inference arguments
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for inference")
    parser.add_argument("--num_inference_steps", type=int, default=1000,
                        help="Number of inference steps (use smaller values like 50-250 for faster sampling)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    
    # Sampler arguments
    parser.add_argument("--sampler", type=str, default="ddpm",
                        choices=["ddpm", "ddim"],
                        help="Sampling method: ddpm (slower, stochastic) or ddim (faster, can be deterministic)")
    parser.add_argument("--ddim_eta", type=float, default=0.0,
                        help="DDIM eta parameter (0=deterministic, 1=DDPM-like). Only used with --sampler ddim")
    
    # Mixed precision
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16"],
                        help="Mixed precision inference mode")
    
    return parser.parse_args()


def main():
    args = get_args()
    
    # Initialize Accelerator for mixed precision inference
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )
    
    # Create output directory (only on main process)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    accelerator.wait_for_everyone()
    
    accelerator.print(f"Using device: {accelerator.device}")
    accelerator.print(f"Mixed precision: {args.mixed_precision}")
    
    # Load checkpoint
    accelerator.print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=accelerator.device)
    
    # Get model args from checkpoint if available
    if "args" in checkpoint:
        saved_args = checkpoint["args"]
        image_size = saved_args.get("image_size", args.image_size)
        num_train_timesteps = saved_args.get("num_train_timesteps", 1000)
    else:
        image_size = args.image_size
        num_train_timesteps = 1000
    
    # Create model
    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=2,  # CT xt + MR condition
        out_channels=1,  # epsilon
        channels=(128, 256, 256),
        attention_levels=(False, True, True),
        num_res_blocks=(1, 1, 1),
        num_head_channels=256,
    )
    
    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    accelerator.print("Model loaded successfully")
    
    # Create scheduler based on sampler type
    if args.sampler == "ddim":
        scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps)
        accelerator.print(f"Using DDIM sampler with eta={args.ddim_eta}")
    else:
        scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)
        accelerator.print("Using DDPM sampler")
    
    # Create test dataset
    test_mr_dir = os.path.join(args.dataset_dir, "mr", "test")
    test_dataset = MRTestDataset(test_mr_dir, image_size)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Prepare model and dataloader with Accelerator
    model, test_loader = accelerator.prepare(model, test_loader)
    
    # Inference loop
    accelerator.print(f"Starting inference with {args.num_inference_steps} steps using {args.sampler.upper()}...")
    
    for batch in tqdm(test_loader, desc="Processing", disable=not accelerator.is_main_process):
        mr_images = batch["mr"]
        filenames = batch["filename"]
        
        # Sample CT from MR
        generated_ct = sample(
            model, scheduler, mr_images, accelerator,
            num_inference_steps=args.num_inference_steps,
            sampler_type=args.sampler,
            eta=args.ddim_eta
        )
        
        # Save each generated image (only on main process)
        if accelerator.is_main_process:
            for i, filename in enumerate(filenames):
                # Change extension to indicate it's generated CT
                base_name = os.path.splitext(filename)[0]
                save_path = os.path.join(args.output_dir, f"{base_name}_generated_ct.jpg")
                
                denormalize_and_save(generated_ct[i], save_path)
    
    accelerator.print(f"Inference complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
