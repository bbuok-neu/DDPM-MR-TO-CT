#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MR-to-CT Conditional DDPM Testing/Inference Script

This script performs inference using a trained conditional DDPM model
to generate CT images from MR images.

Usage:
    python test.py --checkpoint /path/to/checkpoint.pt --dataset_dir /path/to/dataset --output_dir /path/to/output

The script will:
1. Load test MR images from dataset_dir/mr/test/
2. Generate corresponding CT images using the trained DDPM model
3. Save the generated CT images (de-normalized to [0, 255]) to output_dir
"""

import os
import argparse
import glob

import torch
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm

from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler


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


def sample(model, scheduler, mr_condition, device, num_inference_steps=1000):
    """
    Sample CT image from MR condition using DDPM reverse process.
    
    Args:
        model: Trained DDPM model
        scheduler: DDPM scheduler
        mr_condition: MR image condition tensor (B, 1, H, W)
        device: Device to use
        num_inference_steps: Number of inference steps
    
    Returns:
        Generated CT image tensor (B, 1, H, W)
    """
    model.eval()
    
    batch_size = mr_condition.shape[0]
    height = mr_condition.shape[2]
    width = mr_condition.shape[3]
    
    # Start from random noise
    ct_sample = torch.randn((batch_size, 1, height, width), device=device)
    
    # Set timesteps
    scheduler.set_timesteps(num_inference_steps=num_inference_steps)
    
    # Reverse diffusion process
    with torch.no_grad():
        for t in tqdm(scheduler.timesteps, desc="Sampling", leave=False):
            # Create timestep tensor
            timestep = torch.tensor([t] * batch_size, device=device).long()
            
            # Concatenate current sample with MR condition
            model_input = torch.cat([ct_sample, mr_condition], dim=1)
            
            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                # Get model prediction (predicted noise)
                noise_pred = model(model_input, timestep)
            
            # Perform one step of the reverse diffusion
            ct_sample, _ = scheduler.step(noise_pred, t, ct_sample)
    
    return ct_sample


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
                        help="Number of inference steps")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    
    return parser.parse_args()


def main():
    args = get_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
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
    model = model.to(device)
    model.eval()
    print("Model loaded successfully")
    
    # Create scheduler
    scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)
    
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
    
    # Inference loop
    print(f"Starting inference with {args.num_inference_steps} steps...")
    
    for batch in tqdm(test_loader, desc="Processing"):
        mr_images = batch["mr"].to(device)
        filenames = batch["filename"]
        
        # Sample CT from MR
        generated_ct = sample(
            model, scheduler, mr_images, device,
            num_inference_steps=args.num_inference_steps
        )
        
        # Save each generated image
        for i, filename in enumerate(filenames):
            # Change extension to indicate it's generated CT
            base_name = os.path.splitext(filename)[0]
            save_path = os.path.join(args.output_dir, f"{base_name}_generated_ct.jpg")
            
            denormalize_and_save(generated_ct[i], save_path)
    
    print(f"Inference complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
