#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import numpy as np
import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from functools import partial
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Import PyTorch MAE model
from mae_models import MaskedAutoencoderViT
from pos_embed import get_2d_sincos_pos_embed, get_2d_sincos_pos_embed_rect

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train a Masked Autoencoder using PyTorch')
parser.add_argument('--data_dir', type=str, default='input/MAE_dataset', help='Input data directory')
parser.add_argument('--output_dir', type=str, default='output', help='Output directory for models')
args = parser.parse_args()

# Setting seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 128
IMAGE_SIZE = (128, 128)  # Match dimensions from TensorFlow version
PATCH_SIZE = 8           # Match from TensorFlow version
MASK_RATIO = 0.4        # 40% masking ratio
CHANNELS = 1            # Grayscale images
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 20
WARMUP_EPOCHS = int(EPOCHS * 0.15)

# Create output directories
os.makedirs(args.output_dir, exist_ok=True)
model_dir = os.path.join(args.output_dir, 'mae_model')
weights_dir = os.path.join(args.output_dir, 'weights')
vis_dir = os.path.join(args.output_dir, 'visualizations')
os.makedirs(model_dir, exist_ok=True)
os.makedirs(weights_dir, exist_ok=True)
os.makedirs(vis_dir, exist_ok=True)

# Dataset class for loading numpy files
class NumpyDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        # Load the .npy file
        img = np.load(file_path).astype(np.float32)
        
        # Normalize 16-bit data to [0, 1] range
        img = img / 65535.0
        
        # Add channel dimension if needed
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)  # [H, W] -> [1, H, W]
            
        # Pad or resize to target dimensions
        if img.shape[1] != IMAGE_SIZE[0] or img.shape[2] != IMAGE_SIZE[1]:
            # Calculate scaling factor to fit within IMAGE_SIZE while preserving aspect ratio
            height, width = img.shape[1], img.shape[2]
            scale = min(IMAGE_SIZE[0] / height, IMAGE_SIZE[1] / width)
            
            # Calculate new dimensions after scaling
            new_height = int(height * scale)
            new_width = int(width * scale)
            
            # Create a blank canvas of target size
            padded_img = np.zeros((CHANNELS, IMAGE_SIZE[0], IMAGE_SIZE[1]), dtype=np.float32)
            
            # Use simple resizing for numpy arrays
            from PIL import Image
            pil_img = Image.fromarray((img[0] * 255).astype(np.uint8))
            pil_img = pil_img.resize((new_width, new_height), Image.BILINEAR)
            resized_array = np.array(pil_img).astype(np.float32) / 255.0
            
            # Add channel dimension back
            resized_array = np.expand_dims(resized_array, axis=0)
            
            # Calculate padding to center the image
            y_offset = (IMAGE_SIZE[0] - new_height) // 2
            x_offset = (IMAGE_SIZE[1] - new_width) // 2
            
            # Place the resized image on the canvas
            padded_img[:, y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_array
            
            img = padded_img
        
        # Convert to torch tensor
        img = torch.from_numpy(img).float()
        
        # Apply additional transforms if needed
        if self.transform:
            img = self.transform(img)
            
        return img

# Split the data
all_files = glob(os.path.join(args.data_dir, "*.npy"))
print(f"Found {len(all_files)} .npy files in {args.data_dir}")

train_files, test_files = train_test_split(all_files, test_size=0.1, random_state=SEED)
train_files, val_files = train_test_split(train_files, test_size=0.11, random_state=SEED)

print(f"Training files: {len(train_files)}")
print(f"Validation files: {len(val_files)}")
print(f"Test files: {len(test_files)}")

# Base directory for saving dataset splits
base_dir = 'data'
os.makedirs(base_dir, exist_ok=True)

# Save the file lists for reference
with open(os.path.join(base_dir, 'train_files.txt'), 'w') as f:
    for file in train_files:
        f.write(f"{file}\n")

with open(os.path.join(base_dir, 'validation_files.txt'), 'w') as f:
    for file in val_files:
        f.write(f"{file}\n")

with open(os.path.join(base_dir, 'test_files.txt'), 'w') as f:
    for file in test_files:
        f.write(f"{file}\n")

# Create datasets and dataloaders
train_dataset = NumpyDataset(file_paths=train_files)
val_dataset = NumpyDataset(file_paths=val_files)
test_dataset = NumpyDataset(file_paths=test_files)

# DataLoaders with multi-processing
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# Initialize the MAE model
model = MaskedAutoencoderViT(
    img_size=IMAGE_SIZE,
    patch_size=PATCH_SIZE, 
    in_chans=CHANNELS,
    embed_dim=96,                # Encoder embedding dimension 
    depth=12,                    # Encoder depth
    num_heads=8,                 # Encoder number of heads
    decoder_embed_dim=64,        # Decoder embedding dimension
    decoder_depth=4,             # Decoder depth
    decoder_num_heads=8,         # Decoder number of heads
    mlp_ratio=4,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
    norm_pix_loss=False          # Whether to normalize pixel values in loss
)

# Move model to device
model = model.to(device)

# Learning rate schedule function
def get_lr(epoch, warmup_epochs, base_lr):
    if epoch < warmup_epochs:
        # Linear warmup
        return base_lr * epoch / warmup_epochs
    else:
        # Cosine annealing
        progress = (epoch - warmup_epochs) / (EPOCHS - warmup_epochs)
        return base_lr * 0.5 * (1. + np.cos(np.pi * progress))

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Prepare for visualization
def visualize_reconstruction(original_images, pred, mask, epoch, save_path):
    """
    Create a visualization of original, masked, and reconstructed images.
    
    Args:
        original_images: Tensor [B, C, H, W]
        pred: Tensor of predicted patches [B, L, P^2*C]
        mask: Binary mask [B, L] (0=keep, 1=remove)
        epoch: Current epoch number
        save_path: Path to save the visualization
    """
    # Select a random image from the batch
    idx = random.randint(0, original_images.shape[0] - 1)
    
    # Get images
    original = original_images[idx].cpu().detach()
    original_img = original.permute(1, 2, 0).squeeze().numpy()  # [H, W]
    
    # Reconstruct the predicted image
    pred_img = model.unpatchify(pred).cpu().detach()
    pred_img = pred_img[idx].permute(1, 2, 0).squeeze().numpy()  # [H, W]
    
    # Create a masked version of the original image
    masked_img = original_img.copy()
    
    # Apply mask to the image
    mask_idx = mask[idx].cpu().detach().bool().numpy()
    patches = model.patchify(original_images.cpu()).cpu().detach()
    patch_size = model.patch_embed.patch_size[0]
    
    # Create visual representation of the mask
    mask_vis = np.ones_like(original_img)
    h_patches = IMAGE_SIZE[0] // patch_size
    w_patches = IMAGE_SIZE[1] // patch_size
    
    for i in range(len(mask_idx)):
        if mask_idx[i]:  # If patch is masked
            # Convert flat index to 2D patch coordinates
            pi, pj = i // w_patches, i % w_patches
            y_start, x_start = pi * patch_size, pj * patch_size
            
            # Set the patch area to a masked value in the visualization
            masked_img[y_start:y_start+patch_size, x_start:x_start+patch_size] = 0.5
            mask_vis[y_start:y_start+patch_size, x_start:x_start+patch_size] = 0.7
    
    # Create the visualization
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    
    # Original image
    axs[0].imshow(original_img, cmap='gray')
    axs[0].set_title(f'Original (Epoch {epoch})')
    axs[0].axis('off')
    
    # Masked image
    axs[1].imshow(original_img, cmap='gray')
    axs[1].imshow(mask_vis, cmap='Blues', alpha=0.5)
    axs[1].set_title('Masked')
    axs[1].axis('off')
    
    # Reconstructed image
    axs[2].imshow(pred_img, cmap='gray')
    axs[2].set_title('Reconstructed')
    axs[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

# Training loop
print(f"Starting training for {EPOCHS} epochs...")
train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    # Adjust learning rate according to schedule
    lr = get_lr(epoch, WARMUP_EPOCHS, LEARNING_RATE)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # Training phase
    model.train()
    train_loss = 0.0
    
    train_loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] - Train")
    for batch_idx, images in enumerate(train_loop):
        images = images.to(device)
        
        # Forward pass with fixed mask_ratio
        loss, pred, mask = model(images, mask_ratio=MASK_RATIO)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        train_loss += loss.item()
        train_loop.set_postfix(loss=loss.item(), lr=lr)
    
    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        val_loop = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] - Val")
        for batch_idx, images in enumerate(val_loop):
            images = images.to(device)
            
            # Forward pass
            loss, pred, mask = model(images, mask_ratio=MASK_RATIO)
            
            # Update metrics
            val_loss += loss.item()
            val_loop.set_postfix(loss=loss.item())
    
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {lr:.6f}")
    
    # Visualize every 5 epochs
    if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == EPOCHS - 1:
        # Visualize reconstruction
        with torch.no_grad():
            images = next(iter(test_loader)).to(device)
            _, pred, mask = model(images, mask_ratio=MASK_RATIO)
            
            # Save visualization
            vis_path = os.path.join(vis_dir, f'reconstruction_epoch_{epoch+1}.png')
            visualize_reconstruction(images, pred, mask, epoch+1, vis_path)
    
    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
    }, os.path.join(model_dir, f'checkpoint_epoch_{epoch+1}.pt'))
    
    # Save latest model (overwrite)
    torch.save(model.state_dict(), os.path.join(weights_dir, 'latest_model.pt'))

# Save final model
torch.save(model.state_dict(), os.path.join(weights_dir, 'final_model.pt'))

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(vis_dir, 'loss_curve.png'))
plt.close()

# Final test evaluation
model.eval()
test_loss = 0.0

with torch.no_grad():
    test_loop = tqdm(test_loader, desc="Final Test Evaluation")
    for images in test_loop:
        images = images.to(device)
        loss, _, _ = model(images, mask_ratio=MASK_RATIO)
        test_loss += loss.item()

avg_test_loss = test_loss / len(test_loader)
print(f"Final Test Loss: {avg_test_loss:.6f}")

# Final visualization
with torch.no_grad():
    test_images = []
    test_preds = []
    test_masks = []
    
    # Get samples from test set
    for images in test_loader:
        if len(test_images) < 4:  # Get up to 4 batches for visualization
            images = images.to(device)
            _, pred, mask = model(images, mask_ratio=MASK_RATIO)
            
            # Store results
            test_images.append(images)
            test_preds.append(pred)
            test_masks.append(mask)
    
    # Create a grid of visualizations
    fig, axs = plt.subplots(4, 3, figsize=(12, 16))
    
    for i in range(4):
        idx = random.randint(0, test_images[i].shape[0] - 1)
        
        # Get images for this sample
        original = test_images[i][idx].cpu().detach()
        original_img = original.permute(1, 2, 0).squeeze().numpy()
        
        pred_img = model.unpatchify(test_preds[i]).cpu().detach()
        pred_img = pred_img[idx].permute(1, 2, 0).squeeze().numpy()
        
        # Create a masked version for visualization
        mask_idx = test_masks[i][idx].cpu().detach().bool().numpy()
        mask_vis = np.ones_like(original_img)
        patch_size = model.patch_embed.patch_size[0]
        h_patches = IMAGE_SIZE[0] // patch_size
        w_patches = IMAGE_SIZE[1] // patch_size
        
        for j in range(len(mask_idx)):
            if mask_idx[j]:
                pi, pj = j // w_patches, j % w_patches
                y_start, x_start = pi * patch_size, pj * patch_size
                mask_vis[y_start:y_start+patch_size, x_start:x_start+patch_size] = 0.7
        
        # Plot
        axs[i, 0].imshow(original_img, cmap='gray')
        axs[i, 0].axis('off')
        
        axs[i, 1].imshow(original_img, cmap='gray')
        axs[i, 1].imshow(mask_vis, cmap='Blues', alpha=0.5)
        axs[i, 1].axis('off')
        
        axs[i, 2].imshow(pred_img, cmap='gray')
        axs[i, 2].axis('off')
        
        if i == 0:
            axs[i, 0].set_title('Original')
            axs[i, 1].set_title('Masked')
            axs[i, 2].set_title('Reconstructed')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'final_test_samples.png'))
    plt.close()

print(f"Training complete! Models saved to {args.output_dir}")
print(f"Model checkpoints: {model_dir}")
print(f"Model weights: {weights_dir}")
print(f"Visualizations: {vis_dir}")