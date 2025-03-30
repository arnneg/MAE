#!/usr/bin/env python
# coding: utf-8
# # Pre-training a Masked Autoencoder with Keras

# !pip install -U tensorflow-addons
# !pip install tensorflow

from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import tensorflow_addons as tfa  # Import tensorflow_addons for AdamW

import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import random

import os
import shutil
import argparse
from sklearn.model_selection import train_test_split
import glob
import pandas as pd

# Define the debug CSV generation function here, before it's used
def generate_patch_debug_csv(patches, mask_indices, image_idx, save_dir, prefix="debug"):
    """Generate a CSV file for debugging patch selection"""
    # Get the number of patches per dimension
    num_patches = patches.shape[1]
    patch_dim = int(np.sqrt(num_patches))
    
    # Calculate patch means
    patch_means = np.mean(np.abs(patches[image_idx]), axis=1)
    
    # Create a mask status array (1 if masked, 0 if not)
    mask_status = np.zeros(num_patches)
    mask_status[mask_indices[image_idx]] = 1
    
    # Create a dataframe
    rows = []
    for i in range(num_patches):
        # Calculate row and column position
        row = i // patch_dim
        col = i % patch_dim
        
        rows.append({
            'patch_idx': i,
            'row': row,
            'col': col,
            'avg_value': patch_means[i],
            'is_masked': mask_status[i],
            'threshold_check': patch_means[i] > 0.1  # Same threshold as in the smart_indices function
        })
    
    df = pd.DataFrame(rows)
    
    # Save to CSV
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f'{prefix}_patches_image_{image_idx}.csv')
    df.to_csv(csv_path, index=False)
    
    # Generate a heatmap image showing the patch values
    plt.figure(figsize=(8, 8))
    heatmap = df.pivot(index='row', columns='col', values='avg_value')
    plt.imshow(heatmap, cmap='viridis')
    plt.colorbar(label='Average Patch Value')
    
    # Overlay the masked patches with X marks
    masked_df = df[df['is_masked'] == 1]
    plt.scatter(masked_df['col'], masked_df['row'], marker='x', color='red', s=100)
    
    plt.title(f'Patch Values and Masked Regions (Image {image_idx})')
    plt.savefig(os.path.join(save_dir, f'{prefix}_heatmap_image_{image_idx}.png'))
    plt.close()
    
    # Don't print for every file - too verbose
    return df

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train a Masked Autoencoder')
parser.add_argument('--data_dir', type=str, default='MAE_dataset', help='Input data directory')
parser.add_argument('--output_dir', type=str, default='pre-trained-models/mp75/', help='Output directory for models')
args = parser.parse_args()

# Setting seeds for reproducibility.
SEED = 42
keras.utils.set_random_seed(SEED)

from mae_code import *

# ## Hyperparameters for pretraining
# 
# Please feel free to change the hyperparameters and check your results. The best way to
# get an intuition about the architecture is to experiment with it. Our hyperparameters are
# heavily inspired by the design guidelines laid out by the authors in
# [the original paper](https://arxiv.org/abs/2111.06377).

# DATA
BUFFER_SIZE = 512
BATCH_SIZE = 128
AUTO = tf.data.AUTOTUNE

# OPTIMIZER
LEARNING_RATE = 3e-3  # Slightly lower learning rate for stability
WEIGHT_DECAY = 1e-4

# PRETRAINING
EPOCHS = 20  # Increase epochs for better learning

# Update masking proportion to 75%
MASK_PROPORTION = 0.75  # Now represents a percentage (75%) of valid patches to mask

# Model architecture dimensions
ENC_PROJECTION_DIM = 24  # Increased from 12
DEC_PROJECTION_DIM = 12  # Increased from 6
ENC_NUM_HEADS = 8       # Increased from 4
ENC_LAYERS = 12         # Increased from 6

# ## Load and prepare the Numpy 16-bit data
# Directory containing all .npy files
original_data_dir = args.data_dir

# Directory where you want to store training, validation, and test indices
base_dir = 'data'
os.makedirs(base_dir, exist_ok=True)

# Get list of all .npy files
all_files = glob.glob(os.path.join(original_data_dir, "*.npy"))

# Split the files into training, validation, and test sets
train_files, test_files = train_test_split(all_files, test_size=0.1, random_state=42)
train_files, validation_files = train_test_split(train_files, test_size=0.11, random_state=42)

# Save the file lists for reference
with open(os.path.join(base_dir, 'train_files.txt'), 'w') as f:
    for file in train_files:
        f.write(f"{file}\n")

with open(os.path.join(base_dir, 'validation_files.txt'), 'w') as f:
    for file in validation_files:
        f.write(f"{file}\n")

with open(os.path.join(base_dir, 'test_files.txt'), 'w') as f:
    for file in test_files:
        f.write(f"{file}\n")

def load_numpy_files(file_paths, batch_size):
    """Load numpy files and create a tf.data.Dataset"""
    def load_and_preprocess(file_path):
        # Load the numpy file
        img = np.load(file_path.numpy().decode('utf-8'))
        
        # Normalize 16-bit data to [0, 1] range
        img = img.astype(np.float32) / 65535.0
        
        # If grayscale images don't have channel dimension
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        
        # Preserve aspect ratio by padding instead of direct resizing
        if img.shape[0] != IMAGE_SIZE or img.shape[1] != IMAGE_SIZE:
            # Calculate scaling factor to fit within IMAGE_SIZE while preserving aspect ratio
            height, width = img.shape[0], img.shape[1]
            scale = min(IMAGE_SIZE / height, IMAGE_SIZE / width)
            
            # Calculate new dimensions after scaling
            new_height = int(height * scale)
            new_width = int(width * scale)
            
            # Create a blank canvas of target size
            padded_img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, img.shape[2]), dtype=np.float32)
            
            # Resize using numpy/scipy instead of TensorFlow to avoid shape issues
            # First reshape to remove any batch dimension
            img_reshaped = img.reshape(height, width, img.shape[2])
            
            # Use PIL for resizing which is more stable for this operation
            from PIL import Image
            pil_img = Image.fromarray((img_reshaped[:, :, 0] * 255).astype(np.uint8))
            pil_img = pil_img.resize((new_width, new_height), Image.BILINEAR)
            resized_array = np.array(pil_img).astype(np.float32) / 255.0
            
            # Add channel dimension back
            resized_array = np.expand_dims(resized_array, axis=-1)
            
            # Calculate padding to center the image
            y_offset = (IMAGE_SIZE - new_height) // 2
            x_offset = (IMAGE_SIZE - new_width) // 2
            
            # Place the resized image on the canvas
            padded_img[y_offset:y_offset+new_height, x_offset:x_offset+new_width, :] = resized_array
            
            img = padded_img
            
        return img

    def process_path(file_path):
        img = tf.py_function(load_and_preprocess, [file_path], tf.float32)
        # Ensure the shape is correctly set with the exact dimensions
        img.set_shape([IMAGE_SIZE, IMAGE_SIZE, 1])
        return img

    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    dataset = dataset.map(process_path, num_parallel_calls=AUTO)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTO)
    
    return dataset

# Create datasets
train_dataset = load_numpy_files(train_files, BATCH_SIZE)
validation_dataset = load_numpy_files(validation_files, BATCH_SIZE)
test_dataset = load_numpy_files(test_files, BATCH_SIZE)

# ## Data augmentation
# Modified for pre-normalized numpy data

def get_train_augmentation_model():
    model = keras.Sequential(
        [
            # No need for rescaling as we already normalized during loading
            # layers.RandomFlip("horizontal") can be added if desired
            # layers.RandomRotation(factor=0.15) can be added if desired
        ],
        name="train_data_augmentation",
    )
    return model

def get_test_augmentation_model():
    model = keras.Sequential(
        [],  # No need for rescaling as we already normalized during loading
        name="test_data_augmentation",
    )
    return model

# Create visualization directory
vis_dir = os.path.join(args.output_dir, 'visualizations')
os.makedirs(vis_dir, exist_ok=True)

# Let's visualize the image patches.
# Get a batch of images.
image_batch = next(iter(train_dataset))

# Augment the images.
augmentation_model = get_train_augmentation_model()
augmented_images = augmentation_model(image_batch)

# Define the patch layer.
patch_layer = Patches(channels = 1)

# Get the patches from the batched images.
patches = patch_layer(images=augmented_images)

# Now pass the images and the corresponding patches
# to the `show_patched_image` method.
random_index = patch_layer.show_patched_image(images=augmented_images, patches=patches, save_dir=vis_dir)

# Chose the same chose image and try reconstructing the patches
# into the original image.
image = patch_layer.reconstruct_from_patch(patches[random_index])
plt.figure()
plt.imshow(image, cmap = 'gray')
plt.axis("off")
plt.savefig(os.path.join(vis_dir, 'reconstructed_patch.png'))
plt.close()

# Let's see the masking process in action on a sample image.
# Create the patch encoder layer.
patch_encoder = PatchEncoder()

# Get the embeddings and positions.
(
    unmasked_embeddings,
    masked_embeddings,
    unmasked_positions,
    mask_indices,
    unmask_indices,
) = patch_encoder(patches=patches)

# Generate debug CSV for a few example images
for debug_idx in range(min(4, patches.shape[0])):
    generate_patch_debug_csv(patches.numpy(), mask_indices.numpy(), 
                          debug_idx, vis_dir, prefix="single_image_debug")

# Show a masked patch image.
new_patch, random_index = patch_encoder.generate_masked_image(patches, unmask_indices)

# Create a more informative visualization showing masks as overlays
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
# Original image
img = augmented_images[random_index]
plt.imshow(img, cmap='gray')
plt.axis("off")
plt.title("Original")

plt.subplot(1, 3, 2)
# Standard masked image (with white/blank patches)
img_masked = patch_layer.reconstruct_from_patch(new_patch)
plt.imshow(img_masked, cmap='gray')
plt.axis("off")
plt.title("Masked (Standard)")

plt.subplot(1, 3, 3)
# Create overlay visualization
overlay_img = augmented_images[random_index].numpy().copy()
patch_size = PATCH_SIZE
num_patches_per_dim = int(np.sqrt(patches.shape[1]))

# Get coordinates of masked patches
mask_coords = np.unravel_index(mask_indices[random_index].numpy(), 
                             (num_patches_per_dim, num_patches_per_dim))

# Create a semi-transparent overlay mask
mask_overlay = np.ones_like(overlay_img)
for i in range(len(mask_coords[0])):
    row, col = mask_coords[0][i], mask_coords[1][i]
    # Add a colored overlay to masked regions
    mask_overlay[row*patch_size:(row+1)*patch_size, 
                col*patch_size:(col+1)*patch_size] = 0.7  # Partially transparent

# Apply the overlay
plt.imshow(overlay_img, cmap='gray')
plt.imshow(mask_overlay, cmap='Blues', alpha=0.5)  # Blue tint for masked regions
plt.axis("off")
plt.title("Masked (Overlay)")

plt.tight_layout()
plt.savefig(os.path.join(vis_dir, 'mask_example.png'))
plt.close()

# Demonstrate smart masking on actual segmented animal images
# Take 4 images from the dataset to visualize the masking strategy
demo_images = next(iter(train_dataset))[:4]
demo_patches = patch_layer(demo_images)

# Apply smart masking to the demo images
(
    demo_unmasked_embeddings,
    demo_masked_embeddings,
    demo_unmasked_positions,
    demo_mask_indices,
    demo_unmask_indices,
) = patch_encoder(patches=demo_patches)

# Generate debug CSVs for the demo images
for debug_idx in range(len(demo_images)):
    generate_patch_debug_csv(demo_patches.numpy(), demo_mask_indices.numpy(), 
                          debug_idx, vis_dir, prefix="demo_image_debug")

# Add a validation visualization to verify masking strategy
def visualize_masking_strategy(image, patches, mask_indices, patch_layer, vis_dir, idx=0):
    """Create a detailed visualization to verify the masking strategy"""
    patch_means = np.mean(np.abs(patches), axis=1)
    
    # Get grid dimensions
    patch_dim = int(np.sqrt(patches.shape[0]))
    patch_size = PATCH_SIZE
    
    # Create binary masks for visualization
    valid_mask = np.zeros((patch_dim, patch_dim))  # For animal patches (depth > 0)
    masked_animal = np.zeros((patch_dim, patch_dim))  # For masked animal patches
    unmasked_animal = np.zeros((patch_dim, patch_dim))  # For unmasked animal patches
    
    # Count patches for validation
    valid_count = 0
    masked_count = 0
    unmasked_valid_count = 0
    
    # Fill the masks
    for i in range(patches.shape[0]):
        row = i // patch_dim
        col = i % patch_dim
        
        # Check if patch is valid (part of animal)
        is_valid = patch_means[i] > 0
        is_masked = i in mask_indices
        
        if is_valid:
            valid_count += 1
            valid_mask[row, col] = 1
            
            if is_masked:
                masked_count += 1
                masked_animal[row, col] = 1
            else:
                unmasked_valid_count += 1
                unmasked_animal[row, col] = 1
    
    # Calculate percentages for validation
    if valid_count > 0:
        unmasked_percentage = (unmasked_valid_count / valid_count) * 100
        masked_percentage = (masked_count / valid_count) * 100
    else:
        unmasked_percentage = 0
        masked_percentage = 0
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Original image
    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original Image")
    plt.axis("off")
    
    # Animal body mask (valid patches)
    plt.subplot(2, 2, 2)
    # Upscale the mask to match image dimensions
    valid_mask_upscaled = np.repeat(np.repeat(valid_mask, patch_size, axis=0), patch_size, axis=1)
    plt.imshow(image, cmap='gray')
    plt.imshow(valid_mask_upscaled, alpha=0.4, cmap='cool')
    plt.title(f"Animal Body (valid patches): {valid_count} patches")
    plt.axis("off")
    
    # Masked animal parts
    plt.subplot(2, 2, 3)
    # Upscale the mask to match image dimensions
    masked_animal_upscaled = np.repeat(np.repeat(masked_animal, patch_size, axis=0), patch_size, axis=1)
    plt.imshow(image, cmap='gray')
    plt.imshow(masked_animal_upscaled, alpha=0.6, cmap='Reds')
    plt.title(f"Masked Animal Parts: {masked_count} patches ({masked_percentage:.1f}% of animal)")
    plt.axis("off")
    
    # Unmasked animal parts (preserved)
    plt.subplot(2, 2, 4)
    # Upscale the mask to match image dimensions
    unmasked_animal_upscaled = np.repeat(np.repeat(unmasked_animal, patch_size, axis=0), patch_size, axis=1)
    plt.imshow(image, cmap='gray')
    plt.imshow(unmasked_animal_upscaled, alpha=0.6, cmap='Greens')
    plt.title(f"Preserved Animal Parts: {unmasked_valid_count} patches ({unmasked_percentage:.1f}% of animal)")
    plt.axis("off")
    
    # Add validation text at bottom
    plt.figtext(0.5, 0.01, 
               f"Validation: {unmasked_percentage:.1f}% of animal body preserved (target: ≥25%)",
               ha='center', fontsize=12, 
               bbox={'facecolor':'yellow', 'alpha':0.3, 'pad':5})
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'masking_validation_{idx}.png'))
    plt.close()
    
    return {
        'valid_count': valid_count,
        'masked_count': masked_count,
        'unmasked_valid_count': unmasked_valid_count,
        'unmasked_percentage': unmasked_percentage,
        'masked_percentage': masked_percentage
    }

# Verify masking strategy on demo images
print("Validating masking strategy...")
all_stats = []
for i in range(min(4, len(demo_images))):
    stats = visualize_masking_strategy(
        demo_images[i].numpy(), 
        demo_patches[i].numpy(), 
        demo_mask_indices[i].numpy(), 
        patch_layer, 
        vis_dir, 
        idx=i
    )
    all_stats.append(stats)

# Summarize stats in one line instead of multiple prints
avg_unmasked = sum(s['unmasked_percentage'] for s in all_stats) / len(all_stats)
print(f"Average animal body preservation: {avg_unmasked:.1f}% across {len(all_stats)} images")

# Generate masked versions
demo_masked_patches = patch_encoder.generate_masked_images(
    demo_patches, demo_unmask_indices
)

# Create visualization to show smart masking on actual segmented animal images
plt.figure(figsize=(16, 16))
for i in range(4):
    # Original segmented animal image
    plt.subplot(4, 3, i*3+1)
    plt.imshow(demo_images[i], cmap='gray')
    plt.axis("off")
    if i == 0:
        plt.title("Original\nsegmented animal")
    
    # Patches that are masked - show as overlay on original image
    plt.subplot(4, 3, i*3+2)
    # Create a copy of the original image
    overlay_img = demo_images[i].numpy().copy()
    # Get coordinates of masked patches
    patch_coords = np.unravel_index(demo_mask_indices[i].numpy(), 
                                   (int(np.sqrt(demo_patches.shape[1])), 
                                    int(np.sqrt(demo_patches.shape[1]))))
    
    # Create the overlay mask
    mask_overlay = np.ones_like(overlay_img)
    for j in range(len(patch_coords[0])):
        row, col = patch_coords[0][j], patch_coords[1][j]
        # Add a colored highlight to masked regions
        mask_overlay[row*PATCH_SIZE:(row+1)*PATCH_SIZE, 
                    col*PATCH_SIZE:(col+1)*PATCH_SIZE] = 0.7
                    
    # Apply the overlay
    plt.imshow(overlay_img, cmap='gray')
    plt.imshow(mask_overlay, cmap='Blues', alpha=0.5)  # Blue tint for masked regions
    plt.axis("off")
    if i == 0:
        plt.title("Masked Regions\n(highlighted in blue)")
    
    # Result after masking
    plt.subplot(4, 3, i*3+3)
    masked_img = patch_layer.reconstruct_from_patch(demo_masked_patches[i])
    plt.imshow(masked_img, cmap='gray')
    plt.axis("off")
    if i == 0:
        plt.title("Result after\nsmart masking")

plt.tight_layout()
plt.savefig(os.path.join(vis_dir, 'smart_masking_real_examples.png'))
plt.close()

# ## Model initialization

train_augmentation_model = get_train_augmentation_model()
test_augmentation_model = get_test_augmentation_model()
patch_layer = Patches()
patch_encoder = PatchEncoder(mask_proportion=MASK_PROPORTION)
encoder = create_encoder()
decoder = create_decoder()

# ### Learning rate scheduler
# Some code is taken from:
# https://www.kaggle.com/ashusma/training-rfcx-tensorflow-tpu-effnet-b2.

class WarmUpCosine(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps
    ):
        super().__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        if self.total_steps < self.warmup_steps:
            raise ValueError("Total_steps must be larger or equal to warmup_steps.")

        cos_annealed_lr = tf.cos(
            self.pi
            * (tf.cast(step, tf.float32) - self.warmup_steps)
            / float(self.total_steps - self.warmup_steps)
        )
        learning_rate = 0.5 * self.learning_rate_base * (1 + cos_annealed_lr)

        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError(
                    "Learning_rate_base must be larger or equal to "
                    "warmup_learning_rate."
                )
            slope = (
                self.learning_rate_base - self.warmup_learning_rate
            ) / self.warmup_steps
            warmup_rate = slope * tf.cast(step, tf.float32) + self.warmup_learning_rate
            learning_rate = tf.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )
        return tf.where(
            step > self.total_steps, 0.0, learning_rate, name="learning_rate"
        )

    def get_config(self):
        return {
            "learning_rate_base": self.learning_rate_base,
            "total_steps": self.total_steps,
            "warmup_learning_rate": self.warmup_learning_rate,
            "warmup_steps": self.warmup_steps,
        }

# Define the learning rate scheduler
total_steps = int((len(train_dataset)) * EPOCHS)
warmup_epoch_percentage = 0.15
warmup_steps = int(total_steps * warmup_epoch_percentage)
scheduled_lrs = WarmUpCosine(
    learning_rate_base=LEARNING_RATE,
    total_steps=total_steps,
    warmup_learning_rate=0.0,
    warmup_steps=warmup_steps,
)

# Visualize the learning rate schedule
lrs = [scheduled_lrs(step) for step in range(total_steps)]
plt.figure()
plt.plot(lrs)
plt.xlabel("Step", fontsize=14)
plt.ylabel("LR", fontsize=14)
plt.savefig(os.path.join(vis_dir, 'learning_rate_schedule.png'))
plt.close()

mae_model = MaskedAutoencoder(
    train_augmentation_model=train_augmentation_model,
    test_augmentation_model=test_augmentation_model,
    patch_layer=patch_layer,
    patch_encoder=patch_encoder,
    encoder=encoder,
    decoder=decoder,
)

# Compile the model
optimizer = tfa.optimizers.AdamW(
    learning_rate=scheduled_lrs, 
    weight_decay=WEIGHT_DECAY,
    clipnorm=1.0,  # Add gradient clipping for stability
)

# Compile the model with MSE loss for better pixel-level reconstruction
mae_model.compile(
    optimizer=optimizer, 
    loss=keras.losses.MeanSquaredError(), 
    metrics=["mae"],
)

# ## Training callbacks

# ### Visualization callback
# Taking a batch of test inputs to measure model's progress.
test_images = next(iter(test_dataset))

class TrainMonitor(keras.callbacks.Callback):
    def __init__(self, epoch_interval=None, vis_dir=None):
        self.epoch_interval = epoch_interval
        self.vis_dir = vis_dir

    def on_epoch_end(self, epoch, logs=None):
        if self.epoch_interval and epoch % self.epoch_interval == 0:
            test_augmented_images = self.model.test_augmentation_model(test_images)
            test_patches = self.model.patch_layer(test_augmented_images)
            (
                test_unmasked_embeddings,
                test_masked_embeddings,
                test_unmasked_positions,
                test_mask_indices,
                test_unmask_indices,
            ) = self.model.patch_encoder(test_patches)
            test_encoder_outputs = self.model.encoder(test_unmasked_embeddings)
            test_encoder_outputs = test_encoder_outputs + test_unmasked_positions
            test_decoder_inputs = tf.concat(
                [test_encoder_outputs, test_masked_embeddings], axis=1
            )
            test_decoder_outputs = self.model.decoder(test_decoder_inputs)

            # Get mask counts per image for proper visualization
            test_mask_counts = getattr(self.model.patch_encoder, 'mask_counts_per_image', None)
            
            # Show a masked patch image.
            test_masked_patch, idx = self.model.patch_encoder.generate_masked_image(
                test_patches, test_unmask_indices
            )
            
            # Convert tensors to numpy arrays for easier processing
            original_image = test_augmented_images[idx].numpy()
            masked_image = self.model.patch_layer.reconstruct_from_patch(
                test_masked_patch
            ).numpy()
            reconstructed_image = test_decoder_outputs[idx].numpy()
            
            # Calculate value ranges for debugging
            orig_min, orig_max = np.min(original_image), np.max(original_image)
            recon_min, recon_max = np.min(reconstructed_image), np.max(reconstructed_image)
            mask_min, mask_max = np.min(masked_image), np.max(masked_image)
            
            # Calculate Mean Absolute Error for this example
            # Get actual mask count for this image
            mask_count = test_mask_counts[idx].numpy() if test_mask_counts is not None else None
            
            # Use only the actual mask indices (not padding) for visualization
            test_mask_indices_for_image = test_mask_indices[idx].numpy()
            if mask_count is not None:
                test_mask_indices_for_image = test_mask_indices_for_image[:mask_count]
                
            mask_indices_flat = np.unravel_index(test_mask_indices_for_image, 
                                              (int(np.sqrt(test_patches.shape[1])), 
                                               int(np.sqrt(test_patches.shape[1]))))
            
            # Create patch-wise masks for evaluation
            patch_size = PATCH_SIZE
            mask = np.ones_like(original_image)
            for i in range(len(mask_indices_flat[0])):
                row, col = mask_indices_flat[0][i], mask_indices_flat[1][i]
                mask[row*patch_size:(row+1)*patch_size, 
                    col*patch_size:(col+1)*patch_size] = 0
                    
            # Calculate MAE only on masked regions
            masked_region_mae = np.sum(np.abs(original_image * (1-mask) - reconstructed_image * (1-mask))) / np.sum(1-mask)
            
            # Clip reconstructed image values to valid range for visualization
            reconstructed_image = np.clip(reconstructed_image, 0, 1)

            fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
            
            # First row: Standard visualization with fixed range
            ax[0, 0].imshow(original_image, cmap='gray', vmin=0, vmax=1)
            ax[0, 0].set_title(f"Original: {epoch:03d}")
            ax[0, 0].axis("off")

            ax[0, 1].imshow(masked_image, cmap='gray', vmin=0, vmax=1)
            ax[0, 1].set_title(f"Masked: {epoch:03d}")
            ax[0, 1].axis("off")

            ax[0, 2].imshow(reconstructed_image, cmap='gray', vmin=0, vmax=1)
            ax[0, 2].set_title(f"Reconstructed: {epoch:03d}")
            ax[0, 2].axis("off")
            
            # Second row: Enhanced visualization with auto range
            ax[1, 0].imshow(original_image, cmap='gray')
            ax[1, 0].set_title(f"Original (auto range)")
            ax[1, 0].axis("off")

            # Overlay visualization showing what's masked
            overlay_img = original_image.copy()
            mask_overlay = np.ones_like(overlay_img)
            
            for i in range(len(mask_indices_flat[0])):
                row, col = mask_indices_flat[0][i], mask_indices_flat[1][i]
                mask_overlay[row*patch_size:(row+1)*patch_size, 
                            col*patch_size:(col+1)*patch_size] = 0.7
            
            ax[1, 1].imshow(overlay_img, cmap='gray')
            ax[1, 1].imshow(mask_overlay, cmap='Blues', alpha=0.5)
            # Add mask count info to title
            mask_count_text = f" ({mask_count} patches)" if mask_count is not None else ""
            ax[1, 1].set_title(f"Masked regions{mask_count_text}")
            ax[1, 1].axis("off")

            # Difference visualization (error map)
            error_map = np.abs(original_image - reconstructed_image)
            ax[1, 2].imshow(error_map, cmap='hot', vmin=0)
            ax[1, 2].set_title(f"Error Map (MAE: {masked_region_mae:.4f})")
            ax[1, 2].axis("off")
            
            # Add overall statistics
            plt.suptitle(f"Epoch {epoch:03d} | Values - Orig: [{orig_min:.2f}, {orig_max:.2f}], Recon: [{recon_min:.2f}, {recon_max:.2f}] | Loss: {logs.get('val_loss', 0):.6f}", 
                       fontsize=12)
            plt.tight_layout()

            if self.vis_dir:
                plt.savefig(os.path.join(self.vis_dir, f"epoch_{epoch:03d}_test_results.png"), dpi=150)
            plt.close()

# Assemble the callbacks.
train_callbacks = [TrainMonitor(epoch_interval=5, vis_dir=vis_dir)]

# Add early stopping callback
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=5,          # Stop after 5 epochs with no improvement
    min_delta=0.0001,    # Minimum change to qualify as improvement
    mode='min',          # We want to minimize the loss
    restore_best_weights=True,  # Restore model weights from the epoch with the best value
    verbose=1            # Print messages when stopping
)

# Add the early stopping callback to the list
train_callbacks.append(early_stopping)

# Add a progress bar callback that only shows once per epoch
progbar_callback = keras.callbacks.ProgbarLogger(count_mode='steps', stateful_metrics=['loss', 'mae'])

# Add the progress bar to callbacks
train_callbacks.append(progbar_callback)

# Print dataset stats
print(f"\nDataset statistics:")
print(f"Training:   {len(train_files)} images ({len(train_dataset)} batches of size {BATCH_SIZE})")
print(f"Validation: {len(validation_files)} images ({len(validation_dataset)} batches of size {BATCH_SIZE})")
print(f"Test:       {len(test_files)} images ({len(test_dataset)} batches of size {BATCH_SIZE})")
print(f"Total:      {len(train_files) + len(validation_files) + len(test_files)} images\n")

# Train with verbose=1 to show only one progress bar per epoch
history = mae_model.fit(
    train_dataset, 
    epochs=EPOCHS, 
    validation_data=validation_dataset, 
    callbacks=train_callbacks,
    verbose=1  # 0=silent, 1=progress bar, 2=one line per epoch
)

# ## Saving the pre trained model

check_point = args.output_dir
os.makedirs(check_point, exist_ok=True)
print(f"Saving model to directory: {check_point}")

#####################
# SAVE THE COMPLETE MODEL
#####################

# Define the directory to save the complete model checkpoints
checkpoint_directory = os.path.join(check_point, 'mae_model')
os.makedirs(checkpoint_directory, exist_ok=True)

# Create the checkpoint for the entire model
checkpoint = tf.train.Checkpoint(
    patch_encoder=mae_model.patch_encoder,
    encoder=mae_model.encoder,
    decoder=mae_model.decoder
)

# Save the complete model
checkpoint_path = checkpoint.save(file_prefix=os.path.join(checkpoint_directory, "ckpt"))
print(f"Model checkpoint saved to: {checkpoint_path}")

# Also save weights in h5 format for easier loading
weights_dir = os.path.join(check_point, 'weights')
os.makedirs(weights_dir, exist_ok=True)

# Save individual component weights
mae_model.encoder.save_weights(os.path.join(weights_dir, 'encoder_weights.h5'))
mae_model.decoder.save_weights(os.path.join(weights_dir, 'decoder_weights.h5'))
print(f"Model weights saved to: {weights_dir}")

# Verify files were created
checkpoint_files = glob.glob(os.path.join(checkpoint_directory, "*"))
weights_files = glob.glob(os.path.join(weights_dir, "*"))
print(f"Checkpoint files created: {len(checkpoint_files)}")
for f in checkpoint_files:
    print(f"  - {f} ({os.path.getsize(f) / (1024*1024):.2f} MB)")
print(f"Weight files created: {len(weights_files)}")
for f in weights_files:
    print(f"  - {f} ({os.path.getsize(f) / (1024*1024):.2f} MB)")

# Save a simple summary of the model architecture for reference
with open(os.path.join(check_point, 'model_summary.txt'), 'w') as f:
    # Capture encoder summary
    encoder_lines = []
    mae_model.encoder.summary(print_fn=lambda x: encoder_lines.append(x))
    f.write("ENCODER SUMMARY:\n" + "\n".join(encoder_lines) + "\n\n")
    
    # Capture decoder summary
    decoder_lines = []
    mae_model.decoder.summary(print_fn=lambda x: decoder_lines.append(x))
    f.write("DECODER SUMMARY:\n" + "\n".join(decoder_lines))
print(f"Model summary saved to: {os.path.join(check_point, 'model_summary.txt')}")

# ## Loading and Testing the Model

# instantiate the models
train_augmentation_model = get_train_augmentation_model()
test_augmentation_model = get_test_augmentation_model()
patch_layer = Patches()
patch_encoder = PatchEncoder(mask_proportion=MASK_PROPORTION)
encoder = create_encoder()
decoder = create_decoder()

model = MaskedAutoencoder(
    train_augmentation_model=train_augmentation_model,
    test_augmentation_model=test_augmentation_model,
    patch_layer=patch_layer,
    patch_encoder=patch_encoder,
    encoder=encoder,
    decoder=decoder,
)

# Use AdamW optimizer for the testing model as well
optimizer = tfa.optimizers.AdamW(
    learning_rate=scheduled_lrs, 
    weight_decay=WEIGHT_DECAY,
    clipnorm=1.0,  # Add gradient clipping for stability
)

# Compile the model with MSE loss for better pixel-level reconstruction
model.compile(
    optimizer=optimizer, 
    loss=keras.losses.MeanSquaredError(), 
    metrics=["mae"],
)

check_point = args.output_dir

# Creating the checkpoint to restore
checkpoint = tf.train.Checkpoint(
    patch_encoder=model.patch_encoder,
    encoder=model.encoder,
    decoder=model.decoder
)

# Define the directory where the checkpoint was saved
checkpoint_directory = os.path.join(check_point, 'mae_model')

# Check if the checkpoint exists and restore it
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_directory)

if latest_checkpoint:
    checkpoint.restore(latest_checkpoint).expect_partial()
    print(f"Checkpoint restored successfully from: {latest_checkpoint}")
    
    # Additionally, check if h5 weights exist and can be loaded as a fallback
    weights_dir = os.path.join(check_point, 'weights')
    encoder_weights = os.path.join(weights_dir, 'encoder_weights.h5')
    decoder_weights = os.path.join(weights_dir, 'decoder_weights.h5')
    
    if not os.path.exists(encoder_weights) or not os.path.exists(decoder_weights):
        print("Warning: H5 weight files not found.")
    else:
        print(f"H5 weight files found at: {weights_dir}")
else:
    print("No checkpoint found. Will use randomly initialized weights.")
    
    # Check if the directory exists but no checkpoint files
    if os.path.exists(checkpoint_directory):
        files = os.listdir(checkpoint_directory)
        print(f"The checkpoint directory exists but contains: {files}")
    else:
        print(f"Checkpoint directory does not exist: {checkpoint_directory}")

def visualize_results(model, augmented_images, save_dir=None):
    # Patch the augmented images.
    patches = model.patch_layer(augmented_images)

    # Encode the patches.
    (
        unmasked_embeddings,
        masked_embeddings,
        unmasked_positions,
        mask_indices,
        unmask_indices,
    ) = model.patch_encoder(patches)

    # Pass the unmasked patches to the encoder.
    encoder_outputs = model.encoder(unmasked_embeddings)

    # Create the decoder inputs.
    encoder_outputs = encoder_outputs + unmasked_positions
    decoder_inputs = tf.concat([encoder_outputs, masked_embeddings], axis=1)
    decoder_outputs = model.decoder(decoder_inputs)

    # Select 4 example images (using a fixed seed for reproducibility)
    np.random.seed(42)
    sample_indices_np = np.random.choice(len(augmented_images), 4, replace=False)
    # Convert numpy indices to TensorFlow tensor
    sample_indices = tf.convert_to_tensor(sample_indices_np, dtype=tf.int32)
    
    # Get the sample tensors using tf.gather instead of direct indexing
    sample_patches = tf.gather(patches, sample_indices)
    sample_unmask_indices = tf.gather(unmask_indices, sample_indices)
    
    # Show a masked patch image - using numpy arrays for easier processing
    masked_patches = model.patch_encoder.generate_masked_images(
        sample_patches.numpy(), sample_unmask_indices.numpy(), color=0.3
    )

    masked_images = []
    for masked_patch in masked_patches:
        # Apply the reconstruction function to the masked patch
        masked_image = model.patch_layer.reconstruct_from_patch(masked_patch)
        masked_images.append(masked_image)

    # Use numpy indexing for numpy arrays
    original_images = augmented_images.numpy()[sample_indices_np]
    reconstructed_images = decoder_outputs.numpy()[sample_indices_np]
    
    # Calculate min/max stats for proper understanding
    original_min = np.min(original_images)
    original_max = np.max(original_images)
    recon_min = np.min(reconstructed_images)
    recon_max = np.max(reconstructed_images)
    
    # Print value ranges to understand scales
    print(f"Original images value range: [{original_min:.4f}, {original_max:.4f}]")
    print(f"Reconstructed images value range: [{recon_min:.4f}, {recon_max:.4f}]")
    
    # Clip reconstructed values to valid image range [0, 1]
    reconstructed_images = np.clip(reconstructed_images, 0, 1)

    # Minimalistic plot settings
    fig, ax = plt.subplots(4, 3, figsize=(10, 12))
    plt.subplots_adjust(wspace=0.05, hspace=0.1)

    for i in range(4):
        # Original
        ax[i, 0].imshow(original_images[i], cmap='gray', vmin=0, vmax=1)
        ax[i, 0].axis('off')

        # Masked
        ax[i, 1].imshow(masked_images[i], cmap='gray', vmin=0, vmax=1)
        ax[i, 1].axis('off')

        # Reconstructed
        ax[i, 2].imshow(reconstructed_images[i], cmap='gray', vmin=0, vmax=1)
        ax[i, 2].axis('off')

        if i == 0:
          ax[i, 0].set_title("Original")
          ax[i, 1].set_title("Masked")
          ax[i, 2].set_title("Reconstructed")
    
    # Add overall title with value ranges
    plt.suptitle(f"MAE Reconstruction - Value ranges: Original [{original_min:.2f}, {original_max:.2f}], Reconstructed [{recon_min:.2f}, {recon_max:.2f}]", fontsize=10)
    
    # Save with higher DPI for better quality
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'final_results.png'), dpi=150)
        
        # Also save a version with stretched contrast for better visibility
        fig2, ax2 = plt.subplots(4, 3, figsize=(10, 12))
        plt.subplots_adjust(wspace=0.05, hspace=0.1)
        
        for i in range(4):
            # Original
            ax2[i, 0].imshow(original_images[i], cmap='gray')
            ax2[i, 0].axis('off')
    
            # Masked
            ax2[i, 1].imshow(masked_images[i], cmap='gray')
            ax2[i, 1].axis('off')
    
            # Reconstructed - with enhanced contrast
            ax2[i, 2].imshow(reconstructed_images[i], cmap='gray')
            ax2[i, 2].axis('off')
    
            if i == 0:
              ax2[i, 0].set_title("Original")
              ax2[i, 1].set_title("Masked")
              ax2[i, 2].set_title("Enhanced Recon")
              
        plt.suptitle("MAE Reconstruction - Enhanced contrast", fontsize=10)
        plt.savefig(os.path.join(save_dir, 'enhanced_final_results.png'), dpi=150)
    
    plt.close('all')

images = next(iter(test_dataset))

augmented_images = test_augmentation_model(images)
visualize_results(model, augmented_images, save_dir=vis_dir)