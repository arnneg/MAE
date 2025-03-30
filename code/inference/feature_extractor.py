#!/usr/bin/env python
# coding: utf-8

# # Feature Extractor Code 

from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
import os

# Setting seeds for reproducibility.
SEED = 42
keras.utils.set_random_seed(SEED)

from mae_code import *

# DATA
BUFFER_SIZE = 512
BATCH_SIZE = 128
AUTO = tf.data.AUTOTUNE

# OPTIMIZER
LEARNING_RATE = 5e-3
WEIGHT_DECAY = 1e-4

# PRETRAINING
EPOCHS = 100

base_dir = 'png_512'

# Create a dataset from images in the directory
dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join(base_dir),              # Base directory
    labels='inferred',                   # Infer labels from directory structure
    color_mode="grayscale",
    label_mode=None,                     # No labels
    image_size=(IMAGE_SIZE, IMAGE_SIZE), # Image size
    batch_size=BATCH_SIZE,               # Batch size
    shuffle=False,                       # Do not shuffle the data
    interpolation='bilinear'
)
# ## Functions
# 
# These functions are identical to those used in the pre-training code.
# 

def get_train_augmentation_model():
    model = keras.Sequential(
        [
            layers.Rescaling(1 / 255.0),
#             layers.RandomFlip("horizontal"),
#             layers.RandomRotation(factor=0.15),  # Rotação aleatória entre -30 e 30 graus (fator = 0.15 para 15 graus)
         layers.Resizing(IMAGE_SIZE, IMAGE_SIZE)
        ],
        name="train_data_augmentation",
    )
    return model


def get_test_augmentation_model():
    model = keras.Sequential(
        [layers.Rescaling(1 / 255.0),
         layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),],
        name="test_data_augmentation",
    )
    return model

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

total_steps = int((len(dataset)) * EPOCHS)
warmup_epoch_percentage = 0.15
warmup_steps = int(total_steps * warmup_epoch_percentage)
scheduled_lrs = WarmUpCosine(
    learning_rate_base=LEARNING_RATE,
    total_steps=total_steps,
    warmup_learning_rate=0.0,
    warmup_steps=warmup_steps,
)

lrs = [scheduled_lrs(step) for step in range(total_steps)]
plt.plot(lrs)
plt.xlabel("Step", fontsize=14)
plt.ylabel("LR", fontsize=14)
plt.show()

# ## Load Weights from Pre-trained Models
# instantiate the models

train_augmentation_model = get_train_augmentation_model()
test_augmentation_model = get_test_augmentation_model()
patch_layer = Patches()
patch_encoder = PatchEncoder()
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

optimizer = tf.optimizers.AdamW(learning_rate=scheduled_lrs, weight_decay=WEIGHT_DECAY)

#Compile and pretrain the model.
model.compile( optimizer=optimizer, loss=keras.losses.MeanSquaredError(), metrics=["mae"])

check_point = "pre-trained-models/mp75/"

# Creating the checkpoint to restore
checkpoint = tf.train.Checkpoint(
    patch_encoder=model.patch_encoder,
    encoder=model.encoder,
    decoder=model.decoder
)

# Define the directory where the checkpoint was saved
checkpoint_directory = check_point + 'mae_model/'

# Check if the checkpoint exists and restore it
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_directory)

if latest_checkpoint:
    checkpoint.restore(latest_checkpoint).expect_partial()
    print("Checkpoint restored successfully!")
else:
    print("No checkpoint found.")

# ## Visualizing the Reconstruction

images = next(iter(dataset))

augmented_images = test_augmentation_model(images)

def visualize_results(model, augmented_images):

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

    # Show a masked patch image.
    masked_patches = model.patch_encoder.generate_masked_images(
        patches[2:6], unmask_indices[2:6], color = 0.3
)

    masked_images = []
    for masked_patch in masked_patches:
        # Apply the reconstruction function to the masked patch
        masked_image = model.patch_layer.reconstruct_from_patch(masked_patch)
        masked_images.append(masked_image)

    original_image = augmented_images[2:6]
    reconstructed_image = decoder_outputs[2:6]

    # Minimalistic plot settings
    fig, ax = plt.subplots(4, 3, figsize=(7.5, 10))
    plt.subplots_adjust(wspace=0.01, hspace=0.01)

    for i in range(4):
        # Original
        ax[i, 0].imshow(original_image[i], cmap='gray')
        ax[i, 0].axis('off')

        # Masked
        ax[i, 1].imshow(masked_images[i], cmap='gray')
        ax[i, 1].axis('off')

        # Reconstruída
        ax[i, 2].imshow(reconstructed_image[i], cmap='gray')
        ax[i, 2].axis('off')

        if i == 0:
          ax[i, 0].set_title("Original")
          ax[i, 1].set_title("Masked")
          ax[i, 2].set_title("Reconstructed")

    plt.show()

# augmented_images = test_augmentation_model(images)
visualize_results(model, augmented_images)


# ## Extracting Latent Variables
# Assuming you have a train_dataset containing multiple images

# Initialize an empty list to store the results for each image
processed_images = []

# Use tqdm to create a progress bar for the loop
for images in tqdm(dataset, total=len(dataset)):
    # Perform the same operations as in your code snippet

    # Apply augmentation to the image
    augmented_images = test_augmentation_model(images)

    # Patch the augmented images
    patches = patch_layer(augmented_images)

    # Encode the patches
    (
        unmasked_embeddings,
        masked_embeddings,
        unmasked_positions,
        mask_indices,
        unmask_indices,
    ) = model.patch_encoder(patches)

    # Pass the unmasked patches to the encoder
    encoder_outputs = model.encoder(unmasked_embeddings)

    # Append the processed image to the list
    processed_images.append(encoder_outputs)

# processed_images now contains the processed data for the entire dataset, and you will see a progress bar as the loop iterates.
# ## Reshaping and Creating a DF
# number of patches not masked
num_embb = int((1- MASK_PROPORTION) * NUM_PATCHES)

# Initialize a new list to store the reshaped tensors
reshaped_outputs = []

# Use tqdm to create a progress bar for the loop
for output in tqdm(processed_images, total=len(processed_images), desc="Reshaping"):
    reshaped_tensor = tf.reshape(output, [output.shape[0], num_embb  * 16])
    reshaped_outputs.append(reshaped_tensor)

# Assuming reshaped_outputs is a list of shape (553, 128, num_embb  * 16)s)

# Convert the list to a NumPy array
reshaped_array = np.array(reshaped_outputs[:-1])

# Reshape the array to have shape (553*128, num_embb  * 16))
reshaped_array = reshaped_array.reshape(-1, num_embb  * 16)

# Create a DataFrame from the reshaped array
df = pd.DataFrame(reshaped_array)

# Assuming reshaped_outputs is a list of arrays and df is your DataFrame
last_batch = reshaped_outputs[-1]  # Assuming the last element is the last batch
last_batch = last_batch[:28]  # Keep only the first 28 elements
last_batch = np.array(last_batch)

# Append the last batch to the DataFrame
df = pd.concat([df, pd.DataFrame(last_batch.reshape(-1, num_embb * 16))], ignore_index=True)

# Add the filenames that correspond to observations in the DataFrame
all_files = [filename for filename in os.listdir('png_512') if filename.endswith('.png')]

# Remove "resultado_" prefix and ".png" suffix from all elements in the list
cleaned_list = [filename.replace('resultado_', '').replace('.png', '') for filename in all_files]

cleaned_list = sorted(cleaned_list)
df['Filenames'] = cleaned_list