from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

#################################
# Constants
#################################

# AUGMENTATION
IMAGE_SIZE = 128  # We will resize input images to this size.
CHANNELS = 1
PATCH_SIZE = 8  # Changed from 16 to 8 for smaller patches
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # Now 256 instead of 64
MASK_PROPORTION = 0.75  # Now represents a percentage (75%) of valid patches to mask


# ENCODER and DECODER
LAYER_NORM_EPS = 1e-6
ENC_PROJECTION_DIM = 24  # Increased from 12 for better representation power
DEC_PROJECTION_DIM = 12  # Increased from 6 for better reconstruction
ENC_NUM_HEADS = 8       # Increased from 4 for better multi-head attention
ENC_LAYERS = 12         # Increased from 6 for deeper encoder
DEC_NUM_HEADS = 8       # Increased from 4 for better multi-head attention
DEC_LAYERS = 6          # Increased from 2 for better reconstruction
ENC_TRANSFORMER_UNITS = [
    ENC_PROJECTION_DIM * 2,
    ENC_PROJECTION_DIM,
]  # Size of the transformer layers.
DEC_TRANSFORMER_UNITS = [
    DEC_PROJECTION_DIM * 2,
    DEC_PROJECTION_DIM,
]

#################################

class Patches(layers.Layer):
    def __init__(self, patch_size=PATCH_SIZE,
                 channels = CHANNELS, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.channels = channels

        # Assuming the image has three channels each patch would be
        # of size (patch_size, patch_size, 3).
        self.resize = layers.Reshape((-1, patch_size * patch_size * channels))

    def call(self, images):
        # Create patches from the input images
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )

        # Reshape the patches to (batch, num_patches, patch_area) and return it.
        patches = self.resize(patches)
        return patches

    def show_patched_image(self, images, patches, save_dir=None):
        # This is a utility function which accepts a batch of images and its
        # corresponding patches and help visualize one image and its patches
        # side by side.
        idx = np.random.choice(patches.shape[0])
        print(f"Index selected: {idx}.")

        plt.figure(figsize=(4, 4))
        plt.imshow(keras.utils.array_to_img(images[idx]))
        plt.axis("off")
        if save_dir:
            plt.savefig(os.path.join(save_dir, f'original_image_{idx}.png'))
        plt.close()

        n = int(np.sqrt(patches.shape[1]))
        
        # For smaller patches, use a more efficient visualization
        # Only plot a subset of patches if there are many
        if n > 8:
            # Create a figure with reasonable size for many patches
            plt.figure(figsize=(10, 10))
            
            # Option 1: Show a subset of patches
            max_patches_to_show = min(64, n*n)  # Show at most 64 patches
            grid_dim = int(np.ceil(np.sqrt(max_patches_to_show)))
            
            # Select patches evenly distributed across the image
            step = (n*n) // max_patches_to_show
            indices = np.arange(0, n*n, step)[:max_patches_to_show]
            
            for i, patch_idx in enumerate(indices):
                if i < max_patches_to_show:
                    ax = plt.subplot(grid_dim, grid_dim, i + 1)
                    patch_img = tf.reshape(patches[idx][patch_idx], 
                                          (self.patch_size, self.patch_size, self.channels))
                    plt.imshow(keras.utils.img_to_array(patch_img))
                    plt.axis("off")
                    # Add tiny label showing patch index
                    ax.text(0, 0, f"{patch_idx}", color='white', 
                           fontsize=6, backgroundcolor='black')
        else:
            # Original visualization for fewer patches
            plt.figure(figsize=(8, 8))
            for i, patch in enumerate(patches[idx]):
                ax = plt.subplot(n, n, i + 1)
                patch_img = tf.reshape(patch, (self.patch_size, self.patch_size, self.channels))
                plt.imshow(keras.utils.img_to_array(patch_img))
                plt.axis("off")
                
        if save_dir:
            plt.savefig(os.path.join(save_dir, f'patched_image_{idx}.png'))
        plt.close()

        # Return the index chosen to validate it outside the method.
        return idx

    # taken from https://stackoverflow.com/a/58082878/10319735
    def reconstruct_from_patch(self, patch):
        # This utility function takes patches from a *single* image and
        # reconstructs it back into the image. This is useful for the train
        # monitor callback.
        num_patches = patch.shape[0]
        n = int(np.sqrt(num_patches))
        patch = tf.reshape(patch, (num_patches, self.patch_size, self.patch_size, self.channels))
        rows = tf.split(patch, n, axis=0)
        rows = [tf.concat(tf.unstack(x), axis=1) for x in rows]
        reconstructed = tf.concat(rows, axis=0)
        return reconstructed


##############################


class PatchEncoder(layers.Layer):
    def __init__(
        self,
        patch_size=PATCH_SIZE,
        projection_dim=ENC_PROJECTION_DIM,
        mask_proportion=MASK_PROPORTION,
        mask_count=None,  # No longer used as we'll calculate dynamically
        channels = CHANNELS,
        downstream=False,
        zero_threshold=0,  # Threshold to consider a patch as mostly zeros
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.mask_proportion = mask_proportion  # Now represents the proportion of valid patches to mask
        self.mask_count = mask_count  # Keep for backward compatibility but not used
        self.downstream = downstream
        self.zero_threshold = zero_threshold  # New param to identify masked regions

        # This is a trainable mask token initialized randomly from a normal
        # distribution.
        self.mask_token = tf.Variable(
            tf.random.normal([1, patch_size * patch_size * channels]), trainable=True
        )
        

    def build(self, input_shape):
        (_, self.num_patches, self.patch_area) = input_shape

        self.rand_indices = tf.argsort(
            tf.random.uniform(shape=(1, self.num_patches)), axis=-1
        )
        # Create the projection layer for the patches.
        self.projection = layers.Dense(units=self.projection_dim)

        # Create the positional embedding layer.
        self.position_embedding = layers.Embedding(
            input_dim=self.num_patches, output_dim=self.projection_dim
        )

        # Number of patches that will be masked is now dynamically calculated per image
        
    def call(self, patches):
        # Get the positional embeddings.
        batch_size = tf.shape(patches)[0]
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        pos_embeddings = self.position_embedding(positions[tf.newaxis, ...])
        pos_embeddings = tf.tile(
            pos_embeddings, [batch_size, 1, 1]
        )  # (B, num_patches, projection_dim)

        # Embed the patches.
        patch_embeddings = (
            self.projection(patches) + pos_embeddings
        )  # (B, num_patches, projection_dim)

        if self.downstream:
            return patch_embeddings
        else:
            
            mask_indices, unmask_indices = self.get_smart_indices(patches, batch_size)
            
            # Get the actual mask counts for each image
            mask_counts = self.mask_counts_per_image
            max_mask_count = tf.shape(mask_indices)[1]  # Maximum mask count in batch
            
            # The encoder input is the unmasked patch embeddings. Here we gather
            # all the patches that should be unmasked.
            unmasked_embeddings = tf.gather(
                patch_embeddings, unmask_indices, axis=1, batch_dims=1
            )  # (B, unmask_numbers, projection_dim)

            # Get the unmasked and masked position embeddings. We will need them
            # for the decoder.
            unmasked_positions = tf.gather(
                pos_embeddings, unmask_indices, axis=1, batch_dims=1
            )  # (B, unmask_numbers, projection_dim)
            masked_positions = tf.gather(
                pos_embeddings, mask_indices, axis=1, batch_dims=1
            )  # (B, mask_numbers, projection_dim)

            # Create a single mask token embedding for all positions
            mask_token = self.projection(tf.repeat(
                self.mask_token, max_mask_count, axis=0
            ))  # [max_mask_count, projection_dim]
            
            # Broadcast the mask token to batch size
            mask_tokens = tf.repeat(
                mask_token[tf.newaxis, :, :], 
                batch_size, 
                axis=0
            )  # [batch_size, max_mask_count, projection_dim]
            
            # Add position embeddings to mask tokens
            masked_embeddings = mask_tokens + masked_positions
            
            # Apply mask for invalid positions (where padding exists due to variable mask counts)
            if mask_counts is not None:
                # Create a mask with 1s for valid positions and 0s for padding
                sequence = tf.range(max_mask_count, dtype=tf.int32)[tf.newaxis, :]
                sequence = tf.repeat(sequence, batch_size, axis=0)
                mask_counts_expanded = mask_counts[:, tf.newaxis]
                validity_mask = tf.cast(tf.less(sequence, mask_counts_expanded), tf.float32)
                
                # Expand validity mask for broadcasting with embeddings
                validity_mask = tf.expand_dims(validity_mask, axis=-1)
                
                # Zero out invalid positions
                masked_embeddings = masked_embeddings * validity_mask
            
            return (
                unmasked_embeddings,  # Input to the encoder.
                masked_embeddings,  # First part of input to the decoder.
                unmasked_positions,  # Added to the encoder outputs.
                mask_indices,  # The indices that were masked.
                unmask_indices,  # The indices that were unmaksed.
            )

    def get_smart_indices(self, patches, batch_size):
        """Smart masking strategy that masks 75% of patches with content (depth > 0)
        while ensuring at least 25% of the animal body remains unmasked.
        Images without enough valid patches will be marked for skipping."""
        
        # Calculate average value for each patch
        patch_means = tf.reduce_mean(tf.abs(patches), axis=2)  # [batch_size, num_patches]
        
        # Create a mask for valid (non-zero) patches
        valid_mask = tf.cast(patch_means > 0, tf.float32)  # [batch_size, num_patches]
        
        # Count valid patches per image
        valid_counts = tf.reduce_sum(valid_mask, axis=1)  # [batch_size]
        
        # Calculate minimum patches to preserve (25% - inverse of mask_proportion)
        preservation_ratio = tf.constant(1.0 - self.mask_proportion, dtype=tf.float32)
        min_unmasked = tf.cast(tf.math.ceil(valid_counts * preservation_ratio), tf.int32)  # [batch_size]
        
        # Calculate the dynamic mask count for each image (75% of valid patches)
        mask_counts = tf.cast(tf.math.floor(valid_counts * self.mask_proportion), tf.int32)  # [batch_size]
        
        # Make sure we don't mask everything - ensure minimum preservation
        mask_counts = tf.minimum(mask_counts, tf.cast(valid_counts, tf.int32) - min_unmasked)
        
        # Create fully random scores for valid patches
        # Large random values (0-1000) to ensure complete randomization
        random_scores = tf.random.uniform(tf.shape(patch_means), minval=0, maxval=1000, dtype=tf.float32)
        
        # Only assign random scores to valid patches, zero out invalid patches
        randomized_scores = random_scores * valid_mask
        
        # Sort patches by random scores (descending)
        _, indices = tf.nn.top_k(randomized_scores, k=self.num_patches)
        
        # Conditions for each image: does it have enough valid patches?
        has_enough_valid = valid_counts > 0  # Has at least some valid patches
        
        # The maximum mask count is variable now
        max_mask_count = tf.reduce_max(mask_counts)
        
        # Create mask indices for each image (with padding to max count)
        mask_indices = tf.zeros((batch_size, max_mask_count), dtype=tf.int32)
        
        # Create an attention mask to identify skipped images (1 = keep, 0 = skip)
        # This will be used in loss calculation to exclude skipped images
        self.attention_mask = tf.cast(has_enough_valid, tf.float32)
        
        # We also need to track the actual mask count per image for the unmask indices
        self.mask_counts_per_image = mask_counts
        
        # Count total skipped images and only log if there are any
        total_skipped = batch_size - tf.reduce_sum(tf.cast(has_enough_valid, tf.int32))
        if total_skipped > 0:
            tf.print("Skipped", total_skipped, "images due to insufficient valid patches")
        
        # Helper function to process one image - avoids tf.cond type mismatches
        def get_indices_for_image(i, mask_indices_tensor):
            # Get sorted indices for this image
            image_indices = indices[i]
            
            # Choose appropriate indices based on condition
            use_valid_patches = has_enough_valid[i]
            
            # Get dynamic mask count for this specific image
            this_mask_count = mask_counts[i]
            
            # Create placeholder indices (for skipped images)
            placeholder_indices = tf.zeros(max_mask_count, dtype=tf.int32)
            
            # Explicitly handle the two conditions directly
            selected_indices = placeholder_indices
            if use_valid_patches:
                # Select the top indices up to this image's mask count
                selected_indices_raw = image_indices[:this_mask_count]
                # Pad with zeros to reach max_mask_count
                padding = tf.zeros(max_mask_count - this_mask_count, dtype=tf.int32)
                selected_indices = tf.concat([selected_indices_raw, padding], axis=0)
            
            # Update the mask indices for this image
            indices_update = tf.tensor_scatter_nd_update(
                mask_indices_tensor,
                tf.expand_dims(tf.expand_dims(i, 0), 1),  # [[i]]
                tf.expand_dims(selected_indices, 0)  # [1, max_mask_count]
            )
            
            return i + 1, indices_update
        
        # Process each image with a TF while loop
        _, mask_indices = tf.while_loop(
            cond=lambda i, _: i < batch_size,
            body=get_indices_for_image,
            loop_vars=(tf.constant(0), mask_indices)
        )
        
        # Create unmask indices using the same approach, but with variable sized outputs
        # First, determine the maximum number of unmasked patches
        max_unmask_count = tf.reduce_max(self.num_patches - mask_counts)
        unmask_indices = tf.zeros((batch_size, max_unmask_count), dtype=tf.int32)
        
        # Process each image to get unmask indices
        def get_unmask_for_image(i, unmask_indices_tensor):
            # Only process images with enough valid patches
            use_valid_patches = has_enough_valid[i]
            
            # Get mask indices for this image
            image_mask_indices = mask_indices[i][:mask_counts[i]]  # Only use the actual mask count
            
            # This image's unmask count
            this_unmask_count = self.num_patches - mask_counts[i]
            
            # Create a placeholder for invalid images
            placeholder_indices = tf.range(max_unmask_count, dtype=tf.int32)
            
            # For valid images, calculate unmask indices properly
            image_unmask_indices = placeholder_indices  # Default placeholder
            
            if use_valid_patches:
                # Get indices of non-masked patches
                mask_indicator = tf.reduce_sum(
                    tf.one_hot(image_mask_indices, self.num_patches, on_value=1, off_value=0, dtype=tf.int32),
                    axis=0
                )
                unmask_raw = tf.boolean_mask(
                    tf.range(self.num_patches, dtype=tf.int32),
                    mask_indicator == 0
                )
                # Pad with zeros to reach max_unmask_count
                padding = tf.zeros(max_unmask_count - this_unmask_count, dtype=tf.int32)
                image_unmask_indices = tf.concat([unmask_raw, padding], axis=0)
            
            # Update the unmask indices tensor
            indices_update = tf.tensor_scatter_nd_update(
                unmask_indices_tensor,
                tf.expand_dims(tf.expand_dims(i, 0), 1),  # [[i]]
                tf.expand_dims(image_unmask_indices, 0)  # [1, max_unmask_count]
            )
            
            return i + 1, indices_update
        
        # Process each image with a TF while loop
        _, unmask_indices = tf.while_loop(
            cond=lambda i, _: i < batch_size,
            body=get_unmask_for_image,
            loop_vars=(tf.constant(0), unmask_indices)
        )
        
        # Log proportion stats
        valid_avg = tf.reduce_mean(tf.cast(valid_counts, tf.float32))
        masked_avg = tf.reduce_mean(tf.cast(mask_counts, tf.float32))
        if valid_avg > 0:
            mask_ratio = masked_avg / valid_avg
            tf.print("Average valid patches:", valid_avg, "Average masked:", masked_avg, 
                     "Mask ratio:", mask_ratio, "Target:", self.mask_proportion)
        
        return mask_indices, unmask_indices

    def get_random_indices(self, batch_size):
        # Create random indices from a uniform distribution and then split
        # it into mask and unmask indices.
        rand_indices = tf.argsort(
            tf.random.uniform(shape=(batch_size, self.num_patches)), axis=-1
        )
        
        mask_indices = rand_indices[:, : self.num_patches]
        unmask_indices = rand_indices[:, self.num_patches :]
        return mask_indices, unmask_indices

    def get_indices(self, batch_size):
        
        rand_indices = tf.tile(self.rand_indices, [128, 1])

        mask_indices = rand_indices[:, :self.num_patches]
        unmask_indices = rand_indices[:, self.num_patches:]

        return mask_indices, unmask_indices


    def get_ordered_indices(self, batch_size):
        # Create sequential indices.
        sequential_indices = tf.range(self.num_patches, dtype=tf.int32)

        # Separate even and odd indices.
        even_indices = sequential_indices[::2]
        odd_indices = sequential_indices[1::2]

        # Interleave even and odd indices.
        ordered_indices = tf.concat([even_indices, odd_indices], axis=0)

        # Repeat indices to match the batch size.
        ordered_indices = tf.tile(ordered_indices[tf.newaxis, :], [batch_size, 1])

        # Split indices into mask_indices and unmask_indices.
        mask_indices = ordered_indices[:, :self.num_patches]
        unmask_indices = ordered_indices[:, self.num_patches:]

        return mask_indices, unmask_indices

    def get_sorted_indices(self, batch_size):
        # Create sequential indices.
        sequential_indices = tf.range(self.num_patches, dtype=tf.int32)

        # Calculate the midpoint.
        midpoint = self.num_patches // 2

        # Calculate the distance of each index from the midpoint.
        distances = tf.abs(sequential_indices - midpoint)

        # Sort indices based on distances in descending order.
        sorted_indices = tf.argsort(-distances)

        # Repeat indices to match the batch size.
        sorted_indices = tf.tile(sorted_indices[tf.newaxis, :], [batch_size, 1])

        # Split indices into mask_indices and unmask_indices.
        mask_indices = sorted_indices[:, :self.num_patches]
        unmask_indices = sorted_indices[:, self.num_patches:]

        return mask_indices, unmask_indices

    
    def generate_masked_image(self, patches, unmask_indices):
        # Choose a random patch and it corresponding unmask index.
        idx = np.random.choice(patches.shape[0])

        patch = patches[idx]
        
        # For variable-sized unmask indices, get the actual size for this image
        # Unmask indices may be padded with zeros, so we need to filter them out
        # Find the first zero index, or use all if no zeros
        zero_indices = np.where(unmask_indices[idx] == 0)[0]
        if len(zero_indices) > 0 and zero_indices[0] > 0:
            actual_unmask_index = unmask_indices[idx][:zero_indices[0]]
        else:
            actual_unmask_index = unmask_indices[idx]

        # Build a numpy array of same shape as patch.
        new_patch = np.ones_like(patch)

        # Iterate of the new_patch and plug the unmasked patches.
        for i in range(len(actual_unmask_index)):
            new_patch[actual_unmask_index[i]] = patch[actual_unmask_index[i]]
        return new_patch, idx
    
    def generate_masked_images(self, patches, unmask_indices, color = 0.1):
        masked_images = []

        for idx in range(len(patches)):
            patch = patches[idx]
            
            # For variable-sized unmask indices, get the actual size for this image
            # Find the first zero index, or use all if no zeros
            zero_indices = np.where(unmask_indices[idx] == 0)[0]
            if len(zero_indices) > 0 and zero_indices[0] > 0:
                actual_unmask_index = unmask_indices[idx][:zero_indices[0]]
            else:
                actual_unmask_index = unmask_indices[idx]

            # Build a numpy array of the same shape as patch.
            new_patch = np.ones_like(patch)

            # Iterate over the new_patch and plug the unmasked patches.
            for i in range(len(actual_unmask_index)):
                new_patch[actual_unmask_index[i]] = patch[actual_unmask_index[i]]

            masked_images.append(new_patch)

        return masked_images

    
#################################



def mlp(x, dropout_rate, hidden_units):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

########################################


def create_encoder(num_heads=ENC_NUM_HEADS, num_layers=ENC_LAYERS):
    inputs = layers.Input((None, ENC_PROJECTION_DIM))
    x = inputs

    for _ in range(num_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(x)

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=ENC_PROJECTION_DIM, dropout=0.1
        )(x1, x1)

        # Skip connection 1.
        x2 = layers.Add()([attention_output, x])

        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(x2)

        # MLP.
        x3 = mlp(x3, hidden_units=ENC_TRANSFORMER_UNITS, dropout_rate=0.1)

        # Skip connection 2.
        x = layers.Add()([x3, x2])

    outputs = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(x)
    return keras.Model(inputs, outputs, name="mae_encoder")

######################


def create_decoder(
    num_layers=DEC_LAYERS, num_heads=DEC_NUM_HEADS, image_size=IMAGE_SIZE,
    channels = CHANNELS
):
    inputs = layers.Input((None, ENC_PROJECTION_DIM))  # Change to None for variable length
    x = layers.Dense(DEC_PROJECTION_DIM)(inputs)

    for _ in range(num_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(x)

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=DEC_PROJECTION_DIM, dropout=0.1
        )(x1, x1)

        # Skip connection 1.
        x2 = layers.Add()([attention_output, x])

        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(x2)

        # MLP.
        x3 = mlp(x3, hidden_units=DEC_TRANSFORMER_UNITS, dropout_rate=0.1)

        # Skip connection 2.
        x = layers.Add()([x3, x2])

    x = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(x)
    
    # Use GlobalAveragePooling1D instead of Flatten to handle variable-length inputs
    x = layers.GlobalAveragePooling1D()(x)
    
    pre_final = layers.Dense(units=image_size * image_size * channels, activation="sigmoid")(x)
    outputs = layers.Reshape((image_size, image_size, channels))(pre_final)

    return keras.Model(inputs, outputs, name="mae_decoder")

############################


class MaskedAutoencoder(keras.Model):
    def __init__(
        self,
        train_augmentation_model,
        test_augmentation_model,
        patch_layer,
        patch_encoder,
        encoder,
        decoder,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.train_augmentation_model = train_augmentation_model
        self.test_augmentation_model = test_augmentation_model
        self.patch_layer = patch_layer
        self.patch_encoder = patch_encoder
        self.encoder = encoder
        self.decoder = decoder
        
    def compile(self, **kwargs):
        super().compile(**kwargs)
        
        # Adapt the decoder to handle variable input shapes
        # Force build the decoder with a sample input
        # This avoids the dimension mismatch errors when the decoder first runs
        batch_size = 1
        sample_unmasked = tf.zeros((batch_size, NUM_PATCHES // 4, ENC_PROJECTION_DIM))
        sample_masked = tf.zeros((batch_size, NUM_PATCHES - (NUM_PATCHES // 4), ENC_PROJECTION_DIM))
        
        # Get sample encoder output and positions
        sample_encoder_output = self.encoder(sample_unmasked)
        sample_positions = tf.zeros_like(sample_encoder_output)
        sample_encoder_output = sample_encoder_output + sample_positions
        
        # Create decoder inputs
        sample_decoder_inputs = tf.concat([sample_encoder_output, sample_masked], axis=1)
        
        # Run decoder once to build it
        _ = self.decoder(sample_decoder_inputs)

    def calculate_loss(self, images, test=False):
        # Augment the input images.
        if test:
            augmented_images = self.test_augmentation_model(images)
        else:
            augmented_images = self.train_augmentation_model(images)

        # Patch the augmented images.
        patches = self.patch_layer(augmented_images)

        # Encode the patches.
        (
            unmasked_embeddings,
            masked_embeddings,
            unmasked_positions,
            mask_indices,
            unmask_indices,
        ) = self.patch_encoder(patches)

        # Pass the unmaksed patches to the encoder.
        encoder_outputs = self.encoder(unmasked_embeddings)

        # Create the decoder inputs.
        encoder_outputs = encoder_outputs + unmasked_positions
        decoder_inputs = tf.concat([encoder_outputs, masked_embeddings], axis=1)

        # Decode the inputs.
        decoder_outputs = self.decoder(decoder_inputs)
        decoder_patches = self.patch_layer(decoder_outputs)

        # Get actual mask counts for each image
        mask_counts = getattr(self.patch_encoder, 'mask_counts_per_image', None)
        
        # Create a mask to ensure we only compute loss on actual masked regions, not padding
        batch_size = tf.shape(mask_indices)[0]
        max_mask_count = tf.shape(mask_indices)[1]
        
        # Create a simpler approach to mask region identification
        mask_region_mask = None
        if mask_counts is not None:
            # Create a sequence matrix of shape [batch_size, max_mask_count]
            # where each row is [0, 1, 2, ..., max_mask_count-1]
            sequence = tf.range(max_mask_count, dtype=tf.int32)[tf.newaxis, :]
            sequence = tf.repeat(sequence, batch_size, axis=0)
            
            # Create a mask by comparing each position with the mask count for that image
            # This creates a 1 for valid positions and 0 for padding
            mask_counts_expanded = mask_counts[:, tf.newaxis]  # [batch_size, 1]
            mask_region_mask = tf.cast(tf.less(sequence, mask_counts_expanded), tf.float32)
        
        # Gather patches for loss calculation
        loss_patch = tf.gather(patches, mask_indices, axis=1, batch_dims=1)
        loss_output = tf.gather(decoder_patches, mask_indices, axis=1, batch_dims=1)

        # Get the attention mask that identifies which images to skip
        attention_mask = getattr(self.patch_encoder, 'attention_mask', None)
        
        # Compute the total loss, skipping images without enough valid patches if mask exists
        if attention_mask is not None:
            # Expand attention mask to match loss shape for broadcasting
            expanded_mask = tf.expand_dims(attention_mask, axis=1)
            expanded_mask = tf.repeat(expanded_mask, tf.shape(loss_patch)[1], axis=1)
            
            # Combine with mask region mask to handle both skipped images and padding
            if mask_region_mask is not None:
                expanded_mask = expanded_mask * mask_region_mask
            
            # Apply mask to loss calculation - images with mask=0 contribute 0 to loss
            masked_loss = self.compiled_loss(loss_patch, loss_output, sample_weight=expanded_mask)
            
            # Count number of valid elements
            num_valid_elements = tf.reduce_sum(expanded_mask)
            
            # Normalize by number of valid elements (avoid division by zero)
            num_valid_elements = tf.maximum(num_valid_elements, 1.0)
            total_loss = masked_loss * tf.cast(tf.size(expanded_mask), tf.float32) / num_valid_elements
            
            # Only print if we're actually skipping images (reduces verbosity)
            num_skipped = tf.shape(attention_mask)[0] - tf.reduce_sum(tf.cast(attention_mask, tf.int32))
            
            # Use tf.cond instead of Python if statement for tensor conditional
            def log_skipped():
                tf.print("Loss calculation excludes", num_skipped, "images with insufficient patches")
                return tf.no_op()
                
            def no_op():
                return tf.no_op()
            
            # Only log during training, not validation
            if not test:
                # Use tf.cond for tensor conditional
                tf.cond(
                    tf.greater(num_skipped, 0),
                    log_skipped,
                    no_op
                )
        else:
            # If no attention mask, compute loss normally
            total_loss = self.compiled_loss(loss_patch, loss_output)

        return total_loss, loss_patch, loss_output

    def train_step(self, images):
        with tf.GradientTape() as tape:
            total_loss, loss_patch, loss_output = self.calculate_loss(images)

        # Apply gradients.
        train_vars = [
            self.train_augmentation_model.trainable_variables,
            self.patch_layer.trainable_variables,
            self.patch_encoder.trainable_variables,
            self.encoder.trainable_variables,
            self.decoder.trainable_variables,
        ]
        grads = tape.gradient(total_loss, train_vars)
        tv_list = []
        for (grad, var) in zip(grads, train_vars):
            for g, v in zip(grad, var):
                tv_list.append((g, v))
        self.optimizer.apply_gradients(tv_list)

        # Report progress.
        self.compiled_metrics.update_state(loss_patch, loss_output)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, images):
        total_loss, loss_patch, loss_output = self.calculate_loss(images, test=True)

        # Update the trackers.
        self.compiled_metrics.update_state(loss_patch, loss_output)
        return {m.name: m.result() for m in self.metrics}