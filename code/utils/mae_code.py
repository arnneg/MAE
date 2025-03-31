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
MASK_PROPORTION = 0.25  # Now represents a percentage (75%) of valid patches to mask


# ENCODER and DECODER
LAYER_NORM_EPS = 1e-6
ENC_PROJECTION_DIM = 16  # Increased from 12 for better representation power
DEC_PROJECTION_DIM = 16  # Increased from 6 for better reconstruction
ENC_NUM_HEADS = 8       # Increased from 4 for better multi-head attention
ENC_LAYERS = 8         # Increased from 6 for deeper encoder
DEC_NUM_HEADS = 8       # Increased from 4 for better multi-head attention
DEC_LAYERS = 8          # Increased from 2 for better reconstruction
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
        """Smart masking strategy with fixed number of patches (40).
        Skip images where these patches don't cover at least 60% of the animal body."""
        
        # Calculate average value for each patch to determine valid patches (animal body)
        patch_means = tf.reduce_mean(tf.abs(patches), axis=2)  # [batch_size, num_patches]
        
        # Create a mask for valid (non-zero) patches - these are parts of the animal
        valid_mask = tf.cast(patch_means > 0, tf.float32)  # [batch_size, num_patches]
        
        # Count valid patches per image
        valid_counts = tf.reduce_sum(valid_mask, axis=1)  # [batch_size]
        
        # Fixed number of patches to mask
        FIXED_MASK_COUNT = 40
        
        # Check if we have enough valid patches to mask (at least 60% of the animal must remain visible)
        # This means at most 40% of valid patches can be masked
        max_maskable_percentage = 0.4  # At most 40% can be masked (60% must remain visible)
        
        # Calculate the maximum number of masks allowed for each image (40% of valid patches)
        max_masks_allowed = tf.cast(tf.math.ceil(valid_counts * max_maskable_percentage), tf.int32)
        
        # Conditions for each image: 
        # 1. Does it have enough valid patches?
        # 2. Would masking FIXED_MASK_COUNT patches leave at least 60% of animal visible?
        has_enough_valid = valid_counts >= 20  # Has at least 20 valid patches
        masking_allowed = tf.greater_equal(max_masks_allowed, FIXED_MASK_COUNT)  # Fixed mask count doesn't exceed 40% limit
        
        # Image is valid if it has enough valid patches AND masking wouldn't exceed our threshold
        valid_image = tf.logical_and(has_enough_valid, masking_allowed)
        
        # Instead of random scores for all patches, we want to prioritize valid patches (animal body)
        # We'll give high scores to valid patches and low scores to invalid patches
        # This ensures we mask animal body parts first
        
        # Create scores where valid patches get high random values, invalid patches get very low values
        random_scores_valid = tf.random.uniform(tf.shape(patch_means), minval=1000, maxval=2000, dtype=tf.float32)
        random_scores_invalid = tf.random.uniform(tf.shape(patch_means), minval=0, maxval=10, dtype=tf.float32)
        
        # Combine scores - use valid mask to select which score to use for each patch
        randomized_scores = valid_mask * random_scores_valid + (1.0 - valid_mask) * random_scores_invalid
        
        # Sort patches by scores (descending) to get candidates for masking
        # Valid patches with high scores will be at the top
        _, indices = tf.nn.top_k(randomized_scores, k=self.num_patches)
        
        # Create mask indices for each image (fixed count or zeros for skipped images)
        mask_indices = tf.zeros((batch_size, FIXED_MASK_COUNT), dtype=tf.int32)
        
        # Create an attention mask to identify skipped images (1 = keep, 0 = skip)
        # This will be used in loss calculation to exclude skipped images
        self.attention_mask = tf.cast(valid_image, tf.float32)
        
        # We also need to track the actual mask count per image for the unmask indices
        # For valid images, it's FIXED_MASK_COUNT; for invalid ones, it's 0
        self.mask_counts_per_image = tf.where(
            valid_image,
            tf.ones((batch_size,), dtype=tf.int32) * FIXED_MASK_COUNT,
            tf.zeros((batch_size,), dtype=tf.int32)
        )
        
        # Count total skipped images for logging
        total_skipped = batch_size - tf.reduce_sum(tf.cast(valid_image, tf.int32))
        
        # Helper function to process one image
        def get_indices_for_image(i, mask_indices_tensor):
            # Get sorted indices for this image
            image_indices = indices[i]
            
            # Check if this image should be processed
            is_valid = valid_image[i]
            
            # Create placeholder indices (for skipped images)
            placeholder_indices = tf.zeros(FIXED_MASK_COUNT, dtype=tf.int32)
            
            # Get proper mask indices for valid images
            selected_indices = tf.cond(
                is_valid,
                lambda: image_indices[:FIXED_MASK_COUNT],  # Take top FIXED_MASK_COUNT indices
                lambda: placeholder_indices  # Use zeros for invalid images
            )
            
            # Update the mask indices for this image
            indices_update = tf.tensor_scatter_nd_update(
                mask_indices_tensor,
                tf.expand_dims(tf.expand_dims(i, 0), 1),  # [[i]]
                tf.expand_dims(selected_indices, 0)  # [1, FIXED_MASK_COUNT]
            )
            
            return i + 1, indices_update
        
        # Process each image with a TF while loop
        _, mask_indices = tf.while_loop(
            cond=lambda i, _: i < batch_size,
            body=get_indices_for_image,
            loop_vars=(tf.constant(0), mask_indices)
        )
        
        # Create unmask indices using the same approach
        # For uniformity, we'll use a fixed number of unmasked patches as well (NUM_PATCHES - FIXED_MASK_COUNT)
        FIXED_UNMASK_COUNT = self.num_patches - FIXED_MASK_COUNT
        unmask_indices = tf.zeros((batch_size, FIXED_UNMASK_COUNT), dtype=tf.int32)
        
        # Process each image to get unmask indices
        def get_unmask_for_image(i, unmask_indices_tensor):
            # Only process valid images
            is_valid = valid_image[i]
            
            # Get mask indices for this image
            image_mask_indices = mask_indices[i]
            
            # Create a placeholder for invalid images
            placeholder_indices = tf.range(FIXED_UNMASK_COUNT, dtype=tf.int32)
            
            # For valid images, calculate unmask indices properly
            def get_valid_unmask():
                # Get indices of non-masked patches
                mask_indicator = tf.reduce_sum(
                    tf.one_hot(image_mask_indices, self.num_patches, on_value=1, off_value=0, dtype=tf.int32),
                    axis=0
                )
                unmask_raw = tf.boolean_mask(
                    tf.range(self.num_patches, dtype=tf.int32),
                    mask_indicator == 0
                )
                # Pad if necessary to reach FIXED_UNMASK_COUNT
                padding_needed = FIXED_UNMASK_COUNT - tf.shape(unmask_raw)[0]
                padding = tf.zeros(tf.maximum(0, padding_needed), dtype=tf.int32)
                padded_unmask = tf.concat([unmask_raw, padding], axis=0)
                # Make sure we don't exceed FIXED_UNMASK_COUNT
                return padded_unmask[:FIXED_UNMASK_COUNT]
            
            # Choose appropriate indices based on image validity
            image_unmask_indices = tf.cond(
                is_valid,
                get_valid_unmask,
                lambda: placeholder_indices
            )
            
            # Update the unmask indices tensor
            indices_update = tf.tensor_scatter_nd_update(
                unmask_indices_tensor,
                tf.expand_dims(tf.expand_dims(i, 0), 1),  # [[i]]
                tf.expand_dims(image_unmask_indices, 0)  # [1, FIXED_UNMASK_COUNT]
            )
            
            return i + 1, indices_update
        
        # Process each image with a TF while loop
        _, unmask_indices = tf.while_loop(
            cond=lambda i, _: i < batch_size,
            body=get_unmask_for_image,
            loop_vars=(tf.constant(0), unmask_indices)
        )
        
        # Log statistics
        if total_skipped > 0:
            tf.print("Skipped", total_skipped, "images (", 
                     tf.cast(total_skipped, tf.float32) / tf.cast(batch_size, tf.float32) * 100.0,
                     "%) due to insufficient valid patches for required visibility")
        
        # Log proportion stats for valid images (only count non-skipped for averages)
        valid_count_sum = tf.reduce_sum(valid_counts * self.attention_mask)
        valid_image_count = tf.maximum(tf.reduce_sum(self.attention_mask), 1.0)  # Avoid div by zero
        
        valid_avg = valid_count_sum / valid_image_count
        masked_avg = tf.cast(FIXED_MASK_COUNT, tf.float32)
        
        if valid_avg > 0:
            mask_ratio = masked_avg / valid_avg
            tf.print("Average valid patches:", valid_avg, "Fixed mask count:", masked_avg, 
                     "Mask ratio:", mask_ratio, "Target:", max_maskable_percentage)
        
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


########################################


def create_improved_decoder(
    num_layers=DEC_LAYERS, 
    num_heads=DEC_NUM_HEADS, 
    image_size=IMAGE_SIZE,
    channels=CHANNELS,
    patch_size=PATCH_SIZE
):
    """Create a decoder that preserves spatial information for better reconstruction.
    Unlike the GlobalAveragePooling approach, this maintains per-patch representations.
    """
    inputs = layers.Input((None, ENC_PROJECTION_DIM))  # None for variable length
    x = layers.Dense(DEC_PROJECTION_DIM)(inputs)

    # Apply transformer blocks
    for _ in range(num_layers):
        # Layer normalization 1
        x1 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(x)

        # Multi-head attention - allows patches to interact
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=DEC_PROJECTION_DIM, dropout=0.1
        )(x1, x1)

        # Skip connection 1
        x2 = layers.Add()([attention_output, x])

        # Layer normalization 2
        x3 = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(x2)

        # MLP
        x3 = mlp(x3, hidden_units=DEC_TRANSFORMER_UNITS, dropout_rate=0.1)

        # Skip connection 2
        x = layers.Add()([x3, x2])

    # Final layer normalization
    x = layers.LayerNormalization(epsilon=LAYER_NORM_EPS)(x)
    
    # Project each patch embedding to the patch pixel space
    # Each patch gets reconstructed to patch_size * patch_size * channels values
    patch_outputs = layers.Dense(patch_size * patch_size * channels, activation="sigmoid")(x)
    
    # Create a hybrid decoder that works with both approaches:
    # 1. A main branch that treats the output as a sequence of patches (spatial preservation)
    # 2. A backup branch that uses global pooling (for stability when patches don't fit)
    
    # Branch 1: Main spatial information preservation path
    # Just reshape each patch back to proper dimensions
    patches_3d = layers.Reshape((-1, patch_size, patch_size, channels))(patch_outputs)

    # Branch 2: Global pooling fallback (more stable but loses some spatial information)
    global_features = layers.GlobalAveragePooling1D()(x)
    global_output = layers.Dense(image_size * image_size * channels, activation="sigmoid")(global_features)
    global_output = layers.Reshape((image_size, image_size, channels))(global_output)
    
    # Custom layer to combine both approaches
    class HybridDecoder(layers.Layer):
        def __init__(self, patch_size, image_size, channels, mode='spatial'):
            """
            Initialize the HybridDecoder
            
            Args:
                patch_size: Size of each patch
                image_size: Size of the full image
                channels: Number of channels
                mode: Decoder mode - 'global' (use global features only), 'spatial' (use spatial mapping),
                      or 'hybrid' (try spatial first, fall back to global)
            """
            super().__init__(name="patches_to_image_layer")
            self.patch_size = patch_size
            self.image_size = image_size
            self.channels = channels
            self.patches_per_side = image_size // patch_size
            self.num_patches = self.patches_per_side ** 2
            self.mode = mode
            
        def call(self, inputs):
            patches_3d, global_output = inputs
            batch_size = tf.shape(patches_3d)[0]
            actual_patch_count = tf.shape(patches_3d)[1]
            
            # Log info about the shapes for debugging
            tf.print("HybridDecoder shapes - batch_size:", batch_size, 
                    "patches_shape:", tf.shape(patches_3d), 
                    "expected patches:", self.num_patches,
                    "mode:", self.mode)
            
            # Now with fixed patch count, we can use direct spatial mapping
            
            if self.mode == 'global':
                # Just return the global output (fully connected approach)
                return global_output
            
            elif self.mode == 'spatial' or self.mode == 'hybrid':
                # Check if we have the expected number of patches
                patches_match = tf.equal(actual_patch_count, self.num_patches)
                
                def use_spatial_mapping():
                    # Reshape patches to the correct grid shape
                    # [batch_size, num_patches, patch_size, patch_size, channels] ->
                    # [batch_size, patches_per_side, patches_per_side, patch_size, patch_size, channels]
                    grid_shape = [batch_size, self.patches_per_side, self.patches_per_side, 
                                 self.patch_size, self.patch_size, self.channels]
                    
                    # Make sure all dimensions are known to avoid reshape errors
                    patches_reshaped = tf.reshape(patches_3d, grid_shape)
                    
                    # Transpose to get correct ordering for reconstruction
                    # [batch, row, col, patch_h, patch_w, channels] -> [batch, row*patch_h, col*patch_w, channels]
                    reordered = tf.transpose(patches_reshaped, [0, 1, 3, 2, 4, 5])
                    
                    # Reshape to final image dimensions
                    # [batch, row, patch_h, col, patch_w, channels] -> [batch, row*patch_h, col*patch_w, channels]
                    spatial_output = tf.reshape(reordered, 
                                             [batch_size, self.image_size, self.image_size, self.channels])
                    
                    return spatial_output
                
                def use_global_output():
                    # Use the global output as fallback
                    return global_output
                
                if self.mode == 'hybrid':
                    # In hybrid mode, attempt spatial mapping if patch count matches
                    # Otherwise fall back to global output
                    return tf.cond(patches_match, use_spatial_mapping, use_global_output)
                else: 
                    # In spatial mode, always attempt spatial mapping
                    # This might throw an error if patch count doesn't match
                    return use_spatial_mapping()
    
    # Combine both approaches using the hybrid decoder with hybrid mode
    # This will attempt spatial mapping when patch count matches, falling back to global
    # when needed
    outputs = HybridDecoder(patch_size, image_size, channels, mode='hybrid')([patches_3d, global_output])
    
    return keras.Model(inputs, outputs, name="mae_improved_decoder")



######################

'''
def create_decoder(
    num_layers=DEC_LAYERS, num_heads=DEC_NUM_HEADS, image_size=IMAGE_SIZE,
    channels = CHANNELS
):
    """Create a decoder that can handle variable input shapes.
    Now using GlobalAveragePooling1D to handle variable-length inputs."""
    inputs = layers.Input((None, ENC_PROJECTION_DIM))  # None for variable length
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
'''
############################
# 4. Use a custom loss function that focuses on both structure and detail

def combined_reconstruction_loss(y_true, y_pred):
    # MSE for pixel-level reconstruction
    mse_loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    
    # Add a structural similarity term (higher weight to edges and boundaries)
    # Extract edges using gradient approximation
    def extract_edges(x):
        # Simple gradient approximation
        h_grad = x[:, 1:, :, :] - x[:, :-1, :, :]
        v_grad = x[:, :, 1:, :] - x[:, :, :-1, :]
        return tf.concat([h_grad, v_grad], axis=0)
    
    edges_true = extract_edges(y_true)
    edges_pred = extract_edges(y_pred)
    
    # Edge reconstruction loss
    edge_loss = tf.keras.losses.MeanSquaredError()(edges_true, edges_pred)
    
    # Combine losses (give more weight to edge loss)
    combined_loss = 0.7 * mse_loss + 0.3 * edge_loss
    return combined_loss

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