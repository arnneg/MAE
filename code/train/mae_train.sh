#!/bin/bash
set -e

# Get the dataset name from arguments
DATASET_NAME=$1

# Create input and output directories
mkdir -p input output

# Extract input data
echo "Extracting input data..."
tar -xzf /staging/negreiro/training_datasets/${DATASET_NAME}.tar.gz -C input

# Print Python version and path
which python3
python3 --version

# Run the Python script
echo "Running Python script..."
python3 -u mae_train.py --data_dir input/MAE_dataset --output_dir output

# Check for model files
echo "Checking for model files..."
if [ -d "output/mae_model" ]; then
    echo "✓ Model checkpoint directory exists."
    ls -la output/mae_model/
    MODEL_FILES_COUNT=$(find output/mae_model -type f | wc -l)
    echo "Found $MODEL_FILES_COUNT model checkpoint files."
else
    echo "WARNING: Model checkpoint directory 'output/mae_model' not found!"
fi

if [ -d "output/weights" ]; then
    echo "✓ Model weights directory exists."
    ls -la output/weights/
    WEIGHT_FILES_COUNT=$(find output/weights -type f | wc -l)
    echo "Found $WEIGHT_FILES_COUNT weight files."
else
    echo "WARNING: Model weights directory 'output/weights' not found!"
fi

# List all content in output directory
echo "Contents of output directory before tarball creation:"
find output -type f | sort

# Check disk space
echo "Disk space usage:"
df -h

# Compress results with verbose output
echo "Compressing results..."
tar -czvf ${DATASET_NAME}_results.tar.gz -C output .

# Verify the tarball contents
echo "Verifying tarball contents..."
tar -tvf ${DATASET_NAME}_results.tar.gz | grep -E 'mae_model|weights' || echo "No model files found in tarball!"

# Calculate tarball size
TARBALL_SIZE=$(du -h ${DATASET_NAME}_results.tar.gz | cut -f1)
echo "Tarball size: $TARBALL_SIZE"

# Move results to the staging area
echo "Moving results to staging area..."
cp ${DATASET_NAME}_results.tar.gz /staging/negreiro/${DATASET_NAME}_results.tar.gz
echo "Results copied to /staging/negreiro/${DATASET_NAME}_results.tar.gz"

echo "Script completed for dataset: $DATASET_NAME"