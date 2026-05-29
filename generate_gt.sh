#!/bin/bash

# ==============================================================================
# generate_gt.sh - Generate ground-truth files
# The script output is uniquely determined by num_query_sets and K.
# ==============================================================================

set -e # Exit immediately if any command fails

# --- Step 1: Parse command-line arguments ---
while [[ $# -gt 0 ]]; do
    if [[ $1 == --* ]]; then
        key=$(echo "$1" | sed 's/--//' | tr '[:lower:]-' '[:upper:]_')
        if [[ $key == "QUERY_DIR_NAME" ]]; then
            QUERY_DIR_NAME="$2"
            shift 2
            continue
        fi
        if [ -z "$2" ]; then
            echo "Error: Missing value for parameter $1"
            exit 1
        fi
        declare "$key"="$2"
        shift 2
    else
        echo "Unknown parameter: $1"; exit 1
    fi
done

# --- Step 2: Construct the GT directory and filename ---
SAFE_QUERY_NAME=$(echo "$QUERY_DIR_NAME" | tr '/' '_')
GT_DIR_NAME="GT_${SAFE_QUERY_NAME}_K${K}"
GT_OUTPUT_DIR="${EXP_OUTPUT_DIR}/GroundTruth/${GT_DIR_NAME}"
GT_FILE_PATH="${GT_OUTPUT_DIR}/${DATASET}_gt_labels_containment.bin"


# --- Step 3: Skip generation if GT already exists ---
if [ -f "$GT_FILE_PATH" ]; then
    echo "Ground Truth '$GT_FILE_PATH' already exists. Skipping generation."
    exit 0
fi

echo "Ground Truth does not exist. Starting generation: $GT_FILE_PATH"
mkdir -p "$GT_OUTPUT_DIR"

# QUERY_DIR_NAME="query_${QUERY_SUFFIX}"
echo "Using query directory: $QUERY_DIR_NAME"

# --- Step 4: Ensure the query file is in bin format ---
QUERY_BIN_FILE="$DATA_DIR/${QUERY_DIR_NAME}/${DATASET}_query.bin"
if [ ! -f "$QUERY_BIN_FILE" ]; then
    echo "Converting query file format..."
    "$BUILD_DIR"/tools/fvecs_to_bin --data_type float \
        --input_file "$DATA_DIR/${QUERY_DIR_NAME}/${DATASET}_query.fvecs" \
        --output_file "$QUERY_BIN_FILE"
fi

# --- Step 5: Compute GT ---
"$BUILD_DIR"/tools/compute_groundtruth \
      --data_type float --dist_fn L2 --scenario containment --K "$K" --num_threads 32 \
      --base_bin_file "$DATA_DIR/${DATASET}_base.bin" \
      --base_label_file "$DATA_DIR/${DATASET}_base_labels.txt" \
      --query_bin_file "$QUERY_BIN_FILE" \
      --query_label_file "$DATA_DIR/${QUERY_DIR_NAME}/${DATASET}_query_labels.txt" \
      --gt_file "$GT_FILE_PATH"

echo "GT generation completed successfully."
