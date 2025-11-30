#!/usr/bin/env bash

set -e

INPUT_DIR="data/input"
OUTPUT_DIR="data/output"
OPERATION="both"
MAX_IMAGES=300
BLOCK_SIZE=16

mkdir -p "$OUTPUT_DIR"
mkdir -p "artifacts/logs"

LOG_FILE="artifacts/logs/run_$(date +%Y%m%d_%H%M%S).txt"

./bin/cuda_batch_image_processor \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --operation "$OPERATION" \
  --max_images "$MAX_IMAGES" \
  --block_size "$BLOCK_SIZE" \
  | tee "$LOG_FILE"
