# CUDA-Accelerated Batch Image Edge Detection and Enhancement

This project implements a CUDA-powered batch image processing pipeline that
converts images to grayscale and applies Sobel edge detection on the GPU.

The program is designed to process either:
- **Hundreds of small images** in a single run, or
- **Tens of large images** (e.g., HD / 4K)

It demonstrates how to use CUDA to accelerate per-pixel image operations at
scale using a simple CLI tool.

---

## Features

- Reads all `.png`, `.jpg`, `.jpeg` images from an input directory
- Converts each image to **grayscale**
- Applies **Sobel edge detection** on the GPU using CUDA kernels
- Writes the processed result to an output directory
- Accepts **command line arguments** to control behavior:
  - Input directory
  - Output directory
  - Operation (`grayscale`, `sobel`, or `both`)
  - Max number of images
  - Block size (CUDA threads per block side)
- Logs total images processed and timing information

---

## Requirements

- CUDA Toolkit (nvcc)
- C++14 or newer
- A CUDA-capable GPU
- `stb_image.h` and `stb_image_write.h` single-header libraries

Download `stb_image.h` and `stb_image_write.h` from the official stb repository
and place them in the `include/` directory:

- `include/stb_image.h`
- `include/stb_image_write.h`

---

## Building

From the project root:

```bash
make
