
---

## 4. `src/main.cu` (CUDA Code)

Create `src/main.cu` with something like this (you can tweak if needed):

```cpp
// src/main.cu

#include <cuda_runtime.h>

#include <dirent.h>
#include <sys/stat.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Simple CUDA error checking macro.
#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err__ = (call);                                              \
    if (err__ != cudaSuccess) {                                              \
      std::cerr << "CUDA error: " << cudaGetErrorString(err__)              \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;      \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

// Options parsed from command line.
struct Options {
  std::string input_dir;
  std::string output_dir;
  std::string operation = "both";  // grayscale | sobel | both
  int max_images = -1;             // <= 0 means "all"
  int block_size = 16;
};

struct ImageInfo {
  std::string input_path;
  std::string output_path;
};

// ------------------ CUDA Kernels ------------------

// Convert RGB (3 channels, uint8) to grayscale (1 channel, float).
__global__ void RgbToGrayscaleKernel(const unsigned char* rgb,
                                     float* gray,
                                     int width,
                                     int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;

  int idx = (y * width + x) * 3;
  unsigned char r = rgb[idx + 0];
  unsigned char g = rgb[idx + 1];
  unsigned char b = rgb[idx + 2];

  // Standard luminance formula.
  float val = 0.299f * static_cast<float>(r) +
              0.587f * static_cast<float>(g) +
              0.114f * static_cast<float>(b);

  gray[y * width + x] = val;
}

// Sobel edge detection on grayscale input; outputs uint8 magnitude.
__global__ void SobelKernel(const float* gray,
                            unsigned char* edges,
                            int width,
                            int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;

  // Ignore 1-pixel border.
  if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
    edges[y * width + x] = 0;
    return;
  }

  // Sobel kernels.
  float gx =
      -1 * gray[(y - 1) * width + (x - 1)] +
      -2 * gray[(y    ) * width + (x - 1)] +
      -1 * gray[(y + 1) * width + (x - 1)] +
       1 * gray[(y - 1) * width + (x + 1)] +
       2 * gray[(y    ) * width + (x + 1)] +
       1 * gray[(y + 1) * width + (x + 1)];

  float gy =
      -1 * gray[(y - 1) * width + (x - 1)] +
      -2 * gray[(y - 1) * width + (x    )] +
      -1 * gray[(y - 1) * width + (x + 1)] +
       1 * gray[(y + 1) * width + (x - 1)] +
       2 * gray[(y + 1) * width + (x    )] +
       1 * gray[(y + 1) * width + (x + 1)];

  float mag = sqrtf(gx * gx + gy * gy);

  // Clamp to [0, 255].
  mag = fminf(255.0f, fmaxf(0.0f, mag));
  edges[y * width + x] = static_cast<unsigned char>(mag);
}

// ------------------ Utility Functions ------------------

bool EndsWith(const std::string& s, const std::string& suffix) {
  if (suffix.size() > s.size()) return false;
  return std::equal(suffix.rbegin(), suffix.rend(), s.rbegin());
}

bool IsImageFile(const std::string& name) {
  std::string lower = name;
  for (char& c : lower) c = static_cast<char>(std::tolower(c));
  return EndsWith(lower, ".png") || EndsWith(lower, ".jpg") ||
         EndsWith(lower, ".jpeg");
}

bool DirectoryExists(const std::string& path) {
  struct stat st;
  if (stat(path.c_str(), &st) != 0) return false;
  return S_ISDIR(st.st_mode);
}

void EnsureDirectory(const std::string& path) {
  if (!DirectoryExists(path)) {
    // Try to create it.
#ifdef _WIN32
    _mkdir(path.c_str());
#else
    mkdir(path.c_str(), 0755);
#endif
  }
}

std::vector<ImageInfo> ListImages(const std::string& input_dir,
                                  const std::string& output_dir,
                                  int max_images) {
  std::vector<ImageInfo> result;
  DIR* dir = opendir(input_dir.c_str());
  if (!dir) {
    std::cerr << "Failed to open input directory: " << input_dir << std::endl;
    return result;
  }

  struct dirent* entry;
  while ((entry = readdir(dir)) != nullptr) {
    std::string name = entry->d_name;
    if (name == "." || name == "..") continue;
    if (!IsImageFile(name)) continue;

    ImageInfo info;
    info.input_path = input_dir + "/" + name;
    info.output_path = output_dir + "/" + name;  // same file name
    result.push_back(info);

    if (max_images > 0 &&
        static_cast<int>(result.size()) >= max_images) {
      break;
    }
  }

  closedir(dir);
  return result;
}

bool LoadImageRgb(const std::string& path,
                  std::vector<unsigned char>* data,
                  int* width,
                  int* height) {
  int channels = 0;
  unsigned char* img = stbi_load(path.c_str(), width, height,
                                 &channels, 3 /*force RGB*/);
  if (!img) {
    std::cerr << "Failed to load image: " << path << std::endl;
    return false;
  }

  size_t size = static_cast<size_t>(*width) * (*height) * 3;
  data->assign(img, img + size);
  stbi_image_free(img);
  return true;
}

bool SaveImageGray(const std::string& path,
                   const unsigned char* data,
                   int width,
                   int height) {
  // Save as PNG, 1 channel.
  int success = stbi_write_png(path.c_str(), width, height,
                               1, data, width);
  if (!success) {
    std::cerr << "Failed to write image: " << path << std::endl;
    return false;
  }
  return true;
}

void PrintUsage(const char* prog) {
  std::cout << "Usage:\n"
            << prog
            << " --input_dir <path> --output_dir <path>\n"
            << "          [--operation grayscale|sobel|both]\n"
            << "          [--max_images N]\n"
            << "          [--block_size N]\n";
}

// Parse command line arguments into Options.
bool ParseArgs(int argc, char** argv, Options* options) {
  if (argc < 5) {
    PrintUsage(argv[0]);
    return false;
  }

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--input_dir" && i + 1 < argc) {
      options->input_dir = argv[++i];
    } else if (arg == "--output_dir" && i + 1 < argc) {
      options->output_dir = argv[++i];
    } else if (arg == "--operation" && i + 1 < argc) {
      options->operation = argv[++i];
    } else if (arg == "--max_images" && i + 1 < argc) {
      options->max_images = std::atoi(argv[++i]);
    } else if (arg == "--block_size" && i + 1 < argc) {
      options->block_size = std::atoi(argv[++i]);
    } else {
      std::cerr << "Unknown or incomplete argument: " << arg << std::endl;
      PrintUsage(argv[0]);
      return false;
    }
  }

  if (options->input_dir.empty() || options->output_dir.empty()) {
    std::cerr << "Both --input_dir and --output_dir are required.\n";
    PrintUsage(argv[0]);
    return false;
  }

  if (options->operation != "grayscale" &&
      options->operation != "sobel" &&
      options->operation != "both") {
    std::cerr << "Invalid --operation: " << options->operation << std::endl;
    return false;
  }

  if (options->block_size <= 0) {
    options->block_size = 16;
  }

  return true;
}

// Process a single image: host code + CUDA kernels.
bool ProcessSingleImage(const ImageInfo& info,
                        const Options& options) {
  int width = 0;
  int height = 0;
  std::vector<unsigned char> h_rgb;

  if (!LoadImageRgb(info.input_path, &h_rgb, &width, &height)) {
    return false;
  }

  std::cout << "Processing: " << info.input_path
            << " (" << width << "x" << height << ")\n";

  size_t num_pixels = static_cast<size_t>(width) * height;
  size_t rgb_bytes = num_pixels * 3;
  size_t gray_bytes = num_pixels * sizeof(float);
  size_t edge_bytes = num_pixels;  // uint8

  unsigned char* d_rgb = nullptr;
  float* d_gray = nullptr;
  unsigned char* d_edges = nullptr;

  CUDA_CHECK(cudaMalloc(&d_rgb, rgb_bytes));
  CUDA_CHECK(cudaMalloc(&d_gray, gray_bytes));
  CUDA_CHECK(cudaMalloc(&d_edges, edge_bytes));

  CUDA_CHECK(cudaMemcpy(d_rgb, h_rgb.data(), rgb_bytes,
                        cudaMemcpyHostToDevice));

  dim3 block(options.block_size, options.block_size);
  dim3 grid((width + block.x - 1) / block.x,
            (height + block.y - 1) / block.y);

  // Events for timing individual image if desired.
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));

  // Grayscale is always computed if operation != sobel-only? We need it
  // for sobel anyway.
  RgbToGrayscaleKernel<<<grid, block>>>(d_rgb, d_gray, width, height);
  CUDA_CHECK(cudaGetLastError());

  if (options.operation == "sobel" || options.operation == "both") {
    SobelKernel<<<grid, block>>>(d_gray, d_edges, width, height);
    CUDA_CHECK(cudaGetLastError());
  }

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  std::cout << "  GPU processing time: " << ms << " ms\n";

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  bool ok = true;

  if (options.operation == "grayscale") {
    // Download grayscale, convert to uint8, save.
    std::vector<float> h_gray(num_pixels);
    CUDA_CHECK(cudaMemcpy(h_gray.data(), d_gray, gray_bytes,
                          cudaMemcpyDeviceToHost));

    std::vector<unsigned char> h_gray_u8(num_pixels);
    for (size_t i = 0; i < num_pixels; ++i) {
      float v = fminf(255.0f, fmaxf(0.0f, h_gray[i]));
      h_gray_u8[i] = static_cast<unsigned char>(v);
    }

    ok = SaveImageGray(info.output_path, h_gray_u8.data(), width, height);
  } else {
    // Download edges and save.
    std::vector<unsigned char> h_edges(num_pixels);
    CUDA_CHECK(cudaMemcpy(h_edges.data(), d_edges, edge_bytes,
                          cudaMemcpyDeviceToHost));

    // If operation == "both", we still only save edges here for simplicity.
    ok = SaveImageGray(info.output_path, h_edges.data(), width, height);
  }

  CUDA_CHECK(cudaFree(d_rgb));
  CUDA_CHECK(cudaFree(d_gray));
  CUDA_CHECK(cudaFree(d_edges));

  return ok;
}

// ------------------ main ------------------

int main(int argc, char** argv) {
  Options options;
  if (!ParseArgs(argc, argv, &options)) {
    return EXIT_FAILURE;
  }

  if (!DirectoryExists(options.input_dir)) {
    std::cerr << "Input directory does not exist: "
              << options.input_dir << std::endl;
    return EXIT_FAILURE;
  }

  EnsureDirectory(options.output_dir);

  std::vector<ImageInfo> images =
      ListImages(options.input_dir, options.output_dir, options.max_images);

  if (images.empty()) {
    std::cerr << "No images found in: " << options.input_dir << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Found " << images.size() << " images to process.\n";
  std::cout << "Operation: " << options.operation << "\n";
  std::cout << "Block size: " << options.block_size
            << "x" << options.block_size << "\n";

  cudaEvent_t start_all, stop_all;
  CUDA_CHECK(cudaEventCreate(&start_all));
  CUDA_CHECK(cudaEventCreate(&stop_all));
  CUDA_CHECK(cudaEventRecord(start_all));

  int success_count = 0;
  for (const auto& img : images) {
    if (ProcessSingleImage(img, options)) {
      ++success_count;
    }
  }

  CUDA_CHECK(cudaEventRecord(stop_all));
  CUDA_CHECK(cudaEventSynchronize(stop_all));
  float ms_all = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms_all, start_all, stop_all));

  CUDA_CHECK(cudaEventDestroy(start_all));
  CUDA_CHECK(cudaEventDestroy(stop_all));

  std::cout << "Successfully processed " << success_count
            << " / " << images.size() << " images.\n";
  std::cout << "Total GPU time (all images): " << ms_all << " ms\n";

  return (success_count == static_cast<int>(images.size()))
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
