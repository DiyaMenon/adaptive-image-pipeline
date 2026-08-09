#pragma once

#include <cuda_runtime.h>
#include <cstddef>

// Struct holding fine-grained GPU timing metrics in milliseconds
struct GpuTiming {
    float h2d_time_ms;         // Host-to-Device transfer time
    float kernel_time_ms;      // Total kernel execution time (Grayscale + Sobel)
    float d2h_time_ms;         // Device-to-Host transfer time
    float end_to_end_time_ms;  // Total GPU path time (cudaMalloc + H2D + Kernels + D2H + cudaFree)
};

// Filter variants for benchmarking and correctness verification
enum class GpuFilterVariant {
    NAIVE,
    SHARED_MEMORY
};

// CUDA Kernel Launch Wrappers (operate on pre-allocated device memory)
void run_gpu_grayscale_naive(const unsigned char* d_input,
                             unsigned char* d_output,
                             int width,
                             int height,
                             int channels,
                             dim3 block_dim = dim3(16, 16));

void run_gpu_sobel_naive(const unsigned char* d_input_gray,
                         unsigned char* d_output_sobel,
                         int width,
                         int height,
                         dim3 block_dim = dim3(16, 16));

void run_gpu_sobel_shared(const unsigned char* d_input_gray,
                          unsigned char* d_output_sobel,
                          int width,
                          int height,
                          dim3 block_dim = dim3(16, 16));

// Full GPU Pipeline (Host memory -> Device processing -> Host memory)
// Manages allocations, transfers, kernel launches, and timing measurements.
GpuTiming gpu_process_pipeline(const unsigned char* h_input,
                              unsigned char* h_output,
                              int width,
                              int height,
                              int channels,
                              GpuFilterVariant variant = GpuFilterVariant::SHARED_MEMORY,
                              dim3 block_dim = dim3(16, 16));
