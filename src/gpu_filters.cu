#include "gpu_filters.cuh"

#include <iostream>
#include <cmath>
#include <chrono>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__        \
                      << " - " << cudaGetErrorString(err) << std::endl;        \
        }                                                                      \
    } while (0)

// ============================================================================
// CUDA Kernels
// ============================================================================

// Naive Grayscale Kernel: 1 thread per pixel, direct global memory reads/writes.
// Luminosity formula: Y = 0.299*R + 0.587*G + 0.114*B
__global__ void grayscale_naive_kernel(const unsigned char* __restrict__ d_input,
                                       unsigned char* __restrict__ d_output,
                                       int width,
                                       int height,
                                       int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    int idx = y * width + x;
    if (channels >= 3) {
        int src_idx = idx * channels;
        float r = static_cast<float>(d_input[src_idx]);
        float g = static_cast<float>(d_input[src_idx + 1]);
        float b = static_cast<float>(d_input[src_idx + 2]);
        float gray = 0.299f * r + 0.587f * g + 0.114f * b;
        d_output[idx] = static_cast<unsigned char>(gray > 255.0f ? 255.0f : (gray < 0.0f ? 0.0f : gray));
    } else {
        d_output[idx] = d_input[idx * channels];
    }
}

// Naive Sobel Kernel: 1 thread per pixel, reads 3x3 neighborhood from global memory.
// Boundary pixels (x=0, x=w-1, y=0, y=h-1) are set to 0.
__global__ void sobel_naive_kernel(const unsigned char* __restrict__ d_input_gray,
                                   unsigned char* __restrict__ d_output_sobel,
                                   int width,
                                   int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    int idx = y * width + x;

    // Zero out image borders or handle tiny images
    if (width < 3 || height < 3 || x == 0 || x == width - 1 || y == 0 || y == height - 1) {
        d_output_sobel[idx] = 0;
        return;
    }

    // Read 3x3 neighborhood from global memory
    int p00 = d_input_gray[(y - 1) * width + (x - 1)];
    int p01 = d_input_gray[(y - 1) * width + x];
    int p02 = d_input_gray[(y - 1) * width + (x + 1)];

    int p10 = d_input_gray[y * width + (x - 1)];
    int p12 = d_input_gray[y * width + (x + 1)];

    int p20 = d_input_gray[(y + 1) * width + (x - 1)];
    int p21 = d_input_gray[(y + 1) * width + x];
    int p22 = d_input_gray[(y + 1) * width + (x + 1)];

    // Convolve with Sobel horizontal (Gx) and vertical (Gy) kernels
    int gx = -p00 + p02
             - 2 * p10 + 2 * p12
             - p20 + p22;

    int gy = -p00 - 2 * p01 - p02
             + p20 + 2 * p21 + p22;

    float mag = sqrtf(static_cast<float>(gx * gx + gy * gy));
    d_output_sobel[idx] = static_cast<unsigned char>(mag > 255.0f ? 255.0f : mag);
}

// Shared-Memory Sobel Kernel:
// Cooperatively loads a (blockDim.x + 2) x (blockDim.y + 2) tile into shared memory,
// including a 1-pixel boundary halo. Threads synchronize with __syncthreads(),
// then compute Sobel gradients from shared memory without redundant global memory traffic.
extern __shared__ unsigned char s_tile[];

__global__ void sobel_shared_kernel(const unsigned char* __restrict__ d_input_gray,
                                    unsigned char* __restrict__ d_output_sobel,
                                    int width,
                                    int height) {
    int tile_w = blockDim.x + 2;
    int tile_h = blockDim.y + 2;
    int total_tile_elements = tile_w * tile_h;
    int num_threads = blockDim.x * blockDim.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Origin of this block's tile in global coordinates (offset by -1 for the halo)
    int origin_x = blockIdx.x * blockDim.x - 1;
    int origin_y = blockIdx.y * blockDim.y - 1;

    // 1. Cooperative Load: All threads participate to populate the tile and halo
    for (int i = tid; i < total_tile_elements; i += num_threads) {
        int smem_y = i / tile_w;
        int smem_x = i % tile_w;
        int gx = origin_x + smem_x;
        int gy = origin_y + smem_y;

        if (gx >= 0 && gx < width && gy >= 0 && gy < height) {
            s_tile[smem_y * tile_w + smem_x] = d_input_gray[gy * width + gx];
        } else {
            s_tile[smem_y * tile_w + smem_x] = 0; // Pad out-of-bounds halo with 0
        }
    }

    // Ensure entire tile is loaded before computing
    __syncthreads();

    // 2. Compute Phase: Each active thread computes Sobel for its pixel
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    int out_idx = y * width + x;

    // Zero out image borders or handle tiny images
    if (width < 3 || height < 3 || x == 0 || x == width - 1 || y == 0 || y == height - 1) {
        d_output_sobel[out_idx] = 0;
        return;
    }

    // Shared memory coordinates corresponding to thread's pixel (offset by 1 for halo)
    int sx = threadIdx.x + 1;
    int sy = threadIdx.y + 1;

    int p00 = s_tile[(sy - 1) * tile_w + (sx - 1)];
    int p01 = s_tile[(sy - 1) * tile_w + sx];
    int p02 = s_tile[(sy - 1) * tile_w + (sx + 1)];

    int p10 = s_tile[sy * tile_w + (sx - 1)];
    int p12 = s_tile[sy * tile_w + (sx + 1)];

    int p20 = s_tile[(sy + 1) * tile_w + (sx - 1)];
    int p21 = s_tile[(sy + 1) * tile_w + sx];
    int p22 = s_tile[(sy + 1) * tile_w + (sx + 1)];

    int gx = -p00 + p02
             - 2 * p10 + 2 * p12
             - p20 + p22;

    int gy = -p00 - 2 * p01 - p02
             + p20 + 2 * p21 + p22;

    float mag = sqrtf(static_cast<float>(gx * gx + gy * gy));
    d_output_sobel[out_idx] = static_cast<unsigned char>(mag > 255.0f ? 255.0f : mag);
}

// ============================================================================
// Kernel Launch Wrappers
// ============================================================================

void run_gpu_grayscale_naive(const unsigned char* d_input,
                             unsigned char* d_output,
                             int width,
                             int height,
                             int channels,
                             dim3 block_dim) {
    dim3 grid_dim((width + block_dim.x - 1) / block_dim.x,
                  (height + block_dim.y - 1) / block_dim.y);
    grayscale_naive_kernel<<<grid_dim, block_dim>>>(d_input, d_output, width, height, channels);
    CUDA_CHECK(cudaGetLastError());
}

void run_gpu_sobel_naive(const unsigned char* d_input_gray,
                         unsigned char* d_output_sobel,
                         int width,
                         int height,
                         dim3 block_dim) {
    dim3 grid_dim((width + block_dim.x - 1) / block_dim.x,
                  (height + block_dim.y - 1) / block_dim.y);
    sobel_naive_kernel<<<grid_dim, block_dim>>>(d_input_gray, d_output_sobel, width, height);
    CUDA_CHECK(cudaGetLastError());
}

void run_gpu_sobel_shared(const unsigned char* d_input_gray,
                          unsigned char* d_output_sobel,
                          int width,
                          int height,
                          dim3 block_dim) {
    dim3 grid_dim((width + block_dim.x - 1) / block_dim.x,
                  (height + block_dim.y - 1) / block_dim.y);
    size_t smem_size = (block_dim.x + 2) * (block_dim.y + 2) * sizeof(unsigned char);
    sobel_shared_kernel<<<grid_dim, block_dim, smem_size>>>(d_input_gray, d_output_sobel, width, height);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Full GPU Pipeline with Event-Based Timing
// ============================================================================

GpuTiming gpu_process_pipeline(const unsigned char* h_input,
                              unsigned char* h_output,
                              int width,
                              int height,
                              int channels,
                              GpuFilterVariant variant,
                              dim3 block_dim) {
    GpuTiming timing = {0.0f, 0.0f, 0.0f, 0.0f};
    const size_t input_bytes = width * height * channels * sizeof(unsigned char);
    const size_t gray_bytes = width * height * sizeof(unsigned char);
    const size_t output_bytes = width * height * sizeof(unsigned char);

    // Create CUDA events for high-precision device timing
    cudaEvent_t start_total, stop_total;
    cudaEvent_t start_h2d, stop_h2d;
    cudaEvent_t start_kernel, stop_kernel;
    cudaEvent_t start_d2h, stop_d2h;

    CUDA_CHECK(cudaEventCreate(&start_total));
    CUDA_CHECK(cudaEventCreate(&stop_total));
    CUDA_CHECK(cudaEventCreate(&start_h2d));
    CUDA_CHECK(cudaEventCreate(&stop_h2d));
    CUDA_CHECK(cudaEventCreate(&start_kernel));
    CUDA_CHECK(cudaEventCreate(&stop_kernel));
    CUDA_CHECK(cudaEventCreate(&start_d2h));
    CUDA_CHECK(cudaEventCreate(&stop_d2h));

    CUDA_CHECK(cudaEventRecord(start_total));

    // 1. Allocate device memory
    unsigned char *d_input = nullptr, *d_gray = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
    CUDA_CHECK(cudaMalloc(&d_gray, gray_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, output_bytes));

    // 2. Host-to-Device Memory Transfer
    CUDA_CHECK(cudaEventRecord(start_h2d));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop_h2d));

    // 3. Kernel Execution (Grayscale + Sobel)
    dim3 grid_dim((width + block_dim.x - 1) / block_dim.x,
                  (height + block_dim.y - 1) / block_dim.y);

    CUDA_CHECK(cudaEventRecord(start_kernel));
    grayscale_naive_kernel<<<grid_dim, block_dim>>>(d_input, d_gray, width, height, channels);

    if (variant == GpuFilterVariant::SHARED_MEMORY) {
        size_t smem_size = (block_dim.x + 2) * (block_dim.y + 2) * sizeof(unsigned char);
        sobel_shared_kernel<<<grid_dim, block_dim, smem_size>>>(d_gray, d_output, width, height);
    } else {
        sobel_naive_kernel<<<grid_dim, block_dim>>>(d_gray, d_output, width, height);
    }
    CUDA_CHECK(cudaEventRecord(stop_kernel));

    // 4. Device-to-Host Memory Transfer
    CUDA_CHECK(cudaEventRecord(start_d2h));
    CUDA_CHECK(cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop_d2h));

    // 5. Free Device Memory
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_gray));
    CUDA_CHECK(cudaFree(d_output));

    CUDA_CHECK(cudaEventRecord(stop_total));
    CUDA_CHECK(cudaEventSynchronize(stop_total));

    // Compute elapsed times
    CUDA_CHECK(cudaEventElapsedTime(&timing.h2d_time_ms, start_h2d, stop_h2d));
    CUDA_CHECK(cudaEventElapsedTime(&timing.kernel_time_ms, start_kernel, stop_kernel));
    CUDA_CHECK(cudaEventElapsedTime(&timing.d2h_time_ms, start_d2h, stop_d2h));
    CUDA_CHECK(cudaEventElapsedTime(&timing.end_to_end_time_ms, start_total, stop_total));

    // Clean up events
    CUDA_CHECK(cudaEventDestroy(start_total));
    CUDA_CHECK(cudaEventDestroy(stop_total));
    CUDA_CHECK(cudaEventDestroy(start_h2d));
    CUDA_CHECK(cudaEventDestroy(stop_h2d));
    CUDA_CHECK(cudaEventDestroy(start_kernel));
    CUDA_CHECK(cudaEventDestroy(stop_kernel));
    CUDA_CHECK(cudaEventDestroy(start_d2h));
    CUDA_CHECK(cudaEventDestroy(stop_d2h));

    return timing;
}
