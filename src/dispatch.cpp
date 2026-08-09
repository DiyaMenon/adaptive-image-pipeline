#include "dispatch.h"
#include "cpu_filters.h"

#include <iostream>

#if defined(NO_CUDA)
#define CUDA_AVAILABLE 0
#elif defined(__CUDACC__) || __has_include(<cuda_runtime.h>)
#include "gpu_filters.cuh"
#define CUDA_AVAILABLE 1
#else
#define CUDA_AVAILABLE 0
#endif

// Empirically measured on NVIDIA Tesla T4 GPU in Google Colab:
// - 128x128 (16,384 pixels): CPU (0.3008 ms) < GPU Shared E2E (0.3241 ms) -> CPU
// - 144x144 (20,736 pixels): CPU (0.3348 ms) < GPU Shared E2E (0.5604 ms) -> CPU
// - 160x160 (25,600 pixels): CPU (0.4010 ms) > GPU Shared E2E (0.2880 ms) -> GPU
// Note: This threshold was measured on Google Colab's Tesla T4 instance, not the M2 development machine.
ExecutionTarget dispatch(int width, int height) {
    const int pixel_count = width * height;
    if (pixel_count < CROSSOVER_PIXEL_THRESHOLD) {
        return ExecutionTarget::CPU;
    } else {
        return ExecutionTarget::GPU;
    }
}

const char* execution_target_to_string(ExecutionTarget target) {
    switch (target) {
        case ExecutionTarget::CPU:
            return "CPU (Single-threaded)";
        case ExecutionTarget::GPU:
            return "GPU (Shared-Memory Kernel)";
        default:
            return "Unknown";
    }
}

double run_adaptive_pipeline(const unsigned char* input_rgb,
                            unsigned char* output_sobel,
                            int width,
                            int height,
                            int channels,
                            ExecutionTarget* out_target_used) {
    ExecutionTarget target = dispatch(width, height);
    if (out_target_used) {
        *out_target_used = target;
    }

    if (target == ExecutionTarget::CPU) {
        return cpu_process_pipeline(input_rgb, output_sobel, width, height, channels);
    } else {
#if CUDA_AVAILABLE
        GpuTiming timing = gpu_process_pipeline(input_rgb, output_sobel, width, height, channels,
                                                GpuFilterVariant::SHARED_MEMORY);
        return static_cast<double>(timing.end_to_end_time_ms);
#else
        std::cerr << "Warning: GPU path requested but CUDA runtime is not available on this host. Falling back to CPU.\n";
        return cpu_process_pipeline(input_rgb, output_sobel, width, height, channels);
#endif
    }
}
