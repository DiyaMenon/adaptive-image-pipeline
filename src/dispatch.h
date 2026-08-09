#pragma once

#include <string>

// Execution target decided by the adaptive dispatcher
enum class ExecutionTarget {
    CPU,
    GPU
};

// Empirically measured crossover threshold on NVIDIA Tesla T4 in Google Colab:
// - 128x128 (16,384 pixels): CPU (0.1954 ms) < GPU Shared E2E (0.2215 ms) -> CPU is faster
// - 144x144 (20,736 pixels): CPU (0.3009 ms) > GPU Shared E2E (0.2796 ms) -> GPU is faster
// Measured on remote NVIDIA Tesla T4 GPU (Google Colab), NOT on the Apple M2 development machine.
constexpr int CROSSOVER_PIXEL_THRESHOLD = 20736;

// Dispatches workload to CPU or GPU based on image dimensions.
// Returns ExecutionTarget::CPU if width * height < 20736, and ExecutionTarget::GPU otherwise.
ExecutionTarget dispatch(int width, int height);

// Returns a human-readable name of the execution target.
const char* execution_target_to_string(ExecutionTarget target);

// Executes the adaptive pipeline (Grayscale + Sobel) using the optimal target.
// Returns the total elapsed execution time in milliseconds.
double run_adaptive_pipeline(const unsigned char* input_rgb,
                            unsigned char* output_sobel,
                            int width,
                            int height,
                            int channels,
                            ExecutionTarget* out_target_used = nullptr);
