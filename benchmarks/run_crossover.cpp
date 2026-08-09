#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>
#include <numeric>
#include <random>

#include "cpu_filters.h"
#include "gpu_filters.cuh"

struct Stats {
    double mean;
    double stddev;
};

static Stats compute_stats(const std::vector<double>& values) {
    if (values.empty()) return {0.0, 0.0};
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    double mean = sum / values.size();

    double sq_sum = 0.0;
    for (double v : values) {
        sq_sum += (v - mean) * (v - mean);
    }
    double stddev = values.size() > 1 ? std::sqrt(sq_sum / (values.size() - 1)) : 0.0;
    return {mean, stddev};
}

static std::vector<unsigned char> generate_benchmark_image(int w, int h, int c) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<unsigned char> img(w * h * c);
    for (size_t i = 0; i < img.size(); ++i) {
        img[i] = static_cast<unsigned char>(dist(rng));
    }
    return img;
}

int main() {
    // Fine-grained dimensions between 128x128 and 256x256
    const std::vector<int> dimensions = {128, 144, 160, 176, 192, 208, 224, 240, 256};
    const int total_runs = 10;
    const int warmup_runs = 2;
    const std::string csv_path = "results/crossover_benchmark.csv";

    std::cout << "===================================================================\n";
    std::cout << "         Fine-Grained CPU vs GPU Crossover Benchmark Sweep         \n";
    std::cout << "===================================================================\n";
    std::cout << "Testing dimensions from 128x128 to 256x256 (step ~16px)\n";
    std::cout << "10 runs per size (first 2 discarded as warm-up)\n";
    std::cout << "Writing raw output to: " << csv_path << "\n\n";

    std::ofstream csv_file(csv_path);
    if (!csv_file.is_open()) {
        std::cerr << "Error: Unable to open " << csv_path << " for writing!\n";
        return 1;
    }

    // CSV Header matching standard schema
    csv_file << "size,variant,run_index,cpu_time_ms,kernel_time_ms,h2d_time_ms,d2h_time_ms,end_to_end_time_ms\n";

    std::cout << std::left 
              << std::setw(10) << "Size"
              << std::setw(12) << "Pixels"
              << std::setw(16) << "CPU (ms)"
              << std::setw(20) << "GPU Naive E2E(ms)"
              << std::setw(20) << "GPU Shared E2E(ms)"
              << std::setw(14) << "Faster Path"
              << "\n";
    std::cout << "--------------------------------------------------------------------------------------------\n";

    int detected_crossover_dim = -1;
    int detected_crossover_pixels = -1;

    for (int dim : dimensions) {
        const int w = dim;
        const int h = dim;
        const int c = 3;
        const int total_pixels = w * h;

        auto input_img = generate_benchmark_image(w, h, c);
        std::vector<unsigned char> cpu_out(total_pixels, 0);
        std::vector<unsigned char> gpu_naive_out(total_pixels, 0);
        std::vector<unsigned char> gpu_shared_out(total_pixels, 0);

        std::vector<double> cpu_times;
        std::vector<double> gpu_naive_e2e_times;
        std::vector<double> gpu_shared_e2e_times;

        for (int r = 0; r < total_runs; ++r) {
            // 1. CPU Execution
            double cpu_ms = cpu_process_pipeline(input_img.data(), cpu_out.data(), w, h, c);

            // 2. GPU Naive Execution
            GpuTiming naive_timing = gpu_process_pipeline(input_img.data(), gpu_naive_out.data(),
                                                          w, h, c, GpuFilterVariant::NAIVE, dim3(16, 16));

            // 3. GPU Shared-Memory Execution
            GpuTiming shared_timing = gpu_process_pipeline(input_img.data(), gpu_shared_out.data(),
                                                           w, h, c, GpuFilterVariant::SHARED_MEMORY, dim3(16, 16));

            // Write raw CSV rows
            csv_file << dim << ",CPU," << r << "," << std::fixed << std::setprecision(4)
                     << cpu_ms << ",0.0000,0.0000,0.0000," << cpu_ms << "\n";

            csv_file << dim << ",GPU_Naive," << r << "," << std::fixed << std::setprecision(4)
                     << cpu_ms << "," << naive_timing.kernel_time_ms << "," << naive_timing.h2d_time_ms << ","
                     << naive_timing.d2h_time_ms << "," << naive_timing.end_to_end_time_ms << "\n";

            csv_file << dim << ",GPU_Shared," << r << "," << std::fixed << std::setprecision(4)
                     << cpu_ms << "," << shared_timing.kernel_time_ms << "," << shared_timing.h2d_time_ms << ","
                     << shared_timing.d2h_time_ms << "," << shared_timing.end_to_end_time_ms << "\n";

            if (r >= warmup_runs) {
                cpu_times.push_back(cpu_ms);
                gpu_naive_e2e_times.push_back(naive_timing.end_to_end_time_ms);
                gpu_shared_e2e_times.push_back(shared_timing.end_to_end_time_ms);
            }
        }

        Stats cpu_s = compute_stats(cpu_times);
        Stats naive_e2e_s = compute_stats(gpu_naive_e2e_times);
        Stats shared_e2e_s = compute_stats(gpu_shared_e2e_times);

        std::string faster = (cpu_s.mean <= shared_e2e_s.mean) ? "CPU" : "GPU (Shared)";

        if (detected_crossover_dim == -1 && shared_e2e_s.mean < cpu_s.mean) {
            detected_crossover_dim = dim;
            detected_crossover_pixels = total_pixels;
        }

        std::cout << std::left
                  << std::setw(10) << (std::to_string(dim) + "x" + std::to_string(dim))
                  << std::setw(12) << total_pixels
                  << std::setw(16) << (std::to_string(cpu_s.mean).substr(0, 6) + " ms")
                  << std::setw(20) << (std::to_string(naive_e2e_s.mean).substr(0, 6) + " ms")
                  << std::setw(20) << (std::to_string(shared_e2e_s.mean).substr(0, 6) + " ms")
                  << std::setw(14) << faster
                  << "\n";
    }

    csv_file.close();

    std::cout << "--------------------------------------------------------------------------------------------\n";
    if (detected_crossover_dim != -1) {
        std::cout << ">>> First Crossover Point Detected at: " << detected_crossover_dim << "x" << detected_crossover_dim
                  << " (" << detected_crossover_pixels << " pixels) <<<\n";
    } else {
        std::cout << ">>> CPU was faster across all tested fine-grained dimensions <<<\n";
    }
    std::cout << "===================================================================\n";
    std::cout << "Results saved to: " << csv_path << "\n";
    std::cout << "===================================================================\n";

    return 0;
}
