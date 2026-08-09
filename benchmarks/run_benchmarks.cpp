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

struct BenchmarkRun {
    int size;
    std::string variant;
    int run_index;
    double cpu_time_ms;
    float kernel_time_ms;
    float h2d_time_ms;
    float d2h_time_ms;
    float end_to_end_time_ms;
};

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
    const std::vector<int> dimensions = {128, 256, 512, 1024, 2048, 4096};
    const int total_runs = 10;
    const int warmup_runs = 2;
    const std::string csv_path = "results/benchmark.csv";

    std::cout << "===================================================================\n";
    std::cout << "               CPU vs GPU Pipeline Benchmark Suite                 \n";
    std::cout << "===================================================================\n";
    std::cout << "Configurations: 10 runs per size (first 2 discarded as warm-up)\n";
    std::cout << "Writing raw output to: " << csv_path << "\n\n";

    std::ofstream csv_file(csv_path);
    if (!csv_file.is_open()) {
        std::cerr << "Error: Unable to open " << csv_path << " for writing!\n";
        return 1;
    }

    // Exact CSV Header
    csv_file << "size,variant,run_index,cpu_time_ms,kernel_time_ms,h2d_time_ms,d2h_time_ms,end_to_end_time_ms\n";

    std::vector<BenchmarkRun> all_runs;

    // Test sizes across CPU, GPU Naive, and GPU Shared-Memory
    for (int dim : dimensions) {
        std::cout << ">> Benchmarking image size: " << dim << "x" << dim << " (" << (dim * dim) << " pixels)\n";

        const int w = dim;
        const int h = dim;
        const int c = 3;
        const int total_pixels = w * h;

        auto input_img = generate_benchmark_image(w, h, c);
        std::vector<unsigned char> cpu_out(total_pixels, 0);
        std::vector<unsigned char> gpu_naive_out(total_pixels, 0);
        std::vector<unsigned char> gpu_shared_out(total_pixels, 0);

        std::vector<double> cpu_times;
        std::vector<double> gpu_naive_e2e_times, gpu_naive_kernel_times;
        std::vector<double> gpu_shared_e2e_times, gpu_shared_kernel_times;

        for (int r = 0; r < total_runs; ++r) {
            // 1. CPU Execution
            double cpu_ms = cpu_process_pipeline(input_img.data(), cpu_out.data(), w, h, c);

            // 2. GPU Naive Execution
            GpuTiming naive_timing = gpu_process_pipeline(input_img.data(), gpu_naive_out.data(),
                                                          w, h, c, GpuFilterVariant::NAIVE, dim3(16, 16));

            // 3. GPU Shared-Memory Execution
            GpuTiming shared_timing = gpu_process_pipeline(input_img.data(), gpu_shared_out.data(),
                                                           w, h, c, GpuFilterVariant::SHARED_MEMORY, dim3(16, 16));

            // Write CSV rows (all runs logged)
            csv_file << dim << ",CPU," << r << "," << std::fixed << std::setprecision(4)
                     << cpu_ms << ",0.0000,0.0000,0.0000," << cpu_ms << "\n";

            csv_file << dim << ",GPU_Naive," << r << "," << std::fixed << std::setprecision(4)
                     << cpu_ms << "," << naive_timing.kernel_time_ms << "," << naive_timing.h2d_time_ms << ","
                     << naive_timing.d2h_time_ms << "," << naive_timing.end_to_end_time_ms << "\n";

            csv_file << dim << ",GPU_Shared," << r << "," << std::fixed << std::setprecision(4)
                     << cpu_ms << "," << shared_timing.kernel_time_ms << "," << shared_timing.h2d_time_ms << ","
                     << shared_timing.d2h_time_ms << "," << shared_timing.end_to_end_time_ms << "\n";

            // Record measurements past warm-up for summary statistics
            if (r >= warmup_runs) {
                cpu_times.push_back(cpu_ms);
                gpu_naive_e2e_times.push_back(naive_timing.end_to_end_time_ms);
                gpu_naive_kernel_times.push_back(naive_timing.kernel_time_ms);
                gpu_shared_e2e_times.push_back(shared_timing.end_to_end_time_ms);
                gpu_shared_kernel_times.push_back(shared_timing.kernel_time_ms);
            }
        }

        Stats cpu_s = compute_stats(cpu_times);
        Stats naive_e2e_s = compute_stats(gpu_naive_e2e_times);
        Stats naive_k_s = compute_stats(gpu_naive_kernel_times);
        Stats shared_e2e_s = compute_stats(gpu_shared_e2e_times);
        Stats shared_k_s = compute_stats(gpu_shared_kernel_times);

        std::cout << "   CPU Time:              " << std::setw(8) << cpu_s.mean << " ms (±" << cpu_s.stddev << ")\n";
        std::cout << "   GPU Naive End-to-End:  " << std::setw(8) << naive_e2e_s.mean << " ms (±" << naive_e2e_s.stddev << ") | Kernel: " << naive_k_s.mean << " ms\n";
        std::cout << "   GPU Shared End-to-End: " << std::setw(8) << shared_e2e_s.mean << " ms (±" << shared_e2e_s.stddev << ") | Kernel: " << shared_k_s.mean << " ms\n";
        std::cout << "   Speedup (E2E Shared):  " << std::setw(8) << (cpu_s.mean / shared_e2e_s.mean) << "x\n\n";
    }

    // Block-size sweep at fixed size (2048x2048) comparing 16x16 vs 32x32 block dimensions
    std::cout << ">> Running Block-Size Mini-Sweep on 2048x2048 (16x16 vs 32x32 block)...\n";
    const int sweep_dim = 2048;
    auto sweep_img = generate_benchmark_image(sweep_dim, sweep_dim, 3);
    std::vector<unsigned char> sweep_out(sweep_dim * sweep_dim, 0);

    for (int block_dim_size : {16, 32}) {
        dim3 block(block_dim_size, block_dim_size);
        std::string variant_label = "GPU_Shared_Block_" + std::to_string(block_dim_size);
        std::vector<double> sweep_kernel_times, sweep_e2e_times;

        for (int r = 0; r < total_runs; ++r) {
            GpuTiming timing = gpu_process_pipeline(sweep_img.data(), sweep_out.data(),
                                                    sweep_dim, sweep_dim, 3,
                                                    GpuFilterVariant::SHARED_MEMORY, block);

            csv_file << sweep_dim << "," << variant_label << "," << r << ",0.0000,"
                     << timing.kernel_time_ms << "," << timing.h2d_time_ms << ","
                     << timing.d2h_time_ms << "," << timing.end_to_end_time_ms << "\n";

            if (r >= warmup_runs) {
                sweep_kernel_times.push_back(timing.kernel_time_ms);
                sweep_e2e_times.push_back(timing.end_to_end_time_ms);
            }
        }
        Stats k_stats = compute_stats(sweep_kernel_times);
        Stats e2e_stats = compute_stats(sweep_e2e_times);
        std::cout << "   Block " << block_dim_size << "x" << block_dim_size
                  << " -> Kernel: " << k_stats.mean << " ms (±" << k_stats.stddev << ")"
                  << " | E2E: " << e2e_stats.mean << " ms (±" << e2e_stats.stddev << ")\n";
    }

    csv_file.close();
    std::cout << "\n===================================================================\n";
    std::cout << "Benchmark complete. Results saved to: " << csv_path << "\n";
    std::cout << "===================================================================\n";

    return 0;
}
