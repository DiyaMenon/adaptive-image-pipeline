#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <iomanip>
#include <random>
#include <algorithm>

#include "cpu_filters.h"
#include "gpu_filters.cuh"

// Struct representing a synthetic test case
struct TestCase {
    std::string name;
    int width;
    int height;
    int channels;
    std::vector<unsigned char> data;
};

// ============================================================================
// Synthetic Image Generators
// ============================================================================

static std::vector<unsigned char> generate_flat(int w, int h, int c, unsigned char value) {
    return std::vector<unsigned char>(w * h * c, value);
}

static std::vector<unsigned char> generate_random(int w, int h, int c, unsigned int seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<unsigned char> data(w * h * c);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<unsigned char>(dist(rng));
    }
    return data;
}

static std::vector<unsigned char> generate_gradient_noise(int w, int h, int c, unsigned int seed = 123) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> noise_dist(-15, 15);
    std::vector<unsigned char> data(w * h * c);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int grad = (x * 255) / (w > 1 ? w - 1 : 1);
            int idx = (y * w + x) * c;
            for (int ch = 0; ch < c; ++ch) {
                int val = grad + noise_dist(rng);
                data[idx + ch] = static_cast<unsigned char>(std::clamp(val, 0, 255));
            }
        }
    }
    return data;
}

static std::vector<TestCase> create_test_suite() {
    std::vector<TestCase> suite;

    // 1. Normal gradient + noise (512x512)
    suite.push_back({"Gradient + Noise (512x512)", 512, 512, 3, generate_gradient_noise(512, 512, 3)});

    // 2. Small 8x8 image
    suite.push_back({"Small Image (8x8)", 8, 8, 3, generate_random(8, 8, 3, 101)});

    // 3. Non-square image (300x150)
    suite.push_back({"Non-Square Image (300x150)", 300, 150, 3, generate_gradient_noise(300, 150, 3)});

    // 4. Non-multiple of 16x16 block size (257x131)
    suite.push_back({"Non-Block-Multiple (257x131)", 257, 131, 3, generate_random(257, 131, 3, 202)});

    // 5. Tiny 1x1 image edge case
    suite.push_back({"Single Pixel (1x1)", 1, 1, 3, {128, 64, 32}});

    // 6. All-black image (256x256)
    suite.push_back({"All-Black (256x256)", 256, 256, 3, generate_flat(256, 256, 3, 0)});

    // 7. All-white image (256x256)
    suite.push_back({"All-White (256x256)", 256, 256, 3, generate_flat(256, 256, 3, 255)});

    // 8. Full random noise (512x512)
    suite.push_back({"Random Noise (512x512)", 512, 512, 3, generate_random(512, 512, 3, 999)});

    return suite;
}

// ============================================================================
// Verification and Comparison
// ============================================================================

struct ComparisonResult {
    bool passed;
    int max_diff;
    double mean_diff;
    int mismatch_count;
};

static ComparisonResult compare_outputs(const unsigned char* cpu_buf,
                                        const unsigned char* gpu_buf,
                                        int total_pixels,
                                        int tolerance = 1) {
    int max_d = 0;
    long long total_d = 0;
    int mismatches = 0;

    for (int i = 0; i < total_pixels; ++i) {
        int diff = std::abs(static_cast<int>(cpu_buf[i]) - static_cast<int>(gpu_buf[i]));
        if (diff > max_d) {
            max_d = diff;
        }
        total_d += diff;
        if (diff > tolerance) {
            mismatches++;
        }
    }

    ComparisonResult res;
    res.max_diff = max_d;
    res.mean_diff = total_pixels > 0 ? static_cast<double>(total_d) / total_pixels : 0.0;
    res.mismatch_count = mismatches;
    res.passed = (mismatches == 0);
    return res;
}

int main() {
    std::cout << "===================================================================\n";
    std::cout << "             CUDA Image Pipeline Correctness Test Suite            \n";
    std::cout << "===================================================================\n";
    std::cout << "Tolerance: abs(cpu[i] - gpu[i]) <= 1 (on 0-255 scale)\n\n";

    auto test_cases = create_test_suite();
    bool all_passed = true;

    for (const auto& tc : test_cases) {
        const int total_pixels = tc.width * tc.height;
        std::vector<unsigned char> cpu_output(total_pixels, 0);
        std::vector<unsigned char> gpu_naive_output(total_pixels, 0);
        std::vector<unsigned char> gpu_shared_output(total_pixels, 0);

        // 1. Run CPU Reference Pipeline
        cpu_process_pipeline(tc.data.data(), cpu_output.data(), tc.width, tc.height, tc.channels);

        // 2. Run GPU Naive Pipeline
        gpu_process_pipeline(tc.data.data(), gpu_naive_output.data(),
                             tc.width, tc.height, tc.channels,
                             GpuFilterVariant::NAIVE);

        // 3. Run GPU Shared-Memory Pipeline
        gpu_process_pipeline(tc.data.data(), gpu_shared_output.data(),
                             tc.width, tc.height, tc.channels,
                             GpuFilterVariant::SHARED_MEMORY);

        // Compare CPU vs GPU Naive
        ComparisonResult res_naive = compare_outputs(cpu_output.data(), gpu_naive_output.data(), total_pixels);

        // Compare CPU vs GPU Shared Memory
        ComparisonResult res_shared = compare_outputs(cpu_output.data(), gpu_shared_output.data(), total_pixels);

        bool case_passed = res_naive.passed && res_shared.passed;
        if (!case_passed) {
            all_passed = false;
        }

        std::cout << "Test Case: " << tc.name << " (" << tc.width << "x" << tc.height << ")\n";
        std::cout << "  - Naive GPU Kernel:   "
                  << (res_naive.passed ? "[PASS]" : "[FAIL]")
                  << " | Max Diff: " << res_naive.max_diff
                  << " | Mean Diff: " << std::fixed << std::setprecision(4) << res_naive.mean_diff
                  << " | Mismatches: " << res_naive.mismatch_count << "\n";
        std::cout << "  - Shared Mem Kernel:  "
                  << (res_shared.passed ? "[PASS]" : "[FAIL]")
                  << " | Max Diff: " << res_shared.max_diff
                  << " | Mean Diff: " << std::fixed << std::setprecision(4) << res_shared.mean_diff
                  << " | Mismatches: " << res_shared.mismatch_count << "\n";
        std::cout << "-------------------------------------------------------------------\n";
    }

    std::cout << "\n===================================================================\n";
    if (all_passed) {
        std::cout << " >>> ALL CORRECTNESS TESTS PASSED (CPU vs Naive & Shared GPU) <<< \n";
        std::cout << "===================================================================\n";
        return 0;
    } else {
        std::cerr << " >>> SOME CORRECTNESS TESTS FAILED - DO NOT TRUST BENCHMARKS <<< \n";
        std::cout << "===================================================================\n";
        return 1;
    }
}
