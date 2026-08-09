#include "cpu_filters.h"

#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>

void cpu_rgb_to_grayscale(const unsigned char* input,
                          unsigned char* output,
                          int width,
                          int height,
                          int channels) {
    if (!input || !output || width <= 0 || height <= 0 || channels <= 0) {
        return;
    }

    const int total_pixels = width * height;
    for (int i = 0; i < total_pixels; ++i) {
        const int src_idx = i * channels;
        if (channels >= 3) {
            const int r = static_cast<int>(input[src_idx]);
            const int g = static_cast<int>(input[src_idx + 1]);
            const int b = static_cast<int>(input[src_idx + 2]);
            const int gray = (299 * r + 587 * g + 114 * b + 500) / 1000;
            output[i] = static_cast<unsigned char>(std::clamp(gray, 0, 255));
        } else {
            output[i] = input[src_idx];
        }
    }
}

void cpu_sobel_filter(const unsigned char* input_gray,
                      unsigned char* output_sobel,
                      int width,
                      int height) {
    if (!input_gray || !output_sobel || width <= 0 || height <= 0) {
        return;
    }

    // Handle small images where no interior pixels exist
    if (width < 3 || height < 3) {
        std::fill(output_sobel, output_sobel + (width * height), 0);
        return;
    }

    // Zero out the top and bottom boundary rows
    for (int x = 0; x < width; ++x) {
        output_sobel[x] = 0;                             // Row 0
        output_sobel[(height - 1) * width + x] = 0;     // Row height - 1
    }

    // Process interior rows
    for (int y = 1; y < height - 1; ++y) {
        // Zero out left and right boundary pixels for this row
        output_sobel[y * width] = 0;
        output_sobel[y * width + (width - 1)] = 0;

        for (int x = 1; x < width - 1; ++x) {
            // Read 3x3 neighborhood
            const int p00 = input_gray[(y - 1) * width + (x - 1)];
            const int p01 = input_gray[(y - 1) * width + x];
            const int p02 = input_gray[(y - 1) * width + (x + 1)];

            const int p10 = input_gray[y * width + (x - 1)];
            // p11 is center pixel (weight is 0 in both kernels)
            const int p12 = input_gray[y * width + (x + 1)];

            const int p20 = input_gray[(y + 1) * width + (x - 1)];
            const int p21 = input_gray[(y + 1) * width + x];
            const int p22 = input_gray[(y + 1) * width + (x + 1)];

            // Horizontal gradient (Gx)
            const int gx = -p00 + p02
                           - 2 * p10 + 2 * p12
                           - p20 + p22;

            // Vertical gradient (Gy)
            const int gy = -p00 - 2 * p01 - p02
                           + p20 + 2 * p21 + p22;

            // Gradient magnitude
            const float mag = std::sqrt(static_cast<float>(gx * gx + gy * gy));
            output_sobel[y * width + x] = static_cast<unsigned char>(std::min(255.0f, mag));
        }
    }
}

double cpu_process_pipeline(const unsigned char* input_rgb,
                            unsigned char* output_sobel,
                            int width,
                            int height,
                            int channels) {
    std::vector<unsigned char> temp_gray(width * height);

    const auto start = std::chrono::high_resolution_clock::now();
    cpu_rgb_to_grayscale(input_rgb, temp_gray.data(), width, height, channels);
    cpu_sobel_filter(temp_gray.data(), output_sobel, width, height);
    const auto end = std::chrono::high_resolution_clock::now();

    const std::chrono::duration<double, std::milli> duration = end - start;
    return duration.count();
}
