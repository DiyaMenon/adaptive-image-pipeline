#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <cstring>
#include "cpu_filters.h"

static bool test_grayscale_known_values() {
    std::cout << "[Test] CPU Grayscale known colors... ";
    // RGB colors: Red, Green, Blue, White, Black
    std::vector<unsigned char> rgb = {
        255,   0,   0, // Red   -> 0.299 * 255 = 76.245 -> 76
          0, 255,   0, // Green -> 0.587 * 255 = 149.685 -> 149
          0,   0, 255, // Blue  -> 0.114 * 255 = 29.07 -> 29
        255, 255, 255, // White -> 255
          0,   0,   0  // Black -> 0
    };
    std::vector<unsigned char> gray(5);
    cpu_rgb_to_grayscale(rgb.data(), gray.data(), 5, 1, 3);

    assert(gray[0] == 76);
    assert(gray[1] == 149);
    assert(gray[2] == 29);
    assert(gray[3] == 255);
    assert(gray[4] == 0);
    std::cout << "PASSED\n";
    return true;
}

static bool test_sobel_flat_image() {
    std::cout << "[Test] CPU Sobel flat constant image (512x512)... ";
    const int w = 512, h = 512;
    std::vector<unsigned char> gray(w * h, 180); // uniform gray
    std::vector<unsigned char> sobel(w * h, 255);

    cpu_sobel_filter(gray.data(), sobel.data(), w, h);

    for (int i = 0; i < w * h; ++i) {
        if (sobel[i] != 0) {
            std::cerr << "FAIL: expected 0 on flat region at index " << i 
                      << ", got " << static_cast<int>(sobel[i]) << "\n";
            return false;
        }
    }
    std::cout << "PASSED\n";
    return true;
}

static bool test_sobel_vertical_edge() {
    std::cout << "[Test] CPU Sobel vertical step edge... ";
    const int w = 16, h = 16;
    std::vector<unsigned char> gray(w * h, 0);
    // Left half 0, right half 255
    for (int y = 0; y < h; ++y) {
        for (int x = w / 2; x < w; ++x) {
            gray[y * w + x] = 255;
        }
    }
    std::vector<unsigned char> sobel(w * h, 0);
    cpu_sobel_filter(gray.data(), sobel.data(), w, h);

    // Border pixels must be 0
    for (int x = 0; x < w; ++x) {
        assert(sobel[x] == 0);
        assert(sobel[(h - 1) * w + x] == 0);
    }
    // Vertical edge at column w/2 - 1 should have strong response
    for (int y = 1; y < h - 1; ++y) {
        assert(sobel[y * w + (w / 2 - 1)] > 0);
    }
    std::cout << "PASSED\n";
    return true;
}

static bool test_edge_cases() {
    std::cout << "[Test] CPU Sobel edge cases (1x1, 2x2, 257x131)... ";
    // 1x1
    std::vector<unsigned char> img1(1, 100);
    std::vector<unsigned char> out1(1, 99);
    cpu_sobel_filter(img1.data(), out1.data(), 1, 1);
    assert(out1[0] == 0);

    // 2x2
    std::vector<unsigned char> img2(4, 100);
    std::vector<unsigned char> out2(4, 99);
    cpu_sobel_filter(img2.data(), out2.data(), 2, 2);
    for (int i = 0; i < 4; ++i) assert(out2[i] == 0);

    // 257x131
    std::vector<unsigned char> img3(257 * 131, 50);
    std::vector<unsigned char> out3(257 * 131, 99);
    cpu_sobel_filter(img3.data(), out3.data(), 257, 131);
    for (int i = 0; i < 257 * 131; ++i) assert(out3[i] == 0);

    std::cout << "PASSED\n";
    return true;
}

static bool test_pipeline_timing() {
    std::cout << "[Test] CPU Pipeline execution and timing... ";
    const int w = 512, h = 512;
    std::vector<unsigned char> rgb(w * h * 3, 128);
    std::vector<unsigned char> out(w * h, 0);

    const double elapsed_ms = cpu_process_pipeline(rgb.data(), out.data(), w, h, 3);
    assert(elapsed_ms >= 0.0);
    std::cout << "PASSED (" << elapsed_ms << " ms for 512x512)\n";
    return true;
}

int main() {
    std::cout << "=== Running CPU Filter Sanity Checks ===\n";
    bool ok = true;
    ok &= test_grayscale_known_values();
    ok &= test_sobel_flat_image();
    ok &= test_sobel_vertical_edge();
    ok &= test_edge_cases();
    ok &= test_pipeline_timing();

    if (ok) {
        std::cout << "=== All CPU Filter Sanity Checks Passed! ===\n";
        return 0;
    } else {
        std::cerr << "=== CPU Filter Checks Failed ===\n";
        return 1;
    }
}
