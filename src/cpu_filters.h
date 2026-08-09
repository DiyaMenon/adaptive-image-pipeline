#pragma once

#include <cstddef>

// Convert an RGB/RGBA image to single-channel 8-bit grayscale on the CPU.
// Uses standard luminosity formula: Y = 0.299*R + 0.587*G + 0.114*B
void cpu_rgb_to_grayscale(const unsigned char* input,
                          unsigned char* output,
                          int width,
                          int height,
                          int channels);

// Apply 3x3 Sobel edge detection to a grayscale image on the CPU.
// Border pixels (x=0, x=w-1, y=0, y=h-1) are set to 0.
void cpu_sobel_filter(const unsigned char* input_gray,
                      unsigned char* output_sobel,
                      int width,
                      int height);

// Run the full CPU pipeline (Grayscale + Sobel) and return execution time in milliseconds.
double cpu_process_pipeline(const unsigned char* input_rgb,
                            unsigned char* output_sobel,
                            int width,
                            int height,
                            int channels);
