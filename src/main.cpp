#include <iostream>
#include <vector>
#include <string>
#include <iomanip>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "cpu_filters.h"
#include "dispatch.h"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "===================================================================\n";
        std::cout << "             Adaptive CPU/GPU Image Processing Pipeline            \n";
        std::cout << "===================================================================\n";
        std::cout << "Usage: " << argv[0] << " <input_image_path> <output_image_path>\n\n";
        std::cout << "Example: " << argv[0] << " input.jpg output_sobel.png\n";
        return 1;
    }

    const std::string input_path = argv[1];
    const std::string output_path = argv[2];

    int width = 0, height = 0, channels = 0;
    std::cout << "[Pipeline] Loading image: " << input_path << "... ";
    unsigned char* img_data = stbi_load(input_path.c_str(), &width, &height, &channels, 3);
    if (!img_data) {
        std::cerr << "\nError: Failed to load image '" << input_path << "' (" 
                  << stbi_failure_reason() << ")\n";
        return 1;
    }
    channels = 3; // Force 3 channels (RGB)
    std::cout << "Loaded (" << width << "x" << height << ", " << (width * height) << " pixels)\n";

    const int total_pixels = width * height;
    std::vector<unsigned char> output_sobel(total_pixels, 0);

    // Inspect dispatch decision
    ExecutionTarget target = dispatch(width, height);
    std::cout << "[Pipeline] Crossover threshold: " << CROSSOVER_PIXEL_THRESHOLD << " pixels\n";
    std::cout << "[Pipeline] Dispatched Target:  " << execution_target_to_string(target) << "\n";

    // Run adaptive pipeline
    ExecutionTarget used_target;
    double elapsed_ms = run_adaptive_pipeline(img_data, output_sobel.data(),
                                              width, height, channels, &used_target);

    std::cout << "[Pipeline] Execution Time:     " << std::fixed << std::setprecision(4)
              << elapsed_ms << " ms\n";

    // Write output image
    std::cout << "[Pipeline] Saving Sobel output to: " << output_path << "... ";
    int write_ok = stbi_write_png(output_path.c_str(), width, height, 1, output_sobel.data(), width);
    stbi_image_free(img_data);

    if (!write_ok) {
        std::cerr << "FAIL (Error saving image)\n";
        return 1;
    }
    std::cout << "DONE\n";
    std::cout << "===================================================================\n";

    return 0;
}
