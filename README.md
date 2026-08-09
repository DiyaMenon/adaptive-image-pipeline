# Adaptive CPU/GPU Image Processing Pipeline

An adaptive C++17/CUDA image processing pipeline that measures workload execution times across CPU and GPU paths to empirically determine the crossover point where GPU acceleration overcomes PCIe transfer overhead, and automatically dispatches workloads to the optimal execution target.

---

## Motivation

GPU acceleration is frequently assumed to be faster for all parallelizable tasks. However, for smaller image sizes, the fixed overheads—PCIe memory allocation (`cudaMalloc`), Host-to-Device transfer (`cudaMemcpy` H2D), kernel launch latency, and Device-to-Host transfer (`cudaMemcpy` D2H)—can significantly exceed the computation time. 

This project explores the fundamental systems question:
> **"At what workload size does GPU acceleration actually become beneficial versus single-threaded CPU execution, and can we automatically dispatch to the faster path?"**

---

## Problem & Architecture

When an image is submitted for processing, the pipeline inspects its dimensions ($W \times H$) and routes execution to either the CPU or the GPU based on an empirically measured crossover threshold.

```
                   +-----------------------------+
                   |         Input Image         |
                   +-----------------------------+
                                  |
                                  v
                   +-----------------------------+
                   |   Load Image (stb_image)    |
                   +-----------------------------+
                                  |
                                  v
                   +-----------------------------+
                   |     dispatch(width, height) |
                   +-----------------------------+
                                 / \
               Total Pixels < T /   \ Total Pixels >= T
                               /     \
                              v       v
           +--------------------+   +------------------------------------+
           |      CPU Path      |   |              GPU Path              |
           |  (Single-threaded) |   | 1. Allocate device memory          |
           |  - Grayscale       |   | 2. cudaMemcpy (H2D)                |
           |  - Sobel filter    |   | 3. Launch CUDA kernel              |
           |                    |   |    (Naive / Shared Memory Sobel)   |
           |                    |   | 4. cudaMemcpy (D2H)                |
           |                    |   | 5. Free device memory              |
           +--------------------+   +------------------------------------+
                              \       /
                               \     /
                                v   v
                   +-----------------------------+
                   | Write Image (stb_image_write|
                   |      & Log Timings to CSV   |
                   +-----------------------------+
```

---

## Implementation Details

### 1. CPU Baseline (C++17)
- **Grayscale Conversion:** Single-threaded luminosity calculation ($Y = 0.299R + 0.587G + 0.114B$) iterating row-by-row with `std::chrono::high_resolution_clock` timing.
- **Sobel Edge Detection:** $3 \times 3$ convolution using standard horizontal ($G_x$) and vertical ($G_y$) kernels, computing gradient magnitude $\sqrt{G_x^2 + G_y^2}$ clamped to $[0, 255]$.

### 2. CUDA Kernels
- **Naive Grayscale Kernel:** 1 thread per pixel, 2D grid/block layout, direct global memory reads and writes.
- **Naive Sobel Kernel:** 1 thread per pixel, reading $3 \times 3$ neighbor window directly from global memory.
- **Shared-Memory-Optimized Sobel Kernel:** Cooperatively loads a $(B_x + 2) \times (B_y + 2)$ tile containing a 1-pixel halo into `__shared__` memory per thread block, synchronizes with `__syncthreads()`, and computes gradients using cached local values, eliminating redundant global memory fetches.
- *Note on Grayscale Optimization:* A shared-memory grayscale kernel is deliberately **not** implemented because grayscale is a point-wise transformation with zero neighbor data reuse; caching in shared memory would add unnecessary synchronization overhead without reduction in global memory bandwidth.

---

## Benchmark Methodology

- **Image Dimensions:** $128 \times 128$, $256 \times 256$, $512 \times 512$, $1024 \times 1024$, $2048 \times 2048$, and $4096 \times 4096$.
- **Repetitions:** 10 runs per size per variant; the first 2 runs are discarded as warm-up.
- **Timing Metrics:**
  - CPU execution time (`std::chrono`)
  - GPU kernel execution time (isolated via `cudaEvent_t` with `cudaEventSynchronize`)
  - Host-to-Device transfer time (`cudaMemcpy` H2D)
  - Device-to-Host transfer time (`cudaMemcpy` D2H)
  - GPU End-to-End time (H2D + Kernel + D2H + memory management overhead)
- **Output:** Raw timing data exported to `results/benchmark.csv` and summarized with mean and standard deviation.

---

## Results & Crossover Point

*(Note: Real benchmark numbers and generated plots will be populated after executing the benchmark suite on an NVIDIA GPU environment via Google Colab. No synthetic or estimated values are used.)*

- **Target GPU:** *[To be populated upon Colab execution]*
- **CUDA Toolkit Version:** *[To be populated upon Colab execution]*
- **Measured Crossover Threshold:** *[To be derived from measured benchmark.csv]*

---

## Limitations

1. **Single-threaded CPU Baseline:** The CPU implementation is strictly single-threaded to isolate baseline algorithm behavior without multi-threading library overhead (e.g., OpenMP).
2. **Fixed Static Threshold:** The dispatcher uses a compile-time/configured empirical threshold calibrated to specific hardware rather than dynamic run-time runtime adaptation.
3. **No Streams/Overlap:** CUDA execution in the MVP uses synchronous memory copies to clearly delineate individual transfer costs.

---

## Two-Environment Workflow

Due to development on an Apple Silicon (M2) machine with no local CUDA support, the development and execution workflows are strictly separated:

1. **Local Development (Apple M2):**
   - Code writing, repository structure, C++ CPU implementations, and non-GPU test harnesses.
2. **Remote Execution (Google Colab / NVIDIA GPU):**
   - CUDA compilation (`nvcc`), correctness testing against real GPU hardware, benchmark execution, CSV data collection, and plot generation.

### How to Run

#### Standalone CPU Test (Local)
```bash
make cpu_test
./bin/cpu_test
```

#### Full Build & Execution (NVIDIA GPU / Colab)
```bash
# Clone the repository
git clone https://github.com/DiyaMenon/adaptive-image-pipeline.git
cd adaptive-image-pipeline

# Build all targets
make all

# Run correctness test suite
./bin/correctness_test

# Run benchmark suite
./bin/run_benchmarks

# Run end-to-end adaptive pipeline
./bin/pipeline <input_image.png> <output_image.png>
```
