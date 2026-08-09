# Adaptive CPU/GPU Image Processing Pipeline

> A lightweight C++17 and CUDA image processing pipeline that empirically benchmarks CPU vs. GPU offload overheads, determines the workload crossover threshold, and dynamically routes image convolution tasks to the optimal hardware target.

---

## Technical Summary

This project implements an end-to-end adaptive image filtering pipeline (Grayscale conversion followed by $3 \times 3$ Sobel edge detection) in C++17 and CUDA C++. It measures where GPU parallel acceleration overcomes fixed offload latencies (allocation, host-to-device transfer, kernel launch, and device-to-host transfer) and automatically dispatches incoming images to either the single-threaded CPU path or the CUDA GPU path based on workload pixel count.

---

## Why This Project Matters (Systems & GPU Engineering)

In high-performance computing and GPU systems engineering, a common misconception is that offloading any parallelizable algorithm to the GPU will automatically yield a speedup. In practice:

- **PCIe Bus Latency:** Data transfers across the PCIe bus (`cudaMemcpy` H2D and D2H) impose fixed latency and bandwidth limits.
- **Resource Management Overheads:** Device memory allocation (`cudaMalloc`), deallocation (`cudaFree`), and CUDA driver/kernel launch queues introduce measurable fixed overheads.
- **Arithmetic Intensity:** Stencil filters like Sobel have relatively low arithmetic intensity ($O(1)$ operations per memory access). For smaller image resolutions, the total compute time is negligible compared to data migration costs.

This project demonstrates the **crossover principle**: profiling real hardware to identify the precise threshold where GPU compute throughput amortizes data movement costs, allowing the runtime to avoid latency regressions on small inputs.

---

## Architecture Overview

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
                          |   dispatch(width, height)   |
                          +-----------------------------+
                                        / \
         Total Pixels < 25,600 px      /   \      Total Pixels >= 25,600 px
         (e.g., 128x128, 144x144)     /     \     (e.g., 160x160, 512x512)
                                     v       v
                  +--------------------+   +------------------------------------+
                  |      CPU Path      |   |              GPU Path              |
                  |  (Single-threaded) |   | 1. cudaMalloc (device memory)      |
                  |  - Grayscale       |   | 2. cudaMemcpy H2D (RGB buffer)     |
                  |  - Sobel filter    |   | 3. Grayscale Kernel (Naive)        |
                  |                    |   | 4. Sobel Kernel (Shared Memory)    |
                  |                    |   | 5. cudaMemcpy D2H (Sobel map)      |
                  |                    |   | 6. cudaFree (cleanup)              |
                  +--------------------+   +------------------------------------+
                                     \       /
                                      \     /
                                       v   v
                          +-----------------------------+
                          | Write Image (stb_image_write|
                          |     & Log Execution Time    |
                          +-----------------------------+
```

---

## GPU Implementation Details

The CUDA implementation consists of two core stages:

### 1. Naive Grayscale Conversion Kernel
- **Thread Mapping:** 1 thread per pixel over a 2D grid of $16 \times 16$ thread blocks.
- **Computation:** Applies fixed-point luminosity arithmetic:
  $$\text{Gray} = \left\lfloor \frac{299 \cdot R + 587 \cdot G + 114 \cdot B + 500}{1000} \right\rfloor$$
- **Memory Access:** Direct global memory reads from the 3-channel interleaved input buffer to a contiguous 1-channel device array.
- *Deliberate Scope Cut:* A shared-memory grayscale kernel is deliberately not implemented because grayscale conversion is a point-wise transformation with zero neighbor data reuse. Loading to shared memory would introduce synchronization barriers (`__syncthreads()`) without reducing global memory bandwidth.

### 2. Shared-Memory-Optimized Sobel Edge Detection Kernel
- **Stencil Operation:** Convolves the 1-channel grayscale image with $3 \times 3$ horizontal ($G_x$) and vertical ($G_y$) Sobel kernels and calculates gradient magnitude $\sqrt{G_x^2 + G_y^2}$.
- **Shared Memory Tiling:**
  - Each $(B_x \times B_y)$ block allocates an extended shared memory tile of dimensions $(B_x + 2) \times (B_y + 2)$ to accommodate a 1-pixel boundary halo.
  - All threads cooperatively populate the tile and halo in a strided loop from global memory.
  - An explicit `__syncthreads()` barrier ensures all tile elements are resident in fast on-chip shared memory before convolution starts.
  - Interior gradient computations read directly from shared memory, eliminating redundant global memory fetches for overlapping $3 \times 3$ stencils.
- **Boundary Handling:** Boundary pixels ($x=0$, $x=W-1$, $y=0$, $y=H-1$) and out-of-bounds halo regions are zero-padded cleanly without divergence or invalid memory access.

---

## Correctness Validation

Before collecting benchmark data, correctness was verified across 8 synthetic test cases by running the CPU reference pipeline, the Naive GPU pipeline, and the Shared-Memory GPU pipeline side-by-side on an **NVIDIA Tesla T4 GPU in Google Colab**.

| Test Case | Dimensions | Naive GPU Result | Shared GPU Result | Max Diff | Mean Diff | Mismatches ($\Delta > 1$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Gradient + Noise | $512 \times 512$ | **PASS** | **PASS** | 0 | 0.0000 | 0 |
| Small Image | $8 \times 8$ | **PASS** | **PASS** | 0 | 0.0000 | 0 |
| Non-Square Image | $300 \times 150$ | **PASS** | **PASS** | 0 | 0.0000 | 0 |
| Non-Block-Multiple | $257 \times 131$ | **PASS** | **PASS** | 0 | 0.0000 | 0 |
| Single Pixel Edge Case | $1 \times 1$ | **PASS** | **PASS** | 0 | 0.0000 | 0 |
| All-Black Image | $256 \times 256$ | **PASS** | **PASS** | 0 | 0.0000 | 0 |
| All-White Image | $256 \times 256$ | **PASS** | **PASS** | 0 | 0.0000 | 0 |
| Full Random Noise | $512 \times 512$ | **PASS** | **PASS** | 0 | 0.0000 | 0 |

> **Result:** All 8 test cases passed with **0 max difference** and **0 mismatches** against the CPU reference baseline.

---

## Benchmark Methodology & Measured Crossover

### Experimental Setup
- **GPU:** NVIDIA Tesla T4 (16 GB GDDR6, Turing Architecture, SM 7.5) on Google Colab
- **Host Compiler:** GCC / G++ (C++17 `-O3`)
- **CUDA Compiler:** `nvcc` (CUDA Toolkit 12.x, `-O3`)
- **Repetitions:** 10 executions per image dimension; the first 2 runs are discarded as warm-up to eliminate initial driver/JIT warm-up latency.
- **Timing Mechanisms:**
  - CPU wall-clock execution: `std::chrono::high_resolution_clock`
  - GPU kernel execution: Synchronized `cudaEvent_t` pairs (`start_kernel` / `stop_kernel`)
  - GPU memory transfers: Isolated `cudaEvent_t` pairs around `cudaMemcpy` (H2D and D2H)
  - GPU End-to-End time: Total elapsed time encompassing `cudaMalloc` + H2D Transfer + Kernel Launches + D2H Transfer + `cudaFree`.

---

### Measured Crossover Data (NVIDIA Tesla T4)

The fine-grained crossover sweep measured square image sizes between $128 \times 128$ and $256 \times 256$ to pinpoint the transition point:

| Image Size | Total Pixels | CPU Pipeline Time (ms) | GPU Shared End-to-End (ms) | Optimal Execution Target |
| :---: | :---: | :---: | :---: | :---: |
| **$128 \times 128$** | 16,384 | **0.3008 ms** | 0.3241 ms | **CPU** |
| **$144 \times 144$** | 20,736 | **0.3348 ms** | 0.5604 ms | **CPU** |
| **$160 \times 160$** | 25,600 | 0.4010 ms | **0.2880 ms** | **GPU (Shared-Memory)** |
| **$256 \times 256$** | 65,536 | 1.0670 ms | **0.3340 ms** | **GPU (Shared-Memory)** |

```
Execution Time (ms)
  0.60 |                    * GPU (144x144 spike)
  0.50 |
  0.40 |               * CPU (160x160: 0.4010 ms)
  0.30 |     * CPU     \   * GPU (160x160: 0.2880 ms) -> CROSSOVER AT 25,600 PX
  0.20 |     * GPU      \_____________________
       +---------------------------------------
           128x128         144x144         160x160
          (16,384 px)     (20,736 px)     (25,600 px)
```

### Analysis
- At **$128 \times 128$** and **$144 \times 144$**, single-threaded CPU execution is faster than the full GPU offload sequence because PCIe data transfer and memory allocation dominate total runtime.
- At **$160 \times 160$ ($25,600\text{ pixels}$)**, the massive parallelism of the GPU Sobel stencil overcomes offload overheads, achieving a lower end-to-end execution time ($0.2880\text{ ms}$ vs. $0.4010\text{ ms}$).
- **Environment Specificity:** The threshold of **$25,600\text{ pixels}$** was measured empirically on an NVIDIA Tesla T4 in Google Colab. On different GPU architectures (e.g., A100, RTX 4090) or systems with unified memory architectures, the crossover point will vary based on memory bandwidth and PCIe generation.

---

## Adaptive Dispatcher

The dispatcher inspects image dimensions at runtime and routes execution:

```cpp
constexpr int CROSSOVER_PIXEL_THRESHOLD = 25600;

ExecutionTarget dispatch(int width, int height) {
    const int pixel_count = width * height;
    if (pixel_count < CROSSOVER_PIXEL_THRESHOLD) {
        return ExecutionTarget::CPU;
    } else {
        return ExecutionTarget::GPU;
    }
}
```

- When `width * height < 25600`, execution runs in-place on the host CPU, avoiding unnecessary device allocations and PCIe transfers.
- When `width * height >= 25600`, execution offloads to the GPU using the optimized shared-memory kernel.

---

## Project Structure

```
adaptive-image-pipeline/
├── Makefile                        # Unified build system (CPU & CUDA targets)
├── README.md                       # Systems documentation and benchmark results
├── .gitignore                      # Ignore build artifacts and generated outputs
├── include/
│   ├── stb_image.h                 # Vendored public-domain image loader
│   └── stb_image_write.h           # Vendored public-domain PNG/image writer
├── src/
│   ├── cpu_filters.h / .cpp        # Single-threaded C++17 Grayscale & Sobel
│   ├── gpu_filters.cuh / .cu       # CUDA Naive & Shared-Memory kernels + event timing
│   ├── dispatch.h / .cpp           # Adaptive workload routing logic (threshold: 25,600 px)
│   └── main.cpp                    # CLI application entrypoint
├── tests/
│   ├── cpu_test_runner.cpp         # Local CPU sanity & dispatcher boundary assertions
│   └── correctness_test.cpp        # 8-case synthetic CPU vs. GPU correctness test suite
├── benchmarks/
│   ├── run_benchmarks.cpp          # Full multi-resolution benchmark harness (128 to 4096)
│   ├── run_crossover.cpp           # Fine-grained crossover sweep harness (128 to 256)
│   └── plot_benchmarks.py          # Visualization script generating analysis plots
└── results/
    ├── .gitkeep                    # Directory tracking for benchmark CSVs and charts
    ├── benchmark.csv               # Raw multi-resolution benchmark metrics
    ├── crossover_benchmark.csv     # Raw fine-grained crossover measurements
    ├── plot1_cpu_vs_gpu_e2e.png          # Chart: CPU vs GPU End-to-End time
    ├── plot2_naive_vs_optimized_kernel.png # Chart: Naive vs Shared-Memory kernel time
    └── plot3_block_size_sweep.png        # Chart: 16x16 vs 32x32 block dimension comparison
```

---

## Reproducibility: How to Build & Run

### 1. Standalone CPU Build (Local Mac / Linux without GPU)
```bash
# Build and run CPU unit tests & dispatcher boundary checks
make cpu_test
./bin/cpu_test
```

### 2. Full CUDA Build & Execution (Google Colab / NVIDIA GPU Host)
```bash
# Clone the repository
git clone https://github.com/DiyaMenon/adaptive-image-pipeline.git
cd adaptive-image-pipeline

# Build all targets (pipeline, tests, benchmarks, crossover)
make all

# 1. Run Correctness Test Suite (Validates all 8 synthetic edge cases)
./bin/correctness_test

# 2. Run Full Multi-Resolution Benchmark Sweep
./bin/run_benchmarks

# 3. Run Fine-Grained Crossover Benchmark Sweep
./bin/run_crossover

# 4. Generate Benchmark Visualizations
python3 benchmarks/plot_benchmarks.py

# 5. Run Live Adaptive Pipeline on an Image
./bin/pipeline input.jpg output_edges.png
```

---

## Example CLI Output

*(Note: The following outputs are illustrative single-run CLI demonstrations; formal benchmark measurements with warm-up discards across repeated runs are reported in the Benchmark section above.)*

```text
===================================================================
             Adaptive CPU/GPU Image Processing Pipeline            
===================================================================
[Pipeline] Loading image: test_image.png... Loaded (512x512, 262144 pixels)
[Pipeline] Crossover threshold: 25600 pixels
[Pipeline] Dispatched Target:  GPU (Shared-Memory Kernel)
[Pipeline] Execution Time:     0.4812 ms
[Pipeline] Saving Sobel output to: output_edges.png... DONE
===================================================================
```

When run on an image below threshold (e.g., $128 \times 128 = 16,384\text{ px}$):
```text
[Pipeline] Loading image: small_icon.png... Loaded (128x128, 16384 pixels)
[Pipeline] Crossover threshold: 25600 pixels
[Pipeline] Dispatched Target:  CPU (Single-threaded)
[Pipeline] Execution Time:     0.3012 ms
[Pipeline] Saving Sobel output to: small_edges.png... DONE
```

---

## Limitations & Future Work

1. **Static Empirical Threshold:** The current threshold ($25,600\text{ pixels}$) is statically configured from measurements on an NVIDIA Tesla T4. Future work could implement an automated startup micro-benchmark to calibrate the threshold dynamically for the host CPU and GPU pairing.
2. **Memory Transfer Optimizations:** The current implementation uses synchronous `cudaMemcpy` to clearly delineate individual transfer costs. Using pinned host memory (`cudaHostAlloc`), asynchronous streams (`cudaStream_t`), and double buffering could hide transfer latency and shift the crossover threshold to smaller workload sizes.
3. **Multi-Architecture Support:** Evaluating the pipeline across heterogeneous platforms (e.g., NVIDIA Jetson embedded GPUs, Apple Metal unified memory architectures, and datacenter GPUs like A100/H100).
4. **Extended Workload Profiles:** Expanding beyond Sobel edge detection to higher-complexity image pipelines (e.g., multi-pass Gaussian filtering, bilateral filtering, or morphological operators).

---

## License & Acknowledgements

- Image I/O is powered by Sean Barrett's public domain [stb](https://github.com/nothings/stb) single-file libraries (`stb_image.h`, `stb_image_write.h`).
- Developed for educational and GPU systems engineering demonstration purposes.
