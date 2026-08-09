#!/usr/bin/env python3
"""
Benchmark Plotting Script
Reads results/benchmark.csv and generates the three required visualization plots:
1. CPU vs GPU End-to-End Execution Time vs Image Dimension
2. Naive GPU vs Shared-Memory GPU Kernel Execution Time vs Image Dimension
3. Block-Size Sweep Comparison (16x16 vs 32x32) at 2048x2048
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

def generate_plots():
    csv_path = "results/benchmark.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run ./bin/run_benchmarks first.")
        return

    df = pd.read_csv(csv_path)

    # Discard warmup runs (run_index >= 2)
    df_filtered = df[df["run_index"] >= 2]

    os.makedirs("results", exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # -------------------------------------------------------------------------
    # Plot 1: CPU vs GPU End-to-End Time vs Dimension
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 6), dpi=300)
    cpu_data = df_filtered[df_filtered["variant"] == "CPU"].groupby("size")["end_to_end_time_ms"].mean()
    gpu_shared_data = df_filtered[df_filtered["variant"] == "GPU_Shared"].groupby("size")["end_to_end_time_ms"].mean()
    gpu_naive_data = df_filtered[df_filtered["variant"] == "GPU_Naive"].groupby("size")["end_to_end_time_ms"].mean()

    sizes = cpu_data.index.tolist()
    labels = [f"{s}x{s}" for s in sizes]

    plt.plot(labels, cpu_data.values, marker="o", color="#d9534f", linewidth=2, label="CPU (Single-threaded)")
    plt.plot(labels, gpu_naive_data.values, marker="s", color="#f0ad4e", linewidth=2, linestyle="--", label="GPU Naive (End-to-End)")
    plt.plot(labels, gpu_shared_data.values, marker="^", color="#5cb85c", linewidth=2.5, label="GPU Shared-Memory (End-to-End)")

    plt.title("Execution Time vs Image Dimension: CPU vs GPU End-to-End", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Image Dimensions", fontsize=12, labelpad=10)
    plt.ylabel("Time (ms) - Log Scale", fontsize=12, labelpad=10)
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plot1_path = "results/plot1_cpu_vs_gpu_e2e.png"
    plt.savefig(plot1_path)
    plt.close()
    print(f"Saved: {plot1_path}")

    # -------------------------------------------------------------------------
    # Plot 2: Naive vs Shared Memory Kernel Time vs Dimension
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 6), dpi=300)
    k_naive = df_filtered[df_filtered["variant"] == "GPU_Naive"].groupby("size")["kernel_time_ms"].mean()
    k_shared = df_filtered[df_filtered["variant"] == "GPU_Shared"].groupby("size")["kernel_time_ms"].mean()

    plt.plot(labels, k_naive.values, marker="s", color="#e67e22", linewidth=2, label="Naive GPU Kernel (Global Memory)")
    plt.plot(labels, k_shared.values, marker="^", color="#27ae60", linewidth=2.5, label="Optimized GPU Kernel (Shared Memory)")

    plt.title("Kernel Execution Time: Naive vs Shared-Memory Sobel", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Image Dimensions", fontsize=12, labelpad=10)
    plt.ylabel("Kernel Time (ms)", fontsize=12, labelpad=10)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plot2_path = "results/plot2_naive_vs_optimized_kernel.png"
    plt.savefig(plot2_path)
    plt.close()
    print(f"Saved: {plot2_path}")

    # -------------------------------------------------------------------------
    # Plot 3: Block-Size Sweep Bar Chart (16x16 vs 32x32 at 2048x2048)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 6), dpi=300)
    b16 = df_filtered[df_filtered["variant"] == "GPU_Shared_Block_16"]["kernel_time_ms"].mean()
    b32 = df_filtered[df_filtered["variant"] == "GPU_Shared_Block_32"]["kernel_time_ms"].mean()

    bars = plt.bar(["16x16 Block (256 threads)", "32x32 Block (1024 threads)"],
                   [b16, b32], color=["#3498db", "#9b59b6"], width=0.5)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + (yval * 0.02), f"{yval:.3f} ms",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.title("Kernel Execution Time by Thread Block Dimension (2048x2048)", fontsize=13, fontweight="bold", pad=15)
    plt.ylabel("Kernel Time (ms)", fontsize=12, labelpad=10)
    plt.grid(axis="y", ls="--", alpha=0.5)
    plt.tight_layout()
    plot3_path = "results/plot3_block_size_sweep.png"
    plt.savefig(plot3_path)
    plt.close()
    print(f"Saved: {plot3_path}")

if __name__ == "__main__":
    generate_plots()
