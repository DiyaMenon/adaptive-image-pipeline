CXX ?= g++
CXXFLAGS ?= -std=c++17 -O3 -Wall -Wextra -Iinclude -Isrc
NVCC ?= nvcc
NVCCFLAGS ?= -std=c++17 -O3 -Iinclude -Isrc -Xcompiler "-Wall -Wextra"

BIN_DIR = bin
BUILD_DIR = build

.PHONY: all clean cpu_test pipeline tests benchmarks crossover

all: pipeline tests benchmarks

# Directory setup
$(BIN_DIR) $(BUILD_DIR):
	mkdir -p $@

# Standalone CPU test (can be built on macOS / machines without NVCC)
cpu_test: $(BIN_DIR) $(BUILD_DIR) $(BUILD_DIR)/cpu_filters.o
	$(CXX) $(CXXFLAGS) tests/cpu_test_runner.cpp $(BUILD_DIR)/cpu_filters.o -o $(BIN_DIR)/cpu_test

$(BUILD_DIR)/cpu_filters.o: src/cpu_filters.cpp src/cpu_filters.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c src/cpu_filters.cpp -o $@

$(BUILD_DIR)/dispatch.o: src/dispatch.cpp src/dispatch.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c src/dispatch.cpp -o $@

# CUDA Targets (require NVCC and NVIDIA GPU environment)
$(BUILD_DIR)/gpu_filters.o: src/gpu_filters.cu src/gpu_filters.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c src/gpu_filters.cu -o $@

pipeline: $(BIN_DIR) $(BUILD_DIR)/cpu_filters.o $(BUILD_DIR)/dispatch.o $(BUILD_DIR)/gpu_filters.o
	$(NVCC) $(NVCCFLAGS) src/main.cpp $(BUILD_DIR)/cpu_filters.o $(BUILD_DIR)/dispatch.o $(BUILD_DIR)/gpu_filters.o -o $(BIN_DIR)/pipeline

tests: $(BIN_DIR) $(BUILD_DIR)/cpu_filters.o $(BUILD_DIR)/gpu_filters.o
	$(NVCC) $(NVCCFLAGS) tests/correctness_test.cpp $(BUILD_DIR)/cpu_filters.o $(BUILD_DIR)/gpu_filters.o -o $(BIN_DIR)/correctness_test

benchmarks: $(BIN_DIR) $(BUILD_DIR)/cpu_filters.o $(BUILD_DIR)/gpu_filters.o
	$(NVCC) $(NVCCFLAGS) benchmarks/run_benchmarks.cpp $(BUILD_DIR)/cpu_filters.o $(BUILD_DIR)/gpu_filters.o -o $(BIN_DIR)/run_benchmarks

crossover: $(BIN_DIR) $(BUILD_DIR)/cpu_filters.o $(BUILD_DIR)/gpu_filters.o
	$(NVCC) $(NVCCFLAGS) benchmarks/run_crossover.cpp $(BUILD_DIR)/cpu_filters.o $(BUILD_DIR)/gpu_filters.o -o $(BIN_DIR)/run_crossover

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)
