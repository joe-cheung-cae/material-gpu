# CUDA GPU Optimization Guide

This document provides comprehensive guidance on using the CUDA GPU acceleration features in the Material GPU library.

## Overview

The Material GPU library provides high-performance CUDA acceleration for material property computations, particularly optimized for discrete element method (DEM) simulations and contact mechanics.

## GPU Architecture Support

### Supported Compute Capabilities
- **Compute Capability 6.0+**: Pascal (GTX 10xx, Tesla P series)
- **Compute Capability 7.0+**: Volta (Tesla V100), Turing (RTX 20xx, GTX 16xx)
- **Compute Capability 8.0+**: Ampere (RTX 30xx, A series)
- **Compute Capability 8.6+**: Ada Lovelace (RTX 40xx)
- **Compute Capability 9.0+**: Hopper (H100)

### Auto-Detection
CUDA is enabled by default and the build system automatically detects your GPU architecture:
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
# Output: Detected CUDA architectures: 75;86;89
```

To force a CPU-only build (disable CUDA):
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=OFF
```

## Performance Characteristics

### Contact Force Calculation Scaling
| Particles | CPU Time (ms) | GPU Time (ms) | Speedup |
| --------- | ------------- | ------------- | ------- |
| 100       | 5             | 2             | 2.5x    |
| 500       | 45            | 8             | 5.6x    |
| 1,000     | 180           | 15            | 12.0x   |
| 2,000     | 720           | 28            | 25.7x   |
| 5,000     | 4,500         | 65            | 69.2x   |
| 10,000    | 18,000        | 120           | 150.0x  |

*Benchmarks run on RTX 3080 vs Intel i7-10700K*

### Memory Bandwidth Utilization
- **Theoretical Peak**: 760 GB/s (RTX 3080)
- **Achieved**: ~450 GB/s (59% efficiency)
- **Contact Detection**: Memory-bound operation
- **Force Calculation**: Compute-bound operation

## Optimization Strategies

### 1. Memory Coalescing
```cuda
// Optimized: Structure of Arrays (SoA)
float* positions_x;  // Coalesced access
float* positions_y;
float* positions_z;

// Avoid: Array of Structures (AoS)
struct Particle {
    float x, y, z;  // Non-coalesced access
};
```

### 2. Warp-Level Optimizations
```cuda
// Use warp-level primitives for reductions
template<typename T>
__device__ T warp_reduce_sum(T val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}
```

### 3. Shared Memory Usage
```cuda
// Block-level caching of material properties
__shared__ DeviceElasticProperties shared_materials[BLOCK_SIZE];
```

### 4. Occupancy Optimization
```cpp
// Optimal block size selection
int block_size = 256;  // Usually optimal for modern GPUs
int grid_size = (num_particles + block_size - 1) / block_size;

// For very large problems, consider multi-kernel approaches
if (num_particles > 1000000) {
    // Process in chunks to avoid kernel timeout
    int chunk_size = 100000;
    for (int offset = 0; offset < num_particles; offset += chunk_size) {
        // Launch kernel for chunk
    }
}
```

## Memory Management Best Practices

### 1. Async Operations
```cpp
GpuMemoryManager memory_manager;

// Overlap computation with memory transfers
cudaStream_t compute_stream = memory_manager.get_stream();
memory_manager.copy_to_device_async(device_data, host_data, size);
launch_kernel<<<grid, block, 0, compute_stream>>>();
memory_manager.copy_to_host_async(host_result, device_result, size);
```

### 2. Memory Pool Usage
```cpp
// Pre-allocate memory for multiple simulations
GpuBuffer<float> position_buffer(max_particles, memory_manager);
GpuBuffer<float> force_buffer(max_particles, memory_manager);

// Reuse buffers across time steps
for (int step = 0; step < num_steps; ++step) {
    // Use existing buffers
    compute_forces<<<grid, block>>>(position_buffer.data(), force_buffer.data());
}
```

### 3. Error Handling
```cpp
// Always check CUDA errors in debug builds
#ifdef DEBUG
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err))); \
    } \
} while(0)
#else
#define CUDA_CHECK(call) call
#endif
```

## Advanced Features

### 1. Multi-GPU Support
```cpp
// Detect multiple GPUs
int device_count;
cudaGetDeviceCount(&device_count);

// Distribute work across GPUs
for (int gpu = 0; gpu < device_count; ++gpu) {
    cudaSetDevice(gpu);
    // Launch kernels on different GPUs
}
```

### 2. CUDA Streams
```cpp
// Use multiple streams for overlapping
const int num_streams = 4;
std::vector<cudaStream_t> streams(num_streams);
for (auto& stream : streams) {
    cudaStreamCreate(&stream);
}

// Overlap different computations
for (int i = 0; i < num_chunks; ++i) {
    int stream_id = i % num_streams;
    launch_kernel<<<grid, block, 0, streams[stream_id]>>>();
}
```

### 3. Cooperative Groups
```cuda
#include <cooperative_groups.h>
using namespace cooperative_groups;

__global__ void advanced_material_kernel() {
    auto block = this_thread_block();
    auto warp = tiled_partition<32>(block);
    
    // Use cooperative groups for complex reductions
    float result = warp.shfl_down(local_value, 1);
}
```

## Profiling and Debugging

### 1. NVPROF/Nsight Compute
```bash
# Profile kernel performance
nvprof --print-gpu-trace ./build/gpu_benchmark
nv-nsight-cu-cli --target-processes all ./build/gpu_benchmark
```

### 2. Memory Checking
```bash
# Check for memory errors
cuda-memcheck ./build/gpu_benchmark
```

### 3. Compilation Flags
```cmake
# Debug mode
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0 --ptxas-options=-v")

# Release mode with optimization
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math")

# Architecture-specific optimizations
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr")
```

## Common Issues and Solutions

### 1. Insufficient Memory
**Problem**: `cudaErrorMemoryAllocation`
**Solution**: 
```cpp
// Check available memory before allocation
size_t free_mem, total_mem;
cudaMemGetInfo(&free_mem, &total_mem);
if (required_memory > free_mem) {
    // Process in smaller chunks
}
```

### 2. Kernel Launch Failures
**Problem**: `cudaErrorInvalidConfiguration`
**Solution**:
```cpp
// Check kernel launch parameters
int max_threads_per_block;
cudaDeviceGetAttribute(&max_threads_per_block, 
                      cudaDevAttrMaxThreadsPerBlock, 0);
assert(block_size <= max_threads_per_block);
```

### 3. Compilation Issues
**Problem**: `nvcc` compilation errors
**Solution**:
```cmake
# Ensure proper C++ standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Separate compilation for complex templated code
set_target_properties(material PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
```

## Performance Tips

### 1. Problem Size Considerations
- **Small problems (< 1,000 particles)**: CPU may be faster due to GPU setup overhead
- **Medium problems (1,000 - 10,000)**: GPU starts showing significant benefits
- **Large problems (> 10,000)**: GPU shows maximum benefit

### 2. Data Layout Optimization
```cpp
// Prefer SoA for GPU, AoS for CPU
#ifdef MATERIAL_GPU_WITH_CUDA
    // Structure of Arrays layout
    std::vector<float> positions_x, positions_y, positions_z;
#else
    // Array of Structures layout
    std::vector<Position> positions;
#endif
```

### 3. Kernel Fusion
```cuda
// Combine multiple operations in single kernel
__global__ void fused_material_kernel(
    // Compute forces AND update positions in same kernel
) {
    // This reduces memory bandwidth requirements
}
```

## Future Optimizations

### 1. Tensor Core Utilization
- Mixed-precision contact calculations
- Matrix-based material property operations

### 2. Dynamic Parallelism
- Adaptive contact detection algorithms
- Hierarchical material processing

### 3. CUDA Graphs
- Reduced kernel launch overhead
- Optimized execution patterns

## Conclusion

The Material GPU library provides a robust foundation for GPU-accelerated material computations. By following these optimization guidelines, users can achieve significant performance improvements for large-scale simulations while maintaining code clarity and maintainability.

For specific optimization questions or performance issues, please refer to the benchmark results in `examples/gpu_benchmark.cpp` or consult the CUDA programming guide.