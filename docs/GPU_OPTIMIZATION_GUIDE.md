# GPU Optimization Guide for Material GPU Library

## Table of Contents
1. [Overview](#overview)
2. [Hardware Requirements](#hardware-requirements)
3. [Performance Optimization Strategies](#performance-optimization-strategies)
4. [Kernel Selection Guide](#kernel-selection-guide)
5. [Memory Management Best Practices](#memory-management-best-practices)
6. [Multi-GPU Configuration](#multi-gpu-configuration)
7. [Performance Benchmarking](#performance-benchmarking)
8. [Troubleshooting](#troubleshooting)

## Overview

The Material GPU library provides multiple levels of GPU acceleration optimized for different use cases and hardware configurations. This guide helps you achieve optimal performance for your specific simulation requirements.

### Performance Hierarchy

```
CPU Implementation          →  1x baseline performance
Basic GPU Kernel           →  10-50x speedup
Optimized GPU Kernel       →  50-150x speedup
Multi-GPU Implementation   →  100-500x speedup
CUDA Graphs + Multi-GPU    →  200-1000x speedup
```

## Hardware Requirements

### Minimum Requirements
- **GPU**: NVIDIA GPU with Compute Capability 6.0+ (Pascal architecture)
- **CUDA**: Version 10.0 or higher
- **Memory**: 2GB GPU memory minimum
- **Driver**: NVIDIA driver 440.33 or higher

### Recommended Configuration
- **GPU**: RTX 4080/4090, A100, H100 or equivalent
- **CUDA**: Version 12.0 or higher
- **Memory**: 8GB+ GPU memory
- **System**: PCIe 4.0 x16 for optimal memory bandwidth

### Compute Capability Support

| Architecture | Compute Capability | Optimizations Available                     |
| ------------ | ------------------ | ------------------------------------------- |
| Pascal       | 6.0-6.2            | Basic kernels, shared memory                |
| Volta        | 7.0-7.5            | Tensor Cores, warp primitives               |
| Turing       | 7.5                | Enhanced memory subsystem                   |
| Ampere       | 8.0-8.6            | 3rd gen Tensor Cores                        |
| Ada Lovelace | 8.9                | 4th gen Tensor Cores                        |
| Hopper       | 9.0                | 5th gen Tensor Cores, thread block clusters |

## Performance Optimization Strategies

### 1. Kernel Selection Based on Problem Size

```cpp
#include "material/cuda_kernels.cuh"

void choose_optimal_kernel(int num_particles) {
    if (num_particles < 1000) {
        // Use CPU for small problems due to GPU setup overhead
        computeContactForcesCPU();
    }
    else if (num_particles < 10000) {
        // Use basic GPU kernel
        compute_contact_forces_warp_optimized_kernel<<<grid, block>>>();
    }
    else if (num_particles < 100000) {
        // Use optimized kernel with shared memory
        compute_contact_forces_optimized_kernel<<<grid, block, shared_mem_size>>>();
    }
    else {
        // Use spatial optimization for large problems
        compute_contact_forces_spatial_kernel<<<grid, block>>>();
    }
}
```

### 2. Memory Layout Optimization

**Optimal Data Layout:**
```cpp
// Use Structure of Arrays (SoA) instead of Array of Structures (AoS)
struct ParticleDataSoA {
    std::vector<float> x, y, z, radius;  // Better for GPU
};

// Convert to float4 for vectorized access
std::vector<float4> convertToFloat4(const ParticleDataSoA& soa) {
    std::vector<float4> result(soa.x.size());
    for (size_t i = 0; i < soa.x.size(); i++) {
        result[i] = make_float4(soa.x[i], soa.y[i], soa.z[i], soa.radius[i]);
    }
    return result;
}
```

### 3. Occupancy Optimization

```cpp
// Calculate optimal block size for your GPU
void optimize_block_size() {
    int min_grid_size, block_size;
    
    cudaOccupancyMaxPotentialBlockSize(
        &min_grid_size, &block_size,
        compute_contact_forces_warp_optimized_kernel<MaterialView>,
        0, 0  // No shared memory, no block size limit
    );
    
    std::cout << "Optimal block size: " << block_size << std::endl;
    std::cout << "Minimum grid size: " << min_grid_size << std::endl;
}
```

## Kernel Selection Guide

### Basic Warp-Optimized Kernel
**Best for:** 1K-50K particles, single GPU
```cpp
compute_contact_forces_warp_optimized_kernel<<<grid, block>>>(
    materials, particle_data, material_ids, forces, num_particles, threshold
);
```

### Shared Memory Optimized Kernel
**Best for:** 10K-100K particles, memory-bound scenarios
```cpp
size_t shared_mem = block_size * sizeof(float4);
compute_contact_forces_optimized_kernel<<<grid, block, shared_mem>>>(
    materials, particle_data, material_ids, forces, num_particles, threshold
);
```

### Spatial Hash Kernel
**Best for:** 100K+ particles, sparse contact scenarios
```cpp
// Requires pre-computed neighbor lists
compute_contact_forces_spatial_kernel<<<grid, block>>>(
    materials, particle_data, material_ids, neighbor_list, 
    neighbor_count, forces, num_particles, threshold
);
```

## Memory Management Best Practices

### 1. Use Memory Pools
```cpp
#include "material/gpu_memory_manager.hpp"

class OptimizedSimulation {
private:
    GpuMemoryManager memory_manager_;
    
    // Pre-allocate persistent buffers
    GpuBuffer<float4> particle_buffer_;
    GpuBuffer<float4> force_buffer_;
    GpuBuffer<int> material_id_buffer_;
    
public:
    void initialize(int max_particles) {
        // Allocate once, reuse many times
        particle_buffer_ = memory_manager_.allocate<float4>(max_particles);
        force_buffer_ = memory_manager_.allocate<float4>(max_particles);
        material_id_buffer_ = memory_manager_.allocate<int>(max_particles);
    }
    
    void step(const std::vector<float4>& particles) {
        // Efficient updates without reallocation
        memory_manager_.copy_to_device(particle_buffer_, particles);
        
        // Launch kernel
        launch_kernel(particle_buffer_.get(), force_buffer_.get());
        
        // Get results
        std::vector<float4> forces;
        memory_manager_.copy_to_host(forces, force_buffer_, particles.size());
    }
};
```

### 2. Asynchronous Memory Transfers
```cpp
void async_simulation_step() {
    cudaStream_t compute_stream, transfer_stream;
    cudaStreamCreate(&compute_stream);
    cudaStreamCreate(&transfer_stream);
    
    // Overlap computation and memory transfer
    cudaMemcpyAsync(d_new_data, h_new_data, size, 
                   cudaMemcpyHostToDevice, transfer_stream);
    
    // Launch kernel on compute stream
    kernel<<<grid, block, 0, compute_stream>>>(d_current_data);
    
    // Transfer results back
    cudaMemcpyAsync(h_results, d_results, size,
                   cudaMemcpyDeviceToHost, transfer_stream);
    
    cudaStreamSynchronize(compute_stream);
    cudaStreamDestroy(compute_stream);
    cudaStreamDestroy(transfer_stream);
}
```

## Multi-GPU Configuration

### 1. Basic Multi-GPU Setup
```cpp
#include "material/cuda_kernels.cuh"

void multi_gpu_simulation() {
    MultiGpuMaterialProcessor processor;
    
    std::vector<MaterialView> materials = create_materials();
    std::vector<float4> particles = load_particles();
    std::vector<int> material_ids = assign_materials();
    std::vector<float4> forces;
    
    // Automatically distributes across available GPUs
    processor.compute_contact_forces_multi_gpu(
        materials, particles, material_ids, forces
    );
}
```

### 2. Manual GPU Management
```cpp
class ManualMultiGPU {
public:
    void setup_gpus() {
        int device_count;
        cudaGetDeviceCount(&device_count);
        
        for (int gpu = 0; gpu < device_count; gpu++) {
            cudaSetDevice(gpu);
            
            // Set memory pool for each GPU
            size_t pool_size = 1024 * 1024 * 1024; // 1GB per GPU
            cudaMemPoolProps pool_props = {};
            pool_props.allocType = cudaMemAllocationTypePinned;
            pool_props.location.type = cudaMemLocationTypeDevice;
            pool_props.location.id = gpu;
            
            cudaMemPool_t pool;
            cudaMemPoolCreate(&pool, &pool_props);
            cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, 
                                  &pool_size);
        }
    }
};
```

## Performance Benchmarking

### 1. Built-in Benchmark Tool
```bash
# Run comprehensive benchmarks
./build_cuda/gpu_benchmark

# Options:
./build_cuda/gpu_benchmark --particles 10000 --iterations 100
./build_cuda/gpu_benchmark --test-kernels --compare-cpu
```

### 2. Custom Performance Measurement
```cpp
#include <chrono>

class PerformanceMeter {
public:
    struct BenchmarkResult {
        double avg_time_ms;
        double throughput_mps;  // Million particles per second
        double memory_bandwidth_gbps;
    };
    
    BenchmarkResult benchmark_kernel(int num_particles, int iterations) {
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; i++) {
            launch_kernel(num_particles);
            cudaDeviceSynchronize();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        double avg_time_ms = duration.count() / (1000.0 * iterations);
        double throughput = (num_particles * iterations) / (duration.count() / 1e6);
        
        return {avg_time_ms, throughput / 1e6, calculate_bandwidth()};
    }
};
```

### 3. NVIDIA Profiling Tools
```bash
# Use Nsight Compute for kernel analysis
ncu --set full ./build_cuda/gpu_benchmark

# Use Nsight Systems for system-wide profiling
nsys profile --stats=true ./build_cuda/gpu_benchmark

# CUDA profiler for memory analysis
nvprof --print-gpu-trace ./build_cuda/gpu_benchmark
```

## Performance Tuning Guidelines

### Contact Force Computation Performance

| Particle Count | Expected GPU Speedup | Optimal Kernel |
| -------------- | -------------------- | -------------- |
| 100-1K         | 2-5x                 | Warp-optimized |
| 1K-10K         | 10-30x               | Shared memory  |
| 10K-100K       | 50-150x              | Spatial hash   |
| 100K+          | 100-500x             | Multi-GPU      |

### Memory Bandwidth Utilization

```cpp
// Check memory bandwidth efficiency
void check_memory_efficiency() {
    // Theoretical peak bandwidth (e.g., RTX 4090: 1008 GB/s)
    double peak_bandwidth = 1008e9; // bytes/s
    
    // Measure achieved bandwidth
    double achieved = measure_kernel_bandwidth();
    double efficiency = achieved / peak_bandwidth * 100;
    
    std::cout << "Memory bandwidth efficiency: " << efficiency << "%" << std::endl;
    
    if (efficiency < 60) {
        std::cout << "Consider optimizing memory access patterns" << std::endl;
    }
}
```

## CUDA Graphs Optimization

### When to Use CUDA Graphs
- Repeated kernel launches with same parameters
- Complex kernel sequences
- Minimizing CPU-GPU synchronization overhead

```cpp
void setup_cuda_graphs() {
    CudaGraphMaterialProcessor graph_processor;
    
    // Create graph for repeated computation
    graph_processor.create_graph(
        d_materials, d_particles, d_material_ids, d_forces,
        num_particles, contact_threshold
    );
    
    // Execute graph multiple times with minimal overhead
    for (int step = 0; step < num_simulation_steps; step++) {
        update_particle_positions();  // Update data
        graph_processor.execute_graph();  // Fast execution
        process_results();
    }
}
```

## Troubleshooting

### Common Performance Issues

1. **Low GPU utilization (<50%)**
   - Check occupancy with `nsight compute`
   - Increase block size or grid size
   - Ensure sufficient work per thread

2. **Memory bandwidth limited**
   - Use coalesced memory access patterns
   - Minimize random memory access
   - Consider using shared memory

3. **High kernel launch overhead**
   - Batch multiple operations
   - Use CUDA Graphs for repeated patterns
   - Minimize CPU-GPU synchronization

### Error Handling
```cpp
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

// Usage
CUDA_CHECK(cudaMalloc(&d_data, size));
CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
```

### Performance Debugging Checklist

- [ ] Verify optimal block size using occupancy calculator
- [ ] Check memory access patterns are coalesced
- [ ] Ensure sufficient parallelism (>10K threads)
- [ ] Profile with NVIDIA tools
- [ ] Compare against theoretical peak performance
- [ ] Test different kernel variants
- [ ] Verify data layout is GPU-friendly
- [ ] Check for unnecessary memory allocations
- [ ] Ensure proper error handling

## Conclusion

The Material GPU library provides extensive optimization options for different scenarios. Start with the basic kernels and progressively apply optimizations based on your specific requirements and hardware capabilities. Regular profiling and benchmarking will help you achieve optimal performance for your material simulation workloads.