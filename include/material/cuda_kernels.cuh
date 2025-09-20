#pragma once

// Include type definitions first
#include "material/contact_models.hpp"
#include "material/device_material_views.cuh"
#include "material/material_factory.hpp"
#include "material/property_mixins.hpp"

#ifdef MATERIAL_GPU_WITH_CUDA
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif

#include <chrono>
#include <iostream>
#include <vector>

namespace matgpu {

// Forward declarations to ensure types are available
struct DeviceElasticProperties;
struct DeviceThermalView;
struct DeviceElectromagneticView;

#ifdef MATERIAL_GPU_WITH_CUDA

// CUDA kernels for material property computations

// Baseline kernel using Structure-of-Arrays (SoA) inputs
template <typename MaterialView>
__global__ void compute_contact_forces_kernel(const MaterialView* materials, const float* pos_x, const float* pos_y,
                                              const float* pos_z, const float* radii, const int* material_ids,
                                              float* force_x, float* force_y, float* force_z, int num_particles,
                                              float contact_threshold) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles)
        return;

    float xi  = pos_x[i];
    float yi  = pos_y[i];
    float zi  = pos_z[i];
    float ri  = radii[i];
    int mat_i = material_ids[i];

    float fx = 0.0f, fy = 0.0f, fz = 0.0f;

    for (int j = 0; j < num_particles; ++j) {
        if (j == i)
            continue;

        float dx    = xi - pos_x[j];
        float dy    = yi - pos_y[j];
        float dz    = zi - pos_z[j];
        float rj    = radii[j];
        float sum_r = ri + rj;
        float dist2 = dx * dx + dy * dy + dz * dz;
        if (dist2 >= sum_r * sum_r)
            continue;

        float dist    = sqrtf(dist2 + 1e-20f);
        float overlap = sum_r - dist;
        if (overlap <= contact_threshold)
            continue;

        int mat_j       = material_ids[j];
        auto mat_view_i = materials[mat_i];
        auto mat_view_j = materials[mat_j];

        float normal_force = 0.5f * (mat_view_i.contact_model().normal_force(overlap, 0.0f) +
                                     mat_view_j.contact_model().normal_force(overlap, 0.0f));

        float inv_dist = rsqrtf(dist2 + 1e-12f);
        fx += normal_force * dx * inv_dist;
        fy += normal_force * dy * inv_dist;
        fz += normal_force * dz * inv_dist;
    }

    force_x[i] = fx;
    force_y[i] = fy;
    force_z[i] = fz;
}

// Optimized kernel using shared memory and tiled approach
template <typename MaterialView>
__global__ void compute_contact_forces_optimized_kernel(const MaterialView* materials,
                                                        const float4* particle_data, // position (x,y,z) + radius (w)
                                                        const int* material_ids,
                                                        float4* forces, // force (x,y,z) + potential energy (w)
                                                        int num_particles, float contact_threshold) {
    extern __shared__ float4 shared_particles[];

    int tid        = threadIdx.x;
    int bid        = blockIdx.x;
    int particle_i = bid * blockDim.x + tid;

    float4 force_acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    if (particle_i < num_particles) {
        float4 pi = particle_data[particle_i];
        int mat_i = material_ids[particle_i];

        // Process particles in tiles to utilize shared memory
        for (int tile = 0; tile < gridDim.x; tile++) {
            int particle_j = tile * blockDim.x + tid;

            // Load tile data into shared memory
            if (particle_j < num_particles) {
                shared_particles[tid] = particle_data[particle_j];
            } else {
                shared_particles[tid] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }

            __syncthreads();

            // Compute interactions with particles in this tile
            for (int j = 0; j < blockDim.x; j++) {
                int global_j = tile * blockDim.x + j;
                if (global_j >= num_particles || global_j == particle_i)
                    continue;

                float4 pj      = shared_particles[j];
                float dx       = pi.x - pj.x;
                float dy       = pi.y - pj.y;
                float dz       = pi.z - pj.z;
                float distance = sqrtf(dx * dx + dy * dy + dz * dz);
                float overlap  = pi.w + pj.w - distance;

                if (overlap > contact_threshold) {
                    int mat_j = material_ids[global_j];

                    // Get material views
                    auto mat_view_i = materials[mat_i];
                    auto mat_view_j = materials[mat_j];

                    // Compute contact force using mixed material properties
                    float normal_force = 0.5f * (mat_view_i.contact_model().normal_force(overlap, 0.0f) +
                                                 mat_view_j.contact_model().normal_force(overlap, 0.0f));

                    // Normalize contact direction
                    float inv_distance = rsqrtf(dx * dx + dy * dy + dz * dz + 1e-12f);
                    float nx           = dx * inv_distance;
                    float ny           = dy * inv_distance;
                    float nz           = dz * inv_distance;

                    // Accumulate forces
                    force_acc.x += normal_force * nx;
                    force_acc.y += normal_force * ny;
                    force_acc.z += normal_force * nz;
                    force_acc.w += 0.5f * normal_force * overlap; // Potential energy
                }
            }

            __syncthreads();
        }

        // Write final force
        forces[particle_i] = force_acc;
    }
}

// High-performance kernel using warp-level primitives
template <typename MaterialView>
__global__ void compute_contact_forces_warp_optimized_kernel(const MaterialView* materials, const float4* particle_data,
                                                             const int* material_ids, float4* forces, int num_particles,
                                                             float contact_threshold) {
    int tid        = threadIdx.x;
    int particle_i = blockIdx.x * blockDim.x + tid;

    float4 force_acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    if (particle_i < num_particles) {
        float4 pi = particle_data[particle_i];
        int mat_i = material_ids[particle_i];

        // Full-pair interaction for correctness
        for (int j = 0; j < num_particles; ++j) {
            if (j == particle_i)
                continue;

            float4 pj         = particle_data[j];
            float dx          = pi.x - pj.x;
            float dy          = pi.y - pj.y;
            float dz          = pi.z - pj.z;
            float distance_sq = dx * dx + dy * dy + dz * dz;
            float sum_radii   = pi.w + pj.w;

            if (distance_sq < sum_radii * sum_radii) {
                // Match CPU reference path precisely
                float distance = sqrtf(distance_sq);
                float overlap  = sum_radii - distance;

                if (overlap > contact_threshold) {
                    int mat_j       = material_ids[j];
                    auto mat_view_i = materials[mat_i];
                    auto mat_view_j = materials[mat_j];

                    float normal_force = 0.5f * (mat_view_i.contact_model().normal_force(overlap, 0.0f) +
                                                 mat_view_j.contact_model().normal_force(overlap, 0.0f));

                    float inv_distance = 1.0f / (distance + 1e-12f);

                    force_acc.x += normal_force * dx * inv_distance;
                    force_acc.y += normal_force * dy * inv_distance;
                    force_acc.z += normal_force * dz * inv_distance;
                    force_acc.w += 0.5f * normal_force * overlap;
                }
            }
        }

        forces[particle_i] = force_acc;
    }
}

// Advanced kernel using spatial sorting and neighbor lists for O(N) complexity
template <typename MaterialView>
__global__ void compute_contact_forces_spatial_kernel(const MaterialView* materials, const float4* particle_data,
                                                      const int* material_ids, const int* neighbor_list,
                                                      const int* neighbor_count, float4* forces, int num_particles,
                                                      float contact_threshold) {
    int particle_i = blockIdx.x * blockDim.x + threadIdx.x;

    if (particle_i >= num_particles)
        return;

    float4 force_acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 pi        = particle_data[particle_i];
    int mat_i        = material_ids[particle_i];

    int start_idx     = particle_i * 32; // Max 32 neighbors per particle
    int num_neighbors = neighbor_count[particle_i];

    // Process only actual neighbors instead of all particles
    for (int n = 0; n < num_neighbors; n++) {
        int j = neighbor_list[start_idx + n];
        if (j == particle_i)
            continue;

        float4 pj         = particle_data[j];
        float dx          = pi.x - pj.x;
        float dy          = pi.y - pj.y;
        float dz          = pi.z - pj.z;
        float distance_sq = dx * dx + dy * dy + dz * dz;
        float sum_radii   = pi.w + pj.w;

        if (distance_sq < sum_radii * sum_radii) {
            float distance = sqrtf(distance_sq);
            float overlap  = sum_radii - distance;

            if (overlap > contact_threshold) {
                int mat_j       = material_ids[j];
                auto mat_view_i = materials[mat_i];
                auto mat_view_j = materials[mat_j];

                float normal_force = 0.5f * (mat_view_i.contact_model().normal_force(overlap, 0.0f) +
                                             mat_view_j.contact_model().normal_force(overlap, 0.0f));

                float inv_distance = rsqrtf(distance_sq + 1e-12f);

                force_acc.x += normal_force * dx * inv_distance;
                force_acc.y += normal_force * dy * inv_distance;
                force_acc.z += normal_force * dz * inv_distance;
                force_acc.w += 0.5f * normal_force * overlap;
            }
        }
    }

    forces[particle_i] = force_acc;
}

// Kernel to compute material statistics across many particles
__global__ void compute_material_statistics_kernel(const DeviceElasticProperties* elastic_props, int num_materials,
                                                   float* avg_young_modulus, float* max_density,
                                                   float* min_poisson_ratio) {
    extern __shared__ float shared_data[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    float local_young   = 0.0f;
    float local_density = 0.0f;
    float local_poisson = 1.0f;

    // Process multiple elements per thread
    for (int i = gid; i < num_materials; i += blockDim.x * gridDim.x) {
        local_young += elastic_props[i].young;
        local_density = fmaxf(local_density, elastic_props[i].density);
        local_poisson = fminf(local_poisson, elastic_props[i].poisson);
    }

    // Store in shared memory
    shared_data[tid]                  = local_young;
    shared_data[tid + blockDim.x]     = local_density;
    shared_data[tid + 2 * blockDim.x] = local_poisson;

    __syncthreads();

    // Reduction within block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
            shared_data[tid + blockDim.x] =
                fmaxf(shared_data[tid + blockDim.x], shared_data[tid + stride + blockDim.x]);
            shared_data[tid + 2 * blockDim.x] =
                fminf(shared_data[tid + 2 * blockDim.x], shared_data[tid + stride + 2 * blockDim.x]);
        }
        __syncthreads();
    }

    // Write results from block 0
    if (blockIdx.x == 0 && tid == 0) {
        *avg_young_modulus = shared_data[0] / num_materials;
        *max_density       = shared_data[blockDim.x];
        *min_poisson_ratio = shared_data[2 * blockDim.x];
    }
}

// Kernel for thermal diffusion calculation
__global__ void thermal_diffusion_kernel(const DeviceThermalView* thermal_props, const float* temperatures,
                                         float* temperature_derivatives, const float* positions_x,
                                         const float* positions_y, const float* positions_z, int num_particles,
                                         float dt, float thermal_radius) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= num_particles)
        return;

    if (!thermal_props[i].has_thermal()) {
        temperature_derivatives[i] = 0.0f;
        return;
    }

    float T_i = temperatures[i];
    float xi  = positions_x[i];
    float yi  = positions_y[i];
    float zi  = positions_z[i];

    float thermal_diffusivity = thermal_props[i].k() / thermal_props[i].cp();
    float dT_dt               = 0.0f;

    // Simple finite difference thermal diffusion
    for (int j = 0; j < num_particles; j++) {
        if (i == j)
            continue;

        float dx       = xi - positions_x[j];
        float dy       = yi - positions_y[j];
        float dz       = zi - positions_z[j];
        float distance = sqrtf(dx * dx + dy * dy + dz * dz);

        if (distance < thermal_radius) {
            float T_j    = temperatures[j];
            float weight = expf(-distance / thermal_radius);
            dT_dt += thermal_diffusivity * (T_j - T_i) * weight;
        }
    }

    temperature_derivatives[i] = dT_dt;
}

#endif // MATERIAL_GPU_WITH_CUDA

// Advanced GPU optimization and multi-GPU support

#ifdef MATERIAL_GPU_WITH_CUDA

// Multi-GPU manager for scaling across multiple devices
class MultiGpuMaterialProcessor {
  private:
    std::vector<int> device_ids_;
    std::vector<cudaStream_t> streams_;
    std::vector<cudaEvent_t> events_;
    int num_devices_;

  public:
    MultiGpuMaterialProcessor() {
        cudaGetDeviceCount(&num_devices_);

        device_ids_.resize(num_devices_);
        streams_.resize(num_devices_);
        events_.resize(num_devices_);

        for (int i = 0; i < num_devices_; i++) {
            device_ids_[i] = i;
            cudaSetDevice(i);
            cudaStreamCreate(&streams_[i]);
            cudaEventCreate(&events_[i]);
        }
    }

    ~MultiGpuMaterialProcessor() {
        for (int i = 0; i < num_devices_; i++) {
            cudaSetDevice(i);
            cudaStreamDestroy(streams_[i]);
            cudaEventDestroy(events_[i]);
        }
    }

    template <typename MaterialView>
    void compute_contact_forces_multi_gpu(const std::vector<MaterialView>& materials,
                                          const std::vector<float4>& particle_data,
                                          const std::vector<int>& material_ids, std::vector<float4>& forces,
                                          float contact_threshold = 1e-6f) {
        int total_particles   = particle_data.size();
        int particles_per_gpu = (total_particles + num_devices_ - 1) / num_devices_;

        std::vector<MaterialView*> d_materials(num_devices_);
        std::vector<float4*> d_particle_data(num_devices_);
        std::vector<int*> d_material_ids(num_devices_);
        std::vector<float4*> d_forces(num_devices_);

        // Distribute data across GPUs
        for (int gpu = 0; gpu < num_devices_; gpu++) {
            cudaSetDevice(gpu);

            int start_idx     = gpu * particles_per_gpu;
            int end_idx       = std::min(start_idx + particles_per_gpu, total_particles);
            int gpu_particles = end_idx - start_idx;

            if (gpu_particles <= 0)
                continue;

            // Allocate GPU memory
            cudaMalloc(&d_materials[gpu], materials.size() * sizeof(MaterialView));
            cudaMalloc(&d_particle_data[gpu], total_particles * sizeof(float4)); // Need full data for interactions
            cudaMalloc(&d_material_ids[gpu], total_particles * sizeof(int));
            cudaMalloc(&d_forces[gpu], gpu_particles * sizeof(float4));

            // Copy data asynchronously
            cudaMemcpyAsync(d_materials[gpu], materials.data(), materials.size() * sizeof(MaterialView),
                            cudaMemcpyHostToDevice, streams_[gpu]);
            cudaMemcpyAsync(d_particle_data[gpu], particle_data.data(), total_particles * sizeof(float4),
                            cudaMemcpyHostToDevice, streams_[gpu]);
            cudaMemcpyAsync(d_material_ids[gpu], material_ids.data(), total_particles * sizeof(int),
                            cudaMemcpyHostToDevice, streams_[gpu]);
        }

        // Launch kernels on all GPUs
        for (int gpu = 0; gpu < num_devices_; gpu++) {
            cudaSetDevice(gpu);

            int start_idx     = gpu * particles_per_gpu;
            int end_idx       = std::min(start_idx + particles_per_gpu, total_particles);
            int gpu_particles = end_idx - start_idx;

            if (gpu_particles <= 0)
                continue;

            dim3 block_size(256);
            dim3 grid_size((gpu_particles + block_size.x - 1) / block_size.x);

            // Launch optimized kernel
            compute_contact_forces_warp_optimized_kernel<<<grid_size, block_size, 0, streams_[gpu]>>>(
                d_materials[gpu] + start_idx, d_particle_data[gpu], d_material_ids[gpu], d_forces[gpu], gpu_particles,
                contact_threshold);

            cudaEventRecord(events_[gpu], streams_[gpu]);
        }

        // Collect results
        forces.resize(total_particles);
        for (int gpu = 0; gpu < num_devices_; gpu++) {
            cudaSetDevice(gpu);

            int start_idx     = gpu * particles_per_gpu;
            int end_idx       = std::min(start_idx + particles_per_gpu, total_particles);
            int gpu_particles = end_idx - start_idx;

            if (gpu_particles <= 0)
                continue;

            cudaEventSynchronize(events_[gpu]);
            cudaMemcpyAsync(forces.data() + start_idx, d_forces[gpu], gpu_particles * sizeof(float4),
                            cudaMemcpyDeviceToHost, streams_[gpu]);
        }

        // Cleanup
        for (int gpu = 0; gpu < num_devices_; gpu++) {
            cudaSetDevice(gpu);
            cudaStreamSynchronize(streams_[gpu]);

            cudaFree(d_materials[gpu]);
            cudaFree(d_particle_data[gpu]);
            cudaFree(d_material_ids[gpu]);
            cudaFree(d_forces[gpu]);
        }
    }
};

// CUDA Graphs optimization for repeated kernel launches
class CudaGraphMaterialProcessor {
  private:
    cudaGraph_t graph_;
    cudaGraphExec_t graph_exec_;
    cudaStream_t stream_;
    bool graph_created_;

  public:
    CudaGraphMaterialProcessor() : graph_created_(false) { cudaStreamCreate(&stream_); }

    ~CudaGraphMaterialProcessor() {
        if (graph_created_) {
            cudaGraphExecDestroy(graph_exec_);
            cudaGraphDestroy(graph_);
        }
        cudaStreamDestroy(stream_);
    }

    template <typename MaterialView>
    void create_graph(MaterialView* d_materials, float4* d_particle_data, int* d_material_ids, float4* d_forces,
                      int num_particles, float contact_threshold = 1e-6f) {
        if (graph_created_) {
            cudaGraphExecDestroy(graph_exec_);
            cudaGraphDestroy(graph_);
        }

        cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal);

        dim3 block_size(256);
        dim3 grid_size((num_particles + block_size.x - 1) / block_size.x);

        compute_contact_forces_warp_optimized_kernel<<<grid_size, block_size, 0, stream_>>>(
            d_materials, d_particle_data, d_material_ids, d_forces, num_particles, contact_threshold);

        cudaStreamEndCapture(stream_, &graph_);
        cudaGraphInstantiate(&graph_exec_, graph_, NULL, NULL, 0);

        graph_created_ = true;
    }

    void execute_graph() {
        if (graph_created_) {
            cudaGraphLaunch(graph_exec_, stream_);
            cudaStreamSynchronize(stream_);
        }
    }
};

// Single GPU processor with advanced optimizations
class CudaMaterialProcessor {
  private:
    cudaStream_t stream_;

  public:
    CudaMaterialProcessor() { cudaStreamCreate(&stream_); }

    ~CudaMaterialProcessor() { cudaStreamDestroy(stream_); }

    template <typename MaterialView>
    void compute_contact_forces(const std::vector<MaterialView>& host_materials, const std::vector<float>& positions_x,
                                const std::vector<float>& positions_y, const std::vector<float>& positions_z,
                                const std::vector<float>& radii, const std::vector<int>& material_ids,
                                std::vector<float>& forces_x, std::vector<float>& forces_y,
                                std::vector<float>& forces_z, float contact_threshold = 1e-6f) {
        int num_particles = positions_x.size();

        // Allocate device memory
        MaterialView* d_materials;
        float *d_pos_x, *d_pos_y, *d_pos_z, *d_radii;
        float *d_force_x, *d_force_y, *d_force_z;
        int* d_material_ids;

        size_t materials_size   = host_materials.size() * sizeof(MaterialView);
        size_t float_array_size = num_particles * sizeof(float);
        size_t int_array_size   = num_particles * sizeof(int);

        cudaMalloc(&d_materials, materials_size);
        cudaMalloc(&d_pos_x, float_array_size);
        cudaMalloc(&d_pos_y, float_array_size);
        cudaMalloc(&d_pos_z, float_array_size);
        cudaMalloc(&d_radii, float_array_size);
        cudaMalloc(&d_force_x, float_array_size);
        cudaMalloc(&d_force_y, float_array_size);
        cudaMalloc(&d_force_z, float_array_size);
        cudaMalloc(&d_material_ids, int_array_size);

        // Copy data to device
        cudaMemcpyAsync(d_materials, host_materials.data(), materials_size, cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(d_pos_x, positions_x.data(), float_array_size, cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(d_pos_y, positions_y.data(), float_array_size, cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(d_pos_z, positions_z.data(), float_array_size, cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(d_radii, radii.data(), float_array_size, cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(d_material_ids, material_ids.data(), int_array_size, cudaMemcpyHostToDevice, stream_);

        // Initialize forces to zero
        cudaMemsetAsync(d_force_x, 0, float_array_size, stream_);
        cudaMemsetAsync(d_force_y, 0, float_array_size, stream_);
        cudaMemsetAsync(d_force_z, 0, float_array_size, stream_);

        // Launch kernel
        int block_size = 256;
        int grid_size  = (num_particles + block_size - 1) / block_size;

        compute_contact_forces_kernel<<<grid_size, block_size, 0, stream_>>>(
            d_materials, d_pos_x, d_pos_y, d_pos_z, d_radii, d_material_ids, d_force_x, d_force_y, d_force_z,
            num_particles, contact_threshold);

        // Copy results back
        forces_x.resize(num_particles);
        forces_y.resize(num_particles);
        forces_z.resize(num_particles);

        cudaMemcpyAsync(forces_x.data(), d_force_x, float_array_size, cudaMemcpyDeviceToHost, stream_);
        cudaMemcpyAsync(forces_y.data(), d_force_y, float_array_size, cudaMemcpyDeviceToHost, stream_);
        cudaMemcpyAsync(forces_z.data(), d_force_z, float_array_size, cudaMemcpyDeviceToHost, stream_);

        cudaStreamSynchronize(stream_);

        // Cleanup
        cudaFree(d_materials);
        cudaFree(d_pos_x);
        cudaFree(d_pos_y);
        cudaFree(d_pos_z);
        cudaFree(d_radii);
        cudaFree(d_force_x);
        cudaFree(d_force_y);
        cudaFree(d_force_z);
        cudaFree(d_material_ids);
    }

    cudaStream_t get_stream() const { return stream_; }
};

#else // !MATERIAL_GPU_WITH_CUDA

// Stub implementation for non-CUDA builds
class CudaMaterialProcessor {
  public:
    CudaMaterialProcessor()  = default;
    ~CudaMaterialProcessor() = default;

    template <typename MaterialView>
    void compute_contact_forces(const std::vector<MaterialView>&, const std::vector<float>&, const std::vector<float>&,
                                const std::vector<float>&, const std::vector<float>&, const std::vector<int>&,
                                std::vector<float>&, std::vector<float>&, std::vector<float>&, float = 1e-6f) {
        throw std::runtime_error("CUDA support not enabled. Compile with ENABLE_CUDA=ON");
    }
};

#endif // MATERIAL_GPU_WITH_CUDA

} // namespace matgpu