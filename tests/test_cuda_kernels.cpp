#include "material/cuda_kernels.cuh"
#include "material/gpu_memory_manager.hpp"
#include "material/material_factory.hpp"

#include <gtest/gtest.h>

#ifdef MATERIAL_GPU_WITH_CUDA
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <random>
#include <vector>

namespace matgpu {

class CudaKernelTest : public ::testing::Test {
  protected:
    void SetUp() override {
        // Check if CUDA is available
        int device_count;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        if (error != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "CUDA not available on this system";
        }

        // Initialize test data
        num_particles_     = 1000;
        contact_threshold_ = 1e-6f;

        generateTestData();
    }

    void TearDown() override {
        // Cleanup any GPU memory if tests were run
        cudaDeviceReset();
    }

    void generateTestData() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> pos_dist(-10.0f, 10.0f);
        std::uniform_real_distribution<float> rad_dist(0.1f, 0.5f);
        std::uniform_int_distribution<int> mat_dist(0, 2);

        // Generate random particle data
        particle_data_.resize(num_particles_);
        material_ids_.resize(num_particles_);

        for (int i = 0; i < num_particles_; i++) {
            particle_data_[i] = make_float4(pos_dist(gen), pos_dist(gen), pos_dist(gen), rad_dist(gen));
            material_ids_[i]  = mat_dist(gen);
        }

        // Create test materials
        auto eepa_mat =
            MaterialFactoryShortcuts::eepa().young_modulus(2.1e11f).poisson_ratio(0.3f).density(7850.0f).build();

        auto jkr_mat =
            MaterialFactoryShortcuts::jkr().young_modulus(1e7f).poisson_ratio(0.25f).density(2000.0f).build();

        auto standard_mat =
            MaterialFactoryShortcuts::standard().young_modulus(3e9f).poisson_ratio(0.35f).density(1500.0f).build();

        // Convert to device views
        device_materials_.push_back(make_host_material_view(DeviceElasticProperties{2.1e11f, 0.3f, 7850.0f},
                                                            DeviceEEPAContactView{1e6f, 5e5f, 0.3f, 0.15f}));

        // Use EEPA-based device material views consistently in tests
        device_materials_.push_back(make_host_material_view(DeviceElasticProperties{1e7f, 0.25f, 2000.0f},
                                                            DeviceEEPAContactView{8e5f, 4e5f, 0.25f, 0.12f}));

        device_materials_.push_back(make_host_material_view(DeviceElasticProperties{3e9f, 0.35f, 1500.0f},
                                                            DeviceEEPAContactView{5e5f, 2.5e5f, 0.2f, 0.1f}));
    }

    // Helper function to compute CPU reference forces
    std::vector<float4> computeCpuReference() {
        std::vector<float4> cpu_forces(num_particles_, make_float4(0.0f, 0.0f, 0.0f, 0.0f));

        for (int i = 0; i < num_particles_; i++) {
            auto pi    = particle_data_[i];
            auto mat_i = device_materials_[material_ids_[i]];

            for (int j = 0; j < num_particles_; j++) {
                if (i == j)
                    continue;

                auto pj        = particle_data_[j];
                float dx       = pi.x - pj.x;
                float dy       = pi.y - pj.y;
                float dz       = pi.z - pj.z;
                float distance = sqrtf(dx * dx + dy * dy + dz * dz);
                float overlap  = pi.w + pj.w - distance;

                if (overlap > contact_threshold_) {
                    auto mat_j = device_materials_[material_ids_[j]];

                    float normal_force = 0.5f * (mat_i.contact_model().normal_force(overlap, 0.0f) +
                                                 mat_j.contact_model().normal_force(overlap, 0.0f));

                    float inv_distance = 1.0f / (distance + 1e-12f);
                    float nx           = dx * inv_distance;
                    float ny           = dy * inv_distance;
                    float nz           = dz * inv_distance;

                    cpu_forces[i].x += normal_force * nx;
                    cpu_forces[i].y += normal_force * ny;
                    cpu_forces[i].z += normal_force * nz;
                    cpu_forces[i].w += 0.5f * normal_force * overlap;
                }
            }
        }

        return cpu_forces;
    }

  protected:
    int num_particles_;
    float contact_threshold_;
    std::vector<float4> particle_data_;
    std::vector<int> material_ids_;
    using MaterialViewT = DeviceMaterialView<DeviceEEPAContactView, false, false>;
    std::vector<MaterialViewT> device_materials_;
};

TEST_F(CudaKernelTest, BasicKernelExecution) {
    GpuMemoryManager memory_manager;

    // Allocate GPU memory
    auto d_materials     = memory_manager.allocate<decltype(device_materials_[0])>(device_materials_.size());
    auto d_particle_data = memory_manager.allocate<float4>(num_particles_);
    auto d_material_ids  = memory_manager.allocate<int>(num_particles_);
    auto d_forces        = memory_manager.allocate<float4>(num_particles_);

    // Copy data to GPU
    memory_manager.copy_to_device(d_materials, device_materials_);
    memory_manager.copy_to_device(d_particle_data, particle_data_);
    memory_manager.copy_to_device(d_material_ids, material_ids_);

    // Launch kernel
    dim3 block_size(256);
    dim3 grid_size((num_particles_ + block_size.x - 1) / block_size.x);

    compute_contact_forces_warp_optimized_kernel<<<grid_size, block_size>>>(d_materials.get(), d_particle_data.get(),
                                                                            d_material_ids.get(), d_forces.get(),
                                                                            num_particles_, contact_threshold_);

    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Copy results back
    std::vector<float4> gpu_forces;
    memory_manager.copy_to_host(gpu_forces, d_forces, num_particles_);

    EXPECT_EQ(gpu_forces.size(), num_particles_);
}

TEST_F(CudaKernelTest, CpuGpuConsistency) {
    // Compute CPU reference
    auto cpu_forces = computeCpuReference();

    // Compute GPU result
    GpuMemoryManager memory_manager;
    auto d_materials     = memory_manager.allocate<decltype(device_materials_[0])>(device_materials_.size());
    auto d_particle_data = memory_manager.allocate<float4>(num_particles_);
    auto d_material_ids  = memory_manager.allocate<int>(num_particles_);
    auto d_forces        = memory_manager.allocate<float4>(num_particles_);

    memory_manager.copy_to_device(d_materials, device_materials_);
    memory_manager.copy_to_device(d_particle_data, particle_data_);
    memory_manager.copy_to_device(d_material_ids, material_ids_);

    dim3 block_size(256);
    dim3 grid_size((num_particles_ + block_size.x - 1) / block_size.x);

    compute_contact_forces_warp_optimized_kernel<<<grid_size, block_size>>>(d_materials.get(), d_particle_data.get(),
                                                                            d_material_ids.get(), d_forces.get(),
                                                                            num_particles_, contact_threshold_);

    cudaDeviceSynchronize();

    std::vector<float4> gpu_forces;
    memory_manager.copy_to_host(gpu_forces, d_forces, num_particles_);

    // Compare results with combined relative and absolute tolerances
    auto nearly_equal = [](float a, float b, float abs_tol, float rel_tol) {
        float diff  = fabsf(a - b);
        float scale = fmaxf(fabsf(a), fabsf(b));
        return diff <= fmaxf(abs_tol, rel_tol * scale);
    };

    const float abs_tol = 1e-3f;
    const float rel_tol = 1e-5f;
    for (int i = 0; i < num_particles_; i++) {
        EXPECT_TRUE(nearly_equal(cpu_forces[i].x, gpu_forces[i].x, abs_tol, rel_tol))
            << "Force X mismatch at particle " << i << ": cpu " << cpu_forces[i].x << " vs gpu " << gpu_forces[i].x;
        EXPECT_TRUE(nearly_equal(cpu_forces[i].y, gpu_forces[i].y, abs_tol, rel_tol))
            << "Force Y mismatch at particle " << i << ": cpu " << cpu_forces[i].y << " vs gpu " << gpu_forces[i].y;
        EXPECT_TRUE(nearly_equal(cpu_forces[i].z, gpu_forces[i].z, abs_tol, rel_tol))
            << "Force Z mismatch at particle " << i << ": cpu " << cpu_forces[i].z << " vs gpu " << gpu_forces[i].z;
        EXPECT_TRUE(nearly_equal(cpu_forces[i].w, gpu_forces[i].w, abs_tol, rel_tol))
            << "Energy mismatch at particle " << i << ": cpu " << cpu_forces[i].w << " vs gpu " << gpu_forces[i].w;
    }
}

TEST_F(CudaKernelTest, PerformanceBenchmark) {
    const int benchmark_particles = 10000;
    const int num_iterations      = 10;

    // Generate larger dataset for performance testing
    std::vector<float4> large_particle_data(benchmark_particles);
    std::vector<int> large_material_ids(benchmark_particles);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> pos_dist(-50.0f, 50.0f);
    std::uniform_real_distribution<float> rad_dist(0.05f, 0.2f);
    std::uniform_int_distribution<int> mat_dist(0, 2);

    for (int i = 0; i < benchmark_particles; i++) {
        large_particle_data[i] = make_float4(pos_dist(gen), pos_dist(gen), pos_dist(gen), rad_dist(gen));
        large_material_ids[i]  = mat_dist(gen);
    }

    GpuMemoryManager memory_manager;
    auto d_materials     = memory_manager.allocate<decltype(device_materials_[0])>(device_materials_.size());
    auto d_particle_data = memory_manager.allocate<float4>(benchmark_particles);
    auto d_material_ids  = memory_manager.allocate<int>(benchmark_particles);
    auto d_forces        = memory_manager.allocate<float4>(benchmark_particles);

    memory_manager.copy_to_device(d_materials, device_materials_);
    memory_manager.copy_to_device(d_particle_data, large_particle_data);
    memory_manager.copy_to_device(d_material_ids, large_material_ids);

    dim3 block_size(256);
    dim3 grid_size((benchmark_particles + block_size.x - 1) / block_size.x);

    // Warm up
    compute_contact_forces_warp_optimized_kernel<<<grid_size, block_size>>>(d_materials.get(), d_particle_data.get(),
                                                                            d_material_ids.get(), d_forces.get(),
                                                                            benchmark_particles, contact_threshold_);
    cudaDeviceSynchronize();

    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < num_iterations; iter++) {
        compute_contact_forces_warp_optimized_kernel<<<grid_size, block_size>>>(
            d_materials.get(), d_particle_data.get(), d_material_ids.get(), d_forces.get(), benchmark_particles,
            contact_threshold_);
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    auto duration      = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg_time_ms = duration.count() / (1000.0 * num_iterations);

    // Performance metrics
    double interactions_per_sec =
        (benchmark_particles * benchmark_particles * num_iterations) / (duration.count() / 1e6);

    std::cout << "GPU Performance Benchmark Results:" << std::endl;
    std::cout << "  Particles: " << benchmark_particles << std::endl;
    std::cout << "  Average time per iteration: " << avg_time_ms << " ms" << std::endl;
    std::cout << "  Interactions per second: " << interactions_per_sec / 1e6 << " M/s" << std::endl;

    // Should be significantly faster than naive O(N²) CPU implementation
    EXPECT_LT(avg_time_ms, 1000.0); // Should complete in less than 1 second
}

TEST_F(CudaKernelTest, MultiGpuProcessor) {
    int device_count;
    cudaGetDeviceCount(&device_count);

    if (device_count < 2) {
        GTEST_SKIP() << "Multi-GPU test requires at least 2 GPUs";
    }

    MultiGpuMaterialProcessor multi_processor;
    std::vector<float4> forces_result;

    // This should not crash and should distribute work across GPUs
    EXPECT_NO_THROW({
        multi_processor.compute_contact_forces_multi_gpu(device_materials_, particle_data_, material_ids_,
                                                         forces_result, contact_threshold_);
    });

    EXPECT_EQ(forces_result.size(), num_particles_);
}

TEST_F(CudaKernelTest, CudaGraphsOptimization) {
    CudaGraphMaterialProcessor graph_processor;

    GpuMemoryManager memory_manager;
    auto d_materials     = memory_manager.allocate<decltype(device_materials_[0])>(device_materials_.size());
    auto d_particle_data = memory_manager.allocate<float4>(num_particles_);
    auto d_material_ids  = memory_manager.allocate<int>(num_particles_);
    auto d_forces        = memory_manager.allocate<float4>(num_particles_);

    memory_manager.copy_to_device(d_materials, device_materials_);
    memory_manager.copy_to_device(d_particle_data, particle_data_);
    memory_manager.copy_to_device(d_material_ids, material_ids_);

    // Create and execute graph
    EXPECT_NO_THROW({
        graph_processor.create_graph(d_materials.get(), d_particle_data.get(), d_material_ids.get(), d_forces.get(),
                                     num_particles_, contact_threshold_);

        graph_processor.execute_graph();
    });

    // Verify results
    std::vector<float4> graph_forces;
    memory_manager.copy_to_host(graph_forces, d_forces, num_particles_);

    EXPECT_EQ(graph_forces.size(), num_particles_);
}

TEST_F(CudaKernelTest, MemoryCoalescingEfficiency) {
    // Test different memory access patterns
    const int test_particles = 8192; // Power of 2 for better memory alignment

    std::vector<float4> aligned_data(test_particles);
    std::vector<int> aligned_ids(test_particles);

    // Generate aligned data
    for (int i = 0; i < test_particles; i++) {
        aligned_data[i] = make_float4(i * 0.1f, i * 0.1f, i * 0.1f, 0.1f);
        aligned_ids[i]  = i % 3;
    }

    GpuMemoryManager memory_manager;
    auto d_materials     = memory_manager.allocate<decltype(device_materials_[0])>(device_materials_.size());
    auto d_particle_data = memory_manager.allocate<float4>(test_particles);
    auto d_material_ids  = memory_manager.allocate<int>(test_particles);
    auto d_forces        = memory_manager.allocate<float4>(test_particles);

    memory_manager.copy_to_device(d_materials, device_materials_);
    memory_manager.copy_to_device(d_particle_data, aligned_data);
    memory_manager.copy_to_device(d_material_ids, aligned_ids);

    // Test with different block sizes to assess memory coalescing
    std::vector<int> block_sizes = {64, 128, 256, 512};

    for (int block_size : block_sizes) {
        dim3 block(block_size);
        dim3 grid((test_particles + block_size - 1) / block_size);

        auto start = std::chrono::high_resolution_clock::now();

        compute_contact_forces_warp_optimized_kernel<<<grid, block>>>(d_materials.get(), d_particle_data.get(),
                                                                      d_material_ids.get(), d_forces.get(),
                                                                      test_particles, contact_threshold_);

        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        std::cout << "Block size " << block_size << ": " << duration.count() << " μs" << std::endl;

        EXPECT_EQ(cudaGetLastError(), cudaSuccess);
    }
}

} // namespace matgpu

#else // MATERIAL_GPU_WITH_CUDA

// Dummy tests when CUDA is not available
TEST(CudaKernelTest, CudaNotAvailable) { GTEST_SKIP() << "CUDA support not enabled in this build"; }

#endif // MATERIAL_GPU_WITH_CUDA