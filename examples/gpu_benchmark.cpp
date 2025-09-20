#include "material/cuda_kernels.cuh"
#include "material/device_material_views.cuh"

#ifdef MATERIAL_GPU_WITH_CUDA
#include "material/gpu_memory_manager.hpp"
#endif

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace matgpu;

class PerformanceBenchmark {
  private:
    std::mt19937 gen_;
    std::uniform_real_distribution<float> pos_dist_;
    std::uniform_real_distribution<float> radius_dist_;
    std::uniform_int_distribution<int> material_dist_;

  public:
    PerformanceBenchmark()
        : gen_(42) // Fixed seed for reproducibility
          ,
          pos_dist_(-10.0f, 10.0f), radius_dist_(0.1f, 0.5f), material_dist_(0, 2) {}

    void generate_test_data(int num_particles, std::vector<float>& pos_x, std::vector<float>& pos_y,
                            std::vector<float>& pos_z, std::vector<float>& radii, std::vector<int>& material_ids) {
        pos_x.clear();
        pos_y.clear();
        pos_z.clear();
        radii.clear();
        material_ids.clear();

        pos_x.reserve(num_particles);
        pos_y.reserve(num_particles);
        pos_z.reserve(num_particles);
        radii.reserve(num_particles);
        material_ids.reserve(num_particles);

        for (int i = 0; i < num_particles; ++i) {
            pos_x.push_back(pos_dist_(gen_));
            pos_y.push_back(pos_dist_(gen_));
            pos_z.push_back(pos_dist_(gen_));
            radii.push_back(radius_dist_(gen_));
            material_ids.push_back(material_dist_(gen_));
        }
    }

    void cpu_contact_force_calculation(const std::vector<DeviceElasticProperties>& elastic_props,
                                       const std::vector<DeviceEEPAContactView>& contact_models,
                                       const std::vector<float>& pos_x, const std::vector<float>& pos_y,
                                       const std::vector<float>& pos_z, const std::vector<float>& radii,
                                       const std::vector<int>& material_ids, std::vector<float>& forces_x,
                                       std::vector<float>& forces_y, std::vector<float>& forces_z,
                                       float contact_threshold = 1e-6f) {

        int num_particles = pos_x.size();
        forces_x.assign(num_particles, 0.0f);
        forces_y.assign(num_particles, 0.0f);
        forces_z.assign(num_particles, 0.0f);

        // Simple N^2 contact detection and force calculation
        for (int i = 0; i < num_particles; ++i) {
            float fx = 0.0f, fy = 0.0f, fz = 0.0f;
            float xi = pos_x[i], yi = pos_y[i], zi = pos_z[i];
            float ri  = radii[i];
            int mat_i = material_ids[i];

            for (int j = 0; j < num_particles; ++j) {
                if (i == j)
                    continue;

                float xj = pos_x[j], yj = pos_y[j], zj = pos_z[j];
                float rj  = radii[j];
                int mat_j = material_ids[j];

                float dx = xi - xj, dy = yi - yj, dz = zi - zj;
                float distance = std::sqrt(dx * dx + dy * dy + dz * dz);
                float overlap  = (ri + rj) - distance;

                if (overlap > contact_threshold) {
                    // Simple elastic contact force
                    float nx = dx / distance, ny = dy / distance, nz = dz / distance;

                    // Get effective modulus (simplified)
                    float E_i   = elastic_props[mat_i].young;
                    float E_j   = elastic_props[mat_j].young;
                    float E_eff = 2.0f / (1.0f / E_i + 1.0f / E_j);

                    float fn = E_eff * overlap * std::sqrt(overlap);

                    fx += fn * nx;
                    fy += fn * ny;
                    fz += fn * nz;
                }
            }

            forces_x[i] = fx;
            forces_y[i] = fy;
            forces_z[i] = fz;
        }
    }

    void run_cpu_benchmark(int num_particles) {
        std::cout << "=== CPU Benchmark (N=" << num_particles << ") ===" << std::endl;

        // Create test materials
        std::vector<DeviceElasticProperties> elastic_props = {
            {2.1e11f, 0.29f, 7850.0f}, // Steel
            {1.0e7f, 0.25f, 2500.0f},  // Concrete
            {5.0e6f, 0.4f, 1200.0f}    // Polymer
        };

        std::vector<DeviceEEPAContactView> contact_models = {
            {1e6f, 5e5f, 0.3f, 0.15f}, // Steel
            {5e4f, 2e4f, 0.2f, 0.1f},  // Concrete
            {3e4f, 1e4f, 0.15f, 0.08f} // Polymer
        };

        // Generate test data
        std::vector<float> pos_x, pos_y, pos_z, radii;
        std::vector<int> material_ids;
        generate_test_data(num_particles, pos_x, pos_y, pos_z, radii, material_ids);

        // Benchmark CPU calculation
        std::vector<float> cpu_forces_x, cpu_forces_y, cpu_forces_z;

        auto start_cpu = std::chrono::high_resolution_clock::now();
        cpu_contact_force_calculation(elastic_props, contact_models, pos_x, pos_y, pos_z, radii, material_ids,
                                      cpu_forces_x, cpu_forces_y, cpu_forces_z);
        auto end_cpu = std::chrono::high_resolution_clock::now();

        auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu);

        // Calculate some statistics
        float total_force = 0.0f;
        for (int i = 0; i < num_particles; ++i) {
            total_force += std::sqrt(cpu_forces_x[i] * cpu_forces_x[i] + cpu_forces_y[i] * cpu_forces_y[i] +
                                     cpu_forces_z[i] * cpu_forces_z[i]);
        }

        std::cout << "  CPU Time: " << cpu_duration.count() / 1000.0 << " ms" << std::endl;
        std::cout << "  Total Force Magnitude: " << total_force << std::endl;
        std::cout << "  Avg Force per Particle: " << total_force / num_particles << std::endl;
    }

    double run_cpu_benchmark_silent(int num_particles) {
        // Create materials
        std::vector<DeviceElasticProperties> elastic_props = {
            {2.1e11f, 0.29f, 7850.0f}, // Steel
            {1.0e7f, 0.25f, 2500.0f},  // Concrete
            {5.0e6f, 0.4f, 1200.0f}    // Polymer
        };

        std::vector<DeviceEEPAContactView> contact_models = {
            {1e6f, 5e5f, 0.3f, 0.15f}, // Steel
            {5e4f, 2e4f, 0.2f, 0.1f},  // Concrete
            {3e4f, 1e4f, 0.15f, 0.08f} // Polymer
        };

        // Generate test data
        std::vector<float> pos_x, pos_y, pos_z, radii;
        std::vector<int> material_ids;
        generate_test_data(num_particles, pos_x, pos_y, pos_z, radii, material_ids);

        // Benchmark CPU calculation
        std::vector<float> cpu_forces_x, cpu_forces_y, cpu_forces_z;

        auto start_cpu = std::chrono::high_resolution_clock::now();
        cpu_contact_force_calculation(elastic_props, contact_models, pos_x, pos_y, pos_z, radii, material_ids,
                                      cpu_forces_x, cpu_forces_y, cpu_forces_z);
        auto end_cpu = std::chrono::high_resolution_clock::now();

        auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu);
        return cpu_duration.count() / 1000.0; // Convert to milliseconds
    }

#ifdef MATERIAL_GPU_WITH_CUDA
    void run_gpu_benchmark(int num_particles) {
        std::cout << "=== GPU Benchmark (N=" << num_particles << ") ===" << std::endl;

        try {
            // Create test materials
            std::vector<DeviceEEPAMaterial> materials = {
                DeviceEEPAMaterial{{2.1e11f, 0.29f, 7850.0f}, {1e6f, 5e5f, 0.3f, 0.15f}},
                DeviceEEPAMaterial{{1.0e7f, 0.25f, 2500.0f}, {5e4f, 2e4f, 0.2f, 0.1f}},
                DeviceEEPAMaterial{{5.0e6f, 0.4f, 1200.0f}, {3e4f, 1e4f, 0.15f, 0.08f}}};

            // Generate test data
            std::vector<float> pos_x, pos_y, pos_z, radii;
            std::vector<int> material_ids;
            generate_test_data(num_particles, pos_x, pos_y, pos_z, radii, material_ids);

            // GPU calculation using CudaMaterialProcessor
            CudaMaterialProcessor gpu_processor;
            std::vector<float> gpu_forces_x, gpu_forces_y, gpu_forces_z;

            auto start_gpu = std::chrono::high_resolution_clock::now();
            gpu_processor.compute_contact_forces(materials, pos_x, pos_y, pos_z, radii, material_ids, gpu_forces_x,
                                                 gpu_forces_y, gpu_forces_z);
            auto end_gpu = std::chrono::high_resolution_clock::now();

            auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu);

            // Calculate statistics
            float total_force = 0.0f;
            for (int i = 0; i < num_particles; ++i) {
                total_force += std::sqrt(gpu_forces_x[i] * gpu_forces_x[i] + gpu_forces_y[i] * gpu_forces_y[i] +
                                         gpu_forces_z[i] * gpu_forces_z[i]);
            }

            std::cout << "  GPU Time: " << gpu_duration.count() / 1000.0 << " ms" << std::endl;
            std::cout << "  Total Force Magnitude: " << total_force << std::endl;
            std::cout << "  Avg Force per Particle: " << total_force / num_particles << std::endl;

        } catch (const std::exception& e) {
            std::cout << "  GPU Error: " << e.what() << std::endl;
        }
    }

    double run_gpu_benchmark_silent(int num_particles) {
        try {
            // Create test materials
            std::vector<DeviceEEPAMaterial> materials = {
                DeviceEEPAMaterial{{2.1e11f, 0.29f, 7850.0f}, {1e6f, 5e5f, 0.3f, 0.15f}},
                DeviceEEPAMaterial{{1.0e7f, 0.25f, 2500.0f}, {5e4f, 2e4f, 0.2f, 0.1f}},
                DeviceEEPAMaterial{{5.0e6f, 0.4f, 1200.0f}, {3e4f, 1e4f, 0.15f, 0.08f}}};

            // Generate test data
            std::vector<float> pos_x, pos_y, pos_z, radii;
            std::vector<int> material_ids;
            generate_test_data(num_particles, pos_x, pos_y, pos_z, radii, material_ids);

            // GPU calculation using CudaMaterialProcessor
            CudaMaterialProcessor gpu_processor;
            std::vector<float> gpu_forces_x, gpu_forces_y, gpu_forces_z;

            auto start_gpu = std::chrono::high_resolution_clock::now();
            gpu_processor.compute_contact_forces(materials, pos_x, pos_y, pos_z, radii, material_ids, gpu_forces_x,
                                                 gpu_forces_y, gpu_forces_z);
            auto end_gpu = std::chrono::high_resolution_clock::now();

            auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_gpu - start_gpu);
            return gpu_duration.count() / 1000.0; // Convert to milliseconds

        } catch (const std::exception& e) {
            return -1.0; // Error occurred
        }
    }

    void run_memory_benchmark() {
        std::cout << "=== GPU Memory Management Benchmark ===" << std::endl;

        try {
            GpuMemoryManager memory_manager;

            const int num_particles = 10000;
            const int num_materials = 3;

            // Test memory allocation times
            auto start = std::chrono::high_resolution_clock::now();

            // Create GPU buffers
            GpuBuffer<float> pos_x(num_particles, memory_manager);
            GpuBuffer<float> pos_y(num_particles, memory_manager);
            GpuBuffer<float> pos_z(num_particles, memory_manager);
            GpuBuffer<float> forces_x(num_particles, memory_manager);
            GpuBuffer<float> forces_y(num_particles, memory_manager);
            GpuBuffer<float> forces_z(num_particles, memory_manager);

            auto alloc_end = std::chrono::high_resolution_clock::now();

            // Test data transfer
            std::vector<float> host_data(num_particles, 1.0f);
            pos_x.copy_from_host_async(host_data);
            pos_y.copy_from_host_async(host_data);
            pos_z.copy_from_host_async(host_data);

            memory_manager.synchronize();
            auto transfer_end = std::chrono::high_resolution_clock::now();

            auto alloc_time    = std::chrono::duration_cast<std::chrono::microseconds>(alloc_end - start);
            auto transfer_time = std::chrono::duration_cast<std::chrono::microseconds>(transfer_end - alloc_end);

            std::cout << "  Allocation Time: " << alloc_time.count() << " μs" << std::endl;
            std::cout << "  Transfer Time: " << transfer_time.count() << " μs" << std::endl;
            std::cout << "  Memory Allocated: " << (6 * num_particles * sizeof(float)) / 1024 << " KB" << std::endl;

        } catch (const std::exception& e) {
            std::cout << "  Memory Error: " << e.what() << std::endl;
        }
    }
#endif

    void run_comprehensive_benchmark() {
        std::cout << "========================================" << std::endl;
        std::cout << "   Material GPU Performance Benchmark" << std::endl;
        std::cout << "========================================" << std::endl;

        std::vector<int> test_sizes = {100, 500, 1000, 2000, 5000};

        std::cout << std::left << std::setw(8) << "N" << std::setw(12) << "CPU (ms)"
#ifdef MATERIAL_GPU_WITH_CUDA
                  << std::setw(12) << "GPU (ms)" << std::setw(12) << "Speedup"
#endif
                  << std::endl;
        std::cout << std::string(50, '-') << std::endl;

        for (int n : test_sizes) {
            // Run CPU benchmark without output
            double cpu_time_ms = run_cpu_benchmark_silent(n);

            std::cout << std::left << std::setw(8) << n;
            std::cout << std::setw(12) << std::fixed << std::setprecision(2) << cpu_time_ms;

#ifdef MATERIAL_GPU_WITH_CUDA
            // Run GPU benchmark without output
            double gpu_time_ms = run_gpu_benchmark_silent(n);

            float speedup = (gpu_time_ms > 0) ? static_cast<float>(cpu_time_ms) / gpu_time_ms : 0.0f;
            std::cout << std::setw(12) << std::fixed << std::setprecision(2) << gpu_time_ms << std::setw(12)
                      << std::fixed << std::setprecision(2) << speedup << "x";
#endif
            std::cout << std::endl;
        }

#ifdef MATERIAL_GPU_WITH_CUDA
        std::cout << std::endl;
        run_memory_benchmark();
#endif
    }
};

int main() {
    try {
        PerformanceBenchmark benchmark;
        benchmark.run_comprehensive_benchmark();

#ifdef MATERIAL_GPU_WITH_CUDA
        std::cout << "\nCUDA support is enabled." << std::endl;

        // Print GPU information
        int device_count;
        cudaGetDeviceCount(&device_count);
        std::cout << "Number of CUDA devices: " << device_count << std::endl;

        if (device_count > 0) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            std::cout << "GPU: " << prop.name << std::endl;
            std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
            std::cout << "Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
            std::cout << "Multiprocessors: " << prop.multiProcessorCount << std::endl;
        }
#else
        std::cout << "\nCUDA support is NOT enabled. Compile with ENABLE_CUDA=ON for GPU acceleration." << std::endl;
#endif

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}