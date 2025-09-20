#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#ifdef MATERIAL_GPU_WITH_CUDA
#include "material/cuda_kernels.cuh"
#include "material/device_material_views.cuh"

#include <cuda_runtime.h>
#endif

// Simple DEM simulation demonstration
class DEMSimulation {
  private:
    int num_particles_;
    float domain_size_;
    float time_step_;

    // Particle data
    std::vector<float> pos_x_, pos_y_, pos_z_, radius_;
    std::vector<float> vel_x_, vel_y_, vel_z_, mass_;
    std::vector<float> force_x_, force_y_, force_z_;

  public:
    DEMSimulation(int num_particles, float domain_size)
        : num_particles_(num_particles), domain_size_(domain_size), time_step_(1e-5f) {
        initializeParticles();
    }

    void initializeParticles() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> pos_dist(-domain_size_ / 2, domain_size_ / 2);
        std::uniform_real_distribution<float> radius_dist(0.01f, 0.05f);
        std::uniform_real_distribution<float> vel_dist(-1.0f, 1.0f);

        // Resize arrays
        pos_x_.resize(num_particles_);
        pos_y_.resize(num_particles_);
        pos_z_.resize(num_particles_);
        radius_.resize(num_particles_);
        vel_x_.resize(num_particles_);
        vel_y_.resize(num_particles_);
        vel_z_.resize(num_particles_);
        mass_.resize(num_particles_);
        force_x_.resize(num_particles_);
        force_y_.resize(num_particles_);
        force_z_.resize(num_particles_);

        for (int i = 0; i < num_particles_; i++) {
            pos_x_[i]  = pos_dist(gen);
            pos_y_[i]  = pos_dist(gen);
            pos_z_[i]  = pos_dist(gen);
            radius_[i] = radius_dist(gen);

            vel_x_[i] = vel_dist(gen);
            vel_y_[i] = vel_dist(gen);
            vel_z_[i] = vel_dist(gen);

            float density = 2500.0f;
            float volume  = (4.0f / 3.0f) * M_PI * pow(radius_[i], 3);
            mass_[i]      = density * volume;

            force_x_[i] = 0.0f;
            force_y_[i] = 0.0f;
            force_z_[i] = -9.81f * mass_[i]; // Gravity
        }
    }

    void computeForces() {
        // Reset forces (keep gravity)
        for (int i = 0; i < num_particles_; i++) {
            force_x_[i] = 0.0f;
            force_y_[i] = 0.0f;
            force_z_[i] = -9.81f * mass_[i];
        }

        // Simple contact forces
        float contact_stiffness = 1e6f;

        for (int i = 0; i < num_particles_; i++) {
            for (int j = i + 1; j < num_particles_; j++) {
                float dx       = pos_x_[i] - pos_x_[j];
                float dy       = pos_y_[i] - pos_y_[j];
                float dz       = pos_z_[i] - pos_z_[j];
                float distance = sqrt(dx * dx + dy * dy + dz * dz);
                float overlap  = radius_[i] + radius_[j] - distance;

                if (overlap > 0) {
                    float normal_force = contact_stiffness * overlap;
                    float inv_distance = 1.0f / (distance + 1e-12f);
                    float nx           = dx * inv_distance;
                    float ny           = dy * inv_distance;
                    float nz           = dz * inv_distance;

                    force_x_[i] += normal_force * nx;
                    force_y_[i] += normal_force * ny;
                    force_z_[i] += normal_force * nz;

                    force_x_[j] -= normal_force * nx;
                    force_y_[j] -= normal_force * ny;
                    force_z_[j] -= normal_force * nz;
                }
            }
        }
    }

    void integrateMotion() {
        for (int i = 0; i < num_particles_; i++) {
            float inv_mass = 1.0f / mass_[i];

            // Update velocity
            vel_x_[i] += force_x_[i] * inv_mass * time_step_;
            vel_y_[i] += force_y_[i] * inv_mass * time_step_;
            vel_z_[i] += force_z_[i] * inv_mass * time_step_;

            // Update position
            pos_x_[i] += vel_x_[i] * time_step_;
            pos_y_[i] += vel_y_[i] * time_step_;
            pos_z_[i] += vel_z_[i] * time_step_;

            // Boundary conditions
            float radius = radius_[i];
            if (pos_x_[i] < -domain_size_ / 2 + radius || pos_x_[i] > domain_size_ / 2 - radius) {
                vel_x_[i] *= -0.8f;
                pos_x_[i] = std::clamp(pos_x_[i], -domain_size_ / 2 + radius, domain_size_ / 2 - radius);
            }
            if (pos_y_[i] < -domain_size_ / 2 + radius || pos_y_[i] > domain_size_ / 2 - radius) {
                vel_y_[i] *= -0.8f;
                pos_y_[i] = std::clamp(pos_y_[i], -domain_size_ / 2 + radius, domain_size_ / 2 - radius);
            }
            if (pos_z_[i] < -domain_size_ / 2 + radius || pos_z_[i] > domain_size_ / 2 - radius) {
                vel_z_[i] *= -0.8f;
                pos_z_[i] = std::clamp(pos_z_[i], -domain_size_ / 2 + radius, domain_size_ / 2 - radius);
            }
        }
    }

    void run(int num_steps) {
        std::cout << "\nStarting DEM simulation..." << std::endl;
        std::cout << "Particles: " << num_particles_ << std::endl;
        std::cout << "Domain size: " << domain_size_ << " m" << std::endl;

        auto start_time = std::chrono::high_resolution_clock::now();

        for (int step = 0; step < num_steps; step++) {
            computeForces();
            integrateMotion();

            if (step % 100 == 0) {
                float total_energy = 0.0f;
                for (int i = 0; i < num_particles_; i++) {
                    float v_sq = vel_x_[i] * vel_x_[i] + vel_y_[i] * vel_y_[i] + vel_z_[i] * vel_z_[i];
                    total_energy += 0.5f * mass_[i] * v_sq;
                }

                std::cout << "Step " << std::setw(6) << step << " | Energy: " << std::setw(8) << std::fixed
                          << std::setprecision(3) << total_energy << " J" << std::endl;
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "\nSimulation completed in " << duration.count() << " ms" << std::endl;
        std::cout << "Performance: " << (num_particles_ * num_steps) / (duration.count() / 1000.0) / 1e6
                  << " M particle-steps/second" << std::endl;
    }
};

void printSystemInfo() {
    std::cout << "Material GPU Library - DEM Simulation Demo" << std::endl;
    std::cout << std::string(50, '=') << std::endl;

#ifdef MATERIAL_GPU_WITH_CUDA
    int device_count;
    cudaError_t error = cudaGetDeviceCount(&device_count);

    if (error == cudaSuccess && device_count > 0) {
        std::cout << "CUDA enabled - " << device_count << " GPU(s) available" << std::endl;

        for (int i = 0; i < device_count; i++) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            std::cout << "GPU " << i << ": " << prop.name << std::endl;
        }
    } else {
        std::cout << "CUDA runtime not available" << std::endl;
    }
#else
    std::cout << "CPU-only version (compile with CUDA for GPU acceleration)" << std::endl;
#endif

    std::cout << std::string(50, '=') << std::endl;
}

int main(int argc, char* argv[]) {
    printSystemInfo();

    int num_particles = 1000;
    float domain_size = 1.0f;
    int num_steps     = 500;

    if (argc >= 2)
        num_particles = std::atoi(argv[1]);
    if (argc >= 3)
        domain_size = std::atof(argv[2]);
    if (argc >= 4)
        num_steps = std::atoi(argv[3]);

    std::cout << "\nSimulation parameters:" << std::endl;
    std::cout << "Particles: " << num_particles << std::endl;
    std::cout << "Domain size: " << domain_size << " m" << std::endl;
    std::cout << "Steps: " << num_steps << std::endl;

    try {
        DEMSimulation simulation(num_particles, domain_size);
        simulation.run(num_steps);

        std::cout << "\nDemo completed successfully!" << std::endl;
        std::cout << "This demonstrates basic DEM simulation capabilities." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}