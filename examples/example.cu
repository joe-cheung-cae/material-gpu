// CUDA example file - compiled with nvcc
#ifdef __CUDACC__
// When compiling with nvcc, ensure CUDA runtime is available
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif

#include "material/contact_models.hpp"
#include "material/device_material_views.cuh"
#include "material/json_loader.hpp"
#include "material/material_factory.hpp"
#include "material/property_mixins.hpp"

#ifdef MATERIAL_GPU_WITH_CUDA
#include "material/cuda_kernels.cuh"
#include "material/gpu_memory_manager.hpp"
#endif

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

using namespace matgpu;

// Example: Creating materials using the Builder pattern
void example_builder_pattern() {
    std::cout << "=== Builder Pattern Example ===" << std::endl;

    // Create a standard elastic material
    auto standard_material = MaterialBuilder()
                                 .elastic(2.0e11f, 0.3f, 7800.0f) // Steel properties
                                 .no_contact()
                                 .id(1)
                                 .build();

    std::cout << "Created standard material with Young's modulus: " << standard_material->young_modulus() << " Pa"
              << std::endl;

    // Create an EEPA material with thermal properties
    auto eepa_material = MaterialBuilder()
                             .elastic(1.0e7f, 0.25f, 2500.0f)      // Concrete-like properties
                             .eepa_contact(1e5f, 5e4f, 0.2f, 0.1f) // EEPA contact parameters
                             .thermal(1.5f, 880.0f)                // Thermal conductivity & heat capacity
                             .id(2)
                             .build_eepa();

    std::cout << "Created EEPA material with thermal properties" << std::endl;
    std::cout << "  EEPA kn: " << eepa_material->eepa_contact_model()->kn() << std::endl;
    std::cout << "  Thermal conductivity: " << eepa_material->thermal()->conductivity() << " W/(m·K)" << std::endl;

    // Create a JKR material with electromagnetic properties
    auto jkr_material = MaterialBuilder()
                            .elastic(5.0e6f, 0.4f, 1200.0f)              // Polymer-like properties
                            .jkr_contact(0.05f, 1e-4f)                   // JKR adhesion parameters
                            .electromagnetic(8.85e-12f, 1.26e-6f, 1e-8f) // EM properties
                            .id(3)
                            .build_jkr();

    std::cout << "Created JKR material with EM properties" << std::endl;
    std::cout << "  JKR work of adhesion: " << jkr_material->jkr_contact_model()->work_of_adhesion() << " J/m²"
              << std::endl;
    std::cout << "  Permittivity: " << jkr_material->electromagnetic()->permittivity() << " F/m" << std::endl;
}

// Example: Using the Factory pattern
void example_factory_pattern() {
    std::cout << "\n=== Factory Pattern Example ===" << std::endl;

    // Create materials using factory methods
    auto material1 = MaterialFactory::create_standard(2.1e11f, 0.29f, 7850.0f);
    auto material2 = MaterialFactory::create_eepa(1.0e7f, 0.25f, 2500.0f, 1e5f, 5e4f, 0.2f, 0.1f);
    auto material3 = MaterialFactory::create_jkr(5.0e6f, 0.4f, 1200.0f, 0.05f, 1e-4f);

    std::cout << "Created materials using factory methods:" << std::endl;
    std::cout << "  Material 1 type: " << material1->material_type() << std::endl;
    std::cout << "  Material 2 type: " << material2->material_type() << std::endl;
    std::cout << "  Material 3 type: " << material3->material_type() << std::endl;
}

// Example: Loading materials from JSON using enhanced loader
void example_json_loading() {
    std::cout << "\n=== Enhanced JSON Loading Example ===" << std::endl;

    // JSON string with multiple materials
    std::string json_text = R"({
        "materials": [
            {
                "type": "eepa",
                "elastic": {"young_modulus": 2.1e11, "poisson_ratio": 0.29, "density": 7850},
                "contact": {"type": "eepa", "kn": 1e6, "kt": 5e5, "gamma_n": 0.3, "gamma_t": 0.15},
                "thermal": {"conductivity": 50.0, "heat_capacity": 500}
            },
            {
                "type": "jkr",
                "elastic": {"young_modulus": 1.0e7, "poisson_ratio": 0.25, "density": 2500},
                "contact": {"type": "jkr", "work_of_adhesion": 0.08, "contact_radius0": 1.5e-4},
                "electromagnetic": {"permittivity": 8.85e-12, "permeability": 1.26e-6, "conductivity": 1e-6}
            },
            {
                "type": "standard",
                "elastic": {"young_modulus": 3.0e9, "poisson_ratio": 0.35, "density": 1200},
                "contact": {"type": "none"}
            }
        ]
    })";

    Materials materials;
    if (materials.load_from_json_text(json_text)) {
        std::cout << "Successfully loaded " << materials.count() << " materials from JSON" << std::endl;

        for (size_t i = 0; i < materials.materials().size(); ++i) {
            const auto& mat = materials.materials()[i];
            std::cout << "  Material " << i << ": " << mat->material_type() << ", E = " << mat->young_modulus() << " Pa"
                      << std::endl;

            if (mat->thermal_properties()) {
                std::cout << "    Has thermal properties" << std::endl;
            }
            if (mat->electromagnetic_properties()) {
                std::cout << "    Has electromagnetic properties" << std::endl;
            }
        }
    } else {
        std::cout << "Failed to load materials: " << materials.last_error() << std::endl;
    }
}

// Example: Template-based device material views
void example_device_views() {
    std::cout << "\n=== Device Material Views Example ===" << std::endl;

    // Create device elastic properties
    DeviceElasticProperties elastic{2.1e11f, 0.29f, 7850.0f};

    // Create EEPA contact model for device
    DeviceEEPAContactView eepa_contact{1e5f, 5e4f, 0.2f, 0.1f};

    // Create thermal properties for device
    DeviceThermalView thermal{50.0f, 500.0f, true};

    // Create device material view with compile-time type safety
    auto device_material = make_host_material_view(elastic, eepa_contact, thermal);

    std::cout << "Device material properties:" << std::endl;
    std::cout << "  Young's modulus: " << device_material.young_modulus() << " Pa" << std::endl;
    std::cout << "  Density: " << device_material.density() << " kg/m³" << std::endl;

    // Compile-time checks for optional properties
    if constexpr (decltype(device_material)::has_thermal) {
        std::cout << "  Thermal conductivity: " << device_material.thermal_conductivity() << " W/(m·K)" << std::endl;
    }

    // Contact force computation
    float normal_force = device_material.normal_force(1e-3f, 0.1f); // 1mm overlap, 0.1 m/s velocity
    std::cout << "  Normal contact force: " << normal_force << " N" << std::endl;
}

// Example: Using template metaprogramming for compile-time optimization
template <typename MaterialView> void process_material_compile_time(const MaterialView& material) {
    std::cout << "\n=== Compile-Time Material Processing ===" << std::endl;

    std::cout << "Material properties (compile-time optimized):" << std::endl;
    std::cout << "  Has thermal: " << material_has_thermal_v<MaterialView> << std::endl;
    std::cout << "  Has EM: " << material_has_electromagnetic_v<MaterialView> << std::endl;
    std::cout << "  Is EEPA: " << material_is_eepa_v<MaterialView> << std::endl;
    std::cout << "  Is JKR: " << material_is_jkr_v<MaterialView> << std::endl;

    // Conditional compilation based on material properties
    if constexpr (material_has_thermal_v<MaterialView>) {
        std::cout << "  Processing thermal calculations..." << std::endl;
        // Thermal-specific code would go here
    }

    if constexpr (material_is_eepa_v<MaterialView>) {
        std::cout << "  Using EEPA contact model..." << std::endl;
        // EEPA-specific code would go here
    }
}

// Example: Polymorphic material processing
void example_polymorphic_processing() {
    std::cout << "\n=== Polymorphic Material Processing Example ===" << std::endl;

    // Create different types of materials
    std::vector<std::unique_ptr<IMaterial>> materials;

    materials.push_back(MaterialFactory::create_standard(2.1e11f, 0.29f, 7850.0f));
    materials.push_back(MaterialFactory::create_eepa(1.0e7f, 0.25f, 2500.0f, 1e5f, 5e4f, 0.2f, 0.1f));
    materials.push_back(MaterialFactory::create_jkr(5.0e6f, 0.4f, 1200.0f, 0.05f, 1e-4f));

    // Process all materials polymorphically
    for (const auto& material : materials) {
        std::cout << "Processing " << material->material_type() << " material:" << std::endl;
        std::cout << "  Young's modulus: " << material->young_modulus() << " Pa" << std::endl;
        std::cout << "  Contact model: " << material->contact_model().model_name() << std::endl;

        // Polymorphic contact force calculation
        float force = material->contact_model().compute_normal_force(1e-3f, 0.1f);
        std::cout << "  Normal force (1mm overlap): " << force << " N" << std::endl;

        // Check for optional properties
        if (material->thermal_properties()) {
            std::cout << "  Has thermal properties" << std::endl;
        }
        if (material->electromagnetic_properties()) {
            std::cout << "  Has electromagnetic properties" << std::endl;
        }
        std::cout << std::endl;
    }
}

// Example: Fluent interface and method chaining
void example_fluent_interface() {
    std::cout << "\n=== Fluent Interface Example ===" << std::endl;

    // Create complex material using fluent interface
    auto complex_material = MaterialBuilder()
                                .elastic(1.5e10f, 0.28f, 2400.0f)
                                .eepa_contact(2e5f, 1e5f, 0.25f, 0.12f)
                                .thermal(2.5f, 920.0f)
                                .electromagnetic(8.85e-12f, 1.26e-6f, 1e-7f)
                                .id(42)
                                .build();

    std::cout << "Created complex material with all properties:" << std::endl;
    std::cout << "  ID: " << complex_material->material_id() << std::endl;
    std::cout << "  Type: " << complex_material->material_type() << std::endl;
    std::cout << "  All optional properties included" << std::endl;
}

#ifdef MATERIAL_GPU_WITH_CUDA
// Example: GPU acceleration demonstration
void example_gpu_acceleration() {
    std::cout << "\n=== GPU Acceleration Example ===" << std::endl;

    try {
        // Create test materials
        std::vector<DeviceEEPAMaterial> gpu_materials = {
            DeviceEEPAMaterial{{2.1e11f, 0.29f, 7850.0f}, {1e6f, 5e5f, 0.3f, 0.15f}}, // Steel
            DeviceEEPAMaterial{{1.0e7f, 0.25f, 2500.0f}, {5e4f, 2e4f, 0.2f, 0.1f}},   // Concrete
            DeviceEEPAMaterial{{5.0e6f, 0.4f, 1200.0f}, {3e4f, 1e4f, 0.15f, 0.08f}}   // Polymer
        };

        // Create test particle data
        const int num_particles = 1000;
        std::vector<float> pos_x(num_particles), pos_y(num_particles), pos_z(num_particles);
        std::vector<float> radii(num_particles);
        std::vector<int> material_ids(num_particles);

        // Generate random particle positions
        for (int i = 0; i < num_particles; ++i) {
            pos_x[i]        = static_cast<float>(rand()) / RAND_MAX * 10.0f - 5.0f;
            pos_y[i]        = static_cast<float>(rand()) / RAND_MAX * 10.0f - 5.0f;
            pos_z[i]        = static_cast<float>(rand()) / RAND_MAX * 10.0f - 5.0f;
            radii[i]        = 0.1f + static_cast<float>(rand()) / RAND_MAX * 0.2f;
            material_ids[i] = i % 3; // Cycle through material types
        }

        std::cout << "Generated " << num_particles << " particles with 3 material types" << std::endl;

        // GPU Memory Management Example
        std::cout << "Setting up GPU memory management..." << std::endl;
        GpuMemoryManager memory_manager;

        // Create GPU particle data container
        GpuParticleData gpu_particles(pos_x, pos_y, pos_z, radii, memory_manager);
        std::cout << "Uploaded particle data to GPU (" << (6 * num_particles * sizeof(float)) / 1024 << " KB)"
                  << std::endl;

        // Create GPU material container
        GpuMaterialContainer<DeviceEEPAMaterial> gpu_materials_container(gpu_materials, material_ids, memory_manager);
        std::cout << "Uploaded material data to GPU" << std::endl;

        // GPU Contact Force Calculation
        std::cout << "Running GPU contact force calculation..." << std::endl;
        CudaMaterialProcessor gpu_processor;
        std::vector<float> forces_x, forces_y, forces_z;

        auto start = std::chrono::high_resolution_clock::now();
        gpu_processor.compute_contact_forces(gpu_materials, pos_x, pos_y, pos_z, radii, material_ids, forces_x,
                                             forces_y, forces_z);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        // Calculate total force magnitude
        float total_force = 0.0f;
        for (int i = 0; i < num_particles; ++i) {
            total_force += std::sqrt(forces_x[i] * forces_x[i] + forces_y[i] * forces_y[i] + forces_z[i] * forces_z[i]);
        }

        std::cout << "GPU computation completed in " << duration.count() << " ms" << std::endl;
        std::cout << "Total force magnitude: " << total_force << std::endl;
        std::cout << "Average force per particle: " << total_force / num_particles << std::endl;

        // Device property demonstration
        std::cout << "\n--- GPU Device Material Properties ---" << std::endl;
        auto device_mat = make_host_material_view(DeviceElasticProperties{2.1e11f, 0.29f, 7850.0f},
                                                  DeviceEEPAContactView{1e6f, 5e5f, 0.3f, 0.15f});

        std::cout << "Wave speeds (computed on device if CUDA available):" << std::endl;
        std::cout << "  Longitudinal: " << device_mat.elastic().wave_speed_longitudinal() << " m/s" << std::endl;
        std::cout << "  Shear: " << device_mat.elastic().wave_speed_shear() << " m/s" << std::endl;

        // Advanced GPU features demo
        std::cout << "\n--- Advanced GPU Features ---" << std::endl;
        using namespace gpu_compute;

#ifdef __CUDACC__
        auto elastic_constants = compute_elastic_constants_vec(device_mat.elastic());
        std::cout << "Elastic constants (vectorized): λ=" << elastic_constants.x << ", μ=" << elastic_constants.y
                  << ", K=" << elastic_constants.z << std::endl;
#else
        auto elastic_constants = compute_elastic_constants_vec(device_mat.elastic());
        std::cout << "Elastic constants (host): λ=" << elastic_constants.x << ", μ=" << elastic_constants.y
                  << ", K=" << elastic_constants.z << std::endl;
#endif

        std::cout << "GPU acceleration example completed successfully!" << std::endl;

    } catch (const std::exception& e) {
        std::cout << "GPU Error: " << e.what() << std::endl;
        std::cout << "Note: Ensure CUDA is properly installed and GPU is available" << std::endl;
    }
}
#endif

int main() {
    try {
        std::cout << "Material GPU Library - Enhanced Architecture Examples" << std::endl;
        std::cout << "========================================================" << std::endl;

        example_builder_pattern();
        example_factory_pattern();
        example_json_loading();
        example_device_views();
        example_polymorphic_processing();
        example_fluent_interface();

#ifdef MATERIAL_GPU_WITH_CUDA
        example_gpu_acceleration();
#else
        std::cout << "\n=== CUDA GPU Support ===" << std::endl;
        std::cout << "GPU acceleration is NOT enabled." << std::endl;
        std::cout << "To enable GPU support, compile with: cmake -DENABLE_CUDA=ON" << std::endl;
#endif

        // Example of compile-time optimization
        DeviceElasticProperties elastic{1e10f, 0.3f, 2000.0f};
        DeviceEEPAContactView contact{1e5f, 5e4f, 0.2f, 0.1f};
        DeviceThermalView thermal{10.0f, 800.0f, true};
        auto device_mat = make_host_material_view(elastic, contact, thermal);
        process_material_compile_time(device_mat);

        std::cout << "\n========================================================" << std::endl;
        std::cout << "All examples completed successfully!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}