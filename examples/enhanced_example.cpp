#include "material/device_material_views.cuh"
#include "material/enhanced_json_loader.hpp"
#include "material/material_factory.hpp"

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
                "elastic": {"young": 2.1e11, "poisson": 0.29, "density": 7850},
                "contact_model": "EEPA",
                "eepa": {"kn": 1e6, "kt": 5e5, "gamma_n": 0.3, "gamma_t": 0.15},
                "thermal": {"conductivity": 50.0, "heat_capacity": 500}
            },
            {
                "elastic": {"young": 1.0e7, "poisson": 0.25, "density": 2500},
                "contact_model": "JKR",
                "jkr": {"work_of_adhesion": 0.08, "contact_radius0": 1.5e-4},
                "em": {"permittivity": 8.85e-12, "permeability": 1.26e-6, "conductivity": 1e-6}
            },
            {
                "elastic": {"young": 3.0e9, "poisson": 0.35, "density": 1200},
                "contact_model": "None"
            }
        ]
    })";

    MaterialsV2 materials;
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
    auto device_material = make_device_material_view(elastic, eepa_contact, thermal);

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

        // Example of compile-time optimization
        DeviceElasticProperties elastic{1e10f, 0.3f, 2000.0f};
        DeviceEEPAContactView contact{1e5f, 5e4f, 0.2f, 0.1f};
        DeviceThermalView thermal{10.0f, 800.0f, true};
        auto device_mat = make_device_material_view(elastic, contact, thermal);
        process_material_compile_time(device_mat);

        std::cout << "\n========================================================" << std::endl;
        std::cout << "All examples completed successfully!" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}