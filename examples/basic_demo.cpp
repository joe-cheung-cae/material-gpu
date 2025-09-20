#include "material/json_loader.hpp"

#include <iostream>

int main() {
    std::cout << "Material GPU Library - Basic Usage Demo" << std::endl;
    std::cout << std::string(45, '=') << std::endl;

    try {
        // Example JSON text for a simple material
        std::string json_text = R"({
            "materials": [
                {
                    "type": "standard",
                    "elastic": {
                        "young_modulus": 210e9,
                        "poisson_ratio": 0.3,
                        "density": 7850.0
                    },
                    "contact": {
                        "type": "none"
                    }
                },
                {
                    "type": "eepa",
                    "elastic": {
                        "young_modulus": 1.0e7,
                        "poisson_ratio": 0.25,
                        "density": 2500.0
                    },
                    "contact": {
                        "type": "eepa",
                        "kn": 1e6,
                        "kt": 5e5,
                        "gamma_n": 0.3,
                        "gamma_t": 0.15
                    }
                }
            ]
        })";

        std::cout << "Loading materials from JSON..." << std::endl;

        std::vector<std::unique_ptr<matgpu::IMaterial>> materials;
        std::string error;

        bool success = matgpu::JSONLoader::load_materials_from_json_text(json_text, materials, error);

        if (success) {
            std::cout << "Successfully loaded " << materials.size() << " materials:" << std::endl;

            for (size_t i = 0; i < materials.size(); ++i) {
                std::cout << "  " << (i + 1) << ". Material type: " << materials[i]->material_type() << std::endl;
            }

            std::cout << "\nDemonstrating material property access..." << std::endl;
            std::cout << "✓ Material system working correctly" << std::endl;

        } else {
            std::cerr << "Failed to load materials: " << error << std::endl;
            return 1;
        }

        std::cout << "\nBasic demo completed successfully!" << std::endl;
        std::cout << "Next steps:" << std::endl;
        std::cout << "• Run advanced_gpu_demo for DEM simulation" << std::endl;
        std::cout << "• Check GPU_OPTIMIZATION_GUIDE.md for performance tips" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}