# Material GPU Library

A modern C++ material property library for DEM solvers with GPU acceleration support. Features advanced design patterns, compile-time optimization, and flexible material modeling capabilities.

## 🚀 Features

### Core Architecture
- **Modern C++ Design**: CRTP patterns, template metaprogramming, and strategy patterns
- **Builder Pattern**: Fluent interface for material construction
- **Factory Pattern**: Easy material creation with predefined types
- **Polymorphic Interface**: Runtime and compile-time material processing
- **Header-only Core**: High-performance template-based implementation

### Material Models
- **Elastic Properties**: Young's modulus, Poisson's ratio, density
- **Contact Models**: EEPA (Hertz-Mindlin), JKR adhesion, and None
- **Optional Properties**: Thermal conductivity, electromagnetic properties
- **Device Support**: CUDA-optimized device views (optional)

### Data Management
- **JSON Loading**: Flexible material definition from JSON files
- **Type Safety**: Compile-time checks for material capabilities
- **Validation**: Built-in material property validation
- **Serialization**: Export materials back to JSON format

## 📋 Requirements

- **C++17** compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- **CMake 3.20** or higher
- **CUDA Toolkit** (optional, for GPU acceleration)
- **Git** with submodule support

## 🛠️ Getting Started

### 1. Clone Repository

```bash
# Clone with submodules (required for nlohmann/json dependency)
git clone --recursive https://github.com/joe-cheung-cae/material-gpu.git
cd material-gpu

# Or if already cloned without --recursive:
git submodule update --init --recursive
```

### 2. Build Configuration

#### Basic CPU-only Build
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

#### With CUDA Support
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON
cmake --build build -j
```

#### Debug Build
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
```

### 3. Run Examples

```bash
# Run the comprehensive example program
./build/example

# Test with sample materials
./build/example < examples/materials.json
```

## 📖 Usage Examples

### Basic Material Creation

```cpp
#include "material/material_factory.hpp"
#include "material/json_loader.hpp"
using namespace matgpu;

// Using Builder Pattern
auto steel = MaterialBuilder()
    .elastic(2.1e11f, 0.29f, 7850.0f)  // Young's modulus, Poisson ratio, density
    .eepa_contact(1e6f, 5e5f, 0.3f, 0.15f)  // Contact stiffnesses and damping
    .thermal(50.0f, 500.0f)  // Thermal conductivity, heat capacity
    .id(1)
    .build_eepa();

// Using Factory Methods
auto concrete = MaterialFactory::create_eepa(
    1.0e7f, 0.25f, 2500.0f,  // Elastic properties
    5e4f, 2e4f, 0.2f, 0.1f   // EEPA parameters
);

auto polymer = MaterialFactory::create_jkr(
    5.0e6f, 0.4f, 1200.0f,   // Elastic properties  
    0.08f, 1.5e-4f           // JKR adhesion parameters
);
```

### JSON Material Loading

```cpp
// Load materials from JSON
Materials materials;
if (materials.load_from_file("examples/materials.json")) {
    std::cout << "Loaded " << materials.count() << " materials\n";
    
    // Access materials
    for (const auto& mat : materials.materials()) {
        std::cout << "Material: " << mat->material_type() 
                  << ", E = " << mat->young_modulus() << " Pa\n";
    }
} else {
    std::cerr << "Error: " << materials.last_error() << std::endl;
}
```

### Device Material Views (CUDA)

```cpp
#include "material/device_material_views.cuh"

// Create device-optimized material view
DeviceElasticProperties elastic{2.1e11f, 0.29f, 7850.0f};
DeviceEEPAContactView eepa{1e5f, 5e4f, 0.2f, 0.1f};
DeviceThermalView thermal{50.0f, 500.0f, true};

auto device_material = make_device_material_view(elastic, eepa, thermal);

// Compile-time property checks
if constexpr (decltype(device_material)::has_thermal) {
    float k = device_material.thermal_conductivity();
}

// Contact force computation
float force = device_material.normal_force(1e-3f, 0.1f);
```

## 📁 JSON Material Format

```json
{
    "materials": [
        {
            "type": "eepa",
            "name": "High_Strength_Steel",
            "elastic": {
                "young_modulus": 2.1e11,
                "poisson_ratio": 0.29,
                "density": 7850
            },
            "contact": {
                "type": "eepa",
                "kn": 1e6,
                "kt": 5e5,
                "gamma_n": 0.3,
                "gamma_t": 0.15
            },
            "thermal": {
                "conductivity": 50.0,
                "heat_capacity": 500
            }
        }
    ]
}
```

## 🧪 Testing

```bash
# Build and run all examples
cmake --build build
./build/example

# Test different material types
./build/example > output.log

# Validate JSON loading
cat examples/materials.json | ./build/example
```

## 🔧 Development

### Code Formatting

```bash
# Format all source files
./scripts/format.sh

# Check formatting
./scripts/check-format.sh

# Via CMake targets
cmake --build build --target format
cmake --build build --target check-format
```

### Project Structure

```
material_gpu/
├── include/material/           # Header files
│   ├── material_base.hpp      # Base material interfaces
│   ├── material_factory.hpp   # Factory pattern implementation
│   ├── contact_models.hpp     # Contact model definitions
│   ├── json_loader.hpp        # JSON loading functionality
│   └── device_*.cuh           # CUDA device headers
├── src/                       # Source files
│   ├── json_loader.cpp        # JSON implementation
│   └── material_lib.cpp       # Library core
├── examples/                  # Example programs and data
│   ├── example.cpp           # Comprehensive examples
│   └── materials.json        # Sample material definitions
├── third_party/              # External dependencies
│   └── nlohmann_json/        # JSON library (git submodule)
└── CMakeLists.txt            # Build configuration
```

## 📚 Documentation

- [Architecture Guide](ARCHITECTURE_OPTIMIZATION_REPORT.md) - Detailed design patterns and optimization strategies
- [Migration Guide](MIGRATION_GUIDE.md) - Upgrading from legacy versions
- [Examples Directory](examples/) - Comprehensive usage examples

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏗️ Build Options

| Option             | Default   | Description                                 |
| ------------------ | --------- | ------------------------------------------- |
| `ENABLE_CUDA`      | `OFF`     | Enable CUDA support for GPU acceleration    |
| `CMAKE_BUILD_TYPE` | `Release` | Build type (Release, Debug, RelWithDebInfo) |

## 🐛 Troubleshooting

### Common Issues

1. **Submodule not found**: Run `git submodule update --init --recursive`
2. **CUDA compilation errors**: Ensure CUDA toolkit is installed and `ENABLE_CUDA=ON`
3. **Missing nlohmann/json**: Verify git submodules are properly initialized

### Performance Tips

- Use `Release` build for production
- Enable CUDA for large-scale simulations
- Use device views for GPU kernels
- Leverage compile-time optimizations with template specializations
