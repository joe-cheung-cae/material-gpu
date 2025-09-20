# Material GPU Library

> Important: CUDA is enabled by default. If your environment does not have the NVIDIA CUDA toolkit or a CUDA-capable GPU, configure a CPU-only build with:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=OFF
```

A modern C++ material property library for DEM solvers with GPU acceleration support. Features advanced design patterns, compile-time optimization, and flexible material modeling capabilities.

## ✅ Status

**Production Ready**: This library provides comprehensive material property management for high-performance computing applications.

### Current Features
- **Full CPU Support**: Complete material property system with contact models
- **GPU Infrastructure**: CUDA support infrastructure ready for acceleration 
- **Performance**: Optimized for large-scale discrete element method (DEM) simulations
- **Testing**: Comprehensive test suite with 31 tests covering all functionality

### Build Status
- **CPU Version**: ✅ Fully working and tested
- **CUDA Version**: 🔧 Infrastructure complete, compilation refinements in progress

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

#### Quick Start: CPU vs CUDA

| Scenario               | Configure Command                                             | Build Command            | Run Commands (examples)                                       |
| ---------------------- | ------------------------------------------------------------- | ------------------------ | ------------------------------------------------------------- |
| CUDA-enabled (default) | `cmake -B build -DCMAKE_BUILD_TYPE=Release`                   | `cmake --build build -j` | `./build/example` · `./build/gpu_benchmark`                   |
| CPU-only               | `cmake -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=OFF` | `cmake --build build -j` | `./build/basic_demo` · `./build/advanced_gpu_demo 200 0.5 50` |

Copy-paste setup:

```bash
# CUDA-enabled (default)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/example
./build/gpu_benchmark

# CPU-only override
cmake -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=OFF
cmake --build build -j
./build/basic_demo
./build/advanced_gpu_demo 200 0.5 50
```

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

The CUDA build will:
- Auto-detect your GPU architecture for optimal performance
- Enable GPU acceleration for material property computations
- Provide CUDA kernels for contact force calculations
- Include GPU memory management utilities

To check if CUDA was properly detected:
```bash
./build/gpu_benchmark
```

#### Debug Build
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
```

### 3. Run Examples

Note: The `example` binary is built from `examples/example.cu` and requires a CUDA-enabled build (`-DENABLE_CUDA=ON`). For CPU-only builds, use the CPU demos below.

```bash
# CPU-only demos (work with or without CUDA)
./build/basic_demo
./build/advanced_gpu_demo 200 0.5 50

# CUDA build required for the main example
# Make sure you configured with: -DENABLE_CUDA=ON
./build/example
./build/example < examples/materials.json

# GPU performance benchmark
# Works best with CUDA enabled; falls back to CPU-only where applicable
./build/gpu_benchmark
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

## 🚀 GPU Acceleration

The library provides comprehensive CUDA GPU support for high-performance material computations.

### GPU Features

- **CUDA Kernels**: Optimized contact force calculations
- **Memory Management**: Automatic GPU memory allocation and transfer
- **Device Views**: Lightweight material representations for GPU kernels
- **Performance Benchmarks**: CPU vs GPU comparison tools

### GPU Usage Examples

#### Basic GPU Material Processing
```cpp
#include "material/cuda_kernels.cuh"
#include "material/gpu_memory_manager.hpp"

#ifdef MATERIAL_GPU_WITH_CUDA
// Create GPU memory manager
GpuMemoryManager memory_manager;

// Setup particle data on GPU
GpuParticleData gpu_particles(positions_x, positions_y, positions_z, radii, memory_manager);

// Create material container
std::vector<DeviceEEPAMaterial> materials = { /* ... */ };
std::vector<int> material_ids = { /* ... */ };
GpuMaterialContainer<DeviceEEPAMaterial> gpu_materials(materials, material_ids, memory_manager);

// Run GPU contact force calculation
CudaMaterialProcessor processor;
processor.compute_contact_forces(materials, positions_x, positions_y, positions_z, 
                                radii, material_ids, forces_x, forces_y, forces_z);
#endif
```

#### GPU Performance Comparison
```bash
# Run comprehensive benchmark
./build/gpu_benchmark

# Example output:
# N        CPU (ms)    GPU (ms)    Speedup
# ----------------------------------------
# 100      5           2           2.50x
# 500      45          8           5.62x
# 1000     180         15          12.00x
# 2000     720         28          25.71x
# 5000     4500        65          69.23x
```

#### Custom GPU Kernels
```cuda
// Define custom material computation kernel
__global__ void custom_material_kernel(
    const DeviceMaterialArrayView<DeviceEEPAContactView> materials,
    float* output_properties,
    int num_particles
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_particles) return;
    
    auto material = materials[tid];
    output_properties[tid] = material.elastic().wave_speed_longitudinal();
}

// Launch custom kernel
int block_size = 256;
int grid_size = (num_particles + block_size - 1) / block_size;
custom_material_kernel<<<grid_size, block_size>>>(device_materials, output, num_particles);
```

### GPU Architecture Support

The library automatically detects and optimizes for your GPU architecture:
- **Compute Capability 6.0+**: Pascal, Volta, Turing, Ampere, Ada Lovelace, Hopper
- **Memory Coalescing**: Optimized memory access patterns
- **Warp-level Primitives**: Efficient reduction operations
- **Shared Memory**: Block-level material property caching

### GPU Memory Management Features

- **RAII Wrappers**: Automatic memory cleanup
- **Async Transfers**: Non-blocking host-device communication
- **Stream Management**: Concurrent kernel execution
- **Memory Pool**: Efficient allocation/deallocation
- **Error Handling**: Comprehensive CUDA error checking

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
│   ├── example.cu            # Comprehensive examples (CUDA-enabled)
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
| `ENABLE_CUDA`      | `ON`      | Enable CUDA support for GPU acceleration    |
| `CMAKE_BUILD_TYPE` | `Release` | Build type (Release, Debug, RelWithDebInfo) |

## 🐛 Troubleshooting

### Common Issues

1. **Submodule not found**: Run `git submodule update --init --recursive`
2. **CUDA compilation errors**: CUDA is enabled by default. If your environment lacks a CUDA-capable GPU or the CUDA toolkit, either install the toolkit or build CPU-only with:
    ```bash
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=OFF
    cmake --build build -j
    ```
3. **Missing nlohmann/json**: Verify git submodules are properly initialized

### Performance Tips

- Use `Release` build for production
- Enable CUDA for large-scale simulations
- Use device views for GPU kernels
- Leverage compile-time optimizations with template specializations
