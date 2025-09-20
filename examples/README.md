# Examples

This folder contains example programs and data for the material-gpu library.

## Binaries

- `example` (from `example.cu`): CUDA-enabled comprehensive example
- `basic_demo`: CPU-only quick demo
- `advanced_gpu_demo`: Larger demo; accepts optional arguments
- `gpu_benchmark`: CPU vs GPU performance benchmark
- `materials.json`: Sample materials dataset

## Build

CUDA is enabled by default (required for `example`):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON
cmake --build build -j
```

CPU-only override (works for demos and tests):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=OFF
cmake --build build -j
```

## Run

```bash
# CPU demos
./build/basic_demo
./build/advanced_gpu_demo 200 0.5 50

# CUDA example (requires -DENABLE_CUDA=ON)
./build/example
./build/example < examples/materials.json

# Benchmark
./build/gpu_benchmark
```
