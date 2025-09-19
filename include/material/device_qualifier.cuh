#pragma once

// Unified device/host inline qualifier usable from both CUDA and host
// compilers.
#if defined(__CUDACC__)
#define DEVICE_QUALIFIER __device__ __forceinline__
#else
#define DEVICE_QUALIFIER inline
#endif
