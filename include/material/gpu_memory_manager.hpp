#pragma once

#include "material/device_material_views.cuh"

#ifdef MATERIAL_GPU_WITH_CUDA
#include <cuda_runtime.h>
#endif

#include <stdexcept>
#include <type_traits>
#include <vector>

namespace matgpu {

#ifdef MATERIAL_GPU_WITH_CUDA

// Forward declaration so GpuMemoryManager can reference it
template <typename T> class UniqueDeviceBuffer;

/**
 * @brief CUDA memory management utilities for material data
 */
class GpuMemoryManager {
  private:
    cudaStream_t stream_;
    bool owns_stream_;

  public:
    GpuMemoryManager() : owns_stream_(true) { cudaStreamCreate(&stream_); }

    explicit GpuMemoryManager(cudaStream_t stream) : stream_(stream), owns_stream_(false) {}

    ~GpuMemoryManager() {
        if (owns_stream_) {
            cudaStreamDestroy(stream_);
        }
    }

    // No copy constructor/assignment
    GpuMemoryManager(const GpuMemoryManager&)            = delete;
    GpuMemoryManager& operator=(const GpuMemoryManager&) = delete;

    // Move constructor/assignment
    GpuMemoryManager(GpuMemoryManager&& other) noexcept : stream_(other.stream_), owns_stream_(other.owns_stream_) {
        other.owns_stream_ = false;
    }

    GpuMemoryManager& operator=(GpuMemoryManager&& other) noexcept {
        if (this != &other) {
            if (owns_stream_) {
                cudaStreamDestroy(stream_);
            }
            stream_            = other.stream_;
            owns_stream_       = other.owns_stream_;
            other.owns_stream_ = false;
        }
        return *this;
    }

    /**
     * @brief Allocate device memory for array data
     */
    template <typename T> T* allocate_device(size_t count) {
        T* ptr;
        cudaError_t err = cudaMalloc(&ptr, count * sizeof(T));
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA allocation failed: " + std::string(cudaGetErrorString(err)));
        }
        return ptr;
    }

    /**
     * @brief Free device memory
     */
    template <typename T> void free_device(T* ptr) { cudaFree(ptr); }

    /**
     * @brief Copy data from host to device asynchronously
     */
    template <typename T> void copy_to_device_async(T* device_ptr, const T* host_ptr, size_t count) {
        cudaMemcpyAsync(device_ptr, host_ptr, count * sizeof(T), cudaMemcpyHostToDevice, stream_);
    }

    /**
     * @brief Copy data from device to host asynchronously
     */
    template <typename T> void copy_to_host_async(T* host_ptr, const T* device_ptr, size_t count) {
        cudaMemcpyAsync(host_ptr, device_ptr, count * sizeof(T), cudaMemcpyDeviceToHost, stream_);
    }

    /**
     * @brief Set device memory to zero asynchronously
     */
    template <typename T> void zero_device_async(T* device_ptr, size_t count) {
        cudaMemsetAsync(device_ptr, 0, count * sizeof(T), stream_);
    }

    /**
     * @brief Synchronize stream
     */
    void synchronize() { cudaStreamSynchronize(stream_); }

    cudaStream_t get_stream() const { return stream_; }

    // Convenience non-async wrappers expected by some tests
    template <typename T> void copy_to_device(T* device_ptr, const std::vector<T>& host_vec) {
        copy_to_device_async(device_ptr, host_vec.data(), host_vec.size());
        synchronize();
    }

    template <typename T> void copy_to_host(std::vector<T>& host_vec, const T* device_ptr, size_t count) {
        host_vec.resize(count);
        copy_to_host_async(host_vec.data(), device_ptr, count);
        synchronize();
    }

    // Overloads for UniqueDeviceBuffer (accepts decayed types)
    template <typename U> void copy_to_device(UniqueDeviceBuffer<U>& buf, const std::vector<U>& host_vec) {
        copy_to_device(buf.get(), host_vec);
    }

    template <typename U> void copy_to_host(std::vector<U>& host_vec, const UniqueDeviceBuffer<U>& buf, size_t count) {
        copy_to_host(host_vec, buf.get(), count);
    }

    // Member helper to allocate RAII buffers used in tests
    template <typename T> UniqueDeviceBuffer<typename std::remove_reference<T>::type> allocate(size_t count) {
        using BaseT = typename std::remove_reference<T>::type;
        return UniqueDeviceBuffer<BaseT>(allocate_device<BaseT>(count), count, *this);
    }
};

/**
 * @brief RAII wrapper for GPU memory
 */
template <typename T> class GpuBuffer {
  private:
    T* device_ptr_;
    size_t size_;
    GpuMemoryManager* manager_;

  public:
    GpuBuffer(size_t size, GpuMemoryManager& manager) : size_(size), manager_(&manager) {
        device_ptr_ = manager_->template allocate_device<T>(size_);
    }

    ~GpuBuffer() {
        if (device_ptr_) {
            manager_->free_device(device_ptr_);
        }
    }

    // No copy
    GpuBuffer(const GpuBuffer&)            = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;

    // Move semantics
    GpuBuffer(GpuBuffer&& other) noexcept
        : device_ptr_(other.device_ptr_), size_(other.size_), manager_(other.manager_) {
        other.device_ptr_ = nullptr;
    }

    GpuBuffer& operator=(GpuBuffer&& other) noexcept {
        if (this != &other) {
            if (device_ptr_) {
                manager_->free_device(device_ptr_);
            }
            device_ptr_       = other.device_ptr_;
            size_             = other.size_;
            manager_          = other.manager_;
            other.device_ptr_ = nullptr;
        }
        return *this;
    }

    T* data() { return device_ptr_; }
    const T* data() const { return device_ptr_; }
    size_t size() const { return size_; }

    // Compatibility shim for tests using .get()
    T* get() { return device_ptr_; }
    const T* get() const { return device_ptr_; }

    void copy_from_host_async(const std::vector<T>& host_data) {
        if (host_data.size() != size_) {
            throw std::runtime_error("Size mismatch in copy_from_host_async");
        }
        manager_->copy_to_device_async(device_ptr_, host_data.data(), size_);
    }

    void copy_to_host_async(std::vector<T>& host_data) {
        host_data.resize(size_);
        manager_->copy_to_host_async(host_data.data(), device_ptr_, size_);
    }

    void zero_async() { manager_->zero_device_async(device_ptr_, size_); }
};

// Lightweight RAII unique device buffer with .get() interface used in tests
template <typename T> class UniqueDeviceBuffer {
  private:
    using BaseT  = typename std::remove_reference<T>::type;
    BaseT* ptr_  = nullptr;
    size_t size_ = 0;
    GpuMemoryManager* manager_;

  public:
    UniqueDeviceBuffer() = default;
    UniqueDeviceBuffer(BaseT* ptr, size_t n, GpuMemoryManager& mgr) : ptr_(ptr), size_(n), manager_(&mgr) {}
    ~UniqueDeviceBuffer() {
        if (ptr_) {
            manager_->free_device(ptr_);
        }
    }
    UniqueDeviceBuffer(const UniqueDeviceBuffer&)            = delete;
    UniqueDeviceBuffer& operator=(const UniqueDeviceBuffer&) = delete;
    UniqueDeviceBuffer(UniqueDeviceBuffer&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_), manager_(other.manager_) {
        other.ptr_ = nullptr;
    }
    UniqueDeviceBuffer& operator=(UniqueDeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_)
                manager_->free_device(ptr_);
            ptr_       = other.ptr_;
            size_      = other.size_;
            manager_   = other.manager_;
            other.ptr_ = nullptr;
        }
        return *this;
    }
    BaseT* get() { return ptr_; }
    const BaseT* get() const { return ptr_; }
    size_t size() const { return size_; }
};

// Convenience APIs matching tests
template <typename T> UniqueDeviceBuffer<T> allocate(GpuMemoryManager& mgr, size_t count) {
    return UniqueDeviceBuffer<T>(mgr.template allocate_device<T>(count), count, mgr);
}

template <typename T> UniqueDeviceBuffer<T> allocate(GpuMemoryManager& mgr, size_t count, T init_value) {
    auto buf = allocate<T>(mgr, count);
    std::vector<T> host(count, init_value);
    mgr.copy_to_device(buf.get(), host);
    return buf;
}

/**
 * @brief Specialized container for material data on GPU
 */
template <typename MaterialViewType> class GpuMaterialContainer {
  private:
    GpuBuffer<MaterialViewType> materials_;
    GpuBuffer<DeviceElasticProperties> elastic_props_;
    GpuBuffer<int> material_ids_;
    size_t num_materials_;
    size_t num_particles_;
    GpuMemoryManager& manager_;

  public:
    GpuMaterialContainer(const std::vector<MaterialViewType>& host_materials, const std::vector<int>& host_material_ids,
                         GpuMemoryManager& manager)
        : materials_(host_materials.size(), manager), elastic_props_(host_materials.size(), manager),
          material_ids_(host_material_ids.size(), manager), num_materials_(host_materials.size()),
          num_particles_(host_material_ids.size()), manager_(manager) {

        // Upload material data
        materials_.copy_from_host_async(host_materials);
        material_ids_.copy_from_host_async(host_material_ids);

        // Extract elastic properties
        std::vector<DeviceElasticProperties> host_elastic;
        host_elastic.reserve(host_materials.size());
        for (const auto& mat : host_materials) {
            host_elastic.push_back(mat.elastic());
        }
        elastic_props_.copy_from_host_async(host_elastic);

        manager_.synchronize();
    }

    const MaterialViewType* materials() const { return materials_.data(); }
    const DeviceElasticProperties* elastic_properties() const { return elastic_props_.data(); }
    const int* material_ids() const { return material_ids_.data(); }

    size_t num_materials() const { return num_materials_; }
    size_t num_particles() const { return num_particles_; }

    GpuMemoryManager& memory_manager() { return manager_; }
};

/**
 * @brief Particle data container for GPU computations
 */
class GpuParticleData {
  private:
    GpuBuffer<float> positions_x_, positions_y_, positions_z_;
    GpuBuffer<float> velocities_x_, velocities_y_, velocities_z_;
    GpuBuffer<float> forces_x_, forces_y_, forces_z_;
    GpuBuffer<float> radii_;
    GpuBuffer<float> masses_;
    size_t num_particles_;
    GpuMemoryManager& manager_;

  public:
    GpuParticleData(const std::vector<float>& pos_x, const std::vector<float>& pos_y, const std::vector<float>& pos_z,
                    const std::vector<float>& radii, GpuMemoryManager& manager)
        : positions_x_(pos_x.size(), manager), positions_y_(pos_y.size(), manager), positions_z_(pos_z.size(), manager),
          velocities_x_(pos_x.size(), manager), velocities_y_(pos_y.size(), manager),
          velocities_z_(pos_z.size(), manager), forces_x_(pos_x.size(), manager), forces_y_(pos_y.size(), manager),
          forces_z_(pos_z.size(), manager), radii_(radii.size(), manager), masses_(pos_x.size(), manager),
          num_particles_(pos_x.size()), manager_(manager) {

        // Upload position and radii data
        positions_x_.copy_from_host_async(pos_x);
        positions_y_.copy_from_host_async(pos_y);
        positions_z_.copy_from_host_async(pos_z);
        radii_.copy_from_host_async(radii);

        // Initialize velocities and forces to zero
        velocities_x_.zero_async();
        velocities_y_.zero_async();
        velocities_z_.zero_async();
        forces_x_.zero_async();
        forces_y_.zero_async();
        forces_z_.zero_async();

        manager_.synchronize();
    }

    // Position accessors
    const float* positions_x() const { return positions_x_.data(); }
    const float* positions_y() const { return positions_y_.data(); }
    const float* positions_z() const { return positions_z_.data(); }

    // Velocity accessors
    float* velocities_x() { return velocities_x_.data(); }
    float* velocities_y() { return velocities_y_.data(); }
    float* velocities_z() { return velocities_z_.data(); }

    // Force accessors
    float* forces_x() { return forces_x_.data(); }
    float* forces_y() { return forces_y_.data(); }
    float* forces_z() { return forces_z_.data(); }

    // Other properties
    const float* radii() const { return radii_.data(); }
    float* masses() { return masses_.data(); }

    size_t num_particles() const { return num_particles_; }

    void download_forces(std::vector<float>& fx, std::vector<float>& fy, std::vector<float>& fz) {
        forces_x_.copy_to_host_async(fx);
        forces_y_.copy_to_host_async(fy);
        forces_z_.copy_to_host_async(fz);
        manager_.synchronize();
    }

    void zero_forces() {
        forces_x_.zero_async();
        forces_y_.zero_async();
        forces_z_.zero_async();
    }

    GpuMemoryManager& memory_manager() { return manager_; }
};

#else // !MATERIAL_GPU_WITH_CUDA

// Stub implementations for non-CUDA builds
class GpuMemoryManager {
  public:
    GpuMemoryManager() = default;
    explicit GpuMemoryManager(void*) {}

    template <typename T> T* allocate_device(size_t) { throw std::runtime_error("CUDA support not enabled"); }

    template <typename T> void free_device(T*) {}

    void synchronize() {}
};

template <typename T> class GpuBuffer {
  public:
    GpuBuffer(size_t, GpuMemoryManager&) { throw std::runtime_error("CUDA support not enabled"); }
};

template <typename MaterialViewType> class GpuMaterialContainer {
  public:
    GpuMaterialContainer(const std::vector<MaterialViewType>&, const std::vector<int>&, GpuMemoryManager&) {
        throw std::runtime_error("CUDA support not enabled");
    }
};

class GpuParticleData {
  public:
    GpuParticleData(const std::vector<float>&, const std::vector<float>&, const std::vector<float>&,
                    const std::vector<float>&, GpuMemoryManager&) {
        throw std::runtime_error("CUDA support not enabled");
    }
};

#endif // MATERIAL_GPU_WITH_CUDA

} // namespace matgpu