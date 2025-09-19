#pragma once
#include "material/types.hpp"

#include <string>
#include <vector>

namespace matgpu {

class Materials {
  public:
    Materials() = default;
    ~Materials();

    // Load materials from a JSON text (not path). Returns true on success.
    bool load_from_json_text(const std::string& json_text);

    // Load materials from a JSON file path. Returns true on success.
    bool load_from_file(const std::string& json_path);

    // Upload currently loaded materials to device memory.
    // Returns true on success (no-op when CUDA disabled, returns false).
    bool upload_to_device();

    // Fill a device view for kernel launches; returns false if device data is not
    // uploaded.
    bool get_device_view(DeviceMaterialsView& out) const;

    // Convenience: upload and fetch a DeviceMaterialsView in one step.
    bool upload_and_get_view(DeviceMaterialsView& out) {
        if (!upload_to_device())
            return false;
        return get_device_view(out);
    }

    // Free GPU memory (safe to call multiple times).
    void free_device();

    // Access loaded count
    int count() const { return static_cast<int>(materials_.size()); }

    // Access host copy
    const std::vector<Material>& host() const { return materials_; }

    // Access mutable host copy for programmatic construction
    std::vector<Material>& host_mut() { return materials_; }

    // Returns last error message (set by load* methods); empty if none.
    const std::string& last_error() const { return last_error_; }

  private:
    std::vector<Material> materials_;
    std::string last_error_;
};

// Device-side view access (CUDA build only). Returns number of materials on
// device, or 0 if not available.
int device_material_count();

} // namespace matgpu
