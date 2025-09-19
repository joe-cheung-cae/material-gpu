#include "material/material.hpp"

#include "material/device_runtime.hpp"
#include "material/json_loader.hpp"
#include "material/validation.hpp"

#include <fstream>
#include <sstream>

namespace matgpu {

Materials::~Materials() { free_device(); }

bool Materials::load_from_json_text(const std::string& json_text) {
    std::vector<Material> mats;
    std::string err;
    if (!load_materials_from_json_ex(json_text, mats, err)) {
        last_error_ = err;
        return false;
    }
    if (auto verr = validate(mats); !verr.empty()) {
        last_error_ = std::move(verr);
        return false;
    }
    last_error_.clear();
    materials_ = std::move(mats);
    return true;
}

bool Materials::load_from_file(const std::string& json_path) {
    std::ifstream f(json_path);
    if (!f) {
        last_error_ = "Failed to open file: " + json_path;
        return false;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return load_from_json_text(ss.str());
}

bool Materials::upload_to_device() {
    if (auto verr = validate(materials_); !verr.empty()) {
        last_error_ = std::move(verr);
        return false;
    }
    bool ok = device_upload(materials_);
    if (!ok)
        last_error_ = "device_upload failed";
    return ok;
}

void Materials::free_device() { device_free(); }

int device_material_count() { return device_count(); }

bool Materials::get_device_view(DeviceMaterialsView& out) const { return ::matgpu::get_device_view(out); }

} // namespace matgpu
