#pragma once
#include "material/types.hpp"

#include <vector>

namespace matgpu {

#ifdef MATERIAL_GPU_WITH_CUDA
bool device_upload(const std::vector<Material>& materials);
void device_free();
int device_count();
// Fill a host-side DeviceMaterialsView that points to device arrays; returns false if not uploaded
bool get_device_view(DeviceMaterialsView& out);
#else
inline bool device_upload(const std::vector<Material>&) { return false; }
inline void device_free() {}
inline int device_count() { return 0; }
inline bool get_device_view(DeviceMaterialsView&) { return false; }
#endif

} // namespace matgpu
