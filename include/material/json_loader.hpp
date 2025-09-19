#pragma once
#include "material/types.hpp"

#include <string>
#include <vector>

namespace matgpu {

// Minimal JSON loader contract
// Expected schema:
// {
//   "materials": [
//     {
//       "elastic": {"young":1e7, "poisson":0.25, "density": 2500},
//       "contact_model": "EEPA" | "JKR" | "None",
//       "eepa": {"kn":..., "kt":..., "gamma_n":..., "gamma_t":...},
//       "jkr": {"work_of_adhesion":..., "contact_radius0":...},
//       "thermal": {"conductivity":..., "heat_capacity":...},
//       "em": {"permittivity":..., "permeability":..., "conductivity":...}
//     }, ...
//   ]
// }
// Optional objects may be omitted.

// Returns true on success; on failure, 'error' contains a human-readable reason.
bool load_materials_from_json_ex(const std::string& json_text, std::vector<Material>& out, std::string& error);

// Backward-compatible wrapper that discards error details.
inline bool load_materials_from_json(const std::string& json_text, std::vector<Material>& out) {
    std::string err;
    return load_materials_from_json_ex(json_text, out, err);
}

} // namespace matgpu
