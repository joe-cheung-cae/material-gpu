#pragma once
#include "material/types.hpp"

#include <string>
#include <vector>

namespace matgpu {

// Validate a single material; returns empty string if ok, else an error
// message.
std::string validate(const Material& m);

// Validate an array of materials; returns empty string if ok, else the first
// error with index.
std::string validate(const std::vector<Material>& mats);

} // namespace matgpu
