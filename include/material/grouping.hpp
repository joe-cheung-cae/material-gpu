#pragma once
#include "material/types.hpp"

#include <vector>

namespace matgpu {

struct Group {
    std::vector<int> indices; // indices of materials in this group
};

struct GroupingResult {
    Group eepa;
    Group jkr;
    Group none;
};

// Partition materials by contact model into disjoint groups. Stable by index.
GroupingResult group_by_contact_model(const std::vector<Material>& mats);

} // namespace matgpu
