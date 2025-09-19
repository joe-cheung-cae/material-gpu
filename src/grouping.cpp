#include "material/grouping.hpp"

namespace matgpu {

GroupingResult group_by_contact_model(const std::vector<Material>& mats) {
    GroupingResult res;
    res.eepa.indices.reserve(mats.size());
    res.jkr.indices.reserve(mats.size());
    res.none.indices.reserve(mats.size());
    for (int i = 0; i < (int)mats.size(); ++i) {
        switch (mats[i].model) {
        case ContactModel::EEPA:
            res.eepa.indices.push_back(i);
            break;
        case ContactModel::JKR:
            res.jkr.indices.push_back(i);
            break;
        case ContactModel::None:
        default:
            res.none.indices.push_back(i);
            break;
        }
    }
    return res;
}

} // namespace matgpu
