#include "material/validation.hpp"

#include <sstream>

namespace matgpu {

static bool valid_number(float x) { return x == x; }

std::string validate(const Material& m) {
    std::ostringstream err;
    if (!valid_number(m.elastic.young) || m.elastic.young <= 0) {
        err << "invalid elastic.young";
        return err.str();
    }
    if (m.elastic.poisson <= -0.5f || m.elastic.poisson >= 0.5f) {
        err << "invalid elastic.poisson (expected (-0.5,0.5))";
        return err.str();
    }
    if (!valid_number(m.elastic.density) || m.elastic.density <= 0) {
        err << "invalid elastic.density";
        return err.str();
    }
    switch (m.model) {
    case ContactModel::None:
        break;
    case ContactModel::EEPA:
        if (m.eepa.kn < 0 || m.eepa.kt < 0 || m.eepa.gamma_n < 0 || m.eepa.gamma_t < 0) {
            err << "invalid EEPA params (must be non-negative)";
            return err.str();
        }
        break;
    case ContactModel::JKR:
        if (m.jkr.work_of_adhesion < 0 || m.jkr.contact_radius0 < 0) {
            err << "invalid JKR params (must be non-negative)";
            return err.str();
        }
        break;
    }
    if (m.has_thermal) {
        if (m.thermal.conductivity < 0 || m.thermal.heat_capacity <= 0) {
            err << "invalid thermal props";
            return err.str();
        }
    }
    if (m.has_em) {
        if (m.em.permittivity < 0 || m.em.permeability < 0 || m.em.conductivity < 0) {
            err << "invalid EM props";
            return err.str();
        }
    }
    return {};
}

std::string validate(const std::vector<Material>& mats) {
    for (size_t i = 0; i < mats.size(); ++i) {
        std::string e = validate(mats[i]);
        if (!e.empty()) {
            std::ostringstream oss;
            oss << "materials[" << i << "]: " << e;
            return oss.str();
        }
    }
    return {};
}

} // namespace matgpu
