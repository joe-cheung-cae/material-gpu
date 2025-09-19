#include "material/json_loader.hpp"

#include <nlohmann/json.hpp>

#include <sstream>

namespace matgpu {

using nlohmann::json;

static std::string to_upper(std::string s) {
    for (char& c : s)
        c = (char)std::toupper((unsigned char)c);
    return s;
}

static ContactModel parse_contact_model_string(std::string s) {
    s = to_upper(std::move(s));
    if (s == "EEPA" || s == "HERTZ_MINDLIN")
        return ContactModel::EEPA;
    if (s == "JKR")
        return ContactModel::JKR;
    return ContactModel::None;
}

static std::string require_field(const json& j, const char* field, const char* where) {
    if (!j.contains(field)) {
        std::ostringstream oss;
        oss << where << ": missing field '" << field << "'";
        return oss.str();
    }
    return {};
}

bool load_materials_from_json_ex(const std::string& text, std::vector<Material>& out, std::string& error) {
    error.clear();
    out.clear();
    json root;
    try {
        root = json::parse(text);
    } catch (const std::exception& e) {
        error = std::string("JSON parse error: ") + e.what();
        return false;
    }

    if (!root.is_object()) {
        error = "Root JSON must be an object";
        return false;
    }
    if (!root.contains("materials")) {
        error = "Root missing 'materials' array";
        return false;
    }
    const json& arr = root["materials"];
    if (!arr.is_array()) {
        error = "'materials' must be an array";
        return false;
    }

    for (size_t idx = 0; idx < arr.size(); ++idx) {
        const json& jm = arr[idx];
        if (!jm.is_object()) {
            error = "materials[" + std::to_string(idx) + "] must be an object";
            return false;
        }

        Material m{};
        m.model       = ContactModel::None;
        m.has_thermal = false;
        m.has_em      = false;

        if (auto err = require_field(jm, "elastic", ("materials[" + std::to_string(idx) + "]").c_str()); !err.empty()) {
            error = err;
            return false;
        }
        const json& je = jm["elastic"];
        if (!je.is_object()) {
            error = "elastic must be an object";
            return false;
        }
        if (!(je.contains("young") || je.contains("E"))) {
            error = "elastic: missing 'young' (or 'E')";
            return false;
        }
        if (!(je.contains("poisson") || je.contains("nu"))) {
            error = "elastic: missing 'poisson' (or 'nu')";
            return false;
        }
        if (!(je.contains("density") || je.contains("rho"))) {
            error = "elastic: missing 'density' (or 'rho')";
            return false;
        }
        try {
            m.elastic.young   = je.contains("young") ? je.at("young").get<float>() : je.at("E").get<float>();
            m.elastic.poisson = je.contains("poisson") ? je.at("poisson").get<float>() : je.at("nu").get<float>();
            m.elastic.density = je.contains("density") ? je.at("density").get<float>() : je.at("rho").get<float>();
        } catch (const std::exception& e) {
            error = std::string("elastic parsing error: ") + e.what();
            return false;
        }

        if (jm.contains("contact_model")) {
            try {
                m.model = parse_contact_model_string(jm.at("contact_model").get<std::string>());
            } catch (const std::exception& e) {
                error = std::string("contact_model error: ") + e.what();
                return false;
            }
        }

        if (jm.contains("eepa")) {
            const json& jeepa = jm["eepa"];
            if (!jeepa.is_object()) {
                error = "eepa must be an object";
                return false;
            }
            try {
                m.eepa.kn      = jeepa.value("kn", 0.f);
                m.eepa.kt      = jeepa.value("kt", 0.f);
                m.eepa.gamma_n = jeepa.value("gamma_n", 0.f);
                m.eepa.gamma_t = jeepa.value("gamma_t", 0.f);
            } catch (const std::exception& e) {
                error = std::string("eepa error: ") + e.what();
                return false;
            }
        }

        if (jm.contains("jkr")) {
            const json& jjkr = jm["jkr"];
            if (!jjkr.is_object()) {
                error = "jkr must be an object";
                return false;
            }
            try {
                m.jkr.work_of_adhesion = jjkr.value("work_of_adhesion", 0.f);
                m.jkr.contact_radius0  = jjkr.value("contact_radius0", 0.f);
            } catch (const std::exception& e) {
                error = std::string("jkr error: ") + e.what();
                return false;
            }
        }

        if (jm.contains("thermal")) {
            const json& jt = jm["thermal"];
            if (!jt.is_object()) {
                error = "thermal must be an object";
                return false;
            }
            try {
                m.thermal.conductivity  = jt.value("conductivity", 0.f);
                m.thermal.heat_capacity = jt.value("heat_capacity", 0.f);
                m.has_thermal           = true;
            } catch (const std::exception& e) {
                error = std::string("thermal error: ") + e.what();
                return false;
            }
        }

        if (jm.contains("em")) {
            const json& jem = jm["em"];
            if (!jem.is_object()) {
                error = "em must be an object";
                return false;
            }
            try {
                m.em.permittivity = jem.value("permittivity", 0.f);
                m.em.permeability = jem.value("permeability", 0.f);
                m.em.conductivity = jem.value("conductivity", 0.f);
                m.has_em          = true;
            } catch (const std::exception& e) {
                error = std::string("em error: ") + e.what();
                return false;
            }
        }

        out.push_back(m);
    }
    return true;
}

} // namespace matgpu
