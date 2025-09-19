#include "material/json_loader.hpp"

#include <nlohmann/json.hpp>

#include <fstream>
#include <sstream>

namespace matgpu {

using nlohmann::json;

bool JSONLoader::load_materials_from_json_text(const std::string& json_text,
                                               std::vector<std::unique_ptr<IMaterial>>& materials, std::string& error) {
    error.clear();
    materials.clear();

    json root;
    try {
        root = json::parse(json_text);
    } catch (const std::exception& e) {
        error = std::string("JSON parse error: ") + e.what();
        return false;
    }

    if (!root.is_object()) {
        error = "Root JSON must be an object";
        return false;
    }

    if (!root.contains("materials")) {
        error = "Missing 'materials' array";
        return false;
    }

    const auto& materials_array = root["materials"];
    if (!materials_array.is_array()) {
        error = "'materials' must be an array";
        return false;
    }

    materials.reserve(materials_array.size());

    for (size_t i = 0; i < materials_array.size(); ++i) {
        auto material = load_material_from_json_object(materials_array[i], error);
        if (!material) {
            error = "Material " + std::to_string(i) + ": " + error;
            return false;
        }
        material->set_material_id(i);
        materials.push_back(std::move(material));
    }

    return true;
}

std::unique_ptr<IMaterial> JSONLoader::load_material_from_json_object(const json& obj, std::string& error) {
    error.clear();

    if (!obj.is_object()) {
        error = "Material must be a JSON object";
        return nullptr;
    }

    // Validate JSON structure
    if (!validate_material_json(obj, error)) {
        return nullptr;
    }

    // Parse elastic properties
    float young, poisson, density;
    if (!parse_elastic_properties(obj, young, poisson, density, error)) {
        return nullptr;
    }

    // Create material using builder pattern
    MaterialBuilder builder;
    builder.elastic(young, poisson, density);

    // Parse contact model
    auto contact_model = parse_contact_model(obj, error);
    if (!contact_model) {
        return nullptr;
    }
    builder.contact_model(std::move(contact_model));

    // Parse optional thermal properties
    if (obj.contains("thermal")) {
        auto thermal = parse_thermal_properties(obj["thermal"], error);
        if (!thermal) {
            return nullptr;
        }
        builder.thermal_properties(std::move(thermal));
    }

    // Parse optional electromagnetic properties
    if (obj.contains("electromagnetic")) {
        auto em = parse_em_properties(obj["electromagnetic"], error);
        if (!em) {
            return nullptr;
        }
        builder.em_properties(std::move(em));
    }

    // Determine material type and create
    std::string type = "basic";
    if (obj.contains("type")) {
        type = obj["type"].get<std::string>();
    }

    return MaterialFactory::instance().create(type, builder);
}

bool JSONLoader::validate_material_json(const json& obj, std::string& error) {
    error.clear();

    // Check required fields
    if (!obj.contains("elastic")) {
        error = "Missing required 'elastic' properties";
        return false;
    }

    if (!obj.contains("contact")) {
        error = "Missing required 'contact' model";
        return false;
    }

    const auto& elastic = obj["elastic"];
    if (!elastic.is_object()) {
        error = "'elastic' must be an object";
        return false;
    }

    // Validate elastic properties
    if (!elastic.contains("young_modulus") || !elastic.contains("poisson_ratio") || !elastic.contains("density")) {
        error = "Missing required elastic properties: young_modulus, poisson_ratio, density";
        return false;
    }

    return true;
}

bool JSONLoader::parse_elastic_properties(const json& obj, float& young, float& poisson, float& density,
                                          std::string& error) {
    error.clear();

    const auto& elastic = obj["elastic"];

    try {
        young   = elastic["young_modulus"].get<float>();
        poisson = elastic["poisson_ratio"].get<float>();
        density = elastic["density"].get<float>();
    } catch (const std::exception& e) {
        error = std::string("Failed to parse elastic properties: ") + e.what();
        return false;
    }

    // Validate ranges
    if (young <= 0.0f) {
        error = "Young's modulus must be positive";
        return false;
    }
    if (poisson < -1.0f || poisson > 0.5f) {
        error = "Poisson's ratio must be between -1.0 and 0.5";
        return false;
    }
    if (density <= 0.0f) {
        error = "Density must be positive";
        return false;
    }

    return true;
}

std::unique_ptr<ContactModel> JSONLoader::parse_contact_model(const json& obj, std::string& error) {
    error.clear();

    const auto& contact = obj["contact"];
    if (!contact.is_object()) {
        error = "'contact' must be an object";
        return nullptr;
    }

    if (!contact.contains("type")) {
        error = "Missing contact model type";
        return nullptr;
    }

    const std::string type = contact["type"].get<std::string>();

    if (type == "none") {
        return std::make_unique<NoContactModel>();
    } else if (type == "eepa") {
        if (!contact.contains("kn") || !contact.contains("kt") || !contact.contains("gamma_n") ||
            !contact.contains("gamma_t")) {
            error = "EEPA contact model requires 'kn', 'kt', 'gamma_n', and 'gamma_t'";
            return nullptr;
        }
        float kn      = contact["kn"].get<float>();
        float kt      = contact["kt"].get<float>();
        float gamma_n = contact["gamma_n"].get<float>();
        float gamma_t = contact["gamma_t"].get<float>();
        return std::make_unique<EEPAContactModel>(kn, kt, gamma_n, gamma_t);
    } else if (type == "jkr") {
        if (!contact.contains("work_of_adhesion") || !contact.contains("contact_radius0")) {
            error = "JKR contact model requires 'work_of_adhesion' and 'contact_radius0'";
            return nullptr;
        }
        float work_of_adhesion = contact["work_of_adhesion"].get<float>();
        float contact_radius0  = contact["contact_radius0"].get<float>();
        return std::make_unique<JKRContactModel>(work_of_adhesion, contact_radius0);
    } else {
        error = "Unknown contact model type: " + type;
        return nullptr;
    }
}

std::unique_ptr<ThermalProperties> JSONLoader::parse_thermal_properties(const json& obj, std::string& error) {
    error.clear();

    if (!obj.is_object()) {
        error = "Thermal properties must be an object";
        return nullptr;
    }

    if (!obj.contains("conductivity") || !obj.contains("heat_capacity")) {
        error = "Thermal properties require 'conductivity' and 'heat_capacity'";
        return nullptr;
    }

    try {
        float conductivity  = obj["conductivity"].get<float>();
        float heat_capacity = obj["heat_capacity"].get<float>();

        if (conductivity < 0.0f || heat_capacity <= 0.0f) {
            error = "Thermal properties must be positive";
            return nullptr;
        }

        return std::make_unique<ThermalProperties>(conductivity, heat_capacity);
    } catch (const std::exception& e) {
        error = std::string("Failed to parse thermal properties: ") + e.what();
        return nullptr;
    }
}

std::unique_ptr<ElectromagneticProperties> JSONLoader::parse_em_properties(const json& obj, std::string& error) {
    error.clear();

    if (!obj.is_object()) {
        error = "Electromagnetic properties must be an object";
        return nullptr;
    }

    if (!obj.contains("permittivity") || !obj.contains("permeability") || !obj.contains("conductivity")) {
        error = "Electromagnetic properties require 'permittivity', 'permeability', and 'conductivity'";
        return nullptr;
    }

    try {
        float permittivity = obj["permittivity"].get<float>();
        float permeability = obj["permeability"].get<float>();
        float conductivity = obj["conductivity"].get<float>();

        if (permittivity <= 0.0f || permeability <= 0.0f || conductivity < 0.0f) {
            error = "Electromagnetic properties must be positive (conductivity can be zero)";
            return nullptr;
        }

        return std::make_unique<ElectromagneticProperties>(permittivity, permeability, conductivity);
    } catch (const std::exception& e) {
        error = std::string("Failed to parse electromagnetic properties: ") + e.what();
        return nullptr;
    }
}

// JSON serialization helpers
nlohmann::json JSONLoader::serialize_elastic_properties(const IMaterial& material) {
    json elastic;
    elastic["young_modulus"] = material.young_modulus();
    elastic["poisson_ratio"] = material.poisson_ratio();
    elastic["density"]       = material.density();
    return elastic;
}

nlohmann::json JSONLoader::serialize_contact_model(const ContactModel& contact) {
    json contact_json;
    contact_json["type"] = contact.model_name();

    if (auto eepa = dynamic_cast<const EEPAContactModel*>(&contact)) {
        contact_json["kn"]      = eepa->kn();
        contact_json["kt"]      = eepa->kt();
        contact_json["gamma_n"] = eepa->gamma_n();
        contact_json["gamma_t"] = eepa->gamma_t();
    } else if (auto jkr = dynamic_cast<const JKRContactModel*>(&contact)) {
        contact_json["work_of_adhesion"] = jkr->get_parameter("work_of_adhesion");
        contact_json["contact_radius0"]  = jkr->get_parameter("contact_radius0");
    }

    return contact_json;
}

nlohmann::json JSONLoader::serialize_thermal_properties(const ThermalProperties& thermal) {
    json thermal_json;
    thermal_json["conductivity"]  = thermal.conductivity();
    thermal_json["heat_capacity"] = thermal.heat_capacity();
    return thermal_json;
}

nlohmann::json JSONLoader::serialize_em_properties(const ElectromagneticProperties& em) {
    json em_json;
    em_json["permittivity"] = em.permittivity();
    em_json["permeability"] = em.permeability();
    return em_json;
}

// Materials class implementation
bool Materials::load_from_json_text(const std::string& json_text) {
    return JSONLoader::load_materials_from_json_text(json_text, materials_, last_error_);
}

bool Materials::load_from_file(const std::string& json_path) {
    std::ifstream file(json_path);
    if (!file.is_open()) {
        last_error_ = "Cannot open file: " + json_path;
        return false;
    }

    std::ostringstream buffer;
    buffer << file.rdbuf();
    return load_from_json_text(buffer.str());
}

IMaterial* Materials::find_material_by_id(size_t id) {
    for (auto& material : materials_) {
        if (material->material_id() == id) {
            return material.get();
        }
    }
    return nullptr;
}

const IMaterial* Materials::find_material_by_id(size_t id) const {
    for (const auto& material : materials_) {
        if (material->material_id() == id) {
            return material.get();
        }
    }
    return nullptr;
}

std::vector<IMaterial*> Materials::find_materials_by_type(const std::string& type) {
    std::vector<IMaterial*> result;
    for (auto& material : materials_) {
        if (material->material_type() == type) {
            result.push_back(material.get());
        }
    }
    return result;
}

bool Materials::validate_all_materials(std::string& error) const {
    error.clear();

    for (size_t i = 0; i < materials_.size(); ++i) {
        const auto& material = materials_[i];
        if (!material) {
            error = "Material " + std::to_string(i) + " is null";
            return false;
        }

        // Validate elastic properties
        if (material->young_modulus() <= 0.0f) {
            error = "Material " + std::to_string(i) + " has invalid Young's modulus";
            return false;
        }

        if (material->poisson_ratio() < -1.0f || material->poisson_ratio() > 0.5f) {
            error = "Material " + std::to_string(i) + " has invalid Poisson's ratio";
            return false;
        }

        if (material->density() <= 0.0f) {
            error = "Material " + std::to_string(i) + " has invalid density";
            return false;
        }
    }

    return true;
}

std::string Materials::export_to_json() const {
    json root;
    json materials_array = json::array();

    for (const auto& material : materials_) {
        json material_json;

        // Basic properties
        material_json["type"]    = material->material_type();
        material_json["elastic"] = JSONLoader::serialize_elastic_properties(*material);
        material_json["contact"] = JSONLoader::serialize_contact_model(material->contact_model());

        // Optional properties
        if (material->thermal_properties()) {
            material_json["thermal"] = JSONLoader::serialize_thermal_properties(*material->thermal_properties());
        }

        if (material->electromagnetic_properties()) {
            material_json["electromagnetic"] =
                JSONLoader::serialize_em_properties(*material->electromagnetic_properties());
        }

        materials_array.push_back(material_json);
    }

    root["materials"] = materials_array;
    return root.dump(2);
}

bool Materials::export_to_file(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    file << export_to_json();
    return file.good();
}

} // namespace matgpu