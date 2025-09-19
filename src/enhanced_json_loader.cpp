#include "material/enhanced_json_loader.hpp"

#include <nlohmann/json.hpp>

#include <fstream>
#include <sstream>

namespace matgpu {

using nlohmann::json;

bool EnhancedJSONLoader::load_materials_from_json_text(const std::string& json_text,
                                                       std::vector<std::unique_ptr<IMaterial>>& materials,
                                                       std::string& error) {
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

std::unique_ptr<IMaterial> EnhancedJSONLoader::load_material_from_json_object(const json& obj, std::string& error) {
    error.clear();

    if (!obj.is_object()) {
        error = "Material must be a JSON object";
        return nullptr;
    }

    // Parse elastic properties (required)
    float young, poisson, density;
    if (!parse_elastic_properties(obj, young, poisson, density, error)) {
        return nullptr;
    }

    // Create builder with elastic properties
    MaterialBuilder builder;
    builder.elastic(young, poisson, density);

    // Parse contact model (optional, defaults to no contact)
    auto contact_model = parse_contact_model(obj, error);
    if (!error.empty()) {
        return nullptr; // Parse error
    }

    if (contact_model) {
        builder.contact_model(std::move(contact_model));

        // Determine material type based on contact model
        if (dynamic_cast<EEPAContactModel*>(contact_model.get())) {
            builder.material_type("eepa");
        } else if (dynamic_cast<JKRContactModel*>(contact_model.get())) {
            builder.material_type("jkr");
        }
    }

    // Parse optional thermal properties
    auto thermal = parse_thermal_properties(obj, error);
    if (!error.empty()) {
        return nullptr; // Parse error
    }
    if (thermal) {
        builder.thermal_properties(std::move(thermal));
    }

    // Parse optional electromagnetic properties
    auto em = parse_em_properties(obj, error);
    if (!error.empty()) {
        return nullptr; // Parse error
    }
    if (em) {
        builder.em_properties(std::move(em));
    }

    // Build the material
    try {
        return builder.build();
    } catch (const std::exception& e) {
        error = std::string("Failed to build material: ") + e.what();
        return nullptr;
    }
}

bool EnhancedJSONLoader::parse_elastic_properties(const json& obj, float& young, float& poisson, float& density,
                                                  std::string& error) {
    if (!obj.contains("elastic")) {
        error = "Missing required 'elastic' properties";
        return false;
    }

    const auto& elastic = obj["elastic"];
    if (!elastic.is_object()) {
        error = "'elastic' must be an object";
        return false;
    }

    if (!elastic.contains("young")) {
        error = "Missing 'young' modulus in elastic properties";
        return false;
    }
    if (!elastic.contains("poisson")) {
        error = "Missing 'poisson' ratio in elastic properties";
        return false;
    }
    if (!elastic.contains("density")) {
        error = "Missing 'density' in elastic properties";
        return false;
    }

    try {
        young   = elastic["young"].get<float>();
        poisson = elastic["poisson"].get<float>();
        density = elastic["density"].get<float>();
    } catch (const std::exception& e) {
        error = std::string("Invalid elastic property values: ") + e.what();
        return false;
    }

    // Basic validation
    if (young <= 0.0f) {
        error = "Young's modulus must be positive";
        return false;
    }
    if (poisson < -1.0f || poisson > 0.5f) {
        error = "Poisson's ratio must be between -1 and 0.5";
        return false;
    }
    if (density <= 0.0f) {
        error = "Density must be positive";
        return false;
    }

    return true;
}

std::unique_ptr<ContactModel> EnhancedJSONLoader::parse_contact_model(const json& obj, std::string& error) {
    if (!obj.contains("contact_model")) {
        // No contact model specified, return default (no contact)
        return std::make_unique<NoContactModel>();
    }

    const auto& contact_model_name = obj["contact_model"];
    if (!contact_model_name.is_string()) {
        error = "'contact_model' must be a string";
        return nullptr;
    }

    std::string model_name = contact_model_name.get<std::string>();

    if (model_name == "EEPA" || model_name == "eepa" || model_name == "HERTZ_MINDLIN") {
        // Parse EEPA parameters
        if (!obj.contains("eepa")) {
            error = "Missing 'eepa' parameters for EEPA contact model";
            return nullptr;
        }

        const auto& eepa = obj["eepa"];
        if (!eepa.is_object()) {
            error = "'eepa' must be an object";
            return nullptr;
        }

        try {
            float kn      = eepa.value("kn", 1e5f);
            float kt      = eepa.value("kt", 5e4f);
            float gamma_n = eepa.value("gamma_n", 0.2f);
            float gamma_t = eepa.value("gamma_t", 0.1f);

            return std::make_unique<EEPAContactModel>(kn, kt, gamma_n, gamma_t);
        } catch (const std::exception& e) {
            error = std::string("Invalid EEPA parameters: ") + e.what();
            return nullptr;
        }
    } else if (model_name == "JKR" || model_name == "jkr") {
        // Parse JKR parameters
        if (!obj.contains("jkr")) {
            error = "Missing 'jkr' parameters for JKR contact model";
            return nullptr;
        }

        const auto& jkr = obj["jkr"];
        if (!jkr.is_object()) {
            error = "'jkr' must be an object";
            return nullptr;
        }

        try {
            float work_of_adhesion = jkr.value("work_of_adhesion", 0.05f);
            float contact_radius0  = jkr.value("contact_radius0", 1e-4f);

            return std::make_unique<JKRContactModel>(work_of_adhesion, contact_radius0);
        } catch (const std::exception& e) {
            error = std::string("Invalid JKR parameters: ") + e.what();
            return nullptr;
        }
    } else if (model_name == "None" || model_name == "none" || model_name.empty()) {
        return std::make_unique<NoContactModel>();
    } else {
        error = "Unknown contact model: " + model_name;
        return nullptr;
    }
}

std::unique_ptr<ThermalProperties> EnhancedJSONLoader::parse_thermal_properties(const json& obj, std::string& error) {
    if (!obj.contains("thermal")) {
        return nullptr; // No thermal properties
    }

    const auto& thermal = obj["thermal"];
    if (!thermal.is_object()) {
        error = "'thermal' must be an object";
        return nullptr;
    }

    try {
        float conductivity  = thermal.value("conductivity", 10.0f);
        float heat_capacity = thermal.value("heat_capacity", 900.0f);

        return std::make_unique<ThermalProperties>(conductivity, heat_capacity);
    } catch (const std::exception& e) {
        error = std::string("Invalid thermal parameters: ") + e.what();
        return nullptr;
    }
}

std::unique_ptr<ElectromagneticProperties> EnhancedJSONLoader::parse_em_properties(const json& obj,
                                                                                   std::string& error) {
    if (!obj.contains("em") && !obj.contains("electromagnetic")) {
        return nullptr; // No EM properties
    }

    const auto& em = obj.contains("em") ? obj["em"] : obj["electromagnetic"];
    if (!em.is_object()) {
        error = "'em'/'electromagnetic' must be an object";
        return nullptr;
    }

    try {
        float permittivity = em.value("permittivity", 8.85e-12f);
        float permeability = em.value("permeability", 1.26e-6f);
        float conductivity = em.value("conductivity", 1e-6f);

        return std::make_unique<ElectromagneticProperties>(permittivity, permeability, conductivity);
    } catch (const std::exception& e) {
        error = std::string("Invalid EM parameters: ") + e.what();
        return nullptr;
    }
}

// MaterialsV2 implementation
bool MaterialsV2::load_from_json_text(const std::string& json_text) {
    return EnhancedJSONLoader::load_materials_from_json_text(json_text, materials_, last_error_);
}

bool MaterialsV2::load_from_file(const std::string& json_path) {
    std::ifstream f(json_path);
    if (!f) {
        last_error_ = "Failed to open file: " + json_path;
        return false;
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return load_from_json_text(ss.str());
}

const std::vector<Material>& MaterialsV2::host() const {
    update_legacy_cache();
    return legacy_cache_;
}

std::vector<Material>& MaterialsV2::host_mut() {
    update_legacy_cache();
    legacy_cache_valid_ = false; // Mark as invalid since user can modify
    return legacy_cache_;
}

void MaterialsV2::update_legacy_cache() const {
    if (legacy_cache_valid_) {
        return;
    }

    legacy_cache_.clear();
    legacy_cache_.reserve(materials_.size());

    for (const auto& material : materials_) {
        legacy_cache_.push_back(EnhancedJSONLoader::convert_to_legacy_material(*material));
    }

    legacy_cache_valid_ = true;
}

IMaterial* MaterialsV2::find_material_by_id(size_t id) {
    for (auto& material : materials_) {
        if (material->material_id() == id) {
            return material.get();
        }
    }
    return nullptr;
}

const IMaterial* MaterialsV2::find_material_by_id(size_t id) const {
    for (const auto& material : materials_) {
        if (material->material_id() == id) {
            return material.get();
        }
    }
    return nullptr;
}

std::vector<IMaterial*> MaterialsV2::find_materials_by_type(const std::string& type) {
    std::vector<IMaterial*> result;
    for (auto& material : materials_) {
        if (material->material_type() == type) {
            result.push_back(material.get());
        }
    }
    return result;
}

// Legacy material conversion stubs (to be implemented)
std::unique_ptr<IMaterial> EnhancedJSONLoader::convert_legacy_material(const Material& legacy_material) {
    MaterialBuilder builder;
    builder.elastic(legacy_material.elastic.young, legacy_material.elastic.poisson, legacy_material.elastic.density);

    // Convert contact model
    switch (legacy_material.model) {
    case ContactModelType::EEPA:
        builder.eepa_contact(legacy_material.eepa.kn, legacy_material.eepa.kt, legacy_material.eepa.gamma_n,
                             legacy_material.eepa.gamma_t);
        break;
    case ContactModelType::JKR:
        builder.jkr_contact(legacy_material.jkr.work_of_adhesion, legacy_material.jkr.contact_radius0);
        break;
    default:
        builder.no_contact();
        break;
    }

    // Add thermal properties if present
    if (legacy_material.has_thermal) {
        builder.thermal(legacy_material.thermal.conductivity, legacy_material.thermal.heat_capacity);
    }

    // Add EM properties if present
    if (legacy_material.has_em) {
        builder.electromagnetic(legacy_material.em.permittivity, legacy_material.em.permeability,
                                legacy_material.em.conductivity);
    }

    return builder.build();
}

Material EnhancedJSONLoader::convert_to_legacy_material(const IMaterial& material) {
    Material legacy;

    // Copy elastic properties
    legacy.elastic.young   = material.young_modulus();
    legacy.elastic.poisson = material.poisson_ratio();
    legacy.elastic.density = material.density();

    // Convert contact model
    const auto& contact = material.contact_model();
    if (contact.model_name() == "EEPA") {
        legacy.model        = ContactModelType::EEPA;
        legacy.eepa.kn      = contact.get_parameter("kn");
        legacy.eepa.kt      = contact.get_parameter("kt");
        legacy.eepa.gamma_n = contact.get_parameter("gamma_n");
        legacy.eepa.gamma_t = contact.get_parameter("gamma_t");
    } else if (contact.model_name() == "JKR") {
        legacy.model                = ContactModelType::JKR;
        legacy.jkr.work_of_adhesion = contact.get_parameter("work_of_adhesion");
        legacy.jkr.contact_radius0  = contact.get_parameter("contact_radius0");
    } else {
        legacy.model = ContactModelType::None;
    }

    // Convert thermal properties
    const auto* thermal = material.thermal_properties();
    if (thermal && thermal->is_enabled()) {
        legacy.has_thermal           = true;
        legacy.thermal.conductivity  = thermal->conductivity();
        legacy.thermal.heat_capacity = thermal->heat_capacity();
    } else {
        legacy.has_thermal = false;
    }

    // Convert EM properties
    const auto* em = material.electromagnetic_properties();
    if (em && em->is_enabled()) {
        legacy.has_em          = true;
        legacy.em.permittivity = em->permittivity();
        legacy.em.permeability = em->permeability();
        legacy.em.conductivity = em->conductivity();
    } else {
        legacy.has_em = false;
    }

    return legacy;
}

bool MaterialsV2::upload_to_device() {
    // Convert to legacy format and use existing device upload
    update_legacy_cache();
    // This would call the existing device_upload function
    // Implementation depends on the existing device_runtime.hpp
    return true; // Placeholder
}

bool MaterialsV2::get_device_view(DeviceMaterialsView& out) const {
    // This would use the existing get_device_view function
    // Implementation depends on the existing device_runtime.hpp
    return true; // Placeholder
}

void MaterialsV2::free_device() {
    // This would call the existing device_free function
    // Implementation depends on the existing device_runtime.hpp
}

bool MaterialsV2::validate_all_materials(std::string& error) const {
    for (size_t i = 0; i < materials_.size(); ++i) {
        const auto& material = materials_[i];
        if (!material) {
            error = "Material " + std::to_string(i) + " is null";
            return false;
        }
        // Add more validation as needed
    }
    return true;
}

std::string MaterialsV2::export_to_json() const {
    // TODO: Implement JSON serialization
    return "{}"; // Placeholder
}

bool MaterialsV2::export_to_file(const std::string& filename) const {
    std::ofstream f(filename);
    if (!f) {
        return false;
    }
    f << export_to_json();
    return true;
}

} // namespace matgpu