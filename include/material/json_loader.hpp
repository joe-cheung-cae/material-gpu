#pragma once
#include "material/material_factory.hpp"

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

namespace matgpu {

// JSON loader for the new material architecture
class JSONLoader {
  public:
    // Load materials using the new factory system
    static bool load_materials_from_json_text(const std::string& json_text,
                                              std::vector<std::unique_ptr<IMaterial>>& materials, std::string& error);

    // Load single material from JSON object
    static std::unique_ptr<IMaterial> load_material_from_json_object(const nlohmann::json& obj, std::string& error);

    // Validate material JSON structure
    static bool validate_material_json(const nlohmann::json& obj, std::string& error);

    // JSON serialization helpers
    static nlohmann::json serialize_elastic_properties(const IMaterial& material);
    static nlohmann::json serialize_contact_model(const ContactModel& contact);
    static nlohmann::json serialize_thermal_properties(const ThermalProperties& thermal);
    static nlohmann::json serialize_em_properties(const ElectromagneticProperties& em);

  private:
    // Helper methods for parsing specific sections
    static bool parse_elastic_properties(const nlohmann::json& obj, float& young, float& poisson, float& density,
                                         std::string& error);

    static std::unique_ptr<ContactModel> parse_contact_model(const nlohmann::json& obj, std::string& error);

    static std::unique_ptr<ThermalProperties> parse_thermal_properties(const nlohmann::json& obj, std::string& error);

    static std::unique_ptr<ElectromagneticProperties> parse_em_properties(const nlohmann::json& obj,
                                                                          std::string& error);
};

// Modern materials container
class Materials {
  private:
    std::vector<std::unique_ptr<IMaterial>> materials_;
    std::string last_error_;

  public:
    Materials()  = default;
    ~Materials() = default;

    // Move-only semantics for unique_ptr management
    Materials(const Materials&)            = delete;
    Materials& operator=(const Materials&) = delete;
    Materials(Materials&&)                 = default;
    Materials& operator=(Materials&&)      = default;

    // Loading methods
    bool load_from_json_text(const std::string& json_text);
    bool load_from_file(const std::string& json_path);

    // Access to materials
    const std::vector<std::unique_ptr<IMaterial>>& materials() const { return materials_; }
    std::vector<std::unique_ptr<IMaterial>>& materials_mut() { return materials_; }

    // Add material programmatically
    void add_material(std::unique_ptr<IMaterial> material) { materials_.push_back(std::move(material)); }

    // Template methods for adding specific material types
    template <typename MaterialType, typename... Args> MaterialType* emplace_material(Args&&... args) {
        auto material     = std::make_unique<MaterialType>(std::forward<Args>(args)...);
        MaterialType* ptr = material.get();
        add_material(std::move(material));
        return ptr;
    }

    // Factory-based material creation
    bool add_material_from_builder(const std::string& type, MaterialBuilder& builder) {
        auto material = MaterialFactory::instance().create(type, builder);
        if (material) {
            add_material(std::move(material));
            return true;
        }
        last_error_ = "Unknown material type: " + type;
        return false;
    }

    // Utility methods
    int count() const { return static_cast<int>(materials_.size()); }
    void clear() { materials_.clear(); }

    const std::string& last_error() const { return last_error_; }

    // Material queries and manipulation
    IMaterial* find_material_by_id(size_t id);
    const IMaterial* find_material_by_id(size_t id) const;

    std::vector<IMaterial*> find_materials_by_type(const std::string& type);

    // Validation
    bool validate_all_materials(std::string& error) const;

    // JSON export
    std::string export_to_json() const;
    bool export_to_file(const std::string& filename) const;
};

} // namespace matgpu