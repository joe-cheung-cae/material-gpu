#pragma once
#include "material/material_factory.hpp"
#include "material/types.hpp"

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

namespace matgpu {

// Enhanced JSON loader supporting the new architecture
class EnhancedJSONLoader {
  public:
    // Load materials using the new factory system
    static bool load_materials_from_json_text(const std::string& json_text,
                                              std::vector<std::unique_ptr<IMaterial>>& materials, std::string& error);

    // Load single material from JSON object
    static std::unique_ptr<IMaterial> load_material_from_json_object(const nlohmann::json& obj, std::string& error);

    // Convert legacy Material struct to new interface
    static std::unique_ptr<IMaterial> convert_legacy_material(const Material& legacy_material);

    // Convert new interface back to legacy for compatibility
    static Material convert_to_legacy_material(const IMaterial& material);

    // Validate material JSON structure
    static bool validate_material_json(const nlohmann::json& obj, std::string& error);

  private:
    // Helper methods for parsing specific sections
    static bool parse_elastic_properties(const nlohmann::json& obj, float& young, float& poisson, float& density,
                                         std::string& error);

    static std::unique_ptr<ContactModel> parse_contact_model(const nlohmann::json& obj, std::string& error);

    static std::unique_ptr<ThermalProperties> parse_thermal_properties(const nlohmann::json& obj, std::string& error);

    static std::unique_ptr<ElectromagneticProperties> parse_em_properties(const nlohmann::json& obj,
                                                                          std::string& error);

    // JSON serialization helpers
    static nlohmann::json serialize_elastic_properties(const IMaterial& material);
    static nlohmann::json serialize_contact_model(const ContactModel& contact);
    static nlohmann::json serialize_thermal_properties(const ThermalProperties& thermal);
    static nlohmann::json serialize_em_properties(const ElectromagneticProperties& em);
};

// Wrapper class that maintains compatibility with existing code
class MaterialsV2 {
  private:
    std::vector<std::unique_ptr<IMaterial>> materials_;
    std::string last_error_;

    // Legacy compatibility
    mutable std::vector<Material> legacy_cache_;
    mutable bool legacy_cache_valid_ = false;

  public:
    MaterialsV2()  = default;
    ~MaterialsV2() = default;

    // Move-only semantics for unique_ptr management
    MaterialsV2(const MaterialsV2&)            = delete;
    MaterialsV2& operator=(const MaterialsV2&) = delete;
    MaterialsV2(MaterialsV2&&)                 = default;
    MaterialsV2& operator=(MaterialsV2&&)      = default;

    // New interface methods
    bool load_from_json_text(const std::string& json_text);
    bool load_from_file(const std::string& json_path);

    // Access to new material interface
    const std::vector<std::unique_ptr<IMaterial>>& materials() const { return materials_; }
    std::vector<std::unique_ptr<IMaterial>>& materials_mut() {
        legacy_cache_valid_ = false;
        return materials_;
    }

    // Add material programmatically
    void add_material(std::unique_ptr<IMaterial> material) {
        materials_.push_back(std::move(material));
        legacy_cache_valid_ = false;
    }

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

    // Legacy compatibility methods
    const std::vector<Material>& host() const;
    std::vector<Material>& host_mut();

    // Device operations (enhanced)
    bool upload_to_device();
    bool get_device_view(DeviceMaterialsView& out) const;
    void free_device();

    // Utility methods
    int count() const { return static_cast<int>(materials_.size()); }
    void clear() {
        materials_.clear();
        legacy_cache_valid_ = false;
    }

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

  private:
    void invalidate_legacy_cache() { legacy_cache_valid_ = false; }
    void update_legacy_cache() const;
};

} // namespace matgpu