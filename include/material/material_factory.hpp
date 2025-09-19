#pragma once
#include "material/contact_models.hpp"
#include "material/material_base.hpp"
#include "material/property_mixins.hpp"

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace matgpu {

// Forward declarations of concrete material types
class StandardMaterial;
class EEPAMaterial;
class JKRMaterial;
class CompositeMaterial;

// Concrete material implementations using the new architecture
class StandardMaterial : public MaterialCRTP<StandardMaterial>, public CompleteMaterialMixin<StandardMaterial> {
  private:
    std::unique_ptr<ContactModel> contact_model_;

  public:
    StandardMaterial(float young, float poisson, float density, size_t id = 0)
        : MaterialCRTP<StandardMaterial>(young, poisson, density, id),
          contact_model_(std::make_unique<NoContactModel>()) {}

    // IMaterial interface implementation
    const ContactModel& contact_model() const override { return *contact_model_; }

    const ThermalProperties* thermal_properties() const override { return thermal(); }

    const ElectromagneticProperties* electromagnetic_properties() const override { return electromagnetic(); }

    std::string material_type() const override { return "StandardMaterial"; }

    // Set contact model
    void set_contact_model(std::unique_ptr<ContactModel> model) { contact_model_ = std::move(model); }

    // Fluent interface
    StandardMaterial& with_contact_model(std::unique_ptr<ContactModel> model) {
        set_contact_model(std::move(model));
        return *this;
    }

    // Factory method for JSON creation
    static std::unique_ptr<IMaterial> create_from_string(const std::string& json_params);
};

// Specialized material for EEPA contact model
class EEPAMaterial : public MaterialCRTP<EEPAMaterial>, public CompleteMaterialMixin<EEPAMaterial> {
  private:
    std::unique_ptr<EEPAContactModel> eepa_model_;

  public:
    EEPAMaterial(float young, float poisson, float density, float kn, float kt, float gamma_n, float gamma_t,
                 size_t id = 0)
        : MaterialCRTP<EEPAMaterial>(young, poisson, density, id),
          eepa_model_(std::make_unique<EEPAContactModel>(kn, kt, gamma_n, gamma_t)) {}

    // IMaterial interface implementation
    const ContactModel& contact_model() const override { return *eepa_model_; }

    const ThermalProperties* thermal_properties() const override { return thermal(); }

    const ElectromagneticProperties* electromagnetic_properties() const override { return electromagnetic(); }

    std::string material_type() const override { return "EEPAMaterial"; }

    // Direct access to EEPA parameters
    EEPAContactModel* eepa_contact_model() { return eepa_model_.get(); }
    const EEPAContactModel* eepa_contact_model() const { return eepa_model_.get(); }

    static std::unique_ptr<IMaterial> create_from_string(const std::string& json_params);
};

// Specialized material for JKR contact model
class JKRMaterial : public MaterialCRTP<JKRMaterial>, public CompleteMaterialMixin<JKRMaterial> {
  private:
    std::unique_ptr<JKRContactModel> jkr_model_;

  public:
    JKRMaterial(float young, float poisson, float density, float work_of_adhesion, float contact_radius0, size_t id = 0)
        : MaterialCRTP<JKRMaterial>(young, poisson, density, id),
          jkr_model_(std::make_unique<JKRContactModel>(work_of_adhesion, contact_radius0)) {}

    // IMaterial interface implementation
    const ContactModel& contact_model() const override { return *jkr_model_; }

    const ThermalProperties* thermal_properties() const override { return thermal(); }

    const ElectromagneticProperties* electromagnetic_properties() const override { return electromagnetic(); }

    std::string material_type() const override { return "JKRMaterial"; }

    // Direct access to JKR parameters
    JKRContactModel* jkr_contact_model() { return jkr_model_.get(); }
    const JKRContactModel* jkr_contact_model() const { return jkr_model_.get(); }

    static std::unique_ptr<IMaterial> create_from_string(const std::string& json_params);
};

// Builder pattern for complex material construction
class MaterialBuilder {
  private:
    float young_   = 1e7f;
    float poisson_ = 0.25f;
    float density_ = 2500.0f;
    size_t id_     = 0;

    std::unique_ptr<ContactModel> contact_model_;
    std::unique_ptr<ThermalProperties> thermal_;
    std::unique_ptr<ElectromagneticProperties> em_;

    std::string material_type_ = "standard";

  public:
    MaterialBuilder() = default;

    // Elastic properties
    MaterialBuilder& elastic(float young, float poisson, float density) {
        young_   = young;
        poisson_ = poisson;
        density_ = density;
        return *this;
    }

    MaterialBuilder& young_modulus(float young) {
        young_ = young;
        return *this;
    }

    MaterialBuilder& poisson_ratio(float poisson) {
        poisson_ = poisson;
        return *this;
    }

    MaterialBuilder& density(float density) {
        density_ = density;
        return *this;
    }

    MaterialBuilder& id(size_t id) {
        id_ = id;
        return *this;
    }

    // Contact models
    MaterialBuilder& no_contact() {
        contact_model_ = std::make_unique<NoContactModel>();
        material_type_ = "standard";
        return *this;
    }

    MaterialBuilder& eepa_contact(float kn, float kt, float gamma_n, float gamma_t) {
        contact_model_ = std::make_unique<EEPAContactModel>(kn, kt, gamma_n, gamma_t);
        material_type_ = "eepa";
        return *this;
    }

    MaterialBuilder& jkr_contact(float work_of_adhesion, float contact_radius0) {
        contact_model_ = std::make_unique<JKRContactModel>(work_of_adhesion, contact_radius0);
        material_type_ = "jkr";
        return *this;
    }

    MaterialBuilder& contact_model(std::unique_ptr<ContactModel> model) {
        contact_model_ = std::move(model);
        return *this;
    }

    // Set material type for factory selection
    MaterialBuilder& material_type(const std::string& type) {
        material_type_ = type;
        return *this;
    }

    // Optional properties
    MaterialBuilder& thermal(float conductivity, float heat_capacity) {
        thermal_ = std::make_unique<ThermalProperties>(conductivity, heat_capacity);
        return *this;
    }

    MaterialBuilder& electromagnetic(float permittivity, float permeability, float conductivity) {
        em_ = std::make_unique<ElectromagneticProperties>(permittivity, permeability, conductivity);
        return *this;
    }

    MaterialBuilder& thermal_properties(std::unique_ptr<ThermalProperties> thermal) {
        thermal_ = std::move(thermal);
        return *this;
    }

    MaterialBuilder& em_properties(std::unique_ptr<ElectromagneticProperties> em) {
        em_ = std::move(em);
        return *this;
    }

    // Build methods for different material types
    std::unique_ptr<IMaterial> build() {
        if (material_type_ == "eepa") {
            return build_eepa();
        } else if (material_type_ == "jkr") {
            return build_jkr();
        } else {
            return build_standard();
        }
    }

    std::unique_ptr<StandardMaterial> build_standard() {
        auto material = std::make_unique<StandardMaterial>(young_, poisson_, density_, id_);

        if (contact_model_) {
            material->set_contact_model(std::move(contact_model_));
        }

        if (thermal_) {
            material->ThermalMixin<StandardMaterial>::set_property(std::move(thermal_));
        }

        if (em_) {
            material->ElectromagneticMixin<StandardMaterial>::set_property(std::move(em_));
        }

        return material;
    }

    std::unique_ptr<EEPAMaterial> build_eepa() {
        // Extract EEPA parameters from contact model
        auto* eepa = dynamic_cast<EEPAContactModel*>(contact_model_.get());
        if (!eepa) {
            throw std::runtime_error("EEPA contact model required for EEPAMaterial");
        }

        auto material = std::make_unique<EEPAMaterial>(young_, poisson_, density_, eepa->kn(), eepa->kt(),
                                                       eepa->gamma_n(), eepa->gamma_t(), id_);

        if (thermal_) {
            material->ThermalMixin<EEPAMaterial>::set_property(std::move(thermal_));
        }

        if (em_) {
            material->ElectromagneticMixin<EEPAMaterial>::set_property(std::move(em_));
        }

        return material;
    }

    std::unique_ptr<JKRMaterial> build_jkr() {
        // Extract JKR parameters from contact model
        auto* jkr = dynamic_cast<JKRContactModel*>(contact_model_.get());
        if (!jkr) {
            throw std::runtime_error("JKR contact model required for JKRMaterial");
        }

        auto material = std::make_unique<JKRMaterial>(young_, poisson_, density_, jkr->work_of_adhesion(),
                                                      jkr->contact_radius0(), id_);

        if (thermal_) {
            material->ThermalMixin<JKRMaterial>::set_property(std::move(thermal_));
        }

        if (em_) {
            material->ElectromagneticMixin<JKRMaterial>::set_property(std::move(em_));
        }

        return material;
    }

    // Reset builder for reuse
    MaterialBuilder& reset() {
        young_   = 1e7f;
        poisson_ = 0.25f;
        density_ = 2500.0f;
        id_      = 0;
        contact_model_.reset();
        thermal_.reset();
        em_.reset();
        material_type_ = "standard";
        return *this;
    }
};

// Factory for creating materials
class MaterialFactory {
  public:
    using CreateFunction = std::function<std::unique_ptr<IMaterial>(MaterialBuilder&)>;

    static MaterialFactory& instance() {
        static MaterialFactory factory;
        return factory;
    }

    // Register material types
    void register_material_type(const std::string& type_name, CreateFunction creator) {
        creators_[type_name] = std::move(creator);
    }

    // Create material from type name and builder
    std::unique_ptr<IMaterial> create(const std::string& type_name, MaterialBuilder& builder) {
        auto it = creators_.find(type_name);
        if (it != creators_.end()) {
            return it->second(builder);
        }
        return nullptr;
    }

    // Convenience factory methods
    static std::unique_ptr<IMaterial> create_standard(float young, float poisson, float density) {
        return MaterialBuilder().elastic(young, poisson, density).no_contact().build();
    }

    static std::unique_ptr<IMaterial> create_eepa(float young, float poisson, float density, float kn, float kt,
                                                  float gamma_n, float gamma_t) {
        return MaterialBuilder().elastic(young, poisson, density).eepa_contact(kn, kt, gamma_n, gamma_t).build();
    }

    static std::unique_ptr<IMaterial> create_jkr(float young, float poisson, float density, float work_of_adhesion,
                                                 float contact_radius0) {
        return MaterialBuilder()
            .elastic(young, poisson, density)
            .jkr_contact(work_of_adhesion, contact_radius0)
            .build();
    }

    // Get list of registered types
    std::vector<std::string> registered_types() const {
        std::vector<std::string> types;
        for (const auto& pair : creators_) {
            types.push_back(pair.first);
        }
        return types;
    }

  private:
    std::unordered_map<std::string, CreateFunction> creators_;

    MaterialFactory() {
        // Register default material types
        register_material_type("standard", [](MaterialBuilder& builder) { return builder.build_standard(); });

        register_material_type("eepa", [](MaterialBuilder& builder) { return builder.build_eepa(); });

        register_material_type("jkr", [](MaterialBuilder& builder) { return builder.build_jkr(); });
    }
};

// Registration macros for easy extension
#define REGISTER_MATERIAL_FACTORY(MaterialClass, TypeName)                                                             \
    static bool g_##MaterialClass##_registered = []() {                                                                \
        MaterialFactory::instance().register_material_type(                                                            \
            TypeName, [](const MaterialBuilder& builder) -> std::unique_ptr<IMaterial> {                               \
                return std::make_unique<MaterialClass>(builder);                                                       \
            });                                                                                                        \
        return true;                                                                                                   \
    }()

} // namespace matgpu