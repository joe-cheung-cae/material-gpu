#pragma once
#include "material/device_qualifier.cuh"

#include <functional>
#include <memory>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <vector>

namespace matgpu {

// Forward declarations
class ContactModel;
class ThermalProperties;
class ElectromagneticProperties;

// Base interface for all materials
class IMaterial {
  public:
    virtual ~IMaterial() = default;

    // Essential elastic properties - all materials must have these
    virtual float young_modulus() const = 0;
    virtual float poisson_ratio() const = 0;
    virtual float density() const       = 0;

    // Contact model - returns polymorphic contact behavior
    virtual const ContactModel& contact_model() const = 0;

    // Optional properties - return nullptr if not supported
    virtual const ThermalProperties* thermal_properties() const { return nullptr; }
    virtual const ElectromagneticProperties* electromagnetic_properties() const { return nullptr; }

    // Material identification
    virtual std::string material_type() const = 0;
    virtual size_t material_id() const        = 0;
    virtual void set_material_id(size_t id)   = 0;

    // Device compatibility
    virtual bool is_device_compatible() const = 0;
};

// Abstract base class implementing common functionality
class AbstractMaterial : public IMaterial {
  protected:
    float young_;
    float poisson_;
    float density_;
    size_t id_;

  public:
    AbstractMaterial(float young, float poisson, float density, size_t id = 0)
        : young_(young), poisson_(poisson), density_(density), id_(id) {}

    // IMaterial implementation
    float young_modulus() const override { return young_; }
    float poisson_ratio() const override { return poisson_; }
    float density() const override { return density_; }
    size_t material_id() const override { return id_; }
    bool is_device_compatible() const override { return true; }

    // Setters for dynamic modification
    void set_elastic_properties(float young, float poisson, float density) {
        young_   = young;
        poisson_ = poisson;
        density_ = density;
    }

    void set_material_id(size_t id) override { id_ = id; }
};

// CRTP base for type-safe polymorphism with zero runtime overhead
template <typename Derived> class MaterialCRTP : public AbstractMaterial {
  public:
    using AbstractMaterial::AbstractMaterial;

    std::string material_type() const override { return typeid(Derived).name(); }

    // Static downcasting for performance-critical paths
    const Derived* as_derived() const { return static_cast<const Derived*>(this); }

    Derived* as_derived() { return static_cast<Derived*>(this); }
};

// Device-side interface for CUDA kernels
template <typename T> struct DeviceMaterialInterface {
    DEVICE_QUALIFIER float young_modulus() const { return static_cast<const T*>(this)->young_modulus_impl(); }

    DEVICE_QUALIFIER float poisson_ratio() const { return static_cast<const T*>(this)->poisson_ratio_impl(); }

    DEVICE_QUALIFIER float density() const { return static_cast<const T*>(this)->density_impl(); }
};

// Compile-time material type identification
template <typename T> struct MaterialTraits {
    static constexpr bool has_thermal         = false;
    static constexpr bool has_electromagnetic = false;
    static constexpr bool has_eepa_contact    = false;
    static constexpr bool has_jkr_contact     = false;
};

// Type registration system for factory pattern
class MaterialTypeRegistry {
  public:
    using CreateFunc = std::function<std::unique_ptr<IMaterial>(const std::string&)>;

    static MaterialTypeRegistry& instance() {
        static MaterialTypeRegistry registry;
        return registry;
    }

    void register_type(const std::string& type_name, CreateFunc creator) { creators_[type_name] = std::move(creator); }

    std::unique_ptr<IMaterial> create(const std::string& type_name, const std::string& params) {
        auto it = creators_.find(type_name);
        if (it != creators_.end()) {
            return it->second(params);
        }
        return nullptr;
    }

    std::vector<std::string> registered_types() const {
        std::vector<std::string> types;
        for (const auto& pair : creators_) {
            types.push_back(pair.first);
        }
        return types;
    }

  private:
    std::unordered_map<std::string, CreateFunc> creators_;
};

// RAII helper for automatic type registration
template <typename MaterialType> class MaterialRegistrar {
  public:
    MaterialRegistrar(const std::string& type_name) {
        MaterialTypeRegistry::instance().register_type(type_name,
                                                       [](const std::string& params) -> std::unique_ptr<IMaterial> {
                                                           return MaterialType::create_from_string(params);
                                                       });
    }
};

// Macro for easy material type registration
#define REGISTER_MATERIAL_TYPE(MaterialClass, TypeName)                                                                \
    static MaterialRegistrar<MaterialClass> g_##MaterialClass##_registrar(TypeName)

} // namespace matgpu