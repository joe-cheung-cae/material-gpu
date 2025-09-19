#pragma once
#include "material/device_qualifier.cuh"

#include <memory>
#include <string>

namespace matgpu {

// Mixin pattern for optional material properties

// Base class for optional properties
class OptionalProperty {
  public:
    virtual ~OptionalProperty()                             = default;
    virtual bool is_enabled() const                         = 0;
    virtual std::string property_name() const               = 0;
    virtual std::unique_ptr<OptionalProperty> clone() const = 0;
};

// Thermal properties mixin
class ThermalProperties : public OptionalProperty {
  private:
    float conductivity_;  // W/(m·K)
    float heat_capacity_; // J/(kg·K)
    bool enabled_;

  public:
    ThermalProperties(float conductivity = 10.0f, float heat_capacity = 900.0f, bool enabled = true)
        : conductivity_(conductivity), heat_capacity_(heat_capacity), enabled_(enabled) {}

    // OptionalProperty interface
    bool is_enabled() const override { return enabled_; }
    std::string property_name() const override { return "thermal"; }

    std::unique_ptr<OptionalProperty> clone() const override {
        return std::make_unique<ThermalProperties>(conductivity_, heat_capacity_, enabled_);
    }

    // Thermal-specific interface
    float conductivity() const { return conductivity_; }
    float heat_capacity() const { return heat_capacity_; }

    void set_conductivity(float value) { conductivity_ = value; }
    void set_heat_capacity(float value) { heat_capacity_ = value; }
    void set_enabled(bool enabled) { enabled_ = enabled; }

    // Convenience setters
    void set_thermal_properties(float conductivity, float heat_capacity) {
        conductivity_  = conductivity;
        heat_capacity_ = heat_capacity;
        enabled_       = true;
    }

    void disable() { enabled_ = false; }
    void enable() { enabled_ = true; }
};

// Electromagnetic properties mixin
class ElectromagneticProperties : public OptionalProperty {
  private:
    float permittivity_; // F/m (relative permittivity * ε₀)
    float permeability_; // H/m (relative permeability * μ₀)
    float conductivity_; // S/m (electrical conductivity)
    bool enabled_;

  public:
    ElectromagneticProperties(float permittivity = 8.85e-12f, float permeability = 1.26e-6f, float conductivity = 1e-6f,
                              bool enabled = true)
        : permittivity_(permittivity), permeability_(permeability), conductivity_(conductivity), enabled_(enabled) {}

    // OptionalProperty interface
    bool is_enabled() const override { return enabled_; }
    std::string property_name() const override { return "electromagnetic"; }

    std::unique_ptr<OptionalProperty> clone() const override {
        return std::make_unique<ElectromagneticProperties>(permittivity_, permeability_, conductivity_, enabled_);
    }

    // EM-specific interface
    float permittivity() const { return permittivity_; }
    float permeability() const { return permeability_; }
    float conductivity() const { return conductivity_; }

    void set_permittivity(float value) { permittivity_ = value; }
    void set_permeability(float value) { permeability_ = value; }
    void set_conductivity(float value) { conductivity_ = value; }
    void set_enabled(bool enabled) { enabled_ = enabled; }

    // Convenience setters
    void set_em_properties(float permittivity, float permeability, float conductivity) {
        permittivity_ = permittivity;
        permeability_ = permeability;
        conductivity_ = conductivity;
        enabled_      = true;
    }

    void disable() { enabled_ = false; }
    void enable() { enabled_ = true; }
};

// CRTP Mixin base for optional properties
template <typename Derived, typename PropertyType> class PropertyMixin {
  private:
    std::unique_ptr<PropertyType> property_;

  public:
    // Property management
    bool has_property() const { return property_ && property_->is_enabled(); }

    const PropertyType* get_property() const { return has_property() ? property_.get() : nullptr; }

    PropertyType* get_property_mut() { return has_property() ? property_.get() : nullptr; }

    void set_property(std::unique_ptr<PropertyType> prop) { property_ = std::move(prop); }

    template <typename... Args> void emplace_property(Args&&... args) {
        property_ = std::make_unique<PropertyType>(std::forward<Args>(args)...);
    }

    void remove_property() { property_.reset(); }

    // Enable fluent interface
    Derived& with_property(std::unique_ptr<PropertyType> prop) {
        set_property(std::move(prop));
        return static_cast<Derived&>(*this);
    }

    template <typename... Args> Derived& with_property(Args&&... args) {
        emplace_property(std::forward<Args>(args)...);
        return static_cast<Derived&>(*this);
    }

    Derived& without_property() {
        remove_property();
        return static_cast<Derived&>(*this);
    }
};

// Specialized mixins for specific property types
template <typename Derived> using ThermalMixin = PropertyMixin<Derived, ThermalProperties>;

template <typename Derived> using ElectromagneticMixin = PropertyMixin<Derived, ElectromagneticProperties>;

// Combined mixin for materials that support both thermal and EM properties
template <typename Derived>
class CompleteMaterialMixin : public ThermalMixin<Derived>, public ElectromagneticMixin<Derived> {
  public:
    // Convenience methods for accessing specific properties
    const ThermalProperties* thermal() const { return ThermalMixin<Derived>::get_property(); }

    const ElectromagneticProperties* electromagnetic() const { return ElectromagneticMixin<Derived>::get_property(); }

    ThermalProperties* thermal_mut() { return ThermalMixin<Derived>::get_property_mut(); }

    ElectromagneticProperties* electromagnetic_mut() { return ElectromagneticMixin<Derived>::get_property_mut(); }

    bool has_thermal() const { return ThermalMixin<Derived>::has_property(); }

    bool has_electromagnetic() const { return ElectromagneticMixin<Derived>::has_property(); }

    // Fluent interface for chaining
    Derived& with_thermal(float conductivity, float heat_capacity) {
        ThermalMixin<Derived>::emplace_property(conductivity, heat_capacity);
        return static_cast<Derived&>(*this);
    }

    Derived& with_electromagnetic(float permittivity, float permeability, float conductivity) {
        ElectromagneticMixin<Derived>::emplace_property(permittivity, permeability, conductivity);
        return static_cast<Derived&>(*this);
    }

    Derived& without_thermal() {
        ThermalMixin<Derived>::remove_property();
        return static_cast<Derived&>(*this);
    }

    Derived& without_electromagnetic() {
        ElectromagneticMixin<Derived>::remove_property();
        return static_cast<Derived&>(*this);
    }
};

// Device-side views for optional properties
struct DeviceThermalView {
    float conductivity;
    float heat_capacity;
    bool enabled;

    DEVICE_QUALIFIER bool has_thermal() const { return enabled; }
    DEVICE_QUALIFIER float k() const { return conductivity; }
    DEVICE_QUALIFIER float cp() const { return heat_capacity; }
};

struct DeviceElectromagneticView {
    float permittivity;
    float permeability;
    float conductivity;
    bool enabled;

    DEVICE_QUALIFIER bool has_em() const { return enabled; }
    DEVICE_QUALIFIER float eps() const { return permittivity; }
    DEVICE_QUALIFIER float mu() const { return permeability; }
    DEVICE_QUALIFIER float sigma() const { return conductivity; }
};

// Optional wrapper for device views (similar to std::optional but for device code)
template <typename T> struct DeviceOptional {
    T value;
    bool present;

    DEVICE_QUALIFIER DeviceOptional() : present(false) {}
    DEVICE_QUALIFIER DeviceOptional(const T& val) : value(val), present(true) {}

    DEVICE_QUALIFIER bool has_value() const { return present; }
    DEVICE_QUALIFIER const T& get() const { return value; }
    DEVICE_QUALIFIER T& get() { return value; }

    DEVICE_QUALIFIER explicit operator bool() const { return present; }
    DEVICE_QUALIFIER const T& operator*() const { return value; }
    DEVICE_QUALIFIER T& operator*() { return value; }
};

} // namespace matgpu