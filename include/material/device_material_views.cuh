#pragma once
#include "material/contact_models.hpp"
#include "material/device_qualifier.cuh"
#include "material/property_mixins.hpp"

#include <type_traits>

namespace matgpu {

// Forward declarations
template <typename ContactType, bool HasThermal, bool HasEM> class DeviceMaterialView;

// Type traits for compile-time optimization
template <typename T> struct device_material_traits {
    static constexpr bool has_thermal         = false;
    static constexpr bool has_electromagnetic = false;
    static constexpr bool is_eepa             = false;
    static constexpr bool is_jkr              = false;
};

// Specializations for contact models
template <> struct device_material_traits<DeviceEEPAContactView> {
    static constexpr bool has_thermal         = false;
    static constexpr bool has_electromagnetic = false;
    static constexpr bool is_eepa             = true;
    static constexpr bool is_jkr              = false;
};

template <> struct device_material_traits<DeviceJKRContactView> {
    static constexpr bool has_thermal         = false;
    static constexpr bool has_electromagnetic = false;
    static constexpr bool is_eepa             = false;
    static constexpr bool is_jkr              = true;
};

// Device-side elastic properties (always present)
struct DeviceElasticProperties {
    float young;
    float poisson;
    float density;

    DEVICE_QUALIFIER float E() const { return young; }
    DEVICE_QUALIFIER float nu() const { return poisson; }
    DEVICE_QUALIFIER float rho() const { return density; }

    // Derived properties
    DEVICE_QUALIFIER float shear_modulus() const { return young / (2.0f * (1.0f + poisson)); }

    DEVICE_QUALIFIER float bulk_modulus() const { return young / (3.0f * (1.0f - 2.0f * poisson)); }

    DEVICE_QUALIFIER float wave_speed() const { return sqrtf(young / density); }
};

// Template-based device material view using CRTP
template <typename ContactType, bool HasThermal = false, bool HasEM = false> class DeviceMaterialView {
  private:
    DeviceElasticProperties elastic_;
    ContactType contact_;
    DeviceOptional<DeviceThermalView> thermal_;
    DeviceOptional<DeviceElectromagneticView> em_;

  public:
    using contact_type                        = ContactType;
    static constexpr bool has_thermal         = HasThermal;
    static constexpr bool has_electromagnetic = HasEM;
    static constexpr bool is_eepa             = device_material_traits<ContactType>::is_eepa;
    static constexpr bool is_jkr              = device_material_traits<ContactType>::is_jkr;

    // Constructors
    DEVICE_QUALIFIER DeviceMaterialView() = default;

    DEVICE_QUALIFIER DeviceMaterialView(const DeviceElasticProperties& elastic, const ContactType& contact)
        : elastic_(elastic), contact_(contact) {}

    DEVICE_QUALIFIER DeviceMaterialView(const DeviceElasticProperties& elastic, const ContactType& contact,
                                        const DeviceThermalView& thermal)
        : elastic_(elastic), contact_(contact), thermal_(thermal) {}

    DEVICE_QUALIFIER DeviceMaterialView(const DeviceElasticProperties& elastic, const ContactType& contact,
                                        const DeviceElectromagneticView& em)
        : elastic_(elastic), contact_(contact), em_(em) {}

    DEVICE_QUALIFIER DeviceMaterialView(const DeviceElasticProperties& elastic, const ContactType& contact,
                                        const DeviceThermalView& thermal, const DeviceElectromagneticView& em)
        : elastic_(elastic), contact_(contact), thermal_(thermal), em_(em) {}

    // Elastic properties access
    DEVICE_QUALIFIER const DeviceElasticProperties& elastic() const { return elastic_; }
    DEVICE_QUALIFIER float young_modulus() const { return elastic_.E(); }
    DEVICE_QUALIFIER float poisson_ratio() const { return elastic_.nu(); }
    DEVICE_QUALIFIER float density() const { return elastic_.rho(); }

    // Contact model access
    DEVICE_QUALIFIER const ContactType& contact() const { return contact_; }

    // Contact force computation
    DEVICE_QUALIFIER float normal_force(float overlap, float velocity) const {
        return contact_.normal_force(overlap, velocity);
    }

    DEVICE_QUALIFIER float tangential_force(float tangential_overlap, float tangential_velocity) const {
        return contact_.tangential_force(tangential_overlap, tangential_velocity);
    }

    // Optional properties access (compile-time checked)
    template <bool Enabled = HasThermal>
    DEVICE_QUALIFIER typename std::enable_if<Enabled, const DeviceThermalView&>::type thermal() const {
        return thermal_.get();
    }

    template <bool Enabled = HasEM>
    DEVICE_QUALIFIER typename std::enable_if<Enabled, const DeviceElectromagneticView&>::type electromagnetic() const {
        return em_.get();
    }

    // Runtime optional properties check
    DEVICE_QUALIFIER bool has_thermal_properties() const { return thermal_.has_value(); }

    DEVICE_QUALIFIER bool has_electromagnetic_properties() const { return em_.has_value(); }

    // Thermal properties (if available)
    template <bool Enabled = HasThermal>
    DEVICE_QUALIFIER typename std::enable_if<Enabled, float>::type thermal_conductivity() const {
        return thermal_.get().k();
    }

    template <bool Enabled = HasThermal>
    DEVICE_QUALIFIER typename std::enable_if<Enabled, float>::type heat_capacity() const {
        return thermal_.get().cp();
    }

    // EM properties (if available)
    template <bool Enabled = HasEM>
    DEVICE_QUALIFIER typename std::enable_if<Enabled, float>::type permittivity() const {
        return em_.get().eps();
    }

    template <bool Enabled = HasEM>
    DEVICE_QUALIFIER typename std::enable_if<Enabled, float>::type permeability() const {
        return em_.get().mu();
    }

    template <bool Enabled = HasEM>
    DEVICE_QUALIFIER typename std::enable_if<Enabled, float>::type electrical_conductivity() const {
        return em_.get().sigma();
    }
};

// Convenience type aliases for common material configurations
using DeviceStandardMaterial = DeviceMaterialView<DeviceEEPAContactView, false, false>;
using DeviceEEPAMaterial     = DeviceMaterialView<DeviceEEPAContactView, false, false>;
using DeviceJKRMaterial      = DeviceMaterialView<DeviceJKRContactView, false, false>;

using DeviceThermalEEPAMaterial = DeviceMaterialView<DeviceEEPAContactView, true, false>;
using DeviceThermalJKRMaterial  = DeviceMaterialView<DeviceJKRContactView, true, false>;

using DeviceEMMaterial       = DeviceMaterialView<DeviceEEPAContactView, false, true>;
using DeviceCompleteMaterial = DeviceMaterialView<DeviceEEPAContactView, true, true>;

// Factory functions for creating device views
template <typename ContactType>
DEVICE_QUALIFIER auto make_device_material_view(const DeviceElasticProperties& elastic, const ContactType& contact) {
    return DeviceMaterialView<ContactType>(elastic, contact);
}

template <typename ContactType>
DEVICE_QUALIFIER auto make_device_material_view(const DeviceElasticProperties& elastic, const ContactType& contact,
                                                const DeviceThermalView& thermal) {
    return DeviceMaterialView<ContactType, true>(elastic, contact, thermal);
}

template <typename ContactType>
DEVICE_QUALIFIER auto make_device_material_view(const DeviceElasticProperties& elastic, const ContactType& contact,
                                                const DeviceElectromagneticView& em) {
    return DeviceMaterialView<ContactType, false, true>(elastic, contact, em);
}

template <typename ContactType>
DEVICE_QUALIFIER auto make_device_material_view(const DeviceElasticProperties& elastic, const ContactType& contact,
                                                const DeviceThermalView& thermal, const DeviceElectromagneticView& em) {
    return DeviceMaterialView<ContactType, true, true>(elastic, contact, thermal, em);
}

// Array-based device views for SoA (Structure of Arrays) storage
template <typename ContactType, bool HasThermal = false, bool HasEM = false> class DeviceMaterialArrayView {
  private:
    // Elastic properties arrays
    const float* young_;
    const float* poisson_;
    const float* density_;

    // Contact model arrays
    const ContactType* contact_models_;

    // Optional thermal arrays
    const DeviceThermalView* thermal_views_;
    const bool* has_thermal_;

    // Optional EM arrays
    const DeviceElectromagneticView* em_views_;
    const bool* has_em_;

    int count_;

  public:
    using contact_type                        = ContactType;
    static constexpr bool has_thermal         = HasThermal;
    static constexpr bool has_electromagnetic = HasEM;

    DEVICE_QUALIFIER DeviceMaterialArrayView(const float* young, const float* poisson, const float* density,
                                             const ContactType* contact_models, int count)
        : young_(young), poisson_(poisson), density_(density), contact_models_(contact_models), count_(count),
          thermal_views_(nullptr), has_thermal_(nullptr), em_views_(nullptr), has_em_(nullptr) {}

    DEVICE_QUALIFIER int size() const { return count_; }

    // Get individual material view by index
    DEVICE_QUALIFIER DeviceMaterialView<ContactType, HasThermal, HasEM> operator[](int id) const {
        DeviceElasticProperties elastic{young_[id], poisson_[id], density_[id]};

        if constexpr (HasThermal && HasEM) {
            DeviceThermalView thermal    = has_thermal_[id] ? thermal_views_[id] : DeviceThermalView{};
            DeviceElectromagneticView em = has_em_[id] ? em_views_[id] : DeviceElectromagneticView{};
            return DeviceMaterialView<ContactType, true, true>(elastic, contact_models_[id], thermal, em);
        } else if constexpr (HasThermal) {
            DeviceThermalView thermal = has_thermal_[id] ? thermal_views_[id] : DeviceThermalView{};
            return DeviceMaterialView<ContactType, true, false>(elastic, contact_models_[id], thermal);
        } else if constexpr (HasEM) {
            DeviceElectromagneticView em = has_em_[id] ? em_views_[id] : DeviceElectromagneticView{};
            return DeviceMaterialView<ContactType, false, true>(elastic, contact_models_[id], em);
        } else {
            return DeviceMaterialView<ContactType, false, false>(elastic, contact_models_[id]);
        }
    }

    // Direct array access for performance-critical code
    DEVICE_QUALIFIER float young_modulus(int id) const { return young_[id]; }
    DEVICE_QUALIFIER float poisson_ratio(int id) const { return poisson_[id]; }
    DEVICE_QUALIFIER float density(int id) const { return density_[id]; }
    DEVICE_QUALIFIER const ContactType& contact_model(int id) const { return contact_models_[id]; }

    // Optional properties with runtime checks
    DEVICE_QUALIFIER bool material_has_thermal(int id) const { return HasThermal && has_thermal_ && has_thermal_[id]; }

    DEVICE_QUALIFIER bool material_has_em(int id) const { return HasEM && has_em_ && has_em_[id]; }
};

// Template metaprogramming helpers for type deduction
template <typename T> struct is_device_material_view : std::false_type {};

template <typename ContactType, bool HasThermal, bool HasEM>
struct is_device_material_view<DeviceMaterialView<ContactType, HasThermal, HasEM>> : std::true_type {};

template <typename T> constexpr bool is_device_material_view_v = is_device_material_view<T>::value;

// Compile-time material property queries
template <typename MaterialView> constexpr bool material_has_thermal_v = MaterialView::has_thermal;

template <typename MaterialView> constexpr bool material_has_electromagnetic_v = MaterialView::has_electromagnetic;

template <typename MaterialView> constexpr bool material_is_eepa_v = MaterialView::is_eepa;

template <typename MaterialView> constexpr bool material_is_jkr_v = MaterialView::is_jkr;

} // namespace matgpu