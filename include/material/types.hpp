#pragma once
#include <cstdint>

namespace matgpu {

enum class ContactModelType : uint8_t { None = 0, EEPA = 1, JKR = 2 };

struct ElasticProps {
    float young;   // Young's modulus (Pa)
    float poisson; // Poisson's ratio
    float density; // kg/m^3
};

struct EEPAProps {
    float kn;      // normal stiffness
    float kt;      // tangential stiffness
    float gamma_n; // normal damping
    float gamma_t; // tangential damping
};

struct JKRProps {
    float work_of_adhesion; // J/m^2
    float contact_radius0;  // initial radius (optional use)
};

struct ThermalProps {
    float conductivity;  // W/(m K)
    float heat_capacity; // J/(kg K)
};

struct EMProps {
    float permittivity; // F/m
    float permeability; // H/m
    float conductivity; // S/m
};

// Host-side material definition (AoS)
struct Material {
    ElasticProps elastic{};
    ContactModelType model{ContactModelType::None};
    EEPAProps eepa{}; // valid if model==EEPA
    JKRProps jkr{};   // valid if model==JKR
    bool has_thermal{false};
    ThermalProps thermal{};
    bool has_em{false};
    EMProps em{};
};

// Device view uses SoA for coalesced access
struct DeviceMaterialsView {
    // Base elastic
    const float* young{};
    const float* poisson{};
    const float* density{};
    // Contact model tag per material
    const uint8_t* model{}; // ContactModelType
    // EEPA arrays
    const float* eepa_kn{};
    const float* eepa_kt{};
    const float* eepa_gamma_n{};
    const float* eepa_gamma_t{};
    // JKR arrays
    const float* jkr_work{};
    const float* jkr_r0{};
    // Optional thermal
    const uint8_t* has_thermal{}; // 0/1
    const float* thermal_k{};
    const float* thermal_cp{};
    // Optional EM
    const uint8_t* has_em{}; // 0/1
    const float* em_eps{};
    const float* em_mu{};
    const float* em_sigma{};
    // Count
    int count{0};
};

// Device qualifier macro (host/Device unified)
#include "material/device_qualifier.cuh"

// Lightweight device-side optional wrapper with presence flag
template <typename T> struct DeviceOption {
    T val{};
    bool present{false};
    DEVICE_QUALIFIER bool has() const { return present; }
    DEVICE_QUALIFIER const T& get() const { return val; }
};

// Per-capability views (device-side API surface)
struct ElasticView {
    const float* young{};
    const float* poisson{};
    const float* density{};
    DEVICE_QUALIFIER float E(int id) const { return young[id]; }
    DEVICE_QUALIFIER float nu(int id) const { return poisson[id]; }
    DEVICE_QUALIFIER float rho(int id) const { return density[id]; }
};

struct EEPAView {
    const float* kn{};
    const float* kt{};
    const float* gamma_n{};
    const float* gamma_t{};
    DEVICE_QUALIFIER float KN(int id) const { return kn[id]; }
    DEVICE_QUALIFIER float KT(int id) const { return kt[id]; }
    DEVICE_QUALIFIER float GN(int id) const { return gamma_n[id]; }
    DEVICE_QUALIFIER float GT(int id) const { return gamma_t[id]; }
};

struct JKRView {
    const float* work{};
    const float* r0{};
    DEVICE_QUALIFIER float W(int id) const { return work[id]; }
    DEVICE_QUALIFIER float R0(int id) const { return r0[id]; }
};

struct ThermalView {
    const float* k{};
    const float* cp{}; // present only when has_thermal[id]==1
    const uint8_t* has{};
    DEVICE_QUALIFIER bool has_props(int id) const { return has[id] != 0; }
    DEVICE_QUALIFIER float conductivity(int id) const { return k[id]; }
    DEVICE_QUALIFIER float heat_capacity(int id) const { return cp[id]; }
};

struct EMView {
    const float* eps{};
    const float* mu{};
    const float* sigma{}; // present only when has_em[id]==1
    const uint8_t* has{};
    DEVICE_QUALIFIER bool has_props(int id) const { return has[id] != 0; }
    DEVICE_QUALIFIER float permittivity(int id) const { return eps[id]; }
    DEVICE_QUALIFIER float permeability(int id) const { return mu[id]; }
    DEVICE_QUALIFIER float conductivity(int id) const { return sigma[id]; }
};

// A material view that only exposes capabilities available for a given id.
struct MaterialView {
    ElasticView elastic{};
    DeviceOption<EEPAView> eepa{};
    DeviceOption<JKRView> jkr{};
    DeviceOption<ThermalView> thermal{};
    DeviceOption<EMView> em{};
};

} // namespace matgpu
