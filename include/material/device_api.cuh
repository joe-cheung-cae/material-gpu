#pragma once
#include "material/device_qualifier.cuh"
#include "material/types.hpp"

#include <cuda_runtime.h>

namespace matgpu {

// Inline getters using an explicit device view argument
DEVICE_QUALIFIER int mat_count(const DeviceMaterialsView& V) { return V.count; }

DEVICE_QUALIFIER float mat_young(const DeviceMaterialsView& V, int id) { return V.young[id]; }
DEVICE_QUALIFIER float mat_poisson(const DeviceMaterialsView& V, int id) { return V.poisson[id]; }
DEVICE_QUALIFIER float mat_density(const DeviceMaterialsView& V, int id) { return V.density[id]; }
DEVICE_QUALIFIER ContactModelType mat_contact_model(const DeviceMaterialsView& V, int id) {
    return static_cast<ContactModelType>(V.model[id]);
}

// EEPA
DEVICE_QUALIFIER float mat_eepa_kn(const DeviceMaterialsView& V, int id) { return V.eepa_kn[id]; }
DEVICE_QUALIFIER float mat_eepa_kt(const DeviceMaterialsView& V, int id) { return V.eepa_kt[id]; }
DEVICE_QUALIFIER float mat_eepa_gamma_n(const DeviceMaterialsView& V, int id) { return V.eepa_gamma_n[id]; }
DEVICE_QUALIFIER float mat_eepa_gamma_t(const DeviceMaterialsView& V, int id) { return V.eepa_gamma_t[id]; }

// JKR
DEVICE_QUALIFIER float mat_jkr_work(const DeviceMaterialsView& V, int id) { return V.jkr_work[id]; }
DEVICE_QUALIFIER float mat_jkr_r0(const DeviceMaterialsView& V, int id) { return V.jkr_r0[id]; }

// Thermal (optional)
DEVICE_QUALIFIER bool mat_has_thermal(const DeviceMaterialsView& V, int id) { return V.has_thermal[id] != 0; }
DEVICE_QUALIFIER float mat_thermal_k(const DeviceMaterialsView& V, int id) { return V.thermal_k[id]; }
DEVICE_QUALIFIER float mat_thermal_cp(const DeviceMaterialsView& V, int id) { return V.thermal_cp[id]; }

// EM (optional)
DEVICE_QUALIFIER bool mat_has_em(const DeviceMaterialsView& V, int id) { return V.has_em[id] != 0; }
DEVICE_QUALIFIER float mat_em_eps(const DeviceMaterialsView& V, int id) { return V.em_eps[id]; }
DEVICE_QUALIFIER float mat_em_mu(const DeviceMaterialsView& V, int id) { return V.em_mu[id]; }
DEVICE_QUALIFIER float mat_em_sigma(const DeviceMaterialsView& V, int id) { return V.em_sigma[id]; }

} // namespace matgpu
