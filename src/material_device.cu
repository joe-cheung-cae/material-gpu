#include "material/device_api.cuh"
#include "material/types.hpp"

#include <cstring>
#include <cuda_runtime.h>
#include <vector>

namespace matgpu {

// Host-side storage for device arrays
namespace {

struct DevArrays {
    // Elastic properties
    float*   young;
    float*   poisson;
    float*   density;

    // Model type
    uint8_t* model;

    // EEPA parameters
    float*   eepa_kn;
    float*   eepa_kt;
    float*   eepa_gamma_n;
    float*   eepa_gamma_t;

    // JKR parameters
    float*   jkr_work;
    float*   jkr_r0;

    // Thermal properties
    uint8_t* has_thermal;
    float*   thermal_k;
    float*   thermal_cp;

    // Electromagnetic properties
    uint8_t* has_em;
    float*   em_eps;
    float*   em_mu;
    float*   em_sigma;

    // Number of materials
    int      count;
    
    DevArrays() : young(nullptr), poisson(nullptr), density(nullptr), model(nullptr),
                  eepa_kn(nullptr), eepa_kt(nullptr), eepa_gamma_n(nullptr), eepa_gamma_t(nullptr),
                  jkr_work(nullptr), jkr_r0(nullptr), has_thermal(nullptr), thermal_k(nullptr),
                  thermal_cp(nullptr), has_em(nullptr), em_eps(nullptr), em_mu(nullptr),
                  em_sigma(nullptr), count(0) {}
};

DevArrays g_dev;

} // namespace

// No global symbol; we pass DeviceMaterialsView explicitly

static bool alloc_arrays(int n) {
    g_dev.count = n;
    auto alloc = [](auto** p, size_t nbytes) { return cudaMalloc((void**)p, nbytes) == cudaSuccess; };
    auto alloc_u8 = [](uint8_t** p, size_t nbytes) { return cudaMalloc((void**)p, nbytes) == cudaSuccess; };
    size_t nf = sizeof(float) * n;
    size_t nu = sizeof(uint8_t) * n;
    return alloc(&g_dev.young, nf) && alloc(&g_dev.poisson, nf) && alloc(&g_dev.density, nf) &&
           alloc_u8(&g_dev.model, nu) && alloc(&g_dev.eepa_kn, nf) && alloc(&g_dev.eepa_kt, nf) &&
           alloc(&g_dev.eepa_gamma_n, nf) && alloc(&g_dev.eepa_gamma_t, nf) && alloc(&g_dev.jkr_work, nf) &&
           alloc(&g_dev.jkr_r0, nf) && alloc_u8(&g_dev.has_thermal, nu) && alloc(&g_dev.thermal_k, nf) &&
           alloc(&g_dev.thermal_cp, nf) && alloc_u8(&g_dev.has_em, nu) && alloc(&g_dev.em_eps, nf) &&
           alloc(&g_dev.em_mu, nf) && alloc(&g_dev.em_sigma, nf);
}

static void free_arrays() {
    auto fre = [](auto* p) {
        if (p)
            cudaFree(p);
    };
    fre(g_dev.young);
    fre(g_dev.poisson);
    fre(g_dev.density);
    fre(g_dev.model);
    fre(g_dev.eepa_kn);
    fre(g_dev.eepa_kt);
    fre(g_dev.eepa_gamma_n);
    fre(g_dev.eepa_gamma_t);
    fre(g_dev.jkr_work);
    fre(g_dev.jkr_r0);
    fre(g_dev.has_thermal);
    fre(g_dev.thermal_k);
    fre(g_dev.thermal_cp);
    fre(g_dev.has_em);
    fre(g_dev.em_eps);
    fre(g_dev.em_mu);
    fre(g_dev.em_sigma);
    std::memset(&g_dev, 0, sizeof(g_dev));
}

static bool copy_to_device(const std::vector<Material>& m) {
    int n = (int)m.size();
    if (n == 0)
        return false;
    if (!alloc_arrays(n)) {
        free_arrays();
        return false;
    }
    // Prepare host SoA buffers
    std::vector<float> young(n), poisson(n), density(n);
    std::vector<uint8_t> model(n), has_thermal(n), has_em(n);
    std::vector<float> eepa_kn(n), eepa_kt(n), eepa_gn(n), eepa_gt(n);
    std::vector<float> jkr_work(n), jkr_r0(n);
    std::vector<float> tk(n), tcp(n), eps(n), mu(n), sigma(n);
    for (int i = 0; i < n; ++i) {
        young[i]       = m[i].elastic.young;
        poisson[i]     = m[i].elastic.poisson;
        density[i]     = m[i].elastic.density;
        model[i]       = (uint8_t)m[i].model;
        eepa_kn[i]     = m[i].eepa.kn;
        eepa_kt[i]     = m[i].eepa.kt;
        eepa_gn[i]     = m[i].eepa.gamma_n;
        eepa_gt[i]     = m[i].eepa.gamma_t;
        jkr_work[i]    = m[i].jkr.work_of_adhesion;
        jkr_r0[i]      = m[i].jkr.contact_radius0;
        has_thermal[i] = m[i].has_thermal ? 1 : 0;
        tk[i]          = m[i].thermal.conductivity;
        tcp[i]         = m[i].thermal.heat_capacity;
        has_em[i]      = m[i].has_em ? 1 : 0;
        eps[i]         = m[i].em.permittivity;
        mu[i]          = m[i].em.permeability;
        sigma[i]       = m[i].em.conductivity;
    }
    auto cpy = [](auto* dst, const auto* src, size_t nbytes) {
        return cudaMemcpy(dst, src, nbytes, cudaMemcpyHostToDevice) == cudaSuccess;
    };
    size_t nf = sizeof(float) * n;
    size_t nu = sizeof(uint8_t) * n;
    bool ok   = cpy(g_dev.young, young.data(), nf) && cpy(g_dev.poisson, poisson.data(), nf) &&
              cpy(g_dev.density, density.data(), nf) && cpy(g_dev.model, model.data(), nu) &&
              cpy(g_dev.eepa_kn, eepa_kn.data(), nf) && cpy(g_dev.eepa_kt, eepa_kt.data(), nf) &&
              cpy(g_dev.eepa_gamma_n, eepa_gn.data(), nf) && cpy(g_dev.eepa_gamma_t, eepa_gt.data(), nf) &&
              cpy(g_dev.jkr_work, jkr_work.data(), nf) && cpy(g_dev.jkr_r0, jkr_r0.data(), nf) &&
              cpy(g_dev.has_thermal, has_thermal.data(), nu) && cpy(g_dev.thermal_k, tk.data(), nf) &&
              cpy(g_dev.thermal_cp, tcp.data(), nf) && cpy(g_dev.has_em, has_em.data(), nu) &&
              cpy(g_dev.em_eps, eps.data(), nf) && cpy(g_dev.em_mu, mu.data(), nf) &&
              cpy(g_dev.em_sigma, sigma.data(), nf);
    if (!ok) {
        free_arrays();
        return false;
    }

    // No need to upload view; we will construct it on host when launching kernels
    return true;
}

bool device_upload(const std::vector<Material>& materials) {
    free_arrays();
    if (materials.empty())
        return false;
    return copy_to_device(materials);
}

void device_free() { free_arrays(); }

int device_count() { return g_dev.count; }

bool get_device_view(DeviceMaterialsView& out) {
    if (g_dev.count <= 0)
        return false;
    out.young        = g_dev.young;
    out.poisson      = g_dev.poisson;
    out.density      = g_dev.density;
    out.model        = g_dev.model;
    out.eepa_kn      = g_dev.eepa_kn;
    out.eepa_kt      = g_dev.eepa_kt;
    out.eepa_gamma_n = g_dev.eepa_gamma_n;
    out.eepa_gamma_t = g_dev.eepa_gamma_t;
    out.jkr_work     = g_dev.jkr_work;
    out.jkr_r0       = g_dev.jkr_r0;
    out.has_thermal  = g_dev.has_thermal;
    out.thermal_k    = g_dev.thermal_k;
    out.thermal_cp   = g_dev.thermal_cp;
    out.has_em       = g_dev.has_em;
    out.em_eps       = g_dev.em_eps;
    out.em_mu        = g_dev.em_mu;
    out.em_sigma     = g_dev.em_sigma;
    out.count        = g_dev.count;
    return true;
}

} // namespace matgpu
