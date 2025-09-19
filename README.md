# material_gpu

A minimal CUDA-ready material property module for DEM solvers.

## Features
- Base elastic properties: Young's modulus, Poisson's ratio, density
- Contact models: EEPA and JKR
- Optional Thermal and EM properties (can be omitted per material)
- Load material list from JSON text/file
- Upload to device and expose device-side getters for kernels
- New: builder, validation, grouping helpers

## JSON schema (per material)
- elastic: { young, poisson, density } (aliases supported: E, nu, rho)
- contact_model: "EEPA" | "JKR" | "None" (case-insensitive; "HERTZ_MINDLIN" -> EEPA)
- eepa: { kn, kt, gamma_n, gamma_t } (when contact_model==EEPA)
- jkr: { work_of_adhesion, contact_radius0 } (when contact_model==JKR)
- thermal: { conductivity, heat_capacity } (optional)
- em: { permittivity, permeability, conductivity } (optional)

## Build
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON
cmake --build build -j
```

Run example
```
./build/example
```

If CUDA is not available, disable it:
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=OFF
cmake --build build -j
./build/example
```

## Host API quickstart
```c++
#include <material/enhanced_json_loader.hpp>
using namespace matgpu;

MaterialsV2 mats;
if(!mats.load_from_file("enhanced_materials.json")) {
	std::cerr << mats.last_error();
}

// New enhanced material creation
auto material = MaterialBuilder()
    .elastic(2.1e11f, 0.29f, 7850.0f)
    .eepa_contact(1e5f, 5e4f, 0.2f, 0.1f)
    .thermal(50.0f, 500.0f)
    .build();

// Device operations (maintains compatibility)
DeviceMaterialsView V{};
mats.upload_and_get_view(V); // or: upload_to_device(); mats.get_device_view(V);
```

### Builder & Validation
```c++
#include <material/builder.hpp>
#include <material/validation.hpp>
Material m = MaterialBuilder{}
	.elastic(1e7f, 0.25f, 2500.f)
	.contact(ContactModel::EEPA)
	.eepa(1e5f, 5e4f, 0.2f, 0.1f)
	.thermal(10.f, 900.f)
	.build();
std::string err = validate(m);
```

### Grouping
```c++
#include <material/grouping.hpp>
GroupingResult g = group_by_contact_model(mats.host());
```

## Device API
On the device, pass `DeviceMaterialsView` into kernels and use getters or the convenience views in `views.cuh`.
```c++
float E = mat_young(V, id);
auto mv = make_material_view(V);
if(is_eepa(V, id)) {
	float kn = mv.eepa.get().KN(id);
}
```

## License

MIT

## Code style (clang-format)

This repo ships a `.clang-format` at root. You can auto-format sources via:

```
# format all sources in-place
./scripts/format.sh

# check formatting (non-zero exit on issues)
./scripts/check-format.sh

# or via CMake targets
cmake --build build --target format
cmake --build build --target check-format
```
