# 代码清理和迁移说明

## 已删除的文件

### 已删除的头文件
1. **`include/material/builder.hpp`** - 已被 `material_factory.hpp` 中的新 `MaterialBuilder` 替代
2. **`include/material/views.cuh`** - 已被 `device_material_views.cuh` 中的模板化视图替代

### 已删除的示例文件
1. **`examples/example.cu`** - 已被 `enhanced_example.cpp` 替代
2. **`examples/materials.json`** - 已被 `enhanced_materials.json` 替代

## 迁移指南

### 从旧MaterialBuilder迁移到新MaterialBuilder

**旧API:**
```cpp
#include "material/builder.hpp"
auto material = MaterialBuilder()
    .elastic(young, poisson, density)
    .contact(ContactModelType::EEPA)
    .eepa(kn, kt, gamma_n, gamma_t)
    .build();  // 返回 Material 结构体
```

**新API:**
```cpp
#include "material/material_factory.hpp"
auto material = MaterialBuilder()
    .elastic(young, poisson, density)
    .eepa_contact(kn, kt, gamma_n, gamma_t)
    .build();  // 返回 std::unique_ptr<IMaterial>
```

### 从旧MaterialView迁移到新DeviceMaterialView

**旧API:**
```cpp
#include "material/views.cuh"
MaterialView mv = make_material_view(V);
float young = mv.elastic.E(id);
```

**新API:**
```cpp
#include "material/device_material_views.cuh"
auto device_view = make_device_material_view(elastic, contact, thermal);
float young = device_view.young_modulus();
```

### 从Materials迁移到MaterialsV2

**旧API:**
```cpp
#include "material/material.hpp"
Materials mats;
mats.load_from_file("materials.json");
```

**新API:**
```cpp
#include "material/enhanced_json_loader.hpp"
MaterialsV2 mats;
mats.load_from_file("enhanced_materials.json");
// 或者仍然兼容旧的: mats.host() 返回旧格式
```

## 保留的兼容性文件

以下文件被保留以维持向后兼容性，但建议迁移到新API：

1. **`include/material/material.hpp`** - 旧的Materials类，建议使用MaterialsV2
2. **`include/material/types.hpp`** - 原始数据结构，新代码建议使用新接口
3. **`include/material/device_api.cuh`** - 旧的设备API，建议使用新的模板化视图
4. **`src/material.cpp`** - 旧Materials类的实现

## 需要注意的更改

### ContactModel类型更改
- 枚举从 `ContactModel` 重命名为 `ContactModelType`
- 新增了 `ContactModel` 策略类接口

### 编译配置更改
- CMakeLists.txt 已更新以使用新的示例文件
- 添加了 `enhanced_json_loader.cpp` 到编译目标

### JSON格式兼容性
- 新的JSON加载器向后兼容旧格式
- 建议使用新的enhanced_materials.json格式以获得最佳体验

## 未来计划

以下文件可能在未来版本中被废弃：
1. `include/material/grouping.hpp` - 当前只在README中使用，需要重写以支持新架构
2. 旧的Materials类 - 建议所有新代码使用MaterialsV2

## 迁移检查清单

- [ ] 更新所有 `#include "material/builder.hpp"` 为 `#include "material/material_factory.hpp"`
- [ ] 更新所有 `#include "material/views.cuh"` 为 `#include "material/device_material_views.cuh"`
- [ ] 将 `Materials` 类使用替换为 `MaterialsV2`
- [ ] 将 `ContactModel` 枚举使用替换为 `ContactModelType`
- [ ] 更新JSON文件使用新的enhanced格式
- [ ] 测试设备代码使用新的模板化视图

这次清理显著简化了代码库，移除了重复功能，并为将来的维护和扩展奠定了更好的基础。