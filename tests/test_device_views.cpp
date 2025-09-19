#include <gtest/gtest.h>
#include "material/views.cuh"
#include "material/device_runtime.hpp"
#include <vector>

using namespace matgpu;

#ifdef __CUDACC__

class DeviceViewsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize CUDA context if needed
        if (!device_runtime.is_initialized()) {
            device_runtime.initialize();
        }
    }
    
    void TearDown() override {}
    
    DeviceRuntime device_runtime;
};

TEST_F(DeviceViewsTest, MaterialView_BasicProperties) {
    // Create test material data
    MaterialData host_data;
    host_data.material_id = 42;
    host_data.young_modulus = 2.1e11f;
    host_data.poisson_ratio = 0.29f;
    host_data.density = 7850.0f;
    host_data.contact_model_type = ContactModelType::EEPA;
    
    // Copy to device
    MaterialData* device_data = device_runtime.allocate_and_copy(host_data);
    
    // Create device view
    MaterialView view(device_data);
    
    // Test properties access (these would be tested in device code)
    EXPECT_EQ(view.material_id(), 42);
    EXPECT_FLOAT_EQ(view.young_modulus(), 2.1e11f);
    EXPECT_FLOAT_EQ(view.poisson_ratio(), 0.29f);
    EXPECT_FLOAT_EQ(view.density(), 7850.0f);
    EXPECT_EQ(view.contact_model_type(), ContactModelType::EEPA);
    
    device_runtime.deallocate(device_data);
}

TEST_F(DeviceViewsTest, ContactModelView_ForceCalculation) {
    // Test EEPA contact model
    EEPAContactData host_eepa;
    host_eepa.kn = 1e6f;
    host_eepa.kt = 5e5f;
    host_eepa.gamma_n = 0.3f;
    host_eepa.gamma_t = 0.15f;
    
    EEPAContactData* device_eepa = device_runtime.allocate_and_copy(host_eepa);
    EEPAContactView eepa_view(device_eepa);
    
    // Test force calculation parameters
    EXPECT_FLOAT_EQ(eepa_view.normal_stiffness(), 1e6f);
    EXPECT_FLOAT_EQ(eepa_view.tangential_stiffness(), 5e5f);
    EXPECT_FLOAT_EQ(eepa_view.normal_damping(), 0.3f);
    EXPECT_FLOAT_EQ(eepa_view.tangential_damping(), 0.15f);
    
    device_runtime.deallocate(device_eepa);
}

TEST_F(DeviceViewsTest, ThermalPropertiesView) {
    ThermalData host_thermal;
    host_thermal.conductivity = 50.0f;
    host_thermal.heat_capacity = 500.0f;
    host_thermal.expansion_coefficient = 1.2e-5f;
    
    ThermalData* device_thermal = device_runtime.allocate_and_copy(host_thermal);
    ThermalView thermal_view(device_thermal);
    
    EXPECT_FLOAT_EQ(thermal_view.conductivity(), 50.0f);
    EXPECT_FLOAT_EQ(thermal_view.heat_capacity(), 500.0f);
    EXPECT_FLOAT_EQ(thermal_view.expansion_coefficient(), 1.2e-5f);
    
    device_runtime.deallocate(device_thermal);
}

TEST_F(DeviceViewsTest, MaterialArrayView) {
    const size_t num_materials = 5;
    std::vector<MaterialData> host_materials(num_materials);
    
    // Initialize test materials
    for (size_t i = 0; i < num_materials; ++i) {
        host_materials[i].material_id = static_cast<int>(i);
        host_materials[i].young_modulus = 1e9f * (i + 1);
        host_materials[i].poisson_ratio = 0.2f + i * 0.05f;
        host_materials[i].density = 1000.0f * (i + 1);
        host_materials[i].contact_model_type = ContactModelType::STANDARD;
    }
    
    // Copy array to device
    MaterialData* device_materials = device_runtime.allocate_and_copy_array(
        host_materials.data(), num_materials);
    
    // Create array view
    MaterialArrayView array_view(device_materials, num_materials);
    
    EXPECT_EQ(array_view.size(), num_materials);
    EXPECT_NE(array_view.data(), nullptr);
    
    // Test indexing (would be used in device code)
    for (size_t i = 0; i < num_materials; ++i) {
        MaterialView material_view = array_view[i];
        EXPECT_EQ(material_view.material_id(), static_cast<int>(i));
        EXPECT_FLOAT_EQ(material_view.young_modulus(), 1e9f * (i + 1));
    }
    
    device_runtime.deallocate_array(device_materials);
}

TEST_F(DeviceViewsTest, ContactModelPolymorphism) {
    // Test different contact model views
    
    // EEPA model
    EEPAContactData eepa_data;
    eepa_data.kn = 1e6f;
    eepa_data.kt = 5e5f;
    
    EEPAContactData* device_eepa = device_runtime.allocate_and_copy(eepa_data);
    EEPAContactView eepa_view(device_eepa);
    
    EXPECT_EQ(eepa_view.model_type(), ContactModelType::EEPA);
    EXPECT_FLOAT_EQ(eepa_view.normal_stiffness(), 1e6f);
    
    // JKR model
    JKRContactData jkr_data;
    jkr_data.work_of_adhesion = 0.05f;
    jkr_data.contact_radius0 = 1e-4f;
    
    JKRContactData* device_jkr = device_runtime.allocate_and_copy(jkr_data);
    JKRContactView jkr_view(device_jkr);
    
    EXPECT_EQ(jkr_view.model_type(), ContactModelType::JKR);
    EXPECT_FLOAT_EQ(jkr_view.work_of_adhesion(), 0.05f);
    EXPECT_FLOAT_EQ(jkr_view.contact_radius0(), 1e-4f);
    
    device_runtime.deallocate(device_eepa);
    device_runtime.deallocate(device_jkr);
}

__global__ void test_device_material_access(MaterialView* materials, 
                                           size_t count, 
                                           float* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        MaterialView& material = materials[idx];
        // Test device-side property access
        results[idx] = material.young_modulus() * material.density();
    }
}

TEST_F(DeviceViewsTest, DeviceKernelAccess) {
    const size_t num_materials = 4;
    std::vector<MaterialData> host_materials(num_materials);
    
    // Initialize materials
    for (size_t i = 0; i < num_materials; ++i) {
        host_materials[i].material_id = static_cast<int>(i);
        host_materials[i].young_modulus = 1e9f * (i + 1);
        host_materials[i].density = 1000.0f * (i + 1);
    }
    
    // Allocate device memory
    MaterialData* device_materials = device_runtime.allocate_and_copy_array(
        host_materials.data(), num_materials);
    
    MaterialView* device_views = device_runtime.allocate<MaterialView>(num_materials);
    float* device_results = device_runtime.allocate<float>(num_materials);
    
    // Initialize device views
    for (size_t i = 0; i < num_materials; ++i) {
        MaterialView view(&device_materials[i]);
        device_runtime.copy_to_device(&device_views[i], &view, 1);
    }
    
    // Launch kernel
    dim3 block(256);
    dim3 grid((num_materials + block.x - 1) / block.x);
    test_device_material_access<<<grid, block>>>(device_views, num_materials, device_results);
    
    // Check for kernel errors
    cudaError_t error = cudaGetLastError();
    EXPECT_EQ(error, cudaSuccess);
    
    // Copy results back
    std::vector<float> host_results(num_materials);
    device_runtime.copy_to_host(host_results.data(), device_results, num_materials);
    
    // Verify results
    for (size_t i = 0; i < num_materials; ++i) {
        float expected = (1e9f * (i + 1)) * (1000.0f * (i + 1));
        EXPECT_FLOAT_EQ(host_results[i], expected);
    }
    
    // Cleanup
    device_runtime.deallocate_array(device_materials);
    device_runtime.deallocate_array(device_views);
    device_runtime.deallocate_array(device_results);
}

TEST_F(DeviceViewsTest, ViewValidation) {
    // Test null pointer handling
    MaterialView null_view(nullptr);
    EXPECT_FALSE(null_view.is_valid());
    
    // Test valid view
    MaterialData host_data;
    host_data.material_id = 1;
    MaterialData* device_data = device_runtime.allocate_and_copy(host_data);
    
    MaterialView valid_view(device_data);
    EXPECT_TRUE(valid_view.is_valid());
    
    device_runtime.deallocate(device_data);
}

#else

// CPU-only tests when CUDA is not available
TEST(DeviceViewsTest, CUDA_NotAvailable) {
    GTEST_SKIP() << "CUDA not available, skipping device view tests";
}

#endif // __CUDACC__