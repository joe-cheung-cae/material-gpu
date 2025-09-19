#include <gtest/gtest.h>
#include <chrono>
#include <cmath>
#include "material/validation.hpp"
#include "material/material.hpp"
#include "material/builder.hpp"

using namespace matgpu;

class MaterialValidationTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(MaterialValidationTest, ElasticProperties_ValidRanges) {
    MaterialValidator validator;
    
    // Valid elastic properties
    EXPECT_TRUE(validator.validate_young_modulus(200e9f));
    EXPECT_TRUE(validator.validate_poisson_ratio(0.3f));
    EXPECT_TRUE(validator.validate_density(7850.0f));
    
    // Test boundary values
    EXPECT_TRUE(validator.validate_young_modulus(1e6f)); // Minimum reasonable value
    EXPECT_TRUE(validator.validate_poisson_ratio(-0.99f)); // Lower bound
    EXPECT_TRUE(validator.validate_poisson_ratio(0.49f)); // Upper bound
    EXPECT_TRUE(validator.validate_density(0.1f)); // Low density materials
}

TEST_F(MaterialValidationTest, ElasticProperties_InvalidRanges) {
    MaterialValidator validator;
    
    // Invalid Young's modulus
    EXPECT_FALSE(validator.validate_young_modulus(0.0f));
    EXPECT_FALSE(validator.validate_young_modulus(-1e9f));
    
    // Invalid Poisson's ratio
    EXPECT_FALSE(validator.validate_poisson_ratio(-1.1f)); // Too low
    EXPECT_FALSE(validator.validate_poisson_ratio(0.6f));  // Too high
    
    // Invalid density
    EXPECT_FALSE(validator.validate_density(0.0f));
    EXPECT_FALSE(validator.validate_density(-100.0f));
}

TEST_F(MaterialValidationTest, ThermalProperties_ValidRanges) {
    MaterialValidator validator;
    
    // Valid thermal properties
    EXPECT_TRUE(validator.validate_thermal_conductivity(50.0f));
    EXPECT_TRUE(validator.validate_heat_capacity(500.0f));
    EXPECT_TRUE(validator.validate_expansion_coefficient(1.2e-5f));
    
    // Test boundary values
    EXPECT_TRUE(validator.validate_thermal_conductivity(0.01f)); // Low conductivity
    EXPECT_TRUE(validator.validate_heat_capacity(10.0f)); // Low heat capacity
    EXPECT_TRUE(validator.validate_expansion_coefficient(-1e-5f)); // Negative expansion
}

TEST_F(MaterialValidationTest, ThermalProperties_InvalidRanges) {
    MaterialValidator validator;
    
    // Invalid thermal conductivity
    EXPECT_FALSE(validator.validate_thermal_conductivity(0.0f));
    EXPECT_FALSE(validator.validate_thermal_conductivity(-10.0f));
    
    // Invalid heat capacity
    EXPECT_FALSE(validator.validate_heat_capacity(0.0f));
    EXPECT_FALSE(validator.validate_heat_capacity(-100.0f));
    
    // Invalid expansion coefficient (extreme values)
    EXPECT_FALSE(validator.validate_expansion_coefficient(1.0f)); // Unrealistically high
    EXPECT_FALSE(validator.validate_expansion_coefficient(-1.0f)); // Unrealistically low
}

TEST_F(MaterialValidationTest, ContactModel_EEPAValidation) {
    MaterialValidator validator;
    
    // Valid EEPA parameters
    EXPECT_TRUE(validator.validate_contact_stiffness(1e6f));
    EXPECT_TRUE(validator.validate_contact_stiffness(5e5f));
    EXPECT_TRUE(validator.validate_damping_coefficient(0.3f));
    EXPECT_TRUE(validator.validate_damping_coefficient(0.15f));
    
    // Invalid EEPA parameters
    EXPECT_FALSE(validator.validate_contact_stiffness(0.0f));
    EXPECT_FALSE(validator.validate_contact_stiffness(-1e5f));
    EXPECT_FALSE(validator.validate_damping_coefficient(-0.1f));
    EXPECT_FALSE(validator.validate_damping_coefficient(2.0f)); // Too high
}

TEST_F(MaterialValidationTest, ContactModel_JKRValidation) {
    MaterialValidator validator;
    
    // Valid JKR parameters
    EXPECT_TRUE(validator.validate_work_of_adhesion(0.05f));
    EXPECT_TRUE(validator.validate_contact_radius(1e-4f));
    
    // Invalid JKR parameters
    EXPECT_FALSE(validator.validate_work_of_adhesion(0.0f));
    EXPECT_FALSE(validator.validate_work_of_adhesion(-0.01f));
    EXPECT_FALSE(validator.validate_contact_radius(0.0f));
    EXPECT_FALSE(validator.validate_contact_radius(-1e-5f));
}

TEST_F(MaterialValidationTest, MaterialBuilder_ValidationIntegration) {
    // Test that MaterialBuilder validates properties during construction
    
    // Valid material should build successfully
    auto valid_material = MaterialBuilder::create_standard()
        .with_elastic_properties(200e9f, 0.3f, 7850.0f)
        .with_thermal_properties(50.0f, 500.0f)
        .build();
    
    EXPECT_NE(valid_material, nullptr);
    
    // Invalid properties should throw exceptions
    auto builder = MaterialBuilder::create_standard();
    
    EXPECT_THROW(
        builder.with_elastic_properties(-1e9f, 0.3f, 7850.0f), // Invalid Young's modulus
        std::invalid_argument
    );
    
    EXPECT_THROW(
        builder.with_elastic_properties(200e9f, 0.6f, 7850.0f), // Invalid Poisson's ratio
        std::invalid_argument
    );
    
    EXPECT_THROW(
        builder.with_thermal_properties(-10.0f, 500.0f), // Invalid thermal conductivity
        std::invalid_argument
    );
}

TEST_F(MaterialValidationTest, ValidationErrorMessages) {
    MaterialValidator validator;
    
    // Test that validation methods provide descriptive error messages
    std::string error_msg;
    
    EXPECT_FALSE(validator.validate_young_modulus(-1e9f, error_msg));
    EXPECT_FALSE(error_msg.empty());
    EXPECT_NE(error_msg.find("Young's modulus"), std::string::npos);
    
    error_msg.clear();
    EXPECT_FALSE(validator.validate_poisson_ratio(0.6f, error_msg));
    EXPECT_FALSE(error_msg.empty());
    EXPECT_NE(error_msg.find("Poisson's ratio"), std::string::npos);
    
    error_msg.clear();
    EXPECT_FALSE(validator.validate_density(-100.0f, error_msg));
    EXPECT_FALSE(error_msg.empty());
    EXPECT_NE(error_msg.find("density"), std::string::npos);
}

TEST_F(MaterialValidationTest, ComprehensiveMaterialValidation) {
    MaterialValidator validator;
    
    // Create materials with different properties and validate them
    auto standard_material = MaterialBuilder::create_standard()
        .with_elastic_properties(200e9f, 0.3f, 7850.0f)
        .build();
    
    auto eepa_material = MaterialBuilder::create_eepa()
        .with_elastic_properties(100e9f, 0.25f, 2700.0f)
        .with_contact_parameters(1e6f, 5e5f, 0.3f, 0.15f)
        .build();
    
    auto jkr_material = MaterialBuilder::create_jkr()
        .with_elastic_properties(5e6f, 0.4f, 1200.0f)
        .with_adhesion_parameters(0.05f, 1e-4f)
        .build();
    
    ASSERT_NE(standard_material, nullptr);
    ASSERT_NE(eepa_material, nullptr);
    ASSERT_NE(jkr_material, nullptr);
    
    // Validate all materials
    std::string validation_error;
    
    EXPECT_TRUE(validator.validate_material(*standard_material, validation_error));
    EXPECT_TRUE(validation_error.empty());
    
    validation_error.clear();
    EXPECT_TRUE(validator.validate_material(*eepa_material, validation_error));
    EXPECT_TRUE(validation_error.empty());
    
    validation_error.clear();
    EXPECT_TRUE(validator.validate_material(*jkr_material, validation_error));
    EXPECT_TRUE(validation_error.empty());
}

TEST_F(MaterialValidationTest, PhysicalConsistencyChecks) {
    MaterialValidator validator;
    
    // Test physical consistency checks
    
    // Check that derived properties are consistent
    auto material = MaterialBuilder::create_standard()
        .with_elastic_properties(200e9f, 0.3f, 7850.0f)
        .build();
    
    ASSERT_NE(material, nullptr);
    
    // Shear modulus should be positive and reasonable
    float shear_modulus = material->shear_modulus();
    EXPECT_GT(shear_modulus, 0.0f);
    EXPECT_LT(shear_modulus, material->young_modulus());
    
    // Bulk modulus should be positive
    float bulk_modulus = material->bulk_modulus();
    EXPECT_GT(bulk_modulus, 0.0f);
    
    // Wave speed should be reasonable
    float wave_speed = material->elastic_wave_speed();
    EXPECT_GT(wave_speed, 100.0f); // m/s
    EXPECT_LT(wave_speed, 20000.0f); // m/s
}

TEST_F(MaterialValidationTest, ContactModelConsistency) {
    MaterialValidator validator;
    
    // Test EEPA contact model consistency
    auto eepa_material = MaterialBuilder::create_eepa()
        .with_elastic_properties(200e9f, 0.3f, 7850.0f)
        .with_contact_parameters(1e6f, 5e5f, 0.3f, 0.15f)
        .build();
    
    ASSERT_NE(eepa_material, nullptr);
    
    const auto& contact_model = eepa_material->contact_model();
    EXPECT_EQ(contact_model.model_name(), "EEPA");
    
    // Check that tangential stiffness is typically less than normal stiffness
    const auto* eepa_model = dynamic_cast<const EEPAContactModel*>(&contact_model);
    ASSERT_NE(eepa_model, nullptr);
    
    EXPECT_LE(eepa_model->tangential_stiffness(), eepa_model->normal_stiffness());
    EXPECT_GT(eepa_model->normal_damping(), 0.0f);
    EXPECT_GT(eepa_model->tangential_damping(), 0.0f);
}

TEST_F(MaterialValidationTest, BatchValidation) {
    MaterialValidator validator;
    
    // Create a collection of materials
    std::vector<std::unique_ptr<BaseMaterial>> materials;
    
    materials.push_back(
        MaterialBuilder::create_standard()
        .with_elastic_properties(200e9f, 0.3f, 7850.0f)
        .build()
    );
    
    materials.push_back(
        MaterialBuilder::create_eepa()
        .with_elastic_properties(100e9f, 0.25f, 2700.0f)
        .with_contact_parameters(1e6f, 5e5f, 0.3f, 0.15f)
        .build()
    );
    
    materials.push_back(
        MaterialBuilder::create_jkr()
        .with_elastic_properties(5e6f, 0.4f, 1200.0f)
        .with_adhesion_parameters(0.05f, 1e-4f)
        .build()
    );
    
    // Validate all materials in batch
    std::vector<std::string> validation_errors;
    bool all_valid = validator.validate_materials(materials, validation_errors);
    
    EXPECT_TRUE(all_valid);
    EXPECT_TRUE(validation_errors.empty());
}

TEST_F(MaterialValidationTest, ValidationPerformance) {
    MaterialValidator validator;
    
    // Test validation performance with many materials
    const size_t num_materials = 1000;
    std::vector<std::unique_ptr<BaseMaterial>> materials;
    
    for (size_t i = 0; i < num_materials; ++i) {
        float young_mod = 100e9f + i * 1e6f;
        float poisson = 0.2f + (i % 30) * 0.01f;
        float density = 7000.0f + i * 10.0f;
        
        materials.push_back(
            MaterialBuilder::create_standard()
            .with_elastic_properties(young_mod, poisson, density)
            .build()
        );
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<std::string> validation_errors;
    bool all_valid = validator.validate_materials(materials, validation_errors);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    EXPECT_TRUE(all_valid);
    EXPECT_LT(duration.count(), 1000); // Should complete within 1 second
}