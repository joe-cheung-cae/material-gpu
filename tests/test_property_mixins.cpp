#include <gtest/gtest.h>
#include <cmath>
#include "material/property_mixins.hpp"
#include "material/material_factory.hpp"

using namespace matgpu;

class PropertyMixinsTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(PropertyMixinsTest, ElasticProperties_BasicUsage) {
    auto builder = MaterialBuilder::create_standard();
    
    auto material = builder
        .with_elastic_properties(2.1e11f, 0.29f, 7850.0f)
        .build();
    
    ASSERT_NE(material, nullptr);
    EXPECT_FLOAT_EQ(material->young_modulus(), 2.1e11f);
    EXPECT_FLOAT_EQ(material->poisson_ratio(), 0.29f);
    EXPECT_FLOAT_EQ(material->density(), 7850.0f);
    
    // Test derived properties
    EXPECT_GT(material->shear_modulus(), 0.0f);
    EXPECT_GT(material->bulk_modulus(), 0.0f);
    EXPECT_GT(material->elastic_wave_speed(), 0.0f);
}

TEST_F(PropertyMixinsTest, ElasticProperties_DerivedCalculations) {
    auto builder = MaterialBuilder::create_standard();
    
    auto material = builder
        .with_elastic_properties(200e9f, 0.3f, 7800.0f)
        .build();
    
    ASSERT_NE(material, nullptr);
    
    // Test shear modulus calculation: G = E / (2 * (1 + v))
    float expected_shear = 200e9f / (2.0f * (1.0f + 0.3f));
    EXPECT_FLOAT_EQ(material->shear_modulus(), expected_shear);
    
    // Test bulk modulus calculation: K = E / (3 * (1 - 2*v))
    float expected_bulk = 200e9f / (3.0f * (1.0f - 2.0f * 0.3f));
    EXPECT_FLOAT_EQ(material->bulk_modulus(), expected_bulk);
    
    // Test wave speed calculation: c = sqrt(E / ρ)
    float expected_wave_speed = std::sqrt(200e9f / 7800.0f);
    EXPECT_FLOAT_EQ(material->elastic_wave_speed(), expected_wave_speed);
}

TEST_F(PropertyMixinsTest, ThermalProperties_WithValues) {
    auto builder = MaterialBuilder::create_standard();
    
    auto material = builder
        .with_elastic_properties(100e9f, 0.25f, 2000.0f)
        .with_thermal_properties(50.0f, 500.0f, 1.2e-5f)
        .build();
    
    ASSERT_NE(material, nullptr);
    
    const auto* thermal = material->thermal_properties();
    ASSERT_NE(thermal, nullptr);
    
    EXPECT_FLOAT_EQ(thermal->conductivity(), 50.0f);
    EXPECT_FLOAT_EQ(thermal->heat_capacity(), 500.0f);
    EXPECT_FLOAT_EQ(thermal->expansion_coefficient(), 1.2e-5f);
    
    // Test thermal diffusivity calculation
    float expected_diffusivity = 50.0f / (2000.0f * 500.0f);
    EXPECT_FLOAT_EQ(thermal->thermal_diffusivity(), expected_diffusivity);
}

TEST_F(PropertyMixinsTest, ThermalProperties_Optional) {
    auto builder = MaterialBuilder::create_standard();
    
    auto material = builder
        .with_elastic_properties(100e9f, 0.25f, 2000.0f)
        .build();
    
    ASSERT_NE(material, nullptr);
    
    // Material without thermal properties should return nullptr
    EXPECT_EQ(material->thermal_properties(), nullptr);
    EXPECT_FALSE(material->has_thermal_properties());
}

TEST_F(PropertyMixinsTest, ContactModel_EEPA) {
    auto builder = MaterialBuilder::create_eepa();
    
    auto material = builder
        .with_elastic_properties(210e9f, 0.29f, 7850.0f)
        .with_contact_parameters(1e6f, 5e5f, 0.3f, 0.15f)
        .build();
    
    ASSERT_NE(material, nullptr);
    
    const auto& contact_model = material->contact_model();
    EXPECT_EQ(contact_model.model_name(), "EEPA");
    
    // Test EEPA-specific parameters
    const auto* eepa_model = dynamic_cast<const EEPAContactModel*>(&contact_model);
    ASSERT_NE(eepa_model, nullptr);
    
    EXPECT_FLOAT_EQ(eepa_model->normal_stiffness(), 1e6f);
    EXPECT_FLOAT_EQ(eepa_model->tangential_stiffness(), 5e5f);
    EXPECT_FLOAT_EQ(eepa_model->normal_damping(), 0.3f);
    EXPECT_FLOAT_EQ(eepa_model->tangential_damping(), 0.15f);
}

TEST_F(PropertyMixinsTest, ContactModel_JKR) {
    auto builder = MaterialBuilder::create_jkr();
    
    auto material = builder
        .with_elastic_properties(5e6f, 0.4f, 1200.0f)
        .with_adhesion_parameters(0.05f, 1e-4f)
        .build();
    
    ASSERT_NE(material, nullptr);
    
    const auto& contact_model = material->contact_model();
    EXPECT_EQ(contact_model.model_name(), "JKR");
    
    // Test JKR-specific parameters
    const auto* jkr_model = dynamic_cast<const JKRContactModel*>(&contact_model);
    ASSERT_NE(jkr_model, nullptr);
    
    EXPECT_FLOAT_EQ(jkr_model->work_of_adhesion(), 0.05f);
    EXPECT_FLOAT_EQ(jkr_model->contact_radius0(), 1e-4f);
}

TEST_F(PropertyMixinsTest, PropertyValidation_ElasticBounds) {
    auto builder = MaterialBuilder::create_standard();
    
    // Test invalid Young's modulus
    EXPECT_THROW(
        builder.with_elastic_properties(-1.0f, 0.3f, 7850.0f),
        std::invalid_argument
    );
    
    // Test invalid Poisson's ratio (too low)
    EXPECT_THROW(
        builder.with_elastic_properties(200e9f, -1.5f, 7850.0f),
        std::invalid_argument
    );
    
    // Test invalid Poisson's ratio (too high)
    EXPECT_THROW(
        builder.with_elastic_properties(200e9f, 0.6f, 7850.0f),
        std::invalid_argument
    );
    
    // Test invalid density
    EXPECT_THROW(
        builder.with_elastic_properties(200e9f, 0.3f, -100.0f),
        std::invalid_argument
    );
}

TEST_F(PropertyMixinsTest, PropertyValidation_ThermalBounds) {
    auto builder = MaterialBuilder::create_standard()
        .with_elastic_properties(200e9f, 0.3f, 7850.0f);
    
    // Test invalid thermal conductivity
    EXPECT_THROW(
        builder.with_thermal_properties(-10.0f, 500.0f, 1e-5f),
        std::invalid_argument
    );
    
    // Test invalid heat capacity
    EXPECT_THROW(
        builder.with_thermal_properties(50.0f, -100.0f, 1e-5f),
        std::invalid_argument
    );
}

TEST_F(PropertyMixinsTest, PropertyMixins_Chaining) {
    // Test that property mixins can be chained in different orders
    auto material1 = MaterialBuilder::create_eepa()
        .with_elastic_properties(200e9f, 0.3f, 7850.0f)
        .with_thermal_properties(50.0f, 500.0f)
        .with_contact_parameters(1e6f, 5e5f)
        .build();
    
    auto material2 = MaterialBuilder::create_eepa()
        .with_thermal_properties(50.0f, 500.0f)
        .with_contact_parameters(1e6f, 5e5f)
        .with_elastic_properties(200e9f, 0.3f, 7850.0f)
        .build();
    
    auto material3 = MaterialBuilder::create_eepa()
        .with_contact_parameters(1e6f, 5e5f)
        .with_elastic_properties(200e9f, 0.3f, 7850.0f)
        .with_thermal_properties(50.0f, 500.0f)
        .build();
    
    ASSERT_NE(material1, nullptr);
    ASSERT_NE(material2, nullptr);
    ASSERT_NE(material3, nullptr);
    
    // All materials should have the same properties regardless of chain order
    EXPECT_FLOAT_EQ(material1->young_modulus(), material2->young_modulus());
    EXPECT_FLOAT_EQ(material2->young_modulus(), material3->young_modulus());
    
    EXPECT_NE(material1->thermal_properties(), nullptr);
    EXPECT_NE(material2->thermal_properties(), nullptr);
    EXPECT_NE(material3->thermal_properties(), nullptr);
    
    EXPECT_EQ(material1->contact_model().model_name(), "EEPA");
    EXPECT_EQ(material2->contact_model().model_name(), "EEPA");
    EXPECT_EQ(material3->contact_model().model_name(), "EEPA");
}

TEST_F(PropertyMixinsTest, MaterialComparison) {
    auto material1 = MaterialBuilder::create_standard()
        .with_elastic_properties(200e9f, 0.3f, 7850.0f)
        .with_thermal_properties(50.0f, 500.0f)
        .build();
    
    auto material2 = MaterialBuilder::create_standard()
        .with_elastic_properties(200e9f, 0.3f, 7850.0f)
        .with_thermal_properties(50.0f, 500.0f)
        .build();
    
    auto material3 = MaterialBuilder::create_standard()
        .with_elastic_properties(100e9f, 0.25f, 7850.0f) // Different properties
        .build();
    
    ASSERT_NE(material1, nullptr);
    ASSERT_NE(material2, nullptr);
    ASSERT_NE(material3, nullptr);
    
    // Test material equality (should be equal based on properties)
    EXPECT_TRUE(material1->has_same_properties(*material2));
    EXPECT_FALSE(material1->has_same_properties(*material3));
}

TEST_F(PropertyMixinsTest, PropertySerialization) {
    auto material = MaterialBuilder::create_eepa()
        .with_elastic_properties(200e9f, 0.3f, 7850.0f)
        .with_thermal_properties(50.0f, 500.0f, 1.2e-5f)
        .with_contact_parameters(1e6f, 5e5f, 0.3f, 0.15f)
        .build();
    
    ASSERT_NE(material, nullptr);
    
    // Test property serialization to JSON-like structure
    auto properties_map = material->serialize_properties();
    
    EXPECT_GT(properties_map.size(), 0);
    EXPECT_NE(properties_map.find("young_modulus"), properties_map.end());
    EXPECT_NE(properties_map.find("poisson_ratio"), properties_map.end());
    EXPECT_NE(properties_map.find("density"), properties_map.end());
    EXPECT_NE(properties_map.find("thermal_conductivity"), properties_map.end());
    EXPECT_NE(properties_map.find("contact_model"), properties_map.end());
}