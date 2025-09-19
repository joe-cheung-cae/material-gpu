#include <gtest/gtest.h>
#include "material/material_factory.hpp"
#include "material/contact_models.hpp"

using namespace matgpu;

class MaterialFactoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code for each test
    }

    void TearDown() override {
        // Cleanup code for each test
    }
};

TEST_F(MaterialFactoryTest, CreateStandardMaterial) {
    auto material = MaterialFactory::create_standard(2.1e11f, 0.29f, 7850.0f);
    
    ASSERT_NE(material, nullptr);
    EXPECT_EQ(material->material_type(), "StandardMaterial");
    EXPECT_FLOAT_EQ(material->young_modulus(), 2.1e11f);
    EXPECT_FLOAT_EQ(material->poisson_ratio(), 0.29f);
    EXPECT_FLOAT_EQ(material->density(), 7850.0f);
    EXPECT_EQ(material->contact_model().model_name(), "None");
}

TEST_F(MaterialFactoryTest, CreateEEPAMaterial) {
    float young = 1.0e7f, poisson = 0.25f, density = 2500.0f;
    float kn = 1e5f, kt = 5e4f, gamma_n = 0.2f, gamma_t = 0.1f;
    
    auto material = MaterialFactory::create_eepa(young, poisson, density, kn, kt, gamma_n, gamma_t);
    
    ASSERT_NE(material, nullptr);
    EXPECT_EQ(material->material_type(), "EEPAMaterial");
    EXPECT_FLOAT_EQ(material->young_modulus(), young);
    EXPECT_FLOAT_EQ(material->poisson_ratio(), poisson);
    EXPECT_FLOAT_EQ(material->density(), density);
    EXPECT_EQ(material->contact_model().model_name(), "EEPA");
    
    // Test EEPA specific properties
    auto eepa_material = dynamic_cast<EEPAMaterial*>(material.get());
    ASSERT_NE(eepa_material, nullptr);
    EXPECT_FLOAT_EQ(eepa_material->eepa_contact_model()->kn(), kn);
    EXPECT_FLOAT_EQ(eepa_material->eepa_contact_model()->kt(), kt);
    EXPECT_FLOAT_EQ(eepa_material->eepa_contact_model()->gamma_n(), gamma_n);
    EXPECT_FLOAT_EQ(eepa_material->eepa_contact_model()->gamma_t(), gamma_t);
}

TEST_F(MaterialFactoryTest, CreateJKRMaterial) {
    float young = 5.0e6f, poisson = 0.4f, density = 1200.0f;
    float work_of_adhesion = 0.05f, contact_radius0 = 1e-4f;
    
    auto material = MaterialFactory::create_jkr(young, poisson, density, work_of_adhesion, contact_radius0);
    
    ASSERT_NE(material, nullptr);
    EXPECT_EQ(material->material_type(), "JKRMaterial");
    EXPECT_FLOAT_EQ(material->young_modulus(), young);
    EXPECT_FLOAT_EQ(material->poisson_ratio(), poisson);
    EXPECT_FLOAT_EQ(material->density(), density);
    EXPECT_EQ(material->contact_model().model_name(), "JKR");
    
    // Test JKR specific properties
    auto jkr_material = dynamic_cast<JKRMaterial*>(material.get());
    ASSERT_NE(jkr_material, nullptr);
    EXPECT_FLOAT_EQ(jkr_material->jkr_contact_model()->work_of_adhesion(), work_of_adhesion);
    EXPECT_FLOAT_EQ(jkr_material->jkr_contact_model()->contact_radius0(), contact_radius0);
}

TEST_F(MaterialFactoryTest, MaterialValidation) {
    // Test invalid Young's modulus
    EXPECT_THROW(MaterialFactory::create_standard(-1.0f, 0.3f, 7850.0f), std::invalid_argument);
    
    // Test invalid Poisson's ratio
    EXPECT_THROW(MaterialFactory::create_standard(2.1e11f, 0.6f, 7850.0f), std::invalid_argument);
    
    // Test invalid density
    EXPECT_THROW(MaterialFactory::create_standard(2.1e11f, 0.3f, -1000.0f), std::invalid_argument);
}

TEST_F(MaterialFactoryTest, MaterialProperties) {
    auto material = MaterialFactory::create_eepa(2.1e11f, 0.29f, 7850.0f, 1e6f, 5e5f, 0.3f, 0.15f);
    
    // Test basic properties
    EXPECT_GT(material->young_modulus(), 0.0f);
    EXPECT_GE(material->poisson_ratio(), -1.0f);
    EXPECT_LE(material->poisson_ratio(), 0.5f);
    EXPECT_GT(material->density(), 0.0f);
    
    // Test material ID assignment
    EXPECT_EQ(material->material_id(), 0); // Default ID
    material->set_material_id(42);
    EXPECT_EQ(material->material_id(), 42);
}

TEST_F(MaterialFactoryTest, ContactForceCalculation) {
    auto eepa_material = MaterialFactory::create_eepa(1e7f, 0.25f, 2500.0f, 1e5f, 5e4f, 0.2f, 0.1f);
    auto jkr_material = MaterialFactory::create_jkr(5e6f, 0.4f, 1200.0f, 0.05f, 1e-4f);
    
    float overlap = 1e-3f; // 1mm overlap
    float velocity = 0.1f; // 0.1 m/s relative velocity
    
    // EEPA contact should produce positive force
    float eepa_force = eepa_material->contact_model().compute_normal_force(overlap, velocity);
    EXPECT_GT(eepa_force, 0.0f);
    
    // JKR contact should also produce force (including adhesion)
    float jkr_force = jkr_material->contact_model().compute_normal_force(overlap, velocity);
    EXPECT_NE(jkr_force, 0.0f); // Could be positive or negative due to adhesion
}