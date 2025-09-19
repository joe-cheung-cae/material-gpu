#include <gtest/gtest.h>
#include <cmath>
#include "material/contact_models.hpp"
#include "material/material_factory.hpp"

using namespace matgpu;

class ContactModelTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(ContactModelTest, NoContactModel) {
    NoContactModel no_contact;
    
    EXPECT_EQ(no_contact.model_name(), "None");
    EXPECT_FLOAT_EQ(no_contact.compute_normal_force(1e-3f, 0.1f), 0.0f);
    EXPECT_FLOAT_EQ(no_contact.compute_tangential_force(1e-3f, 0.1f), 0.0f);
}

TEST_F(ContactModelTest, EEPAContactModel) {
    float kn = 1e5f, kt = 5e4f, gamma_n = 0.2f, gamma_t = 0.1f;
    EEPAContactModel eepa(kn, kt, gamma_n, gamma_t);
    
    EXPECT_EQ(eepa.model_name(), "EEPA");
    EXPECT_FLOAT_EQ(eepa.kn(), kn);
    EXPECT_FLOAT_EQ(eepa.kt(), kt);
    EXPECT_FLOAT_EQ(eepa.gamma_n(), gamma_n);
    EXPECT_FLOAT_EQ(eepa.gamma_t(), gamma_t);
    
    // Test force computation
    float overlap = 1e-3f; // 1mm
    float velocity = 0.1f; // 0.1 m/s
    float normal_force = eepa.compute_normal_force(overlap, velocity);
    
    // EEPA normal force should be positive for compression
    EXPECT_GT(normal_force, 0.0f);
    
    // Check that force includes both elastic and damping components
    float elastic_force = kn * overlap;
    EXPECT_GT(normal_force, elastic_force); // Should include damping
}

TEST_F(ContactModelTest, JKRContactModel) {
    float work_of_adhesion = 0.05f;
    float contact_radius0 = 1e-4f;
    JKRContactModel jkr(work_of_adhesion, contact_radius0);
    
    EXPECT_EQ(jkr.model_name(), "JKR");
    EXPECT_FLOAT_EQ(jkr.work_of_adhesion(), work_of_adhesion);
    EXPECT_FLOAT_EQ(jkr.contact_radius0(), contact_radius0);
    
    // Test parameter access via get_parameter
    EXPECT_FLOAT_EQ(jkr.get_parameter("work_of_adhesion"), work_of_adhesion);
    EXPECT_FLOAT_EQ(jkr.get_parameter("contact_radius0"), contact_radius0);
    
    // Test invalid parameter name
    EXPECT_THROW(jkr.get_parameter("invalid_param"), std::invalid_argument);
}

TEST_F(ContactModelTest, ContactModelPolymorphism) {
    std::vector<std::unique_ptr<ContactModel>> models;
    
    models.push_back(std::make_unique<NoContactModel>());
    models.push_back(std::make_unique<EEPAContactModel>(1e5f, 5e4f, 0.2f, 0.1f));
    models.push_back(std::make_unique<JKRContactModel>(0.05f, 1e-4f));
    
    // Test polymorphic behavior
    std::vector<std::string> expected_names = {"None", "EEPA", "JKR"};
    
    for (size_t i = 0; i < models.size(); ++i) {
        EXPECT_EQ(models[i]->model_name(), expected_names[i]);
        
        // All models should respond to force computation (may return 0)
        float force = models[i]->compute_normal_force(1e-3f, 0.1f);
        EXPECT_TRUE(std::isfinite(force)); // Force should be finite
    }
}

TEST_F(ContactModelTest, EEPAForceValidation) {
    EEPAContactModel eepa(1e5f, 5e4f, 0.2f, 0.1f);
    
    // Test that force increases with overlap
    float small_overlap = 1e-4f;
    float large_overlap = 1e-3f;
    float velocity = 0.1f;
    
    float force_small = eepa.compute_normal_force(small_overlap, velocity);
    float force_large = eepa.compute_normal_force(large_overlap, velocity);
    
    EXPECT_GT(force_large, force_small);
    
    // Test that force increases with velocity (damping effect)
    float low_velocity = 0.01f;
    float high_velocity = 1.0f;
    float overlap = 1e-3f;
    
    float force_low_v = eepa.compute_normal_force(overlap, low_velocity);
    float force_high_v = eepa.compute_normal_force(overlap, high_velocity);
    
    EXPECT_GT(force_high_v, force_low_v);
}

TEST_F(ContactModelTest, ContactModelCreation) {
    // Test creation with various parameters
    EXPECT_NO_THROW(EEPAContactModel(1e5f, 5e4f, 0.2f, 0.1f));
    EXPECT_NO_THROW(JKRContactModel(0.05f, 1e-4f));
    EXPECT_NO_THROW(NoContactModel());
    
    // Test with boundary values
    EXPECT_NO_THROW(EEPAContactModel(0.0f, 0.0f, 0.0f, 0.0f)); // Zero parameters
    EXPECT_NO_THROW(JKRContactModel(0.0f, 0.0f)); // Zero adhesion
}

TEST_F(ContactModelTest, TangentialForces) {
    EEPAContactModel eepa(1e5f, 5e4f, 0.2f, 0.1f);
    
    float tangential_displacement = 1e-4f;
    float tangential_velocity = 0.05f;
    
    float tangential_force = eepa.compute_tangential_force(tangential_displacement, tangential_velocity);
    
    // Tangential force should be proportional to displacement and velocity
    EXPECT_GT(std::abs(tangential_force), 0.0f);
    
    // Test scaling with displacement
    float larger_displacement = 2e-4f;
    float larger_force = eepa.compute_tangential_force(larger_displacement, tangential_velocity);
    
    EXPECT_GT(std::abs(larger_force), std::abs(tangential_force));
}