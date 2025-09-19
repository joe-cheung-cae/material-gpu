#include "material/contact_models.hpp"

#include <cmath>
#include <gtest/gtest.h>

using namespace matgpu;

TEST(ContactModelSimpleTest, NoContactModel_BasicUsage) {
    NoContactModel no_contact;

    EXPECT_EQ(no_contact.model_name(), "None");

    // No contact model should return zero force
    float normal_force     = no_contact.compute_normal_force(0.1f, 0.05f);
    float tangential_force = no_contact.compute_tangential_force(0.05f, 0.02f);
    EXPECT_FLOAT_EQ(normal_force, 0.0f);
    EXPECT_FLOAT_EQ(tangential_force, 0.0f);
}

TEST(ContactModelSimpleTest, EEPAContactModel_BasicUsage) {
    float kn      = 1e6f;
    float kt      = 5e5f;
    float gamma_n = 0.3f;
    float gamma_t = 0.15f;

    EEPAContactModel eepa(kn, kt, gamma_n, gamma_t);

    EXPECT_EQ(eepa.model_name(), "EEPA");
    EXPECT_FLOAT_EQ(eepa.kn(), kn);
    EXPECT_FLOAT_EQ(eepa.kt(), kt);
    EXPECT_FLOAT_EQ(eepa.gamma_n(), gamma_n);
    EXPECT_FLOAT_EQ(eepa.gamma_t(), gamma_t);

    // Test force calculation
    float normal_force = eepa.compute_normal_force(0.1f, 0.05f);
    EXPECT_GT(normal_force, 0.0f);
    EXPECT_TRUE(std::isfinite(normal_force)); // Force should be finite

    float tangential_force = eepa.compute_tangential_force(0.05f, 0.02f);
    EXPECT_GT(tangential_force, 0.0f);
    EXPECT_TRUE(std::isfinite(tangential_force));
}