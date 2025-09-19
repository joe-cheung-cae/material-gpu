#include "material/material_factory.hpp"

#include <gtest/gtest.h>

using namespace matgpu;

TEST(MaterialFactorySimpleTest, CreateStandardMaterial) {
    MaterialBuilder builder;

    auto material = builder.elastic(200e9f, 0.3f, 7850.0f).no_contact().build();

    ASSERT_NE(material, nullptr);
    EXPECT_FLOAT_EQ(material->young_modulus(), 200e9f);
    EXPECT_FLOAT_EQ(material->poisson_ratio(), 0.3f);
    EXPECT_FLOAT_EQ(material->density(), 7850.0f);
    EXPECT_EQ(material->material_type(), "StandardMaterial");
}

TEST(MaterialFactorySimpleTest, CreateEEPAMaterial) {
    MaterialBuilder builder;

    auto material =
        builder.elastic(100e9f, 0.25f, 2700.0f).eepa_contact(1e6f, 5e5f, 0.3f, 0.15f).material_type("eepa").build();

    ASSERT_NE(material, nullptr);
    EXPECT_FLOAT_EQ(material->young_modulus(), 100e9f);
    EXPECT_FLOAT_EQ(material->poisson_ratio(), 0.25f);
    EXPECT_FLOAT_EQ(material->density(), 2700.0f);
    EXPECT_EQ(material->material_type(), "EEPAMaterial");
    EXPECT_EQ(material->contact_model().model_name(), "EEPA");
}

TEST(MaterialFactorySimpleTest, CreateJKRMaterial) {
    MaterialBuilder builder;

    auto material = builder.elastic(5e6f, 0.4f, 1200.0f).jkr_contact(0.05f, 1e-4f).material_type("jkr").build();

    ASSERT_NE(material, nullptr);
    EXPECT_FLOAT_EQ(material->young_modulus(), 5e6f);
    EXPECT_FLOAT_EQ(material->poisson_ratio(), 0.4f);
    EXPECT_FLOAT_EQ(material->density(), 1200.0f);
    EXPECT_EQ(material->material_type(), "JKRMaterial");
    EXPECT_EQ(material->contact_model().model_name(), "JKR");
}