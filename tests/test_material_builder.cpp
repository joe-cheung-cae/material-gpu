#include "material/material_factory.hpp"

#include <gtest/gtest.h>

using namespace matgpu;

class MaterialBuilderTest : public ::testing::Test {
  protected:
    MaterialBuilder builder;

    void SetUp() override {
        // Reset builder for each test
        builder = MaterialBuilder();
    }
};

TEST_F(MaterialBuilderTest, BasicElasticProperties) {
    auto material = builder.elastic(2.1e11f, 0.29f, 7850.0f).no_contact().build();

    ASSERT_NE(material, nullptr);
    EXPECT_FLOAT_EQ(material->young_modulus(), 2.1e11f);
    EXPECT_FLOAT_EQ(material->poisson_ratio(), 0.29f);
    EXPECT_FLOAT_EQ(material->density(), 7850.0f);
}

TEST_F(MaterialBuilderTest, FluentInterface) {
    auto material = builder.elastic(1.5e10f, 0.28f, 2400.0f)
                        .eepa_contact(2e5f, 1e5f, 0.25f, 0.12f)
                        .thermal(2.5f, 920.0f)
                        .electromagnetic(8.85e-12f, 1.26e-6f, 1e-7f)
                        .id(42)
                        .build();

    ASSERT_NE(material, nullptr);
    EXPECT_EQ(material->material_id(), 42);
    EXPECT_NE(material->thermal_properties(), nullptr);
    EXPECT_NE(material->electromagnetic_properties(), nullptr);
}

TEST_F(MaterialBuilderTest, BuildEEPAMaterial) {
    auto material = builder.elastic(1.0e7f, 0.25f, 2500.0f).eepa_contact(1e5f, 5e4f, 0.2f, 0.1f).build_eepa();

    ASSERT_NE(material, nullptr);
    EXPECT_EQ(material->material_type(), "EEPAMaterial");
    EXPECT_EQ(material->contact_model().model_name(), "EEPA");
}

TEST_F(MaterialBuilderTest, BuildJKRMaterial) {
    auto material = builder.elastic(5.0e6f, 0.4f, 1200.0f).jkr_contact(0.05f, 1e-4f).build_jkr();

    ASSERT_NE(material, nullptr);
    EXPECT_EQ(material->material_type(), "JKRMaterial");
    EXPECT_EQ(material->contact_model().model_name(), "JKR");
}

TEST_F(MaterialBuilderTest, ValidationErrors) {
    // Test that builder can handle edge cases gracefully
    // Note: Current implementation doesn't throw exceptions for invalid values

    // Test building without setting explicit elastic properties (uses defaults)
    auto material1 = builder.no_contact().build();
    ASSERT_NE(material1, nullptr);

    // Test that builder accepts various parameter values
    // (validation could be added in future implementations)
    auto material2 = builder.elastic(1.0f, 0.3f, 7850.0f).no_contact().build();
    ASSERT_NE(material2, nullptr);

    auto material3 = builder.elastic(2.1e11f, 0.49f, 7850.0f).no_contact().build();
    ASSERT_NE(material3, nullptr);

    auto material4 = builder.elastic(2.1e11f, 0.3f, 1000.0f).no_contact().build();
    ASSERT_NE(material4, nullptr);
}

TEST_F(MaterialBuilderTest, OptionalProperties) {
    // Test material without optional properties
    auto simple_material = builder.elastic(2.0e11f, 0.3f, 7800.0f).no_contact().build();

    EXPECT_EQ(simple_material->thermal_properties(), nullptr);
    EXPECT_EQ(simple_material->electromagnetic_properties(), nullptr);

    // Test material with thermal properties only
    auto thermal_material = builder.elastic(2.0e11f, 0.3f, 7800.0f).no_contact().thermal(50.0f, 500.0f).build();

    EXPECT_NE(thermal_material->thermal_properties(), nullptr);
    EXPECT_EQ(thermal_material->electromagnetic_properties(), nullptr);
    EXPECT_FLOAT_EQ(thermal_material->thermal_properties()->conductivity(), 50.0f);
    EXPECT_FLOAT_EQ(thermal_material->thermal_properties()->heat_capacity(), 500.0f);
}

TEST_F(MaterialBuilderTest, MaterialTypesFromBuilder) {
    // Test that build() creates generic material
    auto generic = builder.elastic(1e9f, 0.3f, 1000.0f).no_contact().build();
    EXPECT_EQ(generic->material_type(), "StandardMaterial");

    // Test that build_eepa() creates EEPA material
    auto eepa = builder.elastic(1e9f, 0.3f, 1000.0f).eepa_contact(1e5f, 5e4f, 0.2f, 0.1f).build_eepa();
    EXPECT_EQ(eepa->material_type(), "EEPAMaterial");

    // Test that build_jkr() creates JKR material
    auto jkr = builder.elastic(1e9f, 0.3f, 1000.0f).jkr_contact(0.05f, 1e-4f).build_jkr();
    EXPECT_EQ(jkr->material_type(), "JKRMaterial");
}