#include <gtest/gtest.h>
#include "material/json_loader.hpp"
#include <nlohmann/json.hpp>

using namespace matgpu;
using json = nlohmann::json;

class JSONLoaderTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
    
    std::string create_test_json() {
        json root;
        json materials_array = json::array();
        
        // EEPA material
        json eepa_material;
        eepa_material["type"] = "eepa";
        eepa_material["elastic"]["young_modulus"] = 2.1e11;
        eepa_material["elastic"]["poisson_ratio"] = 0.29;
        eepa_material["elastic"]["density"] = 7850;
        eepa_material["contact"]["type"] = "eepa";
        eepa_material["contact"]["kn"] = 1e6;
        eepa_material["contact"]["kt"] = 5e5;
        eepa_material["contact"]["gamma_n"] = 0.3;
        eepa_material["contact"]["gamma_t"] = 0.15;
        eepa_material["thermal"]["conductivity"] = 50.0;
        eepa_material["thermal"]["heat_capacity"] = 500;
        
        materials_array.push_back(eepa_material);
        
        // JKR material
        json jkr_material;
        jkr_material["type"] = "jkr";
        jkr_material["elastic"]["young_modulus"] = 5.0e6;
        jkr_material["elastic"]["poisson_ratio"] = 0.4;
        jkr_material["elastic"]["density"] = 1200;
        jkr_material["contact"]["type"] = "jkr";
        jkr_material["contact"]["work_of_adhesion"] = 0.05;
        jkr_material["contact"]["contact_radius0"] = 1e-4;
        
        materials_array.push_back(jkr_material);
        
        // Standard material
        json standard_material;
        standard_material["type"] = "standard";
        standard_material["elastic"]["young_modulus"] = 3.0e9;
        standard_material["elastic"]["poisson_ratio"] = 0.35;
        standard_material["elastic"]["density"] = 1200;
        standard_material["contact"]["type"] = "none";
        
        materials_array.push_back(standard_material);
        
        root["materials"] = materials_array;
        return root.dump();
    }
};

TEST_F(JSONLoaderTest, LoadValidJSON) {
    std::string json_text = create_test_json();
    Materials materials;
    
    EXPECT_TRUE(materials.load_from_json_text(json_text));
    EXPECT_EQ(materials.count(), 3);
    EXPECT_TRUE(materials.last_error().empty());
}

TEST_F(JSONLoaderTest, LoadMaterialsWithDifferentTypes) {
    std::string json_text = create_test_json();
    Materials materials;
    
    ASSERT_TRUE(materials.load_from_json_text(json_text));
    
    const auto& material_list = materials.materials();
    
    // Check first material (EEPA)
    EXPECT_EQ(material_list[0]->material_type(), "EEPAMaterial");
    EXPECT_FLOAT_EQ(material_list[0]->young_modulus(), 2.1e11f);
    EXPECT_EQ(material_list[0]->contact_model().model_name(), "EEPA");
    EXPECT_NE(material_list[0]->thermal_properties(), nullptr);
    
    // Check second material (JKR)
    EXPECT_EQ(material_list[1]->material_type(), "JKRMaterial");
    EXPECT_FLOAT_EQ(material_list[1]->young_modulus(), 5.0e6f);
    EXPECT_EQ(material_list[1]->contact_model().model_name(), "JKR");
    
    // Check third material (Standard)
    EXPECT_EQ(material_list[2]->material_type(), "StandardMaterial");
    EXPECT_FLOAT_EQ(material_list[2]->young_modulus(), 3.0e9f);
    EXPECT_EQ(material_list[2]->contact_model().model_name(), "None");
}

TEST_F(JSONLoaderTest, InvalidJSONStructure) {
    Materials materials;
    
    // Test invalid JSON syntax
    EXPECT_FALSE(materials.load_from_json_text("{invalid json}"));
    EXPECT_FALSE(materials.last_error().empty());
    
    // Test missing materials array
    EXPECT_FALSE(materials.load_from_json_text("{}"));
    EXPECT_FALSE(materials.last_error().empty());
    
    // Test materials is not an array
    EXPECT_FALSE(materials.load_from_json_text(R"({"materials": "not_an_array"})"));
    EXPECT_FALSE(materials.last_error().empty());
}

TEST_F(JSONLoaderTest, InvalidMaterialProperties) {
    Materials materials;
    
    // Test missing elastic properties
    json root;
    json materials_array = json::array();
    json material;
    material["type"] = "standard";
    material["contact"]["type"] = "none";
    materials_array.push_back(material);
    root["materials"] = materials_array;
    
    EXPECT_FALSE(materials.load_from_json_text(root.dump()));
    EXPECT_FALSE(materials.last_error().empty());
}

TEST_F(JSONLoaderTest, MaterialValidation) {
    Materials materials;
    
    // Test invalid Young's modulus
    json root;
    json materials_array = json::array();
    json material;
    material["type"] = "standard";
    material["elastic"]["young_modulus"] = -1.0; // Invalid
    material["elastic"]["poisson_ratio"] = 0.3;
    material["elastic"]["density"] = 7850;
    material["contact"]["type"] = "none";
    materials_array.push_back(material);
    root["materials"] = materials_array;
    
    EXPECT_FALSE(materials.load_from_json_text(root.dump()));
    EXPECT_FALSE(materials.last_error().empty());
}

TEST_F(JSONLoaderTest, MaterialQuerying) {
    std::string json_text = create_test_json();
    Materials materials;
    
    ASSERT_TRUE(materials.load_from_json_text(json_text));
    
    // Test find by ID
    const auto* material_0 = materials.find_material_by_id(0);
    ASSERT_NE(material_0, nullptr);
    EXPECT_EQ(material_0->material_id(), 0);
    
    // Test find by type
    auto eepa_materials = materials.find_materials_by_type("EEPAMaterial");
    EXPECT_EQ(eepa_materials.size(), 1);
    
    auto jkr_materials = materials.find_materials_by_type("JKRMaterial");
    EXPECT_EQ(jkr_materials.size(), 1);
    
    auto standard_materials = materials.find_materials_by_type("StandardMaterial");
    EXPECT_EQ(standard_materials.size(), 1);
}

TEST_F(JSONLoaderTest, MaterialSerialization) {
    std::string json_text = create_test_json();
    Materials materials;
    
    ASSERT_TRUE(materials.load_from_json_text(json_text));
    
    // Export back to JSON
    std::string exported_json = materials.export_to_json();
    EXPECT_FALSE(exported_json.empty());
    
    // Parse exported JSON to verify structure
    json exported = json::parse(exported_json);
    EXPECT_TRUE(exported.contains("materials"));
    EXPECT_TRUE(exported["materials"].is_array());
    EXPECT_EQ(exported["materials"].size(), 3);
}

TEST_F(JSONLoaderTest, MaterialValidationAll) {
    std::string json_text = create_test_json();
    Materials materials;
    
    ASSERT_TRUE(materials.load_from_json_text(json_text));
    
    // Validate all materials
    std::string validation_error;
    EXPECT_TRUE(materials.validate_all_materials(validation_error));
    EXPECT_TRUE(validation_error.empty());
}

TEST_F(JSONLoaderTest, EmptyMaterialsList) {
    json root;
    root["materials"] = json::array(); // Empty array
    
    Materials materials;
    EXPECT_TRUE(materials.load_from_json_text(root.dump()));
    EXPECT_EQ(materials.count(), 0);
}