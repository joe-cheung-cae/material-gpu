#include <gtest/gtest.h>
#include "material/json_loader.hpp"
#include <nlohmann/json.hpp>

using namespace matgpu;
using json = nlohmann::json;

TEST(JSONLoaderTest, LoadValidJSON) {
    json root;
    json materials_array = json::array();
    
    // Standard material
    json standard_material;
    standard_material["type"] = "standard";
    standard_material["elastic"]["young_modulus"] = 200e9;
    standard_material["elastic"]["poisson_ratio"] = 0.3;
    standard_material["elastic"]["density"] = 7850;
    standard_material["contact"]["type"] = "none";
    
    materials_array.push_back(standard_material);
    root["materials"] = materials_array;
    
    std::string json_text = root.dump();
    
    Materials materials;
    EXPECT_TRUE(materials.load_from_json_text(json_text));
    EXPECT_EQ(materials.count(), 1);
    EXPECT_TRUE(materials.last_error().empty());
}

TEST(JSONLoaderTest, InvalidJSONStructure) {
    Materials materials;
    
    // Test invalid JSON syntax
    EXPECT_FALSE(materials.load_from_json_text("{invalid json}"));
    EXPECT_FALSE(materials.last_error().empty());
    
    // Test missing materials array
    EXPECT_FALSE(materials.load_from_json_text("{}"));
    EXPECT_FALSE(materials.last_error().empty());
}

TEST(JSONLoaderTest, EmptyMaterialsList) {
    json root;
    root["materials"] = json::array(); // Empty array
    
    Materials materials;
    EXPECT_TRUE(materials.load_from_json_text(root.dump()));
    EXPECT_EQ(materials.count(), 0);
}