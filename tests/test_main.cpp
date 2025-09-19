#include <gtest/gtest.h>
#include <iostream>

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    std::cout << "Running Material GPU Library Tests..." << std::endl;
    std::cout << "=====================================" << std::endl;
    
    int result = RUN_ALL_TESTS();
    
    if (result == 0) {
        std::cout << "\n✅ All tests passed successfully!" << std::endl;
    } else {
        std::cout << "\n❌ Some tests failed!" << std::endl;
    }
    
    return result;
}