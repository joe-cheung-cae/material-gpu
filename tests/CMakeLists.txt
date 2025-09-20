# Tests for Material GPU Library

# Test executable
add_executable(material_tests
    test_main.cpp
    test_contact_models.cpp
    test_contact_models_simple.cpp
    test_factory_simple.cpp
    test_json_loader.cpp
    test_json_simple.cpp
    test_material_builder.cpp
    test_cuda_kernels.cpp
)

# Link with the material library and Google Test
target_link_libraries(material_tests
    PRIVATE
    material
    gtest
    gtest_main
    nlohmann_json::nlohmann_json
)

# Include directories
target_include_directories(material_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party/googletest/googletest/include
)

# Set C++ standard
target_compile_features(material_tests PRIVATE cxx_std_17)

# Add tests to CTest
add_test(NAME ContactModelTests COMMAND material_tests --gtest_filter="ContactModelTest*")
add_test(NAME MaterialFactoryTests COMMAND material_tests --gtest_filter="MaterialFactoryTest*")
add_test(NAME JSONLoaderTests COMMAND material_tests --gtest_filter="JSONLoaderTest*")
add_test(NAME CudaKernelTests COMMAND material_tests --gtest_filter="CudaKernelTest*")
add_test(NAME AllTests COMMAND material_tests)

# Enable CUDA for tests if available
if(ENABLE_CUDA)
    set_property(TARGET material_tests PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    target_compile_definitions(material_tests PRIVATE MATERIAL_GPU_WITH_CUDA=1)

    # Compile CUDA-specific test sources with CUDA language
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/test_cuda_kernels.cpp PROPERTIES LANGUAGE CUDA)

    # Add CUDA runtime library
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(material_tests PRIVATE CUDA::cudart)
endif()