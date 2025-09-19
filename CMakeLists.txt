cmake_minimum_required(VERSION 3.20)
project(material_gpu LANGUAGES C CXX)

# Option to enable CUDA
option(ENABLE_CUDA "Build with CUDA support" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Use nlohmann_json from git submodule
add_subdirectory(third_party/nlohmann_json)

if(ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  # Expand includes instead of using response files so clangd sees headers in compile_commands.json
  set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
endif()

add_library(material STATIC)

target_sources(material PRIVATE src/json_loader.cpp)

# Public headers
target_include_directories(material PUBLIC include)
target_link_libraries(material PUBLIC nlohmann_json::nlohmann_json)

if(ENABLE_CUDA)
  target_compile_definitions(material PUBLIC MATERIAL_GPU_WITH_CUDA=1)
  set_target_properties(material PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  # For device side linking of global symbols
  set_target_properties(material PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

add_executable(example examples/example.cpp)

if(ENABLE_CUDA)
  set_source_files_properties(examples/example.cpp PROPERTIES LANGUAGE CXX)
  set_target_properties(example PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  set_target_properties(example PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
else()
  # allow host-only example
  set_source_files_properties(examples/example.cpp PROPERTIES LANGUAGE CXX)
endif()

target_link_libraries(example PRIVATE material)

# Testing support
option(BUILD_TESTING "Build the testing tree" ON)
if(BUILD_TESTING)
  # Add Google Test
  add_subdirectory(third_party/googletest)
  
  # Enable CTest
  enable_testing()
  
  # Add test subdirectory
  add_subdirectory(tests)
endif()

# Install rules (optional)
install(TARGETS material DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)

# Formatting helpers
find_program(CLANG_FORMAT_EXE NAMES clang-format clang-format-18)

if(CLANG_FORMAT_EXE)
  add_custom_target(format
    COMMAND ${CMAKE_COMMAND} -E env CLANG_FORMAT=${CLANG_FORMAT_EXE} bash ${CMAKE_SOURCE_DIR}/scripts/format.sh
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running clang-format on source files")
  add_custom_target(check-format
    COMMAND ${CMAKE_COMMAND} -E env CLANG_FORMAT=${CLANG_FORMAT_EXE} bash ${CMAKE_SOURCE_DIR}/scripts/check-format.sh
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Checking formatting with clang-format")
else()
  message(STATUS "clang-format not found; 'format' and 'check-format' targets will require it in PATH")
endif()
