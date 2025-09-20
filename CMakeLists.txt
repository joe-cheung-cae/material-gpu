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

  # CUDA architecture detection and configuration
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    include(CheckLanguage)
    check_language(CUDA)

    # Auto-detect GPU architectures if available
    execute_process(
      COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits
      OUTPUT_VARIABLE GPU_COMPUTE_CAPS
      RESULT_VARIABLE GPU_QUERY_RESULT
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(GPU_QUERY_RESULT EQUAL 0 AND GPU_COMPUTE_CAPS)
      string(REPLACE "\n" ";" GPU_COMPUTE_CAPS_LIST ${GPU_COMPUTE_CAPS})
      list(REMOVE_DUPLICATES GPU_COMPUTE_CAPS_LIST)

      # Convert to CMake format (remove decimal points)
      set(CUDA_ARCHITECTURES "")

      foreach(CAP ${GPU_COMPUTE_CAPS_LIST})
        string(REPLACE "." "" CAP_NO_DOT ${CAP})
        list(APPEND CUDA_ARCHITECTURES ${CAP_NO_DOT})
      endforeach()

      set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})
      message(STATUS "Detected CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    else()
      # Fallback to common architectures
      set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86;89;90")
      message(STATUS "Using default CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    endif()
  endif()

  # CUDA compiler flags for optimization
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

  # Allow opting into fast math explicitly to maintain numerical consistency by default
  option(ENABLE_CUDA_FAST_MATH "Enable CUDA --use_fast_math in Release builds" OFF)

  if(ENABLE_CUDA_FAST_MATH)
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 --use_fast_math")
  else()
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")
  endif()

  set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G -O0")

  # Enable verbose compilation for debugging
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")
  endif()
endif()

add_library(material STATIC)

target_sources(material PRIVATE src/json_loader.cpp)

# Public headers
target_include_directories(material PUBLIC include)
target_link_libraries(material PUBLIC nlohmann_json::nlohmann_json)

if(ENABLE_CUDA)
  target_compile_definitions(material PUBLIC MATERIAL_GPU_WITH_CUDA=1)
  set_target_properties(material PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )

  # Add CUDA-specific source files if they exist
  file(GLOB_RECURSE CUDA_SOURCES "src/*.cu")

  if(CUDA_SOURCES)
    target_sources(material PRIVATE ${CUDA_SOURCES})
  endif()

  # For device side linking of global symbols
  set_target_properties(material PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

add_executable(example examples/example.cu)

if(ENABLE_CUDA)
  set_source_files_properties(examples/example.cu PROPERTIES LANGUAGE CUDA)
  set_target_properties(example PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
else()
  # For non-CUDA builds, we need a C++ version
  message(WARNING "CUDA example requires CUDA to be enabled")
endif()

target_link_libraries(example PRIVATE material)

# GPU benchmark executable
add_executable(gpu_benchmark examples/gpu_benchmark.cpp)

if(ENABLE_CUDA)
  set_source_files_properties(examples/gpu_benchmark.cpp PROPERTIES LANGUAGE CUDA)
  set_target_properties(gpu_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
else()
  set_source_files_properties(examples/gpu_benchmark.cpp PROPERTIES LANGUAGE CXX)
endif()

target_link_libraries(gpu_benchmark PRIVATE material)

# Basic demo executable
add_executable(basic_demo examples/basic_demo.cpp)
target_link_libraries(basic_demo PRIVATE material)

# Advanced GPU demo executable
add_executable(advanced_gpu_demo examples/advanced_gpu_demo.cpp)

if(ENABLE_CUDA)
  set_source_files_properties(examples/advanced_gpu_demo.cpp PROPERTIES LANGUAGE CUDA)
  set_target_properties(advanced_gpu_demo PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
else()
  set_source_files_properties(examples/advanced_gpu_demo.cpp PROPERTIES LANGUAGE CXX)
endif()

target_link_libraries(advanced_gpu_demo PRIVATE material)

# Testing support
option(BUILD_TESTING "Build the testing tree" ON)

if(BUILD_TESTING)
  # Add Google Test
  add_subdirectory(third_party/googletest)

  # Enable CTest
  enable_testing()

  # Add test subdirectory
  add_subdirectory(tests)

  # Example and demo executables as tests (quick sanity runs)
  if(TARGET example)
    add_test(NAME Example_Run COMMAND $<TARGET_FILE:example>)
  endif()

  if(TARGET basic_demo)
    add_test(NAME BasicDemo_Run COMMAND $<TARGET_FILE:basic_demo>)
  endif()

  if(TARGET advanced_gpu_demo)
    # Run with smaller problem size to keep CI fast
    add_test(NAME AdvancedGpuDemo_Run COMMAND $<TARGET_FILE:advanced_gpu_demo> 200 0.5 50)
  endif()

  if(TARGET gpu_benchmark)
    add_test(NAME GpuBenchmark_Run COMMAND $<TARGET_FILE:gpu_benchmark>)
  endif()
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
