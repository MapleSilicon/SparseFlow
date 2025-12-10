cmake_minimum_required(VERSION 3.10)
project(SparseFlowRuntime)

find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Runtime library
add_library(sparseflow_runtime SHARED
    src/sparseflow_runtime.cpp
)

target_include_directories(sparseflow_runtime PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(sparseflow_runtime PRIVATE OpenMP::OpenMP_CXX)

# Test executables (only if files exist)
if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_nm_runtime.cpp")
    add_executable(test_nm_runtime tests/test_nm_runtime.cpp)
    target_link_libraries(test_nm_runtime PRIVATE sparseflow_runtime)
    target_include_directories(test_nm_runtime PRIVATE ${CMAKE_SOURCE_DIR}/include)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_pattern_validation.cpp")
    add_executable(test_pattern_validation tests/test_pattern_validation.cpp)
    target_link_libraries(test_pattern_validation PRIVATE sparseflow_runtime)
    target_include_directories(test_pattern_validation PRIVATE ${CMAKE_SOURCE_DIR}/include)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/tests/benchmark_nm_runtime.cpp")
    add_executable(benchmark_nm_runtime tests/benchmark_nm_runtime.cpp)
    target_link_libraries(benchmark_nm_runtime PRIVATE sparseflow_runtime)
    target_include_directories(benchmark_nm_runtime PRIVATE ${CMAKE_SOURCE_DIR}/include)
endif()
