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

# Tests
add_executable(test_nm_runtime tests/test_nm_runtime.cpp)
target_link_libraries(test_nm_runtime PRIVATE sparseflow_runtime)
target_include_directories(test_nm_runtime PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(test_pattern_validation tests/test_pattern_validation.cpp)
target_link_libraries(test_pattern_validation PRIVATE sparseflow_runtime)
target_include_directories(test_pattern_validation PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Benchmark
add_executable(benchmark_nm_runtime tests/benchmark_nm_runtime.cpp)
target_link_libraries(benchmark_nm_runtime PRIVATE sparseflow_runtime)
target_include_directories(benchmark_nm_runtime PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Re-test binary for stability checking
add_executable(retest_8_32 tests/retest_8_32.cpp)
target_link_libraries(retest_8_32 PRIVATE sparseflow_runtime)
target_include_directories(retest_8_32 PRIVATE ${CMAKE_SOURCE_DIR}/include)
