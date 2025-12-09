cmake_minimum_required(VERSION 3.20)
project(SparseFlowRuntime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP)

add_library(sparseflow_runtime SHARED
    src/sparseflow_runtime.cpp
)

target_include_directories(sparseflow_runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(sparseflow_runtime PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(test_runtime test_runtime.cpp)
target_link_libraries(test_runtime PRIVATE sparseflow_runtime)

# N:M pattern test
add_executable(test_nm_runtime test_nm_runtime.cpp)
target_link_libraries(test_nm_runtime PRIVATE sparseflow_runtime)
target_include_directories(test_nm_runtime PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Pattern validation test
add_executable(test_pattern_validation test_pattern_validation.cpp)
target_link_libraries(test_pattern_validation PRIVATE sparseflow_runtime)
target_include_directories(test_pattern_validation PRIVATE ${CMAKE_SOURCE_DIR}/include)
