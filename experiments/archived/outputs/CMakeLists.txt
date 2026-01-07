cmake_minimum_required(VERSION 3.18)
project(SparseFlow CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)

# Check files exist
message(STATUS "Checking source files...")
if(EXISTS "${CMAKE_SOURCE_DIR}/src/kernels/sparse_gemm_fused.cu")
    message(STATUS "✓ Found sparse_gemm_fused.cu")
else()
    message(FATAL_ERROR "✗ Missing sparse_gemm_fused.cu")
endif()

# SparseFlow library
add_library(sparseflow SHARED
    src/kernels/sparse_gemm_fused.cu
    src/runtime/sparse_runtime.cpp
)

target_link_libraries(sparseflow
    CUDA::cudart
)

set_target_properties(sparseflow PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Install rules
install(TARGETS sparseflow LIBRARY DESTINATION lib)
install(DIRECTORY include/sparseflow DESTINATION include)

message(STATUS "✓ Configuration complete")
