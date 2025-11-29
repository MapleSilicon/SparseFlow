cmake_minimum_required(VERSION 3.15)
project(SparseFlowRuntime)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Create standalone executable directly
add_executable(sparseflow_test
    src/main.cpp
)

# Set compiler flags
target_compile_options(sparseflow_test PRIVATE -Wall -Wextra)
