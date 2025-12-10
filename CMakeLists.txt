cmake_minimum_required(VERSION 3.14)
project(SparseFlow)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build compiler
add_subdirectory(compiler)

# Build runtime
add_subdirectory(runtime)
