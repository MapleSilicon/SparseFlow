cmake_minimum_required(VERSION 3.10)
project(SparseFlowRuntime CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -march=native -mtune=native")

add_executable(benchmark_sparse
    masked_matmul.cpp
    benchmark_sparse.cpp
)
