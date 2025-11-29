cmake_minimum_required(VERSION 3.15)
project(SparseFlowRuntime)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(sparseflow_test
    src/main.cpp
)
