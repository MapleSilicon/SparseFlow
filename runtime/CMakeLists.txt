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
