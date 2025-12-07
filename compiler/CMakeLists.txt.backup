cmake_minimum_required(VERSION 3.20)
project(SparseFlowCompiler)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(passes)

# Add sparseflow-opt tool
add_subdirectory(tools/sparseflow-opt)

# Add sparseflow-opt tool
add_subdirectory(tools/sparseflow-opt)
