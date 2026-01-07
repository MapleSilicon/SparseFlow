# MLIR components for SparseFlow

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/mlir/include)
include_directories(${PROJECT_BINARY_DIR}/mlir/include)

# SparseFlow MLIR Dialect library
add_mlir_dialect_library(MLIRSparseFlowDialect
    lib/Dialect/SparseFlowDialect.cpp
    lib/Dialect/SparseFlowOps.cpp
)

# SparseFlow MLIR Passes library
add_library(MLIRSparseFlowPasses
    lib/Passes/TileSizeOptimization.cpp
    lib/Passes/OperationFusion.cpp
    lib/Passes/PassRegistration.cpp
)

target_link_libraries(MLIRSparseFlowPasses
    PUBLIC
    MLIRSparseFlowDialect
    MLIRIR
    MLIRPass
    MLIRTransforms
)
