cmake_minimum_required(VERSION 3.20)
project(SparseFlowCompiler)

set(CMAKE_CXX_STANDARD 17)

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/../runtime/include)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
link_directories(${PROJECT_SOURCE_DIR}/../runtime/build)

add_definitions(${LLVM_DEFINITIONS})

# Add passes subdirectory (builds SparseFlowPasses.so plugin)
add_subdirectory(passes)

# Runtime registration as a SHARED library (not MODULE)
add_library(SparseRuntimeReg SHARED
    runtime/SparseRuntimeRegistration.cpp
)

target_link_libraries(SparseRuntimeReg
    PUBLIC
    MLIRExecutionEngine
    MLIRIR
    LLVMSupport
    sparseflow_runtime
)

# Runner tool
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(sparseflow-runner
    tools/sparseflow-runner.cpp
)

target_link_libraries(sparseflow-runner
    PRIVATE
    SparseRuntimeReg           # Link runtime registration
    ${dialect_libs}
    ${conversion_libs}
    MLIRExecutionEngine
    MLIRIR
    MLIRParser
    MLIRTargetLLVMIRExport
    MLIRSupport
    MLIROptLib
)

llvm_update_compile_flags(sparseflow-runner)

add_executable(test_with_data
    tests/test_with_data.cpp
)

target_link_libraries(test_with_data
    PRIVATE
    sparseflow_runtime
)

target_include_directories(test_with_data PRIVATE
    ${CMAKE_SOURCE_DIR}/../runtime/include
)

add_executable(test_jit_correctness
    tests/test_jit_correctness.cpp
)

target_link_libraries(test_jit_correctness PRIVATE sparseflow_runtime)
target_include_directories(test_jit_correctness PRIVATE ${CMAKE_SOURCE_DIR}/../runtime/include)

add_executable(benchmark_suite
    tests/benchmark_suite.cpp
)

target_link_libraries(benchmark_suite PRIVATE sparseflow_runtime)
target_include_directories(benchmark_suite PRIVATE ${CMAKE_SOURCE_DIR}/../runtime/include)

# N:M pattern benchmark
add_executable(benchmark_nm_patterns
    build/benchmark_nm_patterns.cpp
)

target_link_libraries(benchmark_nm_patterns PRIVATE sparseflow_runtime)
target_include_directories(benchmark_nm_patterns PRIVATE ${CMAKE_SOURCE_DIR}/../runtime/include)
