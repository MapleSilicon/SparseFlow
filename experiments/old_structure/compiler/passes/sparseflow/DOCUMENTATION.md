# SparseFlow Passes Documentation

## Overview
SparseFlow provides MLIR passes for static sparsity analysis and optimization.

## Available Passes

### 1. SparsityPropagationPass (SPA v0.6)
**Purpose**: Propagates 2D sparsity (rows + columns) through operations
**Input**: MLIR with linalg.matmul operations
**Output**: MLIR with sparseflow.spa_rowmask and sparseflow.spa_colmask attributes
**Usage**: `--sparseflow-spa`

### 2. SPAExportPass  
**Purpose**: Exports sparsity information to JSON
**Input**: MLIR with sparsity attributes
**Output**: `spa_sparsity.json` with row/column masks
**Usage**: `--sparseflow-spa-export`

### 3. AnnotateNmPass
**Purpose**: Annotates N:M sparsity patterns
**Input**: MLIR with tensor operations
**Output**: MLIR with sparseflow.n and sparseflow.m attributes
**Usage**: `--sparseflow-annotate-nm`

### 4. ExportMetadataPass
**Purpose**: Exports N:M metadata as JSON
**Input**: MLIR with N:M attributes
**Output**: JSON with sparsity metadata
**Usage**: `--sparseflow-export-metadata`

### 5. FlopCounterPass
**Purpose**: Counts FLOPs for matmul operations
**Input**: MLIR with linalg.matmul
**Output**: Prints FLOP counts to console
**Usage**: `--sparseflow-flop-counter`

## Pipeline Example
