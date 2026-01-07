#!/bin/bash
# SparseFlow Automated Benchmark Suite
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "SparseFlow Benchmark Suite v0.1"
echo "=========================================="
echo ""

mkdir -p benchmarks/results
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="benchmarks/results/$TIMESTAMP"
mkdir -p "$RESULTS_DIR"

CSV_FILE="$RESULTS_DIR/benchmark_results.csv"
echo "Matrix_Size,M,N,K,Total_MACs,Executed_MACs,Density,Theoretical_Speedup,Build_Time_sec" > "$CSV_FILE"

SIZES=(32 64 128 256 512)
echo "Testing ${#SIZES[@]} matrix sizes: ${SIZES[*]}"
echo ""

# Function to create MLIR test file
create_mlir_test() {
    local size=$1
    local filename=$2
    
    cat > "$filename" << MLIREOF
module {
  func.func @matmul_${size}(%arg0: tensor<${size}x${size}xf32>, 
                            %arg1: tensor<${size}x${size}xf32>) -> tensor<${size}x${size}xf32> {
    %c0 = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<${size}x${size}xf32>
    %filled = linalg.fill ins(%c0 : f32) outs(%init : tensor<${size}x${size}xf32>) -> tensor<${size}x${size}xf32>
    %result = linalg.matmul ins(%arg0, %arg1 : tensor<${size}x${size}xf32>, tensor<${size}x${size}xf32>)
                           outs(%filled : tensor<${size}x${size}xf32>) -> tensor<${size}x${size}xf32>
    return %result : tensor<${size}x${size}xf32>
  }
}
MLIREOF
}

# Run benchmark for each size
for SIZE in "${SIZES[@]}"; do
    echo "=========================================="
    echo "Benchmarking ${SIZE}×${SIZE} matrix"
    echo "=========================================="
    
    TEST_FILE="compiler/test/bench_${SIZE}x${SIZE}.mlir"
    create_mlir_test "$SIZE" "$TEST_FILE"
    echo "✓ Created $TEST_FILE"
    
    rm -rf compiler/build runtime/build
    
    BUILD_START=$(date +%s)
    export SPARSEFLOW_MLIR_FILE="compiler/test/bench_${SIZE}x${SIZE}.mlir"
    
    echo "Building and running pipeline..."
    ./build_all.sh > "$RESULTS_DIR/build_${SIZE}.log" 2>&1
    
    BUILD_END=$(date +%s)
    BUILD_TIME=$((BUILD_END - BUILD_START))
    echo "✓ Build completed in ${BUILD_TIME}s"
    
    JSON_FILE="compiler/build/hardware_config.json"
    
    if [ -f "$JSON_FILE" ]; then
        RESULTS=$(python3 << PYEOF
import json
try:
    with open('$JSON_FILE', 'r') as f:
        data = json.load(f)
    
    if 'operations' in data and len(data['operations']) > 0:
        op = data['operations'][0]
        M = op.get('M', 0)
        N = op.get('N', 0)
        K = op.get('K', 0)
        total = op.get('totalMACs', 0)
        executed = op.get('executedMACs', 0)
        density = op.get('density', 0.0)
        speedup = op.get('theoreticalSpeedup', 0.0)
        
        print(f"${SIZE}x${SIZE},{M},{N},{K},{total},{executed},{density:.3f},{speedup:.2f},${BUILD_TIME}")
except:
    print("${SIZE}x${SIZE},0,0,0,0,0,0.0,0.0,${BUILD_TIME}")
PYEOF
)
        
        echo "$RESULTS" >> "$CSV_FILE"
        echo "✓ Results: $RESULTS"
    fi
    echo ""
done

echo "=========================================="
echo "Benchmark Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $CSV_FILE"
echo ""
echo "=== BENCHMARK SUMMARY ==="
column -t -s',' "$CSV_FILE"
echo ""

# Generate analysis
python3 << 'PYEOF'
import csv
import sys

csv_file = sys.argv[1] if len(sys.argv) > 1 else 'benchmark_results.csv'

print("=== PERFORMANCE ANALYSIS ===")
print("")

with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

if rows:
    print(f"{'Size':<12} {'Total MACs':<15} {'Executed MACs':<15} {'Speedup':<10}")
    print("-" * 60)
    
    for row in rows:
        size = row['Matrix_Size']
        total = int(row['Total_MACs'])
        executed = int(row['Executed_MACs'])
        speedup = float(row['Theoretical_Speedup'])
        print(f"{size:<12} {total:>14,} {executed:>14,} {speedup:>9.2f}x")
PYEOF "$CSV_FILE"

echo ""
echo "Next: python3 generate_graphs.py $CSV_FILE"
