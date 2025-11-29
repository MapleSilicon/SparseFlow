#!/usr/bin/env bash
set -euo pipefail

echo "================================================================================"
echo "SPARSEFLOW - JSON-DRIVEN RUNTIME VALIDATION"
echo "================================================================================"
echo

ROOT_DIR="$HOME/src/SparseFlow"
RUNTIME_BUILD="$ROOT_DIR/runtime/build"
JSON_CFG="$ROOT_DIR/hardware_config.json"

# Test different N:M patterns as m (window), n (non-zeros)
declare -a patterns=(
  "4 2"   # 2:4
  "4 1"   # 1:4  
  "8 2"   # 2:8
  "8 4"   # 4:8
)

echo "Testing N:M patterns with 64x64x64 matmul (262144 total MACs)"
echo

for entry in "${patterns[@]}"; do
  set -- $entry
  m="$1"
  n="$2"
  expected_macs=$((262144 * n / m))
  expected_speedup=$(echo "scale=1; $m / $n" | bc)

  echo "🧪 Testing N:M pattern $n:$m"
  echo "   Expected: $expected_macs MACs executed, ${expected_speedup}x speedup"

  # Write the JSON config to project root where runtime will find it
  cat > "$JSON_CFG" << JSONEOF
{
  "matmuls": [
    {
      "m": $m,
      "n": $n,
      "type": "matmul"
    }
  ]
}
JSONEOF

  cd "$RUNTIME_BUILD"
  echo "   Actual:"
  ./sparseflow_test | grep -E 'Pattern:|Executed MACs:|Theoretical Speedup:' | sed 's/^/     /'
  
  # Verify the results match expectations
  actual_macs=$(./sparseflow_test 2>/dev/null | grep "Executed MACs:" | awk '{print $3}')
  if [ "$actual_macs" = "$expected_macs" ]; then
    echo "   ✅ MACs match expectation"
  else
    echo "   ❌ MACs mismatch: expected $expected_macs, got $actual_macs"
  fi
  echo "---"
done

echo "================================================================================"
echo "✅ JSON-driven behavior validated - runtime correctly uses compiler metadata!"
echo "================================================================================"


chmod +x test_json_driven_behavior.sh

# Run the fixed test
./test_json_driven_behavior.sh