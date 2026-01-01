#!/bin/bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║         🚀 SPARSEFLOW v0.2.0 FULL BENCHMARK 🚀                 ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="benchmark_results_${TIMESTAMP}.txt"

exec > >(tee "$OUTPUT_FILE")
exec 2>&1

echo "Results will be saved to: $OUTPUT_FILE"
echo ""

# Step 1: Rebuild Runtime (ensure latest code)
echo "════════════════════════════════════════════════════════════════"
echo "🔨 STEP 1: Rebuilding Runtime"
echo "════════════════════════════════════════════════════════════════"
cd runtime/build
make -j8
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Runtime rebuilt successfully${NC}"
else
    echo -e "${RED}❌ Runtime build failed${NC}"
    exit 1
fi
cd ../..
echo ""

# Step 2: Quick validation
echo "════════════════════════════════════════════════════════════════"
echo "🧪 STEP 2: Quick Validation"
echo "════════════════════════════════════════════════════════════════"
cd runtime/build
export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH

if ./test_nm_runtime > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Runtime tests passed${NC}"
else
    echo -e "${RED}❌ Runtime tests failed${NC}"
    exit 1
fi
cd ../..
echo ""

# Step 3: Run full benchmark
echo "════════════════════════════════════════════════════════════════"
echo "📊 STEP 3: Running Full Benchmark Suite"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Testing matrix sizes: 256, 512, 1024, 2048"
echo "Testing patterns: 1:4, 2:4, 2:8, 4:16, 8:32"
echo ""
echo "This will take 2-5 minutes..."
echo ""

cd runtime/build
export LD_LIBRARY_PATH=$(pwd):$LD_LIBRARY_PATH

./benchmark_nm_runtime

BENCH_RC=$?
cd ../..

if [ $BENCH_RC -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Benchmark completed successfully${NC}"
else
    echo ""
    echo -e "${YELLOW}⚠️  Benchmark completed with warnings${NC}"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "📈 STEP 4: Analyzing Results"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Extract key speedup numbers
echo "Key Speedup Summary:"
echo "───────────────────────────────────────────────────────────────"
grep "Speedup:" "$OUTPUT_FILE" | head -20 || echo "Run completed - see full output above"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "💾 RESULTS SAVED"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Full results saved to: $OUTPUT_FILE"
echo ""
echo "To view again:"
echo "  cat $OUTPUT_FILE"
echo ""
echo "To share:"
echo "  cat $OUTPUT_FILE | grep -A100 'Pattern'"
echo ""

# Create a summary
SUMMARY_FILE="benchmark_summary_${TIMESTAMP}.txt"
cat > "$SUMMARY_FILE" << SUMMARY
╔════════════════════════════════════════════════════════════════╗
║           SPARSEFLOW v0.2.0 BENCHMARK SUMMARY                  ║
╚════════════════════════════════════════════════════════════════╝

Date: $(date)
System: $(uname -a)
CPU: $(lscpu | grep "Model name" | cut -d: -f2 | xargs)

KEY FINDINGS:
─────────────────────────────────────────────────────────────────

SUMMARY

grep "Pattern\|Speedup:" "$OUTPUT_FILE" >> "$SUMMARY_FILE" || true

echo ""
echo "Summary created: $SUMMARY_FILE"
echo ""

cat << 'DONE'
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║              ✅ BENCHMARK COMPLETE! ✅                          ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝

NEXT STEPS:

1. Review the results above
2. Check detailed output: cat benchmark_results_*.txt
3. Add to documentation if results are good
4. Use for investor materials

HONEST PERFORMANCE CLAIMS:
- Report the RANGE of speedups (e.g., "9-20×")
- Note the matrix sizes tested
- Mention this is CPU (GPU coming in v0.3)
- Don't cherry-pick outliers

DONE

