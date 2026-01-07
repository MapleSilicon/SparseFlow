"""Check if your GPU supports SparseFlow sparse operations"""
import sparseflow as sf

print("="*60)
print("SPARSEFLOW GPU COMPATIBILITY CHECK")
print("="*60)

supported, msg = sf.check_sparse_support()
print(f"\n{msg}\n")

if supported:
    print("✅ Your GPU is ready for SparseFlow sparse operations!")
    print("\nNext steps:")
    print("  python3 benchmarks/bench_dense_vs_sparse.py")
else:
    print("❌ Sparse operations NOT available on this GPU")
    print("\nSupported GPUs:")
    print("  • RTX 30-series (3090, 3080, 3070...)")
    print("  • RTX 40-series (4090, 4080...)")
    print("  • A100, A6000, A5000")
    print("  • H100, H200")
    print("\nNot supported:")
    print("  • V100, T4")
    print("  • RTX 20-series")
    print("  • GTX series")

print("="*60)
