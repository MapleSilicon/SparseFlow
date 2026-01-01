#!/usr/bin/env python3
"""SparseFlow Benchmark Graph Generator"""

import csv
import sys
import os
from pathlib import Path

def generate_graphs(csv_file):
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
    except ImportError:
        print("Error: matplotlib not installed")
        print("Install with: pip3 install matplotlib")
        return False
    
    # Read CSV data
    sizes = []
    total_macs = []
    executed_macs = []
    speedups = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sizes.append(row['Matrix_Size'])
            total_macs.append(int(row['Total_MACs']))
            executed_macs.append(int(row['Executed_MACs']))
            speedups.append(float(row['Theoretical_Speedup']))
    
    output_dir = Path(csv_file).parent / "graphs"
    output_dir.mkdir(exist_ok=True)
    
    print(f"Generating graphs in {output_dir}/")
    
    # Graph 1: MACs Comparison
    plt.figure(figsize=(10, 6))
    x = range(len(sizes))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], total_macs, width, 
            label='Dense (Total MACs)', alpha=0.8, color='#e74c3c')
    plt.bar([i + width/2 for i in x], executed_macs, width, 
            label='Sparse (Executed MACs)', alpha=0.8, color='#2ecc71')
    
    plt.xlabel('Matrix Size', fontsize=12, fontweight='bold')
    plt.ylabel('MACs', fontsize=12, fontweight='bold')
    plt.title('SparseFlow: Compute Reduction with 2:4 Sparsity', fontsize=14, fontweight='bold')
    plt.xticks(x, sizes)
    plt.legend(fontsize=10)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'macs_comparison.png', dpi=150)
    plt.close()
    print("  ✓ macs_comparison.png")
    
    # Graph 2: Speedup
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, speedups, marker='o', linewidth=2, markersize=8, color='#3498db')
    plt.axhline(y=2.0, color='#e74c3c', linestyle='--', linewidth=2, label='Target: 2.0x')
    plt.xlabel('Matrix Size', fontsize=12, fontweight='bold')
    plt.ylabel('Theoretical Speedup', fontsize=12, fontweight='bold')
    plt.title('SparseFlow: Speedup Across Matrix Sizes', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'speedup_scaling.png', dpi=150)
    plt.close()
    print("  ✓ speedup_scaling.png")
    
    # Graph 3: Savings
    savings = [(t - e) / t * 100 for t, e in zip(total_macs, executed_macs)]
    plt.figure(figsize=(10, 6))
    plt.bar(sizes, savings, color='#2ecc71', alpha=0.8)
    plt.xlabel('Matrix Size', fontsize=12, fontweight='bold')
    plt.ylabel('Compute Savings (%)', fontsize=12, fontweight='bold')
    plt.title('SparseFlow: Percentage of MACs Eliminated', fontsize=14, fontweight='bold')
    plt.ylim([0, 100])
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'compute_savings.png', dpi=150)
    plt.close()
    print("  ✓ compute_savings.png")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 generate_graphs.py <benchmark_results.csv>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    if not os.path.exists(csv_file):
        print(f"Error: File not found: {csv_file}")
        sys.exit(1)
    
    print("=" * 60)
    print("SparseFlow Graph Generator")
    print("=" * 60)
    print("")
    
    if generate_graphs(csv_file):
        print("")
        print("✅ All graphs generated successfully!")
    else:
        print("❌ Graph generation failed")
        sys.exit(1)
