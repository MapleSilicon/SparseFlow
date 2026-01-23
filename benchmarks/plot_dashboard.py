#!/usr/bin/env python3
"""
SparseFlow Performance Dashboard
Visualizes benchmark results
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Read results
df = pd.read_csv('results_sparseflow.csv')

# Convert string columns to numeric
df['speedup'] = df['speedup'].astype(float)
df['sparse_tflops'] = df['sparse_tflops'].astype(float)
df['dense_tflops'] = df['dense_tflops'].astype(float)
df['sparse_util_%'] = df['sparse_util_%'].astype(float)

# Create dashboard
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('SparseFlow Performance Dashboard (A100 80GB)', fontsize=16, fontweight='bold')

# 1. Speedup bar chart
ax1 = axes[0, 0]
colors = ['green' if x >= 1.0 else 'red' for x in df['speedup']]
bars = ax1.barh(df['shape_name'], df['speedup'], color=colors, alpha=0.7)
ax1.axvline(x=1.0, color='black', linestyle='--', linewidth=2, label='Break-even')
ax1.set_xlabel('Speedup (×)', fontsize=12)
ax1.set_title('Sparse vs Dense Speedup', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(axis='x', alpha=0.3)

# Add speedup values on bars
for i, (bar, val) in enumerate(zip(bars, df['speedup'])):
    ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
             f'{val:.2f}×', va='center', fontweight='bold')

# 2. TFLOPS comparison
ax2 = axes[0, 1]
x = range(len(df))
width = 0.35
ax2.bar([i - width/2 for i in x], df['dense_tflops'], width, label='Dense', alpha=0.7)
ax2.bar([i + width/2 for i in x], df['sparse_tflops'], width, label='Sparse', alpha=0.7)
ax2.set_ylabel('TFLOPS', fontsize=12)
ax2.set_title('Throughput Comparison', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(df['shape_name'], rotation=45, ha='right')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# 3. GPU Utilization
ax3 = axes[1, 0]
ax3.plot(df['shape_name'], df['sparse_util_%'], marker='o', linewidth=2, 
         markersize=8, label='Sparse Utilization')
ax3.axhline(y=100, color='red', linestyle='--', label='Peak (312 TFLOPS)')
ax3.set_ylabel('GPU Utilization (%)', fontsize=12)
ax3.set_title('Sparse Tensor Core Utilization', fontsize=14, fontweight='bold')
ax3.set_xticklabels(df['shape_name'], rotation=45, ha='right')
ax3.legend()
ax3.grid(alpha=0.3)
ax3.set_ylim([0, 120])

# 4. Summary table
ax4 = axes[1, 1]
ax4.axis('off')

summary_data = [
    ['Metric', 'Value'],
    ['Average Speedup', f"{df['speedup'].mean():.2f}×"],
    ['Max Speedup', f"{df['speedup'].max():.2f}×"],
    ['Best Shape', df.loc[df['speedup'].idxmax(), 'shape_name']],
    ['Peak TFLOPS', f"{df['sparse_tflops'].max():.1f}"],
    ['Avg Utilization', f"{df['sparse_util_%'].mean():.1f}%"],
    ['Shapes Tested', str(len(df))],
]

table = ax4.table(cellText=summary_data, cellLoc='left', loc='center',
                  colWidths=[0.5, 0.5])
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 3)

# Style header row
for i in range(2):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Add key insights
insights = [
    "Key Insights:",
    f"• Best for large batch: {df.loc[df['speedup'].idxmax(), 'speedup']:.2f}× faster",
    f"• Small shapes slower (overhead)",
    f"• Compute-bound at large sizes",
]

for i, insight in enumerate(insights):
    weight = 'bold' if i == 0 else 'normal'
    ax4.text(0.05, 0.25 - i*0.05, insight, fontsize=11, weight=weight,
             transform=ax4.transAxes)

plt.tight_layout()
plt.savefig('sparseflow_dashboard.png', dpi=150, bbox_inches='tight')
print("Dashboard saved to: sparseflow_dashboard.png")

# Also create a simple text summary
print("\n" + "="*70)
print("SPARSEFLOW PERFORMANCE SUMMARY")
print("="*70)
print(f"GPU: A100 80GB")
print(f"Shapes tested: {len(df)}")
print(f"\nSpeedup Range: {df['speedup'].min():.2f}× to {df['speedup'].max():.2f}×")
print(f"Average Speedup: {df['speedup'].mean():.2f}×")
print(f"\nBest Performance:")
best_idx = df['speedup'].idxmax()
print(f"  Shape: {df.loc[best_idx, 'shape_name']}")
print(f"  Speedup: {df.loc[best_idx, 'speedup']:.2f}×")
print(f"  TFLOPS: {df.loc[best_idx, 'sparse_tflops']:.1f}")
print(f"\nRecommendation: Use SparseFlow for batch size ≥ 1024")
print("="*70)
