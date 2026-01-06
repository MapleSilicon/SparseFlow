#!/usr/bin/env python3
"""
sparseflow-audit - Analyze deployment costs and savings

Usage:
    sparseflow-audit --model llama-7b --qps 1000
    sparseflow-audit --model gpt2-xl --batch-size 32 --latency-target 50ms
"""

import argparse
import sys
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class ModelConfig:
    """Model configuration"""
    name: str
    params_billions: float
    layers: int
    hidden_size: int
    intermediate_size: int
    mlp_ratio: float = 4.0
    
    @property
    def mlp_params(self) -> int:
        """Compute MLP parameters"""
        # up_proj + down_proj
        return self.layers * (
            self.hidden_size * self.intermediate_size +  # up_proj
            self.intermediate_size * self.hidden_size    # down_proj
        )
    
    @property
    def sparse_mlp_params(self) -> int:
        """Compute sparse MLP parameters (50% reduction)"""
        return self.mlp_params // 2


# Popular model configurations
MODELS = {
    "gpt2": ModelConfig("GPT-2", 0.117, 12, 768, 3072),
    "gpt2-medium": ModelConfig("GPT-2 Medium", 0.345, 24, 1024, 4096),
    "gpt2-large": ModelConfig("GPT-2 Large", 0.774, 36, 1280, 5120),
    "gpt2-xl": ModelConfig("GPT-2 XL", 1.5, 48, 1600, 6400),
    "llama-7b": ModelConfig("LLaMA 7B", 6.7, 32, 4096, 11008),
    "llama-13b": ModelConfig("LLaMA 13B", 13.0, 40, 5120, 13824),
    "llama-30b": ModelConfig("LLaMA 30B", 32.5, 60, 6656, 17920),
    "llama-65b": ModelConfig("LLaMA 65B", 65.2, 80, 8192, 22016),
    "mistral-7b": ModelConfig("Mistral 7B", 7.3, 32, 4096, 14336),
}


@dataclass
class GPUConfig:
    """GPU configuration"""
    name: str
    memory_gb: int
    tflops_fp16: float
    price_per_hour: float
    power_watts: int
    
    
GPUS = {
    "a100-40gb": GPUConfig("A100 40GB", 40, 312, 2.21, 400),
    "a100-80gb": GPUConfig("A100 80GB", 80, 312, 3.67, 400),
    "h100": GPUConfig("H100", 80, 989, 4.76, 700),
    "rtx-3090": GPUConfig("RTX 3090", 24, 142, 0.50, 350),
    "rtx-4090": GPUConfig("RTX 4090", 24, 330, 0.70, 450),
}


class DeploymentAnalyzer:
    """Analyze deployment costs and savings"""
    
    def __init__(self, model: ModelConfig, gpu: GPUConfig):
        self.model = model
        self.gpu = gpu
        
    def compute_tokens_per_second(self, dense: bool = True) -> float:
        """Compute tokens/second for given configuration"""
        # Simplified model: TFLOPS / (params * 2)
        # Real calculation is more complex
        params = self.model.params_billions if dense else (
            self.model.params_billions * 0.75  # 50% reduction in MLP = 25% overall
        )
        
        # FLOPS per token (2 * params for forward pass)
        flops_per_token = params * 2e9
        
        # Tokens per second
        tokens_per_sec = self.gpu.tflops_fp16 * 1e12 / flops_per_token
        
        # Apply efficiency factor (sparse is more efficient)
        efficiency = 0.5 if dense else 0.7  # Sparse has better memory efficiency
        
        return tokens_per_sec * efficiency
    
    def compute_batch_latency(self, batch_size: int, dense: bool = True) -> float:
        """Compute latency for a batch (ms)"""
        tokens_per_sec = self.compute_tokens_per_second(dense)
        return (batch_size / tokens_per_sec) * 1000
    
    def compute_gpus_needed(self, target_qps: int, dense: bool = True) -> int:
        """Compute number of GPUs needed for target QPS"""
        tokens_per_gpu = self.compute_tokens_per_second(dense)
        gpus = max(1, int(target_qps / tokens_per_gpu) + 1)
        return gpus
    
    def compute_annual_cost(self, num_gpus: int) -> Dict[str, float]:
        """Compute annual operational costs"""
        hours_per_year = 24 * 365
        
        # GPU rental cost
        gpu_cost = num_gpus * self.gpu.price_per_hour * hours_per_year
        
        # Power cost (assume $0.12/kWh)
        power_kwh = (num_gpus * self.gpu.power_watts / 1000) * hours_per_year
        power_cost = power_kwh * 0.12
        
        # Carbon emissions (assume 0.5 kg CO2/kWh)
        carbon_tons = (power_kwh * 0.5) / 1000
        
        return {
            "gpu_cost": gpu_cost,
            "power_cost": power_cost,
            "total_cost": gpu_cost + power_cost,
            "carbon_tons": carbon_tons,
        }


def print_header(title: str):
    """Print section header"""
    print()
    print("╔" + "═" * 58 + "╗")
    print(f"║ {title:<56} ║")
    print("╚" + "═" * 58 + "╝")
    print()


def print_table_row(label: str, dense_val: str, sparse_val: str, improvement: str = ""):
    """Print comparison table row"""
    print(f"│ {label:<23} │ {dense_val:>10} │ {sparse_val:>10} │ {improvement:>7} │")


def print_separator():
    """Print table separator"""
    print("├─────────────────────────┼────────────┼────────────┼─────────┤")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze SparseFlow deployment costs and savings"
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODELS.keys()),
        help="Model architecture"
    )
    parser.add_argument(
        "--gpu",
        default="a100-80gb",
        choices=list(GPUS.keys()),
        help="GPU type (default: a100-80gb)"
    )
    parser.add_argument(
        "--qps",
        type=int,
        help="Target queries per second"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Get configurations
    model = MODELS[args.model]
    gpu = GPUS[args.gpu]
    
    # Create analyzer
    analyzer = DeploymentAnalyzer(model, gpu)
    
    # Print header
    print_header(f"SPARSEFLOW DEPLOYMENT ANALYSIS")
    
    print(f"Model: {model.name}")
    print(f"Parameters: {model.params_billions:.1f}B")
    print(f"GPU: {gpu.name}")
    print(f"Target QPS: {args.qps if args.qps else 'N/A'}")
    
    # Performance comparison
    print_header("PERFORMANCE COMPARISON")
    
    dense_tokens = analyzer.compute_tokens_per_second(dense=True)
    sparse_tokens = analyzer.compute_tokens_per_second(dense=False)
    
    dense_latency = analyzer.compute_batch_latency(args.batch_size, dense=True)
    sparse_latency = analyzer.compute_batch_latency(args.batch_size, dense=False)
    
    print("┌─────────────────────────┬────────────┬────────────┬─────────┐")
    print("│ Metric                  │    Dense   │ SparseFlow │  Δ      │")
    print_separator()
    print_table_row(
        "Tokens/sec (per GPU)",
        f"{dense_tokens:.1f}",
        f"{sparse_tokens:.1f}",
        f"+{((sparse_tokens/dense_tokens - 1) * 100):.0f}%"
    )
    print_table_row(
        f"Latency (batch={args.batch_size})",
        f"{dense_latency:.1f}ms",
        f"{sparse_latency:.1f}ms",
        f"-{((1 - sparse_latency/dense_latency) * 100):.0f}%"
    )
    print("└─────────────────────────┴────────────┴────────────┴─────────┘")
    
    # Cost comparison (if QPS specified)
    if args.qps:
        print_header("COST ANALYSIS")
        
        dense_gpus = analyzer.compute_gpus_needed(args.qps, dense=True)
        sparse_gpus = analyzer.compute_gpus_needed(args.qps, dense=False)
        
        dense_costs = analyzer.compute_annual_cost(dense_gpus)
        sparse_costs = analyzer.compute_annual_cost(sparse_gpus)
        
        print("┌─────────────────────────┬────────────┬────────────┬─────────┐")
        print("│ Metric                  │    Dense   │ SparseFlow │  Δ      │")
        print_separator()
        print_table_row("GPUs Required", str(dense_gpus), str(sparse_gpus), 
                       f"-{dense_gpus - sparse_gpus}")
        print_table_row(
            "Annual GPU Cost",
            f"${dense_costs['gpu_cost']/1000:.0f}K",
            f"${sparse_costs['gpu_cost']/1000:.0f}K",
            f"-${(dense_costs['gpu_cost'] - sparse_costs['gpu_cost'])/1000:.0f}K"
        )
        print_table_row(
            "Annual Power Cost",
            f"${dense_costs['power_cost']/1000:.0f}K",
            f"${sparse_costs['power_cost']/1000:.0f}K",
            f"-${(dense_costs['power_cost'] - sparse_costs['power_cost'])/1000:.0f}K"
        )
        print_table_row(
            "Total Annual Cost",
            f"${dense_costs['total_cost']/1000:.0f}K",
            f"${sparse_costs['total_cost']/1000:.0f}K",
            f"-${(dense_costs['total_cost'] - sparse_costs['total_cost'])/1000:.0f}K"
        )
        print_table_row(
            "Carbon (tons CO₂/year)",
            f"{dense_costs['carbon_tons']:.1f}",
            f"{sparse_costs['carbon_tons']:.1f}",
            f"-{dense_costs['carbon_tons'] - sparse_costs['carbon_tons']:.1f}"
        )
        print("└─────────────────────────┴────────────┴────────────┴─────────┘")
        
        # ROI Summary
        print_header("RETURN ON INVESTMENT")
        
        annual_savings = dense_costs['total_cost'] - sparse_costs['total_cost']
        print(f"💰 Annual Savings: ${annual_savings:,.0f}")
        print(f"🌱 Carbon Reduction: {dense_costs['carbon_tons'] - sparse_costs['carbon_tons']:.1f} tons CO₂/year")
        print(f"⚡ Latency Improvement: {((1 - sparse_latency/dense_latency) * 100):.0f}% faster")
        print(f"📊 GPU Reduction: {((1 - sparse_gpus/dense_gpus) * 100):.0f}% fewer GPUs")
        
        print()
        print("Recommendation: ✅ Deploy SparseFlow")
        print(f"ROI Timeline: Immediate (operational savings)")
        print()


if __name__ == "__main__":
    main()
