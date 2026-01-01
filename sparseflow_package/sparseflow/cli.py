import sys
import subprocess
from pathlib import Path

def _repo_root() -> Path:
    # sparseflow_package/sparseflow/cli.py -> repo root is two levels up
    here = Path(__file__).resolve()
    return here.parent.parent.parent  # .../SparseFlow

def _check_exists(path: Path, desc: str):
    if not path.exists():
        print(f"[sparseflow] ERROR: {desc} not found at: {path}", file=sys.stderr)
        sys.exit(1)

def demo_main():
    """Run the full SPA + runtime demo (same as ./spa-runner.sh)."""
    root = _repo_root()
    script = root / "spa-runner.sh"
    _check_exists(script, "spa-runner.sh")

    print(f"[sparseflow] Running demo via {script}")
    try:
        subprocess.run([str(script)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[sparseflow] Demo failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)

def analyze_main():
    """Run SPA + JSON export on a given MLIR file."""
    if len(sys.argv) < 2:
        print("Usage: sparseflow-analyze <input.mlir>", file=sys.stderr)
        sys.exit(1)

    root = _repo_root()
    arg = Path(sys.argv[1])

    candidates = []
    if arg.is_absolute():
        candidates.append(arg)
    else:
        # 1) Relative to repo root (preferred)
        candidates.append(root / arg)
        # 2) Relative to current working directory (fallback)
        candidates.append(Path.cwd() / arg)

    input_path = None
    for c in candidates:
        if c.exists():
            input_path = c
            break

    if input_path is None:
        print("[sparseflow] ERROR: Input MLIR file not found. Tried:", file=sys.stderr)
        for c in candidates:
            print(f"  - {c}", file=sys.stderr)
        sys.exit(1)

    plugin = root / "compiler/build/passes/SparseFlowPasses.so"
    _check_exists(plugin, "SparseFlowPasses.so")

    cmd = [
        "mlir-opt-19",
        "--allow-unregistered-dialect",
        f"--load-pass-plugin={plugin}",
        "-pass-pipeline=builtin.module(sparseflow-spa,sparseflow-spa-export)",
        str(input_path),
    ]

    print("[sparseflow] Running SPA analysis:")
    print("  " + " ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[sparseflow] SPA analysis failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)

def benchmark_main():
    """Run the C++ benchmark only."""
    root = _repo_root()
    exe = root / "runtime/build/benchmark_sparse"
    _check_exists(exe, "benchmark_sparse")

    print(f"[sparseflow] Running benchmark: {exe}")
    try:
        subprocess.run([str(exe)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[sparseflow] Benchmark failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)