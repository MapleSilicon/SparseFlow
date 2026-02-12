"""
Wrapper so users can run:
  python -m sparseflow.tools.bench_prefill_e2e ...

Internally dispatches to the repo-root script: tools/bench_prefill_e2e.py
"""
import runpy
from pathlib import Path

def main():
    repo_root = Path(__file__).resolve().parents[2]
    target = repo_root / "tools" / "bench_prefill_e2e.py"
    if not target.exists():
        raise FileNotFoundError(f"Missing {target}. Repo layout changed?")
    runpy.run_path(str(target), run_name="__main__")

if __name__ == "__main__":
    main()
