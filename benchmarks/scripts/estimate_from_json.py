import json
import sys

def parse_nm(nm):
    try:
        a,b = nm.split(":")
        a,b = int(a), int(b)
        return a / b
    except Exception:
        return 1.0  # treat as dense if missing/bad

def main(path):
    with open(path) as f:
        data = json.load(f)
    mem_cost = 1e-6
    pe_cost  = 5e-7
    total_dense = 0
    total_nnz   = 0
    total_skipped = 0
    total_time_ms = 0.0
    total_energy_mj = 0.0

    for fn in data.get("functions", []):
        for mm in fn.get("matmuls", []):
            M,K,N = mm["M"], mm["K"], mm["N"]
            dense_ops = M*K*N
            rho = parse_nm(mm.get("nm","dense"))
            nnz = int(dense_ops * rho)
            skipped = dense_ops - nnz
            time_ms = mem_cost*nnz + pe_cost*nnz
            energy_mj = 0.001 * time_ms

            total_dense += dense_ops
            total_nnz += nnz
            total_skipped += skipped
            total_time_ms += time_ms
            total_energy_mj += energy_mj

            print(f"{fn['name']}: matmul M={M} K={K} N={N} nm={mm.get('nm')}")
            print(f"  dense_ops={dense_ops} nnz≈{nnz} zero_skipped≈{skipped}")
            print(f"  est_time_ms={time_ms:.6f} est_energy_mj={energy_mj:.6f}")

    print("---- totals ----")
    print(f"dense_ops_total={total_dense} nnz_total={total_nnz} zero_skipped_total={total_skipped}")
    print(f"est_time_ms_total={total_time_ms:.6f} est_energy_mj_total={total_energy_mj:.6f}")

if __name__ == "__main__":
    main(sys.argv[1])
