import os
import math
import random
import argparse
from typing import List, Tuple

# ============================================================
# Parsing and writing (Liu format)
# ============================================================
def read_instance(path: str):
    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip() != ""]

    idx = 0
    m = int(lines[idx]); idx += 1
    n = int(lines[idx]); idx += 1
    H = int(lines[idx]); idx += 1
    n_sync = int(lines[idx]); idx += 1

    sync_ids = []
    if n_sync > 0:
        sync_ids = list(map(int, lines[idx].split()))
        idx += 1

    customers = []
    for _ in range(n + 1):  # depot + customers
        cid, x, y, a, b, s = map(float, lines[idx].split())
        idx += 1
        customers.append({
            "id": int(cid),
            "x": x,
            "y": y,
            "tw_start": a,
            "tw_end": b,
            "s": s
        })

    return m, n, H, sync_ids, customers


def write_instance(path: str,
                   m: int,
                   customers: List[dict],
                   H: int,
                   sync_ids: List[int]):
    n = len(customers) - 1  # excluding depot

    with open(path, "w") as f:
        f.write(f"{m}\n")
        f.write(f"{n}\n")
        f.write(f"{H}\n")
        f.write(f"{len(sync_ids)}\n")
        if sync_ids:
            f.write(" ".join(map(str, sync_ids)) + "\n")

        for c in customers:
            f.write(
                f"{c['id']}\t"
                f"{int(c['x'])}\t{int(c['y'])}\t"
                f"{int(c['tw_start'])}\t{int(c['tw_end'])}\t"
                f"{int(c['s'])}\n"
            )


# ============================================================
# Operator A: Replication + Jitter
# ============================================================
def operator_A(
    m: int,
    n: int,
    H: int,
    sync_ids: List[int],
    customers: List[dict],
    k: float,
    delta_xy: float,
    delta_t: float,
    seed: int
):
    rng = random.Random(seed)

    # Base statistics
    base_customers = [c for c in customers if c["id"] != 0]
    depot = customers[0]

    total_service = sum(c["s"] for c in base_customers)
    base_load = total_service / (m * H)

    target_n = int(math.floor(k * n))
    replicas = int(math.ceil(target_n / n))

    # Spatial bounds
    xs = [c["x"] for c in base_customers]
    ys = [c["y"] for c in base_customers]
    dx = (max(xs) - min(xs)) * delta_xy
    dy = (max(ys) - min(ys)) * delta_xy

    # Replication + jitter
    pool = []

    for r in range(replicas):
        for c in base_customers:
            w = c["tw_end"] - c["tw_start"]
            center = 0.5 * (c["tw_start"] + c["tw_end"])

            # Spatial jitter
            x_new = c["x"] + rng.uniform(-dx, dx)
            y_new = c["y"] + rng.uniform(-dy, dy)

            # Temporal jitter (center-based)
            center_new = center + rng.uniform(-delta_t * H, delta_t * H)
            a_new = max(0, center_new - 0.5 * w)
            b_new = min(H, center_new + 0.5 * w)

            pool.append({
                "orig_id": c["id"],
                "x": x_new,
                "y": y_new,
                "tw_start": a_new,
                "tw_end": b_new,
                "s": c["s"]
            })

    # Truncate to target_n
    rng.shuffle(pool)
    pool = pool[:target_n]

    # Reindex customers
    new_customers = [depot.copy()]
    for idx, c in enumerate(pool, start=1):
        new_customers.append({
            "id": idx,
            "x": c["x"],
            "y": c["y"],
            "tw_start": c["tw_start"],
            "tw_end": c["tw_end"],
            "s": c["s"],
            "orig_id": c["orig_id"]
        })

    # Vehicles (preserve load tightness)
    new_total_service = sum(c["s"] for c in new_customers if c["id"] != 0)
    new_m = int(math.ceil(new_total_service / (base_load * H)))

    # Synchronization
    sync_ratio = len(sync_ids) / n if n > 0 else 0.0
    target_sync = int(math.floor(sync_ratio * target_n))

    # Group by original ID to avoid same-origin duplicates
    candidates = {}
    for c in new_customers[1:]:
        candidates.setdefault(c["orig_id"], []).append(c["id"])

    available = [rng.choice(v) for v in candidates.values()]
    # rng.shuffle(available)
    # to take tighter ones as sync, we could sort by time window width
    tw_width = {
        c["id"]: (c["tw_end"] - c["tw_start"])
        for c in new_customers
        if c["id"] != 0
    }

    available.sort(key=lambda cid: tw_width[cid])
    new_sync_ids = available[:target_sync]

    # Remove helper key
    for c in new_customers:
        c.pop("orig_id", None)

    return new_m, new_customers, H, new_sync_ids

# ============================================================
# Main CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Generate instances using Operator A")
    parser.add_argument("--input_dir", required=True, help="Folder with base instances")
    parser.add_argument("--output_dir", required=True, help="Folder for generated instances")
    parser.add_argument("--k", type=float, required=True, help="Replication factor")
    parser.add_argument("--delta_xy", type=float, default=0.03, help="Spatial jitter ratio")
    parser.add_argument("--delta_t", type=float, default=0.02, help="Temporal jitter ratio")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for fname in sorted(os.listdir(args.input_dir)):
        if not fname.endswith(".txt"):
            continue

        path = os.path.join(args.input_dir, fname)
        m, n, H, sync_ids, customers = read_instance(path)

        new_m, new_customers, H, new_sync_ids = operator_A(
            m, n, H, sync_ids, customers,
            k=args.k,
            delta_xy=args.delta_xy,
            delta_t=args.delta_t,
            seed=args.seed
        )

        family = fname.split("_")[0]
        new_n = len(new_customers) - 1
        new_sync = len(new_sync_ids)

        out_name = (
            f"{family}_{new_n}_{new_sync}_A{k_to_str(args.k)}.txt"
        )
        out_path = os.path.join(args.output_dir, out_name)

        write_instance(out_path, new_m, new_customers, H, new_sync_ids)

        print(f"Generated: {out_name}")


def k_to_str(k: float) -> str:
    """Helper to make filenames clean."""
    return str(k).replace(".", "p")


if __name__ == "__main__":
    main()