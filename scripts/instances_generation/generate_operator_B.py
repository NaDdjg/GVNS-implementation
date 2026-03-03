import argparse
import os
import math
import random
from pathlib import Path
from copy import deepcopy

# ============================================================
# Parsing and writing (Liu format)
# ============================================================
def read_liu_instance(path):
    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

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
    for line in lines[idx:]:
        parts = line.split()
        customers.append({
            "id": int(parts[0]),
            "x": float(parts[1]),
            "y": float(parts[2]),
            "tw_start": float(parts[3]),
            "tw_end": float(parts[4]),
            "service": float(parts[5]),
        })

    return m, n, H, customers, sync_ids


# Writing Liu instance
def write_liu_instance(path, m, n, H, customers, sync_ids):
    with open(path, "w") as f:
        f.write(f"{m}\n")
        f.write(f"{n}\n")
        f.write(f"{H}\n")
        f.write(f"{len(sync_ids)}\n")
        if sync_ids:
            f.write(" ".join(map(str, sorted(sync_ids))) + "\n")

        for c in customers:
            f.write(
                f"{c['id']} {int(c['x'])} {int(c['y'])} "
                f"{int(c['tw_start'])} {int(c['tw_end'])} {int(c['service'])}\n"
            )

# ============================================================
# Operator B: Density increase
# ============================================================
def operator_B(
    m,
    n,
    H,
    customers,
    sync_ids,
    gamma=1.3,          # TW tightening factor (>1)
    alpha=0.2,          # center attraction strength [0, 0.4]
    seed=0
):
    rng = random.Random(seed)
    new_customers = deepcopy(customers)

    non_depot = customers[1:]

    # Compute original centers
    centers = [(c["tw_start"] + c["tw_end"]) / 2 for c in non_depot]
    global_center = sorted(centers)[len(centers) // 2]
    
    # Temporal densification + center attraction
    for c in non_depot:
        a, b = c["tw_start"], c["tw_end"]
        w = b - a
        if w <= 1:
            continue

        center = 0.5 * (a + b)

        # center attraction
        center = (1 - alpha) * center + alpha * global_center

        # shrink window
        new_w = w / gamma

        new_a = max(0, center - new_w / 2)
        new_b = min(H, center + new_w / 2)

        # enforce minimal width
        if new_b - new_a < 1:
            new_b = min(H, new_a + 1)

        c["tw_start"] = new_a
        c["tw_end"] = new_b

    # Synchronization
    target_sync = len(sync_ids)

    # Bias toward tight TW customers
    sorted_by_width = sorted(
        new_customers[1:],
        key=lambda c: c["tw_end"] - c["tw_start"]
    )

    new_sync_ids = [c["id"] for c in sorted_by_width[:target_sync]]

    return m, n, H, new_customers, new_sync_ids

# ============================================================
# Main CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Generate Operator B instances")
    parser.add_argument("--input_dir", required=True, help="Input instances directory")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--gamma", type=float, required=True, help="Density factor (>1)")
    parser.add_argument("--alpha", type=float, default=0.2, help="Center attraction strength [0, 0.4]")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for file in Path(args.input_dir).glob("*.txt"):
        m, n, H, customers, sync_ids = read_liu_instance(file)

        new_m, new_n, H, new_customers, new_sync_ids = operator_B(
            m, n, H, customers, sync_ids,
            gamma=args.gamma,
            alpha=args.alpha,
            seed=args.seed
        )

        base = file.stem

        out_name = (
            f"{base}_B{str(args.gamma).replace('.', 'p')}"
            f"_a{str(args.alpha).replace('.', 'p')}.txt"
        )
        out_path = Path(args.output_dir) / out_name

        write_liu_instance(out_path, new_m, new_n, H, new_customers, new_sync_ids)
        print(f"Generated {out_name}")


if __name__ == "__main__":
    main()