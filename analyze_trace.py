"""
this file will use the json file traces created by gvns/logger.py to analyze and print statistics
it will focus on the accepted moves (so the final step in gvns)
"""

import json

INPUT_FILE = "gvns_trace.jsonl"

def format_visit(rec):
    node = rec["node"]
    start = rec["start"]
    ready = rec["arrival"]
    wait = rec["wait"] if "wait" in rec else max(0, start - ready)
    return f"{node}(start={start:.1f}, [arr={ready:.1f}, wait={wait:.1f}])"


def process_vehicle(vtimeline):
    """vtimeline = dict {route, visits, travel, end_time}"""
    visits = vtimeline["visits"]
    if not visits:
        return "  (empty route)\n"

    lines = []

    # Start at depot
    # We don't know depot id inside timeline so we put 0 (common Liu format)
    depot = 0
    prev = f"{depot}(start=0.0)"
    for rec in visits:
        travel = rec["travel"]
        curr = format_visit(rec)
        lines.append(f"{prev} --{travel:.2f}--> {curr}")
        prev = curr

    # Return to depot
    last_node = visits[-1]["node"]
    # travel back is timeline["travel"] - sum(visits[i].travel)
    # but it's already inside end_time, so better recompute:
    # We attach travel-back only as meta (we do not know exact travel field)
    travel_back = vtimeline["end_time"] - visits[-1]["depart"]
    lines.append(f"{prev} --{travel_back:.2f}--> DEPOT")

    return "\n".join(lines) + "\n"


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except:
                continue

            if obj.get("event") != "accepted":
                continue

            print("\n" + "=" * 80)
            print(f"ACCEPTED STEP {obj['step']} | F={obj['Fval']:.3f}")
            print("=" * 80)

            timeline = obj.get("timeline", None)
            if not timeline:
                continue

            # For each vehicle
            for k, veh in enumerate(timeline):
                print(f"\nVehicle {k}:")
                print(process_vehicle(veh))

            print("-" * 80)


if __name__ == "__main__":
    main()