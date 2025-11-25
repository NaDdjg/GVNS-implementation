"""
This analyzer loads the JSONL trace produced by gvns/logger.py
and extracts all accepted solutions, formatting the vehicle timelines
into a human-readable representation.

The results are written into a file: OUTPUT_FILE
"""

import json

INPUT_FILE = "gvns_trace.jsonl"
OUTPUT_FILE = "analysis_output.txt"


def format_visit(rec):
    node = rec["node"]
    start = rec["start"]
    ready = rec["ready"]
    due = rec["due"]
    depart  = rec["depart"]
    return f"{node}(start={start:.2f}, [ready={ready:.2f}, due={due:.2f}], depart={depart:.2f})"


def process_vehicle(vtimeline):
    """vtimeline = dict {route, visits, travel, end_time}"""
    visits = vtimeline["visits"]
    if not visits:
        return "  (empty route)\n"

    lines = []

    # assume depot=0 (Liu format)
    depot = "DEPOT(start=0.0)"
    prev = depot

    for rec in visits:
        travel = rec["travel"]
        curr = format_visit(rec)
        lines.append(f"{prev} --{travel:.2f}--> {curr}")
        prev = curr

    # Return to depot
    travel_back = vtimeline["end_time"] - visits[-1]["depart"]
    lines.append(f"{prev} --{travel_back:.2f}--> DEPOT")

    return "\n".join(lines) + "\n"


def main():
    out_lines = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except:
                continue

            if obj.get("event") != "accepted":
                continue

            # Header
            out_lines.append("\n" + "=" * 80)
            out_lines.append(
                f"ACCEPTED STEP {obj['step']} | F={obj['Fval']:.3f} | travel={obj['travel_time']:.3f} | penS={obj['penS']}"
            )
            out_lines.append("=" * 80)

            timeline = obj.get("timeline")
            if not timeline:
                continue

            # Each vehicle timeline
            for k, veh in enumerate(timeline):
                out_lines.append(f"\nVehicle {k}:")
                out_lines.append(process_vehicle(veh))
                out_lines.append("-" * 40)

            out_lines.append("\n")

    # Write to file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fw:
        fw.write("\n".join(out_lines))

    print(f"Analysis saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
