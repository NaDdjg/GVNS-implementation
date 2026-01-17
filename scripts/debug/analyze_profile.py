"""
============================================================
GVNS PROJECT – PROFILING ANALYSIS 
============================================================

OUTPUT:
- File-by-file function timings
- Top slow functions
- For each slow function: where the time goes (callees)

"""

import pstats
from pathlib import Path
from collections import defaultdict
from datetime import datetime

PROFILE_FILE = "result_summaries/debug/gvns_profile.prof"
PROJECT_KEYWORD = "gvns"   
TOP_N = 15                 # number of slow functions to inspect

# ============================================================
# Output file
# ============================================================
OUTPUT_DIR = Path("result_summaries/debug")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = OUTPUT_DIR / f"gvns_profile_analysis_{timestamp}.txt"

# ============================================================
# Safety check
# ============================================================
if not Path(PROFILE_FILE).exists():
    raise FileNotFoundError(
        f"Profile file '{PROFILE_FILE}' not found. "
        "Run cProfile first."
    )

# ============================================================
# Load stats
# ============================================================
stats = pstats.Stats(PROFILE_FILE)
stats.strip_dirs()

# stats.stats structure:
# (filename, lineno, funcname) -> (cc, nc, tt, ct, callers)

raw_stats = stats.stats

# ============================================================
# collect only GVNS functions
# ============================================================
gvns_funcs = {}

for (filename, lineno, funcname), (cc, nc, tt, ct, callers) in raw_stats.items():
    if PROJECT_KEYWORD in filename:
        gvns_funcs[(filename, funcname, lineno)] = {
            "calls": nc,
            "self_time": tt,
            "cum_time": ct,
            "callers": callers,
        }

# ============================================================
# aggregate by file
# ============================================================
file_totals = defaultdict(float)

for (filename, _, _), data in gvns_funcs.items():
    file_totals[filename] += data["cum_time"]

# ============================================================
# sort functions by cumulative time
# ============================================================
sorted_funcs = sorted(
    gvns_funcs.items(),
    key=lambda x: x[1]["cum_time"],
    reverse=True
)

# ============================================================
# write report
# ============================================================
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    def write(line=""):
        f.write(line + "\n")

    write("=" * 80)
    write("GVNS PROFILING SUMMARY")
    write("=" * 80)
    write(f"Profile file : {PROFILE_FILE}")
    write(f"Project path : /gvns")
    write("")

    # --------------------------------------------------------
    # file-level summary
    # --------------------------------------------------------
    write("TIME SPENT PER FILE (cumulative)")
    write("-" * 80)

    for filename, total_time in sorted(
        file_totals.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        write(f"{filename:<40} {total_time:10.3f} s")

    write("")
    write("=" * 80)
    write("TOP SLOW FUNCTIONS")
    write("=" * 80)

    # --------------------------------------------------------
    # function-level summary
    # --------------------------------------------------------
    for i, ((filename, funcname, lineno), data) in enumerate(sorted_funcs[:TOP_N], 1):
        write(f"[{i}] {filename}:{lineno} → {funcname}")
        write(f"    calls     : {data['calls']}")
        write(f"    self time : {data['self_time']:.4f} s")
        write(f"    total time: {data['cum_time']:.4f} s")

        # ----------------------------------------------------
        # what causes the time (callees)
        # ----------------------------------------------------
        callees = data["callers"]

        if callees:
            write("    time spent in calls to:")
            sorted_callees = sorted(
                callees.items(),
                key=lambda x: x[1][3],  # cumulative time
                reverse=True
            )

            for (c_file, c_line, c_func), c_data in sorted_callees[:5]:
                if PROJECT_KEYWORD in c_file:
                    write(
                        f"        {c_file}:{c_line} → {c_func} "
                        f"({c_data[3]:.4f} s)"
                    )
        else:
            write("    no significant callees")

        write("-" * 80)

    write("")
    write("INTERPRETATION GUIDE")
    write("- self time high  → heavy loops inside function")
    write("- total time high → expensive calls inside")
    write("- optimize top 1–3 functions first")

print(f"Profiling analysis saved to: {OUTPUT_FILE}")
