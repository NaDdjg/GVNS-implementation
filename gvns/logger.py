import json
import time
from typing import Optional, TextIO

class GVNSLogger:
    def __init__(self, path: Optional[str] = "gvns_trace.jsonl", console: bool = True):
        self.path = path
        self.console = console
        self.handle: Optional[TextIO] = None
        if self.path:
            self.handle = open(self.path, "a", encoding="utf-8")

    def close(self):
        if self.handle:
            self.handle.close()
            self.handle = None

    def _write_json(self, obj):
        if self.handle:
            self.handle.write(json.dumps(obj, default=str) + "\n")
            self.handle.flush()

    def log(self, event_type: str, step: int, solution, note: str = "", extra: dict = None):
        """
        event_type: "init", "shaken", "after_ls", "accepted", "rejected", "vnd_improve", ...
        step: iteration number (or 0 for init)
        solution: Solution object
        note: short human note string
        extra: optional dict with any extra fields
        """
        tstamp = time.time()
        record = {
            "ts": tstamp,
            "event": event_type,
            "step": step,
            "note": note,
            "tours": solution.tours,
            "travel_time": float(solution.travel_time),
            "penT": float(solution.penT),
            "penH": float(solution.penH),
            "penS": float(solution.penS),
            "Fval": float(solution.Fval),
            "w1": float(solution.w1),
            "w2": float(solution.w2),
            "alpha": float(solution.alpha),
            "M": float(solution.M),
        }
        # attach timeline if solution provides it
        try:
            timeline = solution.compute_timeline_postsync()
            record["timeline"] = timeline
        except Exception:
            # compute_timeline may not exist or may fail; ignore
            record["timeline"] = None

        if extra:
            record.update(extra)

        self._write_json(record)

        if self.console:
            # pretty console output (compact)
            summary = f"[{time.strftime('%H:%M:%S', time.localtime(tstamp))}] {event_type.upper()} step={step} F={solution.Fval:.4f} f={solution.travel_time:.4f} PenT={solution.penT:.4f} PenH={solution.penH:.4f} PenS={solution.penS}"
            print(summary)
            # print per-vehicle compact lines
            if record["timeline"]:
                for k, veh in enumerate(record["timeline"]):
                    if not veh["route"]:
                        continue
                    print(f"  V{k}: route={veh['route']} end_time={veh['end_time']:.3f} travel={veh['travel']:.3f}")
            print("-" * 80)
