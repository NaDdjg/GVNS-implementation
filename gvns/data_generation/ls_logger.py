import csv
import os
from typing import Dict, Any

class LSDecisionLogger:
    """
    Logs one row per Local Search (LS) call inside BVND.

    each row corresponds to:
    - one neighborhood attempt
    - at a specific GVNS iteration
    - with a full feature vector
    - and the observed outcome (improvement or not)
    """

    def __init__(
        self,
        output_path: str,
        instance_name: str,
        mode: str,  # "FI" or "BI"
    ):
        self.output_path = output_path
        self.instance_name = instance_name
        self.mode = mode

        # incremental counter for LS calls (purely informational)
        self.ls_call_id = 0

        # ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        header_exists = (
            os.path.exists(output_path)
            and os.path.getsize(output_path) > 0
        )

        # open file in append mode
        self.f = open(output_path, "a", newline="")
        self.writer = None

        # header will be written lazily on first log,
        self.header_written = header_exists

    # =========================================================
    # core logging method
    # =========================================================
    def log(
        self,
        *,
        gvns_iter: int,
        neigh_id: int,
        neigh_name: str,
        features: Dict[str, Any],
        improved: bool,
        F_before: float,
        F_after: float,
        elapsed: float,
    ):
        """
        Log one LS decision.
        will be called after search_state.after_ls() and after features are extracted.
        """
        self.ls_call_id += 1

        # build flat row dictionary
        row = {}

        # identifiers 
        row["instance_name"] = self.instance_name
        row["mode"] = self.mode
        row["gvns_iter"] = gvns_iter
        row["ls_call_id"] = self.ls_call_id
        row["neigh_id"] = neigh_id
        row["neigh_name"] = neigh_name

        # features 
        # features are already flat and numeric
        for k, v in features.items():
            row[k] = v

        # outcome 
        row["improved"] = int(improved)
        row["delta"] = F_before - F_after
        row["elapsed"] = elapsed
        row["F_before"] = F_before
        row["F_after"] = F_after

        # init writer
        if self.writer is None:
            self.writer = csv.DictWriter(
                self.f,
                fieldnames=list(row.keys())
            )

        # write header if needed
        if not self.header_written:
            self.writer.writeheader()
            self.header_written = True

        # write row immediately
        self.writer.writerow(row)
        self.f.flush()  # important for long runs

    # =========================================================
    # cleanup
    # =========================================================
    def close(self):
        """Close the underlying file."""
        if self.f:
            self.f.close()
            self.f = None
