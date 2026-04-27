import json
import os

class PPOActionLogger:
    def __init__(self, output_path):
        self.output_path = output_path
        self.logs = []

    def log_decision(self, iteration, neigh_id, action):
        self.logs.append({
            "iteration": iteration,
            "neigh_id": neigh_id,
            "action": action,
        })

    def log_outcome(self, F_before, F_after, improved):
        if not self.logs:
            return
        self.logs[-1].update({
            "F_before": F_before,
            "F_after": F_after,
            "improved": improved,
        })

    def save(self):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(self.logs, f, indent=2)