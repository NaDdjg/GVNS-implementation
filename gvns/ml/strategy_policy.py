import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path

class MLStrategyPolicy:
    def __init__(self, clf_path, reg_path, features_path, instance_name="unknown", log_decisions=False):
        self.clf = joblib.load(clf_path)
        self.reg = joblib.load(reg_path)

        self.instance_name = instance_name
        self.log_decisions = log_decisions
        self.decision_log = [] if log_decisions else None

        # output directory for plots
        self.output_dir = Path("result_summaries") / "ml_policy_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(features_path) as f:
            self.feature_names_full = json.load(f)

        # features used for decision making (exclude strategy label)
        self.feature_names_state = [
            f for f in self.feature_names_full if f != "strategy"
        ]
        self.strategy_idx = self.feature_names_full.index("strategy")

    def decide(self, features_dict):
        assert "strategy" not in features_dict, (
            "Inference features must NOT include 'strategy'"
        )

        # build full feature vector (strategy left unset for now)
        X = np.zeros((1, len(self.feature_names_full)))

        for i, fname in enumerate(self.feature_names_full):
            if fname == "strategy":
                continue
            X[0, i] = features_dict[fname]

        
        X_bi = X.copy()
        X_fi = X.copy()

        X_bi[0, self.strategy_idx] = 1
        X_fi[0, self.strategy_idx] = 0

        # head 1: classifier (strategy-agnostic)
        p_bi = self.clf.predict_proba(X_bi)[0, 1]
        p_fi = self.clf.predict_proba(X_fi)[0, 1]

        # head 2: regressor (strategy-conditional)
        gain_bi = self.reg.predict(X_bi)[0]
        gain_fi = self.reg.predict(X_fi)[0]

        # final decision
        score_bi = p_bi * gain_bi
        score_fi = p_fi * gain_fi

        # phase aware hard rule
        time_ratio = features_dict.get("time_elapsed_ratio", 1.0)
        stagnation = features_dict.get("stagnation_ls", 0)
        
        # early stabilization
        if time_ratio < 0.05:
            decision = "FI"
            decision_reason = "warmup"
        
        else:
            # bias BI when stagnating
            if stagnation >= 3:
                score_bi *= 1.2   # soft boost, not a gate
        
            decision = "BI" if score_bi > score_fi else "FI"
            decision_reason = "ml_soft_bias"

        result = {
            "decision": decision,
            "decision_reason": decision_reason,
            "p_fi": p_fi,
            "p_bi": p_bi,
            "gain_fi": gain_fi,
            "gain_bi": gain_bi,
            "score_fi": score_fi,
            "score_bi": score_bi,
        }

        if self.log_decisions:
            self.decision_log.append({
                **result,
                "stagnation": features_dict.get("stagnation_ls"),
                "time_elapsed_ratio": features_dict.get("time_elapsed_ratio"),
                "k": features_dict.get("k"),
                "iter_gvns": features_dict.get("iter_gvns"),
            })

        return result
    
    def _plot_path(self, name: str) -> Path:
        safe_inst = (
            self.instance_name
            .replace("/", "_")
            .replace("\\", "_")
            .replace(".txt", "")
        )
        return self.output_dir / f"{safe_inst}__{name}"
    
    def save_decisions_csv(self):
        """
        Save raw ML decisions for offline analysis.
        One row = one decision call.
        """
        if not self.decision_log:
            print("[ML] No decisions to save.")
            return
    
        df = pd.DataFrame(self.decision_log)
    
        path = self._plot_path("ml_decisions_trace.csv")
        df.to_csv(path, index=False)
        
    def plot_decisions(self, show=False):  
        if self.log_decisions is False:
           print("[ML] Decision logging is disabled. No plots to show.")
           return 
        
        if not self.decision_log:
            raise RuntimeError("No decisions logged.")

        import matplotlib.pyplot as plt

        df = pd.DataFrame(self.decision_log)

        # FI vs BI frequency
        plt.figure()
        df["decision"].value_counts(normalize=True).plot(kind="bar")
        plt.title("FI vs BI usage")
        plt.ylabel("Frequency")
        plt.tight_layout()

        path = self._plot_path("fi_vs_bi_usage.png")
        plt.savefig(path, dpi=150)
        if show:
            plt.show()
        plt.close()

        # Decision vs stagnation
        self.save_decisions_csv()

        print(f"[ML] Decision plots saved to: {self.output_dir}")
