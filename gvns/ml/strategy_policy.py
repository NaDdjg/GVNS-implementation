import joblib
import json
import numpy as np
from pathlib import Path

class MLStrategyPolicy:
    def __init__(self, clf_path, reg_path, features_path, log_decisions=False):
        self.clf = joblib.load(clf_path)
        self.reg = joblib.load(reg_path)

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

        # head 1: classifier (strategy-agnostic)
        p_bi = self.clf.predict_proba(X)[0, 1]
        p_fi = 1.0 - p_bi

        # head 2: regressor (strategy-conditional)
        X_bi = X.copy()
        X_fi = X.copy()

        X_bi[0, self.strategy_idx] = 1
        X_fi[0, self.strategy_idx] = 0

        gain_bi = self.reg.predict(X_bi)[0]
        gain_fi = self.reg.predict(X_fi)[0]

        # final decision
        score_bi = p_bi * gain_bi
        score_fi = p_fi * gain_fi

        decision = "BI" if score_bi > score_fi else "FI"

        result = {
            "decision": decision,
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
    
    def plot_decisions(self, show=False):  # call it with ml_policy.plot_decisions()
        if not self.decision_log:
            raise RuntimeError("No decisions logged.")

        import pandas as pd
        import matplotlib.pyplot as plt

        df = pd.DataFrame(self.decision_log)

        # FI vs BI frequency
        plt.figure()
        df["decision"].value_counts(normalize=True).plot(kind="bar")
        plt.title("FI vs BI usage")
        plt.ylabel("Frequency")
        plt.tight_layout()

        path = self.output_dir / "fi_vs_bi_usage.png"
        plt.savefig(path, dpi=150)
        if show:
            plt.show()
        plt.close()

        # Decision vs stagnation
        plt.figure()
        df.boxplot(column="stagnation", by="decision")
        plt.title("Decision vs LS stagnation")
        plt.suptitle("")
        plt.ylabel("LS calls since improvement")
        plt.tight_layout()

        path = self.output_dir / "decision_vs_stagnation.png"
        plt.savefig(path, dpi=150)
        if show:
            plt.show()
        plt.close()

        # Decision over time
        plt.figure()
        score_diff = df["score_bi"] - df["score_fi"]

        plt.scatter(
            df["time_elapsed_ratio"],
            score_diff,
            c=(df["decision"] == "BI"),
            alpha=0.6
        )
        plt.axhline(0, linestyle="--")
        plt.xlabel("Time elapsed ratio")
        plt.ylabel("Score(BI) - Score(FI)")
        plt.title("ML decision boundary over time")
        plt.tight_layout()

        path = self.output_dir / "decision_boundary_over_time.png"
        plt.savefig(path, dpi=150)
        if show:
            plt.show()
        plt.close()

        print(f"[ML] Decision plots saved to: {self.output_dir}")
