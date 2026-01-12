import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from gvns.agents.qlearning_agent import QTableAgent

# ============================================================
# USER CONFIG
# ============================================================
QTABLES_DIR = "results\with RL\qlearning\Best_alpha01_gamma095\qtables"     
RESULTS_DIR = "result_summaries\qtable_results\Best_alpha01_gamma095"
N_ACTIONS = 8                                

os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# HELPERS
# ============================================================
def load_qtable(path):
    """Load Q-table from pickle (supports dict or {'Q': Q})"""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["Q"] if isinstance(data, dict) and "Q" in data else data


def average_q_per_action(Q, n_actions):
    """Average Q-value per action over all states"""
    if len(Q) == 0:
        return np.zeros(n_actions)
    Qmat = np.array(list(Q.values()))
    return Qmat.mean(axis=0)


def print_qtable(Q, max_states=20):
    """Pretty-print Q-table (limit states to avoid spam)"""
    print("===== Q-TABLE (sample) =====")
    for i, (state, qvals) in enumerate(Q.items()):
        print(f"State {state} -> Q {qvals}")
        if i + 1 >= max_states:
            print("... (truncated)")
            break


# ============================================================
# LOAD ALL Q-TABLES
# ============================================================
qtable_files = sorted(
    f for f in os.listdir(QTABLES_DIR) if f.endswith(".pkl")
)

print(f"Found {len(qtable_files)} Q-tables")

agent = QTableAgent(
    n_actions=N_ACTIONS,
    alpha=0.2,
    gamma=0.95,
    eps_start=0.1,
    eps_min=0.01,
    eps_decay=5e-4,
    use_ucb=True,
    ucb_c=1.0,
)

avg_q_history = []

for fname in qtable_files:
    Q = load_qtable(os.path.join(QTABLES_DIR, fname))
    avg_q_history.append(average_q_per_action(Q, N_ACTIONS))

avg_q_history = np.array(avg_q_history)

# ============================================================
# PLOT LEARNING EVOLUTION
# ============================================================
plt.figure()
for a in range(N_ACTIONS):
    plt.plot(avg_q_history[:, a], label=f"Action {a}")

plt.xlabel("Episode")
plt.ylabel("Average Q-value")
plt.title("Q-learning evolution per action")
plt.legend()

plot_path = os.path.join(RESULTS_DIR, "avg_q_per_action.png")
plt.savefig(plot_path)
plt.close()

print("Saved plot to:", plot_path)

# ============================================================
# LAST Q-TABLE ANALYSIS
# ============================================================
if qtable_files:
    last_path = os.path.join(QTABLES_DIR, qtable_files[-1])
    last_Q = load_qtable(last_path)

    print("\nLast Q-table loaded from:", last_path)
    print_qtable(last_Q)

    txt_path = os.path.join(RESULTS_DIR, "last_qtable.txt")
    with open(txt_path, "w") as f:
        for state, qvals in last_Q.items():
            f.write(f"{state} -> {qvals}\n")

    print("Saved readable Q-table to:", txt_path)

print("\nAll results saved in:", RESULTS_DIR)

# ============================================================
# EXPORT LAST Q-TABLE TO CSV (READABLE)
# ============================================================
import csv

def softmax(qvals, tau=1.0):
    q = np.array(qvals, dtype=float) / tau
    q = q - np.max(q)
    exp_q = np.exp(q)
    return exp_q / exp_q.sum() if exp_q.sum() > 0 else np.zeros_like(q)

csv_path = os.path.join(RESULTS_DIR, "last_qtable_readable.csv")

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)

    # Header with semantic meaning
    writer.writerow([
        "penS_flag",
        "penT_flag",
        "penH_flag",
        "stagnation_bucket",
        "effectiveness_bucket",
        "size_bucket",
        "action",
        "Q_value",
        "best_action",
        "policy_probability"
    ])

    for state, qvals in last_Q.items():
        best_action = int(np.argmax(qvals))
        probs = softmax(qvals)

        for a, q in enumerate(qvals):
            writer.writerow([
                state[0], state[1], state[2],
                state[3], state[4], state[5],
                a,
                float(q),
                best_action,
                float(probs[a])
            ])

print("Saved CSV Q-table to:", csv_path)
