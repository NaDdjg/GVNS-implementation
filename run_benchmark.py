from gvns.benchmark import benchmark_folder, summarize_by_class
from gvns.agents.qlearning_agent import QTableAgent

def make_agent():
    return QTableAgent(
        n_actions=4,
        alpha=0.1,
        gamma=0.99,
        eps_start=0.2,
        eps_min=0.01,
        eps_decay=1e-4,
        seed=42
    )

df_summary = benchmark_folder(
    folder=r"C:\Users\HP\Documents\GitHub\GVNS-implementation\vrptws",
    runs_per_instance= 1,
    Tmax_rule= lambda n: 5000,
    gvns_kwargs={
        "contmax": 100,
        "p_init": 1,
        "p_max": 10,
        "Dp": 0.3,
        "w1_init": 0.5,
        "w2_init": 0.1,
        "alpha": 0.1,
        "delta": 0.4,
        "M": 1e6,
        "w1_max": None,
        "w2_max": None,
        "update_every": 1,
    },
    out_csv="gvns_per_instance_qlearning_online.csv",
    agent_factory=make_agent
)

print(df_summary.shape)
print(df_summary)

df_class = summarize_by_class(df_summary)
df_class.to_csv("results/gvns_per_class.csv", index=False)
print(df_class)