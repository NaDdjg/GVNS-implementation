# GVNS with Reinforcement Learning Integration

## Project Overview

This project investigates the integration of **Reinforcement Learning (RL)** into a
**General Variable Neighborhood Search (GVNS)** framework for solving routing-type
optimization problems.

The main objective is to:

1. Establish strong **baseline GVNS behaviors** using different neighborhood orders
2. Analyze the impact of **first vs best improvement strategies**
3. Introduce **RL-based neighborhood selection**
4. Compare classical GVNS against **learning-enhanced GVNS**

---

## Project Structure

```text
gvns/
├── Dockerfile                    # Docker image definition
├── requirements.txt              # Python dependencies
├── gvns/
│   ├── gvns.py                   # Core GVNS algorithm
│   ├── local_search.py           # BVND implementation
│   ├── operators.py              # Neighborhood operators and shaking
│   ├── instance.py               # Instance definition
│   ├── solution.py               # Solution object definition and evaluation
│   ├── initializer.py            # Random initializer for first solution
│   ├── benchmark.py              # Benchmark runner
│   ├── logger.py                 # Logging utilities
│   ├── data_generation/
│   │   ├── ls_logger.py          # Local search logger
│   │   └── offline_labeling.py   # Offline labeling for data generation
│   ├── ml/
│   │   ├── strategy_policy.py    # ML-based strategy policy
│   │   ├── models/               # ML models directory
│   │   ├── notebooks/            # Jupyter notebooks for ML analysis
│   │   └── training/             # ML model training scripts
│   ├── rl/
│   │   ├── rl_env.py             # RL environment definition
│   │   └── agents/
│   │       ├── base_agent.py     # Base agent interface
│   │       ├── dqn_agent.py      # Deep Q-learning agent
│   │       ├── ppo_agent.py      # PPO agent
│   │       └── qlearning_agent.py # Tabular Q-learning agent
│   ├── search_state/
│   │   └── search_state.py       # Search state management
│   └── utils/
│       └── utils.py              # Utility functions
├── instances/
│   ├── 30/                       # Instances with 30 customers
│   ├── 100/                      # Instances with 100 customers
│   └── 200/                      # Instances with 200 customers
├── results/                      # All experimental outputs
├── scripts/                      # Scripts to manage summaries
├── run_benchmark.py              # Main experiment entry point
└── README.md
```

## Experiment design

### Classical GVNS experiments

Before introducing RL, some classical GVNS configurations are evaluated:

- Neighborhood orders:
  - default
  - reverse
  - random
  - suggested

- Improvement strategies:
  - First improvement
  - Best improvement

### Reinforcement Learning experiments

- RL replaces the fixed neighborhood order
- A Q-learning agent dynamically selects neighborhoods
- Neighborhood order arguments are ignored in RL mode
- The same GVNS core is preserved for fair comparison

## Command-line interface

The project is controlled via `run_benchmark.py`, with these available arguments:

| Argument          | Type   | Required | Default        | Description                                                                          |
| ----------------- | ------ | -------- | -------------- | ------------------------------------------------------------------------------------ |
| `--mode`          | string | Yes      | `best`         | Execution mode: `first`, `best`, or `rl_fi` or `rl_bi` or `ml_decision`              |
| `--neigh_order`   | string | No       | `default`      | Neighborhood order (`default`, `reverse`, `random`, `suggested`). Ignored in RL mode |
| `--tmax`          | int    | No       | `500`          | Maximum execution time per instance (in seconds)                                     |
| `--instances_dir` | string | No       | `instances/30` | Path to instance folder (`instances/30`, `instances/100`, `instances/200`)           |
| `--results_dir`   | string | No       | auto-generated | Optional custom directory for saving results                                         |

## Docker-based execution

### 1.Build docker image

in the root folder where `Dockerfile` is, run:

```bash

docker build -t gvns-benchmark .

```

### 2. Execution commands for different experiments

All important folders are mounted so that code can be modified without rebuilding.

Results persist on the host, and experiments are fully reproducible
Mounted folders:

- gvns/
- run_benchmark.py
- instances/
- results/

Docker won't copy files,it will share them with the container, since its an experiment production step, it's better to not rebuild each time.

Here are examples off different execution commands:

#### 2.1. First improvement GVNS (Default order, 30 instances)

```bash
docker run --rm \
  -v $(pwd)/gvns:/app/gvns \
  -v $(pwd)/run_benchmark.py:/app/run_benchmark.py \
  -v $(pwd)/instances:/app/instances \
  -v $(pwd)/result_summaries:/app/result_summaries \
  -v $(pwd)/results:/app/results \
  gvns-benchmark \
  python run_benchmark.py \
    --mode first \
    --neigh_order default \
    --tmax 500 \
    --instances_dir instances/Liu_ALNS/30 \
    --results_dir results/without_RL/predefined_order/First
```

#### 2.2. Best improvement GVNS (Random order, 100 instances)

```bash
docker run --rm \
  -v $(pwd)/gvns:/app/gvns \
  -v $(pwd)/run_benchmark.py:/app/run_benchmark.py \
  -v $(pwd)/instances:/app/instances \
  -v $(pwd)/result_summaries:/app/result_summaries \
  -v $(pwd)/results:/app/results \
  gvns-benchmark \
  python run_benchmark.py \
    --mode best \
    --neigh_order random \
    --tmax 1000 \
    --instances_dir instances/Liu_ALNS/100 \
    --results_dir results/without_RL/random_order/Best
```

#### 2.3. Best improvement GVNS (Suggested order, 200 instances)

A suggested order run must be executed after a default-order run, because it relies on neighborhood-order files generated during the default execution.

```bash
docker run --rm \
  -v $(pwd)/gvns:/app/gvns \
  -v $(pwd)/run_benchmark.py:/app/run_benchmark.py \
  -v $(pwd)/instances:/app/instances \
  -v $(pwd)/result_summaries:/app/result_summaries \
  -v $(pwd)/results:/app/results \
  gvns-benchmark \
  python run_benchmark.py \
    --mode best \
    --neigh_order suggested \
    --tmax 2000 \
    --instances_dir instances/Liu_ALNS/200 \
    --results_dir results/without_RL/suggested_order/Best
```

#### 2.4. RL-based GVNS with best improvement (Q-learning, 30 instances)

```bash
docker run --rm \
  -v $(pwd)/gvns:/app/gvns \
  -v $(pwd)/run_benchmark.py:/app/run_benchmark.py \
  -v $(pwd)/instances:/app/instances \
  -v $(pwd)/result_summaries:/app/result_summaries \
  -v $(pwd)/results:/app/results \
  gvns-benchmark \
  python run_benchmark.py \
    --mode rl_qlearning_bi \
    --tmax 500 \
    --instances_dir instances/Liu_ALNS/30 \
    --results_dir results/with_RL/qlearning/Best
```

#### 2.5. RL-based GVNS with first improvement (Q-learning, 100 instances)

```bash
docker run --rm \
  -v $(pwd)/gvns:/app/gvns \
  -v $(pwd)/run_benchmark.py:/app/run_benchmark.py \
  -v $(pwd)/instances:/app/instances \
  -v $(pwd)/result_summaries:/app/result_summaries \
  -v $(pwd)/results:/app/results \
  gvns-benchmark \
  python run_benchmark.py \
    --mode rl_qlearning_fi \
    --tmax 1000 \
    --instances_dir instances/Liu_ALNS/100 \
    --results_dir results/with_RL/qlearning/First
```

#### 2.6. ML-decision (hybrid FI-BI) GVNS (default order, 30 instances)

```bash
docker run --rm \
  -v $(pwd)/gvns:/app/gvns \
  -v $(pwd)/run_benchmark.py:/app/run_benchmark.py \
  -v $(pwd)/instances:/app/instances \
  -v $(pwd)/result_summaries:/app/result_summaries \
  -v $(pwd)/results:/app/results \
  gvns-benchmark \
  python run_benchmark.py \
    --mode ml_decision \
    --neigh_order default \
    --tmax 500 \
    --instances_dir instances/Liu_ALNS/30 \
    --results_dir results/without_RL/predefined_order/ML_decision
```

#### 2.7. First improvement GVNS on Bredstrom benchmark (Default order, 20 instances)

```bash
docker run --rm \
  -v $(pwd)/gvns:/app/gvns \
  -v $(pwd)/run_benchmark.py:/app/run_benchmark.py \
  -v $(pwd)/instances:/app/instances \
  -v $(pwd)/result_summaries:/app/result_summaries \
  -v $(pwd)/results:/app/results \
  gvns-benchmark \
  python run_benchmark.py \
    --mode first \
    --neigh_order default \
    --tmax 300 \
    --instances_dir instances/Bredstrom/case20 \
    --results_dir results/without_RL/predefined_order/First
```

#### 2.8. Best improvement GVNS on Bredstrom benchmark (Random order, 50 instances)

```bash
docker run --rm \
  -v $(pwd)/gvns:/app/gvns \
  -v $(pwd)/run_benchmark.py:/app/run_benchmark.py \
  -v $(pwd)/instances:/app/instances \
  -v $(pwd)/result_summaries:/app/result_summaries \
  -v $(pwd)/results:/app/results \
  gvns-benchmark \
  python run_benchmark.py \
    --mode best \
    --neigh_order random \
    --tmax 300 \
    --instances_dir instances/Bredstrom/case50 \
    --results_dir results/without_RL/random_order/Best
```

#### 2.9. First improvement GVNS on the new synchronized benchmark (Default order, 200 instances)

```bash
docker run --rm \
  -v $(pwd)/gvns:/app/gvns \
  -v $(pwd)/run_benchmark.py:/app/run_benchmark.py \
  -v $(pwd)/instances:/app/instances \
  -v $(pwd)/result_summaries:/app/result_summaries \
  -v $(pwd)/results:/app/results \
  gvns-benchmark \
  python run_benchmark.py \
    --mode first \
    --neigh_order default \
    --tmax 15000 \
    --instances_dir instances/New_synch/200 \
    --results_dir results/new_benchmark/without_RL/predefined_order/First
```

#### 2.10. ML-decision (hybrid FI-BI) on the new synchronized benchmark (Default order, 200 instances)

```bash
docker run --rm \
  -v $(pwd)/gvns:/app/gvns \
  -v $(pwd)/run_benchmark.py:/app/run_benchmark.py \
  -v $(pwd)/instances:/app/instances \
  -v $(pwd)/result_summaries:/app/result_summaries \
  -v $(pwd)/results:/app/results \
  gvns-benchmark \
  python run_benchmark.py \
    --mode ml_decision \
    --neigh_order default \
    --tmax 15000 \
    --instances_dir instances/New_synch/200 \
    --results_dir results/new_benchmark/without_RL/predefined_order/ML_decision
```

#### 2.11. First improvement GVNS on the new synchronized benchmark (Q-learning, 200 instances)

```bash
docker run --rm \
  -v $(pwd)/gvns:/app/gvns \
  -v $(pwd)/run_benchmark.py:/app/run_benchmark.py \
  -v $(pwd)/instances:/app/instances \
  -v $(pwd)/result_summaries:/app/result_summaries \
  -v $(pwd)/results:/app/results \
  gvns-benchmark \
  python run_benchmark.py \
    --mode rl_qlearning_fi \
    --tmax 15000 \
    --instances_dir instances/New_synch/200 \
    --results_dir results/new_benchmark/with_RL/qlearning/First
```
