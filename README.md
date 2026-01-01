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
├── Dockerfile                  # Docker image definition
├── requirements.txt            # Python dependencies
├── gvns/
│   ├── gvns.py                 # Core GVNS algorithm
│   ├── local_search.py         # BVND implementation
│   ├── operators.py            # Neighborhood operatorsand shaking
│   ├── instance.py             # Instance definition
│   ├── solution.py             #Solution object definition and evaluation step
│   ├── initializer.py          # Random initiliazer for first solution
│   ├── benchmark.py            # Benchmark runner
│   ├── agents/
│   │   ├── base_agent.py       # Interface to define the main struture of all agents
│   │   ├── dqn_agent.py        # Deep Q-learning agent
│   │   ├── ppo_agent.py        # PPO agent
│   │   └── qlearning_agent.py  # Tabular Q-learning agent
│   └── utils.py                # Utilities (saving functions for local sear, rl helpers)
├── instances/
│   ├── 30/                     # Instances with 30 customers
│   ├── 100/                    # Instances with 100 customers
│   └── 200/                    # Instances with 200 customers
├── results/                    # All experimental outputs
├── scripts/                    #Scripts to manage summaries
├── run_benchmark.py            # Main experiment entry point
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
| `--mode`          | string | Yes      | `best`         | Execution mode: `first`, `best`, or `rl`                                             |
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
  -v $(pwd)/results:/app/results \
  gvns-benchmark \
  python run_benchmark.py \
    --mode first \
    --neigh_order default \
    --tmax 500 \
    --instances_dir instances/30 \
    --results_dir results/without_RL/predefined_order/First
```

#### 2.2. Best improvement GVNS (Random order, 100 instances)

```bash
docker run --rm \
  -v $(pwd)/gvns:/app/gvns \
  -v $(pwd)/run_benchmark.py:/app/run_benchmark.py \
  -v $(pwd)/instances:/app/instances \
  -v $(pwd)/results:/app/results \
  gvns-benchmark \
  python run_benchmark.py \
    --mode best \
    --neigh_order random \
    --tmax 1000 \
    --instances_dir instances/100 \
    --results_dir results/without_RL/random_order/Best
```

#### 2.3. Best improvement GVNS (Suggested order, 200 instances)

A suggested order run must be executed after a default-order run, because it relies on neighborhood-order files generated during the default execution.

```bash
docker run --rm \
  -v $(pwd)/gvns:/app/gvns \
  -v $(pwd)/run_benchmark.py:/app/run_benchmark.py \
  -v $(pwd)/instances:/app/instances \
  -v $(pwd)/results:/app/results \
  gvns-benchmark \
  python run_benchmark.py \
    --mode best \
    --neigh_order suggested \
    --tmax 2000 \
    --instances_dir instances/200 \
    --results_dir results/without_RL/suggested_order/Best
```

#### 2.4. RL-based GVNS (Q-learning, 30 instances)

```bash
docker run --rm \
  -v $(pwd)/gvns:/app/gvns \
  -v $(pwd)/run_benchmark.py:/app/run_benchmark.py \
  -v $(pwd)/instances:/app/instances \
  -v $(pwd)/results:/app/results \
  gvns-benchmark \
  python run_benchmark.py \
    --mode rl \
    --tmax 500 \
    --instances_dir instances/30 \
    --results_dir results/with_RL/qlearning/Best
```

#### 2.5. RL-based GVNS (Q-learning, 100 instances)

```bash
docker run --rm \
  -v $(pwd)/gvns:/app/gvns \
  -v $(pwd)/run_benchmark.py:/app/run_benchmark.py \
  -v $(pwd)/instances:/app/instances \
  -v $(pwd)/results:/app/results \
  gvns-benchmark \
  python run_benchmark.py \
    --mode rl \
    --tmax 1000 \
    --instances_dir instances/100 \
    --results_dir results/with_RL/qlearning/Best
```
