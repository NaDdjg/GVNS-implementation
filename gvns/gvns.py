import time
import random

from gvns.solution import Solution
from gvns.initializer import random_insertion
import gvns.operators as ops
import gvns.local_search as ls
from gvns.logger import GVNSLogger

from gvns.rl_env import BVNDEnv

DEFAULT_PARAMS = {
    "Contmax": 100,       # max consecutive non-improving iterations
    "Kmax": 12,           # max shaking strength
    "delta": 0.4,         # acceptance threshold factor
    "w1": 0.5,
    "w2": 0.1,
    "alpha": 0.1,
    "M": 1e6,
    "seed": 0,
    "Tmax": None,
}

def run_gvns(instance, params=None, agent=None, verbose=True, trace_list=None, first_improvement = True, instance_name = "unknown"):
    # ============================================================
    # Load parameters
    # ============================================================
    if params is None:
        cfg = DEFAULT_PARAMS.copy()
    else:
        cfg = DEFAULT_PARAMS.copy()
        cfg.update(params)

    random.seed(cfg["seed"])
    Kmax = cfg["Kmax"]
    depot_id = min(instance.nodes.keys())

    # ============================================================
    # Logger
    # ============================================================
    log_path = cfg.get("log_path", "gvns_trace.jsonl")
    logger = GVNSLogger(path=log_path, console=True)
    step_counter = 0

    # ============================================================
    # Initial solution
    # ============================================================
    current = random_insertion(instance, seed=cfg["seed"])
    current.w1 = cfg["w1"]
    current.w2 = cfg["w2"]
    current.alpha = cfg["alpha"]
    current.M = cfg["M"]
    current.eval()

    best_aug = current.copy()
    best_aug.eval()
    print(best_aug.Fval, best_aug.penS, best_aug.penT, best_aug.penH, best_aug.travel_time)
    for i, r in enumerate(best_aug.tours):
        print(i, len(r))

    best_feasible = None
    best_feasible_f = float("inf")
    best_feasible_time = None

    if current.penS == 0 and current.penT == 0 and current.penH == 0:
        best_feasible = current.copy()
        best_feasible_f = current.travel_time

    if verbose:
        print(f"[GVNS] Init: F={current.Fval:.3f}, f={current.travel_time:.3f}, "
              f"PenS={current.penS}, PenT={current.penT}, PenH={current.penH}")

    # ============================================================
    # Main GVNS loop
    # ============================================================
    cont = 0
    iteration = 0
    start = time.time()
    time_limit = cfg["Tmax"]
    delta = cfg["delta"]

    # ============================================================
    # Helper for BVND budget
    # ============================================================
    def compute_bvnd_time():
        if time_limit is None:
            return None
        elapsed = time.time() - start
        remaining = time_limit - elapsed
        return max(0.01, 0.7 * remaining)

    def compute_bvnd_time(n, sync_pairs, m, Tmax=None,
                      base=5.0,
                      per_customer=2.2,
                      per_sync=15.0,
                      per_vehicle=0.6,
                      min_time=30.0,
                      max_time=3600.0,
                      frac_of_Tmax=0.85):
        # aggressive scaling for large instances
        t = base + per_customer * n + per_sync * sync_pairs + per_vehicle * m
        if Tmax is not None:
            t = min(t, frac_of_Tmax * Tmax)
        t = max(min_time, min(t, max_time))
        return float(t)

    # ============================================================
    # GVNS iterating
    # ============================================================
    while True:
        iteration += 1
        elapsed = time.time() - start

        if time_limit and elapsed >= time_limit:
            print("[GVNS] Time limit reached.")
            break
        if cont >= cfg["Contmax"]:
            print("[GVNS] Non-improvement limit reached.")
            break

        # ============================================================
        # 1. Progressive Shaking
        # ============================================================
        k = 1 + (iteration % Kmax)
        shaken = ops.random_shake(current, k=k)
        shaken.w1 = current.w1
        shaken.w2 = current.w2
        shaken.alpha = current.alpha
        shaken.M = current.M
        shaken.eval()

        # ============================================================
        # 2. BVND (classic local search/RL)
        # ============================================================
        n_customers = len([nid for nid in instance.nodes if nid != depot_id])
        sync_pairs = len(getattr(instance, "sync_customers", []))
        m = instance.m
        bvnd_time = compute_bvnd_time(n=n_customers, sync_pairs=sync_pairs, m=m, Tmax=time_limit)

        if agent is None:
            # ---------- CLASSIC BVND ----------
            improved = ls.b_vnd(
                shaken,
                max_no_improve=1,
                first_improvement=first_improvement,
                save_usage=True,
                use_suggested_order=False,
                instance_name=instance_name
            )
        else: 
            # ---------- RL-DRIVEN BVND ----------
            env = BVNDEnv(
                inst=instance,
                initial_sol=shaken,
                w1=cfg["w1"],
                w2=cfg["w2"],
                M=cfg["M"],
                Tmax=cfg["Tmax"] or 1e9,
                max_steps=20
            )

            obs = env.reset()
            done = False
            while not done:
                action = agent.select_action(obs)
                next_obs, reward, done, info = env.step(action)
                agent.update(obs, action, reward, next_obs, done)
                obs = next_obs

            improved = env.current

        improved.eval()

        # ============================================================
        # 3. Acceptance Criterion (Move-or-Not)
        # ============================================================
        if (improved.penS < current.penS or
            improved.penT < current.penT or
            improved.penH < current.penH):
            accept = True
        else:
            accept = improved.Fval <= (1.0 + delta) * current.Fval

        if accept:
            current = improved
            cont = 0
            logger.log("accepted", step_counter := step_counter + 1, current,
                       note=f"iter_{iteration}")
        else:
            cont += 1
            logger.log("rejected", step_counter := step_counter + 1, improved,
                       note=f"iter_{iteration}")

        # ============================================================
        # 4. Track best solutions
        # ============================================================
        if improved.Fval < best_aug.Fval:
            best_aug = improved.copy()

        if improved.penS == 0 and improved.penT == 0 and improved.penH == 0:
            if improved.travel_time < best_feasible_f:
                best_feasible_f = improved.travel_time
                best_feasible = improved.copy()
                best_feasible_time = elapsed
                print(f"[GVNS][iter {iteration}] New best feasible: "
                      f"f={best_feasible_f:.3f}")

        # ============================================================
        # 5. Penalty Adaptation
        # ============================================================
        current.self_adjust_penalties()

        # ============================================================
        # 6. Print iteration summary
        # ============================================================
        print(f"[GVNS][iter {iteration}] "
              f"F={improved.Fval:.3f} f={improved.travel_time:.3f} "
              f"| PenS={improved.penS} PenT={improved.penT} PenH={improved.penH} "
              f"| accepted={accept}, cont={cont}, k={k}, elapsed={elapsed:.2f}s")

        if trace_list is not None:
            trace_list.append([
                iteration, current.Fval, current.travel_time,
                current.penT, current.penH, current.penS, elapsed
            ])

    if best_feasible is not None:
        best_feasible.eval()
        best_feasible.best_time = best_feasible_time
        return best_feasible

    best_aug.eval()
    best_aug.best_time = None
    logger.close()
    return best_aug