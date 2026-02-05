import os
import time
import random
import numpy as np

# GVNS main module
import gvns.operators as ops
import gvns.local_search as ls
from gvns.solution import Solution
from gvns.initializer import random_insertion

# loggers
from gvns.data_generation.ls_logger import LSDecisionLogger
from gvns.search_state.search_state import SearchState
from gvns.logger import GVNSLogger
from gvns.utils import save_neighborhood_usage, save_neighborhood_improvement_log, save_suggested_order

# RL/ ML
from gvns.rl.rl_env import BVNDEnv
from gvns.ml.strategy_policy import MLStrategyPolicy

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

def run_gvns(instance, params=None, agent=None, verbose=True, trace_list=None, first_improvement = True, instance_name = "unknown", qtables_dir=None):
    # ============================================================
    # load parameters
    # ============================================================
    if params is None:
        cfg = DEFAULT_PARAMS.copy()
    else:
        cfg = DEFAULT_PARAMS.copy()
        cfg.update(params)

    Kmax = cfg["Kmax"]

    base_seed = cfg["seed"]
    # derive deterministic instance-specific seed
    inst_seed = abs(hash(instance_name)) % (2**32)

    final_seed = base_seed + inst_seed

    random.seed(final_seed)
    np.random.seed(final_seed)

    # ============================================================
    # logger
    # ============================================================
    log_path = cfg.get("log_path", "gvns_trace.jsonl")
    logger = GVNSLogger(path=log_path, console=True)
    step_counter = 0

    # ============================================================
    # initial solution
    # ============================================================
    current = random_insertion(instance, seed=cfg["seed"])
    current.w1 = cfg["w1"]
    current.w2 = cfg["w2"]
    current.alpha = cfg["alpha"]
    current.M = cfg["M"]
    current.eval()

    best_aug = current.copy()
    best_aug.eval()

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
    # main GVNS loop
    # ============================================================
    cont = 0
    iteration = 0
    start = time.time()
    time_limit = cfg["Tmax"]
    delta = cfg["delta"]
    Dp = cfg["Dp"]
    order_mode=cfg.get("neigh_order", "default")

    usage_logger = {
        "counter": [0] * len(ls.NEIGHBORHOOD_FUNCS), # usage counts per neighborhood 
        "events": []
    }

    # search state for LS
    search_state = SearchState(
        instance=instance,
        time_limit=time_limit
    )
    search_state.reset()

    # LS decision logger
    log_data = cfg.get("log_ls_data", False)

    if log_data:
        mode = "FI" if first_improvement else "BI"
        output_dir = os.path.join("gvns", "ls_data", mode)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{instance_name}.csv")
        
        ls_logger = LSDecisionLogger(
            output_path=output_path,
            instance_name=instance_name,
            mode=mode
        )

    # ML strategy policy
    use_ml = cfg.get("use_ml", False)
    verbose_ml = cfg.get("verbose_ml", False)

    ml_policy = None
    if use_ml:
        ml_policy = MLStrategyPolicy(
            clf_path="gvns/ml/models/rf_strategy_classifier.pkl",
            reg_path="gvns/ml/models/rf_delta_regressor.pkl",
            features_path="gvns/ml/models/rf_features.json",
        )

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
        # 1. progressive shaking
        # ============================================================
        k = 1 + (iteration % Kmax)
        shaken = ops.random_shake(current, k=k, Dp=Dp)
        shaken.w1 = current.w1
        shaken.w2 = current.w2
        shaken.alpha = current.alpha
        shaken.M = current.M
        shaken.eval()

        # ============================================================
        # 2. BVND (classic local search/RL)
        # ============================================================
        if agent is None:
            #  CLASSIC BVND 
            improved = ls.b_vnd(
                shaken,
                max_no_improve=1,
                first_improvement=first_improvement,
                order_mode=order_mode,
                usage_logger=usage_logger,
                instance_name=instance_name,
                iteration=iteration,                   
                search_state=search_state, 
                ls_logger=ls_logger,
                ml_policy=ml_policy,
                verbose_ml=verbose_ml
            )
        else: 
            #  RL-DRIVEN BVND 
            env = BVNDEnv(
                inst=instance,
                initial_sol=shaken,
                w1=cfg["w1"],
                w2=cfg["w2"],
                M=cfg["M"],
                Tmax=cfg["Tmax"] or 1e9,
            )

            obs = env.reset()
            done = False

            new_sol, action, improved = ls.rl_bvnd_step(
                sol=env.current,
                agent=agent,
                obs=obs,
                neighborhoods=ls.NEIGHBORHOOD_FUNCS,
                first_improvement=first_improvement,
                iteration=iteration, 
                usage_logger=usage_logger
            )

            next_obs, reward, done, info = env.step(
                action=action,
                improved=improved,
                new_sol=new_sol if improved else None,
                elapsed=elapsed
            )

            # Q update
            agent.update(obs, action, reward, next_obs, done)

            obs = next_obs
            improved = env.current

            # logs for qtables
            # ts = time.strftime("%Y%m%d_%H%M%S")
            # save_path = f"{qtables_dir}/qtable_{ts}.pkl"
            # agent.save(save_path)

        improved.eval()

        # ============================================================
        # 3. acceptance criterion (Move-or-Not)
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
            # logger.log("accepted", step_counter := step_counter + 1, current,
            #            note=f"iter_{iteration}")
        else:
            cont += 1
            # logger.log("rejected", step_counter := step_counter + 1, improved,
            #            note=f"iter_{iteration}")

        # ============================================================
        # 4. track best solutions
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
        # 5. penalty Adaptation
        # ============================================================
        current.self_adjust_penalties()

        # ============================================================
        # 6. print iteration summary
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

    # save neighborhood statistics
    # if order_mode != "suggested":
    #     save_neighborhood_usage(
    #         usage_logger["counter"],
    #         instance_name
    #     )
        
    #     save_neighborhood_improvement_log(
    #         usage_logger["events"],
    #         instance_name,
    #         suffix="rl" if agent is not None else "bvnd"
    #     )
        
    #     save_suggested_order(
    #         usage_logger["counter"],
    #         instance_name,
    #         list(enumerate(ls.NEIGHBORHOOD_FUNCS))
    #     )

    if best_feasible is not None:
        best_feasible.eval()
        best_feasible.best_time = best_feasible_time
        return best_feasible

    best_aug.eval()
    best_aug.best_time = None
    logger.close()
    ls_logger.close()
    return best_aug