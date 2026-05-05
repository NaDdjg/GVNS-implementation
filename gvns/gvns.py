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
from gvns.data_generation.ml.ls_logger import LSDecisionLogger
from gvns.data_generation.search_state import SearchState
from gvns.logger import GVNSLogger
from gvns.utils.utils import (
    save_neighborhood_usage,
    save_neighborhood_improvement_log,
    save_suggested_order,
    save_operator_stats
)

# RL/ ML
from gvns.rl.rl_env import BVNDEnv
from gvns.ml.strategy_policy import MLStrategyPolicy

DEFAULT_PARAMS = {
    "Contmax": 100,       # max consecutive non-improving iterations
    "p_max": 12,           # max shaking strength
    "delta": 0.4,         # acceptance threshold factor
    "w1": 0.5,
    "w2": 0.1,
    "alpha": 0.1,
    "M": 1e6,
    "seed": 42,
    "Tmax": 500,
}

def run_gvns(
        instance, 
        results_dir="results",
        params=None, 
        agent=None, 
        ppo_agent=None,
        verbose=True, 
        trace_list=None, 
        first_improvement = True, 
        instance_name = "unknown", 
        qtables_dir=None,
        collector=None, 
        param_controller=None
    ):
    # ============================================================
    # load parameters
    # ============================================================
    cfg = DEFAULT_PARAMS.copy()
    if params :
        cfg.update(params)

    Kmax = cfg["p_max"]
    time_limit = cfg["Tmax"]
    use_controller = cfg.get("use_controller", False)
    
    # ============================================================
    # seed
    # ============================================================
    base_seed = cfg["seed"]
    inst_seed = abs(hash(instance_name)) % (2**32)
    final_seed = base_seed + inst_seed

    random.seed(final_seed)
    np.random.seed(final_seed)

    # ============================================================
    # controller 
    # ============================================================
    if use_controller:
        if param_controller is None:
            from gvns.data_generation.rl.params.controller import ParamController
            param_controller = ParamController()
        else:
            param_controller.reset()

    # ============================================================
    # logger
    # ============================================================
    log_path = cfg.get("log_path", "gvns_trace.jsonl")
    logger = GVNSLogger(path=log_path, console=True)

    # ============================================================
    # initial solution
    # ============================================================
    current = random_insertion(instance, seed=final_seed)
    current.w1 = cfg["w1"]
    current.w2 = cfg["w2"]
    current.alpha = cfg["alpha"] 
    current.M = cfg["M"]
    current.eval()

    best_aug = current.copy()

    best_feasible = None
    best_feasible_f = float("inf")
    best_feasible_time = None

    if current.penS == 0 and current.penT == 0 and current.penH == 0:
        best_feasible = current.copy()
        best_feasible_f = current.obj_value

    if verbose:
        print(f"[GVNS] Init: F={current.Fval:.3f}, f={current.obj_value:.3f}")

    # ============================================================
    # main GVNS loop init
    # ============================================================
    cont = 0
    iteration = 0
    start = time.time()
    
    delta_cfg  = cfg["delta"]
    base_Dp  = cfg.get("Dp", 1)
    order_mode=cfg.get("neigh_order", "default")
    usage_logger = {
        "counter": [0] * len(ls.NEIGHBORHOOD_FUNCS), # usage counts per neighborhood 
        "events": []
    }

    # RL operator statistics
    operator_stats = {
        i: {
            "uses": 0,
            "success": 0,
            "reward_sum": 0.0
        }
        for i in range(len(ls.NEIGHBORHOOD_FUNCS))
    }

    # ============================================================
    # search state
    # ============================================================
    search_state = SearchState(instance, time_limit)
    search_state.reset()

    if use_controller:
        search_state.controller = param_controller

    # ============================================================
    # LS logger
    # ============================================================
    log_data = cfg.get("log_ls_data", False)
    ls_logger = None # will remain None if logging is disabled

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

    # ============================================================
    # PPO logger
    # ============================================================
    ppo_logger = None
    if ppo_agent is not None:
        from gvns.data_generation.rl.ppo_action_logger import PPOActionLogger

        ppo_logger = PPOActionLogger(
            output_path=f"results/methods/Hyperparams_controller/ppo_logs/{instance_name}.json"
        )
        
    # ============================================================
    # ML policy 
    # ============================================================
    use_ml = cfg.get("use_ml", False)
    verbose_ml = cfg.get("verbose_ml", False)
    ml_policy = None

    if use_ml:
        ml_policy = MLStrategyPolicy(
            clf_path="gvns/ml/models/rf_success_classifier.pkl",
            reg_path="gvns/ml/models/rf_delta_regressor.pkl",
            features_path="gvns/ml/models/rf_features.json",
            instance_name=instance_name,
            log_decisions=True,
        )
    
    # ============================================================
    # GVNS Loop
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

        # ========================================================
        # Paparmeters switch
        # ========================================================
        if use_controller:
            params_now = param_controller.get_params()
            delta = params_now["acceptance_delta"]
            Dp = base_Dp * params_now["shaking_scale"]
        else:
            delta = delta_cfg
            Dp = base_Dp

        # ============================================================
        # Shaking
        # ============================================================
        k = 1 + (iteration % Kmax)
        shaken = ops.random_shake(current, k=k, Dp=Dp)
        shaken.w1 = current.w1
        shaken.w2 = current.w2
        shaken.alpha = current.alpha
        shaken.M = current.M
        shaken.eval()

        # ============================================================
        # 2. Local search
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
                results_dir=results_dir,
                iteration=iteration,                   
                search_state=search_state, 
                ls_logger=ls_logger,
                ml_policy=ml_policy,
                verbose_ml=verbose_ml,
                collector=collector,
                param_controller=param_controller if use_controller else None,
                ppo_agent=ppo_agent,
                ppo_logger=ppo_logger
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
                reward_mode=cfg.get("reward_mode", "qlearning"),
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
                usage_logger=usage_logger,
                ml_policy=ml_policy,
                search_state=search_state,
                verbose_ml=verbose_ml
            )

            next_obs, reward, done, info = env.step(
                action=action,
                improved=improved,
                new_sol=new_sol if improved else None,
                elapsed=elapsed
            )

            # update operator statistics
            operator_stats[action]["uses"] += 1
            operator_stats[action]["reward_sum"] += reward

            if improved:
                operator_stats[action]["success"] += 1

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
        # Acceptance criterion (Move-or-Not)
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
        # Track best solutions
        # ============================================================
        if improved.Fval < best_aug.Fval:
            best_aug = improved.copy()

        if improved.penS == 0 and improved.penT == 0 and improved.penH == 0:
            if improved.obj_value < best_feasible_f:
                best_feasible_f = improved.obj_value
                best_feasible = improved.copy()
                best_feasible_time = elapsed
                print(f"[GVNS][iter {iteration}] New best feasible: "
                    f"f={best_feasible_f:.3f}")
                
        # ============================================================
        # Penalty Adaptation
        # ============================================================
        current.self_adjust_penalties()

        # ============================================================
        # Trace logging
        # ============================================================
        print(f"[GVNS][iter {iteration}] "
              f"F={improved.Fval:.3f} f={improved.obj_value:.3f} "
              f"| PenS={improved.penS} PenT={improved.penT} PenH={improved.penH} "
              f"| accepted={accept}, cont={cont}, k={k}, elapsed={elapsed:.2f}s")

        if trace_list is not None:
            trace_list.append([
                iteration, current.Fval, current.obj_value,
                current.penT, current.penH, current.penS, elapsed
            ])

    # ============================================================
    # Save collector
    # ============================================================
    if collector is not None:
        collector.save()

    # ============================================================
    # Statistics and logs
    # ============================================================
    # operator statistics
    for op in operator_stats:
        uses = operator_stats[op]["uses"]
        success = operator_stats[op]["success"]
        reward_sum = operator_stats[op]["reward_sum"]

        operator_stats[op]["success_rate"] = success / uses if uses > 0 else 0.0
        operator_stats[op]["avg_reward"] = reward_sum / uses if uses > 0 else 0.0

    # neighborhood statistics
    if order_mode != "suggested":
        save_neighborhood_usage(
            usage_logger["counter"],
            instance_name,
            results_dir=results_dir
        )
        
        save_neighborhood_improvement_log(
            usage_logger["events"],
            instance_name,
            suffix="rl" if agent is not None else "bvnd",
            results_dir=results_dir
        )
        
        save_suggested_order(
            usage_logger["counter"],
            instance_name,
            list(enumerate(ls.NEIGHBORHOOD_FUNCS)),
            results_dir=results_dir
        )

    # ML policy logs
    if ml_policy is not None:
        ml_policy.plot_decisions(show=False)

    # RL operator stats
    if agent is not None and cfg.get("reward_mode", "qlearning") == "dqn":
        save_operator_stats(
            operator_stats,
            instance_name,
            suffix="dqn",
            results_dir=results_dir
        )

    # PPO action logs
    if ppo_logger is not None:
        ppo_logger.save()

    # ============================================================
    # Cleanup
    # ============================================================
    logger.close()
    if ls_logger is not None:
        ls_logger.close()   

    # ============================================================
    # Return
    # ============================================================
    if best_feasible is not None:
        best_feasible.best_time = best_feasible_time
        return best_feasible

    best_aug.best_time = None
    return best_aug