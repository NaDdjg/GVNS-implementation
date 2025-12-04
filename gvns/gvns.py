import time
import random

from gvns.solution import Solution
from gvns.initializer import random_insertion
import gvns.operators as ops
import gvns.local_search as ls
from gvns.logger import GVNSLogger

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

def run_gvns(instance, params=None, verbose=True, trace_list=None):
    # ---------------------------------------------------------
    # Load parameters
    # ---------------------------------------------------------
    if params is None:
        cfg = DEFAULT_PARAMS.copy()
    else:
        cfg = DEFAULT_PARAMS.copy()
        cfg.update(params)

    random.seed(cfg["seed"])
    Kmax = cfg["Kmax"]
    depot_id = min(instance.nodes.keys())

    # ---------------------------------------------------------
    # Logger
    # ---------------------------------------------------------
    log_path = cfg.get("log_path", "gvns_trace.jsonl")
    logger = GVNSLogger(path=log_path, console=True)
    step_counter = 0

    # ---------------------------------------------------------
    # Initial solution
    # ---------------------------------------------------------
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
    if current.penS == 0 and current.penT == 0 and current.penH == 0:
        best_feasible = current.copy()
        best_feasible_f = current.travel_time

    if verbose:
        print(f"[GVNS] Init: F={current.Fval:.3f}, f={current.travel_time:.3f}, "
              f"PenS={current.penS}, PenT={current.penT}, PenH={current.penH}")

    # ---------------------------------------------------------
    # Main GVNS loop
    # ---------------------------------------------------------
    cont = 0
    iteration = 0
    start = time.time()
    time_limit = cfg["Tmax"]
    delta = cfg["delta"]

    # ---------------------------------------------------------
    # Helper for BVND budget
    # ---------------------------------------------------------
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

    # def compute_bvnd_time(n, sync_pairs, m,
    #                     base=2.5,
    #                     per_customer=0.8,
    #                     per_sync=10.0,
    #                     per_vehicle=0.5,
    #                     min_time=20.0,
    #                     max_time=500.0,
    #                     Tmax=None,
    #                     frac_of_Tmax=0.85):

    #     # aggressively scale with n and sync
    #     t = (base +
    #         per_customer * n +
    #         per_sync * sync_pairs +
    #         per_vehicle * m)

    #     # respect global Tmax
    #     if Tmax is not None:
    #         t = min(t, frac_of_Tmax * Tmax)

    #     # clamp into allowed bounds
    #     t = max(min_time, min(t, max_time))
    #     return float(t)

    # ---------------------------------------------------------
    # GVNS iterating
    # ---------------------------------------------------------
    while True:
        iteration += 1

        elapsed = time.time() - start
        if time_limit and elapsed >= time_limit:
            print("[GVNS] Time limit reached.")
            break
        if cont >= cfg["Contmax"]:
            print("[GVNS] Non-improvement limit reached.")
            break

        # ---------------------------------------------
        # 1. Progressive Shaking: k = 1..Kmax
        # ---------------------------------------------
        k = 1 + (iteration % Kmax)
        shaken = ops.random_shake(current, k=k)
        shaken.w1 = current.w1
        shaken.w2 = current.w2
        shaken.alpha = current.alpha
        shaken.M = current.M
        shaken.eval()

        # logger.log("shake", step_counter := step_counter + 1, shaken,
        #            note=f"shake_k={k}")

        # ---------------------------------------------
        # 2. Apply BVND (local search)
        # ---------------------------------------------
        n_customers = len([nid for nid in instance.nodes if nid != depot_id])
        sync_pairs = len(getattr(instance, "sync_customers", []))
        m = instance.m
        bvnd_time = compute_bvnd_time(n=n_customers, sync_pairs=sync_pairs, m=m, Tmax=time_limit)
        # bvnd_time = compute_bvnd_time()

        improved = ls.b_vnd(
            shaken,
            time_limit=bvnd_time,
            max_no_improve=1
        )
        improved.eval()
        # logger.log("after_ls", step_counter := step_counter + 1, improved, note="after_LS")

        # ---------------------------------------------
        # 3. Acceptance Criterion (Move-or-Not)
        # ---------------------------------------------
        if improved.penS < current.penS:
            accept = True
        elif improved.penT < current.penT:
            accept = True
        elif improved.penH < current.penH:
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

        # ---------------------------------------------
        # 4. Track best solutions
        # ---------------------------------------------
        if improved.Fval < best_aug.Fval:
            best_aug = improved.copy()
            print(f"[GVNS][iter {iteration}] New best augmented: "
                  f"F={best_aug.Fval:.3f}, f={best_aug.travel_time:.3f}, "
                  f"PenS={best_aug.penS}")

        if improved.penS == 0 and improved.penT == 0 and improved.penH == 0:
            if improved.travel_time < best_feasible_f:
                best_feasible_f = improved.travel_time
                best_feasible = improved.copy()
                print(f"[GVNS][iter {iteration}] New best feasible: "
                      f"f={best_feasible_f:.3f}")

        # ---------------------------------------------
        # 5. Penalty Adaptation
        # ---------------------------------------------
        current.self_adjust_penalties()

        # ---------------------------------------------
        # 6. Print iteration summary
        # ---------------------------------------------
        print(f"[GVNS][iter {iteration}] "
              f"F={improved.Fval:.3f} f={improved.travel_time:.3f} "
              f"| PenS={improved.penS} PenT={improved.penT} PenH={improved.penH} "
              f"| accepted={accept}, cont={cont}, k={k}, elapsed={elapsed:.2f}s")

        if trace_list is not None:
            trace_list.append([
                iteration, current.Fval, current.travel_time,
                current.penT, current.penH, current.penS, elapsed
            ])

    best_aug.eval()
    logger.close()
    return best_aug


# import time
# import random
# from gvns.solution import Solution
# from gvns.initializer import random_insertion
# import gvns.operators as ops
# import gvns.local_search as ls
# from gvns.logger import GVNSLogger

# DEFAULT_PARAMS = {
#     "Contmax": 100,
#     "p_init": 1,
#     "p_max": 10,
#     "Dp": 0.3,
#     "w1": 0.5,
#     "w2": 0.1,
#     "alpha": 0.1,
#     "delta": 0.4,
#     "M": 1e6,
#     "seed": 0,
#     "Tmax": None,
# }

# def run_gvns(instance, params=None, verbose=True, trace_list=None):
#     if params is None:
#         cfg = DEFAULT_PARAMS.copy()
#     else:
#         cfg = params
#     random.seed(cfg["seed"])

#     depot_id = min(instance.nodes.keys())
#     all_non_depot = [nid for nid in instance.nodes.keys() if nid != depot_id]
#     expected_total = len([c for c in all_non_depot if c not in instance.sync_customers]) + \
#                      sum(instance.sync_requirements.get(c, 1) for c in instance.sync_customers)

#     def total_visits(sol: Solution):
#         return sum(len(rt) for rt in sol.tours)

#     def occ_distribution(sol: Solution):
#         from collections import Counter
#         cnt = Counter()
#         for rt in sol.tours:
#             for v in rt:
#                 cnt[v] += 1
#         return cnt
    
#     # open logger
#     log_path = cfg.get("log_path", "gvns_trace.jsonl")
#     logger = GVNSLogger(path=log_path, console=True)  
#     step_counter = 0

#     # initialization 
#     current = random_insertion(instance, seed=cfg["seed"])
#     current.w1 = cfg["w1"]; current.w2 = cfg["w2"]; current.alpha = cfg["alpha"]; current.M = cfg["M"]

#     tv_init = total_visits(current)
#     if tv_init != expected_total:
#         print("[GVNS][ERROR] initializer built unexpected total_visits:", tv_init, "expected:", expected_total)
#         print(" occurrences sample:", dict(list(occ_distribution(current).items())[:20]))

#     current.eval()
#     # logger.log("init", step_counter, current, note="initial_solution")

#     best_aug = current.copy()
#     best_aug.eval()

#     best_feasible = None
#     best_feasible_f = float("inf")
#     if current.penT == 0.0 and current.penH == 0.0 and current.penS == 0.0:
#         best_feasible = current.copy()
#         best_feasible_f = current.travel_time

#     p = cfg["p_init"]
#     cont = 0
#     iteration_num = 0
#     start_time = time.time()
#     time_limit = cfg["Tmax"]
#     delta = cfg["delta"]
#     p_min, p_max = cfg["p_init"], cfg["p_max"]
#     Dp = cfg["Dp"]

#     if verbose:
#         print(f"[GVNS] start: F={current.Fval:.4f}, f(travel)={current.travel_time:.4f}, "
#               f"PenT={current.penT:.4f}, PenH={current.penH:.4f}, PenS={current.penS}")

#     def compute_bvnd_time(n, sync_pairs, m,
#                           base_per_customer=0.4,
#                           per_sync_pair=4.0,
#                           per_vehicle=0.05,
#                           min_time=10.0,
#                           max_time=600.0,
#                           Tmax=None,
#                           frac_of_Tmax=0.8):
#         t = base_per_customer * n + per_sync_pair * sync_pairs + per_vehicle * m
#         if Tmax is not None:
#             t = min(t, frac_of_Tmax * Tmax)
#         t = max(min_time, min(t, max_time))
#         return float(t)

#     # main loop
#     while True:
#         iteration_num += 1
#         elapsed = time.time() - start_time
#         if time_limit and elapsed >= time_limit:
#             if verbose: print("[GVNS] time limit reached.")
#             break
#         if cont >= cfg["Contmax"]:
#             if verbose: print("[GVNS] cont limit reached.")
#             break

#         # shake
#         k = max(1, int(p))
#         shaken = ops.random_shake(current, k=k, insert_prob=Dp)
#         shaken.w1 = current.w1; shaken.w2 = current.w2; shaken.alpha = current.alpha; shaken.M = current.M

#         # log shaken solution
#         step_counter += 1
#         shaken.eval()
#         # logger.log("shaken", step_counter, shaken, note=f"shake_k={k}")

#         # local search (B-VND)
#         n_customers = len([nid for nid in instance.nodes if nid != depot_id])
#         sync_pairs = len(getattr(instance, "sync_customers", []))
#         m = instance.m
#         bvnd_time = compute_bvnd_time(n=n_customers, sync_pairs=sync_pairs, m=m, Tmax=time_limit)

#         sample_limit = max(8000, int(expected_total**2 / 6))
#         max_no_improve_iters = int(max(5, min(200, sample_limit // 500)))

#         improved = ls.b_vnd(
#             shaken,
#             neighborhoods=["N1", "N2", "N3", "N4"],
#             max_no_improve_iters=max_no_improve_iters,
#             time_limit=bvnd_time,
#             rl_p_pass=2,                     
#             rl_i_pass=int(max(1000, expected_total)),  
#             # l=1,                             
#             # k_shake_small=2,
#             require_penS_zero=True,            # enforce feasibility-first
#             allow_fallback_if_no_feasible=True            
#         )
#         improved.eval()

#         step_counter += 1
#         # logger.log("after_ls", step_counter, improved, note="after_local_search")
#         new = improved

#         # defensive checks
#         tv_new = total_visits(new)
#         if tv_new != expected_total or (new.Fval == 0.0 and (new.penT + new.penH + new.penS) > 0):
#             cont += 1
#             p = max(p_min, p - 1)
#             if trace_list is not None:
#                 trace_list.append([
#                     iteration_num, current.Fval, current.travel_time,
#                     current.penT, current.penH, current.penS, elapsed, best_feasible_f
#                 ])
#             continue

#         # acceptance (Move-or-Not)
#         threshold = (1.0 + delta) * current.Fval
#         accept = (new.Fval <= threshold)
#         if accept:
#             current = new
#             cont = 0
#             accepted_str = "ACCEPTED"
#             step_counter += 1
#             logger.log("accepted", step_counter, current, note=f"iter_{iteration_num}")
#         else:
#             cont += 1
#             accepted_str = "REJECTED"
#             step_counter += 1
#             logger.log("rejected", step_counter, new, note=f"iter_{iteration_num}")

#         # update augmented best (penalized objective) 
#         if new.Fval < best_aug.Fval and new.Fval > 1e-6:
#             best_aug = new.copy()
#             print(f"[GVNS][iteration {iteration_num}] NEW BEST (augmented): "
#                   f"F={best_aug.Fval:.4f} f={best_aug.travel_time:.4f} "
#                   f"PenT={best_aug.penT:.4f} PenH={best_aug.penH:.4f} PenS={best_aug.penS:.1f} "
#                   f"elapsed={elapsed:.2f}s")

#         # update feasible best (pure f) 
#         if new.penT == 0.0 and new.penH == 0.0 and new.penS == 0.0 and new.travel_time < best_feasible_f:
#             best_feasible = new.copy()
#             best_feasible_f = new.travel_time
#             print(f"[GVNS][iteration {iteration_num}] NEW BEST (feasible f): "
#                   f"f={best_feasible_f:.4f} elapsed={elapsed:.2f}s")

#         # penalty adaptation
#         current.eval()
#         current.self_adjust_penalties()

#         # update p
#         if cont > 0 and cont % 10 == 0:
#             p = min(p + 1, p_max)
#         if accept:
#             p = max(p_min, p - 1)

#         # iteration summary
#         print(f"[GVNS][iteration {iteration_num}] {accepted_str} | "
#               f"F={new.Fval:.4f} f={new.travel_time:.4f} PenT={new.penT:.4f} "
#               f"PenH={new.penH:.4f} PenS={new.penS} | "
#               f"current_F={current.Fval:.4f} best_F={best_aug.Fval:.4f} "
#               f"cont={cont} p={p} elapsed={elapsed:.2f}s")

#         if trace_list is not None:
#             trace_list.append([
#                 iteration_num,
#                 current.Fval,
#                 current.travel_time,
#                 current.penT,
#                 current.penH,
#                 current.penS,
#                 elapsed
#             ])

#     best_aug.eval()
#     # logger.log("finished", step_counter, best_aug, note="final_best_aug")
#     logger.close()
#     return best_aug
