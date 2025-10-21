from .solution import Solution

def compute_route_schedule(inst, route, speed=1.0):
    """
    return (schedule_list, total_time)
    schedule_list: [(vid, arrival, start, depart), ...]
    """
    def tt(i, j):
      if hasattr(inst, "travel_time"):
          if callable(inst.travel_time):
              return inst.travel_time(i, j)
          elif isinstance(inst.travel_time, dict):
              return inst.travel_time[(i, j)]
      elif hasattr(inst, "distance_matrix"):
          return inst.distance_matrix[i][j]
      raise ValueError("No travel-time data available")


    depot_id = getattr(inst.depot, "id", 0)
    times = []
    current = depot_id
    current_time = getattr(inst.depot, "start_tw", 0.0)
    for vid in route:
        v = inst.visits[vid]
        arrival = current_time + tt(current, vid)
        start_service = max(arrival, getattr(v,'tw_early', arrival))
        departure = start_service + getattr(v, 'service_time', 0.0)
        times.append((vid, arrival, start_service, departure))
        current = vid
        current_time = departure
    # return to depot
    total_time = current_time + tt(current, depot_id)
    return times, total_time


def is_feasible(inst, sol, eval_dict=None, epsilon=1e-6, verbose=False):
    """
    params
    ----------
    inst : Instance
        problem instance containing visits, depot, horizons, and sync pairs.
    sol : Solution
        candidate solution with .routes (list of list of visit IDs).
    eval_dict : dict, optional
        pre-computed evaluation dictionary from evaluate_solution().
        If None, evaluate_solution is called internally.
    epsilon : float
        allowed tolerance for synchronization timing differences.
    verbose : bool
        if True, print details of each violation.
    """

    if eval_dict is None:
        eval_dict = evaluate_solution(inst, sol)

    feasible = True
    messages = []

    #  1. time-window feasibility 
    if eval_dict.get("PenT", 0.0) > epsilon:
        feasible = False
        messages.append(f"Time-window violations total = {eval_dict['PenT']:.3f}")

    #  2. route horizon feasibility 
    if eval_dict.get("PenH", 0.0) > epsilon:
        feasible = False
        messages.append(f"Route-horizon violations total = {eval_dict['PenH']:.3f}")

    #  3. synchronization feasibility 
    # each synchronized pair (i,j) must start within epsilon of each other.
    if getattr(inst, "sync_pairs", None):
        # ee need the actual start times per visit
        start_times = eval_dict.get("start_times")
        if start_times is None:
            # evaluate_solution already computes them; if not, recompute
            _, _, start_times = evaluate_solution(inst, sol, return_times=True)
        for i, j in inst.sync_pairs:
            si, sj = start_times.get(i), start_times.get(j)
            if si is None or sj is None or abs(si - sj) > 300:
                feasible = False
                messages.append(
                    f"Sync violation between visits {i} and {j}: |{si:.3f}-{sj:.3f}|>{epsilon}"
                )

    #  4. unplanned-visit feasibility 
    # every required visit must appear in some route
    planned = {v for r in sol.routes for v in r}
    all_required = set(inst.visits.keys())
    missing = all_required - planned
    if missing:
        feasible = False
        messages.append(f"Missing visits: {sorted(missing)}")

    if verbose and not feasible:
        print("Infeasible solution details:", "; ".join(messages))

    return feasible

def clone_solution(sol):
    """
    return a deep copy of a Solution so that moves can modify the copy
    without affecting the original.
    """
    new_sol = Solution(routes=[list(r) for r in sol.routes])
    for attr in ["cost", "meta", "objective"]:
        if hasattr(sol, attr):
            setattr(new_sol, attr, getattr(sol, attr))
    return new_sol

def quick_repair_greedy(inst, sol, max_tries=200, verbose=False):
    """
    - its greedy and local so avoids expensive global search
    - focuses only on visits that actually violate constraintss
    - accept ny lower-f condidate even if still infeasible (we accept some violation as worst case) so it monotonically improves
    """
    best_sol = clone_solution(sol)
    best_eval = evaluate_solution(inst, best_sol)
    if is_feasible(inst, best_sol, best_eval):
        return best_sol, best_eval

    # 1) collect offending visits
    # herer i try to find nodes that cause penalties or are missing
    offending = set()
    for r_idx, route in enumerate(sol.routes):
        #find any customer whose start time is after its latest allowed time window
        schedule, _ = compute_route_schedule(inst, route)
        for (vid, _, start, _) in schedule:
            if start > inst.visits[vid].tw_late + 1e-9:
                offending.add(vid)

    planned = {v for r in best_sol.routes for v in r}
    missing = set(inst.visits.keys()) - planned
    offending.update(missing)

    # for each sync pair (i, j), if their start times differ or one is missing, mark both as offending
    for (i, j) in getattr(inst, "sync_pairs", []):
        si = best_eval.get("start_times", {}).get(i)
        sj = best_eval.get("start_times", {}).get(j)
        if si is None or sj is None or abs(si - sj) > 1e-6:
            offending.add(i); offending.add(j)

    if verbose:
        print(f"[quick_repair] offending visits (count={len(offending)}):", sorted(offending))

    current_F = best_eval["F"]

    tries = 0
    for vid in list(offending):
        if tries >= max_tries:
            break
        tries += 1
        # create base by removing vid from any current route
        base = clone_solution(best_sol)
        for rr in base.routes:
            if vid in rr:
                rr.remove(vid)
        # try all insertion positions
        improved_for_vid = False
        best_local_for_vid = None
        best_local_val = float('inf')

        for ridx in range(len(base.routes)):
            route_len = len(base.routes[ridx])
            for pos in range(route_len + 1):
                cand = clone_solution(base)
                cand.routes[ridx].insert(pos, vid)
                #evaluate F
                ev_c = evaluate_solution(inst, cand)
                if ev_c["F"] + 1e-9 < best_local_val:
                    best_local_val = ev_c["F"]
                    best_local_for_vid = (cand, ev_c)
                    if ev_c["F"] + 1e-9 < current_F:
                        # immediate accept improvement for this vid
                        best_sol, best_eval = cand, ev_c
                        current_F = ev_c["F"]
                        improved_for_vid = True
                        break
            if improved_for_vid:
                break

        # try new route for vid
        cand = clone_solution(base)
        cand.routes.append([vid])
        ev_c = evaluate_solution(inst, cand)
        # if it gives an improvement, update best_sol, and best_eval
        if ev_c["F"] + 1e-9 < best_local_val:
            best_local_val = ev_c["F"]
            best_local_for_vid = (cand, ev_c)
            if ev_c["F"] + 1e-9 < current_F:
                best_sol, best_eval = cand, ev_c
                current_F = ev_c["F"]
                improved_for_vid = True

        # if we found some improvement at least keep the best local (even if not feasible yet)
        if not improved_for_vid and best_local_for_vid is not None and best_local_val + 1e-9 < current_F:
            best_sol, best_eval = best_local_for_vid

        # quick exit if now feasible
        if is_feasible(inst, best_sol, best_eval):
            if verbose:
                print("[quick_repair] feasible achieved after repairing vid", vid)
            return best_sol, best_eval

    return best_sol, best_eval

def evaluate_solution(inst, sol: Solution, w1=0.5, w2=0.1, M=None):
    """
    evaluate a solution and also return diagnostic timing info.

    returns a dict with keys:
      'f', 'PenT', 'PenH', 'PenS', 'sync_time_diff', 'sync_pair_mismatch_count', 'F',
      plus diagnostics: 'visit_service_starts' (dict visit -> [start_times]),
      'start_times' (visit -> first start), 'route_schedules' (list of per-route schedules).

    this is compatible with previously-existing code that expects 'f' and 'F'.
    """
    from .instance import Instance 
    assert isinstance(inst, Instance)

    try:
        if M is None:
            if getattr(inst, "travel_time", None):
                try:
                    max_tt = max(inst.travel_time.values())
                except Exception:
                    max_tt = 1.0
            else:
                max_tt = 1.0
            n_vis = max(1, len(getattr(inst, "visits", {})))
            M = max_tt * n_vis * 100.0
    except Exception:
        M = 1e6

    f = 0.0
    total_lateness = 0.0
    horizon_violation = 0.0

    # store start times per visit (list) and per-route schedules
    visit_service_starts = {}
    route_schedules = []

    # helper to read service time and time windows (works with both .service/.service_time and .tw_early/.tw_late)
    def _svc(v):
        return getattr(v, "service", getattr(v, "service_time", 0.0))
    def _tw_early(v):
        return getattr(v, "tw_early", 0.0)
    def _tw_late(v):
        return getattr(v, "tw_late", float("inf"))

    # helper to get travel time
    def _tt(i, j):
        # support dict-style inst.travel_time[(i,j)] or method inst.travel_time(i,j)
        if hasattr(inst, "travel_time"):
            if isinstance(inst.travel_time, dict):
                return inst.travel_time.get((getattr(i, 'id', i), j if isinstance(j, int) else getattr(j,'id',j)), 0.0)
            elif callable(inst.travel_time):
                return inst.travel_time(i, j)
        # fallback: distance_matrix or euclidean
        if hasattr(inst, "distance_matrix"):
            try:
                return inst.distance_matrix[getattr(i,'id', i)][getattr(j,'id', j)]
            except Exception:
                pass
        # last fallback: compute euclidean from attributes
        xi, yi = (inst.depot.x, inst.depot.y) if getattr(i, 'id', i) == getattr(inst.depot, 'id', 0) else (inst.visits[getattr(i,'id',i)].x, inst.visits[getattr(i,'id',i)].y)
        xj, yj = (inst.depot.x, inst.depot.y) if getattr(j, 'id', j) == getattr(inst.depot, 'id', 0) else (inst.visits[getattr(j,'id',j)].x, inst.visits[getattr(j,'id',j)].y)
        return ((xi-xj)**2 + (yi-yj)**2)**0.5

    # evaluate route by route, building schedules
    for route in sol.routes:
        cur_time = 0.0
        cur_node = inst.depot
        route_travel = 0.0
        schedule = []  # list of (vid, arrival, start, depart)
        for v_id in route:
            travel = _tt(cur_node, v_id)
            arrival = cur_time + travel
            visit = inst.visits[v_id]
            start = max(arrival, _tw_early(visit))
            # lateness
            if start > _tw_late(visit):
                total_lateness += (start - _tw_late(visit))
            depart = start + _svc(visit)
            # store start
            visit_service_starts.setdefault(v_id, []).append(start)
            schedule.append((v_id, arrival, start, depart))
            cur_time = depart
            cur_node = visit
            route_travel += travel

        # return to depot travel
        route_travel += _tt(cur_node, inst.depot.id if hasattr(inst.depot, 'id') else inst.depot)
        f += route_travel
        route_schedules.append(schedule)

        # compute route return time for horizon
        # last depart time is cur_time; add travel back to depot:
        return_time = cur_time + _tt(cur_node, inst.depot.id if hasattr(inst.depot, 'id') else inst.depot)
        horizon_val = inst.horizon if inst.horizon is not None else float("inf")
        if return_time > horizon_val:
            horizon_violation += (return_time - horizon_val)

    # PenS: missing visits
    total_visits_expected = len(getattr(inst, "visits", {}))
    assigned_visits = sum(len(route) for route in sol.routes)
    missing_penalty  = max(0, total_visits_expected - assigned_visits)

    # synchronize visits
    # visit_positions = {}
    # for r_idx, schedule in enumerate(route_schedules):
    #     for v_idx, (vid, arr, start, dep) in enumerate(schedule):
    #         visit_positions[vid] = (r_idx, v_idx)

    # for (a, b) in getattr(inst, "sync_pairs", []):
    #     if a not in visit_positions or b not in visit_positions:
    #         continue
    #     ra, ia = visit_positions[a]
    #     rb, ib = visit_positions[b]

    #     # Get current start times
    #     sa = route_schedules[ra][ia][2]
    #     sb = route_schedules[rb][ib][2]
    #     t_sync = max(sa, sb)

    #     # Propagate delay if needed
    #     for (r_idx, v_idx, s_old) in [(ra, ia, sa), (rb, ib, sb)]:
    #         delay = t_sync - s_old
    #         if delay > 0:
    #             for j in range(v_idx, len(route_schedules[r_idx])):
    #                 vid_j, arr_j, start_j, dep_j = route_schedules[r_idx][j]
    #                 arr_j += delay
    #                 start_j += delay
    #                 dep_j += delay
    #                 route_schedules[r_idx][j] = (vid_j, arr_j, start_j, dep_j)
    #                 # update visit_service_starts as well
    #                 visit_service_starts[vid_j][0] = start_j

    # total_lateness = 0.0
    # for r_idx, schedule in enumerate(route_schedules):
    #     for (vid, arr, start, dep) in schedule:
    #         if start > _tw_late(inst.visits[vid]):
    #             total_lateness += (start - _tw_late(inst.visits[vid]))

    # total_visits_expected = len(inst.visits)
    # assigned_visits = sum(len(route) for route in sol.routes)
    # missing_penalty = max(0, total_visits_expected - assigned_visits)
    # #---

    # synchronization diagnostic
    sync_penalty = 0.0
    tolerance = 240
    for (a, b) in getattr(inst, "sync_pairs", []):
        starts_a = visit_service_starts.get(a, [])
        starts_b = visit_service_starts.get(b, [])
        if starts_a and starts_b:
            diff = abs(starts_a[0] - starts_b[0])
            if diff > tolerance:
                sync_penalty += diff - tolerance

    PenT = total_lateness
    PenH = horizon_violation
    PenS = missing_penalty + sync_penalty

    Pen = w1 * PenT + w2 * PenH + M * PenS
    F = f + Pen

    # prepare start_times: single representative start (first) per visit
    start_times = {vid: starts[0] for vid, starts in visit_service_starts.items()}

    return {
        'f': f,
        'PenT': PenT,
        'PenH': PenH,
        'PenS': PenS,
        'F': F,
        'sync_penalty': sync_penalty,
        'missing_penalty': missing_penalty,
        'sync_pairs_count': len(getattr(inst, "sync_pairs", [])),
        # diagnostics:
        'visit_service_starts': visit_service_starts,
        'start_times': start_times,
        'route_schedules': route_schedules
    }