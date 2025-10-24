from .solution import Solution

def compute_route_schedule(inst, route, speed=1.0):
    """
    return (schedule_list, total_time)
    schedule_list: [(vid, arrival, start, depart), ...]
    """
    def _get_id(x):
        return getattr(x, "id", x)
    
    def tt(i, j):
        # normalize to ids
        ii = _get_id(i)
        jj = _get_id(j)
        if hasattr(inst, "travel_time"):
            if callable(inst.travel_time):
                return inst.travel_time(ii, jj)
            elif isinstance(inst.travel_time, dict):
                key = (ii, jj)
                if key not in inst.travel_time:
                    raise KeyError(f"travel_time missing key {key} in instance {getattr(inst, 'name', '')}")
                return inst.travel_time[key]
        if hasattr(inst, "distance_matrix"):
            return inst.distance_matrix[ii][jj]
        raise ValueError("No travel-time data available")


    depot_id = getattr(inst.depot, "id", 0)
    times = []
    current = depot_id
    current_time = 0.0
    for vid in route:
        v = inst.visits[vid]
        arrival = current_time + tt(current, vid)
        start_service = max(arrival, getattr(v,'tw_early', arrival))
        departure = start_service + getattr(v, 'service', 0.0)
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
            eval_dict = evaluate_solution(inst, sol)
            start_times = eval_dict.get("start_times", {})
        for i, j in inst.sync_pairs:
            si, sj = start_times.get(i), start_times.get(j)
            if si is None or sj is None or abs(si - sj) > epsilon:
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

def repair_missing_and_sync(inst, sol, verbose=False, snap_tol=1e-3):
    """
    Repair missing visits and synchronization violations.

    Behavior:
      - Reinserts missing visits (best insertion into existing routes).
      - For each sync pair (i,j) with different start times, make them consecutive
        (merge routes or reorder) and align temporally by delaying the earlier
        visit to the later start time (target = max(si, sj)).
      - Only commit time-window changes when the candidate *improves* PenS, or
        keeps PenS equal and improves F.
      - After accepting a candidate, run quick_repair_greedy to heal TW violations.
    """
    from .operators import clone_solution
    from .evaluate import evaluate_solution, quick_repair_greedy

    repaired = clone_solution(sol)

    # helper to save/restore original TWs (so we can rollback on rejection)
    original_tws = {}  # vid -> (orig_early, orig_late)
    def save_tw(vid):
        if vid not in original_tws:
            v = inst.visits[vid]
            original_tws[vid] = (getattr(v, "tw_early", 0.0), getattr(v, "tw_late", float("inf")))
    def restore_saved_tws():
        for vid, (e, l) in original_tws.items():
            v = inst.visits.get(vid)
            if v is not None:
                v.tw_early = e
                v.tw_late = l
        original_tws.clear()
    def clear_saved_tws_commit():
        # candidate accepted -> commit changes (forget originals)
        original_tws.clear()

    # small helpers for travel/service (robust)
    def _svc(vid):
        v = inst.visits[vid]
        return getattr(v, "service", getattr(v, "service_time", 0.0))
    def _tt(a, b):
        # try dict keyed by (a,b) or callable
        try:
            if hasattr(inst, "travel_time"):
                if isinstance(inst.travel_time, dict):
                    # keys may be ints or visit ids; assume a,b are ids (int)
                    return inst.travel_time.get((a, b), inst.travel_time.get((getattr(a,'id',a), getattr(b,'id',b)), 0.0))
                elif callable(inst.travel_time):
                    return inst.travel_time(a, b)
        except Exception:
            pass
        return 0.0

    # ---- 1) Fix missing visits by best insertion into existing routes ----
    planned = {v for r in repaired.routes for v in r}
    missing = set(inst.visits.keys()) - planned
    for vid in missing:
        best_F = float("inf")
        best_sol = None
        # try insert into every route and every position
        for ridx, route in enumerate(repaired.routes):
            for pos in range(len(route) + 1):
                cand = clone_solution(repaired)
                cand.routes[ridx].insert(pos, vid)
                ev = evaluate_solution(inst, cand)
                if ev["F"] < best_F:
                    best_F, best_sol = ev["F"], cand
        # also allow creating a new route if nothing found
        if best_sol is None:
            cand = clone_solution(repaired)
            cand.routes.append([vid])
            best_sol = cand
        repaired = best_sol
        if verbose:
            print(f"[repair] reinserted missing visit {vid}")

    # ---- 2) Fix synchronization ----
    eval_rep = evaluate_solution(inst, repaired)
    sync_pairs = getattr(inst, "sync_pairs", []) or []
    for (i, j) in sync_pairs:
        si = eval_rep.get("start_times", {}).get(i)
        sj = eval_rep.get("start_times", {}).get(j)
        if si is None or sj is None:
            # missing visit(s) will be penalized as PenS; nothing to do here
            if verbose:
                print(f"[repair] pair ({i},{j}) has missing start(s) -> skipped")
            continue

        if abs(si - sj) <= 1e-6:
            # already aligned (within tiny tolerance)
            continue

        # identify early, late and their current routes
        if si < sj:
            early, late = i, j
        else:
            early, late = j, i

        ri = rj = None
        for r_idx, r in enumerate(repaired.routes):
            if early in r:
                ri = r_idx
            if late in r:
                rj = r_idx
        if ri is None or rj is None:
            if verbose:
                print(f"[repair] couldn't locate routes for ({early},{late}), skipping")
            continue

        # candidate solution: place late right after early (merge or reorder)
        cand = clone_solution(repaired)
        if ri != rj:
            # remove late from its route and put it after early
            for r in cand.routes:
                if late in r:
                    r.remove(late)
            pos = cand.routes[ri].index(early) + 1
            cand.routes[ri].insert(pos, late)
            # remove empty routes
            cand.routes = [r for r in cand.routes if len(r) > 0]
            if verbose:
                print(f"[repair] merged routes for pair ({early},{late})")
        else:
            # same route: make consecutive if not already
            r_e = cand.routes[ri]
            idx_e = r_e.index(early)
            idx_l = r_e.index(late)
            if abs(idx_e - idx_l) > 1:
                r_e.remove(late)
                r_e.insert(idx_e + 1, late)
                if verbose:
                    print(f"[repair] reordered pair ({early},{late}) to be consecutive")

        # evaluate candidate ordering to get up-to-date start times
        ev_before = evaluate_solution(inst, cand)
        si2 = ev_before["start_times"].get(early)
        sj2 = ev_before["start_times"].get(late)
        if si2 is None or sj2 is None:
            if verbose:
                print(f"[repair] candidate ordering for ({early},{late}) has missing times -> skip")
            continue

        # Save TWs of involved visits so we can rollback if we reject
        save_tw(early)
        save_tw(late)

        # Liu-style: align earlier to the later start (use max)
        target_time = max(si2, sj2)
        # Delay the earlier visit to the target time (force early start == target)
        inst.visits[early].tw_early = target_time
        inst.visits[early].tw_late = target_time
        # Make sure 'late' permits a start at target_time (expand if needed)
        inst.visits[late].tw_early = min(getattr(inst.visits[late], "tw_early", 0.0), target_time)
        inst.visits[late].tw_late = max(getattr(inst.visits[late], "tw_late", float("inf")), target_time)
        if verbose:
            print(f"[repair] aligned ({early},{late}) -> delayed earlier to {target_time:.3f}")

        # Evaluate candidate after committing temporary TW changes
        ev = evaluate_solution(inst, cand)

        # Acceptance rule: prefer strictly smaller PenS, or same PenS and smaller F.
        better_pen_s = ev["PenS"] < eval_rep["PenS"]
        same_pen_better_f = (ev["PenS"] == eval_rep["PenS"]) and (ev["F"] + 1e-9 < eval_rep["F"])

        # if better_pen_s or same_pen_better_f:
            # accept candidate
        repaired, eval_rep = cand, ev
        if verbose:
            print(f"[repair] accepted cand for pair ({early},{late}) -> PenS {ev['PenS']}, F {ev['F']:.3f}")
        # try to heal new TW violations created by forcing the alignment
        repaired, eval_rep = quick_repair_greedy(inst, repaired, max_tries=300, verbose=verbose)
        # commit TW changes (we already modified inst in place), forget originals
        clear_saved_tws_commit()
        # else:
        #     # reject: rollback TW changes
        #     restore_saved_tws()
        #     if verbose:
        #         print(f"[repair] rejected cand for pair ({early},{late}) (no PenS/F improvement)")

    final_eval = evaluate_solution(inst, repaired)
    if verbose:
        print(f"[repair] final PenS={final_eval['PenS']}, PenT={final_eval['PenT']}, f={final_eval['f']}")
    return repaired, final_eval

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
            try:
                max_tt = max(inst.travel_time.values())
            except Exception:
                max_tt = 1.0
            n_vis = max(1, len(getattr(inst, "visits", {})))
            M = max(1e6, min(max_tt * n_vis * 100.0, 1e12))
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

    # synchronization diagnostic
    sync_violation_count = 0
    tolerance = 1e-4
    sync_time_diff = 0.0
    start_times = {vid: starts[-1] for vid, starts in visit_service_starts.items()}

    for (a, b) in getattr(inst, "sync_pairs", []):
        sa, sb = start_times.get(a), start_times.get(b)
        if sa is None or sb is None:
            sync_violation_count += 1
            continue

        diff = abs(sa - sb)
        if diff > tolerance:
            sync_violation_count += 1
            sync_time_diff += diff

            # avg_time = 0.5 * (sa + sb)
            # start_times[a] = avg_time
            # start_times[b] = avg_time
            # # adjust time windows locally (light relaxation)
            # if a in inst.visits and b in inst.visits:
            #     va = inst.visits[a]
            #     vb = inst.visits[b]
            #     va.tw_early = min(va.tw_early, avg_time)
            #     va.tw_late  = max(va.tw_late, avg_time)
            #     vb.tw_early = min(vb.tw_early, avg_time)
            #     vb.tw_late  = max(vb.tw_late, avg_time)

    PenT = total_lateness
    PenH = horizon_violation
    PenS = missing_penalty + sync_violation_count

    Pen = w1 * PenT + w2 * PenH + M * PenS
    F = f + Pen

    return {
        'f': f,
        'PenT': PenT,
        'PenH': PenH,
        'PenS': PenS,
        'F': F,
        'sync_violation_count': sync_violation_count,
        'missing_penalty': missing_penalty,
        'sync_pairs_count': len(getattr(inst, "sync_pairs", [])),
        # diagnostics:
        'visit_service_starts': visit_service_starts,
        'start_times': start_times,
        'route_schedules': route_schedules
    }