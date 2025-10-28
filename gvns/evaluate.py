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

def repair_missing_and_sync(inst, sol, verbose=False, sync_tol=1e-4):
    """
    Repair missing visits and synchronization violations WITHOUT changing instance TWs.
    - Reinsert missing visits (best insertion).
    - For each sync pair (i,j):
        * if both are in the same route -> move one to a different route (or new route)
        * if in different routes and start times differ > sync_tol -> try local reinsertion
          (delay the earlier visit by moving it later in its route) or move the later earlier.
    Accept a candidate only if it reduces PenS or keeps PenS and reduces F.
    """
    from .operators import clone_solution
    from .evaluate import evaluate_solution, quick_repair_greedy

    repaired = clone_solution(sol)

    # ---- 1) fix missing visits by best insertion ----
    planned = {v for r in repaired.routes for v in r}
    missing = set(inst.visits.keys()) - planned
    for vid in missing:
        best_F = float("inf")
        best_sol = None
        for ridx, route in enumerate(repaired.routes):
            for pos in range(len(route) + 1):
                cand = clone_solution(repaired)
                cand.routes[ridx].insert(pos, vid)
                ev = evaluate_solution(inst, cand)
                if ev["F"] < best_F:
                    best_F, best_sol = ev["F"], cand
        # try new route
        cand = clone_solution(repaired)
        cand.routes.append([vid])
        ev = evaluate_solution(inst, cand)
        if ev["F"] < best_F:
            best_F, best_sol = ev["F"], cand
        if best_sol is not None:
            repaired = best_sol
            if verbose:
                print(f"[repair] reinserted missing visit {vid}")

    # ---- 2) handle sync pairs ----
    eval_rep = evaluate_solution(inst, repaired)
    sync_pairs = getattr(inst, "sync_pairs", []) or []

    for (a, b) in sync_pairs:
        start_times = eval_rep.get("start_times", {})
        sa = start_times.get(a)
        sb = start_times.get(b)
        if sa is None or sb is None:
            # missing will be penalized by PenS: nothing to do
            if verbose:
                print(f"[repair] pair ({a},{b}) missing start -> skipped")
            continue

        diff = abs(sa - sb)
        if diff <= sync_tol:
            # already fine
            continue

        # locate their routes
        rai = rbi = None
        for idx, r in enumerate(repaired.routes):
            if a in r:
                rai = idx
            if b in r:
                rbi = idx

        # if both in same route -> must separate them (they should belong to different vehicles)
        if rai is not None and rbi is not None and rai == rbi:
            # attempt to move one of them to a different route (best move)
            moved = False
            best_candidate = None
            best_val = float("inf")
            for to_idx in range(len(repaired.routes)):
                if to_idx == rai:
                    continue
                # try move a to route to_idx (every pos)
                base = clone_solution(repaired)
                # remove a (and b might still be there)
                for rr in base.routes:
                    if a in rr:
                        rr.remove(a)
                for pos in range(len(base.routes[to_idx]) + 1):
                    cand = clone_solution(base)
                    cand.routes[to_idx].insert(pos, a)
                    ev = evaluate_solution(inst, cand)
                    if ev["F"] < best_val:
                        best_val = ev["F"]
                        best_candidate = (cand, ev)
                # try move b instead
                base = clone_solution(repaired)
                for rr in base.routes:
                    if b in rr:
                        rr.remove(b)
                for pos in range(len(base.routes[to_idx]) + 1):
                    cand = clone_solution(base)
                    cand.routes[to_idx].insert(pos, b)
                    ev = evaluate_solution(inst, cand)
                    if ev["F"] < best_val:
                        best_val = ev["F"]
                        best_candidate = (cand, ev)

            # also consider creating a new route for a or b
            base = clone_solution(repaired)
            for rr in base.routes:
                if a in rr:
                    rr.remove(a)
            cand = clone_solution(base)
            cand.routes.append([a])
            ev = evaluate_solution(inst, cand)
            if ev["F"] < best_val:
                best_val = ev["F"]
                best_candidate = (cand, ev)

            base = clone_solution(repaired)
            for rr in base.routes:
                if b in rr:
                    rr.remove(b)
            cand = clone_solution(base)
            cand.routes.append([b])
            ev = evaluate_solution(inst, cand)
            if ev["F"] < best_val:
                best_val = ev["F"]
                best_candidate = (cand, ev)

            if best_candidate is not None:
                cand, ev = best_candidate
                # accept if decreases PenS or same PenS and decreases F
                if ev["PenS"] < eval_rep["PenS"] or (ev["PenS"] == eval_rep["PenS"] and ev["F"] + 1e-9 < eval_rep["F"]):
                    repaired, eval_rep = cand, ev
                    if verbose:
                        print(f"[repair] separated pair ({a},{b}) (same-route) -> PenS {ev['PenS']}, F {ev['F']:.3f}")
                    # attempt quick repair for TW issues
                    # repaired, eval_rep = quick_repair_greedy(inst, repaired, max_tries=200, verbose=verbose)
                    moved = True

            if not moved:
                if verbose:
                    print(f"[repair] couldn't separate pair ({a},{b}) without worsening F -> left as-is")
            continue

        # if in different routes, attempt to align by moving the earlier visit later in its route
        # pick early and late
        if sa < sb:
            early, late = a, b
            r_early = rai
        else:
            early, late = b, a
            r_early = rbi

        # try to move 'early' later inside its route (all possible insertion positions after current idx)
        best_candidate = None
        best_val = float("inf")
        if r_early is not None:
            route = repaired.routes[r_early]
            idx_early = route.index(early)
            # try moving early to later positions inside the same route
            for new_pos in range(idx_early + 1, len(route)):
                cand = clone_solution(repaired)
                rr = cand.routes[r_early]
                rr.remove(early)
                rr.insert(new_pos, early)
                ev = evaluate_solution(inst, cand)
                # compute new sync penalty count
                if ev["F"] < best_val:
                    best_val = ev["F"]
                    best_candidate = (cand, ev)

            # also try moving 'early' to another route (or new route) to increase its start time via travel/wait
            for to_idx in range(len(repaired.routes)):
                if to_idx == r_early:
                    continue
                for pos in range(len(repaired.routes[to_idx]) + 1):
                    cand = clone_solution(repaired)
                    # remove early
                    for rr in cand.routes:
                        if early in rr:
                            rr.remove(early)
                    cand.routes[to_idx].insert(pos, early)
                    ev = evaluate_solution(inst, cand)
                    if ev["F"] < best_val:
                        best_val = ev["F"]
                        best_candidate = (cand, ev)
            # try new route
            cand = clone_solution(repaired)
            for rr in cand.routes:
                if early in rr:
                    rr.remove(early)
            cand.routes.append([early])
            ev = evaluate_solution(inst, cand)
            if ev["F"] < best_val:
                best_val = ev["F"]
                best_candidate = (cand, ev)

        # accept candidate if it reduces PenS or same PenS and reduces F
        if best_candidate is not None:
            cand, ev = best_candidate
            if ev["PenS"] < eval_rep["PenS"] or (ev["PenS"] == eval_rep["PenS"] and ev["F"] + 1e-9 < eval_rep["F"]):
                repaired, eval_rep = cand, ev
                if verbose:
                    print(f"[repair] realigned pair ({a},{b}) by moving {early} -> PenS {ev['PenS']}, F {ev['F']:.3f}")
                # repaired, eval_rep = quick_repair_greedy(inst, repaired, max_tries=200, verbose=verbose)
            else:
                if verbose:
                    print(f"[repair] tried reinsertions for ({a},{b}) but no improvement")

    final_eval = evaluate_solution(inst, repaired)
    if verbose:
        print(f"[repair] final PenS={final_eval['PenS']}, PenT={final_eval['PenT']}, f={final_eval['f']}")
    return repaired, final_eval

def evaluate_solution(inst, sol: Solution, w1=0.5, w2=0.1, M=None, sync_tol=1e-4):
    """
    Pure evaluation: computes routing cost f, penalties, and reports synchronization violations.
    Does NOT mutate inst (time windows, etc).

    Returns diagnostic dict including:
      - 'f', 'PenT', 'PenH', 'PenS', 'F'
      - 'visit_service_starts': dict vid -> [start_times] (list in chronological order)
      - 'start_times': dict vid -> canonical start (chosen as the last recorded start if multiple)
      - 'visit_route_pos': dict vid -> (route_idx, position_index, start_time)
      - 'route_schedules': list of per-route schedule lists
      - 'sync_time_diff': sum of |sa-sb| over violating pairs
      - 'sync_violation_count'
    """
    from .instance import Instance
    assert isinstance(inst, Instance)

    # compute a large M if not provided
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

    visit_service_starts = {}   # vid -> [start_times]
    route_schedules = []        # per-route list of (vid, arrival, start, depart)
    visit_route_pos = {}        # vid -> (route_idx, pos_idx, start)

    def _svc(v):
        return getattr(v, "service", getattr(v, "service_time", 0.0))
    def _tw_early(v):
        return getattr(v, "tw_early", 0.0)
    def _tw_late(v):
        return getattr(v, "tw_late", float("inf"))

    def _tt(i, j):
        # support dict-style inst.travel_time[(i,j)] or callable inst.travel_time(i,j)
        if hasattr(inst, "travel_time"):
            if isinstance(inst.travel_time, dict):
                key = (getattr(i, 'id', i), j if isinstance(j, int) else getattr(j, 'id', j))
                return inst.travel_time.get(key, 0.0)
            elif callable(inst.travel_time):
                return inst.travel_time(i, j)
        if hasattr(inst, "distance_matrix"):
            return inst.distance_matrix[getattr(i,'id', i)][getattr(j,'id', j)]
        # fallback euclidean if data present
        xi, yi = (inst.depot.x, inst.depot.y) if getattr(i, 'id', i) == getattr(inst.depot, 'id', 0) else (inst.visits[getattr(i,'id',i)].x, inst.visits[getattr(i,'id',i)].y)
        xj, yj = (inst.depot.x, inst.depot.y) if getattr(j, 'id', j) == getattr(inst.depot, 'id', 0) else (inst.visits[getattr(j,'id',j)].x, inst.visits[getattr(j,'id',j)].y)
        return ((xi-xj)**2 + (yi-yj)**2)**0.5

    depot_node = inst.depot
    for r_idx, route in enumerate(sol.routes):
        cur_time = 0.0
        cur_node = depot_node
        route_travel = 0.0
        schedule = []
        for pos_idx, v_id in enumerate(route):
            travel = _tt(cur_node, v_id)
            arrival = cur_time + travel
            visit = inst.visits[v_id]
            start = max(arrival, _tw_early(visit))
            if start > _tw_late(visit):
                total_lateness += (start - _tw_late(visit))
            depart = start + _svc(visit)
            # record
            visit_service_starts.setdefault(v_id, []).append(start)
            schedule.append((v_id, arrival, start, depart))
            visit_route_pos[v_id] = (r_idx, pos_idx, start)
            cur_time = depart
            cur_node = visit
            route_travel += travel

        # return to depot
        route_travel += _tt(cur_node, getattr(inst.depot, 'id', 0))
        f += route_travel
        route_schedules.append(schedule)

        # horizon check
        return_time = cur_time + _tt(cur_node, getattr(inst.depot, 'id', 0))
        horizon_val = inst.horizon if inst.horizon is not None else float("inf")
        if return_time > horizon_val:
            horizon_violation += (return_time - horizon_val)

    # Pen: missing visits
    total_visits_expected = len(getattr(inst, "visits", {}))
    assigned_visits = sum(len(route) for route in sol.routes)
    missing_penalty = max(0, total_visits_expected - assigned_visits)

    # Build canonical start_times (if a visit appears multiple times we keep the last start)
    start_times = {vid: starts[-1] for vid, starts in visit_service_starts.items()}

    # Synchronization diagnostics
    sync_violation_count = 0
    sync_time_diff = 0.0
    for (a, b) in getattr(inst, "sync_pairs", []):
        sa = start_times.get(a)
        sb = start_times.get(b)
        if sa is None or sb is None:
            sync_violation_count += 1
            continue
        diff = abs(sa - sb)
        if diff > sync_tol:
            sync_violation_count += 1
            sync_time_diff += diff

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
        'sync_time_diff': sync_time_diff,
        'missing_penalty': missing_penalty,
        'sync_pairs_count': len(getattr(inst, "sync_pairs", [])),
        'visit_service_starts': visit_service_starts,
        'start_times': start_times,
        'visit_route_pos': visit_route_pos,
        'route_schedules': route_schedules
    }