from .solution import Solution
import math
import random

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

def quick_repair_greedy(inst, sol, max_tries=200, verbose=False, w1=0.5, w2=0.1, M=None):
    """
    Fast feasibility repair for VRPTW-S solutions.

    Goals:
      - Quickly reduce PenT / PenH / PenS to restore feasibility.
      - Avoid expensive nested searches (no recursive b_vnd calls).
      - Evaluate only the most promising insertions.
      - Stop early if no improvement in penalty is observed.

    Returns:
      (best_sol, best_eval)
    """
    import heapq, random, math, itertools
    from .local_search import intra_route_swap, evaluate_solution, clone_solution, is_feasible

    # --- helper: composite score combining raw f + weighted penalties ---
    def score_from_eval(ev, w1_, w2_, M_):
        if M_ is None:
            M_ = max(1e6, getattr(inst, "horizon", 1e6))
        return float(ev.get("f", 0.0)) + float(
            w1_ * ev.get("PenT", 0.0) + w2_ * ev.get("PenH", 0.0) + M_ * ev.get("PenS", 0.0)
        )

    # --- fallback M ---
    if M is None:
        try:
            max_tt = max(inst.travel_time.values()) if hasattr(inst, "travel_time") and inst.travel_time else 1.0
            n_vis = max(1, len(getattr(inst, "visits", {})))
            M = max(1e6, min(max_tt * n_vis * 100.0, 1e12))
        except Exception:
            M = 1e6

    best_sol = clone_solution(sol)
    best_eval = evaluate_solution(inst, best_sol)
    best_score = score_from_eval(best_eval, w1, w2, M)
    last_penT = best_eval.get("PenT", 0.0)

    if is_feasible(inst, best_sol, best_eval):
        return best_sol, best_eval

    # --- identify offenders ---
    offending = set()
    severity = {}

    for r_idx, route in enumerate(best_sol.routes):
        schedule, _ = compute_route_schedule(inst, route)
        for idx, (vid, _, start, _) in enumerate(schedule):
            visit = inst.visits[vid]
            late = max(0.0, start - getattr(visit, "tw_late", float("inf")))
            if late > 1e-9:
                offending.add(vid)
                severity[vid] = severity.get(vid, 0.0) + late
                if idx > 0:
                    pred = route[idx - 1]
                    offending.add(pred)
                    severity[pred] = severity.get(pred, 0.0) + late * 0.5
                if idx + 1 < len(route):
                    succ = route[idx + 1]
                    offending.add(succ)
                    severity[succ] = severity.get(succ, 0.0) + late * 0.2

    # also handle missing & unsynced
    planned = {v for r in best_sol.routes for v in r}
    missing = set(inst.visits.keys()) - planned
    for vid in missing:
        offending.add(vid)
        severity[vid] = severity.get(vid, 0.0) + 1e6

    for (a, b) in getattr(inst, "sync_pairs", []):
        sa = best_eval.get("start_times", {}).get(a)
        sb = best_eval.get("start_times", {}).get(b)
        if sa is None or sb is None or abs(sa - sb) > 1e-6:
            offending.add(a)
            offending.add(b)
            severity[a] = severity.get(a, 0.0) + 1e5
            severity[b] = severity.get(b, 0.0) + 1e5

    offenders = sorted(list(offending), key=lambda x: -severity.get(x, 0.0))
    if verbose:
        print(f"[quick_repair] offenders={len(offenders)}, top={offenders[:10]}")

    # --- proximity helper ---
    def is_near(visit, route_first, route_last, thresh_factor=0.25):
        try:
            if hasattr(inst, "travel_time") and inst.travel_time:
                def tt(a, b):
                    key = (getattr(a, 'id', a), getattr(b, 'id', b))
                    return inst.travel_time.get(key, 0.0)
                d1 = tt(visit, route_first)
                d2 = tt(visit, route_last)
                scale = max(1.0, max(inst.travel_time.values()))
            else:
                d1 = math.hypot(visit.x - route_first.x, visit.y - route_first.y)
                d2 = math.hypot(visit.x - route_last.x, visit.y - route_last.y)
                scale = 1.0
            thr = max(1e-6, thresh_factor * scale * max(1.0, len(getattr(inst, "visits", {}))))
            return min(d1, d2) <= thr
        except Exception:
            return True

    # --- main repair loop ---
    tries = 0
    no_penT_improve = 0
    counter = itertools.count()  # <--- tie-breaker for heapq

    for vid in offenders:
        if tries >= max_tries or no_penT_improve >= 5:
            break
        tries += 1

        base = clone_solution(best_sol)
        for rr in base.routes:
            while vid in rr:
                rr.remove(vid)

        visit = inst.visits[vid]
        near_routes = []
        for ridx, r in enumerate(base.routes):
            if not r:
                near_routes.append(ridx)
                continue
            first, last = inst.visits[r[0]], inst.visits[r[-1]]
            if is_near(visit, first, last):
                near_routes.append(ridx)
        if not near_routes:
            near_routes = list(range(len(base.routes)))

        # --- try cheap insertions, keep top few ---
        top_candidates = []
        for ridx in near_routes:
            rlen = len(base.routes[ridx])
            positions = list(range(rlen + 1))
            if len(positions) > 30:
                random.shuffle(positions)
                positions = positions[:30]
            for pos in positions:
                cand = clone_solution(base)
                cand.routes[ridx].insert(pos, vid)
                ev = evaluate_solution(inst, cand)
                sc = score_from_eval(ev, w1, w2, M)
                heapq.heappush(top_candidates, (sc, next(counter), cand, ev))
                if len(top_candidates) > 5:
                    heapq.heappop(top_candidates)

        # --- optional: try new single route ---
        cand = clone_solution(base)
        cand.routes.append([vid])
        ev = evaluate_solution(inst, cand)
        sc = score_from_eval(ev, w1, w2, M)
        heapq.heappush(top_candidates, (sc, next(counter), cand, ev))
        if len(top_candidates) > 5:
            heapq.heappop(top_candidates)

        # --- refine only the best few candidates ---
        best_local = None
        best_local_score = float("inf")
        for sc, _, cand, ev in top_candidates:
            # quick intra-route improvement on the route containing vid
            for ridx, r in enumerate(cand.routes):
                if vid in r and len(r) > 2:
                    for i in range(len(r) - 1):
                        tmp = intra_route_swap(cand, ridx, i, i + 1)
                        ev2 = evaluate_solution(inst, tmp)
                        sc2 = score_from_eval(ev2, w1, w2, M)
                        if sc2 < sc:
                            cand, ev, sc = tmp, ev2, sc2
            if sc < best_local_score:
                best_local_score = sc
                best_local = (cand, ev)

        # --- accept improvement if better score or PenT reduced ---
        if best_local:
            cand_sol, cand_ev = best_local
            cand_penT = cand_ev.get("PenT", 0.0)
            penT_reduced = cand_penT < best_eval.get("PenT", 0.0) - 1e-6
            cand_score = score_from_eval(cand_ev, w1, w2, M)

            if cand_score + 1e-9 < best_score or penT_reduced:
                best_sol, best_eval, best_score = cand_sol, cand_ev, cand_score
                no_penT_improve = 0
                if verbose:
                    print(f"[quick_repair] accepted vid={vid} PenT={cand_penT:.2f} score={cand_score:.2f}")
            else:
                no_penT_improve += 1

        # --- early exit if feasible ---
        if is_feasible(inst, best_sol, best_eval):
            if verbose:
                print(f"[quick_repair] feasible after vid={vid}, tries={tries}")
            break

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