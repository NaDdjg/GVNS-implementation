from .solution import Solution
from .instance import Instance
from .evaluate import evaluate_solution
from .viz import visualize_solution
import random

def clone_solution(sol: Solution) -> Solution:
    return Solution(routes=[list(r) for r in sol.routes])

def intra_route_insertion(sol: Solution, route_idx: int, from_pos: int, to_pos: int) -> Solution:
    new_sol = clone_solution(sol)
    route = new_sol.routes[route_idx]
    v = route.pop(from_pos)
    # adjust to_pos if removal occurred before insertion point
    if to_pos > from_pos:
        to_pos -= 1
    route.insert(to_pos, v)
    return new_sol

def inter_route_insertion(sol: Solution, from_route: int, pos: int, to_route: int, insert_pos: int) -> Solution:
    new_sol = clone_solution(sol)
    v = new_sol.routes[from_route].pop(pos)
    new_sol.routes[to_route].insert(insert_pos, v)
    # remove empty routes optionally
    new_sol.routes = [r for r in new_sol.routes if len(r) > 0]
    return new_sol

def intra_route_swap(sol: Solution, route_idx: int, pos1: int, pos2: int) -> Solution:
    new_sol = clone_solution(sol)
    r = new_sol.routes[route_idx]
    r[pos1], r[pos2] = r[pos2], r[pos1]
    return new_sol

def inter_route_swap(sol: Solution, r1:int, p1:int, r2:int, p2:int) -> Solution:
    new_sol = clone_solution(sol)
    a = new_sol.routes[r1][p1]
    b = new_sol.routes[r2][p2]
    new_sol.routes[r1][p1] = b
    new_sol.routes[r2][p2] = a
    return new_sol

def shake_solution(inst: Instance, sol: Solution, p:int, Dp:float=0.3, seed=None, debug_visual: bool = False):
    """
    apply p randomized moves: with prob Dp pick a swap move (N3/N4), else insertion (N1/N2)
    will return shaken solution and applied moves (surtout for debug)
    """
    if debug_visual:
        ev0 = evaluate_solution(inst, sol)
        visualize_solution(inst, sol,
                           start_times=ev0.get("start_times"),
                           title="shake_before", show=True)

    if seed is not None:
        random.seed(seed)
    cur = clone_solution(sol)
    moves = []
    for _ in range(p):
        if random.random() < Dp:
            # swap move: choose intra or inter swap with 50-50
            if random.random() < 0.5 and len(cur.routes) > 0:
                # intra-route swap
                r = random.randrange(len(cur.routes))
                if len(cur.routes[r]) >= 2:
                    i, j = random.sample(range(len(cur.routes[r])), 2)
                    cur = intra_route_swap(cur, r, i, j)
                    moves.append(('intra_swap', r, i, j))
            else:
                if len(cur.routes) >= 2:
                    r1, r2 = random.sample(range(len(cur.routes)), 2)
                    if len(cur.routes[r1])>0 and len(cur.routes[r2])>0:
                        p1 = random.randrange(len(cur.routes[r1]))
                        p2 = random.randrange(len(cur.routes[r2]))
                        cur = inter_route_swap(cur, r1, p1, r2, p2)
                        moves.append(('inter_swap', r1, p1, r2, p2))
        else:
            # insertion move
            if random.random() < 0.5:
                # intra insertion if possible
                if len(cur.routes)>0:
                    r = random.randrange(len(cur.routes))
                    if len(cur.routes[r]) >= 1:
                        from_pos = random.randrange(len(cur.routes[r]))
                        to_pos = random.randrange(len(cur.routes[r]))
                        cur = intra_route_insertion(cur, r, from_pos, to_pos)
                        moves.append(('intra_ins', r, from_pos, to_pos))
            else:
                # inter insertion
                if len(cur.routes) >= 2:
                    r1, r2 = random.sample(range(len(cur.routes)), 2)
                    if len(cur.routes[r1])>0:
                        p = random.randrange(len(cur.routes[r1]))
                        inspos = random.randrange(len(cur.routes[r2])+1)
                        cur = inter_route_insertion(cur, r1, p, r2, inspos)
                        moves.append(('inter_ins', r1, p, r2, inspos))

    if debug_visual:
          ev1 = evaluate_solution(inst, cur)
          visualize_solution(inst, cur,
                            start_times=ev1.get("start_times"),
                            title="shake_after", show=True)

    return cur, moves