from .solution import Solution
from .evaluate import evaluate_solution
from .viz import plot_side_by_side, which_nodes_changed
import random

def clone_solution(sol: Solution) -> Solution:
    return Solution(routes=[list(r) for r in sol.routes])

def two_opt(sol: Solution, route_idx: int, i: int, k: int) -> Solution:
    """Reverse segment i..k inclusive (i < k)."""
    if i >= k:
        return sol
    new = clone_solution(sol)
    r = new.routes[route_idx]
    # safety
    if k >= len(r) or i < 0:
        return new
    r[i:k+1] = list(reversed(r[i:k+1]))
    return new

def or_opt(sol: Solution, from_route:int, i:int, block_size:int, to_route:int, j:int) -> Solution:
    """Remove block [i,i+block_size) from from_route and insert at position j in to_route."""
    new = clone_solution(sol)
    block = new.routes[from_route][i:i+block_size]
    # remove block
    del new.routes[from_route][i:i+block_size]
    # inserting in same route and i < j needs index adjustment
    if from_route == to_route and j > i:
        j -= block_size
    for k, v in enumerate(block):
        new.routes[to_route].insert(j+k, v)
    # optionally remove empty route
    if len(new.routes[from_route]) == 0:
        new.routes.pop(from_route)
    return new

def cross_exchange(sol: Solution, r1:int, p1:int, r2:int, p2:int) -> Solution:
    new = clone_solution(sol)
    tail1 = new.routes[r1][p1+1:]
    tail2 = new.routes[r2][p2+1:]
    new.routes[r1] = new.routes[r1][:p1+1] + tail2
    new.routes[r2] = new.routes[r2][:p2+1] + tail1
    return new

def intra_route_insertion(sol: Solution, route_idx: int, from_pos: int, to_pos: int) -> Solution:
    if from_pos == to_pos:
        return sol
    new_sol = clone_solution(sol)
    route = new_sol.routes[route_idx]
    if len(route) < 2:
        return new_sol
    v = route.pop(from_pos)
    if to_pos > from_pos:
        to_pos -= 1
    route.insert(to_pos, v)
    return new_sol

def inter_route_insertion(sol: Solution, from_route: int, pos: int, to_route: int, insert_pos: int) -> Solution:
    if from_route == to_route:
        return sol
    new_sol = clone_solution(sol)
    if len(new_sol.routes[from_route]) <= 1:
        return new_sol
    v = new_sol.routes[from_route].pop(pos)
    new_sol.routes[to_route].insert(insert_pos, v)
    # remove route if empty
    if len(new_sol.routes[from_route]) == 0:
        new_sol.routes.pop(from_route)
    return new_sol

def intra_route_swap(sol: Solution, route_idx: int, pos1: int, pos2: int) -> Solution:
    new_sol = clone_solution(sol)
    r = new_sol.routes[route_idx]
    r[pos1], r[pos2] = r[pos2], r[pos1]
    return new_sol

def inter_route_swap(sol: Solution, r1:int, p1:int, r2:int, p2:int) -> Solution:
    if r1 == r2:
        return sol
    new_sol = clone_solution(sol)
    a = new_sol.routes[r1][p1]
    b = new_sol.routes[r2][p2]
    new_sol.routes[r1][p1] = b
    new_sol.routes[r2][p2] = a
    return new_sol

def shake_solution(inst, sol: Solution, p:int, Dp:float=0.3, seed=None, verbose: bool = False, debug_visual: bool = False):
    """
    apply p randomized moves. With prob Dp pick a swap-like move (intra/inter),
    otherwise pick insertion-like moves. Additionally sometimes apply stronger
    structural moves (2-opt, or-opt, cross-exchange) to jump basins.

    Returns (shaken_solution, moves_list)
    """
    from .instance import Instance
    assert isinstance(inst, Instance)

    if seed is not None:
        random.seed(seed)

    cur = clone_solution(sol)
    moves = []

    # Visualize whole sol before if requested
    if debug_visual:
        plot_side_by_side(inst, sol, sol, title="shake_solution_before",
                          highlight_nodes=set(), move_desc="initial", show=True)

    max_visual_moves = 6
    visual_count = 0

    # helper small wrappers to call new operators if available
    def try_two_opt(sol_):
        # choose a random route with length>=4
        cand = sol_
        routes_idx = [i for i, r in enumerate(sol_.routes) if len(r) >= 4]
        if not routes_idx:
            return sol_, None
        r = random.choice(routes_idx)
        L = len(sol_.routes[r])
        i = random.randrange(0, L-2)
        k = random.randrange(i+1, min(L-1, i+4))  # small 2-opt segment
        return two_opt(sol_, r, i, k), ('two_opt', r, i, k)

    def try_or_opt(sol_):
        # choose random block size 1..3, a source route and target route
        cand = sol_
        non_empty = [i for i,r in enumerate(sol_.routes) if len(r) > 0]
        if len(non_empty) == 0:
            return sol_, None
        fr = random.choice(non_empty)
        block_size = random.choice([1, 2, 3])
        if len(sol_.routes[fr]) < block_size: 
            return sol_, None
        i = random.randrange(0, len(sol_.routes[fr]) - block_size + 1)
        # pick target route (could be same)
        tr = random.randrange(len(sol_.routes))
        j = random.randrange(0, len(sol_.routes[tr]) + 1)
        return or_opt(sol_, fr, i, block_size, tr, j), ('or_opt', fr, i, block_size, tr, j)

    def try_cross(sol_):
        # swap tails after positions
        if len(sol_.routes) < 2:
            return sol_, None
        r1, r2 = random.sample(range(len(sol_.routes)), 2)
        if len(sol_.routes[r1]) < 2 or len(sol_.routes[r2]) < 2:
            return sol_, None
        p1 = random.randrange(len(sol_.routes[r1]) - 1)
        p2 = random.randrange(len(sol_.routes[r2]) - 1)
        return cross_exchange(sol_, r1, p1, r2, p2), ('cross', r1, p1, r2, p2)

    # main shake loop
    for step in range(p):
        before = clone_solution(cur)
        move = None

        # occasionally do a stronger structural move (small prob)
        strong_roll = random.random()
        if strong_roll < 0.12:
            # priority: 2-opt > or-opt > cross
            cur, move = try_two_opt(cur)
            if move is None: 
                cur, move = try_or_opt(cur)
            if move is None:
                cur, move = try_cross(cur)
        else:
            # standard shake using swap/insertion logic but with some improvements
            if random.random() < Dp:
                # swap-like: intra or inter
                if random.random() < 0.5 and len(cur.routes) > 0:
                    r = random.randrange(len(cur.routes))
                    if len(cur.routes[r]) >= 2:
                        i, j = random.sample(range(len(cur.routes[r])), 2)
                        cur = intra_route_swap(cur, r, i, j)
                        move = ('intra_swap', r, i, j)
                else:
                    if len(cur.routes) >= 2:
                        r1, r2 = random.sample(range(len(cur.routes)), 2)
                        if len(cur.routes[r1])>0 and len(cur.routes[r2])>0:
                            p1 = random.randrange(len(cur.routes[r1]))
                            p2 = random.randrange(len(cur.routes[r2]))
                            # 25% chance of cross-exchange instead of simple swap
                            if random.random() < 0.25:
                                maybe, mv = try_cross(cur)
                                if mv is not None:
                                    cur, move = maybe, mv
                                else:
                                    cur = inter_route_swap(cur, r1, p1, r2, p2)
                                    move = ('inter_swap', r1, p1, r2, p2)
                            else:
                                cur = inter_route_swap(cur, r1, p1, r2, p2)
                                move = ('inter_swap', r1, p1, r2, p2)
            else:
                # insertion-like
                if random.random() < 0.5 and len(cur.routes) > 0:
                    # intra insertion if possible
                    r = random.randrange(len(cur.routes))
                    if len(cur.routes[r]) >= 2:
                        from_pos, to_pos = random.sample(range(len(cur.routes[r])), 2)
                        cur = intra_route_insertion(cur, r, from_pos, to_pos)
                        move = ('intra_ins', r, from_pos, to_pos)
                else:
                    # inter insertion or or-opt
                    if len(cur.routes) >= 2:
                        if random.random() < 0.15:
                            cand, mv = try_or_opt(cur)
                            if mv is not None:
                                cur, move = cand, mv
                        else:
                            r1, r2 = random.sample(range(len(cur.routes)), 2)
                            if len(cur.routes[r1])>0:
                                p = random.randrange(len(cur.routes[r1]))
                                inspos = random.randrange(len(cur.routes[r2])+1)
                                cur = inter_route_insertion(cur, r1, p, r2, inspos)
                                move = ('inter_ins', r1, p, r2, inspos)

        if move is not None:
            moves.append(move)
            if verbose:
                # best-effort neighborhood label
                nb = move[0]
                print(f"[shake] step {step+1}/{p}: {move} (nb={nb})")

            if debug_visual and visual_count < max_visual_moves:
                diff = which_nodes_changed(before, cur)
                highlight = set(diff['added'] + diff['removed'] + [d[0] for d in diff['moved']])
                desc = f"shake step {step+1} | move={move} | changed={len(highlight)}"
                savepath = plot_side_by_side(inst, before, cur,
                                             title=f"shake_move_{step+1}",
                                             highlight_nodes=highlight,
                                             move_desc=desc,
                                             show=True)
                if verbose:
                    print("Saved:", savepath)
                visual_count += 1

    if debug_visual:
        ev1 = evaluate_solution(inst, cur)
        plot_side_by_side(inst, sol, cur, title="shake_solution_after",
                          highlight_nodes=set(), move_desc=f"Total moves: {len(moves)}", show=True)

    return cur, moves