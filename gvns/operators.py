from .solution import Solution
from .evaluate import evaluate_solution
from .viz import plot_side_by_side, which_nodes_changed
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

def inter_route_insertion(sol: Solution, from_route: int, pos: int,
                          to_route: int, insert_pos: int) -> Solution:
    new_sol = clone_solution(sol)
    v = new_sol.routes[from_route].pop(pos)
    new_sol.routes[to_route].insert(insert_pos, v)
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

def shake_solution(inst, sol: Solution, p:int, Dp:float=0.3, seed=None, verbose: bool = False, debug_visual: bool = False):
    """
    apply p randomized moves: with prob Dp pick a swap move (N3/N4), else insertion (N1/N2)
    returns shaken solution and applied moves
    debug_visual -> show small side-by-side plots for each move (limited)
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

    for step in range(p):
        before = clone_solution(cur)
        move = None

        if random.random() < Dp:
            # swap move: choose intra or inter swap with 50-50
            if random.random() < 0.5 and len(cur.routes) > 0:
                # intra-route swap
                r = random.randrange(len(cur.routes))
                if len(cur.routes[r]) >= 2:
                    i, j = random.sample(range(len(cur.routes[r])), 2)
                    cur = intra_route_swap(cur, r, i, j)
                    move = ('intra_swap', r, i, j)
                else:
                    move = None
            else:
                if len(cur.routes) >= 2:
                    r1, r2 = random.sample(range(len(cur.routes)), 2)
                    if len(cur.routes[r1])>0 and len(cur.routes[r2])>0:
                        p1 = random.randrange(len(cur.routes[r1]))
                        p2 = random.randrange(len(cur.routes[r2]))
                        cur = inter_route_swap(cur, r1, p1, r2, p2)
                        move = ('inter_swap', r1, p1, r2, p2)
        else:
            # insertion move
            if random.random() < 0.5 and len(cur.routes) > 0:
                # intra insertion if possible
                r = random.randrange(len(cur.routes))
                if len(cur.routes[r]) >= 2:
                    from_pos, to_pos = random.sample(range(len(cur.routes[r])), 2)
                    cur = intra_route_insertion(cur, r, from_pos, to_pos)
                    move = ('intra_ins', r, from_pos, to_pos)
            else:
                # inter insertion
                if len(cur.routes) >= 2:
                    r1, r2 = random.sample(range(len(cur.routes)), 2)
                    if len(cur.routes[r1])>0:
                        p = random.randrange(len(cur.routes[r1]))
                        inspos = random.randrange(len(cur.routes[r2])+1)
                        cur = inter_route_insertion(cur, r1, p, r2, inspos)
                        move = ('inter_ins', r1, p, r2, inspos)

        if move is not None:
            moves.append(move)
            if verbose:
                nb = {'intra_ins':'N1','inter_ins':'N2',
                      'intra_swap':'N3','inter_swap':'N4'}.get(move[0], 'UNK')
                print(f"[shake] step {step+1}/{p}: {move} (neigh={nb})")

            # visualize the move if debug and under cap
            if debug_visual and visual_count < max_visual_moves:
                diff = which_nodes_changed(before, cur)
                highlight = set(diff['added'] + diff['removed'] + [d[0] for d in diff['moved']])
                desc = f"shake step {step+1} | move={move} | nb={nb} | changed={len(highlight)}"
                savepath = plot_side_by_side(inst, before, cur,
                                             title=f"shake_move_{step+1}",
                                             highlight_nodes=highlight,
                                             move_desc=desc,
                                             show=True)
                print("Saved:", savepath)
                visual_count += 1

    if debug_visual:
        ev1 = evaluate_solution(inst, cur)
        plot_side_by_side(inst, sol, cur, title="shake_solution_after",
                          highlight_nodes=set(), move_desc=f"Total moves: {len(moves)}", show=True)

    return cur, moves