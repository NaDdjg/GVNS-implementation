from typing import Tuple, Dict, Any, List
import matplotlib.pyplot as plt
import os
from gvns.instance import Instance
from gvns.solution import Solution

VIZ_DIR = "gvns_viz"
os.makedirs(VIZ_DIR, exist_ok=True)

def visualize_solution(inst, sol, start_times=None, title=None, savepath=None, show=True):
    """
    Lightweight visualizer — does NOT use networkx. Much lower memory footprint.
    inst: instance with depot.x/y and inst.visits[v].x/.y
    sol: solution with sol.routes
    start_times: optional dict vid->start
    """
    dpi = 100
    fig, ax = plt.subplots(figsize=(7, 6), dpi=dpi)

    # depot coords
    dx, dy = inst.depot.x, inst.depot.y
    ax.scatter([dx], [dy], s=120, marker='s', label='Depot', zorder=5)
    ax.text(dx, dy, " depot", fontsize=8)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for k, route in enumerate(sol.routes):
        if len(route) == 0:
            continue
        xs = [dx]  # start at depot
        ys = [dy]
        for v in route:
            vobj = inst.visits[v]
            xs.append(vobj.x)
            ys.append(vobj.y)
        xs.append(dx)  # return
        ys.append(dy)
        color = colors[k % len(colors)]
        ax.plot(xs, ys, linewidth=1.5, label=f"veh {k}", zorder=3)
        ax.scatter(xs[1:-1], ys[1:-1], s=35, zorder=4)
        # annotate node ids (small)
        for i, vid in enumerate(route):
            tx = inst.visits[vid].x
            ty = inst.visits[vid].y
            lab = str(vid)
            if start_times and vid in start_times:
                lab += f"\n{start_times[vid]:.0f}"
            ax.text(tx, ty, lab, fontsize=7, zorder=6)

    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    if title:
        ax.set_title(title)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(False)

    if savepath is None:
        savepath = f"{VIZ_DIR}/viz_{title or 'sol'}.png"
    plt.savefig(savepath, bbox_inches='tight', dpi=120)
    if show:
        plt.show()
    plt.close(fig)
    return savepath

# small helper: which nodes changed from sol_a -> sol_b
def which_nodes_changed(sol_a: Solution, sol_b: Solution) -> Dict[str, Any]:
    """
    Compare two solutions and report node differences.
    Always returns keys: 'added', 'removed', 'moved' (lists).
    """
    map_a = {}
    for ridx, route in enumerate(sol_a.routes):
        for pos, vid in enumerate(route):
            map_a[vid] = (ridx, pos)
    map_b = {}
    for ridx, route in enumerate(sol_b.routes):
        for pos, vid in enumerate(route):
            map_b[vid] = (ridx, pos)

    vids_a = set(map_a)
    vids_b = set(map_b)

    added = sorted(vids_b - vids_a)
    removed = sorted(vids_a - vids_b)

    moved = []
    for vid in vids_a & vids_b:
        ra, pa = map_a[vid]
        rb, pb = map_b[vid]
        if ra != rb or pa != pb:
            moved.append((vid, ra, rb, pa, pb))

    # ensure all keys exist
    return {
        "added": added,
        "removed": removed,
        "moved": moved
    }

# small side-by-side plotter (keeps style similar to visualize_solution)
def plot_side_by_side(inst: Instance, sol_a: Solution, sol_b: Solution,
                      title: str = "move", highlight_nodes: set = None,
                      move_desc: str = None, savepath=None, show=True):
    """
    Draw two subplots: left = sol_a, right = sol_b.
    Uses the same plotting style as visualize_solution.
    """
    if highlight_nodes is None:
        highlight_nodes = set()

    dpi = 90
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=dpi)
    all_axes = axes.flatten()
    depot_x, depot_y = inst.depot.x, inst.depot.y
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    def draw_one(ax, sol, subtitle):
        ax.scatter([depot_x], [depot_y], s=100, marker='s', label='Depot', zorder=5)
        ax.text(depot_x, depot_y, " depot", fontsize=8)
        for k, route in enumerate(sol.routes):
            if len(route) == 0:
                continue
            xs = [depot_x]
            ys = [depot_y]
            for v in route:
                vobj = inst.visits[v]
                xs.append(vobj.x); ys.append(vobj.y)
            xs.append(depot_x); ys.append(depot_y)
            color = colors[k % len(colors)]
            ax.plot(xs, ys, linewidth=1.5, label=f"veh {k}", zorder=3)
            ax.scatter(xs[1:-1], ys[1:-1], s=35, zorder=4)
            for vid in route:
                tx = inst.visits[vid].x; ty = inst.visits[vid].y
                lab = str(vid)
                if vid in highlight_nodes:
                    ax.scatter([tx], [ty], s=120, edgecolor='k', linewidth=1.4, zorder=6)
                ax.text(tx, ty, lab, fontsize=7, zorder=6)
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.set_title(subtitle)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(False)

    draw_one(all_axes[0], sol_a, f"{title} (before)")
    draw_one(all_axes[1], sol_b, f"{title} (after)")

    if move_desc:
        fig.suptitle(move_desc, fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if savepath is None:
        os.makedirs("gvns_viz", exist_ok=True)
        savepath = f"gvns_viz/{title.replace(' ','_')}.png"
    plt.savefig(savepath, bbox_inches='tight', dpi=110)
    if show:
        plt.show()
    plt.close(fig)
    return savepath