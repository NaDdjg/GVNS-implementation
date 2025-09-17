import matplotlib.pyplot as plt
import os

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