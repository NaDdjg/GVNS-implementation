from .evaluate import evaluate_solution
from .viz import visualize_solution

def _maybe_get_inst_and_sol_from_args(args, kwargs):
    if len(args) >= 2:
        return args[0], args[1]
    elif 'inst' in kwargs and 'sol' in kwargs:
        return kwargs['inst'], kwargs['sol']
    return None, None

def wrap_with_visualization(func_name, before=True, after=True, debug_visual=True, max_nodes=60):
    if func_name not in globals():
        print(f"[wrap] {func_name} not found — skip")
        return False
    orig = globals()[func_name]
    if not callable(orig):
        print(f"[wrap] {func_name} not callable — skip")
        return False

    def wrapped(*args, **kwargs):
        inst, sol = _maybe_get_inst_and_sol_from_args(args, kwargs)
        if debug_visual and inst and sol:
            try:
                ev = evaluate_solution(inst, sol)
            except Exception:
                ev = {}
            nodes = sum(len(r) for r in sol.routes)
            if nodes <= max_nodes and before:
                visualize_solution(inst, sol, start_times=ev.get('start_times'), title=f"{func_name}_before", show=True)
        res = orig(*args, **kwargs)
        new_sol = None
        if isinstance(res, tuple):
            for item in res:
                if hasattr(item, 'routes'):
                    new_sol = item
                    break
        elif hasattr(res, 'routes'):
            new_sol = res
        else:
            new_sol = args[1] if len(args)>=2 else kwargs.get('sol', None)

        if debug_visual and inst and new_sol and after:
            try:
                ev2 = evaluate_solution(inst, new_sol)
            except Exception:
                ev2 = {}
            nodes2 = sum(len(r) for r in new_sol.routes)
            if nodes2 <= max_nodes:
                visualize_solution(inst, new_sol, start_times=ev2.get('start_times'), title=f"{func_name}_after", show=True)
        return res

    globals()[func_name] = wrapped
    print(f"[wrap] {func_name} wrapped for visualization (before={before}, after={after})")
    return True