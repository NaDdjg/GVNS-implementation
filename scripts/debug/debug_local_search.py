from gvns.instance import generate_synthetic_vrptw, augment_with_synchronization, random_insertion
from gvns.solution import pretty_print_solution, set_deterministic
from gvns.evaluate import evaluate_solution
from gvns.operators import shake_solution
from gvns.local_search import b_vnd

def main():
    set_deterministic(42)

    inst_dbg = generate_synthetic_vrptw(8, tw_window=50, service_mean=8.0)
    inst_dbg = augment_with_synchronization(inst_dbg, sync_fraction=0.25, seed=42)
    print("Visits:", sorted(inst_dbg.visits.keys()), "sync_pairs:", inst_dbg.sync_pairs)

    s0 = random_insertion(inst_dbg, num_vehicles_estimate=3, seed=42)
    print("Initial solution:")
    pretty_print_solution(s0)
    print("Eval initial:", evaluate_solution(inst_dbg, s0))

    s_shaken, sh_moves = shake_solution(inst_dbg, s0, p=4, Dp=0.4, seed=123, debug_visual=True)
    print("Shake moves:", sh_moves)
    print("Shaken solution:")
    pretty_print_solution(s_shaken)
    print("Eval shaken:", evaluate_solution(inst_dbg, s_shaken))

    s_impr, imp_flag, val, _ = b_vnd(
        inst_dbg, s_shaken,
        neighborhoods=['N1','N2','N3','N4'],
        eval_f=lambda s: evaluate_solution(inst_dbg, s)['F'],
        max_total_iters=2000,
        sample_limit=500,
        debug_visual=True
    )
    print("B-VND improved?", imp_flag, "value F:", val)
    pretty_print_solution(s_impr)
    print("Eval final:", evaluate_solution(inst_dbg, s_impr))

if __name__ == "__main__":
    main()