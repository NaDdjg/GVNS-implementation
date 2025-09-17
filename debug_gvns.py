from gvns.instance import generate_synthetic_vrptw, augment_with_synchronization
from gvns.solution import pretty_print_solution, set_deterministic
from gvns.evaluate import evaluate_solution
from gvns.local_search import gvns  

def main():
    set_deterministic(42)

    inst_dbg = generate_synthetic_vrptw(
        8,                
        tw_window=50,
        service_mean=8.0
    )
    inst_dbg = augment_with_synchronization(inst_dbg, sync_fraction=0.25, seed=42)

    print("Visits:", sorted(inst_dbg.visits.keys()), "sync_pairs:", inst_dbg.sync_pairs)

    best_sol, history = gvns(
        inst=inst_dbg,
        Tmax_seconds=60,      
        seed=42,
        verbose=True
    )

    print("\n========= FINAL BEST SOLUTION =========")
    pretty_print_solution(best_sol)
    print("Final evaluation:", evaluate_solution(inst_dbg, best_sol))
    print("History records:", len(history))

if __name__ == "__main__":
    main()
