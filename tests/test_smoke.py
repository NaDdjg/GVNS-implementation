def test_compute_schedule_and_eval():
    from gvns.instance import Instance, Visit
    from gvns.solution import Solution
    from gvns.evaluate import compute_route_schedule, evaluate_solution

    # tiny instance: depot + 1 visit
    depot = Visit(id=0, x=0.0, y=0.0, service=0.0, tw_early=0.0, tw_late=1000)
    v1 = Visit(id=1, x=1.0, y=0.0, service=0.0, tw_early=0.0, tw_late=1000)
    inst = Instance(name="tiny", depot=depot)
    inst.visits = {1: v1}
    # minimal travel matrix
    inst.distance_matrix = {0:{0:0,1:1}, 1:{0:1,1:0}}

    sol = Solution(routes=[[1]])
    sched, total = compute_route_schedule(inst, sol.routes[0])
    assert total >= 0
    stats = evaluate_solution(inst, sol)
    assert 'F' in stats
