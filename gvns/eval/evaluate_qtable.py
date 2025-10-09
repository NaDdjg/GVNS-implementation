from gvns.rl_env import BVNDEnv
from gvns.local_search import paper_initial_fixed, clone_solution, evaluate_solution
from gvns.instance import parse_vrptws_txt
from gvns.agents.qlearning_agent import QTableAgent

inst = parse_vrptws_txt("")
init_sol, _ = paper_initial_fixed(inst, num_vehicles_estimate=30, seed=1)
env = BVNDEnv(inst, init_sol, Tmax=600, b_vnd_max_iters=200, max_steps=100)

agent = QTableAgent()
agent.load("results/qtable.pkl")

obs = env.reset()
done = False
while not done:
    a = agent.select_action(obs) 
    obs, reward, done, info = env.step(a)
print("Final F:", env.eval['F'])
