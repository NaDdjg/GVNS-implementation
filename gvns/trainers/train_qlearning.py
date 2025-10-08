import os
import time
import argparse
import numpy as np

from gvns.rl_env import BVNDEnv
from gvns.agents.qlearning_agent import QTableAgent
from gvns.local_search import paper_initial_fixed
from gvns.instance import parse_vrptws_txt

def train_on_instance(instance_path: str, num_episodes: int = 500, save_path: str = "models/qlearning.pkl"):
    inst = parse_vrptws_txt(instance_path)
    init_sol, _ = paper_initial_fixed(inst, num_vehicles_estimate=30, seed=42)

    env = BVNDEnv(inst, init_sol, Tmax=300, b_vnd_max_iters=500, max_steps=50)
    agent = QTableAgent(n_actions=4, alpha=0.1, gamma=0.99, eps_start=0.2, eps_min=0.01, eps_decay=1e-4)

    stats = []
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        t0 = time.time()
        while not done:
            a = agent.select_action(obs)
            next_obs, reward, done, info = env.step(a)
            agent.update(obs, a, reward, next_obs, done)
            obs = next_obs
            total_reward += reward
        stats.append(total_reward)
        if (ep+1) % 10 == 0:
            print(f"Episode {ep+1}/{num_episodes}  total_reward={total_reward:.3f}  eps={agent.eps:.4f}")
    # save q-table
    agent.save(save_path)
    print("Saved Q-table to", save_path)
    return agent, stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--out", type=str, default="models/qlearning.pkl")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    agent, stats = train_on_instance(args.instance, num_episodes=args.episodes, save_path=args.out)
