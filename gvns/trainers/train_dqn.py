import os
import argparse
import numpy as np
from tqdm import trange

from gvns.rl_env import BVNDEnv
from gvns.agents.dqn_agent import DQNAgent
from gvns.local_search import paper_initial_fixed
from gvns.instance import parse_vrptws_txt

def train_dqn_on_instance(instance_path: str, num_episodes: int = 500,
                          save_path: str = "results/agents_results/dqn.pth",
                          device: str = "cpu"):
    inst = parse_vrptws_txt(instance_path)
    init_sol, _ = paper_initial_fixed(inst, num_vehicles_estimate=30, seed=42)

    env = BVNDEnv(inst, init_sol, Tmax=300, b_vnd_max_iters=500, max_steps=50)
    obs_dim = env.observation_space.shape[0]
    agent = DQNAgent(input_dim=obs_dim, n_actions=env.action_space.n, device=device,
                     lr=1e-4, buffer_size=50_000, batch_size=64,
                     target_update_freq=500, min_buffer_size=2000, start_train_steps=1000)

    total_steps = 0
    losses = []
    for ep in trange(num_episodes, desc="Episodes"):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = agent.select_action(obs, deterministic=False)
            next_obs, reward, done, info = env.step(action)
            agent.store_transition(obs, action, reward, next_obs, done)
            obs = next_obs
            ep_reward += reward
            total_steps += 1

            # perform a training step every env step once buffer has enough transitions
            if total_steps >= agent.start_train_steps and len(agent.buffer) >= agent.min_buffer_size:
                loss = agent.train_step()
                if loss is not None:
                    losses.append(loss)

        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1} reward={ep_reward:.3f} eps={agent.eps:.4f} buffer={len(agent.buffer)}")

    agent.save(save_path)
    print("Saved DQN model to", save_path)
    return agent, losses

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--out", type=str, default="results/agents_results/dqn.pth")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    train_dqn_on_instance(args.instance, num_episodes=args.episodes, save_path=args.out, device=args.device)
