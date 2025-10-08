import argparse
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from gvns.rl_env import BVNDEnv
from gvns.instance import parse_vrptws_txt
from gvns.local_search import paper_initial_fixed
import os

def main(args):
    # load instance
    inst = parse_vrptws_txt(args.instance)
    initial_sol, _ = paper_initial_fixed(inst, num_vehicles_estimate=args.vehicles, seed=args.seed)

    def make_env():
        env = BVNDEnv(
            inst,
            initial_sol,
            Tmax=args.tmax,
            b_vnd_max_iters=args.bvnd_max_iters,
            max_steps=args.max_steps,
            seed=args.seed
        )
        env = Monitor(env)
        return env

    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

    policy_kwargs = dict(net_arch=[128, 64])  # MLP backbone

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        learning_rate= 3e-4,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
        seed=args.seed
    )

    # train
    print(f"Training PPO on {args.instance} for {args.timesteps} timesteps")
    model.learn(total_timesteps=args.timesteps)

    # save 
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    model.save(args.save_path)
    print(f"Model saved to {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--instance", type=str, required=True, help="Path to VRPTW instance file")
    parser.add_argument("--save_path", type=str, default="models/ppo_bvnd_mlp", help="Where to save the model")

    parser.add_argument("--timesteps", type=int, default=50000, help="Total timesteps to train")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--n_steps", type=int, default=1024, help="Number of steps to run per update")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="Entropy coefficient")

    parser.add_argument("--vehicles", type=int, default=30, help="Estimated number of vehicles")
    parser.add_argument("--tmax", type=int, default=300, help="Maximum time for the environment")
    parser.add_argument("--bvnd_max_iters", type=int, default=500, help="Max iterations for B-VND")
    parser.add_argument("--max_steps", type=int, default=50, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    main(args)
