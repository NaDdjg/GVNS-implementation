import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from gvns.rl_env import BVNDEnv
from gvns.instance import parse_vrptws_txt  
from gvns.local_search import paper_initial_fixed

inst = parse_vrptws_txt("vrptws/100/R110_100_5.txt")
initial_sol, _ = paper_initial_fixed(inst, num_vehicles_estimate=30, seed=42)

def make_env():
    env = BVNDEnv(inst, initial_sol, Tmax=300, b_vnd_max_iters=500, max_steps=50, seed=42)
    env = Monitor(env)  
    return env

vec_env = DummyVecEnv([make_env])
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)  # feature normalization

policy_kwargs = dict(net_arch=[128, 64])  # MLP backbone

model = PPO("MlpPolicy", vec_env,
            verbose=1,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            ent_coef=0.01,
            seed=42)

# logs to check the chosen neighborhood 
obs = vec_env.reset()
action, _ = model.predict(obs, deterministic=False)
print("Action chosen:", action) 

model.learn(total_timesteps=50_000)
model.save("models/ppo_bvnd_mlp")
