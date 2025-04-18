import ray
from ray import tune
from config import ppo_config # for One-way strategy
# from config_long import ppo_config # for Long only strategy
import os 
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "envs"))

ray.shutdown()
ray.init()

tune.run(
    "PPO",
    stop={"timesteps_total": int(1e10)},
    config=ppo_config,
    local_dir="./results", # default folder "~ray_results" 
    checkpoint_freq=12,
    checkpoint_at_end=False,
    keep_checkpoints_num=None,
    verbose=2,
    reuse_actors=False,
    # resume=True,
    # restore="C:/Users/tmpou/ray_results/PPO_2025-04-16_17-59-27/PPO_CryptoEnv_64a14_00000_0_2025-04-16_17-59-27/checkpoint_000007"
)