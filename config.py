import gymnasium as gym
import numpy as np

from ray.tune import registry
from ray.rllib.models.catalog import ModelCatalog

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

from envs.training_env import LearningCryptoEnv
from simple_transformer import SimpleTransformer


ModelCatalog.register_custom_model(
    model_name='SimpleTransformer',
    model_class=SimpleTransformer
)

registry.register_env(
    name='CryptoEnv',
    env_creator=lambda env_config: LearningCryptoEnv(**env_config)
)

ppo_config = (
    PPOConfig()
    # .rl_module(_enable_rl_module_api=False)
    .framework('torch')
    .environment(
        env='CryptoEnv',
        env_config={
            "dataset_name": "dataset",  # .npy files should be in ./data/dataset/
            "leverage": 2, # leverage for perpetual futures
            "episode_max_len": 168 * 2, # train episode length, 2 weeks
            "lookback_window_len": 168, 
            "train_start": [2000, 8000, 14000, 20000, 26000, 32000, 38000],
            "train_end": [7000, 13000, 19000, 25000, 31000, 37000, 43000], 
            "test_start": [7000, 13000, 19000, 25000, 31000, 37000, 43000],
            "test_end": [8000, 14000, 20000, 26000, 32000, 38000, 44424-1], 
            "order_size": 100, # dollars
            "initial_capital": 1000, # dollars
            "open_fee": 0.12e-2, # taker_fee
            "close_fee": 0.12e-2, # taker_fee
            "maintenance_margin_percentage": 0.012, # 1.2 percent
            "initial_random_allocated": 0, # opened initial random long/short position up to initial_random_allocated $
            "regime": "training",
            "record_stats": False, # True for backtesting
            # "cold_start_steps": 0, # do nothing at the beginning of the episode
        },
        observation_space=gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(76 * 168,),
            dtype=np.float32
        ),
        action_space=gym.spaces.Discrete(4),
    )
    .training(
        lr=5e-5,
        gamma=0.995, # 1.
        grad_clip=30,
        entropy_coeff=0.03,
        kl_coeff=0.05,
        kl_target=0.01, # not used if kl_coeff == 0.
        num_sgd_iter=10,
        use_gae=True,
        # lambda=0.95,
        clip_param=0.2, # larger values for more policy change
        vf_clip_param=10,
        train_batch_size=8 * 6 * 168, # num_rollout_workers * num_envs_per_worker * rollout_fragment_length * multiplier
        shuffle_sequences=True,
        model={
            "custom_model": "SimpleTransformer",
            "custom_model_config": {
                "embed_size": 128,
                "nhead": 4, 
                "nlayers": 3,
                "seq_len": 168,
                "dropout": 0.2,
                "cnn_enabled": False,
                "freeze_cnn": False,
            }
        }
    )
    .evaluation(
        evaluation_interval=1,
        evaluation_duration=8,
        evaluation_duration_unit='episodes',
        evaluation_parallel_to_training=False,
        evaluation_config={
            "explore": False,
            "env_config": {
                "regime": "evaluation",
                "record_stats": False, # True for backtesting
                "episode_max_len": 168 * 2, # validation episode length
                "lookback_window_len": 168, 
            }
        },
        evaluation_num_workers=4
    )
    .rollouts(
        num_rollout_workers=6,
        num_envs_per_worker=4,
        rollout_fragment_length=168,
        batch_mode='complete_episodes',
        preprocessor_pref=None
    )
    .resources(
        num_gpus=1
    )
    .debugging(
        log_level='WARN'
    )
)