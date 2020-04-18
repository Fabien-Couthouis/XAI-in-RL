from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import argparse
import gfootball.env as football_env
import gym
import ray
import numpy as np
from ray import tune
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.tune.config_parser import make_parser
from experiments.RllibGFootball import RllibGFootball, policy_agent_mapping
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy

from ray.rllib.agents.sac.sac_policy import SACTFPolicy


# Try to import both backends for flag checking/warnings.
tf = try_import_tf()
torch, _ = try_import_torch()
CHECKPOINT_PATH = "./multiagent-checkpoint"
EXAMPLE_USAGE = """
Training example via RLlib CLI:
    rllib train --run DQN --env CartPole-v0
Grid search example via RLlib CLI:
    rllib train -f tuned_examples/cartpole-grid-search-example.yaml
Grid search example via executable:
    ./train.py -f tuned_examples/cartpole-grid-search-example.yaml
Note that -f overrides all other trial-specific command-line options.
"""


def create_parser(parser_creator=None):
    parser = make_parser(
        parser_creator=parser_creator,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Train a reinforcement learning agent.",
        epilog=EXAMPLE_USAGE)
    parser.add_argument(
        "--scenario-name", default=None, help="Change scenario name.")
    parser.add_argument('--num-agents', type=int, default=None)

    parser.add_argument('--num-iters', type=int, default=10000)
    parser.add_argument('--simple', action='store_true')
    parser.add_argument(
        "--ray-num-gpus",
        default=1,
        type=int,
        help="--num-gpus to use if starting a new cluster.")

    parser.add_argument("--resume", action="store_true",
                        help="Whether to attempt to resume previous Tune experiments.")

    parser.add_argument("--experiment-name", default="default", type=str,
                        help="Name of the subdirectory under `local_dir` to put results in.")
    parser.add_argument('--policy-type', default="PPOTF",
                        type=str, help="PPOTF|PPOTORCH|SACTF: agent policy type to use to train the model with.")
    return parser



def gen_policies(args):
    'Generate policies dict {policy_name: (policy_type,obs_space, act_space, {})}'
    if args.policy_type.upper() == "SACTF":
        policy_type = SACTFPolicy
    elif args.policy_type.upper() == "PPOTF":
        policy_type = PPOTFPolicy
    elif args.policy_type.upper() == "PPOTORCH":
        policy_type = PPOTorchPolicy
    else:
        raise ValueError(
            "Policy is not valid. Valid policies are either: \"PPO\" or \"SAC\"")
    policy = (policy_type, obs_space, act_space, {})
    agent_names = [f"agent_{agent_id}" for agent_id in range(args.num_agents)]
    policies = {policy_agent_mapping(agent_name): policy
                for agent_name in agent_names}
    return policies


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    ray.init(num_gpus=args.ray_num_gpus)

    register_env('g_football', lambda _: RllibGFootball(
        args.num_agents, args.scenario_name, render=False))

    # Get obs/act spaces to feed the policy with
    single_env = RllibGFootball(
        args.num_agents, args.scenario_name, render=False)
    obs_space = single_env.observation_space
    act_space = single_env.action_space

    policies = gen_policies(args)

    if args.policy_type.upper().startswith("PPO"):

        tune.run(
            'PPO',
            stop={'training_iteration': args.num_iters},
            checkpoint_freq=args.checkpoint_freq,
            resume=args.resume,
            config={
                # === PPO SPECIFIC CONFIG ===
                'lambda': 0.95,
                'kl_coeff': 0.2,
                'clip_rewards': False,
                'vf_clip_param': 10.0,
                'entropy_coeff': 0.01,
                'sgd_minibatch_size': 500,
                'num_sgd_iter': 10,
                'use_pytorch': args.policy_type.upper().endswith("TORCH"),
                'observation_filter': 'NoFilter',
                'vf_share_layers': True,
                'simple_optimizer': args.simple,
                # === COMMON CONFIG ===
                'env': 'g_football',
                'train_batch_size': 2000,
                'rollout_fragment_length': 100,  # NOTE: same as sample_batch_size in older versions
                'num_workers': 3,
                'num_envs_per_worker': 1,
                'num_cpus_per_worker': 1,
                'batch_mode': 'truncate_episodes',
                'num_gpus': args.ray_num_gpus,
                'lr': 2.5e-4,
                'log_level': 'WARN',
                'multiagent': {
                    'policies': policies,
                    'policy_mapping_fn': policy_agent_mapping,
                },
            },
        )

    elif args.policy_type.upper().startswith("SAC"):

        tune.run(
            'SAC',
            stop={'training_iteration': args.num_iters},
            checkpoint_freq=args.checkpoint_freq,
            resume=args.resume,
            config={
                # === SAC SPECIFIC CONFIG ===
                # === Model ===
                "twin_q": True,
                "use_state_preprocessor": False,
                # RLlib model options for the Q function
                "Q_model": {
                    "hidden_activation": "relu",
                    "hidden_layer_sizes": (256, 256),
                },
                # RLlib model options for the policy function
                "policy_model": {
                    "hidden_activation": "relu",
                    "hidden_layer_sizes": (256, 256),
                },
                # Unsquash actions to the upper and lower bounds of env's action space.
                # Ignored for discrete action spaces.
                "normalize_actions": False,
                # Disable setting done=True at end of episode. This should be set to True
                # for infinite-horizon MDPs (e.g., many continuous control problems).
                "no_done_at_end": False,
                # Update the target by \tau * policy + (1-\tau) * target_policy.
                "tau": 5e-3,
                # Initial value to use for the entropy weight alpha.
                "initial_alpha": 1.0,
                # Target entropy lower bound. If "auto", will be set to -|A| (e.g. -2.0 for
                # Discrete(2), -3.0 for Box(shape=(3,))).
                # This is the inverse of reward scale, and will be optimized automatically.
                "target_entropy": "auto",
                # N-step target updates.
                "n_step": 1,
                # === Optimization ===
                "optimization": {
                    "actor_learning_rate": 3e-4,
                    "critic_learning_rate": 3e-4,
                    "entropy_learning_rate": 3e-4,
                },
                # === COMMON CONFIG ===
                'env': 'g_football',
                'num_workers': 3,
                'num_envs_per_worker': 1,
                'num_cpus_per_worker': 1,
                'num_gpus': args.ray_num_gpus,
                'rollout_fragment_length': 100,  # NOTE: same as sample_batch_size in older versions
                "train_batch_size": 2000,
                'batch_mode': 'truncate_episodes',
                'lr': 2.5e-4,
                'log_level': 'WARN',
                'multiagent': {
                    'policies': policies,
                    'policy_mapping_fn': policy_agent_mapping,
                },
            }
        )
