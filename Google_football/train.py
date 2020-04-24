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
from ray.rllib.agents.impala.vtrace_policy import VTraceTFPolicy
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
    parser.add_argument('--checkpoint-freq', type=int, default=100)
    parser.add_argument('--num-iters', type=int, default=10000)
    parser.add_argument('--simple', action='store_true')
    parser.add_argument("--ray-num-gpus", default=1, type=int,
                        help="number of gpus to use if starting a new cluster.")
    parser.add_argument("--resume", action="store_true",
                        help="Whether to attempt to resume previous Tune experiments.")
    parser.add_argument("--experiment-name", default="default", type=str,
                        help="Name of the subdirectory under `local_dir` to put results in.")
    parser.add_argument('--policy-type', default="PPOTF",
                        type=str, help="PPO|SAC|IMPALA: agent policy type to use to train the model with.")
    return parser


def gen_policies(args):
    'Generate policies dict {policy_name: (policy_type,obs_space, act_space, {})}'
    if args.policy_type.upper() == "SAC":
        policy_type = SACTFPolicy
    elif args.policy_type.upper() == "PPO":
        policy_type = PPOTFPolicy
    elif args.policy_type.upper() == "IMPALA":
        policy_type = VTraceTFPolicy
    else:
        raise ValueError(
            "Policy is not valid. Valid policies are either: \"PPO\", \"IMPALA\", or \"SAC\"")
    policy = (policy_type, obs_space, act_space)
    agent_names = [f"agent_{agent_id}" for agent_id in range(args.num_agents)]
    policies = {policy_agent_mapping(agent_name): policy + ({"agent_id": agent_id},)
                for agent_id, agent_name in enumerate(agent_names)}
    return policies


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    ray.init(num_gpus=args.ray_num_gpus, lru_evict=True)

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
                'use_pytorch': False,
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
                'rollout_fragment_length': 50,  # NOTE: same as sample_batch_size in older versions
                "train_batch_size": 1000,
                'batch_mode': 'truncate_episodes',
                'lr': 2.5e-4,
                'log_level': 'WARN',
                'multiagent': {
                    'policies': policies,
                    'policy_mapping_fn': policy_agent_mapping,
                },
            }
        )

    elif args.policy_type.upper().startswith("IMPALA"):

        tune.run(
            'IMPALA',
            stop={'training_iteration': args.num_iters},
            checkpoint_freq=args.checkpoint_freq,
            resume=args.resume,
            config={
                # === IMPALA SPECIFIC CONFIG ===
                # V-trace params (see vtrace.py).
                "vtrace": True,
                "vtrace_clip_rho_threshold": 1.0,
                "vtrace_clip_pg_rho_threshold": 1.0,
                # System params.
                #
                # == Overview of data flow in IMPALA ==
                # 1. Policy evaluation in parallel across `num_workers` actors produces
                #    batches of size `rollout_fragment_length * num_envs_per_worker`.
                # 2. If enabled, the replay buffer stores and produces batches of size
                #    `rollout_fragment_length * num_envs_per_worker`.
                # 3. If enabled, the minibatch ring buffer stores and replays batches of
                #    size `train_batch_size` up to `num_sgd_iter` times per batch.
                # 4. The learner thread executes data parallel SGD across `num_gpus` GPUs
                #    on batches of size `train_batch_size`.
                #
                "min_iter_time_s": 10,
                # set >1 to load data into GPUs in parallel. Increases GPU memory usage
                # proportionally with the number of buffers.
                "num_data_loader_buffers": 1,
                # how many train batches should be retained for minibatching. This conf
                # only has an effect if `num_sgd_iter > 1`.
                "minibatch_buffer_size": 500,
                # number of passes to make over each train batch
                "num_sgd_iter": 10,
                # set >0 to enable experience replay. Saved samples will be replayed with
                # a p:1 proportion to new data samples.
                "replay_proportion": 0.0,
                # number of sample batches to store for replay. The number of transitions
                # saved total will be (replay_buffer_num_slots * rollout_fragment_length).
                "replay_buffer_num_slots": 0,
                # max queue size for train batches feeding into the learner
                "learner_queue_size": 16,
                # wait for train batches to be available in minibatch buffer queue
                # this many seconds. This may need to be increased e.g. when training
                # with a slow environment
                "learner_queue_timeout": 300,
                # level of queuing for sampling.
                "max_sample_requests_in_flight_per_worker": 2,
                # max number of workers to broadcast one set of weights to
                "broadcast_interval": 1,
                # use intermediate actors for multi-level aggregation. This can make sense
                # if ingesting >2GB/s of samples, or if the data requires decompression.
                "num_aggregation_workers": 0,

                # Learning params.
                "grad_clip": 40.0,
                # either "adam" or "rmsprop"
                "opt_type": "adam",
                "lr": 0.0005,
                "lr_schedule": None,
                # rmsprop considered
                "decay": 0.99,
                "momentum": 0.0,
                "epsilon": 0.1,
                # balancing the three losses
                "vf_loss_coeff": 0.5,
                "entropy_coeff": 0.01,
                "entropy_coeff_schedule": None,

                # use fake (infinite speed) sampler for testing
                "_fake_sampler": False,
                # === COMMON CONFIG ===
                'env': 'g_football',
                'num_workers': 3,
                'num_envs_per_worker': 1,
                'num_cpus_per_worker': 1,
                'num_gpus': args.ray_num_gpus,
                'rollout_fragment_length': 50,
                "train_batch_size": 1000,
                'batch_mode': 'truncate_episodes',
                'log_level': 'WARN',
                'multiagent': {
                    'policies': policies,
                    'policy_mapping_fn': policy_agent_mapping,
                },
            }
        )

    else:
        raise ValueError(f"Unsupported algorithm: \"{args.policy_type}\"")
