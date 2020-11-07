from __future__ import absolute_import, division, print_function

import argparse

import gym
import os
import numpy as np
from gym.spaces import Box, Dict, Discrete, MultiDiscrete, Tuple

import ray
from experiments.kaz_wrapper import ParallelPettingZooEnv

from ray import tune
from ray.rllib.env.multi_agent_env import ENV_STATE, MultiAgentEnv
from ray.rllib.utils.framework import try_import_tf
from ray.tune.config_parser import make_parser
from ray.tune.registry import register_env

# Try to import both backends for flag checking/warnings.
tf = try_import_tf()
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
    parser.add_argument('--num-iters', type=int, default=100)
    parser.add_argument('--simple', action='store_true')
    parser.add_argument("--resume", action="store_true",
                        help="Whether to attempt to resume previous Tune experiments.")
    parser.add_argument("--experiment-name", default="default", type=str,
                        help="Name of the subdirectory under `local_dir` to put results in.")
    return parser


def gen_policies(obs_space, act_space, agent_names, qmix=False):
    'Generate policies dict {policy_name: (policy_type, obs_space, act_space, {"agent_id": i})}'
    policy = (None, obs_space, act_space)
    policies = {policy_agent_mapping(agent_name): policy + ({"agent_id": agent_id},)
                for agent_id, agent_name in enumerate(agent_names)}
    return policies


def policy_agent_mapping(agent_name):
    'Maps agent name to policy name'
    return f"policy_{agent_name}"


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    ray.init(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))

    # Get obs/act spaces to feed the policy with
    single_env = ParallelPettingZooEnv()

    obs_space = single_env.observation_space
    act_space = single_env.action_space

    agent_names = single_env.reset().keys()
    archers_list = single_env.archers_list
    knights_list = single_env.knights_list
    num_agents = single_env.num_agents

    if args.run.upper() == "QMIX":
        # Qmix spaces
        obs_space = Box(low=0, high=255, shape=(
                        512, 512, 3), dtype=np.uint8)
        obs_space = Tuple([
            Dict({
                "obs": obs_space,
                ENV_STATE: obs_space
            }) for _ in range(num_agents)
        ])
        act_space = Tuple([
            act_space for _ in range(num_agents)
        ])

    single_env.close()

    policies = gen_policies(obs_space, act_space, agent_names=agent_names)

    if args.run.upper() == "PPO":

        config = {
            # === PPO SPECIFIC CONFIG ===
            # Should use a critic as a baseline (otherwise don't use value baseline;
            # required for using GAE).
            "use_critic": True,
            # If true, use the Generalized Advantage Estimator (GAE)
            # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
            "use_gae": True,
            # The GAE(lambda) parameter.
            "lambda": 1.0,
            # Initial coefficient for KL divergence.
            "kl_coeff": 0.2,
            # Size of batches collected from each worker.
            "rollout_fragment_length": 200,
            # Number of timesteps collected for each SGD round. This defines the size
            # of each SGD epoch.
            "train_batch_size": 2000,
            # Total SGD batch size across all devices for SGD. This defines the
            # minibatch size within each epoch.
            "sgd_minibatch_size": 128,
            # Whether to shuffle sequences in the batch when training (recommended).
            "shuffle_sequences": True,
            # Number of SGD iterations in each outer loop (i.e., number of epochs to
            # execute per train batch).
            "num_sgd_iter": 30,
            # Stepsize of SGD.
            "lr": 5e-5,
            # Learning rate schedule.
            "lr_schedule": None,
            # Share layers for value function. If you set this to True, it's important
            # to tune vf_loss_coeff.
            "vf_share_layers": False,
            # Coefficient of the value function loss. IMPORTANT: you must tune this if
            # you set vf_share_layers: True.
            "vf_loss_coeff": 1.0,
            # Coefficient of the entropy regularizer.
            "entropy_coeff": 0.0,
            # Decay schedule for the entropy regularizer.
            "entropy_coeff_schedule": None,
            # PPO clip parameter.
            "clip_param": 0.3,
            # Clip param for the value function. Note that this is sensitive to the
            # scale of the rewards. If your expected V is large, increase this.
            "vf_clip_param": 10.0,
            # If specified, clip the global norm of gradients by this amount.
            "grad_clip": None,
            # Target value for KL divergence.
            "kl_target": 0.01,
            # Whether to rollout "complete_episodes" or "truncate_episodes".
            "batch_mode": "truncate_episodes",
            # Which observation filter to apply to the observation.
            "observation_filter": "NoFilter",
            # Uses the sync samples optimizer instead of the multi-gpu one. This is
            # usually slower, but you might want to try it if you run into issues with
            # the default optimizer.
            "simple_optimizer": False,
            # Whether to fake GPUs (using CPUs).
            # Set this to True for debugging on non-GPU machines (set `num_gpus` > 0).
            "_fake_gpus": False,
            # Switch on Trajectory View API for PPO by default.
            # NOTE: Only supported for PyTorch so far.
            "_use_trajectory_view_api": False,

            "model": {"dim": 168, "conv_filters":
                      [[16, [16, 16], 8],
                       [32, [4, 4], 2],
                          [256, [11, 11], 1],
                       ]},
            # === COMMON CONFIG ===
            'num_workers': 3,
            'num_envs_per_worker': 1,
            'num_cpus_per_worker': 1,
            # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
            "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            # "framework": "TF" if args.torch else "tf",
            'log_level': 'WARN',
            'multiagent': {
                'policies': policies,
                'policy_mapping_fn': policy_agent_mapping,
            },
        }

    elif args.run.upper() == "CONTRIB/MADDPG":

        config = {

            # === Framework to run the algorithm ===
            "framework": "tf",

            # === Settings for each individual policy ===
            # ID of the agent controlled by this policy
            "agent_id": None,
            # Use a local critic for this policy.
            "use_local_critic": False,

            # === Evaluation ===
            # Evaluation interval
            "evaluation_interval": None,
            # Number of episodes to run per evaluation period.
            "evaluation_num_episodes": 10,

            # === Model ===
            # Apply a state preprocessor with spec given by the "model" config option
            # (like other RL algorithms). This is mostly useful if you have a weird
            # observation shape, like an image. Disabled by default.
            "use_state_preprocessor": True,
            # Postprocess the policy network model output with these hidden layers. If
            # use_state_preprocessor is False, then these will be the *only* hidden
            # layers in the network.
            "actor_hiddens": [64, 64],
            # Hidden layers activation of the postprocessing stage of the policy
            # network
            "actor_hidden_activation": "relu",
            # Postprocess the critic network model output with these hidden layers;
            # again, if use_state_preprocessor is True, then the state will be
            # preprocessed by the model specified with the "model" config option first.
            "critic_hiddens": [64, 64],
            # Hidden layers activation of the postprocessing state of the critic.
            "critic_hidden_activation": "relu",
            # N-step Q learning
            "n_step": 1,
            # Algorithm for good policies.
            "good_policy": "maddpg",
            # Algorithm for adversary policies.
            "adv_policy": "maddpg",

            # === Replay buffer ===
            # Size of the replay buffer. Note that if async_updates is set, then
            # each worker will have a replay buffer of this size.
            "buffer_size": int(1e6),
            # Observation compression. Note that compression makes simulation slow in
            # MPE.
            "compress_observations": False,
            # If set, this will fix the ratio of replayed from a buffer and learned on
            # timesteps to sampled from an environment and stored in the replay buffer
            # timesteps. Otherwise, the replay will proceed at the native ratio
            # determined by (train_batch_size / rollout_fragment_length).
            "training_intensity": None,

            # === Optimization ===
            # Learning rate for the critic (Q-function) optimizer.
            "critic_lr": 1e-2,
            # Learning rate for the actor (policy) optimizer.
            "actor_lr": 1e-2,
            # Update the target network every `target_network_update_freq` steps.
            "target_network_update_freq": 0,
            # Update the target by \tau * policy + (1-\tau) * target_policy
            "tau": 0.01,
            # Weights for feature regularization for the actor
            "actor_feature_reg": 0.001,
            # If not None, clip gradients during optimization at this value
            "grad_norm_clipping": 0.5,
            # How many steps of the model to sample before learning starts.
            "learning_starts": 1024 * 25,
            # Update the replay buffer with this many samples at once. Note that this
            # setting applies per-worker if num_workers > 1.
            "rollout_fragment_length": 100,
            # Size of a batched sampled from replay buffer for training. Note that
            # if async_updates is set, then each worker returns gradients for a
            # batch of this size.
            "train_batch_size": 16,
            # Number of env steps to optimize for before returning
            "timesteps_per_iteration": 0,
            # Prevent iterations from going lower than this time span
            "min_iter_time_s": 0,


            "model": {"dim": 168, "conv_filters":
                      [[16, [16, 16], 8],
                       [32, [4, 4], 2],
                       [256, [11, 11], 1],
                       ]},

            # === COMMON CONFIG ===
            "env_config": {
                "actions_are_logits": True,
            },
            'num_workers': 1,
            'num_envs_per_worker': 1,
            'num_cpus_per_worker': 3,
            'num_gpus': int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            'batch_mode': 'truncate_episodes',
            'log_level': 'DEBUG',

            'multiagent': {
                'policies': policies,
                'policy_mapping_fn': policy_agent_mapping,
                # "replay_mode": "lockstep",

            },
        }

    elif args.run == "QMIX":
        grouping = {
            "archers": archers_list+knights_list,
            # "knights": knights_list,
        }

        print("obs_space", obs_space)
        print("act_space", act_space)

        # Create and register google football env

        def create_env_qmix(_): return ParallelPettingZooEnv(config={"with_state": True}).with_agent_groups(grouping,
                                                                                                            obs_space=obs_space,
                                                                                                            act_space=act_space)

        register_env('kaz_qmix', create_env_qmix)

        config = {

            # === QMix ===
            # Mixing network. Either "qmix", "vdn", or None
            "mixer": "qmix",
            # Size of the mixing network embedding
            "mixing_embed_dim": 32,
            # Whether to use Double_Q learning
            "double_q": True,
            # Optimize over complete episodes by default.
            "batch_mode": "complete_episodes",

            # === Evaluation ===
            # Evaluate with epsilon=0 every `evaluation_interval` training iterations.
            # The evaluation stats will be reported under the "evaluation" metric key.
            # Note that evaluation is currently not parallelized, and that for Ape-X
            # metrics are already only reported for the lowest epsilon workers.
            "evaluation_interval": None,
            # Number of episodes to run per evaluation period.
            "evaluation_num_episodes": 10,

            # === Replay buffer ===
            # Size of the replay buffer in steps.
            "buffer_size": 10000,

            # === Optimization ===
            # Learning rate for RMSProp optimizer
            "lr": 0.0005,
            # RMSProp alpha
            "optim_alpha": 0.99,
            # RMSProp epsilon
            "optim_eps": 0.00001,
            # If not None, clip gradients during optimization at this value
            "grad_norm_clipping": 10,
            # How many steps of the model to sample before learning starts.
            "learning_starts": 1000,
            # Size of a batched sampled from replay buffer for training. Note that
            # if async_updates is set, then each worker returns gradients for a
            # batch of this size.
            "train_batch_size": 32,


            # 'num_gpus': int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            # Whether to use a distribution of epsilons across workers for exploration.
            "per_worker_exploration": False,
            # Whether to compute priorities on workers.
            "worker_side_prioritization": False,
            # Prevent iterations from going lower than this time span
            "min_iter_time_s": 1,

            # === Model ===
            "model": {
                "lstm_cell_size": 64,
            },

            # === COMMON CONFIG ===
            'env': 'kaz_qmix',
        }
    else:
        raise ValueError(
            "Algorithm configuration has not been provided:", args.run)

    config["env"] = ParallelPettingZooEnv

    tune.run(
        args.run,
        stop={'training_iteration': args.num_iters},
        checkpoint_freq=args.checkpoint_freq,
        resume=args.resume,
        config=config)
    ray.shutdown()
