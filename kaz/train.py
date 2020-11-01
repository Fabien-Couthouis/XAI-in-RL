from __future__ import absolute_import, division, print_function

import argparse

import gym
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
    parser.add_argument('--num-iters', type=int, default=60)
    parser.add_argument('--simple', action='store_true')
    parser.add_argument("--num-gpus", default=1, type=int,
                        help="number of gpus to use if starting a new cluster.")
    parser.add_argument("--resume", action="store_true",
                        help="Whether to attempt to resume previous Tune experiments.")
    parser.add_argument("--experiment-name", default="default", type=str,
                        help="Name of the subdirectory under `local_dir` to put results in.")
    return parser


def gen_policies(obs_space, act_space, agent_names):
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
    ray.init(num_gpus=args.num_gpus)

    # MADDPG emits action logits instead of actual discrete actions
    actions_are_logits = (args.run.upper() == "MADDPG")

    # Create and register google football env
    def create_env(config=None): return ParallelPettingZooEnv(config=config)
    register_env('kaz', create_env)

    # Get obs/act spaces to feed the policy with
    single_env = create_env()
    obs_space = single_env.observation_space
    act_space = single_env.action_space
    agent_names = single_env.reset().keys()
    single_env.close()

    policies = gen_policies(obs_space, act_space, agent_names=agent_names)

    print(policies)

    if args.run.upper() == "PPO":
        tune.run(
            'PPO',
            stop={'training_iteration': args.num_iters},
            checkpoint_freq=args.checkpoint_freq,
            resume=args.resume,
            config={
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
                # Number of timesteps collected for each SGD round. This defines the size
                # of each SGD epoch.

                # Total SGD batch size across all devices for SGD. This defines the
                # minibatch size within each epoch.
                "sgd_minibatch_size": 256,
                # Whether to shuffle sequences in the batch when training (recommended).
                "shuffle_sequences": True,
                # Number of SGD iterations in each outer loop (i.e., number of epochs to
                # execute per train batch).
                "num_sgd_iter": 30,
                # Stepsize of SGD.
                "lr": 5e-4,
                # Learning rate schedule.
                "lr_schedule": None,
                # Share layers for value function. If you set this to True, it's important
                # to tune vf_loss_coeff.
                "vf_share_layers": False,
                # Coefficient of the value function loss. IMPORTANT: you must tune this if
                # you set vf_share_layers: True.
                "vf_loss_coeff": 1.0,
                # Coefficient of the entropy regularizer.
                "entropy_coeff": 0.01,
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
                "simple_optimizer": args.simple,
                # Whether to fake GPUs (using CPUs).
                # Set this to True for debugging on non-GPU machines (set `num_gpus` > 0).
                "_fake_gpus": False,
                # === COMMON CONFIG ===
                'env': 'kaz',
                'train_batch_size': 512,
                # NOTE: same as sample_batch_size in older versions
                'rollout_fragment_length': 256,
                'num_workers': 1,
                'num_envs_per_worker': 1,
                'num_cpus_per_worker': 3,
                'batch_mode': 'truncate_episodes',
                'num_gpus': args.num_gpus,
                'log_level': 'WARN',
                'multiagent': {
                    'policies': policies,
                    'policy_mapping_fn': policy_agent_mapping,
                },
            },
        )

    elif args.run.upper() == "SAC":

        tune.run(
            'SAC',
            stop={'training_iteration': args.num_iters},
            checkpoint_freq=args.checkpoint_freq,
            resume=args.resume,
            config={

                # === Model ===
                # Use two Q-networks (instead of one) for action-value estimation.
                # Note: Each Q-network will have its own target network.
                "twin_q": True,
                # Use a e.g. conv2D state preprocessing network before concatenating the
                # resulting (feature) vector with the action input for the input to
                # the Q-networks.
                "use_state_preprocessor": False,
                # Model options for the Q network(s).
                "Q_model": {
                    "fcnet_activation": "relu",
                    "fcnet_hiddens": [256, 256],
                },
                # Model options for the policy function.
                "policy_model": {
                    "fcnet_activation": "relu",
                    "fcnet_hiddens": [256, 256],
                },
                # Unsquash actions to the upper and lower bounds of env's action space.
                # Ignored for discrete action spaces.
                "normalize_actions": True,

                # === Learning ===
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
                "target_entropy": None,
                # N-step target updates. If >1, sars' tuples in trajectories will be
                # postprocessed to become sa[discounted sum of R][s t+n] tuples.
                "n_step": 1,
                # Number of env steps to optimize for before returning.
                "timesteps_per_iteration": 100,

                # === Replay buffer ===
                # Size of the replay buffer. Note that if async_updates is set, then
                # each worker will have a replay buffer of this size.
                "buffer_size": int(1e6),
                # If True prioritized replay buffer will be used.
                "prioritized_replay": False,
                "prioritized_replay_alpha": 0.6,
                "prioritized_replay_beta": 0.4,
                "prioritized_replay_eps": 1e-6,
                "prioritized_replay_beta_annealing_timesteps": 20000,
                "final_prioritized_replay_beta": 0.4,
                # Whether to LZ4 compress observations
                "compress_observations": False,
                # If set, this will fix the ratio of replayed from a buffer and learned on
                # timesteps to sampled from an environment and stored in the replay buffer
                # timesteps. Otherwise, the replay will proceed at the native ratio
                # determined by (train_batch_size / rollout_fragment_length).
                "training_intensity": None,

                # === Optimization ===
                "optimization": {
                    "actor_learning_rate": 3e-4,
                    "critic_learning_rate": 3e-4,
                    "entropy_learning_rate": 3e-4,
                },
                # If not None, clip gradients during optimization at this value.
                "grad_clip": None,
                # How many steps of the model to sample before learning starts.
                "learning_starts": 1500,
                # Update the replay buffer with this many samples at once. Note that this
                # setting applies per-worker if num_workers > 1.
                "rollout_fragment_length": 1,
                # Size of a batched sampled from replay buffer for training. Note that
                # if async_updates is set, then each worker returns gradients for a
                # batch of this size.
                "train_batch_size": 256,
                # Update the target network every `target_network_update_freq` steps.
                "target_network_update_freq": 0,


                # === COMMON CONFIG ===
                'env': 'kaz',
                'num_workers': 3,
                'num_envs_per_worker': 1,
                'num_cpus_per_worker': 1,
                'num_gpus': args.num_gpus,
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

    elif args.run.upper() == "IMPALA":

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

                # === COMMON CONFIG ===
                'env': 'kaz',
                'num_workers': 3,
                'num_envs_per_worker': 1,
                'num_cpus_per_worker': 1,
                'num_gpus': args.num_gpus,
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
    elif args.run.upper() == "MADDPG":

        tune.run(
            'contrib/MADDPG',
            stop={'training_iteration': args.num_iters},
            checkpoint_freq=args.checkpoint_freq,
            resume=args.resume,
            config={
                # === MADDPG SPECIFIC CONFIG ===
                # === Settings for each individual policy ===
                # ID of the agent controlled by this policy
                # "agent_id": None,
                # # Use a local critic for this policy.
                # "use_local_critic": False,

                # # # === Evaluation ===
                # # # Evaluation interval
                # # "evaluation_interval": None,
                # # Number of episodes to run per evaluation period.
                # "evaluation_num_episodes": 10,

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

                # Size of the replay buffer. Note that if async_updates is set, then
                # each worker will have a replay buffer of this size.
                "buffer_size": int(1e6),
                # Observation compression. Note that compression makes simulation slow in
                # MPE.
                # "compress_observations": False,

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
                # Number of env steps to optimize for before returning
                "timesteps_per_iteration": 0,

                # === Parallelism ===
                # Prevent iterations from going lower than this time span
                "min_iter_time_s": 0,
                # === COMMON CONFIG ===
                'env': 'kaz',
                'num_workers': 3,
                'num_envs_per_worker': 1,
                'num_cpus_per_worker': 1,
                'num_gpus': args.num_gpus,
                'rollout_fragment_length': 100,
                "train_batch_size": 1024,
                'batch_mode': 'truncate_episodes',
                'log_level': 'DEBUG',
                'multiagent': {
                    'policies': policies,
                    'policy_mapping_fn': policy_agent_mapping,
                },
            }
        )
    # elif args.run == "QMIX":
    #     print("obs_space: ", obs_space)
    #     grouping = {
    #         "group_1": list(range(args.num_agents)),
    #     }

    #     obs_space_qmix = Tuple([
    #         Dict({
    #             "obs": obs_space,
    #             ENV_STATE: obs_space
    #         }) for _ in range(args.num_agents)
    #     ])
    #     act_space_qmix = Tuple([
    #         act_space for _ in range(args.num_agents)
    #     ])

    #     # Create and register google football env

    #     def create_env_qmix(_): return ParallelPettingZooEnv(args.num_agents, args.scenario_name, render=False,
    #                                                   with_state=True).with_agent_groups(grouping,
    #                                                                                      obs_space=obs_space_qmix,
    #                                                                                      act_space=act_space_qmix)

    #     register_env('kaz_qmix', create_env_qmix)

    #     tune.run(
    #         'QMIX',
    #         stop={'training_iteration': args.num_iters},
    #         checkpoint_freq=args.checkpoint_freq,
    #         resume=args.resume,
    #         config={
    #             "rollout_fragment_length": 4,
    #             "train_batch_size": 32,
    #             "exploration_config": {
    #                 "epsilon_timesteps": 5000,
    #                 "final_epsilon": 0.05,
    #             },
    #             "mixer": "qmix",
    #             "env_config": {
    #                 "separate_state_space": True,
    #                 "one_hot_state_encoding": False
    #             },

    #             # === Parallelism ===
    #             # Number of workers for collecting samples with. This only makes sense
    #             # to increase if your environment is particularly slow to sample, or if
    #             # you"re using the Async or Ape-X optimizers.
    #             "num_workers": 0,
    #             'env': 'kaz_qmix',
    #             'num_envs_per_worker': 1,
    #             'num_cpus_per_worker': 1,
    #             'num_gpus': args.num_gpus,
    #             # Whether to use a distribution of epsilons across workers for exploration.
    #             "per_worker_exploration": False,
    #             # Whether to compute priorities on workers.
    #             "worker_side_prioritization": False,
    #             # Prevent iterations from going lower than this time span
    #             "min_iter_time_s": 1,

    #             # === Model ===
    #             "model": {
    #                 "lstm_cell_size": 64,
    #                 # "max_seq_len": 999999,
    #             },
    #         })

    else:
        raise ValueError(
            f"Unsupported algorithm: \"{args.run}\". Please use one of: \"PPO\", \"IMPALA\", \"QMIX\" or \"SAC\"")
