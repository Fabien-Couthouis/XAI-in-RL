from __future__ import absolute_import

import argparse

import ray
import supersuit
from ray import tune
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env

import simple_tag_v2_custom as simple_tag_v2


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Evaluates a reinforcement learning agent '
                    'given a checkpoint.')

    # required input parameters
    parser.add_argument(
        '--result_dir', type=str, default="results", help='Directory containing results')
    parser.add_argument('--checkpoint_freq', type=str,
                        default=200, help='Checkpoint frequency.')

    # optional input parameters
    parser.add_argument(
        '--run',
        type=str,
        default="PPO",
        help='The algorithm or model to train. This may refer to '
             'the name of a built-on algorithm (e.g. RLLib\'s DQN '
             'or PPO), or a user-defined trainable function or '
             'class registered in the tune registry. ')

    return parser


def env_creator(env_config=None):
    env_config = env_config or {}
    env = simple_tag_v2.env(**env_config)
    supersuit_wrapped = supersuit.aec_wrappers.pad_observations(env)
    return PettingZooEnv(supersuit_wrapped)


def get_ppo_config(policies):
    return {
        # Should use a critic as a baseline (otherwise don't use value baseline;
        # required for using GAE).
        "use_critic": True,
        # If true, use the Generalized Advantage Estimator (GAE)
        # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
        "use_gae": True,
        # The GAE (lambda) parameter.
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
        # Coefficient of the value function loss. IMPORTANT: you must tune this if
        # you set vf_share_layers=True inside your model's config.
        "vf_loss_coeff": 1.0,
        "model": {
            # Share layers for value function. If you set this to True, it's
            # important to tune vf_loss_coeff.
            "vf_share_layers": False,
        },
        # Coefficient of the entropy regularizer.
        "entropy_coeff": 0.001,
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

        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": lambda agent_id: agent_id
        }
    }


def gen_policies(obs_space, act_space, agents):
    'Generate policies dict {policy_name: (policy_type, obs_space, act_space, {"agent_id": i})}'
    policies = {agent_id:  (None, obs_space, act_space, {})
                for agent_id in agents}
    return policies


# def gen_policies_maddpg(obs_space, act_space, agents):
#     obs_space_dict = {agent_id: obs_space for agent_id in agents}
#     act_space_dict = {agent_id: act_space for agent_id in agents}

#     policies = {
#        f"policy_{agent_id}": (None, obs_space, act_space,
#             {"agent_id": agent_id,
#              "use_local_critic": False,
#              "obs_space_dict": obs_space_dict,
#              "act_space_dict": act_space_dict})
#         for agent_id in agents}
#     return policies

if __name__ == "__main__":
    from ray.tune.registry import get_trainable_cls
    parser = create_parser()
    args = parser.parse_args()

    ray.init()
    register_env("prey_predator", env_creator)

    config = {
        "num_workers": 4,
        "num_envs_per_worker": 10,
        "env": "prey_predator",
    }

    # Get obs/act spaces to feed the policy with
    single_env = env_creator()
    obs_space = single_env.observation_space
    act_space = single_env.action_space
    agents = single_env.agents

    single_env.close()
    policies = gen_policies(obs_space, act_space, agents)

    if args.run.upper() == "PPO":
        algo_config = get_ppo_config(policies)

    # elif args.run.upper() == "MADDPG":
    #     policies = gen_policies_maddpg(obs_space, act_space,agents)
    #     algo_config = get_maddpg_config(policies)

    tune.run(
        args.run,
        num_samples=5,
        stop={"time_total_s": 42000},
        config=dict(config, **algo_config),
        checkpoint_freq=args.checkpoint_freq,
        local_dir=args.result_dir
    )
