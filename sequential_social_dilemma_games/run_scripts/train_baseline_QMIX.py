import argparse

import ray
from gym.spaces import Dict, Tuple
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

from social_dilemmas.envs.harvest import HarvestEnv

def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Evaluates a reinforcement learning agent "
                    "given a checkpoint.")

    # required input parameters
    parser.add_argument(
        "--result_dir", type=str, default="results", help="Directory containing results")
    parser.add_argument("--checkpoint_freq", type=str,
                        default=200, help="Checkpoint frequency.")

    # optional input parameters
    parser.add_argument(
        "--run",
        type=str,
        default="QMIX",
        help="The algorithm or model to train. This may refer to "
             "the name of a built-on algorithm (e.g. RLLib\"s DQN "
             "or PPO), or a user-defined trainable function or "
             "class registered in the tune registry. ")
    parser.add_argument("--num-agents", type=int, default=5)
    parser.add_argument("--training-iterations", type=int, default=10000)
    parser.add_argument("--horizon", type=int, default=1000)
    return parser


def env_creator(env_config=None):
    num_agents = env_config["num_agents"]
    return HarvestEnv(num_agents=num_agents)


def get_ppo_config():
    return {
        # Should use a critic as a baseline (otherwise don"t use value baseline;
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
        # you set vf_share_layers=True inside your model"s config.
        "vf_loss_coeff": 1.0,
        "model": {
            # Share layers for value function. If you set this to True, it"s
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
            "policy_mapping_fn": lambda agent_id: agent_id,
        }
    }


def get_qmix_config():
    return {
        # === QMix ===
        # Mixing network. Either "qmix", "vdn", or None
        "mixer": "vdn",
        # Size of the mixing network embedding
        "mixing_embed_dim": 32,
        # Whether to use Double_Q learning
        "double_q": True,
        # Optimize over complete episodes by default.
        "batch_mode": "complete_episodes",

        # === Exploration Settings ===
        "exploration_config": {
            # The Exploration class to use.
            "type": "EpsilonGreedy",
            # Config for the Exploration class" constructor:
            "initial_epsilon": 1.0,
            "final_epsilon": 0.02,
            # Timesteps over which to anneal epsilon.
            "epsilon_timesteps": 10000,

            # For soft_q, use:
            # "exploration_config" = {
            #   "type": "SoftQ"
            #   "temperature": [float, e.g. 1.0]
            # }
        },

        # === Evaluation ===
        # Evaluate with epsilon=0 every `evaluation_interval` training iterations.
        # The evaluation stats will be reported under the "evaluation" metric key.
        # Note that evaluation is currently not parallelized, and that for Ape-X
        # metrics are already only reported for the lowest epsilon workers.
        "evaluation_interval": None,
        # Number of episodes to run per evaluation period.
        "evaluation_num_episodes": 10,
        # Switch to greedy actions in evaluation workers.
        "evaluation_config": {
            "explore": False,
        },

        # Number of env steps to optimize for before returning
        "timesteps_per_iteration": 500,
        # Update the target network every `target_network_update_freq` steps.
        "target_network_update_freq": 300,

        # === Replay buffer ===
        # Size of the replay buffer in batches (not timesteps!).
        "buffer_size": 1000,

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
        # Update the replay buffer with this many samples at once. Note that
        # this setting applies per-worker if num_workers > 1.
        "rollout_fragment_length": 4,
        # Size of a batched sampled from replay buffer for training. Note that
        # if async_updates is set, then each worker returns gradients for a
        # batch of this size.
        "train_batch_size": 64,

        # === Parallelism ===
        # Number of workers for collecting samples with. This only makes sense
        # to increase if your environment is particularly slow to sample, or if
        # you"re using the Async or Ape-X optimizers.
        "num_workers": 0,
        # Whether to use a distribution of epsilons across workers for exploration.
        "per_worker_exploration": False,
        # Whether to compute priorities on workers.
        "worker_side_prioritization": False,
        # Prevent iterations from going lower than this time span
        "min_iter_time_s": 1,

        # Custom model registration
        "model": {
            "lstm_cell_size": 64
        }
    }


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    ray.init()
    register_env("harvest", env_creator)

    # register the custom model
    model_name = "conv_to_fc_net"

    base_config = {
        "num_workers": 8,
        "num_envs_per_worker": 10,
        "horizon": args.horizon,
        "env": "harvest",
        "env_config": {
            "num_agents": args.num_agents,
        },
    }

    if args.run.upper() == "PPO":
        algo_config = get_ppo_config()
        single_env = env_creator(env_config=base_config["env_config"])
        policies = {}
        for i in range(args.num_agents):
            policies[f"agent-{i}"] = (None, single_env.observation_space, single_env.action_space, {"agent_id": f"agent-{i}"})
        algo_config["multiagent"]["policies"] = policies
        

    elif args.run.upper() == "QMIX":
        single_env = env_creator(env_config=base_config["env_config"])

        grouping = {"group_1": [
            f"agent-{i}" for i in range(single_env.num_agents)]}

        obs_space = Tuple(
            [single_env.observation_space for _ in range(single_env.num_agents)])
        act_space = Tuple([single_env.action_space
                          for _ in range(single_env.num_agents)])

        register_env("grouped_harvest", lambda env_config: env_creator(env_config).with_agent_groups(
            grouping, obs_space=obs_space, act_space=act_space))

        base_config["env"] = "grouped_harvest"
        algo_config = get_qmix_config()

    tune.run(
        args.run,
        config=dict(base_config, **algo_config),
        checkpoint_freq=args.checkpoint_freq,
        local_dir=args.result_dir,
        stop={
            "training_iteration": args.training_iterations
        },
    )
