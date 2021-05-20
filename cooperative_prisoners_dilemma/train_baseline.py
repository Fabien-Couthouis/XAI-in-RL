import argparse

import ray
from gym.spaces import Dict, Tuple
from ray import tune
from ray.tune.registry import register_env

from env import PrisonerDilemma

import ray
import supersuit
from pettingzoo.mpe import simple_tag_v2
from ray import tune
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env
import argparse

<<<<<<< HEAD

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
    parser.add_argument("--num-agents", type=int, default=5)
    parser.add_argument("--num-repetitions", type=int, default=10)
    parser.add_argument("--training-iterations", type=int, default=1e4)
=======
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
        default="PPO",
        help="The algorithm or model to train. This may refer to "
             "the name of a built-on algorithm (e.g. RLLib\"s DQN "
             "or PPO), or a user-defined trainable function or "
             "class registered in the tune registry. ")
    parser.add_argument("--num-agents", type=int, default=5)
    parser.add_argument("--num-repetitions", type=int, default=10)
    parser.add_argument("--training-iterations", type=int, default=1200)
>>>>>>> 189568557c02f525b1c529071f6a93986228d4b3
    return parser


def env_creator(env_config=None):
    env_config = env_config or {}
    return PrisonerDilemma(env_config)


def get_ppo_config():
    return {
<<<<<<< HEAD
        # Should use a critic as a baseline (otherwise don't use value baseline;
=======
        # Should use a critic as a baseline (otherwise don"t use value baseline;
>>>>>>> 189568557c02f525b1c529071f6a93986228d4b3
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
<<<<<<< HEAD
    }


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    ray.init()
    register_env("prey_predator", env_creator)
    config = {
        "num_workers": 4,
        "num_envs_per_worker": 10,
        "env": "prey_predator",
        "env_config": {'num_players': args.num_agents, 'num_repetitions': args.num_repetitions}
    }

    if args.run.upper() == "PPO":
        algo_config = get_ppo_config()
    elif args.run.upper()=="QMIX":
        single_env = PrisonerDilemma(env_config)
        obs_space = Tuple([Dict({"obs": single_env.observation_space})]*single_env.num_agents)
        act_space = Tuple([single_env.action_space]*args.num_agents)

    tune.run(
        args.run,
        config=dict(config, **algo_config),
        checkpoint_freq=args.checkpoint_freq,
        local_dir=args.result_dir,
        stop={
                "training_iteration": args.training_iterations
            },
    )



def setup(args):
    
    env_config = {'num_players': args.num_agents, 'num_repetitions': args.num_repetitions}

    single_env = PrisonerDilemma(env_config)
    obs_space = Tuple([Dict({"obs": single_env.observation_space})]*args.num_agents)
    act_space = Tuple([single_env.action_space]*args.num_agents)
    
    if args.algo == "QMIX":
        def env_creator(config):
            return PrisonerDilemma(config).with_agent_groups({"group_1": [f"agent-{i}" for i in range(args.num_agents)]}, obs_space=obs_space, act_space=act_space)
    else:
        def env_creator(config):
            return PrisonerDilemma(config)

    env_name = args.env + "_env"
    register_env(env_name, env_creator)

    # # Each policy can have a different configuration (including custom model)
    # def gen_policy():
    #     return (PPOTorchPolicy, obs_space, act_space, {})

    # policy_graphs = {}
    # for i in range(args.num_agents):
    #     policy_graphs['agent-' + str(i)] = gen_policy()

    # def policy_mapping_fn(agent_id):
    #     return agent_id

    agent_cls = get_agent_class(args.algo)
    config = agent_cls._default_config.copy()

    config['env_config']['num_players'] = args.num_agents
    config['env_config']['num_repetitions'] = args.num_repetitions

    # information for replay
    #config['env_config']['func_create'] = tune.function(env_creator)
    config['env_config']['env_name'] = env_name
    config['env_config']['run'] = args.algo

    # hyperparams
    config.update({
                "mixer": args.mixer,
                "train_batch_size": args.train_batch_size,
                "num_workers": 0,
                "num_gpus": args.num_gpus,  # The number of GPUs for the driver
                "framework": "torch"
    })
    return env_name, config


if __name__ == '__main__':
=======
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
    }


if __name__ == "__main__":
>>>>>>> 189568557c02f525b1c529071f6a93986228d4b3
    parser = create_parser()
    args = parser.parse_args()

    ray.init()
    register_env("prisoner_dilemma", env_creator)
    base_config = {
        "num_workers": 8,
        "num_envs_per_worker": 10,
        "env": "prisoner_dilemma",
        "env_config": {
            "num_players": args.num_agents,
            "num_repetitions": args.num_repetitions,
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
            f"agent-{i}" for i in range(single_env.num_players)]}

        obs_space = Tuple(
            [single_env.observation_space for _ in range(single_env.num_players)])
        act_space = Tuple([single_env.action_space
                          for _ in range(single_env.num_players)])

        register_env("grouped_prisoner_dilemma", lambda env_config: env_creator(env_config).with_agent_groups(
            grouping, obs_space=obs_space, act_space=act_space))

        base_config["env"] = "grouped_prisoner_dilemma"
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
