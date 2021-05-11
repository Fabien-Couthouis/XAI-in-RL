import ray
from ray import tune
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.tune import run_experiments
from ray.tune.registry import register_env
from gym.spaces import Tuple, Dict
import argparse

from env import PrisonerDilemma


def create_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="PrisonerDilemma")
    parser.add_argument("--algo", type=str, default="QMIX")
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--checkpoint-freq", type=int, default=100)
    parser.add_argument("--rollout-frag-length", type=int, default=4)
    parser.add_argument("--mixer", type=str, default="vdn")
    parser.add_argument("--num-agents", type=int, default=5)
    parser.add_argument("--num-repetitions", type=int, default=10)
    parser.add_argument("--num-cpus", type=int, default=8)
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument("--exp-name", type=str, default=None)
    parser.add_argument("--restore", type=str, default=None)
    parser.add_argument("--training-iterations", type=int, default=1e4)

    return parser


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
    parser = create_parser()
    args = parser.parse_args()
    ray.init(num_cpus=args.num_cpus)
    
    env_name, config = setup(args)

    if args.exp_name is None:
        exp_name = args.env + '_' + args.algo
    else:
        exp_name = args.exp_name
    print('Commencing experiment', exp_name)

    run_experiments({
        exp_name: {
            "run": args.algo,
            "env": env_name,
            "stop": {
                "training_iteration": args.training_iterations
            },
            'checkpoint_freq': args.checkpoint_freq,
            "config": config,
            "restore": args.restore
        }
    })
