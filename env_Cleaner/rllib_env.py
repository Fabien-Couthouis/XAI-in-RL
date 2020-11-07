import argparse

import numpy as np
import ray
from gym.spaces import Box, Discrete
from ray import tune
from ray.rllib.agents import ppo
from ray.rllib.env import MultiAgentEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.tune.config_parser import make_parser

from env_Cleaner import EnvCleaner


def create_parser(parser_creator=None):
    parser = make_parser(
        parser_creator=parser_creator,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Train a reinforcement learning agent.")
    parser.add_argument('--num-iters', type=int, default=100)
    parser.add_argument("--resume", action="store_true",
                        help="Whether to attempt to resume previous Tune experiments.")
    return parser


class CleanerWrapper(MultiAgentEnv):

    def __init__(self, env_config):
        self.env = EnvCleaner(
            env_config['N_agent'], env_config['map_size'], env_config['seed'], env_config['max_iters'])

    def step(self, action_dict):
        action_list = []
        for k in action_dict:
            action_list.append(action_dict[k])
        reward = self.env.step(action_list)
        obs = self.env.get_global_obs()
        ep_over = self.env.is_episode_over()
        reward_dict = {}
        obs_dict = {}
        dones = {'__all__': True if ep_over else False}
        infos = {}
        for k in action_dict:
            reward_dict[k] = reward
            obs_dict[k] = obs
            infos[k] = {}
            dones[k] = True if ep_over else False
        return obs_dict, reward_dict, dones, infos

    def reset(self):
        self.env.reset()
        obs = self.env.get_global_obs()
        obs_dict = {f'agent_{i}': obs for i in range(self.env.N_agent)}
        return obs_dict

    def render(self):
        self.env.render()


class CustomModel(TFModelV2):
    """Example of a keras custom model that just delegates to an fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(CustomModel, self).__init__(obs_space, action_space, num_outputs,
                                          model_config, name)
        self.model = FullyConnectedNetwork(obs_space, action_space,
                                           num_outputs, model_config, name)
        self.register_variables(self.model.variables())

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    ModelCatalog.register_custom_model(
        "my_model", CustomModel)
    # env = EnvCleaner(4, 42, 0, 500)
    # while True:
    #     env.render()
    n_agent = 3
    map_size = 43
    max_iters = 1000

    config = {
        "env": CleanerWrapper,
        "env_config": {
            "N_agent": n_agent,
            "map_size": map_size,
            "seed": None,
            "max_iters": max_iters
        },
        "model": {
            "custom_model": "my_model"
        },
        "multiagent": {
            "policies": {
                f"pol_agent_{agent_id}": (None, Box(low=0.0, high=1.0, shape=(map_size, map_size, 3), dtype=np.float32), Discrete(4), {}) for agent_id in range(n_agent)
            },
            "policy_mapping_fn":
                lambda agent_id:
                    f"pol_{agent_id}"
        }
    }
    ray.init()

    tune.run(
        args.run,
        stop={'training_iteration': args.num_iters},
        checkpoint_freq=args.checkpoint_freq,
        resume=args.resume,
        config=config)
    ray.shutdown()
