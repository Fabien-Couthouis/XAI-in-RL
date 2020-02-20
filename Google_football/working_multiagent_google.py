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

parser = argparse.ArgumentParser()

parser.add_argument('--num-agents', type=int, default=11)
parser.add_argument('--num-policies', type=int, default=11)
parser.add_argument('--num-iters', type=int, default=10000)
parser.add_argument('--simple', action='store_true')
parser.add_argument(
    "--scenario-name", default="11_vs_11_easy_stochastic", help="Change scenario name.")

CHECKPOINT_PATH = "./multiagent-checkpoint"


class RllibGFootball(MultiAgentEnv):
    """An example of a wrapper for GFootball to make it compatible with rllib."""

    def __init__(self, num_agents, env_name, render=True, save_replays=False):
        self.env = football_env.create_environment(
            env_name=env_name, stacked=False, physics_steps_per_frame=1,
            logdir='./replays', write_video=save_replays,
            write_goal_dumps=save_replays, write_full_episode_dumps=save_replays, render=render,
            dump_frequency=1,
            number_of_left_players_agent_controls=num_agents,
            channel_dimensions=(42, 42))
        self.action_space = gym.spaces.Discrete(self.env.action_space.nvec[1])
        self.observation_space = gym.spaces.Box(
            low=self.env.observation_space.low[0],
            high=self.env.observation_space.high[0],
            dtype=self.env.observation_space.dtype)
        self.num_agents = num_agents

    def reset(self):
        original_obs = self.env.reset()
        obs = {}
        for x in range(self.num_agents):
            if self.num_agents > 1:
                obs['agent_%d' % x] = original_obs[x]
            else:
                obs['agent_%d' % x] = original_obs
        return obs

    def step(self, action_dict):
        actions = []
        for key, value in sorted(action_dict.items()):
            actions.append(value)
        o, r, d, i = self.env.step(actions)
        rewards = {}
        obs = {}
        infos = {}
        for pos, key in enumerate(sorted(action_dict.keys())):
            infos[key] = i
            if self.num_agents > 1:
                rewards[key] = r[pos]
                obs[key] = o[pos]
            else:
                rewards[key] = r
                obs[key] = o
        dones = {'__all__': d}
        return obs, rewards, dones, infos

    def render(self):
        self.env.render()

    def get_random_actions(self):
        'Return: list of random actions for each agent '
        agents_actions = {}
        for agent_id in range(self.num_agents):
            key = "agent_"+str(agent_id)
            agents_actions[key] = self.action_space.sample()
        return agents_actions

    def test(self):
        print(self.env.observation())


def gen_policy(_):
    return (None, obs_space, act_space, {})


def gib_to_octets(gib):
    'Convert Gib value to octets'
    return gib*100*1024 * 1024


if __name__ == '__main__':
    args = parser.parse_args()
    ray.init(num_gpus=1, object_store_memory=gib_to_octets(
        7), redis_max_memory=gib_to_octets(3))

    register_env('g_football', lambda _: RllibGFootball(
        args.num_agents, args.scenario_name, render=False))
    single_env = RllibGFootball(
        args.num_agents, args.scenario_name, render=False)
    obs_space = single_env.observation_space
    act_space = single_env.action_space

    # Setup PPO with an ensemble of `num_policies` different policies
    policies = {
        'policy_{}'.format(i): gen_policy(i) for i in range(args.num_policies)
    }
    policy_ids = list(policies.keys())

    tune.run(
        'PPO',
        stop={'training_iteration': args.num_iters},
        checkpoint_freq=100,
        config={
            'env': 'g_football',
            'lambda': 0.95,
            'kl_coeff': 0.2,
            'clip_rewards': False,
            'vf_clip_param': 10.0,
            'entropy_coeff': 0.01,
            'train_batch_size': 32,
            'sample_batch_size': 16,
            'sgd_minibatch_size': 16,
            'num_sgd_iter': 10,
            'num_workers': 3,
            'num_envs_per_worker': 1,
            'num_cpus_per_worker': 1,
            'batch_mode': 'truncate_episodes',
            'observation_filter': 'NoFilter',
            'vf_share_layers': 'true',
            'num_gpus': 1,
            'lr': 2.5e-4,
            'use_pytorch': 'true',
            'log_level': 'WARN',
            'simple_optimizer': args.simple,
            'multiagent': {
                'policies': policies,
                'policy_mapping_fn': tune.function(
                    lambda agent_id: policy_ids[int(agent_id[6:])]),
            },
        },
    )
