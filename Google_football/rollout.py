#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import json
import os
import random
import pickle
import shelve
import numpy as np
from pathlib import Path

import gym
import ray
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.env import MultiAgentEnv
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.evaluation.episode import _flatten_action
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils import merge_dicts
from working_multiagent_google import RllibGFootball
from ray.tune.registry import register_env
from itertools import combinations, permutations
from statistics import mean
from scipy.spatial.distance import euclidean


EXAMPLE_USAGE = """
Example Usage via RLlib CLI:
    rllib rollout /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    --env CartPole-v0 --steps 1000000 --out rollouts.pkl

Example Usage via executable:
    ./rollout.py /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    --env CartPole-v0 --steps 1000000 --out rollouts.pkl
"""


class RolloutSaver:
    """Utility class for storing rollouts.

    Currently supports two behaviours: the original, which
    simply dumps everything to a pickle file once complete,
    and a mode which stores each rollout as an entry in a Python
    shelf db file. The latter mode is more robust to memory problems
    or crashes part-way through the rollout generation. Each rollout
    is stored with a key based on the episode number (0-indexed),
    and the number of episodes is stored with the key "num_episodes",
    so to load the shelf file, use something like:

    with shelve.open('rollouts.pkl') as rollouts:
       for episode_index in range(rollouts["num_episodes"]):
          rollout = rollouts[str(episode_index)]

    If outfile is None, this class does nothing.
    """

    def __init__(self,
                 outfile=None,
                 use_shelve=False,
                 write_update_file=False,
                 target_steps=None,
                 target_episodes=None,
                 save_info=False):
        self._outfile = outfile
        self._update_file = None
        self._use_shelve = use_shelve
        self._write_update_file = write_update_file
        self._shelf = None
        self._num_episodes = 0
        self._rollouts = []
        self._current_rollout = []
        self._total_steps = 0
        self._target_episodes = target_episodes
        self._target_steps = target_steps
        self._save_info = save_info

    def _get_tmp_progress_filename(self):
        outpath = Path(self._outfile)
        return outpath.parent / ("__progress_" + outpath.name)

    @property
    def outfile(self):
        return self._outfile

    def __enter__(self):
        if self._outfile:
            if self._use_shelve:
                # Open a shelf file to store each rollout as they come in
                self._shelf = shelve.open(self._outfile)
            else:
                # Original behaviour - keep all rollouts in memory and save
                # them all at the end.
                # But check we can actually write to the outfile before going
                # through the effort of generating the rollouts:
                try:
                    with open(self._outfile, "wb") as _:
                        pass
                except IOError as x:
                    print("Can not open {} for writing - cancelling rollouts.".
                          format(self._outfile))
                    raise x
            if self._write_update_file:
                # Open a file to track rollout progress:
                self._update_file = self._get_tmp_progress_filename().open(
                    mode="w")
        return self

    def __exit__(self, type, value, traceback):
        if self._shelf:
            # Close the shelf file, and store the number of episodes for ease
            self._shelf["num_episodes"] = self._num_episodes
            self._shelf.close()
        elif self._outfile and not self._use_shelve:
            # Dump everything as one big pickle:
            pickle.dump(self._rollouts, open(self._outfile, "wb"))
        if self._update_file:
            # Remove the temp progress file:
            self._get_tmp_progress_filename().unlink()
            self._update_file = None

    def _get_progress(self):
        if self._target_episodes:
            return "{} / {} episodes completed".format(self._num_episodes,
                                                       self._target_episodes)
        elif self._target_steps:
            return "{} / {} steps completed".format(self._total_steps,
                                                    self._target_steps)
        else:
            return "{} episodes completed".format(self._num_episodes)

    def begin_rollout(self):
        self._current_rollout = []

    def end_rollout(self):
        if self._outfile:
            if self._use_shelve:
                # Save this episode as a new entry in the shelf database,
                # using the episode number as the key.
                self._shelf[str(self._num_episodes)] = self._current_rollout
            else:
                # Append this rollout to our list, to save laer.
                self._rollouts.append(self._current_rollout)
        self._num_episodes += 1
        if self._update_file:
            self._update_file.seek(0)
            self._update_file.write(self._get_progress() + "\n")
            self._update_file.flush()

    def append_step(self, obs, action, next_obs, reward, done, info):
        """Add a step to the current rollout, if we are saving them"""
        if self._outfile:
            if self._save_info:
                self._current_rollout.append(
                    [obs, action, next_obs, reward, done, info])
            else:
                self._current_rollout.append(
                    [obs, action, next_obs, reward, done])
        self._total_steps += 1


def create_parser(parser_creator=None):
    parser_creator = parser_creator or argparse.ArgumentParser
    parser = parser_creator(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Roll out a reinforcement learning agent "
        "given a checkpoint.",
        epilog=EXAMPLE_USAGE)

    parser.add_argument(
        "checkpoint", type=str, help="Checkpoint from which to roll out.")
    required_named = parser.add_argument_group("required named arguments")
    required_named.add_argument(
        "--run",
        type=str,
        required=True,
        help="The algorithm or model to train. This may refer to the name "
        "of a built-on algorithm (e.g. RLLib's DQN or PPO), or a "
        "user-defined trainable function or class registered in the "
        "tune registry.")
    required_named.add_argument(
        "--env", type=str, help="The gym environment to use.")
    parser.add_argument(
        "--no-render",
        default=False,
        action="store_const",
        const=True,
        help="Surpress rendering of the environment.")
    parser.add_argument(
        "--monitor",
        default=False,
        action="store_const",
        const=True,
        help="Wrap environment in gym Monitor to record video.")
    parser.add_argument(
        "--steps", default=100, help="Number of steps to roll out.")
    parser.add_argument("--out", default=None, help="Output filename.")
    parser.add_argument(
        "--config",
        default="{}",
        type=json.loads,
        help="Algorithm-specific configuration (e.g. env, hyperparams). "
        "Surpresses loading of configuration from checkpoint.")
    parser.add_argument(
        "--episodes",
        default=0,
        help="Number of complete episodes to roll out. (Overrides --steps)")
    parser.add_argument(
        "--save-info",
        default=False,
        action="store_true",
        help="Save the info field generated by the step() method, "
        "as well as the action, observations, rewards and done fields.")
    parser.add_argument(
        "--use-shelve",
        default=False,
        action="store_true",
        help="Save rollouts into a python shelf file (will save each episode "
        "as it is generated). An output filename must be set using --out.")
    parser.add_argument(
        "--track-progress",
        default=False,
        action="store_true",
        help="Write progress to a temporary file (updated "
        "after each episode). An output filename must be set using --out; "
        "the progress file will live in the same folder.")
    parser.add_argument(
        "--compute-shapley",
        default=False,
        action="store_true",
        help="Compute Shapley values.")
    parser.add_argument(
        "--scenario-name",
        default="shapley_no_adversary",
        help="Change scenario name.")
    parser.add_argument(
        "--num-agents",
        default=3,
        help="Change number of agents.")
    parser.add_argument(
        "--save-replays",
        default=False,
        action="store_true",
        help="Save dump replays, goals and videos.")

    return parser


def run(args, parser):
    config = {}
    # Load configuration from file
    config_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(config_dir, "params.pkl")
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "../params.pkl")
    if not os.path.exists(config_path):
        if not args.config:
            raise ValueError(
                "Could not find params.pkl in either the checkpoint dir or "
                "its parent directory.")
    else:
        with open(config_path, "rb") as f:
            config = pickle.load(f)
    if "num_workers" in config:
        config["num_workers"] = min(2, config["num_workers"])
    config = merge_dicts(config, args.config)
    if not args.env:
        if not config.get("env"):
            parser.error("the following arguments are required: --env")
        args.env = config.get("env")

    ray.init()

    cls = get_agent_class(args.run)
    agent = cls(env=args.env, config=config)
    agent.restore(args.checkpoint)
    num_steps = int(args.steps)
    num_episodes = int(args.episodes)
    env = agent.workers.local_worker().env
    with RolloutSaver(
            args.out,
            args.use_shelve,
            write_update_file=args.track_progress,
            target_steps=num_steps,
            target_episodes=num_episodes,
            save_info=args.save_info) as saver:
        if args.compute_shapley:
            monte_carlo_shapley_estimation(args.env, env.num_agents,
                                           agent, num_steps, num_episodes)
        else:
            rollout(agent, args.env, num_steps, num_episodes, saver,
                    args.no_render, args.monitor)


class DefaultMapping(collections.defaultdict):
    """default_factory now takes as an argument the missing key."""

    def __missing__(self, key):
        self[key] = value = self.default_factory(key)
        return value


DEBUG = False


def default_policy_agent_mapping(unused_agent_id):
    return DEFAULT_POLICY_ID


def keep_going(steps, num_steps, episodes, num_episodes):
    """Determine whether we've collected enough data"""
    # if num_episodes is set, this overrides num_steps
    if num_episodes:
        return episodes < num_episodes
    # if num_steps is set, continue until we reach the limit
    if num_steps:
        return steps < num_steps
    # otherwise keep going forever
    return True


def get_ball_owned_player(env):
    obs = env.env.env.env._env.observation()  # Thanks google
    return obs['ball_owned_player']


def get_nearest_agent_from_ball(env, team="left_team"):
    obs = env.env.env.env._env.observation()
    ball_x, ball_y, ball_z = obs["ball"]
    agents_positions = obs[team]
    nearest_agent = None
    min_dist = None
    for i, (x, y) in enumerate(agents_positions):
        dist = euclidean([ball_x, x], [ball_y, y])
        # print("i", i, dist)

        # print(dist)
        if min_dist is None or dist < min_dist:
            min_dist = dist
            nearest_agent = i-1  # do not take goal in account
    return nearest_agent


def get_ball_owned_team(env):
    obs = env.env.env.env._env.observation()
    if obs['ball_owned_team'] == -1:
        return None
    else:
        return "right" if obs['ball_owned_team'] == 1 else "left"


def rollout(agent,
            env_name,
            num_steps,
            num_episodes=0,
            saver=RolloutSaver(),
            no_render=True,
            monitor=False,
            coalition=None):
    'Play game'

    policy_agent_mapping = default_policy_agent_mapping

    if hasattr(agent, "workers"):
        env = agent.workers.local_worker().env
        multiagent = isinstance(env, MultiAgentEnv)
        if agent.workers.local_worker().multiagent:
            policy_agent_mapping = agent.config["multiagent"][
                "policy_mapping_fn"]

        policy_map = agent.workers.local_worker().policy_map
        state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
        use_lstm = {p: len(s) > 0 for p, s in state_init.items()}
        action_init = {
            p: _flatten_action(m.action_space.sample())
            for p, m in policy_map.items()
        }
    else:
        env = gym.make(env_name)
        multiagent = False
        use_lstm = {DEFAULT_POLICY_ID: False}

    if monitor and not no_render and saver and saver.outfile is not None:
        # If monitoring has been requested,
        # manually wrap our environment with a gym monitor
        # which is set to record every episode.
        env = gym.wrappers.Monitor(
            env, os.path.join(os.path.dirname(saver.outfile), "monitor"),
            lambda x: True)

    steps = 0
    episodes = 0
    reward_table = []
    while keep_going(steps, num_steps, episodes, num_episodes):
        mapping_cache = {}  # in case policy_agent_mapping is stochastic
        saver.begin_rollout()
        obs = env.reset()
        agent_states = DefaultMapping(
            lambda agent_id: state_init[mapping_cache[agent_id]])
        prev_actions = DefaultMapping(
            lambda agent_id: action_init[mapping_cache[agent_id]])
        prev_rewards = collections.defaultdict(lambda: 0.)
        done = False
        reward_total = 0.0
        while not done and keep_going(steps, num_steps, episodes,
                                      num_episodes):
            multi_obs = obs if multiagent else {_DUMMY_AGENT_ID: obs}

            if coalition is not None:
                action = take_actions_for_coalition(env, agent, multi_obs, mapping_cache, use_lstm,
                                                    agent_states, prev_actions, prev_rewards, policy_agent_mapping, coalition)
            else:
                action = take_action(agent, multi_obs, mapping_cache, use_lstm,
                                     agent_states, prev_actions, prev_rewards, policy_agent_mapping)

            action = action if multiagent else action[_DUMMY_AGENT_ID]

            next_obs, reward, done, info = env.step(action)

            if multiagent:
                for agent_id, r in reward.items():
                    prev_rewards[agent_id] = r
            else:
                prev_rewards[_DUMMY_AGENT_ID] = reward

            if multiagent:
                done = done["__all__"]
                reward_total += sum(reward.values())
            else:
                reward_total += reward
            if not no_render:
                env.render()

            saver.append_step(obs, action, next_obs, reward, done, info)
            steps += 1
            obs = next_obs

        saver.end_rollout()

        print("Episode #{}: reward: {}".format(episodes, reward_total))

        if done:
            episodes += 1
            reward_table.append(reward_total)

    return reward_table


def get_combinations(features):
    'Get all possible coalitions between features'
    combinations_list = []
    for i in range(1, len(features)+1):
        oc = combinations(features, i)
        for c in oc:
            combinations_list.append(list(c))
    return combinations_list


def get_combinations_for_feature(features, feature_id):
    combinations = get_combinations(features)
    with_player, without_player = [], []
    for feature in combinations:
        if feature_id in feature:
            with_player.append(feature)
        else:
            without_player.append(feature)

    return with_player, without_player


def monte_carlo_shapley_estimation(env_name, n_agents, agent, num_steps=0, num_episodes=1, M=100):
    """
    Monte Carlo estimation of shapley values (o(M*n_agents) complexity).
    Parameters:
        M: Number of coalitions to evaluate for each player (optional, default=100)
        See rollout function for other parameters.
    """

    # Episodes override steps in the API, so we keep this behaviour here
    if num_episodes != 0:
        print("Starting Shapley value estimation on:", n_agents,
              "agents,", num_episodes, "episodes, M=", M)
    else:
        print("Starting Shapley value estimation on:",
              n_agents, "agents,", num_steps, "steps, M=", M)

    estimated_values = []
    features = range(n_agents)
    for feature in features:
        print(f"Starting computation for agent {feature}")
        with_player, without_player = get_combinations_for_feature(
            features, feature)
        marginal_contributions = []
        for m in range(M):
            print(f"Step {m}/{M} for Agent {feature}:")
            coalition_with_player = random.choice(with_player)
            coalition_without_player = random.choice(without_player)

            value_with_player = rollout(
                agent, env_name, num_steps, num_episodes, coalition=coalition_with_player)
            value_without_player = rollout(
                agent, env_name, num_steps, num_episodes, coalition=coalition_without_player)

            marginal_contribution = (
                sum(value_with_player)-sum(value_without_player))/num_episodes
            marginal_contributions.append(marginal_contribution)

        estimated_values.append(mean(marginal_contributions))

    norm_estimated_values = np.divide(estimated_values, sum(estimated_values))
    print("Normalized Shapley values:", norm_estimated_values)
    print("Shapley values:", estimated_values)

    return estimated_values


def get_marginal_contributions(env_name, features, num_steps, num_episodes, agent):
    'Get mean reward for each agent for each coalitions '
    coalition_values = dict()
    for coalition in get_combinations(features):
        total_rewards = rollout(
            agent, env_name, num_steps, num_episodes, coalition=coalition)

        coalition_values[str(coalition)] = round(mean(total_rewards), 2)
    if DEBUG:
        print("Coalition values: ", coalition_values)
    return coalition_values


def shapley_values(env_name, n_agents, agent, num_steps=10, num_episodes=1):
    'Naive implementation (not optimized at all)'
    agents_ids = range(n_agents)
    coalition_values = get_marginal_contributions(
        env_name, agents_ids, num_steps, num_episodes, agent)
    shapley_values = []

    for agent_id in agents_ids:
        if DEBUG:
            print("Computing shap value for agent: ", agent_id)
        shapley_value = 0

        for permutation in permutations(agents_ids):
            to_remove = []
            if DEBUG:
                print("permutation ", permutation)
            for i, x in enumerate(permutation):
                if x == agent_id:
                    coalition = sorted(permutation[:i+1])
                    if DEBUG:
                        print("coalition", coalition)
                    shapley_value += coalition_values[str(coalition)]

                    if len(to_remove) > 0:
                        to_remove = str(sorted(to_remove))
                        shapley_value -= coalition_values[to_remove]
                        if DEBUG:
                            print("to remove ", to_remove)
                    break
                else:
                    to_remove.append(x)
        shapley_values.append(shapley_value)

    result = np.divide(shapley_values, np.math.factorial(agent_nb))
    norm_result = np.divide(result, sum(result))
    print("Normalized Shapley values:", norm_result)
    print("Shapley values:", result)

    return result


def take_action(agent, multi_obs, mapping_cache, use_lstm, agent_states, prev_actions, prev_rewards, policy_agent_mapping):
    "take actions"
    action_dict = {}
    for agent_id, a_obs in multi_obs.items():
        if a_obs is not None:
            policy_id = mapping_cache.setdefault(
                agent_id, policy_agent_mapping(agent_id))
            p_use_lstm = use_lstm[policy_id]
            if p_use_lstm:
                a_action, p_state, _ = agent.compute_action(
                    a_obs,
                    state=agent_states[agent_id],
                    prev_action=prev_actions[agent_id],
                    prev_reward=prev_rewards[agent_id],
                    policy_id=policy_id)
                agent_states[agent_id] = p_state
            else:
                a_action = agent.compute_action(
                    a_obs,
                    prev_action=prev_actions[agent_id],
                    prev_reward=prev_rewards[agent_id],
                    policy_id=policy_id)
            a_action = _flatten_action(a_action)  # tuple actions
            action_dict[agent_id] = a_action
            prev_actions[agent_id] = a_action
    return action_dict


def take_actions_for_coalition(env, agent, multi_obs, mapping_cache, use_lstm,
                               agent_states, prev_actions, prev_rewards, policy_agent_mapping, coalition):
    'Return actions where each agent in coalition follow the policy, others play at random'
    actions = take_action(agent, multi_obs, mapping_cache, use_lstm,
                          agent_states, prev_actions, prev_rewards, policy_agent_mapping)
    random_actions = env.get_random_actions()

    for agent_id in coalition:
        key = "agent_"+str(agent_id)
        random_actions[key] = actions[key]

    return random_actions


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    # Register custom envs
    register_env("gfootball", lambda _: RllibGFootball(
        num_agents=args.num_agents, env_name=args.scenario_name, render=(not args.no_render), save_replays = args.save_replays))
    run(args, parser)
