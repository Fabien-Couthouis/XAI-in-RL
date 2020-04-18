#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import random
import argparse
import pickle
import shelve
from pathlib import Path

import gym
import ray
import matplotlib.pyplot as plt
from ray.rllib.utils import merge_dicts
from ray.tune.registry import get_trainable_cls
from ray.tune.registry import register_env

from experiments.RllibGFootball import RllibGFootball
from experiments.rollout import *
from experiments.shapley_values import shapley_values
from experiments.plot import plot_shap_barchart, plot_shap_piechart


EXAMPLE_USAGE = """
Example Usage via RLlib CLI:
    rllib rollout /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    --env CartPole-v0 --steps 1000000 --out rollouts.pkl

Example Usage via executable:
    ./rollout.py /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    --env CartPole-v0 --steps 1000000 --out rollouts.pkl
"""
# Agent names used in charts
AGENTS_NAMES = ["agent_0", "agent_1", "agent_2",
                "agent_3", "agent_4", "agent_5",
                "agent_6", "agent_7", "agent_8",
                "agent_9", "agent_10"]


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
        "--steps",
        default=100,
        type=int,
        help="Number of steps to roll out.")
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
        type=int,
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
        "--n-random-coalitions",
        default=100,
        type=int,
        help="Number of random coalitions used to estimate shapley value for each player.")
    parser.add_argument(
        "--missing-agents-behaviour",
        default="random",
        type=str,
        help="random|idle: Behaviour for missing (from the coalition) agents in Shapley value computation. Random for random actions, idle for no action.")
    parser.add_argument(
        "--scenario-name",
        default="shapley_no_adversary",
        help="Change scenario name.")
    parser.add_argument(
        "--num-agents",
        default=3,
        type=int,
        help="Change number of agents.")
    parser.add_argument(
        "--save-replays",
        default=False,
        action="store_true",
        help="Save dump replays, goals and videos.")
    parser.add_argument(
        "--plot-shapley",
        default=False,
        action="store_true",
        help="Plot shapley values (if computed).")

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

    # Create the Trainer from config.
    cls = get_trainable_cls(args.run)
    agent = cls(env=args.env, config=config)
    # Load state from checkpoint.
    agent.restore(args.checkpoint)
    num_steps = args.steps
    num_episodes = args.episodes

    env = agent.workers.local_worker().env
    with RolloutSaver(
            args.out,
            args.use_shelve,
            write_update_file=args.track_progress,
            target_steps=num_steps,
            target_episodes=num_episodes,
            save_info=args.save_info) as saver:
        if args.compute_shapley:
            s_values = shapley_values(args.env, env.num_agents,
                                      agent, num_steps, num_episodes, n_random_coalitions=args.n_random_coalitions, replace_missing_agents=args.missing_agents_behaviour)
            ax = plot_shap_barchart(s_values, AGENTS_NAMES[:env.num_agents])
            if args.plot_shapley:
                plt.show()
                print("PLOT")

        else:
            return rollout(agent, args.env, num_steps, num_episodes, saver,
                           args.no_render, args.monitor)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    # Register custom envs
    register_env("gfootball", lambda _: RllibGFootball(num_agents=args.num_agents,
                                                       env_name=args.scenario_name,
                                                       render=(
                                                           not args.no_render),
                                                       save_replays=args.save_replays))
    run(args, parser)
