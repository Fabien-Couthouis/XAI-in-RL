from __future__ import absolute_import

import copy
import csv
import json
import os
import random
from time import time

import numpy as np
from ray.cloudpickle import cloudpickle
from ray.rllib import env
from ray.rllib.agents.registry import get_agent_class, get_trainer_class
from ray.rllib.env import MultiAgentEnv
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.tune.registry import get_trainable_cls, register_env
from ray.tune.utils import merge_dicts

from train import env_creator


def load_agent_config(args):
    # Load configuration from checkpoint file.
    config_path = ""
    if args.checkpoint:
        config_dir = os.path.dirname(args.checkpoint)
        config_path = os.path.join(config_dir, "params.pkl")
        # Try parent directory.
        if not os.path.exists(config_path):
            config_path = os.path.join(config_dir, "../params.pkl")

    # Load the config from pickled.
    if os.path.exists(config_path):
        with open(config_path, "rb") as f:
            config = cloudpickle.load(f)
    # If no pkl file found, require command line `--config`.
    else:
        # If no config in given checkpoint -> Error.
        if args.checkpoint:
            raise ValueError(
                "Could not find params.pkl in either the checkpoint dir or "
                "its parent directory AND no `--config` given on command "
                "line!")

        # Use default config for given agent.
        _, config = get_trainer_class(args.run, return_config=True)

    # Make sure worker 0 has an Env.
    config["num_workers"] = 0
    config["num_envs_per_worker"] = 1
    config["create_env_on_driver"] = True

    # Merge with `evaluation_config` (first try from command line, then from
    # pkl file).
    evaluation_config = copy.deepcopy(
        args.config.get("evaluation_config", config.get(
            "evaluation_config", {})))
    config = merge_dicts(config, evaluation_config)
    # Merge with command line `--config` settings (if not already the same
    # anyways).
    config = merge_dicts(config, args.config)
    if not args.env:
        args.env = config.get("env")

    # Make sure we have evaluation workers.
    # if not config.get("evaluation_num_workers"):
    #     config["evaluation_num_workers"] = config.get("num_workers", 0)
    if not config.get("evaluation_num_episodes"):
        config["evaluation_num_episodes"] = 1
    config["render_env"] = args.render
    config["record_env"] = args.video_dir

    if config.get("env_config") is None:
        config["env_config"] = {}

    print(args.agent_speeds)
    config["env_config"]["agent_speeds"] = args.agent_speeds

    register_env(args.env, env_creator)

    # Create the Trainer from config.
    cls = get_trainable_cls(args.run)
    agent = cls(env=args.env, config=config)

    # Load state from checkpoint, if provided.
    if args.checkpoint:
        agent.restore(args.checkpoint)

    return agent, config


def rollout(args, agent, config, num_episodes, considered_player=None, coalition=None):

    if hasattr(agent, "workers") and isinstance(agent.workers, WorkerSet):
        env = agent.workers.local_worker().env

        multiagent = isinstance(env, MultiAgentEnv)
        if agent.workers.local_worker().multiagent:
            policy_agent_mapping = agent.config["multiagent"][
                "policy_mapping_fn"]
            mapping_cache = {}

        policy_map = agent.workers.local_worker().policy_map
        state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
        use_lstm = {p: len(s) > 0 for p, s in state_init.items()}

    else:
        multiagent = False
        use_lstm = {DEFAULT_POLICY_ID: False}

    agents_active = [
        agent_name for agent_name in env.agents if agent_name.startswith("adversary")]

    # Rollout
    episode = 0
    rewards_list = []
    while episode < num_episodes:
        steps = 0
        state = env.reset()

        done = False
        reward_total = 0.0
        max_steps = 50

        if args.shapley_M is not None:
            last_players_actions = {
                agent_name: 0 for agent_name in agents_active}
        while not done and steps < max_steps:
            if args.render:
                env.render()
            if multiagent:
                if args.shapley_M is not None:
                    action = take_actions_for_coalition(env, agent, considered_player, state, mapping_cache, use_lstm,
                                                        policy_agent_mapping, state_init, coalition, args.missing_agents_behaviour, agents_active, last_players_actions)
                    for agent_id, agent_action in action.items():
                        last_players_actions[agent_id] = agent_action
                else:
                    action = take_action(env, agent, state, mapping_cache, use_lstm,
                                         policy_agent_mapping, state_init, agents_active)

            else:
                if use_lstm[DEFAULT_POLICY_ID]:
                    action, state_init, _ = agent.compute_action(
                        state, state=state_init)
                else:
                    action = agent.compute_action(state)

            next_state, reward, done, _ = env.step(action)

            if multiagent:
                done = done["__all__"]
                reward_total += sum(reward.values())
            else:
                reward_total += reward

            if args.social_metrics:
                with open(f'{args.save_dir}/{args.exp_name}.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',')
                    row = [episode] + [steps] + list(reward.values())
                    writer.writerow(row)

            steps += 1
            state = next_state

        episode += 1
        rewards_list.append(reward_total)
    return rewards_list


def take_action(env, agent, state, mapping_cache, use_lstm, policy_agent_mapping, state_init, agents_active):
    "Take agents actions"
    action_dict = {}
    agent_ids = [agent_id for agent_id in state.keys()]
    for agent_id in state.keys():
        a_state = state[agent_id]
        if a_state is not None:
            policy_id = mapping_cache.setdefault(
                agent_id, policy_agent_mapping(agent_id))
            p_use_lstm = use_lstm[policy_id]
            if p_use_lstm:
                a_action, p_state_init, _ = agent.compute_action(
                    a_state,
                    state=state_init[policy_id],
                    policy_id=policy_id)
                state_init[policy_id] = p_state_init
            else:
                a_action = agent.compute_action(
                    a_state, policy_id=policy_id)
            action_dict[agent_id] = a_action

    return action_dict


def take_actions_for_coalition(env, agent, considered_player, state, mapping_cache, use_lstm, policy_agent_mapping, state_init, coalition, missing_agents_behaviour, agents_active, last_players_actions):
    'Return actions where each agent in coalition follow the policy, others play at random if replace_missing_players=="random" or do not move if replace_missing_players=="idle'
    actions = take_action(env, agent, state, mapping_cache,
                          use_lstm, policy_agent_mapping, state_init, agents_active)
    if coalition is None or considered_player is None:
        return actions

    actions_for_coalition = actions.copy()

    for agent_id in actions.keys():
        if agent_id not in coalition:
            if missing_agents_behaviour == "random_player_action":
                # Action from another random player
                agents_without_player = [agent_id for agent_id in last_players_actions.keys() if agent_id !=
                                         considered_player]
                random_player = random.choice(agents_without_player)
                action = last_players_actions[random_player]

            elif missing_agents_behaviour == "random":
                # Random action
                action = env.action_space.sample()

            elif missing_agents_behaviour == "idle":
                # Idle action
                action = 0

            else:
                raise ValueError(
                    f"Value: {missing_agents_behaviour} for parameter missing_agents_bahaviour is not valid. Valid values are: \"random\" \"random_player_action\" or \"idle\".")

            actions_for_coalition[agent_id] = action

    return actions_for_coalition
