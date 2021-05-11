import csv
import json
import os
import random
import shutil
import sys
from time import time

import numpy as np
import ray
import utility_funcs
from ray.cloudpickle import cloudpickle
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.evaluation.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

# from ray.rllib.evaluation.sampler import clip_action


def get_rllib_config(path):
    """Return the data from the specified rllib configuration file."""
    jsonfile = path + '/params.json'  # params.json is the config file
    jsondata = json.loads(open(jsonfile).read())
    return jsondata


def get_rllib_pkl(path):
    """Return the data from the specified rllib configuration file."""
    pklfile = path + '/params.pkl'  # params.json is the config file
    with open(pklfile, 'rb') as file:
        pkldata = cloudpickle.load(file)
    return pkldata


def load_agent_config(args):
    result_dir = args.result_dir if args.result_dir[-1] != '/' \
        else args.result_dir[:-1]

    config = get_rllib_config(result_dir)
    pkl = get_rllib_pkl(result_dir)

    # check if we have a multiagent scenario but in a
    # backwards compatible way
    if config.get('multiagent', {}).get('policy_graphs', {}):
        multiagent = True
        config['multiagent'] = pkl['multiagent']
    else:
        multiagent = False

    # Create and register a gym+rllib env
    env_creator = pkl['env_config']['func_create']
    env_name = config['env_config']['env_name']
    register_env(env_name, env_creator.func)

    ModelCatalog.register_custom_model("conv_to_fc_net", ConvToFCNet)

    # Determine agent and checkpoint
    config_run = config['env_config']['run'] if 'run' in config['env_config'] \
        else None
    if (args.run and config_run):
        if (args.run != config_run):
            print('visualizer_rllib.py: error: run argument '
                  + '\'{}\' passed in '.format(args.run)
                  + 'differs from the one stored in params.json '
                  + '\'{}\''.format(config_run))
            sys.exit(1)
    if (args.run):
        agent_cls = get_agent_class(args.run)
    elif (config_run):
        agent_cls = get_agent_class(config_run)
    else:
        print('visualizer_rllib.py: error: could not find flow parameter '
              '\'run\' in params.json, '
              'add argument --run to provide the algorithm or model used '
              'to train the results\n e.g. '
              'python ./visualizer_rllib.py /tmp/ray/result_dir 1 --run PPO')
        sys.exit(1)

    # Run on only one cpu for rendering purposes if possible; A3C requires two
    if config_run == 'A3C':
        config['num_workers'] = 1
        config["sample_async"] = False
    else:
        config['num_workers'] = 0

    # create the agent that will be used to compute the actions
    agent = agent_cls(env=env_name, config=config)
    checkpoint = result_dir + '/checkpoint_' + args.checkpoint_num
    checkpoint = checkpoint + '/checkpoint-' + args.checkpoint_num
    print('Loading checkpoint', checkpoint)
    agent.restore(checkpoint)
    return agent, config


def rollout(args, agent, config, num_episodes, considered_player=None, coalition=None):
    if hasattr(agent, "local_evaluator"):
        env = agent.local_evaluator.env

    if hasattr(agent, "local_evaluator"):
        multiagent = agent.local_evaluator.multiagent
        if multiagent:
            policy_agent_mapping = agent.config["multiagent"][
                "policy_mapping_fn"]
            mapping_cache = {}
        policy_map = agent.local_evaluator.policy_map
        state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
        use_lstm = {p: len(s) > 0 for p, s in state_init.items()}
    else:
        multiagent = False
        use_lstm = {DEFAULT_POLICY_ID: False}

    agents_active = [f"agent-{i}" for i in range(args.agents_active)]

    # Rollout
    episode = 0
    rewards_list = []
    while episode < num_episodes:
        steps = 0

        state = env.reset()
        done = False
        reward_total = 0.0
        while not done and steps < (config['horizon'] or steps + 1):
            if args.render:
                print("render")
                env.render()
            if multiagent:
                if args.shapley_M is not None:
                    action = take_actions_for_coalition(env, agent, considered_player, state, mapping_cache, use_lstm,
                                                        policy_agent_mapping, state_init, coalition, args.missing_agents_behaviour, agents_active)
                else:
                    action = take_action(env, agent, state, mapping_cache, use_lstm,
                                         policy_agent_mapping, state_init, agents_active)

            else:
                if use_lstm[DEFAULT_POLICY_ID]:
                    action, state_init, _ = agent.compute_action(
                        state, state=state_init)
                else:
                    action = agent.compute_action(state)

            if agent.config["clip_actions"]:
                # action = clip_action(action, env.action_space)
                next_state, reward, done, _ = env.step(action)
            else:
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

        print("Episode reward", reward_total)
        episode += 1
        rewards_list.append(reward_total)

    return rewards_list


def take_action(env, agent, state, mapping_cache, use_lstm, policy_agent_mapping, state_init, agents_active):
    "Take agents actions"
    action_dict = {}
    agent_ids = [agent_id for agent_id in state.keys()]
    for agent_id in agents_active:
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


def take_actions_for_coalition(env, agent, considered_player, state, mapping_cache, use_lstm, policy_agent_mapping, state_init, coalition, missing_agents_behaviour, agents_active):
    'Return actions where each agent in coalition follow the policy, others play at random if replace_missing_players=="random" or do not move if replace_missing_players=="idle'
    actions = take_action(env, agent, state, mapping_cache,
                          use_lstm, policy_agent_mapping, state_init, agents_active)
    if coalition is None or considered_player is None:
        return actions

    actions_for_coalition = actions.copy()
    agents_without_player = [agent_id for agent_id in actions.keys() if agent_id !=
                             considered_player]
    for agent_id in actions.keys():
        if agent_id not in coalition:
            if missing_agents_behaviour == "random_player_action":
                # Action from another random player
                random_player = random.choice(agents_without_player)
                action = actions[random_player]

            elif missing_agents_behaviour == "random":
                # Random action
                action = np.random.randint(0, 6)

            # elif missing_agents_behaviour == "idle":
            #     # Idle action
            #     action = 4  # env.ACTIONS["STAY"]

            else:
                raise ValueError(
                    f"Value: {missing_agents_behaviour} for parameter missing_agents_bahaviour is not valid. Valid values are: \"random\" \"random_player_action\" or \"idle\".")

            actions_for_coalition[agent_id] = action

    return actions_for_coalition
