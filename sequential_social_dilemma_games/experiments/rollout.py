import csv
import json
import os
import random
import shutil
import sys
from time import time

import numpy as np
import ray
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import utility_funcs
from ray.cloudpickle import cloudpickle
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from models.conv_to_fc_net import ConvToFCNet

from gym.spaces import Tuple
from social_dilemmas.envs.harvest import HarvestEnv
# from ray.rllib.evaluation.sampler import clip_action

def env_creator(env_config=None):
    num_agents = env_config["num_agents"]
    return HarvestEnv(num_agents=num_agents)

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

    single_env = env_creator(pkl['env_config'])
    env_name = pkl['env']
    # Create and register a gym+rllib env
    obs_space = Tuple(
            [single_env.observation_space for _ in range(single_env.num_agents)])
    act_space = Tuple([single_env.action_space
                          for _ in range(single_env.num_agents)])
    
    grouping = {"group_1": [
            f"agent-{i}" for i in range(single_env.num_agents)]}

    register_env(env_name, lambda env_config: env_creator(env_config).with_agent_groups(grouping, obs_space=obs_space, act_space=act_space))

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
    if hasattr(agent, "workers"):
        env = agent.workers.local_worker().env

        if args.save_video:
            shape = env.base_map.shape
            full_obs = [np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
                        for i in range(config["horizon"])]

        multiagent = isinstance(env, MultiAgentEnv)
        policy_agent_mapping = agent.config["multiagent"]["policy_mapping_fn"] if agent.workers.local_worker().multiagent else None
        mapping_cache = {}
        policy_map = agent.workers.local_worker().policy_map
        state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
        use_lstm = {p: len(s) > 0 for p, s in state_init.items()}
    else:
        multiagent = False
        use_lstm = {DEFAULT_POLICY_ID: False}

    if config["agents_fov"] is not None:
        env.set_agents_fov(config["agents_fov"])

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
                                                        policy_agent_mapping, state_init, coalition, args.missing_agents_behaviour, agents_active, args.run)
                else:
                    action = take_action(env, agent, state, mapping_cache, use_lstm,
                                         policy_agent_mapping, state_init, agents_active, args.run)

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

            if args.save_video:
                rgb_arr = env.map_to_colors()
                full_obs[steps] = rgb_arr.astype(np.uint8)

            steps += 1
            state = next_state

        print("Episode reward", reward_total)
        episode += 1
        rewards_list.append(reward_total)

        if args.save_video:
            path = os.path.abspath(os.path.dirname(__file__)) + '/videos'
            if not os.path.exists(path):
                os.makedirs(path)
            images_path = path + '/images/'
            if not os.path.exists(images_path):
                os.makedirs(images_path)
            utility_funcs.make_video_from_rgb_imgs(full_obs, path)

            # Clean up images
            shutil.rmtree(images_path)

    return rewards_list


def take_action(env, agent, state, mapping_cache, use_lstm, policy_agent_mapping, state_init, agents_active, algo):
    "Take agents actions"
    group_id =  list(state.keys())[0]
    action_dict = {} if algo != "QMIX" else {group_id: []}
    for i, agent_id in enumerate(agents_active):
        a_state = state[agent_id] if algo != "QMIX" else state[group_id]
        if a_state is not None:
            if algo != "QMIX":
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
            else:
                policy_id = list(state_init.keys())[0]
                a_action, p_state_init = agent.compute_action(
                        a_state, state=state_init[policy_id],
                        policy_id=policy_id)
                state_init[policy_id] = p_state_init
            action_dict[group_id].append(a_action)

    return action_dict


def take_actions_for_coalition(env, agent, considered_player, state, mapping_cache, use_lstm, policy_agent_mapping, state_init, coalition, missing_agents_behaviour, agents_active, algo):
    'Return actions where each agent in coalition follow the policy, others play at random if replace_missing_players=="random" or do not move if replace_missing_players=="idle'
    actions = take_action(env, agent, state, mapping_cache,
                          use_lstm, policy_agent_mapping, state_init, agents_active, algo)
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
                action = np.random.randint(0, 7)

            elif missing_agents_behaviour == "idle":
                # Idle action
                action = 4  # env.ACTIONS["STAY"]

            else:
                raise ValueError(
                    f"Value: {missing_agents_behaviour} for parameter missing_agents_bahaviour is not valid. Valid values are: \"random\" \"random_player_action\" or \"idle\".")

            actions_for_coalition[agent_id] = action

    return actions_for_coalition
