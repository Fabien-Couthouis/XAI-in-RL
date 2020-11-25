"""Defines a multi-agent controller to rollout environment episodes w/
   agent policies."""

import argparse
import json
import numpy as np
import os
import shutil
import sys

import ray
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.cloudpickle import cloudpickle
from ray.rllib.evaluation.sample_batch import DEFAULT_POLICY_ID
# from ray.rllib.evaluation.sampler import clip_action

from models.conv_to_fc_net import ConvToFCNet
import utility_funcs


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


def visualizer_rllib(args):
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
    if hasattr(agent, "local_evaluator"):
        env = agent.local_evaluator.env

    if args.save_video:
        shape = env.base_map.shape
        full_obs = [np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
                    for i in range(config["horizon"])]

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

    steps = 0
    while steps < (config['horizon'] or steps + 1):
        state = env.reset()
        done = False
        reward_total = 0.0
        while not done and steps < (config['horizon'] or steps + 1):
            if multiagent:
                action_dict = {}
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
                action = action_dict
            else:
                if use_lstm[DEFAULT_POLICY_ID]:
                    action, state_init, _ = agent.compute_action(
                        state, state=state_init)
                else:
                    action = agent.compute_action(state)

            if agent.config["clip_actions"]:
                # clipped_action = clip_action(action, env.action_space)
                next_state, reward, done, _ = env.step(action)
            else:
                next_state, reward, done, _ = env.step(action)

            if multiagent:
                done = done["__all__"]
                reward_total += sum(reward.values())
            else:
                reward_total += reward

            if args.save_video:
                rgb_arr = env.map_to_colors()
                full_obs[steps] = rgb_arr.astype(np.uint8)

            steps += 1
            state = next_state
        print("Episode reward", reward_total)

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


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Evaluates a reinforcement learning agent '
                    'given a checkpoint.')

    # required input parameters
    parser.add_argument(
        'result_dir', type=str, help='Directory containing results')
    parser.add_argument('checkpoint_num', type=str, help='Checkpoint number.')

    # optional input parameters
    parser.add_argument(
        '--run',
        type=str,
        help='The algorithm or model to train. This may refer to '
             'the name of a built-on algorithm (e.g. RLLib\'s DQN '
             'or PPO), or a user-defined trainable function or '
             'class registered in the tune registry. '
             'Required for results trained with flow-0.2.0 and before.')
    parser.add_argument(
        '--num-rollouts',
        type=int,
        default=1,
        help='The number of rollouts to visualize.')
    parser.add_argument(
        '--save-video',
        action='store_true',
        help='whether to save a movie or not')
    parser.add_argument(
        '--render',
        action='store_true',
        help='whether to watch the rollout while it happens')
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    ray.init(num_cpus=2)
    visualizer_rllib(args)
