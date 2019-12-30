import numpy as np
import torch
import time
from itertools import combinations
from utilities.env_wrapper import EnvWrapper
from utilities.util import *
from models.maddpg import MADDPG
from arguments import *


def load_model(path):
    checkpoint = torch.load(path)
    model = Model[model_name]
    strategy = Strategy[model_name]

    if args.target:
        target_net = model(args)
        behaviour_net = model(args, target_net)
    else:
        behaviour_net = model(args)

    checkpoint = torch.load(
        path, map_location='cpu') if not args.cuda else torch.load(path)
    behaviour_net.load_state_dict(checkpoint['model_state_dict'])
    behaviour_net = behaviour_net.cuda().eval() if args.cuda else behaviour_net.eval()
    return behaviour_net


def take_action(behaviour_net, env, state, last_action):
    'Return: list of actions corresponding to each agent'
    if behaviour_net is None:
        actions = env.get_random_actions()
        return actions

    cuda = args.cuda and torch.cuda.is_available()
    if last_action is None:
        action = cuda_wrapper(torch.zeros(
            (1, args.agent_num, args.action_dim)), cuda=cuda)

    state = cuda_wrapper(prep_obs(state).contiguous().view(
        1, args.agent_num, args.obs_size), cuda=cuda)

    action_logits = behaviour_net.policy(
        state, schedule=None, last_act=last_action, last_hid=None, info={})
    action = select_action(args, action_logits, status='test')

    _, actual = translate_action(args, action, env)
    return actual  # list


def play(env, combination=None, behaviour_net=None, num_episodes=2000, max_steps_per_episode=500, render=True):
    actions = None
    total_rewards = []

    for episode in range(1, num_episodes+1):
        observations = env.reset()
        episode_rewards = []

        for step in range(max_steps_per_episode):
            if combination is not None:
                actions = take_actions_for_combination(
                    combination, behaviour_net, env, observations, actions)
            else:
                actions = take_action(behaviour_net, env,
                                      observations, actions)

            observations, rewards, dones, info = env.step(actions)
            episode_rewards.append(rewards[0])

            if render:
                env.render()
                time.sleep(0.1)

            if any(dones):
                break

        sum_rewards = sum(episode_rewards)
        episode_rewards.clear()
        total_rewards.append(sum_rewards)
        print("End of episode ", episode, "\nTotal reward is ", sum_rewards)
    return total_rewards


def get_combinations(features):
    combinations_list = []
    for i in range(1, len(features)+1):
        oc = combinations(features, i)
        for c in oc:
            combinations_list.append(list(c))
    return combinations_list


def take_actions_for_combination(combination, behaviour_net, env, state, last_action):
    'Return actions where each agent in combination plays at random'
    actions = take_action(behaviour_net, env, state, last_action)
    random_actions = env.get_random_actions()

    for agent_id in combination:
        actions[agent_id] = random_actions[agent_id]

    return actions


if __name__ == "__main__":
    env = EnvWrapper("simple_tag")
    model_path = "model_save/simple_tag_maddpg/model.pt"
    behaviour_net = load_model(model_path)

    agents_ids = range(env.get_num_of_agents())

    possible_combinations = get_combinations(agents_ids)
    all_rewards = []
    for combination in possible_combinations:
        total_rewards = play(env, combination=combination,
                             behaviour_net=behaviour_net, num_episodes=3, render=False)
        all_rewards.append(
            dict(random_agents=combination, rewards=total_rewards))
        print(combination, "effectued")

    print(all_rewards)
