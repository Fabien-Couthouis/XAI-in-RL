import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import csv
import random
from maddpg.trainer.maddpg import MADDPGAgentTrainer
from env_wrapper import EnvWrapper
from utils import get_trainers, mlp_model


def rollout(env, arglist, trainers, considered_player=None, coalition=None, missing_agents_bahaviour="random_player"):
    'Play simulations'
    num_adversaries = min(env.n, arglist.num_adversaries)

    obs_n = env.reset()
    episode_step = 0
    rollout_step = 0
    total_goal_agents, total_agent_rewards, total_episode_rewards = [], [], []

    for episode in range(arglist.num_episodes):
        goal_agents, rewards, agent_rewards = [], [], [[]
                                                       for _ in range(env.n)]
        while True:
            # get action
            action_n = take_actions_for_coalition(
                trainers, obs_n, considered_player, coalition, env, missing_agents_bahaviour, num_adversaries)

            # environment step
            obs_n, rew_n, done_n, _ = env.step(action_n)

            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)

            # increment global step counter
            rollout_step += 1

            # for displaying
            if arglist.display:
                env.render(mode='other')
                # time.sleep(0.01)

            rewards.append(sum(rew_n[:-num_adversaries]))
            if rew_n[0] > 0:
                # prey  got caught
                goal_agents.append(env.winning_agent().name)
                print(env.winning_agent().name)
            for i, rew in enumerate(rew_n):
                agent_rewards[i].append(rew)

            if done or terminal:
                obs_n = env.reset()
                episode_step = 0
                total_agent_rewards.append(agent_rewards)
                total_episode_rewards.append(rewards)
                total_goal_agents.append(goal_agents)
                # print(total_goal_agents)
                break

    rollout_info = {"goal_agents": total_goal_agents,
                    "episode_rewards": total_episode_rewards, "agent_rewards": total_agent_rewards}
    return rollout_info


def take_action(trainers, obs_n):
    'Return: list of actions corresponding to each agent'
    action_n = [agent.action(obs)
                for agent, obs in zip(trainers, obs_n)]
    return action_n


def take_actions_for_coalition(trainers, obs_n, considered_player, coalition, env, missing_agents_bahaviour="random_player", num_adversaries=1):
    'Return actions where each agent in coalition follow the policy, others play at random'
    actions = take_action(trainers, obs_n)
    if coalition is None or considered_player is None:
        return actions

    actions_for_coalition = actions.copy()
    if missing_agents_bahaviour == "idle":
        idle_actions = env.idle_actions()
    if missing_agents_bahaviour == "random":
        random_actions = env.random_actions()

    n_good_agents = min(env.n, env.n-num_adversaries)
    agents_without_player = [agent_id for agent_id in range(
        n_good_agents) if agent_id != considered_player]
    for agent_id in range(n_good_agents):
        if agent_id not in coalition:
            # take dummy action
            if missing_agents_bahaviour == "random_player":
                random_agent = random.choice(agents_without_player)
                random_agent_action = actions[random_agent]
                actions_for_coalition[agent_id] = random_agent_action
            elif missing_agents_bahaviour == "idle":
                actions_for_coalition[agent_id] = idle_actions[agent_id]
            elif missing_agents_bahaviour == "random":
                actions_for_coalition[agent_id] = random_actions[agent_id]
            else:
                raise ValueError(
                    f"Value: {missing_agents_bahaviour} for parameter missing_agents_bahaviour is not valid. Valid values are: \"random\" \"random_player\" or \"idle\".")
    return actions_for_coalition


def take_actions_for_coalition_observations(trainers, obs_n, considered_player, coalition, env, missing_agents_bahaviour="random_player", num_adversaries=1):
    'Return actions where each agent in coalition follow the policy, others play at random'
    if coalition is None or considered_player is None:
        return take_action(trainers, obs_n)

    if missing_agents_bahaviour == "idle":
        idle_actions = env.idle_actions()
    if missing_agents_bahaviour == "random":
        random_actions = env.random_actions()

    n_good_agents = min(env.n, env.n-num_adversaries)
    agents_without_player = [agent_id for agent_id in range(
        n_good_agents) if agent_id != considered_player]
    for agent_id in range(n_good_agents):
        if agent_id not in coalition:
            # take dummy action
            if missing_agents_bahaviour == "random_player":
                random_agent = random.choice(agents_without_player)
                random_agent_action = actions[random_agent]
                actions_for_coalition[agent_id] = random_agent_action
            elif missing_agents_bahaviour == "idle":
                actions_for_coalition[agent_id] = idle_actions[agent_id]
            elif missing_agents_bahaviour == "random":
                actions_for_coalition[agent_id] = random_actions[agent_id]
            else:
                raise ValueError(
                    f"Value: {missing_agents_bahaviour} for parameter missing_agents_bahaviour is not valid. Valid values are: \"random\" \"random_player\" or \"idle\".")
    return actions_for_coalition
