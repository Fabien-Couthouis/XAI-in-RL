import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import random
import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers
from env_wrapper import EnvWrapper
from utils import get_trainers, mlp_model


def rollout(env, arglist, coalition=None, missing_agents_bahaviour="random_player"):
    tf.reset_default_graph()
    with U.single_threaded_session():
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(
            env, num_adversaries, obs_shape_n, arglist)

        # Initialize
        U.initialize()
        U.load_state(arglist.load_dir)

        obs_n = env.reset()
        episode_step = 0
        rollout_step = 0
        goal_agents = []
        goal_agents, total_agent_rewards, total_episode_rewards = [], [], []

        for episode in range(arglist.num_episodes):
            rewards, agent_rewards = [], []
            while True:
                # get action
                action_n = take_actions_for_coalition(
                    trainers, obs_n, coalition, env, missing_agents_bahaviour)

                # environment step
                obs_n, rew_n, done_n, info_n = env.step(action_n)

                episode_step += 1
                done = all(done_n)
                terminal = (episode_step >= arglist.max_episode_len)

                # increment global step counter
                rollout_step += 1

                # for displaying
                if arglist.display:
                    time.sleep(0.1)
                    env.render()
                    continue

                # update all trainers, if not in display or benchmark mode
                for agent in trainers:
                    agent.preupdate()
                for agent in trainers:
                    agent.update(trainers, rollout_step)

                agent_rewards.append(rew_n)
                rewards.append(sum(rew_n[:-num_adversaries]))
                if done or terminal:
                    obs_n = env.reset()
                    episode_step = 0
                    total_agent_rewards.append(agent_rewards)
                    total_episode_rewards.append(rewards)
                    goal_agents.append(env.winning_agent().name)
                    break

    rollout_info = {"goal_agents": goal_agents,
                    "episode_rewards": total_episode_rewards, "agent_rewards": total_agent_rewards}
    return rollout_info


def take_action(trainers, obs_n):
    'Return: list of actions corresponding to each agent'
    action_n = [agent.action(obs)
                for agent, obs in zip(trainers, obs_n)]
    return action_n


def take_actions_for_coalition(trainers, obs_n, coalition, env, missing_agents_bahaviour="random_player"):
    'Return actions where each agent in coalition follow the policy, others play at random'
    actions = take_action(trainers, obs_n)
    if coalition is None:
        return actions

    if missing_agents_bahaviour == "random_player_action":
        actions_for_coalition = actions.copy()
        for agent_id in actions.keys():
            if agent_id not in coalition:
                # take dummy action
                random_agent = random.choice(coalition)
                random_agent_action = actions[random_agent]
                actions_for_coalition[agent_id] = random_agent_action
    else:
        if missing_agents_bahaviour == "idle":
            actions_for_coalition = env.idle_actions()
        else:  # missing_agents_bahaviour == "random":
            actions_for_coalition = env.random_actions()
        for agent_id in coalition:
            actions_for_coalition[agent_id] = actions[agent_id]

    return actions_for_coalition


def save_reward(arglist, feature, m, marginal_contribution):
    'Keep track of marginal contributions in csv file while computing shapley values'
    file_name = f"{arglist.save_dir}/{arglist.exp_name}.csv"

    with open(file_name, "a", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(
            [feature, m, arglist.missing_agent_behaviour, marginal_contribution])
