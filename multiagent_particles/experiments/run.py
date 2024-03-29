import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import csv
import maddpg.common.tf_util as U
from env_wrapper import EnvWrapper
from utils import get_trainers, mlp_model
from shapley_values import monte_carlo_shapley_estimation, shapley_values
from rollout import rollout


def parse_args():
    parser = argparse.ArgumentParser(
        "Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str,
                        default="simple_tag", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int,
                        default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int,
                        default=1000000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int,
                        default=1, help="number of adversaries")
    parser.add_argument("--good-policy", type=str,
                        default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str,
                        default="ddpg", help="policy of adversaries")
    parser.add_argument(
        "--agent-speeds", nargs="+", default=[1.0, 1.0, 1.0, 1.3], help="Speed of agents (first are adversaries)")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float,
                        default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=128,
                        help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None,
                        help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/",
                        help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="saves/model",
                        help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore-episode", type=int, default=0)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000,
                        help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/",
                        help="directory where benchmark data is saved")
    parser.add_argument("--shapley-M", type=int, default=None,
                        help="compute or not shapley values with given number of simulation episodes (M)")
    parser.add_argument("--true-shapley", action="store_true",
                        default=False, help="Do not use Monte Carlo approximations")

    parser.add_argument("--missing-agents-behaviour", type=str, default="random_player",
                        help="behaviour of agent not in the coalition: random_player (take a random player mode from a from in the coalition) or random (random move) or idle (do not move)")
    parser.add_argument("--rollout", action="store_true", default=False)
    return parser.parse_args()


# def make_env(scenario_name, benchmark=False):
#     from multiagent.environment import MultiAgentEnv
#     import multiagent.scenarios as scenarios

#     # load scenario from script
#     scenario = scenarios.load(scenario_name + ".py").Scenario()
#     # create world
#     world = scenario.make_world()
#     # create multiagent environment
#     if benchmark:
#         env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
#                             scenario.observation, scenario.benchmark_data)
#     else:
#         env = MultiAgentEnv(world, scenario.reset_world,
#                             scenario.reward, scenario.observation)
#     return env


def train(env, arglist, trainers):

    episode_rewards = [0.0]  # sum of rewards for all agents
    agent_rewards = [[0.0]
                     for _ in range(env.n)]  # individual agent reward
    agent_info = [[[]]]  # placeholder for benchmarking info
    saver = tf.train.Saver()
    obs_n = env.reset()
    episode_step = 0
    train_step = 0
    t_start = time.time()

    print('Starting iterations...')
    while True:
        # get action
        action_n = [agent.action(obs)
                    for agent, obs in zip(trainers, obs_n)]
        # environment step
        new_obs_n, rew_n, done_n, info_n = env.step(action_n)
        episode_step += 1
        done = all(done_n)
        terminal = (episode_step >= arglist.max_episode_len)
        # collect experience
        for i, agent in enumerate(trainers):
            agent.experience(
                obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
        obs_n = new_obs_n

        for i, rew in enumerate(rew_n):
            episode_rewards[-1] += rew
            agent_rewards[i][-1] += rew

        if done or terminal:
            obs_n = env.reset()
            episode_step = 0
            episode_rewards.append(0)
            for a in agent_rewards:
                a.append(0)
            agent_info.append([[]])

        # increment global step counter
        train_step += 1

        # for benchmarking learned policies
        if arglist.benchmark:
            for i, info in enumerate(info_n):
                agent_info[-1][i].append(info_n['n'])
            if train_step > arglist.benchmark_iters and (done or terminal):
                file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                print('Finished benchmarking, now saving...')
                with open(file_name, 'wb') as fp:
                    pickle.dump(agent_info[:-1], fp)
                break
            continue

        # for displaying learned policies
        if arglist.display:
            time.sleep(0.1)
            env.render()
            continue

        # update all trainers, if not in display or benchmark mode
        for agent in trainers:
            agent.preupdate()
        for agent in trainers:
            loss = agent.update(trainers, train_step)

        episode = len(episode_rewards) + arglist.restore_episode
        # save model, display training output
        if (done or terminal) and (episode % arglist.save_rate == 0):
            mean_reward = np.mean(episode_rewards[-arglist.save_rate:])
            agents_episode_rewards = [
                np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards]
            time_spent = round(time.time()-t_start, 3)

            U.save_state(
                f"{arglist.save_dir}/{arglist.exp_name}/episode_{episode}/model", saver=saver)
            # print statement depends on whether or not there are adversaries
            if num_adversaries == 0:
                print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                    train_step, episode, mean_reward, time_spent))
            else:
                print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                    train_step, episode, mean_reward, agents_episode_rewards, time_spent))
            t_start = time.time()

            # Keep track of rewards
            save_rewards(arglist, episode, mean_reward,
                         agents_episode_rewards)

        # saves final episode reward for plotting training curve later
        if episode >= arglist.num_episodes:
            print('...Finished total of {} episodes.'.format(episode))
            break


def save_rewards(arglist, episode, mean_reward, agents_episode_rewards):
    'Keep track of rewards in csv file'
    file_name = f"{arglist.save_dir}/{arglist.exp_name}/rewards.csv"
    with open(file_name, "a", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow([episode, mean_reward] +
                        [ag_rew for ag_rew in agents_episode_rewards])


def save_goal_agents(arglist, rollout_info):
    file_name = f"{arglist.exp_name}.csv"
    goal_agents_list = rollout_info["goal_agents"]
    episode_rewards = np.sum(rollout_info["episode_rewards"], axis=-1)
    num_good_agents = min(env.n, env.n-arglist.num_adversaries)

    with open(file_name, "a", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        for episode in range(len(goal_agents_list)):
            goal_agents = goal_agents_list[episode]

            episode_reward = episode_rewards[episode]
            row = [episode, episode_reward]
            row.extend([goal_agents.count(f"agent {agent}")
                        for agent in range(num_good_agents)])
            writer.writerow(row)


if __name__ == '__main__':
    arglist = parse_args()
    # Create environment
    env = EnvWrapper(arglist.scenario, arglist.benchmark,
                     agent_speeds=arglist.agent_speeds)
    with U.single_threaded_session():
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        # Initialize
        U.initialize()
        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.rollout or arglist.shapley_M or arglist.true_shapley or arglist.restore_episode != 0 or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        if arglist.true_shapley:
            shapley_values(env, arglist, trainers)
        elif arglist.shapley_M is not None:
            monte_carlo_shapley_estimation(
                env, arglist, trainers)
        elif arglist.rollout:
            rollout_info = rollout(env, arglist, trainers)
            save_goal_agents(arglist, rollout_info)

        else:
            train(env, arglist, trainers)
