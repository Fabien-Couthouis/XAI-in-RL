from ddpg_agent import Agent as DA
from maddpg_agent import Agent as MA
from gym_unity.envs import UnityEnv
import gym
import numpy as np
# from baselines import deepq
# from baselines import logger
import torch
import numpy as np
from collections import deque
import os
# import matplotlib.pyplot as plt

multi_env_name = "envs/Tennis"
multi_env = UnityEnv(multi_env_name, worker_id=1,
                     use_visual=False, multiagent=True)

# Examine environment parameters
print("Env parameters:", str(multi_env))

#
num_agents = multi_env.number_agents
action_size = multi_env.action_space.shape[0]
state_size = multi_env.observation_space.shape[0]
print("Number of agents is:", num_agents, ".\nAction size is:", action_size)

initial_observations = multi_env.reset()
# Look at observations
if len(multi_env.observation_space.shape) == 1:
    # Examine the initial vector observation
    print("Agent observations look like: \n{}".format(initial_observations[0]))
else:
    # Examine the initial visual observation
    print("Agent observations look like:")
    if multi_env.observation_space.shape[2] == 3:
        plt.imshow(initial_observations[0][:, :, :])
    else:
        plt.imshow(initial_observations[0][:, :, 0])


def select_agent(agent):
    if agent == 'ddpg':
        return DA(state_size, action_size, num_agents, fc1=400, fc2=300, seed=0, update_times=10)
    elif agent == 'maddpg':
        return MA(state_size, action_size, num_agents, fc1=400, fc2=300, seed=0, update_times=10)
    else:
        print("wrong selection. select from 1. ddpg, 2. maddpg")


agent = select_agent('maddpg')


scores = [] 


def solve_environment(n_episodes=20):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
    """
                           # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    global scores
    for i_episode in range(1, n_episodes+1):
        observations = multi_env.reset() # reset the environment
        agent.reset_random()              #reset noise object
        
        score = 0
        t=0
        reward_this_episode_1=0
        reward_this_episode_2=0
        while True:
            t=t+1
            action=agent.act(np.array(observations)).tolist()
            # env_info = env.step(np.array(action))[brain_name] 
            print(action)
            observations, rewards, dones, info = multi_env.step(action)

            agent.step(state, action, rewards, next_state, dones)
            state = next_state
            #print(reward)
            reward_this_episode_1+=rewards[0]
            reward_this_episode_2+=rewards[1]
            
            if np.any(dones):
                break 
 
        score = max(reward_this_episode_1,reward_this_episode_2)
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        # if np.mean(scores_window)>=2:
        #     print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
        #     torch.save(agent.critic_local.state_dict(), 'trained_weights/checkpoint_critic.pth')
        #     torch.save(agent.actor_local.state_dict(), 'trained_weights/checkpoint_actor.pth')
        #     break
    return 



solve_environment()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()




# try:
#     for episode in range(100):
#         observations = multi_env.reset()
#         done = False
#         episode_rewards = 0
#         while not done:
#             actions = [multi_env.action_space.sample()
#                        for agent in range(multi_env.number_agents)]
#             observations, rewards, dones, info = multi_env.step(actions)
#             episode_rewards += np.mean(rewards)
#             done = dones[0]
#         print("Total reward this episode: {}".format(episode_rewards))

# except KeyboardInterrupt:
#     multi_env.close()

multi_env.close()
# # logger.configure('./logs') # Çhange to log in a different directory
# # act = deepq.learn(
# #     env,
# #     "cnn", # conv_only is also a good choice for GridWorld
# #     lr=2.5e-4,
# #     total_timesteps=1000000,
# #     buffer_size=50000,
# #     exploration_fraction=0.05,
# #     exploration_final_eps=0.1,
# #     print_freq=20,
# #     train_freq=5,
# #     learning_starts=20000,
# #     target_network_update_freq=50,
# #     gamma=0.99,
# #     prioritized_replay=False,
# #     checkpoint_freq=1000,
# #     checkpoint_path='./logs', # Change to save model in a different directory
# #     dueling=True
# # )
# # print("Saving model to unity_model.pkl")
# # act.save("unity_model.pkl")
