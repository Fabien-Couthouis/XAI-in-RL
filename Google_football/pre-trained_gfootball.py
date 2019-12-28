import numpy as np
import gfootball.env as football_env
from gfootball.env.players.ppo2_cnn import Player

players = []
players.append(
    "ppo2_cnn:right_players=1,policy=gfootball_impala_cnn,checkpoint=CP_11_vs_11_easy_stochastic_v2")

env = football_env.create_environment(env_name='11_vs_11_stochastic', render=True,
                                      stacked=False, number_of_left_players_agent_controls=1, extra_players=players)

player_config = {'index': 0, 'left_players': 1, 'right_players': 0,
                 'policy': 'gfootball_impala_cnn', 'stacked': True, 'checkpoint': 'CP_11_vs_11_easy_stochastic_v2'}
agent = Player(player_config, env_config={})

observation = env.reset()
agent.reset()
rewards = []
done = False
while not done:
    action = agent.take_action([observation])
    observation, reward, done, info = env.step(action)
    rewards.append(reward)

print(f'Agent(s) obtained mean reward: {rewards}')
