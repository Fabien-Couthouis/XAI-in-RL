import gfootball.env as football_env
from gfootball.env.players.ppo2_cnn import Player

env = football_env.create_environment(
    env_name='11_vs_11_stochastic', render=True)
player_config = {"checkpoint": "11_vs_11_easy_stochastic_v2"}
player = Player(player_config=player_config, env_config={})

env.reset()
done = False
while not done:
    action = player.take_action()
    observation, reward, done, info = env.step(action)
