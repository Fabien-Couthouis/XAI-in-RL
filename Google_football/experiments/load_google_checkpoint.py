import gfootball.env as football_env
from gfootball.env.players.ppo2_cnn import Player

# !! needs dm-sonnet==1.36 & tensorflow==.15 !!
players = ["ppo2_cnn:right_players=1,policy=gfootball_impala_cnn,checkpoint=models/CP_11_vs_11_easy_stochastic_v2"]
env = football_env.create_environment(env_name='11_vs_11_easy_stochastic', render=False, write_full_episode_dumps=True,
                                      stacked=False, number_of_left_players_agent_controls=1, write_video=True, logdir=".")

player_config = {'index': 0, 'left_players': 11, 'right_players': 0,
                 'policy': 'gfootball_impala_cnn', 'stacked': True, 'checkpoint': 'models/CP_11_vs_11_easy_stochastic_v2'}
agent = Player(player_config, env_config={})

observations = env.reset()
agent.reset()
done = False
while not done:
    actions = agent.take_action([observations])
    # print(actions)
    observation, reward, done, info = env.step(actions)
