from knights_archers_zombies import env, raw_env, parallel_env, manual_control


def random_actions(env, agent, observation):
    return env.action_spaces[agent].sample()


env = parallel_env()
print(env.knight_list)

# observations = env.reset()
# done = False
# while not done:
#     env.render()
#     actions = {agent: random_actions(
#         env, agent, observations[agent]) for agent in env.agents}
#     observations, rewards, dones, infos = env.step(actions)
#     done = all(dones.values())
#     print(rewards)
