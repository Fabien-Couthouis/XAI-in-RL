from env_Cleaner import EnvCleaner

n_iter = 1000
env = EnvCleaner(3, 23, 0, 1000)
obs = env.reset()
for i in range(n_iter-1):
    print(i)
    obs = env.step([0, 3, 0])
    env.render()
