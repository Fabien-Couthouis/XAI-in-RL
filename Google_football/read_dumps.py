import pickle
import os


# for file_path in os.listdir("replays"):
#     print(file_path)
with open("replays/score_20200206-111243889571.dump", 'rb') as file:
    dumps = pickle.load(file)
    print(len(dumps))
print(dumps[-2])
# for dump in dumps:
# obs = dump['observation']

# print(dump['debug']['action'])
# print(obs['ball_owned_team'])
