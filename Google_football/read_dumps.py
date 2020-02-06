import pickle
import os


def show_all():
    for file_path in os.listdir("replays"):
        if not file_path.endswith(".avi"):
            print(file_path)
            show_one(file_path)


def show_one(file_path, show_all=False):
    with open("replays/"+file_path, 'rb') as file:
        dumps = pickle.load(file)

        if show_all:
            print(dumps)
        else:
            for dump in dumps:
                obs = dump['observation']
                if obs['ball_owned_team'] != -1:
                    print(dump['debug']['action'])
                    print(obs['ball_owned_team'])


show_one("episode_done_20200206-160931439122.dump", True)
