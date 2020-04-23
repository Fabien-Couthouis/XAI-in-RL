import pickle
import os
"""
# Special action set that includes all the core actions in the same order.
full_action_set = [
    action_idle, action_left, action_top_left, action_top,
    action_top_right, action_right, action_bottom_right,
    action_bottom, action_bottom_left, action_long_pass,
    action_high_pass, action_short_pass, action_shot,
    action_keeper_rush, action_sliding, action_pressure,
    action_team_pressure, action_switch, action_sprint,
    action_dribble, action_release_direction,
    action_release_long_pass, action_release_high_pass,
    action_release_short_pass, action_release_shot,
    action_release_keeper_rush, action_release_sliding,
    action_release_pressure, action_release_team_pressure,
    action_release_switch, action_release_sprint,
    action_release_dribble
]
"""


def show_all():
    for file_path in os.listdir("replays"):
        if not file_path.endswith(".avi"):
            print(file_path)
            show_one(file_path)


def get_ball_owned_team(obs):
    if obs['ball_owned_team'] == -1:
        return None
    else:
        return "right" if obs['ball_owned_team'] == 1 else "left"


def get_empty_player_dict(file_path, team="left"):
    dumps = get_dumps(file_path)
    obs = dumps[0]['observation']
    players = obs[f"{team}_agent_controlled_player"]
    return {player: 0 for player in players}


def get_players_n_goals(file_path, team="left"):
    dumps = get_dumps(file_path)
    player_n_goals = get_empty_player_dict(file_path, team)
    scorer = None

    for frame, dump in enumerate(dumps, 1):
        obs = dump['observation']
        # End before all steps ended
        if frame == len(dumps) and obs['steps_left'] > 0:
            player_n_goals[scorer] += 1

        # Rewarded
        elif dump['reward'] > 0:
            player_n_goals[scorer] += 1

        if get_ball_owned_team(obs) == team:
            scorer = obs['ball_owned_player']

    return player_n_goals


def get_steps_per_player(file_path, team="left"):
    dumps = get_dumps(file_path)
    player_n_goals = get_empty_player_dict(file_path, team)
    for frame, dump in enumerate(dumps, 1):
        obs = dump['observation']
        current_player = obs['ball_owned_player']
        # Check if ball is controlled
        if current_player != -1:
            player_n_goals[current_player] += 1

    return player_n_goals


def get_dumps(file_path):
    with open("replays/"+file_path, 'rb') as file:
        dumps = pickle.load(file)
    return dumps


def show_one(file_path, show_all=False):
    dumps = get_dumps(file_path)
    print(dumps)

    if show_all:
        print(dumps)
    else:
        for e, dump in enumerate(dumps, 1):
            # print(dump['debug']['action'])

            obs = dump['observation']

            # if obs['ball_owned_team'] != -1:
            if e < 1600 and e > 1360:
                if obs['ball_owned_player'] != -1:
                    print("Step:", e)
                    # print(dump)

                    print(dump['debug']['action'])

                    print("team", obs['ball_owned_team'],
                          "player", obs['ball_owned_player'], "\n")


show_one("episode_done_20200418-141133054416.dump", True)
# scorers = get_scorers("score_20200206-174813545665.dump")
# print(scorers)
