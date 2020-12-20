import subprocess
import os
import csv
from rollout import rollout


def save_rewards(N, M, folder_name, agent_speeds):
    runs = ["run_10", "run_11", "run_12", "run_13", "run_14"]
    missing_agents_behaviours = ["idle", "random_player", "random"]
    for n in range(N):
        for run in runs:
            for behaviour in missing_agents_behaviours:
                fname = f"{folder_name}/{run}_{behaviour}_{n}.csv"
                if not os.path.exists(fname):
                    print(fname)
                    command = f'python run.py --load-dir "saves/{run}/episode_200000/model" --missing-agents-behaviour {behaviour} --exp-name {run}_{behaviour}_{n} --save-dir {folder_name} --shapley-M {M} --num-episodes 1 --agent-speeds'
                    for speed in agent_speeds:
                        command += f" {speed}"
                    subprocess.run(command, shell=True)


def save_rewards_true_shapley(N, n_episodes, folder_name, agent_speeds):
    runs = ["run_10", "run_11", "run_12", "run_13", "run_14"]
    missing_agents_behaviours = ["idle", "random_player", "random"]
    for n in range(N):
        for run in runs:
            for behaviour in missing_agents_behaviours:
                fname = f"{folder_name}/{run}_{behaviour}_{n}.csv"
                if not os.path.exists(fname):
                    print(fname)
                    command = f'python run.py --load-dir "saves/{run}/episode_200000/model" --missing-agents-behaviour {behaviour} --exp-name {run}_{behaviour}_{n} --true-shapley --save-dir {folder_name} --num-episodes {n_episodes} --agent-speeds'
                    for speed in agent_speeds:
                        command += f" {speed}"
                    subprocess.run(command, shell=True)


def save_goal_agents(N, folder_name, agent_speeds):
    runs = ["run_10", "run_11", "run_12", "run_13", "run_14"]
    for run in runs:
        fname = f"{folder_name}/{run}_goal_agents_{N}.csv"

        if not os.path.exists(fname):
            command = f'python run.py --load-dir "saves/{run}/episode_200000/model" --exp-name {fname[:-4]} --save-dir {folder_name} --rollout --num-episodes {N} --agent-speeds'
            for speed in agent_speeds:
                command += f" {speed}"
            subprocess.run(command, shell=True)


if __name__ == "__main__":
    SPEEDS_EXP1 = [1.0, 1.0, 1.0, 1.3]
    SPEEDS_EXP2 = [0.2, 0.8, 2.0, 1.3]
    # save_rewards(N=1, M=1000, folder_name="rewards/exp1",
    #              agent_speeds=SPEEDS_EXP2)
    # save_goal_agents(2000, "goal_agents/exp1",
    #                  agent_speeds=SPEEDS_EXP1)
    save_rewards_true_shapley(N=1, n_episodes=1000, folder_name="rewards/true-shap-exp1",
                              agent_speeds=SPEEDS_EXP1)
