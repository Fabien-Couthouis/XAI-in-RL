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


def save_goal_agents(N, folder_name, agent_speeds):
    runs = ["run_10", "run_11", "run_12", "run_13", "run_14"]  #
    for run in runs:
        fname = f"{folder_name}/{run}_goal_agents_{N}.csv"

        if not os.path.exists(fname):
            command = f'python run.py --load-dir "saves/{run}/episode_200000/model" --exp-name {fname[:-4]} --save-dir {folder_name} --rollout --display --num-episodes {N} --agent-speeds'
            for speed in agent_speeds:
                command += f" {speed}"
            subprocess.run(command, shell=True)


if __name__ == "__main__":
    # save_rewards(N=3, M=1000, folder_name="rewards/exp2",
    #              agent_speeds=[0.2, 0.7, 1.3, 1.3])
    save_goal_agents(10000, "goal_agents/exp2",
                     agent_speeds=[0.1, 1.0, 3.0, 1.3])
