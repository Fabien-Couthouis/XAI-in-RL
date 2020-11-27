import subprocess
import os
import csv
from rollout import rollout


def save_rewards(N, M, folder_name):
    runs = ["run_1"]
    missing_agents_behaviours = ["idle", "random_player_action", "random"]
    for n in range(N):
        for run in runs:
            for behaviour in missing_agents_behaviours:
                fname = f"{folder_name}/{run}_{behaviour}_{n}.csv"
                if not os.path.exists(fname):
                    command = f'python run.py models/harvest 7840 --missing-agents-behaviour {behaviour} --exp-name {run}_{behaviour}_{n} --save-dir {folder_name} --shapley-M {M}'
                    subprocess.run(command, shell=True)


if __name__ == "__main__":
    save_rewards(N=1, M=100, folder_name="rewards")
