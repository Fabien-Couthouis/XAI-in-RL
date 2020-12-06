import subprocess
import os
import csv
from rollout import rollout


def save_rewards(N, M, folder_name):
    runs = ["run_1"]
    missing_agents_behaviours = ["idle", "random", "random_player_action"]
    processes = []
    for n in range(N):
        for run in runs:
            for behaviour in missing_agents_behaviours:
                fname = f"{folder_name}/{run}_{behaviour}_{n}.csv"
                if not os.path.exists(fname):
                    command = f'python run.py models/harvest 7840 --missing-agents-behaviour {behaviour} --exp-name {run}_{behaviour}_{n} --save-dir {folder_name} --shapley-M {M}'
                    processes.append(subprocess.Popen(command, shell=True))
    for p in processes: p.wait()


if __name__ == "__main__":
    save_rewards(N=1, M=1000, folder_name="rewards")
