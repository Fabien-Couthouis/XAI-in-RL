import subprocess
import os
import csv
from rollout import rollout


def save_rewards(N, M, folder_name, agents_fov, agents_active):
    runs = ["run_2"]
    missing_agents_behaviours = ["idle", "random", "random_player_action"]
    processes = []
    for n in range(N):
        for run in runs:
            for behaviour in missing_agents_behaviours:
                fname = f"{folder_name}/{run}_{behaviour}_{n}.csv"
                if not os.path.exists(fname):
                    command = f'python run.py models/harvest_6_agents 7840 --missing-agents-behaviour {behaviour} --exp-name {run}_{behaviour}_{n} --save-dir {folder_name} --shapley-M {M} --agents-active {agents_active} --agents-fov'
                    for fov in agents_fov:
                        command += f" {fov}"
                    processes.append(subprocess.Popen(command, shell=True))
                else:
                    print("File already exists!!")
    for p in processes:
        p.wait()


if __name__ == "__main__":
    AGENTS_FOV = [40, 30, 20, 10, 5]
    save_rewards(N=1, M=1000, folder_name="rewards", agents_fov=AGENTS_FOV, agents_active=5)
