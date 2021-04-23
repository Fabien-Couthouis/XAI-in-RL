import subprocess
import os
import csv
from rollout import rollout


def save_rewards(N, M, model_name, folder_name, agents_fov, agents_active, checkpoints):
    runs = ["run_1", "run_2", "run_3"]
    missing_agents_behaviours = ["idle"] #"random", "random_player_action"]
    processes = []
    for checkpoint in checkpoints:
        for n in range(N):
            for run in runs:
                for behaviour in missing_agents_behaviours:
                    fname = f"{folder_name}/{folder_name}_{checkpoint}/{run}_{behaviour}_{n}.csv"
                    if not os.path.exists(fname):
                        command = f'python run.py {model_name} {checkpoint} --missing-agents-behaviour {behaviour} --exp-name {run}_{behaviour}_{n} --save-dir {folder_name}/{folder_name}_{checkpoint} --shapley-M {M} --agents-active {agents_active}'
                        if agents_fov is not None:
                            command += ' --agents-fov'
                            for fov in agents_fov:
                                command += f" {fov}"
                        processes.append(subprocess.Popen(command, shell=True))
                    else:
                        print("File already exists!!")
    for p in processes:
        p.wait()


def save_rewards_metrics(n_episodes, save_folder_name, checkpoint_folder, checkpoints, agents_fov=None, agents_active=6):
    processes = []
    for checkpoint in checkpoints:
        fname = f"{save_folder_name}/social_metrics_ckpt_{checkpoints}.csv"
        if not os.path.exists(fname):
            command = f'python run.py {checkpoint_folder} {checkpoint} --num-rollouts {n_episodes} --exp-name social_metrics_ckpt_{checkpoint} --social-metrics --save-dir {save_folder_name} --agents-active {agents_active}'
            if agents_fov is not None:
                command += ' --agents-fov'
                for fov in agents_fov:
                    command += f" {fov}"
            processes.append(subprocess.Popen(command, shell=True))
        else:
            print("File already exists!!")
    for p in processes:
        p.wait()


if __name__ == "__main__":
    # AGENTS_FOV = [40, 30, 20, 10, 5]
    checkpoints = list(range(1000, 9000, 1000))
    save_rewards(N=1, M=1000, model_name="models/harvest_5_agents_additionnal", folder_name="rewards3", agents_fov=None, agents_active=5, checkpoints=checkpoints)

    
    # save_rewards_metrics(100, "social_metrics4", "models/harvest_5_agents_additionnal",
    #                      checkpoints, agents_active=5)
