import subprocess
import os
import csv
from rollout import rollout


def save_rewards(N, M, folder_name, agents_active, checkpoints):
    runs = ["run_1"]
    missing_agents_behaviours = ["idle","random", "random_player_action"]
    processes = []
    for checkpoint in checkpoints:
        for n in range(N):
            for run in runs:
                for behaviour in missing_agents_behaviours:
                    fname = f"{folder_name}/{folder_name}_{checkpoint}/{run}_{behaviour}_{n}.csv"
                    if not os.path.exists(fname):
                        command = f'python run_sv.py {checkpoint} --missing-agents-behaviour {behaviour} --exp-name {run}_{behaviour}_{n} --save-dir {folder_name}/{folder_name}_{checkpoint} --shapley-M {M} --agents-active {agents_active}'
                        processes.append(subprocess.Popen(command, shell=True))
                    else:
                        print("File already exists!!")
    for p in processes:
        p.wait()




if __name__ == "__main__":
    checkpoints = [3400]
    model_names = [
        r"results\PPO\PPO_prey_predator_f9fc2_00002_2_2021-05-18_14-10-03\checkpoint_003400\checkpoint-3400",
        # r"results\PPO\PPO_prey_predator_f9fc2_00003_3_2021-05-19_01-54-50\checkpoint_003400\checkpoint-3400",
        # r"results\PPO\PPO_prey_predator_f9fc2_00000_0_2021-05-18_14-10-03\checkpoint_003400\checkpoint-3400",
        # r"results\PPO\PPO_prey_predator_5964a_00001_1_2021-05-17_22-13-31\checkpoint_003400\checkpoint-3400",
        # r"results\PPO\PPO_prey_predator_5964a_00000_0_2021-05-17_22-13-31\checkpoint_003400\checkpoint-3400"
        ]
    save_rewards(N=1, M=1000, model_names=model_names, folder_name="rewards",  agents_active=3, checkpoints=checkpoints)

