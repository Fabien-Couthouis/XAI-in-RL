import subprocess
import os


def save_rewards(N, M, folder_name, checkpoints, missing_agents_behaviours, agent_speeds):
    processes = []
    for checkpoint_i, checkpoint in enumerate(checkpoints):
        for n in range(N):
            for behaviour in missing_agents_behaviours:
                save_dir = "{folder_name}/{folder_name}_model_{checkpoint_i}"
                fname = f"{save_dir}/model_{checkpoint_i}_{behaviour}_{n}_speeds={str(agent_speeds)}.csv"
                if not os.path.exists(fname):
                    command = f'python run_sv.py {checkpoint} --run PPO --missing-agents-behaviour {behaviour} --exp-name model_{checkpoint_i}_{behaviour}_{n} --save-dir {save_dir} --shapley-M {M} --agent-speeds'
                    for s in agent_speeds:
                        command += f' {s}'
                    processes.append(subprocess.Popen(command, shell=True))
                else:
                    print("File already exists!!")
    for p in processes:
        p.wait()


if __name__ == "__main__":
    checkpoints = [
        r"results/PPO/PPO_prey_predator_f9fc2_00002_2_2021-05-18_14-10-03/checkpoint_003400/",
        r"results/PPO/PPO_prey_predator_f9fc2_00003_3_2021-05-19_01-54-50/checkpoint_003400/",
        r"results/PPO/PPO_prey_predator_f9fc2_00000_0_2021-05-18_14-10-03/checkpoint_003400/",
        r"results/PPO/PPO_prey_predator_5964a_00001_1_2021-05-17_22-13-31/checkpoint_003400/",
        r"results/PPO/PPO_prey_predator_5964a_00000_0_2021-05-17_22-13-31/checkpoint_003400/"
    ]
    missing_agents_behaviours = ["idle", "random", "random_player_action"]

    agent_speeds = [1.3, 1.0, 1.0, 1.0]
    save_rewards(N=1, M=1000, folder_name="rewards", checkpoints=checkpoints,
                 missing_agents_behaviours=missing_agents_behaviours, agent_speeds=agent_speeds)

    agent_speeds = [1.3, 0.2, 0.8, 2.0]
    save_rewards(N=1, M=1000, folder_name="rewards", checkpoints=checkpoints,
                 missing_agents_behaviours=missing_agents_behaviours, agent_speeds=agent_speeds)
