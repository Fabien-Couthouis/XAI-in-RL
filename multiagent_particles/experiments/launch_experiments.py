import subprocess


if __name__ == "__main__":
    M = 50
    runs = ["run_10", "run_11", "run_12", "run_13", "run_14"]
    missing_agents_behaviours = ["random_player", "random", "idle"]
    for run in runs:
        for behaviour in missing_agents_behaviours:
            command = f'python run.py --load-dir "saves/{run}/episode_200000/model" --missing-agents-behaviour {behaviour} --exp-name {run}_{behaviour} --save-dir marginal_contributions --shapley-M {M} --num-episodes 1 '
            subprocess.run(command, shell=True)
