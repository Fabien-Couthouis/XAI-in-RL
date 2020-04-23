# XAI-in-RL: Google Football env

## Packages installation
```
sudo apt-get update && 
sudo apt-get install git cmake build-essential libgl1-mesa-dev libsdl2-dev \
libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
libdirectfb-dev libst-dev mesa-utils xvfb x11vnc libsdl-sge-dev python3-pip \
cmake libopenmpi-dev python3-dev zlib1g-dev
```
## Environment installation (Anaconda)
*Note: Install Anaconda first. See [official website](https://docs.anaconda.com/anaconda/install/linux/) for further information.*
```bash
conda env create -f conda_env_football.yml
```

## Google football environment installation
```bash
cd gfootball_env
pip3 install .
```
## Training agents

You can start training new agents by running the `working_multiagent_google.py` script.
Here is an example for a 11 vs 11 match (the --no-render argument is here to disable environment rendering. Remove it to watch training.):
```bash
python train.py --scenario-name "11_vs_11_stochastic" --num-agents 11 --num-iters 1000 --checkpoint-freq 100
```

Another example for the shapley_adversary.py scenario (3 players + the goal = 4 agents vs one adversaries player and a goal):
```bash
python train.py --scenario-name "shapley_adversary" --num-agents 4 --num-iters 1000 --checkpoint-freq 100
```

This will create files called "checkpoints" that will be used to store model weights every **--checkpoint-freq** iterations.
They will be located on `~/ray_results/default/...`.

### Usefull arguments:

* **--no-render** - Disable environment rendering.
* **--resume** - Resume training from last checkpoint available.
* **--compute-shapley** - Compute Shapley values for each controlled agent.
* **--save-replays** - Save video replays of the rollouts.
* **--policy-type** - Choose which policy type to use between: "PPOTF"(default), "PPOTORCH", "SACTF" and "IMPALATF". 

## Evaluating an agent

You can evaluate a trained agent by running the `test.py` script and passing the path to your model checkpoint file as an argument. Here is an example for a 11 vs 11 match:
```bash
python test.py path_to_your_checkpoint --env gfootball --run PPO --scenario-name "11_vs_11_stochastic" --num-agents 11 --episodes 20 --steps 10000
```
If you want to compute the agents' contributions using Shapley values, just pass the argument `--compute-shapley` to `rollout.py`. To plot Shapley value associated to each agent, pass the `--plot-shapley` argument.
