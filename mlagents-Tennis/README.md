[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"
[image3]: https://github.com/abhismatrix1/Tennis-MultiAgent/blob/master/training_graph.png "training graph"
[image4]: https://github.com/abhismatrix1/Tennis-MultiAgent/blob/master/training_graph2.png "training graph2"
[image5]: https://github.com/abhismatrix1/Tennis-MultiAgent/blob/master/training_graph_maddpg.png "training graph3"


# Notes Fabien
Fork d'un article sur Medium, code mis à jour pour mlagents 0.12. VOir environment.yml pour la liste des modules. Pas encore réussi à lancer l'entrainement, le modèle ne sort pas les actions pour chaque agent (à adapter).

# Tennis: Collaboration and Competition
links: [Medium article on improved stability](https://medium.com/p/27719f3cafd4)

### Introduction

This project is solving the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, the rewards that each agent received (without discounting) is added, to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Traiining result

1. With DDPG agent (single -agent RL)
The environment was solved in 3036 episodes. Average running scores graph is below
<a href="url"><img src="/training_graph.png " align="center" height="374" width="558" ></a>

2. Extended training
If the training is continued after solving the environment (i.e avg score > .5) we see that agent is continously improving reaching to avg score >3 in another 765 episodes. In this training i have reduced the noise. 
<a href="url"><img src="/training_graph2.png" align="center" height="374" width="558" ></a>

3. With maddpg agent (multi-agent reinforcement training.)
The environment was solved in 900 episodes.
<a href="url"><img src="/training_graph_maddpg.png" align="center" height="374" width="558" ></a>

### Getting Started

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name game python=3.6
	source activate game
	```
	- __Windows__: 
	```bash
	conda create --name game python=3.6 
	activate game
	```
	
2. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Next, install the **classic control** environment group by following the instructions [here](https://github.com/openai/gym#classic-control).
	- Then, install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
	
3. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/abhismatrix1/Tennis-MultiAgent.git
cd Tennis-MultiAgent/python
pip install .
```

4. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

5. Place the file in the this GitHub repository, in the `Tennis-MultiAgent/` folder, and unzip (or decompress) the file. 

### Instructions

Follow the instructions in `Tennis..ipynb` to get started with training your own agent!  




