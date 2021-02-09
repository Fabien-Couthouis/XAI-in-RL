# XAI-in-RL

## About

Explaining RL multiagent environments by computing Shapley values.
This is the official implementation of [Collective eXplainable AI: Explaining Cooperative strategies and agent contribution in Multiagent Reinforcement Learning with Shapley Values](arxivlink).

Experiments were conducted for two environments: [Sequential Social Dilemmas](https://github.com/eugenevinitsky/sequential_social_dilemma_games) and [OpenAI's Multiagent Particle](https://github.com/openai/multiagent-particle-envs).
The implementation for each environment is available in the corresponding subfolder.

## Installation

See Readme in subfolders. Instructions will be given for each env to reproduce the experiments presented in the article.

## Experiments

For each environment, you need to go to the `experiments` subfolder to launch the experiments:

```bash
cd multiagent-particles/experiments
```
```bash
cd sequential_social_dilemmas_games/experiments
```
### Multiagent Particle

To launch an experiment, you must place your models in the `saves` directory.
Several pretrained models have already been placed (#TODO rename model folders).

If you do not have a pre-trained model, you can train a new one with default settings using this command:

```bash
python run.py
```

To launch experiments with the provided models and reproduce the paper's results, use the following command:

```bash
python launch_experiments.py 
```

### Sequential Social Dilemmas


## Results

pass

## Paper Citation

If you use our work in your article, please cite the following paper:

```bibtex

```


