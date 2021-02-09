# XAI-in-RL

## About

Explaining RL multiagent environments by computing Shapley values.
This is the official implementation of [Collective eXplainable AI: Explaining Cooperative strategies and agent contribution in Multiagent Reinforcement Learning with Shapley Values](arxivlink). (TODO: add link to article)

Experiments were conducted for two environments: [Sequential Social Dilemmas](https://github.com/eugenevinitsky/sequential_social_dilemma_games) and [OpenAI's Multiagent Particle](https://github.com/openai/multiagent-particle-envs).
The implementation for each environment is available in the corresponding subfolder.

## Installation

See Readme in subfolders. Instructions will be given for each env to reproduce the experiments presented in the article.

To use any of the pre-configured Python environments, you need [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

## Experiments

There is a unified API to launch and configured experiments for both environments.
For each environment, you need to go to the `experiments` subfolder to launch the experiments:

```bash
cd multiagent-particles/experiments
```
```bash
cd sequential_social_dilemmas_games/experiments
```

To launch an experiment, you must place your models in the `saves` (for Multiagent Particle) or `models` (for Sequential Social Dilemmas) directory.
Several pretrained models have already been placed (TODO: rename model folders).

If you do not have a pre-trained model, you can train a new one with default settings using this command:

```bash
python run.py
```

To launch experiments with the provided models and reproduce the paper's results, use the following command:

```bash
python launch_experiments.py 
```

To get help and a list of all arguments (to configure the environment and the experiments) you can add `--help` after any of the two commands above.

## Plots

Once the experiments have been ran, it is possible to plot different diagrams using the following command:

```bash
python plots.py rewards --plot_type your_plot_type
```
`plot_type` can be one of:
- `model_rewards` for plotting global rewards obtained during training phase by a model. It is required to give the model location with `--model_dir your_model_dir`.
- `shapley_barchart` for plotting a barchart featuring the Shapley value of each agent computed using a specific player exclusion method. The player exclusion method must be given as argument with `--exclusion_method`, it can be one of `noop`, `random` or `random_player`.
- `shapley_cat_plot` for plotting a categorical plot featuring the Shapley value of each agent for each model used to run the experiments


## Results

pass

## Paper Citation

If you use our work in your article, please cite the following paper:

```bibtex

```


