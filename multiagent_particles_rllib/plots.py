import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use('bmh')

BARCHAR_TEXTPROPS = {"fontsize": 12, "weight": "bold"}
PIECHART_TEXTPROPS = {"fontsize": 12}


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Plots different diagrams from already ran experiments.')

    # required input parameters
    parser.add_argument(
        '--result-dir', type=str, default="multiagent_particles_rllib/rewards_mparticle_rllib", help='Directory containing results of experiments')

    # optional input parameters
    parser.add_argument(
        '--num-agents',
        type=int,
        default=3,
        help='The number of agents.')
    parser.add_argument("--plot-type", type=str, default="shapley_cat_plot",
                        help="type of diagram to plot: shapley_barchart/model_rewards/cat_plot")
    parser.add_argument("--model-dir", type=str, default="models/harvest_6_agents/",
                        help='model location, required when plotting model rewards')
    return parser


def plot_shap_barchart(shapley_values: List[float], agent_names: List[str]):
    'Plot barchart: agent with corresponding shapley value'
    print("shapley_values", shapley_values)

    fig, ax = plt.subplots()
    data_df = pd.DataFrame(
        {agent_names[i]: [shapley_values[i]] for i in range(len(shapley_values))})
    ax = sns.barplot(data=data_df, orient="h")

    ax.set_title("Shapley value for each agent",
                 BARCHAR_TEXTPROPS)
    ax.set_xlabel("Shapley value (contribution of each agent)",
                  BARCHAR_TEXTPROPS)
    ax.set_ylabel('Agent names', BARCHAR_TEXTPROPS)
    fig.tight_layout()
    return fig, ax


def plot_model_rewards(file_path: str):
    'Plot: reward per episode for each agent'
    fig, ax = plt.subplots()

    df = pd.read_csv(file_path)

    ax = sns.lineplot(x="timesteps_total", y="episode_reward_mean", data=df)
    ax.set_xlabel('Training episode number')
    ax.set_ylabel('Mean global reward')
    # ax.set_title(f"Reward per episode on {len(agent_rewards[0])} episodes")

    # Set 1e label
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    fig.tight_layout()
    return fig, ax


# def cat_plot(shapley_values: List[List[float]], agent_names: List[str], method_names: List[str]):
# def cat_plot(player_names: List[str], shapley_values: List[float], methods: List[str]):
def cat_plot(data_df):
    fig, ax = plt.subplots()
    g = sns.catplot(x="Method_idx", y="Shapley value", height=4, aspect=.35, hue="Method_idx",
                    data=data_df, dodge=True,  col="Player", kind="swarm")

    g.set(xticks=[])
    g.set(xlabel='')
    # g.set(ylim=(-5, 20))
    axes = g.axes.flatten()
    player_names = sorted(list(set(data['Player'])))
    for ax, player_name in zip(axes, player_names):
        ax.set_title(player_name)
    g.fig.tight_layout()
    # Let space for legend
    g.fig.subplots_adjust(right=0.7)

    return ax


def compute_shapley_value(rewards_with: np.ndarray, rewards_without: np.ndarray) -> float:
    'Compute marginal contributions and Shapley values from rewards on random coalitions with and without the considerd player'
    # Compute marginal contributions from rewards

    # shapley_value = np.sum(rewards_with - rewards_without, axis=-1)
    shapley_value = rewards_with.sum(
        axis=-1) - rewards_without.sum(axis=-1)
    shapley_value = shapley_value.mean()

    return shapley_value


def load_cat_plot_data(path: str):
    names = ["player", "episode", "method", "rewards_with", "rewards_without"]
    player_names_list = []
    methods_list = []
    shapley_values = []
    run_list = []
    for f in Path(path).rglob('*.csv'):
        df = pd.read_csv(f, names=names)
        if 'model' in str(f):
            run_id = str(f).split("model_")[1].split("_")[0]
        else:
            raise ValueError(
                'Wrong file name, fn should contain either run or ckpt')
        player_names = df["player"].unique()
        methods = df["method"].unique()
        for method in methods:
            for player in player_names:
                run_list.append(int(run_id))
                df_sel = df[df['player'] == player]
                df_sel = df_sel[df_sel['method'] == method]
                r_with = np.vstack(df_sel['rewards_with'].values)
                r_without = np.vstack(df_sel['rewards_without'].values)
                # print(type(r_with), type(r_with[0][0]), r_with)
                shapley_value = compute_shapley_value(r_with, r_without)
                # return

                player_names_list.append(player)
                methods_list.append(method)
                shapley_values.append(shapley_value)

    methods_list = ["noop" if m == "idle" else m for m in methods_list]
    data = {
        'Player': player_names_list,
        'Shapley value': shapley_values,
        'Method_idx': methods_list,
        'Method': methods_list.copy(),
        'Run': run_list
    }

    return pd.DataFrame(data)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    print(args.plot_type)
    data = load_cat_plot_data(args.result_dir)
    # print(data)
    # print(data[data['Method'] == 'noop'])

    agent_names = [f"Agent {i}" for i in range(args.num_agents)]

    if args.plot_type == "shapley_barchart":
        plot_shap_barchart(data[data['Method'] == 'noop']
                           ['Shapley value'], agent_names)
        plot_shap_barchart(data[data['Method'] == 'random']
                           ['Shapley value'], agent_names)
        plot_shap_barchart(
            data[data['Method'] == 'random_player']['Shapley value'], agent_names)
    elif args.plot_type == "shapley_cat_plot":
        cat_plot(data)
    elif args.plot_type == "model_rewards":
        plot_model_rewards(args.model_location)

    else:
        raise Exception("Unknown plot type")

    plt.show()
