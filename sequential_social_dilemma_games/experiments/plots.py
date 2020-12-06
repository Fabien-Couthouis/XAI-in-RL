import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import os
from pathlib import Path
from typing import List

plt.style.use('bmh')

BARCHAR_TEXTPROPS = {"fontsize": 12, "weight": "bold"}
PIECHART_TEXTPROPS = {"fontsize": 12}


def plot_shap_barchart(shapley_values: List[float], agent_names: List[str]):
    'Plot barchart: agent with corresponding shapley value'
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


def plot_model_rewards(folder_path: str):
    'Plot: reward per episode for each agent'
    fig, ax = plt.subplots()

    all_files = [path for path in Path(folder_path).rglob('*.csv')]

    df = pd.concat((pd.read_csv(f)
                    for f in all_files))

    ax = sns.lineplot(x="episodes_total", y="episode_reward_mean", data=df)
    ax.set_xlabel('Training episode number')
    ax.set_ylabel('Mean global reward')
    # ax.set_title(f"Reward per episode on {len(agent_rewards[0])} episodes")

    # Set 1e label
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    fig.tight_layout()
    return fig, ax


# def cat_plot(shapley_values: List[List[float]], agent_names: List[str], method_names: List[str]):
# def cat_plot(player_names: List[str], shapley_values: List[float], methods: List[str]):
def cat_plot(player_names: List[str], shapley_values: List[float], methods: List[str]):
    # fig, ax = plt.subplots()
    methods = ["noop" if m == "idle" else m for m in methods]
    data = {
        'Player': player_names,
        'Shapley value': shapley_values,
        'Method_idx': methods,
        'Method': methods.copy()
    }

    data_df = pd.DataFrame(data)
    g = sns.catplot(x="Method_idx", y="Shapley value", height=4, aspect=.35, hue="Method",
                    data=data_df, dodge=True, palette="husl", col="Player", kind="swarm")

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
    all_files = [p for p in Path(path).rglob('*.csv')]
    print(Path(path))

    names = ["player", "episode", "method","rewards_with", "rewards_without"]
    player_names_list = []
    methods_list = []
    shapley_values = []
    for f in all_files:
        df = pd.read_csv(f, names=names)
        player_names = df["player"].unique()
        methods = df["method"].unique()
        for method in methods:
            for player in player_names:
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
    return player_names_list, shapley_values, methods_list




if __name__ == "__main__":
    agent_names = [f"Agent {i}" for i in range(6)]
    path_pp_mc = r"rewards"

    data = load_cat_plot_data(path_pp_mc)
    # shapley_values = data[1]
    # plot_shap_barchart(shapley_values, agent_names)

    cat_plot(*data)

    #plot_model_rewards("rewards")


    plt.show()
