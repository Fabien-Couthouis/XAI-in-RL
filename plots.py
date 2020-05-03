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

    ax.set_title("Contribution of each agent (Shapley values)",
                 BARCHAR_TEXTPROPS)
    ax.set_xlabel('Agent names', BARCHAR_TEXTPROPS)
    ax.set_ylabel("Shapley value", BARCHAR_TEXTPROPS)
    fig.tight_layout()
    return fig, ax


def plot_model_rewards_pp(path: str):
    'Plot: reward per episode for each agent'
    fig, ax = plt.subplots()

    all_files = [path for path in Path(path).rglob('*.csv')]
    names = ["episode", "reward", "r1", "r2", "r3", "r4"]

    df = pd.concat((pd.read_csv(f, names=names)
                    for f in all_files))
    df['sum_good_agents'] = df['r1']+df['r2']+df['r3']

    ax = sns.lineplot(x="episode", y="sum_good_agents", data=df)
    ax.set_xlabel('Training episode number')
    ax.set_ylabel('Reward for predators (averrage over 5 models)')
    # ax.set_title(f"Reward per episode on {len(agent_rewards[0])} episodes")

    # Set 1e label
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    fig.tight_layout()
    return fig, ax


def plot_shap_piechart(shapley_values: List[float], agent_names: List[str]):
    'Plot piechart: % contribution for each agent based on Shapley values '
    values = [shapley_value/sum(shapley_values)
              for shapley_value in shapley_values]
    fig, ax = plt.subplots()
    ax.pie(values, labels=agent_names, autopct='%1.1f%%',
           startangle=90, textprops=PIECHART_TEXTPROPS)
    # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.axis('equal')
    fig.tight_layout()
    return fig, ax


# def cat_plot(shapley_values: List[List[float]], agent_names: List[str], method_names: List[str]):
def cat_plot(player_names: List[str], shapley_values: List[float], methods: List[str]):
    # fig, ax = plt.subplots()
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
    axes = g.axes.flatten()
    player_names = sorted(list(set(data['Player'])))
    for ax, player_name in zip(axes, player_names):
        ax.set_title(player_name)
    g.fig.tight_layout()
    # Let space for legend
    g.fig.subplots_adjust(right=0.7)

    return ax


def load_cat_plot_data_pp(path: str):
    all_files = [path for path in Path(path).rglob('*.csv')]

    names = ["player", "m", "missing_agents_behaviour", "discounted_rewards_with_player_gamma_1", "discounted_rewards_with_player_gamma_0.99",
             "discounted_rewards_with_player_gamma_0.95", "discounted_rewards_with_player_gamma_0.90",
             "discounted_rewards_without_player_gamma_1", "discounted_rewards_without_player_gamma_0.99",
             "discounted_rewards_without_player_gamma_0.95", "discounted_rewards_without_player_gamma_0.90"]
    player_names_list = []
    methods_list = []
    shapley_values = []
    for f in all_files:
        df = pd.read_csv(f, names=names)
        player_names = df["player"].unique()
        methods = df["missing_agents_behaviour"].unique()
        for method in methods:
            for player in player_names:
                df_sel = df[df['player'] == player]
                df_sel = df_sel[df_sel['missing_agents_behaviour'] == method]
                dr_with = df_sel['discounted_rewards_with_player_gamma_1']
                dr_without = df_sel['discounted_rewards_without_player_gamma_1']

                shapley_value = np.mean(dr_with-dr_without)

                player_names_list.append(player)
                methods_list.append(method)
                shapley_values.append(shapley_value)

    return player_names_list, shapley_values, methods_list


if __name__ == "__main__":
    # Replace example values by yours! :)
    agent_names = ["agent1", "agent2", "agent3", "agent4"]
    model_names = ["model1", "model2", "model3", "model4"]
    agent_rewards = [[1, 2, 3, 4, 1], [28, 69, 45, 58, 7],
                     [8, 9, 5, 8, 7], [8, 29, 18, 8, 7]]
    shapley_values = [3, 20, 50, 100]

    player_names = ['player1', "player1", 'player1', 'player1', 'player2',
                    'player2', 'player2', 'player3', 'player3', 'player3']
    shapley_values_2 = [34, 38, 30, 32, 16, 15, 14, 25, 23, 25]
    methods = ['random', "random", 'idle', 'random_player', 'random',
               'idle', 'random_player', 'random', 'idle', 'random_player']

    path_pp_models = r"multiagent_particles/experiments/saves"
    plot_model_rewards_pp(path=path_pp_models)
    path_pp_mc = r"multiagent_particles\experiments\marginal_contributions"
    # data = load_cat_plot_data_pp(path_pp_mc)
    # cat_plot(*data)

    # plot_shap_barchart(shapley_values, agent_names)
    # plot_shap_piechart(shapley_values, agent_names)
    # cat_plot(player_names, shapley_values_2, methods)

    plt.show()
