import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import os
from pathlib import Path
from typing import List

GAMMA = 0.95
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


def plot_model_rewards_pp(folder_path: str):
    'Plot: reward per episode for each agent'
    fig, ax = plt.subplots()

    all_files = [path for path in Path(folder_path).rglob('*.csv')]
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
# def cat_plot(player_names: List[str], shapley_values: List[float], methods: List[str]):
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


def discount_rewards(rewards, gamma) -> np.ndarray:
    'Returns rewards discounted by GAMMA factor'
    discounted_rewards = np.zeros_like(rewards)
    # print(discounted_rewards)
    for r in range(len(rewards)):
        R = 0
        for t in reversed(range(0, len(rewards[r]))):
            # print("rewards t", t, rewards[r][t])
            R = R * gamma + rewards[r][t]
            discounted_rewards[r][t] = R
            # print("R", t, R)
    return discounted_rewards


def compute_shapley_value(rewards_with: np.array, rewards_without: np.array) -> float:
    'Compute marginal contributions and Shapley values from rewards on random coalitions with and without the considerd player'
    # Compute marginal contributions from rewards
    # rewards_with = rewards_with[10:15]
    # rewards_without = rewards_without[10:15]
    # print("R WITH", rewards_with)
    # print("R WITHOUT", rewards_without)

    # d_rewards_with = discount_rewards(rewards_with, GAMMA).sum(axis=-1)
    # d_rewards_without = discount_rewards(rewards_without, GAMMA).sum(axis=-1)
    # print(d_rewards_with[10:15])

    # print("D WITH", d_rewards_with)
    # print("D WITHOUT", d_rewards_without)
    # print("LESS", d_rewards_with-d_rewards_without)
    # d_rewards_with = (d_rewards_with-d_rewards_with.mean()) / \
    #     d_rewards_with.std()
    # d_rewards_without = (
    #     d_rewards_without-d_rewards_without.mean())/d_rewards_without.std()

    # shapley_value = np.sum(rewards_with - rewards_without, axis=-1)
    shapley_value = rewards_with.sum(axis=-1) - rewards_without.sum(axis=-1)
    # min_s = shapley_value.min()
    # max_s = shapley_value.max()
    # shapley_value = (shapley_value - min_s)/(max_s-min_s)
    shapley_value = shapley_value.mean()
    # print(shapley_value)

    # print(r_with)
    # r_without = np.sum(rewards_without, axis=-1)
    # min_with = np.min(r_with)
    # max_with = np.max(r_with)
    # min_without = np.min(r_without)
    # max_without = np.max(r_without)

    # # Compute Shapley values
    # r_with = (r_with - min_with)/(max_with-min_with)
    # r_without = (r_without - min_without)/(max_without-min_without)

    # shapley_value = np.mean(r_with-r_without)

    return shapley_value


def load_cat_plot_data_pp(path: str):
    all_files = [path for path in Path(path).rglob('*.csv')]

    names = ["player", "episode", "method", "goal_agents_with", "total_good_agents_rewards_with",
             "good_agents_rewards_with", "goal_agents_without", "total_good_agents_rewards_without", "good_agents_rewards_without"]
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
                r_with = np.vstack(df_sel['total_good_agents_rewards_with'].apply(
                    eval).apply(np.array).values)
                r_without = np.vstack(df_sel['total_good_agents_rewards_without'].apply(
                    eval).apply(np.array).values)
                # print(type(r_with), type(r_with[0][0]), r_with)
                shapley_value = compute_shapley_value(r_with, r_without)
                # return

                player_names_list.append(player)
                methods_list.append(method)
                shapley_values.append(shapley_value)
    return player_names_list, shapley_values, methods_list


def plot_goal_agents_pp(folder_path: str, agent_names: List[str]):
    'Plot barchart: agent with corresponding shapley value'
    fig, ax = plt.subplots()
    all_files = [path for path in Path(folder_path).rglob('*.csv')]
    # names = ["episode", "reward", "r1", "r2", "r3", "r4"]

    df = pd.concat((pd.read_csv(f, names=["Episode", "Reward"]+agent_names)
                    for f in all_files))
    print(df.head())
    df_sum = df[agent_names].sum().reset_index().rename(
        columns={0: "value"})
    print(df_sum)

    ax = sns.barplot(x="value", y="index", data=df_sum, palette="husl",
                     orient="h", order=agent_names, estimator=sum)
    ax.set(ylabel="Percentage of time catching the prey")

    # ax.set_title("Contribution of each agent (Shapley values)",
    #              BARCHAR_TEXTPROPS)
    # ax.set_xlabel('Agent names', BARCHAR_TEXTPROPS)
    # ax.set_ylabel("Shapley value", BARCHAR_TEXTPROPS)
    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    # Replace example values by yours! :)
    # agent_names = ["agent1", "agent2", "agent3", "agent4"]
    # model_names = ["model1", "model2", "model3", "model4"]
    # agent_rewards = [[1, 2, 3, 4, 1], [28, 69, 45, 58, 7],
    #                  [8, 9, 5, 8, 7], [8, 29, 18, 8, 7]]
    # shapley_values = [3, 20, 50, 100]

    # player_names = ['player1', "player1", 'player1', 'player1', 'player2',
    #                 'player2', 'player2', 'player3', 'player3', 'player3']
    # shapley_values_2 = [34, 38, 30, 32, 16, 15, 14, 25, 23, 25]
    # methods = ['random', "random", 'idle', 'random_player', 'random',
    #            'idle', 'random_player', 'random', 'idle', 'random_player']

    # path_pp_models = r"multiagent_particles/experiments/saves"
    # # plot_model_rewards_pp(path=path_pp_models)
    # path_pp_mc = r"multiagent_particles\experiments\rewards"
    # data = load_cat_plot_data_pp(path_pp_mc)
    # cat_plot(*data)

    plot_goal_agents_pp(r"multiagent_particles\experiments\goal_agents", [
                        "agent 0", "agent 1", "agent 2"])

    # plot_shap_barchart(shapley_values, agent_names)
    # plot_shap_piechart(shapley_values, agent_names)
    # cat_plot(player_names, shapley_values_2, methods)

    plt.show()
