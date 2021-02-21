from collections import defaultdict
from itertools import permutations
from pathlib import Path
from typing import List
import argparse
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
        'result_dir', type=str, help='Directory containing results of experiments')

    # optional input parameters
    parser.add_argument(
        '--num-agents',
        type=int,
        default=3,
        help='The number of rollouts to visualize.')
    parser.add_argument("--plot-type", type=str, default="shapley_barchart",
                        help="type of diagram to plot: shapley_barchart/model_rewards/cat_plot")
    parser.add_argument("--model-dir", type=str, default="saves/run_3_vs_9/",
                        help='model location, required when plotting model rewards')
    return parser


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


def plot_model_rewards_pp(folder_path: str, num_good_agents: int, num_preys: int):
    'Plot: reward per episode for each agent'
    fig, ax = plt.subplots()

    all_files = [path for path in Path(folder_path).rglob('*.csv')]
    names = ["episode", "reward"]
    names += [f"r{i}" for i in range(num_good_agents+num_preys)]

    df = pd.concat((pd.read_csv(f, names=names)
                    for f in all_files))
    df['sum_good_agents'] = sum([df[f"r{i}"] for i in range(num_good_agents)])

    ax = sns.lineplot(x="episode", y="sum_good_agents", data=df)
    ax.set_xlabel('Training episode number')
    ax.set_ylabel('Reward for predators (average over 5 models)')
    # ax.set_title(f"Reward per episode on {len(agent_rewards[0])} episodes")

    # Set 1e label
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    fig.tight_layout()
    return fig, ax


# def cat_plot(shapley_values: List[List[float]], agent_names: List[str], method_names: List[str]):
# def cat_plot(player_names: List[str], shapley_values: List[float], methods: List[str]):
def cat_plot(player_names: List[str], shapley_values: List[float], methods: List[str]):
    # fig, ax = plt.subplots()
    data = {
        'Player': [f"Predator {name}" for name in player_names],
        'Shapley value': shapley_values,
        'Method_idx': methods,
        'Method': methods.copy()
    }

    data_df = pd.DataFrame(data)
    g = sns.catplot(x="Method_idx", y="Shapley value", height=4, aspect=.35, hue="Method",
                    data=data_df, dodge=True, palette=sns.color_palette("husl", 6), col="Player", kind="swarm")

    g.set(xticks=[])
    g.set(xlabel='')
    g.set(ylim=(-5, 20))
    axes = g.axes.flatten()
    player_names = sorted(list(set(data['Player'])))
    for ax, player_name in zip(axes, player_names):
        ax.set_title(player_name)
    g.fig.tight_layout()
    # Let space for legend
    g.fig.subplots_adjust(right=0.7)

    return ax


def compute_shapley_value(rewards_with: np.array, rewards_without: np.array) -> float:
    'Compute marginal contributions and Shapley values from rewards on random coalitions with and without the considerd player'
    # Compute marginal contributions from rewards

    # shapley_value = np.sum(rewards_with - rewards_without, axis=-1)
    shapley_value = rewards_with.sum(
        axis=-1) - rewards_without.sum(axis=-1)
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


def load_cat_plot_data_pp(path: str, M: int = None):
    all_files = [p for p in Path(path).rglob('*.csv')]
    print(Path(path))

    names = ["player", "episode", "method", "goal_agents_with", "total_good_agents_rewards_with",
             "good_agents_rewards_with", "goal_agents_without", "total_good_agents_rewards_without", "good_agents_rewards_without"]
    player_names_list = []
    methods_list = []
    shapley_values = []
    run_list = []
    for f in all_files:
        df = pd.read_csv(f, names=names)
        run_id = str(f).split("run_")[1].split("_")[0]

        player_names = df["player"].unique()
        methods = df["method"].unique()
        for method in methods:
            for player in player_names:
                run_list.append(run_id)
                df_sel = df[df['player'] == player]
                df_sel = df_sel[df_sel['method'] == method]
                if M is not None:
                    df_sel = df_sel.sample(n=M)
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

    methods_list = ["noop" if m == "idle" else m for m in methods_list]
    for i in range(len(methods_list)):
        methods_list[i] += " (MC estimation)"

    return player_names_list, shapley_values, methods_list, run_list


def load_cat_plot_data_pp_true(path: str):
    all_files = [p for p in Path(path).rglob('*.csv')]
    print(Path(path))

    names = ["coalition", "n", "method", "value"]
    methods_list = []
    shapley_values = []
    players_list = []
    for f in all_files:
        df = pd.read_csv(f, names=names)
        players = eval(df.iloc[-1]['coalition'])
        for player in players:
            shapley_value = 0
            for permutation in permutations(players):
                to_remove = []
                for i, x in enumerate(permutation):
                    if x == player:
                        coalition = sorted(permutation[:i+1])
                        shapley_value += df[df['coalition'] ==
                                            str(coalition)]["value"].values[0]
                        to_remove = str(sorted(to_remove))
                        shapley_value -= df[df['coalition']
                                            == to_remove]["value"].values[0]
                        break
                    else:
                        to_remove.append(x)

            shapley_value /= np.math.factorial(len(players))
            shapley_values.append(shapley_value)
            methods_list.append(df.iloc[0]["method"])
        players_list.extend(players)

    for i in range(len(methods_list)):
        methods_list[i] += " (real)"
        methods_list[i] = methods_list[i].replace("idle", "noop")
    return players_list, shapley_values, methods_list


def plot_goal_agents_pp(folder_path: str, agent_names: List[str]):
    'Plot barchart: agent with corresponding shapley value'
    fig, ax = plt.subplots()
    all_files = [path for path in Path(folder_path).rglob('*.csv')]

    dfs = [pd.read_csv(f, index_col=0, names=["Episode", "Reward"]+agent_names)[agent_names].sum().T.reset_index().assign(model=i)
           for i, f in enumerate(all_files)]

    df_c = pd.concat(dfs).rename(columns={0: "value"})
    ax = sns.barplot(x="value", y="index", hue="model", data=df_c, palette="husl",
                     orient="h", order=agent_names, estimator=sum)
    # ax.set_title("Statistics for each agent",
    #              BARCHAR_TEXTPROPS)
    ax.set(xlabel="Number of times the prey get caught")
    ax.set(ylabel="Agent")

    fig.tight_layout()
    return fig, ax


def plot_goal_agents_pp_one(folder_path: str, agent_names: List[str]):
    'Plot barchart: agent with corresponding shapley value (one bar)'
    fig, ax = plt.subplots()
    all_files = [path for path in Path(folder_path).rglob('*.csv')]
    # names = ["episode", "reward", "r1", "r2", "r3", "r4"]

    df = pd.concat((pd.read_csv(f, names=["Episode", "Reward"]+agent_names)
                    for f in all_files))
    df_sum = df[agent_names].sum().reset_index().rename(
        columns={0: "value"})

    ax = sns.barplot(x="value", y="index", data=df_sum, palette="husl",
                     orient="h", order=agent_names, estimator=sum)
    ax.set(xlabel="Number of times the prey get caught")
    ax.set(ylabel="Agent")

    fig.tight_layout()
    return fig, ax


# def plot_M_graph(path_pp_mc, path_pp_true, M_list):
#     fig, ax = plt.subplots()
#     true_data = load_cat_plot_data_pp_true(path_pp_mc_true)
#     true_shapley = np.asarray(true_data[1])

#     shapley_diff = []
#     for M in M_list:
#         mc_data = load_cat_plot_data_pp(path_pp_mc, M=M)
#         mc_shapley = np.asarray(mc_data[1])
#         shapley_diff.extend((np.absolute(true_shapley - mc_shapley)).tolist())

#     data = pd.DataFrame({
#         'Player': [f"Predator {name}" for name in true_data[0]]*len(M_list),
#         'Shapley_diff': shapley_diff,
#         'M': M_list*len(true_data[0]),
#         'Method': mc_data[2]*len(M_list)
#     })
#     data = data[(data["Method"] == "noop") & (data["Player"] == "Predator 1")]

#     data = data.groupby(['Method', 'Player', 'M']).mean()  # .reset_index()
#     print(data)

#     ax = sns.lineplot(data=data, x="M",
#                       y="Shapley_diff")
#     # MSE between real Shapley value and estimated Shapley value
#     fig.tight_layout()
#     return fig, ax
def plot_shapley_vs_speed(path: str, agent_id: int):
    data = load_cat_plot_data_pp(path)
    # Hardcoded :S ; can use file names if needed
    speeds = []
    for speed_a1 in range(0, 22, 2):
        speeds.append(speed_a1/10)

    data_df = pd.DataFrame({
        'Player_id': [player_id for player_id in data[0]],
        'Shapley_value': data[1],
        'Run': data[3]
    })
    print(data_df)
    data_df = data_df[data_df["Player_id"] == agent_id]
    # *len(set(data[3])) to fit the nb of runs
    data_df["Speed"] = speeds*len(set(data[3]))
    # data_df["Run"] = data[3]

    print(data_df)
    # print(len(shapley_values))
    fig, ax = plt.subplots()

    ax = sns.lineplot(x="Speed", y="Shapley_value",  # hue="Run",
                      data=data_df, palette="husl")
    ax.set(xlabel="Speed of Predator 0")
    ax.set(ylabel="Shapley Value")

    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    args = create_parser()
    args = args.parse_args()
    agent_names = [f"Predator {i}" for i in range(args.num_agents)]
    path_pp_mc = args.result_dir  # r"rewards/exp1"
    path_pp_mc_true = r"rewards/true-shap-exp1"

    data = load_cat_plot_data_pp(path_pp_mc)

    if args.plot_type == "shapley_barchart":
        shapley_values = data[1]
        plot_shap_barchart(shapley_values[0:9], agent_names)  # NOOP
        plot_shap_barchart(shapley_values[9:18], agent_names)
        plot_shap_barchart(shapley_values[18:27], agent_names)
    elif args.plot_type == "shapley_cat_plot":
        cat_plot(*data)
    elif args.plot_type == "shapley_true":
        data_true = load_cat_plot_data_pp_true(path_pp_mc_true)

        # # Add true shapley value to MC approximation
        for i in range(len(data_true)):
            data[i].extend(data_true[i])
        cat_plot(*data)
    elif args.plot_type == "shapley_speed":
        plot_shapley_vs_speed("rewards/exp2-speeds-chart", 0)
    elif args.plot_type == "goal_agents":
        plot_goal_agents_pp(r"goal_agents/exp2", agent_names)
        plot_goal_agents_pp_one(r"goal_agents/exp1", agent_names)
    elif args.plot_type == "model_rewards":
        plot_model_rewards_pp(
            "saves/run_3_vs_9", 9, 3)
    else:
        raise Exception(f"Unknown plot type: {args.plot_type}")

    plt.show()
