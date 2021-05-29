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
        '--result-dir', type=str, help='Directory containing results of experiments')

    # optional input parameters
    parser.add_argument(
        '--num-agents',
        type=int,
        default=6,
        help='The number of agents.')
    parser.add_argument("--plot-type", type=str, default="shapley_barchart",
                        help="type of diagram to plot: shapley_barchart/model_rewards/cat_plot")
    parser.add_argument("--model-dir", type=str, default="models/harvest_6_agents/",
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


def efficiency(returns: List[float], num_agents: int):
    'Efficiency over the episodes'
    return (np.sum(returns)/num_agents)/len(returns)


def equality(returns: List[float], num_agents: int):
    'Equality over the episodes'
    equalities = []
    for episode_returns in returns:
        diff = 0
        sum_ri = 0
        for i in range(num_agents):
            sum_ri += episode_returns[i]
            for j in range(num_agents):
                diff += np.abs(episode_returns[i] - episode_returns[j])

        equality = 1-(diff/(2*num_agents*sum_ri))
        equalities.append(equality)
    return np.mean(equalities)


def sustainability(tr_lst: List[float], num_agents: int):
    'Sustainability over the episodes'
    return np.sum(np.mean(tr_lst, axis=0), axis=-1)/num_agents


def plot_sm_shap_linechart(data_df, efficiencies, equalities, sustainabilities, mean_returns, ckpts):
    'Plot barchart: agent with corresponding shapley value'
    fig, ax = plt.subplots()
    data_df = data_df.rename(columns={"Shapley value": "Value",  "Run": "Episode"})

    def create_df_social_metric(social_metrics):
        df = pd.DataFrame({'Value': [], 'Episode': []})

        for social_metric, ckpt in zip(social_metrics, ckpts):
            for sm in social_metric.flatten():
                row = [sm,  ckpt]
                df.loc[len(df)] = row

        return df

    df_efficiency = create_df_social_metric(efficiencies)
    df_equality = create_df_social_metric(equalities)
    df_sustainability = create_df_social_metric(sustainabilities)

    print(df_efficiency)

    # for agent_returns, ckpt in zip(mean_returns, ckpts):
    #     print(len(mean_returns), len(ckpts))
    #     for agent_id, agent_return in enumerate(agent_returns):
    #         row = [f'Agent {agent_id} mean return', agent_return, 'noop', 'noop', ckpt]
    #         data_df.loc[len(data_df)] = row

    data_df = data_df[data_df['Method'] == 'noop']

    ax = sns.lineplot(data=data_df, x='Episode', y='Value',
                      sort=True, hue='Player')

    # ax.set_title("Shapley value for each agent at different training step (random action selection method)",
    #              BARCHAR_TEXTPROPS)
    ax.set_xlabel("Episode",
                  BARCHAR_TEXTPROPS)
    ax.set_ylabel('Shapley value', BARCHAR_TEXTPROPS)

    fig.tight_layout()

    fig, ax = plt.subplots()
    df_mean_sv = data_df.drop(['Player'], axis=1)
    df_mean_sv = df_mean_sv.groupby(['Episode']).mean()
    ax = sns.lineplot(data=df_mean_sv, x='Episode', y='Value',
                      sort=True, ci="sd")
    ax.set_xlabel("Episode",
                  BARCHAR_TEXTPROPS)
    ax.set_ylabel('Mean of Shapley values of all agents', BARCHAR_TEXTPROPS)
    fig.tight_layout()

    fig, ax = plt.subplots()
    ax = sns.lineplot(data=df_efficiency, x='Episode', y='Value',
                      ci="sd")
    ax.set_xlabel("Episode",
                  BARCHAR_TEXTPROPS)
    ax.set_ylabel('Efficiency', BARCHAR_TEXTPROPS)
    fig.tight_layout()

    fig, ax = plt.subplots()
    ax = sns.lineplot(data=df_equality, x='Episode', y='Value',
                      sort=True, ci="sd")
    ax.set_xlabel("Episode",
                  BARCHAR_TEXTPROPS)
    ax.set_ylabel('Equality', BARCHAR_TEXTPROPS)

    fig.tight_layout()

    fig, ax = plt.subplots()
    ax = sns.lineplot(data=df_sustainability, x='Episode', y='Value',
                      sort=True, ci="sd")
    ax.set_xlabel("Episode",
                  BARCHAR_TEXTPROPS)
    ax.set_ylabel('Sustainability', BARCHAR_TEXTPROPS)
    ax.set_ylim(440, 500)

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
    print(data_df)
    fig, ax = plt.subplots()
    g = sns.catplot(x="Method_idx", y="Shapley value", height=4, aspect=.35, hue="Method",
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


def compute_shapley_value_social_metric(rewards_with: np.ndarray, rewards_without: np.ndarray, social_metric: str, num_agents: int):
    assert social_metric in ["efficiency", "equality", "sustainability"]

    if social_metric == "efficiency":
        shapley_value = efficiency(rewards_with, num_agents) - \
            efficiency(rewards_without, num_agents)
    elif social_metric == "equality":
        shapley_value = equality(rewards_with, num_agents) - equality(rewards_without, num_agents)
    else:
        pass
        #TODO: "sustainability"

    return shapley_value.mean()


def load_cat_plot_data(path: str):
    all_files = [p for p in Path(path).rglob('*.csv')]

    names = ["player", "episode", "method", "rewards_with", "rewards_without"]
    player_names_list = []
    methods_list = []
    shapley_values = []
    run_list = []
    for f in all_files:
        df = pd.read_csv(f, names=names)
        if 'run' in str(f):
            run_id = str(f).split("run_")[1].split("_")[0]
        elif 'ckpt' in str(f):
            run_id = str(f).split("ckpt_")[1].split(".")[0]
        else:
            raise ValueError('Wrong file name, fn should contain either run or ckpt')
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
    methods_list = ["replace" if m ==
                    "random_player_action" else m for m in methods_list]
    data = {
        'Player': player_names_list,
        'Shapley value': shapley_values,
        'Method_idx': methods_list,
        'Method': methods_list.copy(),
        'Run': run_list
    }

    return pd.DataFrame(data)


def load_social_metrics(paths: str, ckpts: List[int], num_agents: int = 5):
    def compute_returns(episode):
        df_episode = df[df['episode'] == episode]
        returns = [df_episode[f'reward_{agent_id}'].sum() for agent_id in range(num_agents)]
        return returns

    def compute_ti(episode):
        df_episode = df[df['episode'] == episode]
        ti_list = [df_episode[df_episode[f'reward_{agent_id}'] > 0]
                   ['tr'].mean() for agent_id in range(num_agents)]
        return ti_list

    efficiencies = []
    equalities = []
    sustainabilities = []
    mean_returns = []

    for ckpt in ckpts:
        sub_efficiencies = []
        sub_equalities = []
        sub_sustainabilities = []
        sub_mean_returns = []

        for path in paths:
            df = pd.read_csv(f'{path}/social_metrics_ckpt_{ckpt}.csv', sep=',', names=['episode', 'tr'] +
                             [f'reward_{agent_id}' for agent_id in range(num_agents)]).astype('float32')

            n_episodes = int(max(df['episode'].values))+1
            returns = [compute_returns(episode) for episode in range(n_episodes)]
            ti_lst = [compute_ti(episode) for episode in range(n_episodes)]

            sub_efficiencies.append(efficiency(returns, num_agents))
            sub_equalities.append(equality(returns, num_agents))
            sub_sustainabilities.append(sustainability(ti_lst, num_agents))

            sub_mean_returns.append(np.mean(returns, axis=0))

        efficiencies.append(np.dstack(sub_efficiencies))
        equalities.append(np.dstack(sub_equalities))
        sustainabilities.append(np.dstack(sub_sustainabilities))
        mean_returns.append(np.dstack(sub_mean_returns))

    return efficiencies, equalities, sustainabilities, mean_returns


def plot_social_metrics(data, path, checkpoints, num_agents=5):
    efficiencies, equalities, sustainabilies, mean_returns = load_social_metrics(
        path, checkpoints, num_agents)
    plot_sm_shap_linechart(data, efficiencies, equalities, sustainabilies, mean_returns, checkpoints)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    print(args.plot_type)
    data = load_cat_plot_data(args.result_dir)

    agent_names = [f"Agent {i}" for i in range(args.num_agents)]

    if args.plot_type == "shapley_barchart":
        plot_shap_barchart(data[data['Method' == 'idle']]['Shapley values'], agent_names)
        plot_shap_barchart(data[data['Method' == 'random']]['Shapley values'], agent_names)
        plot_shap_barchart(data[data['Method' == 'random_player']]['Shapley values'], agent_names)
    elif args.plot_type == "shapley_cat_plot":
        cat_plot(data)
    elif args.plot_type == "model_rewards":
        plot_model_rewards(args.model_location)
    elif args.plot_type == "social_metrics":
        plot_social_metrics(data, ['social_metrics', 'social_metrics2',
                            'social_metrics3', 'social_metrics4'], range(1000, 9000, 1000))

    else:
        raise Exception("Unknown plot type")

    plt.show()
