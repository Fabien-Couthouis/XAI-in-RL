import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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


def plot_model_rewards(agent_rewards: List[float], agent_names: List[str]):
    'Plot: reward per episode for each agent'
    fig, ax = plt.subplots()

    for rewards in agent_rewards:
        x = range(1, len(rewards)+1)
        ax.plot(x, rewards)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')

    leg = ax.legend(agent_names)
    # Set legend dragable to move it outside plots if needed
    leg.set_draggable(True)
    ax.set_title(f"Reward per episode on {len(agent_rewards[0])} runs")
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

    # plot_model_rewards(agent_rewards, model_names)
    # plot_shap_barchart(shapley_values, agent_names)
    # plot_shap_piechart(shapley_values, agent_names)
    cat_plot(player_names, shapley_values_2, methods)

    plt.show()
