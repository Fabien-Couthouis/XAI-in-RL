import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import List
plt.style.use('seaborn-deep')

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
    return fig, ax


if __name__ == "__main__":
    agent_names = ["agent1", "agent2", "agent3", "agent4"]
    model_names = ["model1", "model2", "model3", "model4"]
    agent_rewards = [[1, 2, 3, 4, 1], [28, 69, 45, 58, 7],
                     [8, 9, 5, 8, 7], [8, 29, 18, 8, 7]]
    shapley_values = [3, 20, 50, 100]

    plot_model_rewards(agent_rewards, model_names)
    plot_shap_barchart(shapley_values, agent_names)
    plot_shap_piechart(shapley_values, agent_names)
    plt.show()
