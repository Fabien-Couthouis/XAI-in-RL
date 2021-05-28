from __future__ import absolute_import

import csv
import os
import random
from itertools import combinations, permutations
from statistics import mean

import numpy as np

from rollout import rollout

DEBUG = False


def compute_coalitions(features):
    'Get all possible coalitions (combinations) between features (including empty list)'
    coalitions = []
    for i in range(0, len(features)+1):
        oc = combinations(features, i)
        for c in oc:
            coalitions.append(list(c))
    return coalitions


def compute_coalitions_with_player(player_id, players_ids):
    'Compute all coalitions of players including current player'
    coalitions = compute_coalitions(players_ids)
    coalitions_with_player = [
        coalition for coalition in coalitions if player_id in coalition]
    return coalitions_with_player


def monte_carlo_shapley_values(args, agent, config):
    """
    Monte Carlo estimation of shapley values (if k = num_episodes*n_random_coalitions, o(k*n_agents) complexity).
    """

    env = agent.workers.local_worker().env
    players_ids = [
        agent_name for agent_name in env.agents if agent_name.startswith("adversary")]

    n_players = len(players_ids)
    print("Starting Shapley value estimation on:", n_players, "players",
          " n_random_coalitions=", args.shapley_M, " missing agents behaviour=", args.missing_agents_behaviour)

    shapley_values = []

    for player_id in players_ids:
        print(f"Starting computation for player {player_id}...")
        # Get all possible combinations with and without the current player
        coalitions_with_player = compute_coalitions_with_player(
            player_id, players_ids)
        marginal_contributions = []
        for m in range(1, 1+args.shapley_M):
            # Select random coalitions
            coalition_with_player = random.choice(coalitions_with_player)
            coalition_without_player = [
                n for n in coalition_with_player if n != player_id]
            # print(
            #     f"Coalition {m}/{args.shapley_M}: {coalition_with_player}")

            # Simulate num_episodes episodes on selected coalitions with and without current player
            reward_with_player = rollout(
                args, agent, config, 1, player_id, coalition_with_player)
            reward_without_player = rollout(
                args, agent, config, 1, player_id, coalition_without_player)

            save_rollout_info(args, player_id, m,
                              reward_with_player, reward_without_player)

            # Compute estimated marginal contribution
            marginal_contribution = reward_with_player[0] - \
                reward_without_player[0]
            marginal_contributions.append(marginal_contribution)

        # Compute shapley value as the mean of marginal contributions
        shapley_value = mean(marginal_contributions)
        shapley_values.append(shapley_value)
        print(f"Shapley value for player {player_id}: {shapley_value}")

    print("Shapley values:", shapley_values)

    return shapley_values


def save_rollout_info(arglist, feature, m, reward_with, reward_without):
    'Keep track of marginal contributions in csv file while computing shapley values'
    file_name = f"{arglist.save_dir}/{arglist.exp_name}.csv"
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    with open(file_name, "a", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        row = [feature, m, arglist.missing_agents_behaviour]
        row.extend(reward_with)
        row.extend(reward_without)
        writer.writerow(row)

# Old version


# def compute_marginal_contributions(env_name, features, num_steps, num_episodes, agent, replace_missing_players):
#     'Get mean reward for each agent for each coalitions '
#     marginal_contributions = dict()
#     for coalition in compute_coalitions(features):
#         total_rewards = rollout(
#             agent, env_name, num_steps, num_episodes, coalition=coalition, replace_missing_players=replace_missing_players, verbose=False)
#
#         marginal_contributions[str(coalition)] = round(mean(total_rewards), 2)
#     if DEBUG:
#         print("Coalition values: ", marginal_contributions)
#     return marginal_contributions


# def exact_shapley_values(env_name, n_agents, agent, num_steps=10, num_episodes=1, replace_missing_players="random"):
#     'Naive implementation (not optimized at all)'
#     agents_ids = range(n_agents)
#     coalition_values = compute_marginal_contributions(
#         env_name, agents_ids, num_steps, num_episodes, agent, replace_missing_players)
#     shapley_values = []

#     for agent_id in agents_ids:
#         if DEBUG:
#             print("Computing shap value for agent: ", agent_id)
#         shapley_value = 0

#         for permutation in permutations(agents_ids):
#             to_remove = []
#             if DEBUG:
#                 print("permutation ", permutation)
#             for i, x in enumerate(permutation):
#                 if x == agent_id:
#                     coalition = sorted(permutation[:i+1])
#                     if DEBUG:
#                         print("coalition", coalition)
#                     shapley_value += coalition_values[str(coalition)]

#                     if len(to_remove) > 0:
#                         to_remove = str(sorted(to_remove))
#                         shapley_value -= coalition_values[to_remove]
#                         if DEBUG:
#                             print("to remove ", to_remove)
#                     break
#                 else:
#                     to_remove.append(x)
#         shapley_values.append(shapley_value)

#     result = np.divide(shapley_values, np.math.factorial(n_agents))
#     print("Shapley values:", result)

#     return result
