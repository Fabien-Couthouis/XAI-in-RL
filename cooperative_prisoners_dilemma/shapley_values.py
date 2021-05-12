from __future__ import absolute_import

import csv
import os
import random
from itertools import combinations
from statistics import mean

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


def monte_carlo_shapley_values(args, agent, config, n_players):
    """
    Monte Carlo estimation of shapley values (if k = num_episodes*n_random_coalitions, o(k*n_agents) complexity).
    """

    print("Starting Shapley value estimation on:", n_players,
          " n_random_coalitions=", args.shapley_M, " missing agents behaviour=", args.missing_agents_behaviour)

    shapley_values = []
    players_ids = [f"agent-{i}" for i in range(n_players)]
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
            print(
                f"Coalition {m}/{args.shapley_M}: {coalition_with_player}")

            # Simulate num_episodes episodes on selected coalitions with and without current player
            reward_with_player = rollout(args, agent, config, 1, player_id, coalition_with_player)
            reward_without_player = rollout(
                args, agent, config, 1, player_id, coalition_without_player)

            save_rollout_info(args, player_id, m, reward_with_player, reward_without_player)

            # Compute estimated marginal contribution
            marginal_contribution = reward_with_player[0] - reward_without_player[0]
            marginal_contributions.append(marginal_contribution)

        # Compute shapley value as the mean of marginal contributions
        shapley_value = mean(marginal_contributions)
        shapley_values.append(shapley_value)
        print(f"Shapley value for player {player_id}: {shapley_value}")

    print("Shapley values:", shapley_values)

    return shapley_values

def exact_shapley_values(args, agent, config, n_players):
    shapley_values = []
    players_ids = [f"agent-{i}" for i in range(n_players)]
    for player_id in players_ids:
        print(f"Starting computation for player {player_id}...")
        # Get all possible combinations with and without the current player
        coalitions_with_player = compute_coalitions_with_player(
            player_id, players_ids)
        marginal_contributions = []
        for i, coalition_with_player in enumerate(coalitions_with_player):
            coalition_without_player = [
                n for n in coalition_with_player if n != player_id]
            print(
                f"Coalition {i}/{len(coalitions_with_player)}: {coalition_with_player}")

            # Simulate num_episodes episodes on selected coalitions with and without current player
            reward_with_player = rollout(args, agent, config, 1, player_id, coalition_with_player)
            reward_without_player = rollout(
                args, agent, config, 1, player_id, coalition_without_player)

            save_rollout_info(args, player_id, i, reward_with_player, reward_without_player)

            # Compute estimated marginal contribution
            marginal_contribution = reward_with_player[0] - reward_without_player[0]
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