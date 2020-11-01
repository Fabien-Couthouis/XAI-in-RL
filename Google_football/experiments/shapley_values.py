from __future__ import absolute_import

import numpy as np
import random
from itertools import combinations, permutations
from statistics import mean
from scipy.spatial.distance import euclidean
from experiments.rollout import rollout
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


def monte_carlo_shapley_values(env_name, n_players, agent, num_steps, n_random_coalitions=100, num_episodes=1, replace_missing_players="random"):
    """
    Monte Carlo estimation of shapley values (if k = num_episodes*n_random_coalitions, o(k*n_agents) complexity).
    Parameters:
        n_random_coalitions (optional, default=100): Number of random coalitions used to estimate shapley value for each player. 
        replace_missing_players (optional, default="random"): "random" to replace a player absent from the coalition by random actions
            "random_player_action" to replace action with a random action from another player or "idle" to let it doing nothing
        See rollout function for other parameters.
    """

    print("Starting Shapley value estimation on:", n_players,
          " n_random_coalitions=", n_random_coalitions)

    shapley_values = []
    players_ids = list(range(n_players))
    for player_id in players_ids:
        print(f"Starting computation for player {player_id}...")
        # Get all possible combinations with and without the current player
        coalitions_with_player = compute_coalitions_with_player(
            player_id, players_ids)
        marginal_contributions = []
        for c in range(1, 1+n_random_coalitions):

            # Select random coalitions
            coalition_with_player = random.choice(coalitions_with_player)
            coalition_without_player = [
                n for n in coalition_with_player if n != player_id]
            print(
                f"Coalition {c}/{n_random_coalitions}: {coalition_with_player}")

            # Simulate num_episodes episodes on selected coalitions with and without current player
            # num episode replaces num steps when used (num steps value is useless here)
            reward_with_player = rollout(
                agent, env_name, num_steps=1, num_episodes=1, coalition=coalition_with_player,
                verbose=False)[0]
            reward_without_player = rollout(
                agent, env_name, num_steps=1, num_episodes=1,
                coalition=coalition_without_player, replace_missing_players=replace_missing_players,
                verbose=False)[0]

            # Compute estimated marginal contribution
            marginal_contribution = reward_with_player - reward_without_player
            marginal_contributions.append(marginal_contribution)

        # Compute shapley value as the mean of marginal contributions
        shapley_value = mean(marginal_contributions)
        shapley_values.append(shapley_value)
        print(f"Shapley value for player {player_id}: {shapley_value}")

    print("Shapley values:", shapley_values)

    return shapley_values


# Old version

def compute_marginal_contributions(env_name, features, num_steps, num_episodes, agent, replace_missing_players):
    'Get mean reward for each agent for each coalitions '
    marginal_contributions = dict()
    for coalition in compute_coalitions(features):
        total_rewards = rollout(
            agent, env_name, num_steps, num_episodes, coalition=coalition, replace_missing_players=replace_missing_players, verbose=False)

        marginal_contributions[str(coalition)] = round(mean(total_rewards), 2)
    if DEBUG:
        print("Coalition values: ", marginal_contributions)
    return marginal_contributions


def exact_shapley_values(env_name, n_agents, agent, num_steps=10, num_episodes=1, replace_missing_players="random"):
    'Naive implementation (not optimized at all)'
    agents_ids = range(n_agents)
    coalition_values = compute_marginal_contributions(
        env_name, agents_ids, num_steps, num_episodes, agent, replace_missing_players)
    shapley_values = []

    for agent_id in agents_ids:
        if DEBUG:
            print("Computing shap value for agent: ", agent_id)
        shapley_value = 0

        for permutation in permutations(agents_ids):
            to_remove = []
            if DEBUG:
                print("permutation ", permutation)
            for i, x in enumerate(permutation):
                if x == agent_id:
                    coalition = sorted(permutation[:i+1])
                    if DEBUG:
                        print("coalition", coalition)
                    shapley_value += coalition_values[str(coalition)]

                    if len(to_remove) > 0:
                        to_remove = str(sorted(to_remove))
                        shapley_value -= coalition_values[to_remove]
                        if DEBUG:
                            print("to remove ", to_remove)
                    break
                else:
                    to_remove.append(x)
        shapley_values.append(shapley_value)

    result = np.divide(shapley_values, np.math.factorial(n_agents))
    print("Shapley values:", result)

    return result
