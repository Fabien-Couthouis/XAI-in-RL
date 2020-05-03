import numpy as np
import random
import csv
from itertools import combinations, permutations
from statistics import mean
from math import factorial
from rollout import rollout

DEBUG = False  # print debugging info if set to true
GAMMA_LIST = [1, 0.99, 0.95, 0.9]  # discount factors


def get_combinations(features):
    'Get all possible coalitions between features'
    combinations_list = []
    for i in range(1, len(features)+1):
        oc = combinations(features, i)
        for c in oc:
            combinations_list.append(list(c))
    return combinations_list


def get_combinations_for_feature(features, feature_id):
    combinations = get_combinations(features)
    with_player = []
    for feature in combinations:
        if feature_id in feature:
            with_player.append(feature)

    return with_player


def monte_carlo_shapley_estimation(env, arglist, M):
    estimated_values = []
    num_good = min(env.n, env.n-arglist.num_adversaries)
    features = range(num_good)
    for feature in features:
        print("Feature: ", feature, "/", num_good-1)
        coalitions_with_player = get_combinations_for_feature(
            features, feature)
        marginal_contributions = []
        for m in range(1, 1+M):
            print("m=", m, "/", M)
            coalition_with_player = random.choice(coalitions_with_player)
            coalition_without_player = coalition_with_player.copy().remove(feature)
            rollout_info_with = rollout(env,
                                        arglist, coalition_with_player, arglist.missing_agents_behaviour)
            rollout_info_without = rollout(env,
                                           arglist, coalition_without_player, arglist.missing_agents_behaviour)

            values_with_player = np.mean(
                rollout_info_with["episode_rewards"])
            values_without_player = np.mean(
                rollout_info_without["episode_rewards"])
            # print(values_with_player)
            marginal_contribution = values_with_player - values_without_player
            marginal_contributions.append(marginal_contribution)
            save_rollout_info(arglist, feature, m,
                              rollout_info_with, rollout_info_without)

        # TODO: normalization (minmax?)

        shapley_value = sum(
            marginal_contributions)/len(marginal_contributions)
        estimated_values.append(shapley_value)

    print("Estimated values:", estimated_values)

    return estimated_values


def save_rollout_info(arglist, feature, m, rollout_info_with, rollout_info_without):
    'Keep track of marginal contributions in csv file while computing shapley values'
    file_name = f"{arglist.save_dir}/{arglist.exp_name}.csv"
    rewards_with = np.mean(rollout_info_with["episode_rewards"], axis=0)
    discounted_rewards_with = []
    rewards_without = np.mean(rollout_info_without["episode_rewards"], axis=0)
    discounted_rewards_without = []
    for gamma in GAMMA_LIST:
        discounted_rewards_with.append(
            np.mean(discount_rewards(rewards_with, gamma)))
        discounted_rewards_without.append(np.mean(
            discount_rewards(rewards_without, gamma)))
    with open(file_name, "a", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(
            [feature, m, arglist.missing_agents_behaviour] + [reward for reward in discounted_rewards_with] + [reward for reward in discounted_rewards_without])


def discount_rewards(rewards, gamma):
    'Returns rewards discounted by GAMMA factor'
    discounted_rewards = np.zeros_like(rewards)
    R = 0
    for t in reversed(range(0, len(rewards))):
        R = R * gamma + rewards[t]
        discounted_rewards[t] = R
    return discounted_rewards

# def get_marginal_contributions(env, features, num_episodes, behaviour_nets):
#     'Get mean reward for each agent for each coalitions '
#     coalition_values = dict()
#     for coalition in get_combinations(features):
#         total_rewards = rollout(
#                 arglist, coalition=coalition, missing_agents_bahaviour="random_player")
#         coalition_values[str(coalition)] = round(mean(total_rewards), 2)
#     if DEBUG:
#         print("Coalition values: ", coalition_values)
#     return coalition_values


# def shapley_values(env, behaviour_nets, num_episodes=10):
#     'Naive implementation (not optimized at all)'
#     agents_ids = range(env.n)
#     coalition_values = get_marginal_contributions(
#         env, agents_ids, num_episodes, behaviour_nets)
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

#     return np.divide(shapley_values, np.math.factorial(env.get_num_of_agents()))
