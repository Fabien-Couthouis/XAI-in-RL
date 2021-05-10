
import argparse
import ray

from rollout import rollout, load_agent_config
from shapley_values import monte_carlo_shapley_values


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Evaluates a reinforcement learning agent '
                    'given a checkpoint.')

    # required input parameters
    parser.add_argument(
        'result_dir', type=str, help='Directory containing results')
    parser.add_argument('checkpoint_num', type=str, help='Checkpoint number.')

    # optional input parameters
    parser.add_argument(
        '--run',
        type=str,
        help='The algorithm or model to train. This may refer to '
             'the name of a built-on algorithm (e.g. RLLib\'s DQN '
             'or PPO), or a user-defined trainable function or '
             'class registered in the tune registry. '
             'Required for results trained with flow-0.2.0 and before.')
    parser.add_argument(
        '--num-rollouts',
        type=int,
        default=1,
        help='The number of rollouts to visualize.')
    parser.add_argument(
        '--render',
        action='store_true',
        help='whether to watch the rollout while it happens')
    parser.add_argument("--shapley-M", type=int, default=None,
                        help="compute or not shapley values with given number of simulation episodes (M)")
    parser.add_argument("--missing-agents-behaviour", type=str, default="random_player_action",
                        help="behaviour of agents not in the coalition: random_player (take a random player mode from a from in the coalition) or random (random move) or idle (do not move)")
    parser.add_argument("--exp-name", type=str, default="run_0",
                        help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="logs",
                        help="directory in which shapley logs should be saved")
    parser.add_argument(
        "--agents-active", type=int, default=6, help='number of active agents'
    )
    parser.add_argument(
        "--social-metrics", action='store_true',
        help='whether to save rewards to compute social metrics')
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    ray.init(num_cpus=2)
    agent, config = load_agent_config(args)

    if args.shapley_M is not None:
        monte_carlo_shapley_values(args, agent, config, args.agents_active)
    elif args.social_metrics:
        rollout(args, agent, config, args.num_rollouts)

    else:
        rollout(args, agent, config, args.num_rollouts)
