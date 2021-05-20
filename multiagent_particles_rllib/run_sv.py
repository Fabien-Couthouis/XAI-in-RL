
import argparse
import ray

from rollout import rollout, load_agent_config
from shapley_values import monte_carlo_shapley_values

import json


def create_parser(parser_creator=None):
    parser_creator = parser_creator or argparse.ArgumentParser
    parser = parser_creator(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Roll out a reinforcement learning agent "
        "given a checkpoint.")

    parser.add_argument(
        "checkpoint",
        type=str,
        nargs="?",
        help="(Optional) checkpoint from which to roll out. "
        "If none given, will use an initial (untrained) Trainer.")

    required_named = parser.add_argument_group("required named arguments")
    required_named.add_argument(
        "--run",
        type=str,
        required=True,
        help="The algorithm or model to train. This may refer to the name "
        "of a built-on algorithm (e.g. RLLib's `DQN` or `PPO`), or a "
        "user-defined trainable function or class registered in the "
        "tune registry.")
    required_named.add_argument(
        "--env",
        type=str,
        help="The environment specifier to use. This could be an openAI gym "
        "specifier (e.g. `CartPole-v0`) or a full class-path (e.g. "
        "`ray.rllib.examples.env.simple_corridor.SimpleCorridor`).")
    parser.add_argument(
        "--local-mode",
        action="store_true",
        help="Run ray in local mode for easier debugging.")
    parser.add_argument(
        "--no-render",
        default=False,
        action="store_const",
        const=True,
        help="Suppress rendering of the environment.")
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Specifies the directory into which videos of all episode "
        "rollouts will be stored.")
    parser.add_argument(
        "--steps",
        default=10000,
        help="Number of timesteps to roll out. Rollout will also stop if "
        "`--episodes` limit is reached first. A value of 0 means no "
        "limitation on the number of timesteps run.")
    parser.add_argument(
        "--episodes",
        default=0,
        help="Number of complete episodes to roll out. Rollout will also stop "
        "if `--steps` (timesteps) limit is reached first. A value of 0 means "
        "no limitation on the number of episodes run.")
    parser.add_argument("--out", default=None, help="Output filename.")
    parser.add_argument(
        "--config",
        default="{}",
        type=json.loads,
        help="Algorithm-specific configuration (e.g. env, hyperparams). "
        "Gets merged with loaded configuration from checkpoint file and "
        "`evaluation_config` settings therein.")
    parser.add_argument(
        "--save-info",
        default=False,
        action="store_true",
        help="Save the info field generated by the step() method, "
        "as well as the action, observations, rewards and done fields.")
    parser.add_argument(
        "--use-shelve",
        default=False,
        action="store_true",
        help="Save rollouts into a python shelf file (will save each episode "
        "as it is generated). An output filename must be set using --out.")
    parser.add_argument(
        "--track-progress",
        default=False,
        action="store_true",
        help="Write progress to a temporary file (updated "
        "after each episode). An output filename must be set using --out; "
        "the progress file will live in the same folder.")

   
    parser.add_argument(
        '--num-rollouts',
        type=int,
        default=1,
        help='The number of rollouts to visualize.')
    parser.add_argument("--shapley-M", type=int, default=None,
                        help="compute or not shapley values with given number of simulation episodes (M)")
    parser.add_argument("--missing-agents-behaviour", type=str, default="random_player_action",
                        help="behaviour of agents not in the coalition: random_player (take a random player mode from a from in the coalition) or random (random move) or idle (do not move)")
    parser.add_argument("--exp-name", type=str, default="run_0",
                        help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="logs",
                        help="directory in which shapley logs should be saved")
    parser.add_argument(
        "--agents-active", type=int, default=3, help='number of active agents'
    )
    parser.add_argument(
        "--social-metrics", action='store_true',
        help='whether to save rewards to compute social metrics')
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    ray.init()
    agent, config = load_agent_config(args)

    if args.shapley_M is not None:
        monte_carlo_shapley_values(args, agent, config, args.agents_active)

    else:
        rollout(args, agent, config, args.num_rollouts)
