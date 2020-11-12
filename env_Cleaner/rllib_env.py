import argparse

import cv2
import numpy as np
import ray
from gym.spaces import Box, Discrete, Tuple, Dict
from ray import tune
from ray.rllib.agents import ppo
from ray.rllib.env import MultiAgentEnv
from ray.rllib.env.multi_agent_env import ENV_STATE
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.tune.config_parser import make_parser
from ray.tune.registry import register_env

from env_Cleaner import EnvCleaner


def create_parser(parser_creator=None):
    parser = make_parser(
        parser_creator=parser_creator,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Train a reinforcement learning agent.")
    parser.add_argument('--num-iters', type=int, default=100)
    parser.add_argument('--render', action="store_true",
                        help="Render the environement during training")
    parser.add_argument("--resume", action="store_true",
                        help="Whether to attempt to resume previous Tune experiments.")
    return parser


def gen_policies(obs_space, act_space, num_agents):
    'Generate policies dict {policy_name: (policy_type, obs_space, act_space, {"agent_id": i})}'
    policy = (None, obs_space, act_space)
    agent_names = [f"agent_{i}" for i in range(num_agents)]
    policies = {policy_agent_mapping(agent_name): policy + ({"agent_id": agent_id},)
                for agent_id, agent_name in enumerate(agent_names)}
    return policies


def policy_agent_mapping(agent_name):
    'Maps agent name to policy name'
    return f"policy_{agent_name}"


OBS_SIZE = 42
ACT_SPACE = 4


class CleanerWrapper(MultiAgentEnv):

    def __init__(self, env_config):
        self.env = EnvCleaner(
            env_config['N_agent'], env_config['map_size'], env_config['seed'], env_config['max_iters'])
        self.with_state = env_config['with_state']
        self._render = env_config.get("render", False)
        self._actions_are_logits = env_config.get("actions_are_logits", False)

    def step(self, action_dict):
        if self._actions_are_logits:
            action_dict = {
                k: np.random.choice(range(ACT_SPACE), p=v)
                for k, v in action_dict.items()
            }

        if self._render:
            self.env.render()
        action_list = []
        for k in action_dict:
            action_list.append(action_dict[k])
        reward = self.env.step(action_list)
        obs = self.env.get_global_obs()
        ep_over = self.env.is_episode_over()
        reward_dict = {}
        obs_dict = {}
        dones = {'__all__': True if ep_over else False}
        infos = {}
        for k in action_dict:
            reward_dict[k] = reward
            obs_dict[k] = self._preprocess(obs)
            infos[k] = {}
            dones[k] = True if ep_over else False
        return obs_dict, reward_dict, dones, infos

    def reset(self):
        self.env.reset()
        obs = self.env.get_global_obs()
        obs = self._preprocess(obs)
        obs_dict = {f'agent_{i}': obs for i in range(self.env.N_agent)}

        return obs_dict

    def _preprocess(self, obs):
        obs = cv2.resize(np.float32(obs), (OBS_SIZE, OBS_SIZE))
        if self.with_state:
            obs = {
                "obs": obs,
                ENV_STATE: obs
            }
        return obs

    def render(self):
        self.env.render()


class CustomModel(TFModelV2):
    """Example of a keras custom model that just delegates to an fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(CustomModel, self).__init__(obs_space, action_space, num_outputs,
                                          model_config, name)
        self.model = FullyConnectedNetwork(obs_space, action_space,
                                           num_outputs, model_config, name)
        self.register_variables(self.model.variables())

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    ModelCatalog.register_custom_model(
        "my_model", CustomModel)
    # env = EnvCleaner(4, 42, 0, 500)
    # while True:
    #     env.render()
    n_agent = 3
    map_size = 43
    max_iters = 1000
    obs_space = Box(low=0.0, high=1.0, shape=(
        OBS_SIZE, OBS_SIZE, 3), dtype=np.float32)
    act_space = Discrete(ACT_SPACE)
    policies = gen_policies(obs_space, act_space, n_agent)

    env_config = {
        "N_agent": n_agent,
        "map_size": map_size,
        "seed": None,
        "max_iters": max_iters,
        "render": args.render,
        "actions_are_logits": args.run.upper() == "CONTRIB/MADDPG",
        "with_state": True
    }

    if args.run.upper() == "PPO":
        config = {
            "env": CleanerWrapper,
            "env_config": {
                "N_agent": n_agent,
                "map_size": map_size,
                "seed": None,
                "max_iters": max_iters,
                "render": args.render
            },
            "model": {
                "custom_model": "my_model"
            },
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_agent_mapping
            }
        }

    elif args.run.upper() == "QMIX":
        # Qmix spaces
        obs_space = Tuple([
            Dict({
                "obs": obs_space,
                ENV_STATE: obs_space
            }) for _ in range(n_agent)
        ])
        act_space = Tuple([
            act_space for _ in range(n_agent)
        ])
        grouping = {
            "agents": [f'agent_{i}' for i in range(n_agent)],
        }

        # Create and register env

        def create_env_qmix(env_config): return CleanerWrapper(env_config).with_agent_groups(grouping,
                                                                                             obs_space=obs_space,
                                                                                             act_space=act_space)

        register_env('cleaner_qmix', create_env_qmix)

        config = {

            # === QMix ===
            # Mixing network. Either "qmix", "vdn", or None
            "mixer": "qmix",
            # Size of the mixing network embedding
            "mixing_embed_dim": 32,
            # Whether to use Double_Q learning
            "double_q": True,
            # Optimize over complete episodes by default.
            "batch_mode": "complete_episodes",

            # === Evaluation ===
            # Evaluate with epsilon=0 every `evaluation_interval` training iterations.
            # The evaluation stats will be reported under the "evaluation" metric key.
            # Note that evaluation is currently not parallelized, and that for Ape-X
            # metrics are already only reported for the lowest epsilon workers.
            "evaluation_interval": None,
            # Number of episodes to run per evaluation period.
            "evaluation_num_episodes": 10,

            # === Replay buffer ===
            # Size of the replay buffer in steps.
            "buffer_size": 10000,

            # === Optimization ===
            # Learning rate for RMSProp optimizer
            "lr": 0.0005,
            # RMSProp alpha
            "optim_alpha": 0.99,
            # RMSProp epsilon
            "optim_eps": 0.00001,
            # If not None, clip gradients during optimization at this value
            "grad_norm_clipping": 10,
            # How many steps of the model to sample before learning starts.
            "learning_starts": 1000,
            # Size of a batched sampled from replay buffer for training. Note that
            # if async_updates is set, then each worker returns gradients for a
            # batch of this size.
            "train_batch_size": 32,


            # 'num_gpus': int(os.environ.get("RLLIB_NUM_GPUS", "0")),
            # Whether to use a distribution of epsilons across workers for exploration.
            "per_worker_exploration": False,
            # Whether to compute priorities on workers.
            "worker_side_prioritization": False,
            # Prevent iterations from going lower than this time span
            "min_iter_time_s": 1,

            # === Model ===
            "model": {
                "lstm_cell_size": 64,
            },

            # === COMMON CONFIG ===
            'env': 'cleaner_qmix',
            'env_config': env_config
        }

    elif args.run.upper() == "CONTRIB/MADDPG":

        config = {

            "num_workers": 2,

            # === Framework to run the algorithm ===
            "framework": "tf",
            # === Evaluation ===
            # Evaluation interval
            "evaluation_interval": None,
            # Number of episodes to run per evaluation period.
            "evaluation_num_episodes": 10,

            # === Model ===
            # Apply a state preprocessor with spec given by the "model" config option
            # (like other RL algorithms). This is mostly useful if you have a weird
            # observation shape, like an image. Disabled by default.
            "use_state_preprocessor": True,
            # === Replay buffer ===
            # Size of the replay buffer. Note that if async_updates is set, then
            # each worker will have a replay buffer of this size.
            "buffer_size": int(1e6),
            # Observation compression. Note that compression makes simulation slow in
            # MPE.
            "compress_observations": False,
            # If set, this will fix the ratio of replayed from a buffer and learned on
            # timesteps to sampled from an environment and stored in the replay buffer
            # timesteps. Otherwise, the replay will proceed at the native ratio
            # determined by (train_batch_size / rollout_fragment_length).
            "training_intensity": None,

            # === Optimization ===
            # Learning rate for the critic (Q-function) optimizer.
            "critic_lr": 1e-2,
            # Learning rate for the actor (policy) optimizer.
            "actor_lr": 1e-2,
            # Update the target network every `target_network_update_freq` steps.
            "target_network_update_freq": 0,
            # Update the target by \tau * policy + (1-\tau) * target_policy
            "tau": 0.01,
            # Weights for feature regularization for the actor
            "actor_feature_reg": 0.001,
            # If not None, clip gradients during optimization at this value
            "grad_norm_clipping": 0.5,
            # How many steps of the model to sample before learning starts.
            "learning_starts": 1024 * 25,
            # Update the replay buffer with this many samples at once. Note that this
            # setting applies per-worker if num_workers > 1.
            "rollout_fragment_length": 10,
            # Size of a batched sampled from replay buffer for training. Note that
            # if async_updates is set, then each worker returns gradients for a
            # batch of this size.
            "train_batch_size": 16,
            # Number of env steps to optimize for before returning
            "timesteps_per_iteration": 0,
            # Prevent iterations from going lower than this time span
            "min_iter_time_s": 0,


            # "model": {
            #     "custom_model": "my_model"
            # },

            # === COMMON CONFIG ===
            "env": CleanerWrapper,

            "env_config": env_config,

            'batch_mode': 'truncate_episodes',
            'log_level': 'DEBUG',

            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_agent_mapping
            }
        }

    ray.init()

    tune.run(
        args.run,
        stop={'training_iteration': args.num_iters},
        checkpoint_freq=args.checkpoint_freq,
        resume=args.resume,
        config=config)
    ray.shutdown()
