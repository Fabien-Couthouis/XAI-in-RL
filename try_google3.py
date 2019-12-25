import multiprocessing
import os

from baselines import logger
from baselines.bench import monitor
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.ppo2 import ppo2
import gfootball.env as env
from gfootball.examples import models
import tensorflow as tf
multiprocessing.set_start_method('spawn', True)

load_path = "CP_11_vs_11_easy_stochastic_v2"
# ncpu = multiprocessing.cpu_count()
# config = tf.ConfigProto(allow_soft_placement=True,
#                         intra_op_parallelism_threads=ncpu,
#                         inter_op_parallelism_threads=ncpu)
# config.gpu_options.allow_growth = True
# tf.Session(config=config).__enter__()

# vec_env = SubprocVecEnv([
#     (lambda _i=i: create_single_football_env(_i))
#     for i in range(4)
# ], context=None)
# env= env.create_environment("11_vs_11_stochastic")
cfg = config.Config({
      'action_set': "default",
      'dump_full_episodes': True,
      'players': players,
      'real_time': False,
  })

env.football_env.FootballEnv()
ppo2.learn(network='gfootball_impala_cnn',
           total_timesteps=0,
           env=env,
           #    seed=FLAGS.seed,
           #    nsteps=FLAGS.nsteps,
           #    nminibatches=FLAGS.nminibatches,
           #    noptepochs=FLAGS.noptepochs,
           #    max_grad_norm=FLAGS.max_grad_norm,
           #    gamma=FLAGS.gamma,
           #    ent_coef=FLAGS.ent_coef,
           #    lr=FLAGS.lr,
           #    log_interval=1,
           #    save_interval=FLAGS.save_interval,
           #    cliprange=FLAGS.cliprange,
           load_path=load_path)
