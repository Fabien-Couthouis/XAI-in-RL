#python rollout.py /home/fabien/ray_results/PPO/PPO_g_football_7cae6c8e_2020-01-20_14-05-09e0rnlm86/checkpoint_100/checkpoint-100 --env gfootball --run PPO --compute-shapley True --episodes 10 --steps 10000
python rollout.py good_model/checkpoint_2250/checkpoint-2250 --env gfootball --run PPO --episodes 100 --steps 10000
