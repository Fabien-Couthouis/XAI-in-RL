#!/usr/bin/bash

python test.py experiments/models/MADDPG/shapley_5_vs_5/checkpoint_120000/checkpoint-120000 --env gfootball --run contrib/MADDPG --scenario-name "shapley_5_vs_5" --episodes 4 --num-agents 5 --no-render --save-replays #--compute-shapley --n-random-coalitions 50 --plot-shapley --no-render #--idle-missing-agents