import numpy as np
from utilities.trainer import *
import torch
from arguments import *
import os
from utilities.util import *
from utilities.logger import Logger
import argparse


print("Initialisation...\n")

parser = argparse.ArgumentParser(description='Test rl agent.')
parser.add_argument('--save-path', type=str, nargs='?', default='./',
                    help='Please input the directory of saving model.')
argv = parser.parse_args()

if argv.save_path[-1] is '/' or argv.save_path[-1] is './':
    save_path = argv.save_path
else:
    save_path = argv.save_path+'/'

# create save folders
if 'model_save' not in os.listdir(save_path):
    os.mkdir(save_path+'model_save')
if 'tensorboard' not in os.listdir(save_path):
    os.mkdir(save_path+'tensorboard')
if log_name not in os.listdir(save_path+'model_save/'):
    os.mkdir(save_path+'model_save/'+log_name)
if log_name not in os.listdir(save_path+'tensorboard/'):
    os.mkdir(save_path+'tensorboard/'+log_name)
else:
    path = save_path+'tensorboard/'+log_name
    for f in os.listdir(path):
        file_path = os.path.join(path, f)
        if os.path.isfile(file_path):
            os.remove(file_path)

logger = Logger(save_path+'tensorboard/' + log_name)

model = Model[model_name]

strategy = Strategy[model_name]

print("Args passed are:", '{}\n'.format(args))

if strategy == 'pg':
    train = PGTrainer(args, model, env(), logger, args.online)
elif strategy == 'q':
    raise NotImplementedError('This needs to be implemented.')
else:
    raise RuntimeError('Please input the correct strategy, e.g. pg or q.')

stat = dict()
print("Start training for {} episodes... \n".format(args.train_episodes_num))

for i in range(args.train_episodes_num):
    train.run(stat)
    train.logging(stat)
    if i % args.save_model_freq == args.save_model_freq-1:
        train.print_info(stat)
        torch.save({'model_state_dict': train.behaviour_net.state_dict()},
                   save_path+'model_save/'+log_name+'/model.pt')
        print('The model is saved!\n')
        with open(save_path+'model_save/'+log_name + '/log.txt', 'w+') as file:
            file.write(str(args)+'\n')
            file.write(str(i))

print("End of training")
