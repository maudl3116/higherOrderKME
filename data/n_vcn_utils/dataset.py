import os
import random
import itertools

import numpy as np

from config import gen_args
from n_bodies_generator import PhysicsDataset, load_data
from utils import rand_int, count_parameters, Tee, AverageMeter, get_lr, to_np, set_seed

args = gen_args()
set_seed(args.random_seed)

np.random.seed(args.random_seed)

print(args)

datasets = {}
dataloaders = {}
data_n_batches = {}
for phase in ['train', 'valid']:
    datasets[phase] = PhysicsDataset(args, phase=phase) 

    if args.gen_data:
        datasets[phase].gen_data()
    else:
        datasets[phase].load_data()
args.stat = datasets['train'].stat
