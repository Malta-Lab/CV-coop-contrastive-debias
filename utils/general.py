import random

import numpy as np
import torch
import os
from torch.utils.data import Dataset
from loaders import CelebA, UTKface

def set_seeds(seed: int) -> None:
    '''
    Sets the seeds for commonly-used pseudorandom number generators (python,
    numpy, torch). Works for CPU and GPU computations. Also configures the
    cuDNN backend to be as deterministic as possible.
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # for CPU and GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_dataset(args, split:str, is_biased:str, bias_prop:float) -> Dataset:
    if args.dataset == 'celeba':
            return CelebA(
            root = args.dataset_path,
            split = split,
            target = args.target,
            bias = args.bias,
            biased = is_biased,
            bias_prop=bias_prop,
            seed = args.seed,
            attributes_path = os.path.join(args.dataset_path, 'celeba', 'list_attr_celeba.csv'),
            download = False
        )
    
    if args.dataset == 'utkface':
        return UTKface(
            root = args.dataset_path,
            split = split,
            bias = args.bias,
            label = args.label
        )