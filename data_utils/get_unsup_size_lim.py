import argparse
import os
import pandas as pd
import numpy as np
import math
from sklearn.utils import shuffle
import ast
from datetime import datetime
import random
import torch
import time


parser = argparse.ArgumentParser()
parser.add_argument('--sub_size', type=int, nargs='+', help='')
# parser.add_argument('--unsup_file', default='preprocessed_data/sosc_data/back/sosc_unsup_train_last_9.csv', help='')
parser.add_argument('--unsup_file', default='data_utils/data/proc_data/github_raw/github_golden_train_bt_7.csv', help='')
parser.add_argument('-s', '--random_seed', type=int, default=43, help='set random seed')
parser.add_argument('--exp_time', type=int, help='the exp time')

def set_seeds(seed):
    "set random seeds"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':


    args = parser.parse_args()
    exp_time = args.exp_time
    unsup_examples = pd.read_csv(args.unsup_file, sep=',')
    set_seeds(args.random_seed)

    sub_sizes = args.sub_size
    for s in sub_sizes:
        _idx = np.arange(unsup_examples.shape[0])
        _idx_select = np.random.permutation(_idx)[:s]

        unsup_examples_selected = unsup_examples.iloc[_idx_select, :]

        # dir_path = 'preprocessed_data/sosc_data/back/'
        # unsup_examples_selected.to_csv(os.path.join(dir_path, f'sosc_unsup_train_last_9_{s}_{exp_time}.csv'), index=False, encoding='utf-8')

        dir_path = 'preprocessed_data/github_data/back_trans/unsup'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        unsup_examples_selected.to_csv(os.path.join(dir_path, f'github_golden_train_bt_7_{s}_{exp_time}.csv'),
                                       index=False, encoding='utf-8')


