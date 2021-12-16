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
parser.add_argument('--sub_size', type=int, default='', help='')
parser.add_argument('--train_file', default='data_utils/data/proc_data/github_raw/train/githubr_sup_train_all.csv', help='')
parser.add_argument('-w', '--with_same_distribution', type=ast.literal_eval, dest='flag', help='')
parser.add_argument('-m', '--method', type=str, default='random', help='top bottom random')    # only random
parser.add_argument('-s', '--random_seed', type=int, default=43, help='set random seed')
parser.add_argument('-t', '--times', type=int, default=10, help='how many times to generate')
parser.add_argument('-u', '--use_label', type=str, default='multiple', help='choose in ["binary", "multiple"]')
parser.add_argument('--task_name', type=str, help='task name for choosing e.g. github')

def set_seeds(seed):
    "set random seeds"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)




if __name__ == '__main__':


    args = parser.parse_args()
    train_examples = pd.read_csv(args.train_file, sep=',')
    _label = args.use_label
    set_seeds(args.random_seed)


    train_pos = train_examples[train_examples['label_id'] == 1]
    train_neg = train_examples[train_examples['label_id'] == 0]

    if _label == 'multiple':
        train_neu = train_examples[train_examples['label_id'] == 1]
        train_pos = train_examples[train_examples['label_id'] == 2]


    wsd = args.flag

    if wsd:
        # we don't need wsd, it's a wrong function
        pos_prob = len(train_pos) / (len(train_pos) + len(train_neg))
    else:
        if _label == 'binary':
            prob = 0.5
        else:
            prob = 1.0/3

    if _label == 'binary':
        pos_select = math.ceil(args.sub_size * prob)
        neg_select = args.sub_size - pos_select
    else:
        pos_select = math.ceil(args.sub_size * prob)
        neu_select = math.ceil(args.sub_size * prob)
        neg_select = args.sub_size - pos_select - neu_select

    for i in range(args.times):

        time.sleep(2)

        r = datetime.now().strftime("%H%M%S")


        if args.method == 'top':
            df_pos = train_pos.iloc[:pos_select, :]
            df_neg = train_neg.iloc[:neg_select, :]

        elif args.method == 'bottom':
            df_pos = train_pos.iloc[-pos_select:, :]
            df_neg = train_neg.iloc[-neg_select:, :]

        else:
            idx_pos = np.arange(len(train_pos))
            idx_neg = np.arange(len(train_neg))
            idx_pos_select = np.random.permutation(idx_pos)[:pos_select]
            idx_neg_select = np.random.permutation(idx_neg)[:neg_select]

            df_pos = train_pos.iloc[idx_pos_select, :]
            df_neg = train_neg.iloc[idx_neg_select, :]

            if _label == 'multiple':
                idx_neu = np.arange(len(train_neu))
                idx_neu_select = np.random.permutation(idx_neu)[:neu_select]
                df_neu = train_neu.iloc[idx_neu_select, :]

        if _label == 'multiple':
            df = pd.concat([df_pos, df_neg, df_neu])
        else:
            df = pd.concat([df_pos, df_neg])
        df = shuffle(df)

        _task = args.task_name
        dir_path = 'preprocessed_data/{}_data/back_trans/{}'.format(_task, args.sub_size)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        if wsd:
            df.to_csv('{}/{}_sup_train_{}_{}_wsd.csv'.format(dir_path, _task, args.sub_size, args.method), index=False, encoding='utf-8')
        else:
            print('{}/{}_sup_train_{}_{}_{}.csv'.format(dir_path, _task, args.sub_size, args.method, r))
            df.to_csv('{}/{}_sup_train_{}_{}_{}.csv'.format(dir_path, _task, args.sub_size, args.method, r), index=False, encoding='utf-8')



