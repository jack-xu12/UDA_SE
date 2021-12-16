import argparse
import os
import pandas as pd
import numpy as np
import math

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_dir', default='./data_utils/data/github_raw', help='Directory for the old sosc dataset')
parser.add_argument('-o', '--out_dir', default='./data_utils/data/proc_data/github_raw', help='Directory for the output')
parser.add_argument('-s','--select_method', default='last', help='select method in (first, random, last)')
parser.add_argument('-r','--split_rate', default='7:3', help='split Proportion in train:test')
parser.add_argument('-a','--aug_method', default='bt', help='method in [bt, tf_idf]')


if __name__ == '__main__':

    args = parser.parse_args()
    select_method = args.select_method
    source_dir = args.data_dir
    task_name = source_dir.split('/')[-1].split('_')[0]  # data_dir as XX/XX/taskname_XX
    task_name = task_name if task_name.find("github") == -1 else task_name+'_golden'
    out_dir = args.out_dir
    if args.aug_method  == 'bt':
        sup_data_path = os.path.join(source_dir, f'{task_name}_all.csv')
        unsup_data_path = os.path.join(source_dir, f'bt_{task_name}_all.csv')
    else:
        # TODO: add this part if we need tf_idf augment
        pass
        # unsup_data_path = os.path.join(source_dir, 'sosc_unsup_all_tf_idf.csv')


    if not os.path.exists(unsup_data_path):
        raise RuntimeError('we need Specified data files in raw data dir')
    
    df_sup = pd.read_csv(sup_data_path, sep=',')
    df_unsup = pd.read_csv(unsup_data_path, sep=',')

    df = pd.concat([df_sup, df_unsup], axis=1)
    assert df.shape[0] == df_unsup.shape[0], 'we need sup dataset has the same shape of the unsup one(same index)'


    df_pos = df[df['label'] == 'Positive']
    df_neu = df[df['label'] == 'Neutral']
    df_neg = df[df['label'] == 'Negative']

    split_rate = args.split_rate.split(':')
    test_prob = float(int(split_rate[1]) / (int(split_rate[0]) + int(split_rate[1])))
    len_pos = df_pos.shape[0]
    len_neg = df_neg.shape[0]
    len_neu = df_neu.shape[0]

    idx_pos = np.arange(len_pos)
    idx_neg = np.arange(len_neg)
    idx_neu = np.arange(len_neu)

    if select_method == 'first':
        idx_pos_select = idx_pos[:math.ceil(len_pos * test_prob)]
        idx_neg_select = idx_neg[:math.ceil(len_neg * test_prob)]
        idx_neu_select = idx_neu[:math.ceil(len_neu * test_prob)]

    elif select_method == 'last':
        idx_pos_select = idx_pos[-math.ceil(len_pos * test_prob):]
        idx_neg_select = idx_neg[-math.ceil(len_neg * test_prob):]
        idx_neu_select = idx_neu[-math.ceil(len_neu * test_prob):]

    elif select_method == 'random':
        idx_pos_select = np.sort(np.random.permutation(idx_pos)[:math.ceil(len_pos * test_prob)])
        idx_neg_select = np.sort(np.random.permutation(idx_neg)[:math.ceil(len_neg * test_prob)])
        idx_neu_select = np.sort(np.random.permutation(idx_neu)[:math.ceil(len_neu * test_prob)])

    else:
        raise RuntimeError('we should set select method in (first, last, random)')

    idx_pos_left = np.sort(np.setdiff1d(idx_pos, idx_pos_select))
    idx_neg_left = np.sort(np.setdiff1d(idx_neg, idx_neg_select))
    idx_neu_left = np.sort(np.setdiff1d(idx_neu, idx_neu_select))

    ###############  get output dataset  ###############

    _df_train_out = pd.concat([df_neg.iloc[idx_neg_left, :], df_neu.iloc[idx_neu_left, :], df_pos.iloc[idx_pos_left, :]])
    df_train_out = pd.DataFrame(_df_train_out, columns=['content', 'label', 'id'])
    df_unsup_train_out = pd.DataFrame(_df_train_out, columns=['bt_content', 'bt_label'])
    df_unsup_train_out.columns = ['content', 'label']
    df_test_out = pd.concat([df_neg.iloc[idx_neg_select, :],df_neu.iloc[idx_neu_select, :], df_pos.iloc[idx_pos_select, :]])
    df_test_out = pd.DataFrame(df_test_out, columns=['content', 'label', 'id'])



    if not os.path.exists(out_dir):
        os.mkdir(out_dir)


    train_sup_out_path = '{}_train_{}.csv'.format(task_name, split_rate[0])
    test_sup_out_path = '{}_test_{}.csv'.format(task_name, split_rate[1])
    train_unsup_out_path = '{}_train_bt_{}.csv'.format(task_name, split_rate[0])

    train_sup_out_path = os.path.join(out_dir, train_sup_out_path)
    test_sup_out_path = os.path.join(out_dir, test_sup_out_path)
    train_unsup_out_path = os.path.join(out_dir, train_unsup_out_path)

    df_train_out.to_csv(train_sup_out_path, index=False, encoding='utf-8')
    df_test_out.to_csv(test_sup_out_path, index=False, encoding='utf-8')
    df_unsup_train_out.to_csv(train_unsup_out_path,index=False, encoding='utf-8')






