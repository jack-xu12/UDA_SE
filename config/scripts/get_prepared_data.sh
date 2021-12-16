#! /bin/bash

# split train_test dataset when you need
# python data_utils/get_dataset_split.py

python data_utils/preprocess.py \
	--max_seq_length=128 \
	--task_name=oracler \
	--raw_data_dir=data/oracle_raw \
	--output_base_dir=data/proc_data/oracle_raw/dev \
	--data_type=sup \
	--sub_set=dev \
	--vocab_file=../BERT_Base_Uncased/vocab.txt

# get sup train data

python data_utils/preprocess.py \
	--max_seq_length=128 \
	--task_name=githubr \
	--raw_data_dir=data/proc_data/github_raw \
	--output_base_dir=data/proc_data/github_raw/train \
	--data_type=sup \
	--sub_set=train \
	--vocab_file=../BERT_Base_Uncased/vocab.txt \
	--use_label=multiple \

# get sup dev data

python data_utils/preprocess.py \
	--max_seq_length=128 \
	--task_name=githubr \
	--raw_data_dir=data/proc_data/github_raw \
	--output_base_dir=data/proc_data/github_raw/dev \
	--data_type=sup \
	--sub_set=dev \
	--vocab_file=../BERT_Base_Uncased/vocab.txt \
	--use_label=multiple \


# get unsup data ori for 3 columns, aug for 3 lines, label

python data_utils/preprocess.py \
	--max_seq_length=128 \
	--task_name=githubr \
	--raw_data_dir=data/proc_data/github_raw \
	--output_base_dir=data/proc_data/github_raw/unsup \
	--data_type=unsup \
	--sub_set=unsup_in \
	--vocab_file=../BERT_Base_Uncased/vocab.txt \
	--back_translation_dir=data/proc_data/github_raw \
	--aug_ops=bt-0.9 \
	--aug_copy_num=0 \
	--use_label=multiple \

