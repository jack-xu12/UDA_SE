#! /bash/sh

declare -A sup_num

sup_num['60']='062312/062314/062316/062318/062320'
sup_num['120']='062717/062719/062721/062723/062725'
sup_num['180']='062805/062807/062809/062811/062813'

for key in $(echo ${!sup_num[*]})
do
	array=(`echo ${sup_num[$key]} | tr '/' ' '`)
	t=1
	for v in "${array[@]}"
	do
		sdd="preprocessed_data/github_data/back_trans/${key}/github_sup_train_${key}_random_${v}.csv"
		rd="res/github/results_lr_3e-5_warmup_0.2_cosine_${key}_10000_random_${t}"
		lf="diff_sup_log_${key}.txt"
		cat config/uda.json | jq \
		   --arg SDD "$sdd" --arg RD "$rd" --arg LF "$lf"\
		   'to_entries | 
		   map(if .key == "sup_data_dir"
			  then . + {"value":$SDD}
			  elif .key == "results_dir"
			  then . + {"value":$RD}
			  elif .key == "log_file"
			  then . + {"value":$LF}
			  else .
			  end
			 ) |
		  from_entries' > config/sup_diff.json
		  
		cat config/sup_diff.json | jq '.sup_data_dir'
		python main.py --cfg='config/sup_diff.json' --model_cfg='config/bert_base.json' --freeze=True
		  
		t=$(($t+1))
		
	done
done
