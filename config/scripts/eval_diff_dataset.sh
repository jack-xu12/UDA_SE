#! /bash/sh

declare -A model_step_dict

model_step_dict['80_5']=5500
model_step_dict['80_6']=2000
model_step_dict['100_4']=2000
model_step_dict['100_9']=7750
model_step_dict['100_22']=3000
diff_dataset=('data_utils/data/proc_data/AppReviews/dev/appr_sup_dev.csv' 'data_utils/data/proc_data/github_raw/dev/githubr_sup_dev.csv' 'data_utils/data/proc_data/oracle_raw/dev/oracler_sup_dev.csv')

for eval_dataset in "${diff_dataset[@]}"
do
	echo "============================================"
    echo "start eval on ${eval_dataset}"
	for key in $(echo ${!model_step_dict[*]})
	do
	  array=(`echo $key | tr '_' ' '` )
	  path="res/results_lr_3e-5_warmup_0.2_cosine_${array[0]}_10000_random_${array[1]}/save/model_steps_${model_step_dict[$key]}.pt"
	  echo "${path}"
	  cat config/eval.json | jq \
		   --arg ED "$eval_dataset" --arg MP "$path"\
		   'to_entries | 
		   map(if .key == "model_file"
			  then . + {"value":$MP}
			  elif .key == "eval_data_dir"
			  then . + {"value":$ED}
			  else .
			  end
			 ) |
		  from_entries' > config/eval_diff_dataset.json
	  cat config/eval_diff_dataset.json | jq '.eval_data_dir'
	  echo '#################################################'
	  python main.py --cfg='config/eval_diff_dataset.json' --model_cfg='config/bert_base.json'
	  echo '################################################'
	done
	
	echo "============================================"

done