#! /bash/sh


exp_time=$1
echo "exp_time $exp_time"
unsup_size=(800 1000 1200 1400 1600)

for u_s in "${unsup_size[@]}"
do
	echo "#########################################"
	unsup_data_dir="preprocessed_data/sosc_data/back/sosc_unsup_train_last_9_${u_s}_${exp_time}.csv"
	results_dir="res/results_lr_3e-5_warmup_0.2_cosine_100_8000_random_22_${u_s}_${exp_time}"
	echo "$unsup_data_dir"
	echo "$results_dir"
	cat config/uda.json | jq \
	   --arg UDD "$unsup_data_dir" --arg RD "$results_dir"\
	   'to_entries | 
       map(if .key == "unsup_data_dir" 
          then . + {"value":$UDD}
		  elif .key == "results_dir"
		  then . + {"value":$RD}
		  elif .key == "check_steps"
		  then . + {"value": 250}
          else . 
          end
         ) | 
      from_entries' > config/uda_diff_unsup_size.json
	
	cat config/uda_diff_unsup_size.json | jq '.unsup_data_dir'
	python main.py --cfg='config/uda_diff_unsup_size.json' --model_cfg='config/bert_base.json' --freeze=True --save_pt=False
	
	
	echo "#########################################"
	
	
done
