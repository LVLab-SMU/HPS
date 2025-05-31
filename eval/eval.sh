reward_model=''
output_dir=''
output_dir1=''

accelerate launch --config_file ./configs/inference.yaml ./eval/eval_score.py --dataset_name_or_path ${output_dir} --output_dir ${output_dir1} --reward_name_or_path ${reward_model}