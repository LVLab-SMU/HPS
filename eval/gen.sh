my_world_size=4
infer_model=''
prompt_dir=''

# mkdir data
output_dir=''
output_dir1=''

# conda activate vllm
CUDA_VISIBLE_DEVICES=0 python ./eval/eval_gene.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 5 --temperature 0.9 --local_index 0 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=1 python ./eval/eval_gene.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 5 --temperature 0.9 --local_index 1 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=2 python ./eval/eval_gene.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 5 --temperature 0.9 --local_index 2 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=3 python ./eval/eval_gene.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 5 --temperature 0.9 --local_index 3 --my_world_size ${my_world_size}  &

wait
python ./eval/merge_data.py --base_path ${output_dir} --output_dir ${output_dir1}