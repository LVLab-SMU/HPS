# The generated samples will be stored at output_dir + local_index + ".jsonl
mkdir data

my_world_size=8
infer_model=RLHFlow/Llama3-v2-iterative-DPO-iter3
prompt_dir='./data/online_hh.json'

output_dir=./data/online_hh_data

conda activate vllm
CUDA_VISIBLE_DEVICES=0 python ./generation/gen_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 100 --temperature 0.9 --local_index 0 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=1 python ./generation/gen_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 100 --temperature 0.9 --local_index 1 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=2 python ./generation/gen_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 100 --temperature 0.9 --local_index 2 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=3 python ./generation/gen_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 100 --temperature 0.9 --local_index 3 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=4 python ./generation/gen_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 100 --temperature 0.9 --local_index 4 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=5 python ./generation/gen_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 100 --temperature 0.9 --local_index 5 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=6 python ./generation/gen_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 100 --temperature 0.9 --local_index 6 --my_world_size ${my_world_size}  &
CUDA_VISIBLE_DEVICES=7 python ./generation/gen_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 100 --temperature 0.9 --local_index 7 --my_world_size ${my_world_size}  &

# Then we merge the generated datasets into one dataset
wait
python ./generation/merge_data.py --base_path ${output_dir} --output_dir ./data/online_hh_data.json --num_datasets ${my_world_size}

# Data annotation
accelerate launch ./annotate_data/get_rewards.py --dataset_name_or_path ./data/online_hh_data.json --output_dir ./data/hh_data_all.jsonl