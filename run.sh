conda activate rlhflow
accelerate launch --config_file ./configs/zero2.yaml dpo_iteration/run_dpo.py ./configs/training.yaml