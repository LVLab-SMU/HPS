import json
import os
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline, AutoModelForSequenceClassification
from accelerate import Accelerator
import re

tqdm.pandas()

#####
# This script takes a dataset as the input, where each sample is {"prompt": "the pormpt", "responses": ["response1", "response2", "response3", ...]}
# The script will compute the reward for each input-output pair, and eventually output a new dataset, where each sample contains {"prompt": "the pormpt", "responses": ["response1", "response2", "response3", ...], "rewards": [reward1, reward2, ...]}
#####


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    dataset_name_or_path: Optional[str] = field(
        default="uf_split0_responses_K8.jsonl",
        metadata={"help": "the location of the dataset name or path"},
    )
    output_dir: Optional[str] = field(
        default="uf_split0_responses_K8_reward.json",
        metadata={"help": "the location of the output file"},
    )
    record_dir: Optional[str] = field(
        default=None,
        metadata={"help": "the location of the recording file"},
    )
    reward_name_or_path: Optional[str] = field(
        default="Skywork/Skywork-Reward-Llama-3.1-8B-v0.2",
        metadata={"help": "the name of the reward model"},
    )
    input_output_delimiter: Optional[str] = field(
        default="",
        metadata={"help": "the delimiter between input and output"},
    )
    K: Optional[int] = field(
        default=100,
        metadata={"help": "the number of responses per prompt"},
    )


accelerator = Accelerator()

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

device = accelerator.device
reward_model = script_args.reward_name_or_path


rm = AutoModelForSequenceClassification.from_pretrained(
    reward_model,
    torch_dtype=torch.bfloat16,
    device_map=device,
    attn_implementation="flash_attention_2",
    num_labels=1,
)
rm_tokenizer = AutoTokenizer.from_pretrained(reward_model)



ds_dir = script_args.dataset_name_or_path
world_size = int(os.getenv("WORLD_SIZE", "1"))
ds = load_dataset("json", data_files=ds_dir, split="train")

local_rank = Accelerator().local_process_index

data_size = len(ds["prompt"])

share = int(data_size / world_size) + 1
ds = ds.select(np.arange(local_rank * share, min((local_rank + 1) * share, len(ds))))

"""
We process the data format here and query the reward model to get the rewards.
"""


def get_reward(sample):
    rewards = []
    for i in range(len(sample["responses"])):
        message1 = re.sub(r'<\|start_header_id\|>system<\|end_header_id\|>.*?<\|eot_id\|>', '', sample["prompt"], flags=re.DOTALL)
        message2 = [{"role": "assistant", "content": sample["responses"][i]}]
        message2 = rm_tokenizer.apply_chat_template(message2, tokenize=False)
        message2 = re.sub(r'<\|begin_of_text\|><\|start_header_id\|>system<\|end_header_id\|>.*?<\|eot_id\|>', '', message2, flags=re.DOTALL)

        message_tokenized = rm_tokenizer(message1+message2, return_tensors="pt").to(device)
        
        with torch.no_grad():
            tmp_reward = rm(**message_tokenized).logits[0][0].item()
        rewards.append(tmp_reward)
    return rewards

data = []

# tqdm is used to show the progress bar
with torch.no_grad():
    for sample in tqdm(ds):
        # The VLLM may not generate responses for some prompts because it is too long, we skip them
        if len(sample["responses"]) < script_args.K:
            continue
        
        rewards = get_reward(sample)
        data.append({"prompt": sample["prompt"], "responses": sample["responses"], "rewards": rewards})


# Send the data to other GPUs
world_size = int(os.getenv("WORLD_SIZE", "1"))
all_process_list = [{}] * world_size

data_to_send = {
    "data": [[data[i]] for i in range(len(data))],
}

import torch.distributed as dist

dist.all_gather_object(all_process_list, data_to_send)
gathered_data = []


for i in range(world_size):
    tmp_data = [tmp[0] for tmp in all_process_list[i]["data"]]
    gathered_data.extend(tmp_data)

all_rewards = [sample["rewards"] for sample in gathered_data]
top1_scores = np.mean(np.max(all_rewards, axis=1))
mean_scores = np.mean(all_rewards)


if local_rank == 0:
    print(
        "Collect {} data from {} inputs. mean score {} top1 score: {}".format(
            len(gathered_data), data_size, mean_scores, top1_scores
        )
    )
    if len(gathered_data) < data_size:
        print(
            "Some of the prompts are with responses < {}. This can happen because the prompt is too long and is ignored by VLLM".format(
                script_args.K
            )
        )

    with open(script_args.output_dir, "w", encoding="utf8") as f:
        for i in range(len(gathered_data)):
            json.dump(gathered_data[i], f, ensure_ascii=False)
            f.write('\n')
            
    if script_args.record_dir is not None:
        with open(script_args.record_dir, "a") as f:
            f.write(str(mean_scores) + "\t" + str(top1_scores) + "\n")
