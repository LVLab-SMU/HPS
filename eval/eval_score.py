import json
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, AutoModelForSequenceClassification
from accelerate import Accelerator
import torch.distributed as dist
import nltk

tqdm.pandas()

@dataclass
class ScriptArguments:
    dataset_name_or_path: Optional[str] = field(
        default="",
        metadata={"help": "the location of the dataset name or path"},
    )
    output_dir: Optional[str] = field(
        default="",
        metadata={"help": "the location of the output file"},
    )
    record_dir: Optional[str] = field(
        default=None,
        metadata={"help": "the location of the recording file"},
    )
    reward_name_or_path: Optional[str] = field(
        default="",
        metadata={"help": "the name of the reward model"},
    )
    input_output_delimiter: Optional[str] = field(
        default="",
        metadata={"help": "the delimiter between input and output"},
    )
    K: Optional[int] = field(
        default=1,
        metadata={"help": "the number of responses per prompt"},
    )

accelerator = Accelerator()

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

device = accelerator.device
reward_model = script_args.reward_name_or_path

ds_dir = script_args.dataset_name_or_path
world_size = 8
ds = load_dataset("json", data_files=ds_dir)

local_rank = Accelerator().local_process_index

data_size = len(ds["prompt"])

share = int(data_size / world_size) + 1
ds = ds.select(np.arange(local_rank * share, min((local_rank + 1) * share, len(ds))))

rm = AutoModelForSequenceClassification.from_pretrained(reward_model, device_map=device, torch_dtype=torch.bfloat16)

rm_tokenizer = AutoTokenizer.from_pretrained(reward_model)

def get_reward(sample):
    rewards = []
    rewards1 = []

    for i in range(len(sample["generated"])):
        message1 = sample["prompt"]
        message1 = rm_tokenizer.apply_chat_template(message1, tokenize=False)
        message2 = [{"role": "assistant", "content": sample["generated"][i]}]
        message2 = rm_tokenizer.apply_chat_template(message2, tokenize=False)
        message_tokenized = rm_tokenizer(message1+message2, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output = rm(**message_tokenized)
            multi_obj_rewards = output.rewards.cpu().float()
            tmp_reward = multi_obj_rewards.mean().item()
        rewards.append(tmp_reward)
    
    for i in range(len(sample["responses"])):
        message1 = sample["prompt"]
        message1 = rm_tokenizer.apply_chat_template(message1, tokenize=False)
        message2 = [{"role": "assistant", "content": sample["responses"][i]}]
        message2 = rm_tokenizer.apply_chat_template(message2, tokenize=False)
        message_tokenized = rm_tokenizer(message1+message2, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output = rm(**message_tokenized)
            multi_obj_rewards = output.rewards.cpu().float()
            tmp_reward = multi_obj_rewards.mean().item()
        rewards1.append(tmp_reward)

    return rewards, rewards1

def get_bleu(sample, rewards, rewards1):
    bleu_list = []

    max_reward_index = rewards.index(max(rewards))
    hyp = sample['generated'][max_reward_index]

    max_reward_index = rewards1.index(max(rewards1))
    ref = sample['responses'][max_reward_index]

    hyp = hyp.strip()
    ref = ref.strip()
    bleu_score = nltk.translate.bleu_score.sentence_bleu([ref], hyp)
    bleu_list.append(bleu_score)
    return bleu_list

data = []

# tqdm is used to show the progress bar
with torch.no_grad():
    for sample in tqdm(ds):
        # The VLLM may not generate responses for some prompts because it is too long, we skip them
        if len(sample["generated"]) < script_args.K:
            continue
        
        rewards, rewards1 = get_reward(sample)
        bleu_list = get_bleu(sample, rewards, rewards1)
        data.append({"prompt": sample["prompt"], "generated": sample["generated"], "responses": sample["responses"], "rewards": rewards, "rewards1": rewards1, "bleu": bleu_list})

world_size = 8
all_process_list = [{}] * world_size

data_to_send = {
    "data": [[data[i]] for i in range(len(data))],
}

dist.all_gather_object(all_process_list, data_to_send)
gathered_data = []

for i in range(world_size):
    tmp_data = [tmp[0] for tmp in all_process_list[i]["data"]]
    gathered_data.extend(tmp_data)

# rewards
all_rewards = [sample["rewards"] for sample in gathered_data]
top1_scores = np.mean(np.max(all_rewards, axis=1))
mean_scores = np.mean(all_rewards)

# bleu
all_rewards_b = [sample["bleu"] for sample in gathered_data]
top1_scores_b = np.mean(np.max(all_rewards_b, axis=1))
mean_scores_b = np.mean(all_rewards_b)

if local_rank == 0:
    print(
        "Collect {} data from {} inputs. mean score {} top1 score: {}".format(
            len(gathered_data), data_size, mean_scores, top1_scores
        )
    )
    print(
        "Collect {} data from {} inputs. bleu mean score {} bleu top1 score: {}".format(
            len(gathered_data), data_size, mean_scores_b, top1_scores_b
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