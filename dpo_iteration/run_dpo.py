import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from alignment import H4ArgumentParser
from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
)

from trl.commands.cli_utils import TrlParser
from peft import LoraConfig, get_peft_model

@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    ref_model: Optional[str] = field(
        default="",
        metadata={"help": "the location of the SFT model name or path"},
    )
    
    train_dir: Optional[str] = field(
        default="",
        metadata={"help": "the location of the dataset name or path"},
    )

    eval_dir: Optional[str] = field(
        default="",
        metadata={"help": "the location of the evalset name or path"},
    )

    eos_padding: Optional[bool] = field(default=True, metadata={"help": "whether to pad with eos token"})

    margin_scale: Optional[float] = field(default=1.0, metadata={"help": "the margin scale"})

    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})

    max_training_samples: Optional[int] = field(default=-1, metadata={"help": "the maximum sample size"})

    choose_type: Optional[str] = field(default="max_min", metadata={"help": "the choose type"})

    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers."
        },
    )
    eot_token: Optional[str] = field(default="", metadata={"help": "the end of text token"})
    len_penalty: Optional[float] = field(default=0, metadata={"help": "the length penalty"})

epoch = 2

# gamma, beta
gamma = -5
beta = 0.25

initial_gamma = -5
final_gamma = 5
gamma_range = final_gamma - initial_gamma
gamma_i = initial_gamma
ds_num_sample = 100

def weighted_uniform_sample(sample_rewards, gamma, num_samples=1):
    origin = np.array(sample_rewards)
    sample_rewards = np.exp(sample_rewards)
    max_reward = np.max(sample_rewards)
    mask = sample_rewards != max_reward
    filtered_indices = np.arange(len(sample_rewards))[mask]  # Get indices of remaining rewards

    if len(filtered_indices) == 0:
        return 0
    else:
        weights = np.exp(origin[mask]*gamma)
        probabilities = weights / weights.sum()
        sampled_indices = np.random.choice(filtered_indices, size=num_samples, p=probabilities)
        return sampled_indices.item()

def prepare_data(
    data_dir: str = "",
    sanity_check: bool = False,
    cache_dir: str = None,
    num_proc=24,
    margin_scale=1,
    choose_type="random",
    eot_token="",
    length_penalty=0,
) -> Dataset:
    """Prepare the dataset for DPO training by rejection sampling.
    """
    ds = load_dataset("json", data_files=data_dir, split="train")
    ds = ds.shuffle(seed=42)
    ds = ds.map(lambda x: {"responses": x["responses"][:ds_num_sample], "rewards": x["rewards"][:ds_num_sample]})

    if epoch == 2:
        ds = concatenate_datasets([ds, ds])
        ds = ds.shuffle(seed=42)
        num_samples = len(ds)
    elif epoch == 1:
        ds = ds.shuffle(seed=42)
        num_samples = len(ds)

    update_points = [int(num_samples * frac) for frac in [0.2, 0.4, 0.6, 0.8]]

    pos = []
    neg = []
    prompts = []

    margin = []
    for i, sample in enumerate(ds):
        # P = tokenizer.apply_chat_template(sample["prompt"], tokenize = False, add_generation_prompt= True)
        P = sample["prompt"]
        if choose_type == "random":
            idx0 = 0
            idx1 = 1
        elif choose_type == "max_random":
            idx0 = np.argmax(sample["rewards"])
            if idx0 == 0:
                idx1 = 1
            else:
                idx1 = 0
        elif choose_type == "max_min":
            idx0 = np.argmax(sample["rewards"])
            idx1 = np.argmin(sample["rewards"])
        elif choose_type == "max_max":
            sorted_indices = np.argsort(sample["rewards"])
            idx0 = sorted_indices[-1]
            idx1 = sorted_indices[-2]
        elif choose_type == "max_min_p":
            r = [
                sample["rewards"][i] - length_penalty * len(sample["responses"][i])
                for i in range(len(sample["rewards"]))
            ]
            idx0 = np.argmax(r)
            idx1 = np.argmin(r)
        elif choose_type == "hard":
            idx0 = np.argmax(sample["rewards"])
            idx1 = weighted_uniform_sample(sample["rewards"], gamma)
        elif choose_type == "hard_i":
            if i in update_points:
                progress = i / (num_samples - 1)
                gamma_i = initial_gamma + (gamma_range * progress)
            idx0 = np.argmax(sample["rewards"])
            idx1 = weighted_uniform_sample(sample["rewards"], gamma_i)
        else:
            raise NotImplementedError

        if type(idx0) == np.ndarray or type(idx0) == list:
            assert len(idx0) == len(idx1)
            for i in range(len(idx0)):
                prompts.append(P)
                pos.append(sample["responses"][idx0[i]] + eot_token)
                neg.append(sample["responses"][idx1[i]] + eot_token)
                margin.append((sample["rewards"][idx0[i]] - sample["rewards"][idx1[i]]) * margin_scale)
        else:
            if sample["rewards"][idx0] > sample["rewards"][idx1]:
                prompts.append(P)
                pos.append(sample["responses"][idx0] + eot_token)
                neg.append(sample["responses"][idx1] + eot_token)
                margin.append((sample["rewards"][idx0] - sample["rewards"][idx1]) * margin_scale)
            elif sample["rewards"][idx0] < sample["rewards"][idx1]:
                prompts.append(P)
                pos.append(sample["responses"][idx1] + eot_token)
                neg.append(sample["responses"][idx0] + eot_token)
                margin.append((-sample["rewards"][idx0] + sample["rewards"][idx1]) * margin_scale)
    dataset = Dataset.from_dict({"prompt": prompts, "chosen": pos, "rejected": neg, "margin": margin})

    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 100)))

    return dataset


if __name__ == "__main__":

    
    parser = H4ArgumentParser((ScriptArguments, DPOConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse()

    # 1. Load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16,
    )
    
    use_l = False

    if use_l:
        # Define LoRA configuration
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.01,
            bias="none",
            task_type="CAUSAL_LM"
        )
        # Apply LoRA to the model
        model = get_peft_model(model, lora_config)

    model.config.use_cache = False

    if script_args.ignore_bias_buffers:
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    if script_args.ref_model:
        ref_name = script_args.ref_model
    else:
        ref_name = model_config.model_name_or_path

    model_ref = AutoModelForCausalLM.from_pretrained(
        ref_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    if script_args.eos_padding:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.config.vocab_size += 1
        model_ref.config.vocab_size += 1
        model.config.pad_token_id = tokenizer.pad_token_id
        model_ref.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))
        model_ref.resize_token_embeddings(len(tokenizer))


    # 2. Load the Stack-exchange paired dataset
    train_dataset = prepare_data(
        data_dir=script_args.train_dir,
        margin_scale=script_args.margin_scale,
        sanity_check=script_args.sanity_check,
        choose_type=script_args.choose_type,
        eot_token=script_args.eot_token,
        length_penalty=script_args.len_penalty,
    )

    if script_args.max_training_samples > 0:
        train_dataset = train_dataset.select(range(script_args.max_training_samples))

    # 3. Load evaluation dataset
    eval_dataset = prepare_data(
        data_dir=script_args.eval_dir,
        sanity_check=True,
        margin_scale=script_args.margin_scale,
        eot_token=script_args.eot_token,
    )

    print(training_args)

    # 4. Initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        beta=beta,
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
        loss_type=training_args.loss_type,
    )
    print("begin to train")

    # 5. Train
    dpo_trainer.train()

    # 6. Save
    output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)