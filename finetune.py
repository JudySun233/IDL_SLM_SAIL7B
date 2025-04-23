# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import sys
import json
import pathlib
from dataclasses import dataclass, field
from typing import Optional, Sequence, Dict

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers import (
    Trainer,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
    DataCollatorForLanguageModeling
)
from transformers.trainer_pt_utils import LabelSmoother
from fastchat.conversation import get_conv_template, SeparatorStyle

import wandb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="luohy/SAIL-7b",
        metadata={"help": "Path to pretrained model or model identifier"}
    )

@dataclass
class DataArguments:
    data_path: str = field(
        default=None,
        metadata={"help": "Path to the JSON training data file"}
    )
    lazy_preprocess: bool = field(
        default=False,
        metadata={"help": "Whether to preprocess the dataset lazily"}
    )

@dataclass
class LoraArguments:
    lora_r: int = field(default=16, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.1, metadata={"help": "LoRA dropout"})
    target_modules: Sequence[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"],
        metadata={"help": "Modules to apply LoRA to"}
    )
    qlora: bool = field(
        default=False,
        metadata={"help": "Whether to use 4-bit QLoRA quantization"}
    )

@dataclass
class WandbArguments:
    project_name: str = field(
        default="", metadata={"help": "Weights & Biases project name"}
    )
    run_name: str = field(
        default="", metadata={"help": "Weights & Biases run name"}
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(
        default=1600,
        metadata={"help": "Maximum sequence length"}
    )
    report_to: Optional[str] = field(
        default="wandb",
        metadata={"help": "The integration to report to ('wandb', 'tensorboard', etc.)"}
    )
    logging_dir: Optional[str] = field(
        default="./logs",
        metadata={"help": "Directory for tensorboard logs"}
    )


def rank0_print(*args):
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    if rank == 0:
        print(*args)


def preprocess(sources, tokenizer: transformers.PreTrainedTokenizer) -> Dict[str, torch.Tensor]:
    # Build your prompts exactly as before...
    conv = get_conv_template("dolly_v2").copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    prompts = []
    for source in sources:
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]
        conv.messages = []
        for sentence in source:
            role = roles[sentence["from"]]
            conv.append_message(role, sentence["value"])
        prompts.append(conv.get_prompt())

    # Tokenize, pad/truncate to max_length
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
    )

    return {
        "input_ids":      encoded.input_ids,
        "attention_mask": encoded.attention_mask,
        # no labels here!
    }

   
class SupervisedDataset(Dataset):
    def __init__(self, raw_data, tokenizer, lazy=False):
        self.raw = raw_data
        self.tokenizer = tokenizer
        self.lazy = lazy
        if not lazy:
            data = preprocess([ex["conversations"] for ex in raw_data], tokenizer)
            self.input_ids = data["input_ids"]
            self.attn_mask = data["attention_mask"]


    def __len__(self):
        return len(self.raw)

    def __getitem__(self, idx):
        if self.lazy:
            data = preprocess([self.raw[idx]["conversations"]], self.tokenizer)
            return data
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attn_mask[idx],
        }

def make_data_module(args, tokenizer):
    raw = json.load(open(args.data_path, "r"))
    perm = np.random.default_rng(42).permutation(len(raw))
    split = int(len(raw) * 0.8)
    train_raw = [raw[i] for i in perm[:split]]
    val_raw = [raw[i] for i in perm[split:]]
    rank0_print(f"#train {len(train_raw)}, #val {len(val_raw)}")
    return (
        SupervisedDataset(train_raw, tokenizer, lazy=args.lazy_preprocess),
        SupervisedDataset(val_raw, tokenizer, lazy=args.lazy_preprocess)
    )

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, LoraArguments,TrainingArguments)
    )
    model_args, data_args, lora_args, training_args = \
        parser.parse_args_into_dataclasses()

    # seed
    set_seed(42)

    # init wandb
    wandb.init(project="llama2-lora-entail",
               name=training_args.run_name or None)


    TRUST = {"trust_remote_code": True}

    if lora_args.qlora:
        bnb_conf = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            quantization_config=bnb_conf,
            device_map="auto",
            cache_dir=training_args.cache_dir,
            **TRUST
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            device_map="auto",
            torch_dtype=torch.float16,
            cache_dir=training_args.cache_dir,
            **TRUST
        )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        padding_side="right",
        use_fast=False,
        **TRUST
    )
    tokenizer.pad_token = tokenizer.eos_token

    data_collator = DataCollatorForLanguageModeling(
        tokenizer,
        mlm=False,               # causal LM (not masked LM)
        pad_to_multiple_of=None,
        return_tensors="pt",
    )

    

    # LoRA wrap
    lora_cfg = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=list(lora_args.target_modules),
        lora_dropout=lora_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    train_dataset, eval_dataset = make_data_module(data_args, tokenizer)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )

    trainer.train()
    model.save_pretrained(training_args.output_dir)
    print("Finished fine-tuning with LoRA/QLoRA and wandb tracking!")

if __name__ == "__main__":
    train()
