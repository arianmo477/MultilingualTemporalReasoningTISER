#!/usr/bin/env python3
"""
QLoRA Training Script (RTX 4090 Optimized)
"""

import argparse
import os
import sys
from typing import Any, List, Dict
from dataclasses import dataclass

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from utils.io_gpu import balance_by_dataset_name

# -------------------------
# OPTIMIZATION FLAGS
# -------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@dataclass
class DataCollatorForCausalLMWithPadding:
    tokenizer: Any

    def __call__(self, features):
        labels = [f["labels"] for f in features]
        features_no_labels = []
        for f in features:
            # Create a copy to avoid modifying original dictionary
            f_copy = {k: v for k, v in f.items() if k != "labels"}
            features_no_labels.append(f_copy)

        batch = self.tokenizer.pad(features_no_labels, return_tensors="pt")
        
        # Determine max length in this batch
        max_len = batch["input_ids"].shape[1]
        padded_labels = []

        for lbl in labels:
            # Pad labels with -100 (ignore index)
            padded_labels.append(lbl + [-100] * (max_len - len(lbl)))

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def find_lora_targets(model) -> List[str]:
    names = {n.split(".")[-1] for n, _ in model.named_modules()}
    
    preferred = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    return [m for m in preferred if m in names]


def main():
    parser = argparse.ArgumentParser()

    # Model / Data
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--only_passed", action="store_true")

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--per_device_batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    
    # LoRA
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Optimization
    parser.add_argument("--gradient_checkpointing", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)

    args = parser.parse_args()

    # -------------------------
    # 1. Tokenizer
    # -------------------------
    print(f"Loading tokenizer for: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    # Trainer expects right padding usually for training, unlike inference
    tokenizer.padding_side = "right" 

    # -------------------------
    # 2. Dataset Loading & Balancing
    # -------------------------
    print(f"Loading dataset from: {args.train_file}")
    dataset = load_dataset("json", data_files=args.train_file)["train"]

    if args.only_passed and "validation_status" in dataset.column_names:
        dataset = dataset.filter(lambda x: x["validation_status"] == "PASS")
        print(f"Filtered to PASS samples: {len(dataset)}")

    if args.max_train_samples:
        print(f"Balancing dataset to max {args.max_train_samples} samples...")
        dataset = balance_by_dataset_name(dataset.shuffle(seed=42), max_samples=args.max_train_samples)

    # -------------------------
    # 3. Preprocessing
    # -------------------------
    def preprocess(ex):
        # Construct the prompt exactly as used in inference
        instr = (ex.get("prompt", "") or "").strip()
        qst = (ex.get("question", "") or "").strip()
        ctx = (ex.get("temporal_context", "") or ex.get("context", "") or "").strip()
        
        full_prompt = (
            f"{instr}\n\n"
            f"Context:\n{ctx}\n\n"
            f"Question:\n{qst}\n\n"
            f"<reasoning>"
        )
        
        # The expected completion
        # We assume the output in the dataset is the answer. 
        # Ideally, your training data 'output' field contains the full chain of thought + answer.
        # If your 'output' only has the answer, the model will learn to jump straight to answer.
        output = (ex.get("output", "") or ex.get("answer", "") or "").strip()

        # Tokenize separately to manage lengths
        prompt_tokens = tokenizer(full_prompt, add_special_tokens=False)["input_ids"]
        output_tokens = tokenizer(output + tokenizer.eos_token, add_special_tokens=False)["input_ids"]

        # Concatenate
        input_ids = prompt_tokens + output_tokens
        
        # Create Labels: -100 for prompt (masked), actual IDs for output
        labels = [-100] * len(prompt_tokens) + output_tokens
        
        # Truncate if necessary
        if len(input_ids) > args.max_length:
            input_ids = input_ids[:args.max_length]
            labels = labels[:args.max_length]

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    # === CRITICAL FIX IS HERE ===
    # Convert list back to Hugging Face Dataset if necessary
    if isinstance(dataset, list):
        print("Converting list dataset back to Hugging Face Dataset object...")
        dataset = Dataset.from_list(dataset)
    # ============================

    print("Tokenizing dataset...")
    # remove_columns prevents the old text columns from being passed to the trainer
    dataset = dataset.map(preprocess, remove_columns=dataset.column_names, desc="Tokenizing")

    # -------------------------
    # 4. Model (4-bit QLoRA)
    # -------------------------
    print("Loading Model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
    )
    
    # Enable gradient checkpointing and prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False 

    # -------------------------
    # 5. LoRA Config
    # -------------------------
    targets = find_lora_targets(model)
    print(f"Targeting modules for LoRA: {targets}")
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # -------------------------
    # 6. Training
    # -------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        bf16=True,     # RTX 4090 supports BF16
        fp16=False,
        optim="paged_adamw_8bit", # Saves memory
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        gradient_checkpointing=(args.gradient_checkpointing == 1),
        dataloader_num_workers=args.dataloader_num_workers,
        report_to="none",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForCausalLMWithPadding(tokenizer),
    )

    print("Starting Training...")
    trainer.train()

    print(f"Saving model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training complete.")

if __name__ == "__main__":
    main()