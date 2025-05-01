#!/usr/bin/env python
# run_imdb_qwen_finetune.py
"""
Two-stage IMDb sentiment classification with Qwen2.5-0.5B:
1) Baseline generative inference without training
2) Standard prompt-based fine-tuning, then re-evaluate
"""

import os
import argparse
import logging

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from tqdm.auto import tqdm
import copy

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generative baseline + fine-tune Qwen2.5-0.5B on stanfordnlp/imdb"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="stanfordnlp/imdb",
        help="HF dataset identifier for IMDb",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-tmp/output-of-result",
        help="Where to store checkpoints and logs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Per-device training batch size",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=4,
        help="Per-device evaluation batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Accumulate gradients this many steps",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Max token length for prompt + answer",
    )
    parser.add_argument(
        "--max_gen_length",
        type=int,
        default=10,
        help="Max new tokens to generate during inference",
    )
    parser.add_argument(
        "--do_train",
        default=True,
        action="store_true",
        help="If set, run fine-tuning after baseline",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="qwen-imdb-finetune-mlp",
        help="Experiment name: will be used to name the output folder. Defaults to timestamp."
    )
    return parser.parse_args()


def preprocess_function(examples, tokenizer, max_seq_length):
    texts  = examples["text"]
    labels = examples["label"]
    
    prompts   = [f"Review: {t}\n. Output the sentiment the of the review as a single word, either positive or negative." for t in texts]
    responses = ["positive" if lab == 1 else "negative" for lab in labels]

    batch = tokenizer(
        prompts,
        responses,
        max_length=max_seq_length,
        padding="max_length",
        truncation="only_first",      
        return_tensors="pt",
    )
    input_ids     = batch["input_ids"]
    attention_mask= batch["attention_mask"]

    labels = input_ids.clone()
    for i in range(input_ids.size(0)):
        seq_ids = batch.sequence_ids(i)   
        for j, s in enumerate(seq_ids):
            if s != 1:                   
                labels[i, j] = -100

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
    }


def baseline_generation(model, tokenizer, dataset, device, max_gen_length):
    """
    Loop over dataset with tqdm, prompt the model,
    generate up to max_gen_length tokens, then map to {0,1}.
    """
    model.eval()
    correct = 0
    total = 0

    # Wrap dataset in tqdm for a progress bar
    for ex in tqdm(dataset, desc="Baseline inference", leave=True):
        text = ex["text"]
        gold = ex["label"]
        prompt = f"Review: {text}\n. Output the sentiment the of the review as a single word, either positive or negative."
        
        # Tokenize prompt
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding=True,
        ).to(device)

        # Generate classification string
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_gen_length,
            pad_token_id=tokenizer.eos_token_id,
        )
        # Only decode the new tokens
        gen_tokens = out_ids[0, inputs["input_ids"].shape[-1] :]
        pred_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip().lower()

        # Simple rule: if "positive" appears → 1 else → 0
        pred = 1 if "positive" in pred_text else 0

        correct += int(pred == gold)
        total += 1

    acc = correct / total
    print(f"\nBaseline generation accuracy on {total} examples: {acc:.4f}")
    return acc

def main():
    args = parse_args()
    #os.makedirs(args.output_dir, exist_ok=True)

    from datetime import datetime
    exp_name = args.exp_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    # Logging setup
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    # Device and model/tokenizer loading
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(">>> Tokenizer vocab size:", tokenizer.vocab_size)
    print(">>> Tokenizer pad token ID:", tokenizer.pad_token_id)
    print(">>> Tokenizer EOS token ID:", tokenizer.eos_token_id)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        cache_dir=os.environ.get("HF_HOME"),
        trust_remote_code=True
    ).to(device)

    # Load IMDb
    raw_dsets = load_dataset(args.dataset_name)
    # ⬅️ 新增：在 train 上拆出一小部分做验证集
    split = raw_dsets["train"].train_test_split(test_size=0.1, seed=42)  # 10% 作 eval
    train_ds = split["train"].shuffle(seed=42).select(range(2500))                       # 训练集取前2500
    eval_ds  = split["test"]                                            # 验证集（约250条）
    test_ds  = raw_dsets["test"].shuffle(seed=42).select(range(2500))                    # 最终测试集，留给最最后跑baseline

    # # 1) Baseline without any fine-tuning
    # logger.info(">>> Running baseline generation on test set")
    # baseline_generation(
    #     model, tokenizer, test_ds, device, args.max_gen_length
    # )

    # 2) Optional fine-tuning
    if args.do_train:
        logger.info(">>> Preprocessing train & test splits")
        tokenized_train = train_ds.map(
            lambda ex: preprocess_function(ex, tokenizer, args.max_seq_length),
            batched=True,
            remove_columns=train_ds.column_names,
        )
        # ⬅️ 用 eval_ds 生成验证集张量
        tokenized_eval = eval_ds.map(
            lambda ex: preprocess_function(ex, tokenizer, args.max_seq_length),
            batched=True,
            remove_columns=eval_ds.column_names,
        )

        # TrainingArguments for causal-LM fine-tuning
        training_args = TrainingArguments(
            output_dir=save_dir,
            overwrite_output_dir=True,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=100,
            fp16=True, 
            report_to=[],
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,   # ⬅️ 这里替换成验证集
            tokenizer=tokenizer,
            data_collator=default_data_collator,
        )

        # Fine-tune
        logger.info(">>> Starting fine-tuning")
        trainer.train()

        # Save final model & tokenizer
        logger.info(f">>> Saving final model & tokenizer to {save_dir}")
        trainer.save_model(save_dir)
        tokenizer.save_pretrained(save_dir)

        # Re-evaluate after fine-tuning
        logger.info(">>> Re-running generation on test set after fine-tuning")
        baseline_generation(
            model, tokenizer, test_ds, device, args.max_gen_length
        )
        
        
if __name__ == "__main__":
    main()
