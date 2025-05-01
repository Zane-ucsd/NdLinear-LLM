#!/usr/bin/env python
# run_imdb_qwen_infer.py
import os
import argparse
import logging
from datetime import datetime

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from tqdm.auto import tqdm


# ---------- CLI ----------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference on IMDb with a (fine-tuned) Qwen-style causal-LM"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HF hub model ID *or* local checkpoint directory",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="stanfordnlp/imdb",
        help="HuggingFace dataset identifier",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-tmp/output-of-result/qwen-imdb-finetune-mlp/checkpoint-78",
        help="Where to write simple accuracy log",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Prompt truncation length",
    )
    parser.add_argument(
        "--max_gen_length",
        type=int,
        default=10,
        help="Max new tokens to generate",
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=25,
        help="Use only first N examples of test split for speed",
    )
    return parser.parse_args()


# ---------- Core inference loop ----------
@torch.no_grad()
def run_inference(model, tokenizer, dataset, device, max_gen_len):
    model.eval()
    correct = total = 0

    for ex in tqdm(dataset, desc="Inference"):
        prompt = (
            f"Review: {ex['text']}\n"
            f". Output the sentiment the of the review as a single word, either positive or negative."
        )
        inputs = tokenizer(
            prompt,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)

        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_gen_len,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_tokens = out_ids[0, inputs["input_ids"].shape[-1] :]
        pred_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip().lower()

        pred = 1 if "positive" in pred_text else 0
        correct += int(pred == ex["label"])
        total += 1

    return correct / total


# ---------- Main ----------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    # device / model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading model from {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    ).to(device)

    # dataset
    logger.info("Loading IMDb test split")
    raw_dsets = load_dataset(args.dataset_name)
    test_ds = raw_dsets["test"].shuffle(seed=42).select(range(args.subset_size))
    print(test_ds['label'])

    # run
    logger.info("Running inference …")
    acc = run_inference(
        model, tokenizer, test_ds, device, args.max_gen_length
    )
    logger.info(f"Accuracy on {len(test_ds)} examples: {acc:.4%}")

    # simple text log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(args.output_dir, f"infer_{timestamp}.txt"), "w") as f:
        f.write(f"model = {args.model_name_or_path}\n")
        f.write(f"accuracy = {acc:.4%}\n")
    logger.info(f"Saved log to {args.output_dir}")

if __name__ == "__main__":
    main()
