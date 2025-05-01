#!/usr/bin/env python
"""IMDb sentiment fine‑tune with *NdLinear (1‑D)* replacement – random‑init.

This script is a minimal variant of the original `run_NDlinear_replace_finetune.py`
that swaps every selected `nn.Linear` in the MLPs with our **NdLinear** layer but
keeps **1‑D shapes** so no extra reshaping is needed.  All new layers are randomly
(initialised with Xavier) – **no weight copy or tensor factorisation**.

The goal is simply to verify the NdLinear implementation trains & runs end‑to‑end.

Usage example
-------------

```bash
python run_NDlinear_replace_finetune_new.py \
    --use_ndlinear \
    --replacement_mode last_n --last_n 4 \
    --epochs 1
```
"""
from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
from typing import List

import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

# ‑‑ NdLinear import ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑
try:
    from nd import NdLinear  # path: nd.py in project root
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Cannot import NdLinear – make sure nd.py is on PYTHONPATH"
    ) from e

# ============================================================
#  Argument parsing
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generative baseline + fine‑tune Qwen2.5‑0.5B on IMDb with optional NdLinear replacement"
        )
    )
    # ‑ basic
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--dataset_name", type=str, default="stanfordnlp/imdb")
    p.add_argument("--output_dir", type=str, default="/root/autodl-tmp/output‑result")
    p.add_argument("--exp_name", type=str, default="qwen‑imdb‑nd1d")

    # ‑ training
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=0.00002)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--max_gen_length", type=int, default=10)
    p.add_argument("--do_train", action="store_true", default=True)

    # ‑ NdLinear replace opts
    p.add_argument("--use_ndlinear", action="store_true", default=True)
    p.add_argument(
        "--replacement_mode",
        choices=["all", "last_n", "even", "reinit_last_n"],
        default="last_n",
    )
    p.add_argument("--last_n", type=int, default=1)
    return p.parse_args()

# ============================================================
#  Replacement helpers (1‑D NdLinear, *no* weight copy)
# ============================================================

def _find_transformer_layers(model) -> List[nn.Module]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers  # Qwen/Llama style
    if hasattr(model, "base_model") and hasattr(model.base_model, "layers"):
        return model.base_model.layers
    raise RuntimeError("Cannot locate transformer layers – adapt for your model.")


def _swap_linear(lin: nn.Linear) -> NdLinear:
    """Return an NdLinear with **1‑D dims** (d_in→d_out), Xavier‑initialised."""
    return NdLinear(
        input_dims=lin.in_features,
        hidden_size=lin.out_features,
        bias=lin.bias is not None,
    )


def replace_all_mlp(model):
    for name, module in list(model.named_modules()):
        if ".mlp." in name and isinstance(module, nn.Linear):
            parent = model
            *path, attr = name.split(".")
            for p in path:
                parent = getattr(parent, p)
            setattr(parent, attr, _swap_linear(module))


def replace_last_n_mlp(model, n: int):
    layers = _find_transformer_layers(model)
    for block in layers[-n:]:
        for name, lin in block.mlp.named_children():
            if isinstance(lin, nn.Linear):
                setattr(block.mlp, name, _swap_linear(lin))


def replace_even_mlp(model):
    for i, block in enumerate(_find_transformer_layers(model)):
        if i % 2 == 0:
            for name, lin in block.mlp.named_children():
                if isinstance(lin, nn.Linear):
                    setattr(block.mlp, name, _swap_linear(lin))


def reinit_last_n_mlp(model, n: int, mean: float = 0.0, std: float = 0.02):
    """(Optional) Reinit last‑n *existing* Linear layers (if no swap)."""
    for block in _find_transformer_layers(model)[n:]:
        for lin in block.mlp.modules():
            if isinstance(lin, nn.Linear):
                nn.init.normal_(lin.weight, mean, std)
                if lin.bias is not None:
                    nn.init.zeros_(lin.bias)

# ============================================================
#  Data pre‑/post‑processing helpers (same as original)
# ============================================================

def preprocess_function(examples, tokenizer, max_seq_length):
    texts = examples["text"]
    labels = examples["label"]
    prompts = [
        f"Review: {t}\nOutput the sentiment of the review as a single word, either positive or negative."
        for t in texts
    ]
    responses = ["positive" if lab == 1 else "negative" for lab in labels]

    batch = tokenizer(
        prompts,
        responses,
        max_length=max_seq_length,
        padding="max_length",
        truncation="only_first",
        return_tensors="pt",
    )
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels_tensor = input_ids.clone()
    for i in range(input_ids.size(0)):
        for j, s in enumerate(batch.sequence_ids(i)):
            if s != 1:  # mask prompt tokens
                labels_tensor[i, j] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels_tensor,
    }


def baseline_generation(model, tokenizer, dataset, device, max_gen_length):
    model.eval()
    correct = total = 0
    for ex in tqdm(dataset, desc="Inference", leave=True):
        text, gold = ex["text"], ex["label"]
        prompt = (
            f"Review: {text}\nOutput the sentiment of the review as a single word, either positive or negative."
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_gen_length,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_tokens = out_ids[0, inputs["input_ids"].shape[-1] :]
        pred_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip().lower()
        pred = 1 if "positive" in pred_text else 0
        correct += int(pred == gold)
        total += 1
    print(f"\nAccuracy on {total} examples: {correct / total:.4f}")

# ============================================================
#  Main routine
# ============================================================

def main() -> None:  # noqa: D401
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("nd1d")

    # ‑ output dir
    exp_name = args.exp_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    # ‑ device & model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, cache_dir=os.environ.get("HF_HOME"), trust_remote_code=True
    ).to(device)

    # ‑‑ NdLinear replace ‑‑
    if args.use_ndlinear:
        if args.replacement_mode == "all":
            replace_all_mlp(model)
        elif args.replacement_mode == "last_n":
            replace_last_n_mlp(model, args.last_n)
        elif args.replacement_mode == "even":
            replace_even_mlp(model)
    if args.replacement_mode == "reinit_last_n":
        reinit_last_n_mlp(model, args.last_n)

    # ‑ dataset
    raw_dsets = load_dataset(args.dataset_name)
    split = raw_dsets["train"].train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"].select(range(2500))
    eval_ds = split["test"]
    test_ds = raw_dsets["test"].select(range(25))

    logger.info(">>> Tokenising …")
    tokenized_train = train_ds.map(
        lambda ex: preprocess_function(ex, tokenizer, args.max_seq_length),
        batched=True,
        remove_columns=train_ds.column_names,
    )
    tokenized_eval = eval_ds.map(
        lambda ex: preprocess_function(ex, tokenizer, args.max_seq_length),
        batched=True,
        remove_columns=eval_ds.column_names,
    )

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
        logging_steps=50,
        fp16=True,
        gradient_checkpointing=True,
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    if args.do_train:
        logger.info(">>> Fine‑tuning …")
        trainer.train()
        logger.info(">>> Saving model …")
        trainer.save_model(save_dir)
        tokenizer.save_pretrained(save_dir)

    logger.info(">>> Evaluating …")
    baseline_generation(model, tokenizer, test_ds, device, args.max_gen_length)


if __name__ == "__main__":
    main()
