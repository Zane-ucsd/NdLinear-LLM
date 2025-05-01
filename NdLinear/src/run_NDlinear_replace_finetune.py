#!/usr/bin/env python
# run_imdb_qwen_finetune_ndlinear.py
"""
IMDb 情感分类两阶段实验脚本（基线 + 后训练），并提供可选的 NdLinear 替换 / 重新初始化功能。

使用示例：

```bash
python run_imdb_qwen_finetune_ndlinear.py \
    --use_ndlinear \
    --replacement_mode last_n --last_n 4 \
    --epochs 1
```

参数说明请查看 `parse_args()`。
"""

import os
import argparse
import logging
from datetime import datetime
from typing import List

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from tqdm.auto import tqdm

# ==========================================================
#  NdLinear 依赖
# ==========================================================
try:
    from ndlinear import NdLinear  # noqa: F401
except ImportError as e:
    raise ImportError("请确保 ndlinear 包在 PYTHONPATH 中：pip install -e ./ndlinear 或相应路径") from e


# ==========================================================
#  参数解析
# ==========================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generative baseline + fine‑tune Qwen2.5‑0.5B on IMDb with optional NdLinear replacement"
        )
    )
    # 基础参数
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--dataset_name", type=str, default="stanfordnlp/imdb")
    p.add_argument("--output_dir", type=str, default="/root/autodl-tmp/output-of-result")
    p.add_argument("--exp_name", type=str, default="qwen-imdb-finetune-Ndlinear-1")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--max_gen_length", type=int, default=10)
    p.add_argument("--do_train", default=True, action="store_true")
    # NdLinear & 替换策略
    p.add_argument("--use_ndlinear", default=True, action="store_true", help="是否启用 NdLinear 替换")
    p.add_argument(
        "--replacement_mode",
        choices=["all", "last_n", "even", "reinit_last_n"],
        default="last_n",
        help="NdLinear 替换 / 重新初始化模式",
    )
    p.add_argument("--last_n", type=int, default=1, help="last_n 或 reinit_last_n 模式下的 N")
    return p.parse_args()


# ==========================================================
#  NdLinear 替换工具函数
# ==========================================================

def _find_transformer_layers(model) -> List[nn.Module]:
    """兼容 HuggingFace Qwen / GPT 等模型的层定位"""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers  # Qwen2 / LLaMA
    if hasattr(model, "base_model") and hasattr(model.base_model, "layers"):
        return model.base_model.layers  # 部分模型包装
    raise RuntimeError("无法定位到 Transformer 层，请手动检查模型结构")


def _swap_linear(lin: nn.Linear) -> NdLinear:
    nd = NdLinear((lin.in_features,), (lin.out_features,))
    nd.align_layers[0].weight.data.copy_(lin.weight.data)
    if lin.bias is not None:
        nd.align_layers[0].bias.data.copy_(lin.bias.data)
    return nd


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
    layers = _find_transformer_layers(model)
    for i, block in enumerate(layers):
        if i % 2 == 0:
            for name, lin in block.mlp.named_children():
                if isinstance(lin, nn.Linear):
                    setattr(block.mlp, name, _swap_linear(lin))


def reinit_last_n_mlp(model, n: int, mean: float = 0.0, std: float = 0.02):
    layers = _find_transformer_layers(model)
    for block in layers[-n:]:
        for lin in block.mlp.modules():
            if isinstance(lin, nn.Linear):
                nn.init.normal_(lin.weight, mean=mean, std=std)
                if lin.bias is not None:
                    nn.init.zeros_(lin.bias)


# ==========================================================
#  数据预处理 & 评估
# ==========================================================

def preprocess_function(examples, tokenizer, max_seq_length):
    texts = examples["text"]
    labels = examples["label"]
    prompts = [
        f"Review: {t}\nOutput the sentiment of the review as a single word, either positive or negative."  # noqa: E501
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
        seq_ids = batch.sequence_ids(i)
        for j, s in enumerate(seq_ids):
            if s != 1:
                labels_tensor[i, j] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels_tensor,
    }

def baseline_generation(model, tokenizer, dataset, device, max_gen_length):
    model.eval()
    correct = total = 0
    for ex in tqdm(dataset, desc="Baseline inference", leave=True):
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
    acc = correct / total
    print(f"\nAccuracy on {total} examples: {acc:.4f}")
    return acc


# ==========================================================
#  主流程
# ==========================================================

def main():
    args = parse_args()

    # 日志
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
    logger = logging.getLogger("imdb_ndlinear")

    # 输出目录
    exp_name = args.exp_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)

    # 设备 & 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        cache_dir=os.environ.get("HF_HOME"),
        trust_remote_code=True,
    ).to(device)

    # —— NdLinear 替换 ————————————
    if args.use_ndlinear:
        if args.replacement_mode == "all":
            replace_all_mlp(model)
        elif args.replacement_mode == "last_n":
            replace_last_n_mlp(model, args.last_n)
        elif args.replacement_mode == "even":
            replace_even_mlp(model)

    if args.replacement_mode == "reinit_last_n":
        reinit_last_n_mlp(model, args.last_n)

    # 数据集加载 & 切分
    raw_dsets = load_dataset(args.dataset_name)
    split = raw_dsets["train"].train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"].select(range(2500))
    eval_ds = split["test"]
    test_ds = raw_dsets["test"].select(range(25))

    # 预处理
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

    # 训练参数
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

    # 训练
    if args.do_train:
        logger.info(">>> Start fine‑tuning …")
        trainer.train()
        logger.info(f">>> Saving model to {save_dir}")
        trainer.save_model(save_dir)
        tokenizer.save_pretrained(save_dir)

    # 基线 / 微调后评估
    logger.info(">>> Evaluating on test set …")
    baseline_generation(model, tokenizer, test_ds, device, args.max_gen_length)


if __name__ == "__main__":
    main()
