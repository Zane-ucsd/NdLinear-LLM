#!/usr/bin/env python
# finetune_pubmedqa.py
import os
import math
import argparse
from functools import partial

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments,default_data_collator,DataCollatorWithPadding
)
from transformers.trainer_utils import EvalPrediction

from ndlinear import NdLinear  # 确保已 pip install ndlinear

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Qwen2.5-0.5B on PubMedQA")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--grad_acc", type=int, default=8)
    p.add_argument("--max_length", type=int, default=512,
                   help="max total tokens of prompt+answer")
    p.add_argument("--use_ndlinear", action="store_true",
                   help="whether to replace MLP layers with NdLinear")
    return p.parse_args()

def replace_mlp_with_ndlinear(model):
    """
    遍历 model 中所有名为 '*.mlp.*' 的 nn.Linear，并替换成等价的 NdLinear。
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and ".mlp." in name:
            parent = model
            *path, attr = name.split(".")
            for key in path:
                parent = getattr(parent, key)
            in_f, out_f = module.in_features, module.out_features
            ndl = NdLinear((in_f,), (out_f,))
            # 初始化成原始权重
            ndl.align_layers[0].weight.data.copy_(module.weight.data)
            if module.bias is not None:
                ndl.align_layers[0].bias.data.copy_(module.bias.data)
            setattr(parent, attr, ndl)

def preprocess(ex, tokenizer, max_length):
    """
    构造 prompt + label：
      prompt = "Context: {passage}\nQuestion: {question}\nAnswer:"
      label  = "{answer}"
    然后 tokenize，两者拼一起，并把 prompt 部分 labels 置 -100。
    """
    passage = ex["passage"]["text"] if "passage" in ex else ex["context"]
    question = ex["question"]
    answer = ex["answer"].lower()  # yes,no,maybe
    # 拼串
    prompt = f"Context: {passage}\nQuestion: {question}\nAnswer:"
    full = prompt + " " + answer + tokenizer.eos_token
    tokenized = tokenizer(
        full,
        truncation=True,
        padding=max_length,
        max_length=max_length,
        
    )
    # 构造 labels：prompt 部分 = -100
    prompt_tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
        
    )["input_ids"]
    labels = tokenized["input_ids"].copy()
    labels[:len(prompt_tokens)] = [-100] * len(prompt_tokens)
    tokenized["labels"] = labels
    return tokenized

def compute_metrics(p: EvalPrediction):
    """
    在验证集上测 Accuracy & Perplexity。
    """
    preds = p.predictions
    if isinstance(preds, tuple): preds = preds[0]
    # 我们用 greedy 生成时只需要取 argmax
    pred_ids = preds.argmax(axis=-1)
    # 把所有非 -100 标签位置计算 accuracy
    labels = p.label_ids
    mask = labels != -100
    correct = (pred_ids == labels) & mask
    acc = correct.sum() / mask.sum()
    # Perplexity from avg loss
    loss = p.metrics.get("eval_loss", None)
    ppl = math.exp(loss) if loss is not None else None
    return {"accuracy": acc, "perplexity": ppl}



def main():
    args = parse_args()
    # 设备 & Tokenizer & Model 加载
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    if args.use_ndlinear:
        replace_mlp_with_ndlinear(model)
    model = model.to(device)

    # 数据加载 & 预处理

    # 确认缓存路径环境变量
    hf_ds_cache = os.environ.get("HF_DATASETS_CACHE")
    if not hf_ds_cache:
        raise RuntimeError("请先 export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/datasets")
    raw = load_dataset("bigbio/pubmed_qa", cache_dir=hf_ds_cache)
    train_ds = raw["train"]
    val_ds   = raw["validation"]

    preprocess_fn = partial(
        preprocess,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    train_tok = train_ds.map(
        preprocess_fn,
        batched=False,
        remove_columns=train_ds.column_names
    )
    val_tok = val_ds.map(
        preprocess_fn,
        batched=False,
        remove_columns=val_ds.column_names
    )

    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        eval_strategy="epoch",
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        fp16=True,
        logging_steps=100,
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to=[]
    )

    # data_collator = DataCollatorWithPadding(
    #     tokenizer,
    #     padding='max_length',
    #     max_length=args.max_length,          # 和你 map 里用的 max_len 保持一致
    #     pad_to_multiple_of=None,
    #     return_tensors='pt',
    #     label_pad_token_id=-100, # 对 labels 用 -100
    # )
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator = default_data_collator
    )

    # 统计参数
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"⚙️  Model params: {params:,}")

    # 启动训练
    trainer.train()
    trainer.evaluate()

if __name__ == "__main__":
    main()
