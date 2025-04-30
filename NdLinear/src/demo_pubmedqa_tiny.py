#!/usr/bin/env python
# demo_pubmedqa_tiny.py
import os, math, argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, EvalPrediction,default_data_collator
)
import torch
import torch.nn as nn
from ndlinear import NdLinear

import logging
from transformers import TrainerCallback

# 1) 在脚本开头配置 logger
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# 2) 定义一个 Callback，用来在每次 Trainer.log() 时打印
class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # 只打印 loss/accuracy 之类的 scalar 项
            scalar_logs = {k: v for k, v in logs.items() if isinstance(v, (float, int))}
            msg = f"Step {state.global_step}"
            msg += " | " + " | ".join(f"{k}={v:.4f}" for k, v in scalar_logs.items())
            logger.info(msg)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            # metrics 里会包含 eval_loss／eval_accuracy
            msg = f"*** Eval at step {state.global_step} ***  "
            msg += " | ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, (float, int)))
            logger.info(msg)


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Qwen2.5-0.5B on PubMedQA")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--output_dir", type=str, default="/root/autodl-tmp/output-of-all")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--grad_acc", type=int, default=8)
    p.add_argument("--max_length", type=int, default=512,
                   help="max total tokens of prompt+answer")
    p.add_argument("--use_ndlinear", action="store_true",
                   help="whether to replace MLP layers with NdLinear")
    return p.parse_args()

def replace_mlp(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and ".mlp." in name:
            # 定位父模块并替换
            parent = model; *path, attr = name.split(".")
            for p in path: parent = getattr(parent, p)
            in_f, out_f = module.in_features, module.out_features
            ndl = NdLinear((in_f,), (out_f,))
            # 拷贝权重
            ndl.align_layers[0].weight.data.copy_(module.weight.data)
            if module.bias is not None:
                ndl.align_layers[0].bias.data.copy_(module.bias.data)
            setattr(parent, attr, ndl)


def preprocess(ex, tok, max_len):
    # 1) 取 question
    question = ex['QUESTION']
    # 2) CONTEXTS 是个 list，取首个
    contexts = ex['CONTEXTS']
    context = contexts[0] if isinstance(contexts, (list, tuple)) else contexts
    # 3) LONG_ANSWER 作为生成目标
    answer = ex['LONG_ANSWER'].strip()
    
    # 构造 prompt
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    full   = prompt + " " + answer + tok.eos_token

    tkn = tok(
        full, 
        truncation=True, 
        max_length=max_len, 
        padding="max_length"
        )
    # prompt length
    pl = len(tok(prompt, add_special_tokens=False)["input_ids"])
    labels = tkn["input_ids"].copy()
    labels[:pl] = [-100]*pl
    tkn["labels"] = labels
    return tkn

def compute_metrics(p: EvalPrediction):
    preds = p.predictions
    if isinstance(preds, tuple): preds=preds[0]
    pred_ids = preds.argmax(axis=-1)
    labels = p.label_ids
    mask = labels!=-100
    acc = ( (pred_ids==labels)&mask ).sum()/mask.sum()
    #ppl = math.exp(p.metrics["eval_loss"]) if "eval_loss" in p.metrics else None
    return {"accuracy":acc}

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--use_nd", action="store_true", default=False,
    #                     help="替换 MLP 为 NdLinear")
    args=parse_args()
    #print(torch.cuda.is_available())

    # 确认缓存路径环境变量
    hf_ds_cache = os.environ.get("HF_DATASETS_CACHE")
    if not hf_ds_cache:
        raise RuntimeError("请先 export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/datasets")

    # 加载 tokenizer & 模型
    model_name="Qwen/Qwen2.5-0.5B-Instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=os.environ.get("HF_HOME"))
    if tok.pad_token_id is None:
        tok.pad_token     = tok.eos_token
        tok.pad_token_id  = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=os.environ.get("HF_HOME"),
        #torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        #torch_dtype=torch.float16,
        device_map="auto"
    )
    # if args.use_nd:
    #     replace_mlp(model)
    model.to(device)

    log_file = os.path.join(args.output_dir, "train.log")
    handlers = [
        logging.StreamHandler(),                # 控制台
        logging.FileHandler(log_file, mode="w") # 输出到文件
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers
    )
    logger = logging.getLogger(__name__)

    # 3) 打印一下确认
    logger.info(f"Logging both to console and '{log_file}'")

    # 下载并截取小数据集
    ds = load_dataset(
        "bigbio/pubmed_qa",
        "pubmed_qa_labeled_fold0_source",
        cache_dir=os.environ["HF_DATASETS_CACHE"]
    )
    train = ds["train"]        # 500 条
    val   = ds["validation"]   # 250 条
    test  = ds["test"]         # 250 条
    # 打印列名
    #print("Columns in PubMedQA train split:", ds["train"].column_names)
    # tokenize
    max_len=512
    preprocess_fn = lambda ex: preprocess(ex, tok, max_len)
    train = train.map(preprocess_fn, remove_columns=train.column_names)
    val   = val.map(preprocess_fn, remove_columns=val.column_names)
    test  = test.map(preprocess_fn, remove_columns=test.column_names)

    # Trainer 设置
    args_tr = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        prediction_loss_only=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        eval_strategy="epoch",
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        fp16=True,
        bf16 = False,  
        logging_steps=100,
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to=[]
    )
    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=train,
        eval_dataset=val,
        tokenizer=tok,
        compute_metrics=compute_metrics,
        data_collator = default_data_collator,
        callbacks=[LoggingCallback]
    )

    # 训练 & 验证
    #print(f"{'NdLinear' if args.use_nd else 'Baseline'} 开始训练 …")
    trainer.train()
 # —— 在验证集上评估 —— 
    logger.info("=== Validation ===")
    val_metrics = trainer.evaluate(eval_dataset=val)
    for k, v in val_metrics.items():
        if isinstance(v, (float, int)):
            logger.info(f"  {k}: {v:.4f}")

   # —— 在测试集上评估 —— 
    # logger.info("=== Test ===")
    # test_metrics = trainer.evaluate(eval_dataset=test)
    # for k, v in test_metrics.items():
    #     if isinstance(v, (float, int)):
    #         logger.info(f"  {k}: {v:.4f}")

if __name__=="__main__":
    main()
