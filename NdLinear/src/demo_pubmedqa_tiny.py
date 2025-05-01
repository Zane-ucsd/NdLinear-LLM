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
def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Qwen2.5-0.5B on PubMedQA")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--output_dir", type=str, default="/root/autodl-tmp/output-of-all")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--eval_batch_size", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--grad_acc", type=int, default=8)
    p.add_argument("--max_length", type=int, default=256,
                   help="max total tokens of prompt+answer")
    p.add_argument("--use_ndlinear", action="store_true",
                   help="whether to replace MLP layers with NdLinear")
    p.add_argument("--replacement_mode", type=str,
                   choices=["all", "last_n", "even",  "reinit_last_n"],
                   default="all",
                   help="哪种替换策略")
    p.add_argument("--last_n", type=int, default=8,
                   help="当 mode=last_n 时，替换最后 N 层 MLP")
    p.add_argument("--topk", type=int, default=5,
                   help="当 mode=topk 时，替换 out_features 最大的前 K 层 MLP")
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

def replace_all_mlp(model):
    # 原来的“全替换”逻辑
    replace_mlp(model)

def replace_last_n_mlp(model, n):
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    else:
        layers = model.base_model.layers
    total = len(layers)
    for idx in range(total - n, total):
        mlp = layers[idx].mlp
        # mlp 里三个线性层：gate_proj / up_proj / down_proj
        for name, lin in mlp.named_children():
            if isinstance(lin, nn.Linear):
                ndl = NdLinear((lin.in_features,), (lin.out_features,))
                ndl.align_layers[0].weight.data.copy_(lin.weight.data)
                if lin.bias is not None:
                    ndl.align_layers[0].bias.data.copy_(lin.bias.data)
                setattr(mlp, name, ndl)

def reinit_last_n_mlp(model, n, mean=0.0, std=0.02):
    """
    对模型最后 n 层的每个 mlp（gate_proj/up_proj/down_proj）执行
    Gaussian 初始化，以构造一个效果更差的 baseline。
    """
    # 先拿到 layer list
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    else:
        layers = model.base_model.layers

    total = len(layers)
    start = max(0, total - n)
    for idx in range(start, total):
        mlp = layers[idx].mlp
        for name, lin in mlp.named_children():
            if isinstance(lin, nn.Linear):
                # 重高斯初始化
                nn.init.normal_(lin.weight, mean=mean, std=std)
                if lin.bias is not None:
                    nn.init.zeros_(lin.bias)

def replace_even_mlp(model):
    if hasattr(model.model, 'layers'):
        layers = model.model.layers
        prefix = "model.model.layers"
    elif hasattr(model.base_model, 'layers'):
        layers = model.base_model.layers
        prefix = "model.base_model.layers"
    else:
        raise RuntimeError("找不到 transformer 层，请检查模型结构")
    for i, block in enumerate(layers):
        if i % 2 == 0:
            mlp = layers.mlp
            for name, lin in mlp.named_children():
                if isinstance(lin, nn.Linear):
                    ndl = NdLinear((lin.in_features,), (lin.out_features,))
                    ndl.align_layers[0].weight.data.copy_(lin.weight.data)
                    if lin.bias is not None:
                        ndl.align_layers[0].bias.data.copy_(lin.bias.data)
                    setattr(mlp, name, ndl)

# def replace_topk_by_dim(model, k):
#     mlps = []
#     for name, module in model.named_modules():
#         if isinstance(module, nn.Linear) and ".mlp." in name:
#             mlps.append((name, module, module.out_features))
#     for name, module, _ in sorted(mlps, key=lambda x: x[2], reverse=True)[:k]:
#         parent, attr = locate_parent(model, name)
#         setattr(parent, attr, new_ndlinear(module))



# class HybridMLP(nn.Module):
#     def __init__(self, orig: nn.Linear):
#         super().__init__()
#         self.orig = orig
#         self.nd   = new_ndlinear(orig)
#         self.alpha = nn.Parameter(torch.zeros(1))
#     def forward(self, x):
#         w = torch.sigmoid(self.alpha)
#         return w * self.nd(x) + (1 - w) * self.orig(x)

#     def replace_hybrid_mlp(model):
#         for name, module in list(model.named_modules()):
#             if isinstance(module, nn.Linear) and ".mlp." in name:
#                 parent, attr = locate_parent(model, name)
#                 hybrid = HybridMLP(module)
#                 setattr(parent, attr, hybrid)

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
    # 只保留最后一个token的预测，减少内存使用
    if len(preds.shape) > 2:
        preds = preds[:, -1, :]  # 只取最后一个位置的预测
    pred_ids = preds.argmax(axis=-1)
    labels = p.label_ids
    if len(labels.shape) > 1:
        labels = labels[:, -1]  # 只取最后一个位置的标签
    mask = labels != -100
    acc = (pred_ids[mask] == labels[mask]).mean()
    return {"accuracy": float(acc)}

class MemoryEfficientTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # 重写prediction_step以减少内存使用
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        
        # 使用torch.no_grad()减少内存使用
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs.loss if has_labels else None
            
            # 只保留最后一个token的logits
            if not prediction_loss_only:
                logits = outputs.logits[:, -1:, :]  # 只保留最后一个位置
            else:
                logits = None
                
        return (loss, logits, inputs["labels"]) if has_labels else (loss, logits, None)

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--use_nd", action="store_true", default=False,
    #                     help="替换 MLP 为 NdLinear")
    args = parse_args()
    #print(torch.cuda.is_available())

    log_dir = "/root/autodl-tmp/output-of-all"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir,
        f"train_{args.replacement_mode}"
        + (f"_n{args.last_n}" if args.replacement_mode=="last_n" else "")
        + (f"_k{args.topk}"    if args.replacement_mode=="topk"  else "")
        + ".log"
    )

    handlers = [
        logging.StreamHandler(),                # 控制台
        logging.FileHandler(log_file, mode="w") # 输出到文件
    ]
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=handlers
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
        device_map="auto",
        trust_remote_code=True
    )
    # print(model)               # 整个模型
    # print(model.base_model)    # Qwen2Model
    # print(dir(model.base_model))

    # if args.use_nd:
    #     replace_mlp(model)
    if args.use_ndlinear:
        if   args.replacement_mode == "all":
            replace_all_mlp(model)
        elif args.replacement_mode == "last_n":
            replace_last_n_mlp(model, args.last_n)
        elif args.replacement_mode == "even":
            replace_even_mlp(model)
        elif args.replacement_mode == "topk":
            replace_topk_by_dim(model, args.topk)
        elif args.replacement_mode == "hybrid":
            replace_hybrid_mlp(model)

    if args.replacement_mode == "reinit_last_n":
        # 注意：reinit_last_n 不依赖于 use_ndlinear
        reinit_last_n_mlp(model, args.last_n, mean=0.0, std=0.02)
    model.to(device)

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
        prediction_loss_only=False,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_acc,
        eval_strategy="epoch",
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        fp16=True,
        bf16=False,
        logging_steps=100,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        report_to=[],
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        dataloader_num_workers=0,
    )
    trainer = MemoryEfficientTrainer(
        model=model,
        args=args_tr,
        train_dataset=train,
        eval_dataset=val,
        tokenizer=tok,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
        callbacks=[LoggingCallback]
    )

    # 训练
    trainer.train()
    
    # 评估验证集
    logger.info("=== Validation ===")
    val_metrics = trainer.evaluate(eval_dataset=val)
    for k, v in val_metrics.items():
        if isinstance(v, (float, int)):
            logger.info(f"  {k}: {v:.4f}")
    
    # 评估测试集
    logger.info("=== Test ===")
    # 分批评估测试集
    test_batch_size = 50  # 每次评估50个样本
    test_results = []
    for i in range(0, len(test), test_batch_size):
        batch_test = test.select(range(i, min(i + test_batch_size, len(test))))
        batch_metrics = trainer.evaluate(eval_dataset=batch_test, metric_key_prefix="test")
        test_results.append(batch_metrics)
    
    # 合并结果
    final_test_metrics = {}
    for k in test_results[0].keys():
        if isinstance(test_results[0][k], (float, int)):
            values = [r[k] for r in test_results]
            final_test_metrics[k] = sum(values) / len(values)
    
    for k, v in final_test_metrics.items():
        logger.info(f"  {k}: {v:.4f}")

if __name__=="__main__":
    main()
