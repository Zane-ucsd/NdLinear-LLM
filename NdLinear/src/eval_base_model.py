#!/usr/bin/env python
# eval_base_model.py
import os, time
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, EvalPrediction, default_data_collator
)
import torch
import logging

# 配置日志
output_dir = "/root/autodl-tmp/output-of-all"
os.makedirs(output_dir, exist_ok=True)
log_file = os.path.join(output_dir, "eval_base_model.log")

handlers = [
    logging.StreamHandler(),                # 控制台输出
    logging.FileHandler(log_file, mode="w") # 文件输出
]

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=handlers
)
logger = logging.getLogger(__name__)
logger.info(f"Evaluation results will be saved to: {log_file}")

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
    full = prompt + " " + answer + tok.eos_token

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
    return {"accuracy":acc}

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
                if has_labels:
                    labels = inputs["labels"][:, -1:]  # 确保labels和logits的shape匹配
            else:
                logits = None
                
        return (loss, logits, inputs["labels"]) if has_labels else (loss, logits, None)

def main():
    # 确保设置了缓存路径
    if not os.environ.get("HF_DATASETS_CACHE"):
        os.environ["HF_DATASETS_CACHE"] = "/root/autodl-tmp/hf_cache/datasets"
    
    # 加载模型和tokenizer
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading model {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=os.environ.get("HF_HOME"))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=os.environ.get("HF_HOME"),
        device_map="auto"
    )
    
    # 加载数据集
    logger.info("Loading dataset...")
    ds = load_dataset(
        "bigbio/pubmed_qa",
        "pubmed_qa_labeled_fold0_source",
        cache_dir=os.environ["HF_DATASETS_CACHE"]
    )
    test_dataset = ds["test"]
    
    # 预处理数据集
    max_len = 256
    preprocess_fn = lambda ex: preprocess(ex, tokenizer, max_len)
    test_dataset = test_dataset.map(preprocess_fn, remove_columns=test_dataset.column_names)
    
    # 配置Trainer
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=1,
        prediction_loss_only=False,
        fp16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
    )
    
    trainer = MemoryEfficientTrainer(
        model=model,
        args=args,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
    )
    
    # 评估
    logger.info("Starting evaluation...")
    start_time = time.time()
    
    # 分批评估测试集
    test_batch_size = 50  # 每次评估50个样本
    test_results = []
    for i in range(0, len(test_dataset), test_batch_size):
        batch_test = test_dataset.select(range(i, min(i + test_batch_size, len(test_dataset))))
        batch_metrics = trainer.evaluate(eval_dataset=batch_test, metric_key_prefix="test")
        test_results.append(batch_metrics)
        logger.info(f"Batch {i//test_batch_size + 1} metrics: {batch_metrics}")
    
    # 合并结果
    final_test_metrics = {}
    for k in test_results[0].keys():
        if isinstance(test_results[0][k], (float, int)):
            values = [r[k] for r in test_results]
            final_test_metrics[k] = sum(values) / len(values)
    
    eval_time = time.time() - start_time
    
    # 打印最终结果
    logger.info("\nFinal Evaluation Results:")
    for k, v in final_test_metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    logger.info(f"Runtime: {eval_time:.2f} seconds")
    logger.info(f"Samples per second: {len(test_dataset)/eval_time:.2f}")

if __name__ == "__main__":
    main() 