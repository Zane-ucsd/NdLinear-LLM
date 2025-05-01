#!/usr/bin/env python
# explore_pubmedqa.py

import os
import random
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

def text_stats(texts):
    """计算一组文本的 token 数和 word 数的统计信息"""
    lens_words = [len(t.split()) for t in texts]
    return {
        "count": len(lens_words),
        "min_words": int(np.min(lens_words)),
        "max_words": int(np.max(lens_words)),
        "avg_words": float(np.mean(lens_words)),
        "p50_words": int(np.percentile(lens_words, 50)),
        "p90_words": int(np.percentile(lens_words, 90)),
    }

def main():
    # 1. 加载数据集
    cache_dir = os.environ.get("HF_DATASETS_CACHE", None)
    ds = load_dataset(
        "bigbio/pubmed_qa",
        "pubmed_qa_labeled_fold0_source",
        cache_dir=cache_dir
    )

    # 2. 打印各 split 大小和列名
    print("==== Splits and sizes ====")
    for split in ["train", "validation", "test"]:
        print(f"  {split:10s}: {len(ds[split])} examples")
    print("\n==== Columns in train split ====")
    print(" ", ds["train"].column_names)

    # 3. 随机抽几个样本看看
    print("\n==== 5 random train samples ====")
    for idx in random.sample(range(len(ds["train"])), 5):
        ex = ds["train"][idx]
        print(ex.keys())
        # dict_keys(['QUESTION', 'CONTEXTS', 'LABELS', 'MESHES', 'YEAR', 'reasoning_required_pred', 'reasoning_free_pred', 'final_decision', 'LONG_ANSWER'])
        context = ex["CONTEXTS"][0] if isinstance(ex["CONTEXTS"], (list,tuple)) else ex["CONTEXTS"]
        print(f"\n--- example #{idx} ---")
        print("QUESTION   :", ex["QUESTION"])
        print("LABELS  :", ex["LABELS"])
        print("CONTEXT     :", context[:200].replace("\n"," ")+"…" if len(context)>200 else context)
        print("LONG_ANSWER :", ex["LONG_ANSWER"][:200].replace("\n"," ")+"…" if len(ex["LONG_ANSWER"])>200 else ex["LONG_ANSWER"])

    # 4. 统计文本长度分布
    print("\n==== Text length stats (word counts) ====")
    # QUESTION
    q_stats = text_stats([ex["QUESTION"] for ex in ds["train"]])
    print(" QUESTION :", q_stats)
    # CONTEXT (只取第一个元素)
    ctxs = [ex["CONTEXTS"][0] if isinstance(ex["CONTEXTS"], (list,tuple)) else ex["CONTEXTS"]
            for ex in ds["train"]]
    c_stats = text_stats(ctxs)
    print(" CONTEXT  :", c_stats)
    # LONG_ANSWER
    a_stats = text_stats([ex["LONG_ANSWER"] for ex in ds["train"]])
    print(" LONG_ANSWER:", a_stats)

    # 5. 如果你想看 token 长度分布，也可以用 tokenizer
    print("\n==== Token length stats (using model tokenizer) ====")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",
                                              cache_dir=os.environ.get("HF_HOME"))
    tok_lens = [len(tokenizer(context, truncation=True)["input_ids"]) for context in ctxs]
    print(" CONTEXT tokens: ",
          {"min": min(tok_lens), "max": max(tok_lens), "avg": float(np.mean(tok_lens))})

if __name__ == "__main__":
    main()
