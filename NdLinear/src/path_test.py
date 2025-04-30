# import os, glob
# from transformers import AutoTokenizer, AutoModel

# # 1. 验证 env
# print("HF_HOME =", os.environ.get("HF_HOME"))
# print("TRANSFORMERS_CACHE =", os.environ.get("TRANSFORMERS_CACHE"))

# # 2. 清理旧缓存（可选，跑一次就删一次）
# import shutil
# shutil.rmtree(os.environ["HF_HOME"], ignore_errors=True)

# # 3. 下载模型
# AutoTokenizer.from_pretrained("bert-base-uncased")
# AutoModel.from_pretrained("bert-base-uncased")

# # 4. 列出缓存目录
# print("Cache contents:")
# for path in glob.glob(os.path.join(os.environ["HF_HOME"], "**"), recursive=True):
#     print(" ", path.replace(os.environ["HF_HOME"]+"/", ""))
import os
from datasets import load_dataset
# 触发 datasets 缓存
load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=os.environ["HF_DATASETS_CACHE"])
