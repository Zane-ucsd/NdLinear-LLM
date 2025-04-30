from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, StoppingCriteriaList, StoppingCriteria
import torch, threading

# 加载
model_name = "Qwen/Qwen3-4B"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 输入
prompt = "你写一个two sum的代码"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# 流式输出
streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, timeout=10.0)

# 自定义停止：双换行
class StopOnDoubleNL(StoppingCriteria):
    def __call__(self, input_ids, scores):
        text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return text.endswith("\n\n")

# 解码参数
gen_kwargs = {
    "input_ids": inputs["input_ids"],
    "max_new_tokens": 512,
    "streamer": streamer,
    "do_sample": True,
    "temperature": 0.8,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "eos_token_id": tokenizer.eos_token_id,
    "stopping_criteria": StoppingCriteriaList([StopOnDoubleNL()]),
}

import logging

# 在脚本最上面配置日志
logging.basicConfig(
    filename="stream_llm_debug.log",
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s"
)

def safe_generate(**gen_kwargs):
    try:
        model.generate(**gen_kwargs)
    except Exception as e:
        logging.exception("Error in model.generate:")

# 然后把线程 target 改成 safe_generate
thread = threading.Thread(target=safe_generate, kwargs=gen_kwargs)
thread.start()

try:
    for chunk in streamer:
        print(chunk, end="", flush=True)
except Exception as e:
    logging.exception("Error in streamer loop:")
finally:
    thread.join()

