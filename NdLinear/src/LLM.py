from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, StoppingCriteriaList, StoppingCriteria
import torch, threading

# 加载
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 输入
prompt = "我操你妈！"
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

# 启动生成
thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
thread.start()

# 打印流式输出
for chunk in streamer:
    print(chunk, end="", flush=True)

thread.join()
