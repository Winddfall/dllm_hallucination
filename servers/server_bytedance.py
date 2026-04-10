'''
python server_pro_v2_bytedance.py
'''
import asyncio
import uuid
import time
import concurrent.futures
from fastapi import FastAPI, Request
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import uvicorn

app = FastAPI()

# ═══════════════════════════════════════════
# 模型加载
# ═══════════════════════════════════════════
base_path = "./models--ByteDance-Seed--Stable-DiffCoder-8B-Instruct/snapshots/10cdaf9b486f1cf0273ad968459dbe2d21a1482a"
NUM_GPUS = torch.cuda.device_count()

tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
# 确保有 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

models = []
for gpu_id in range(NUM_GPUS):
    m = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(f"cuda:{gpu_id}").eval()
    models.append(m)

# ═══════════════════════════════════════════
# 配置：强制 BATCH_SIZE = 1
# ═══════════════════════════════════════════
REQUEST_QUEUE = asyncio.Queue()
BATCH_SIZE = 1  # 关键：当前模型 block-wise 模式只支持 batch=1
RESULT_STORE = {}
PENDING_EVENTS = {}

gpu_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=NUM_GPUS,
    thread_name_prefix="gpu-inference"
)

def _run_inference(gpu_id: int, prompts: list[str]) -> list[str]:
    # 清理markdown标记
    def clean_and_normalize_output(raw_output_str: str) -> str:
        cleaned_str = raw_output_str.strip()
        
        if cleaned_str.startswith('"') and cleaned_str.endswith('"'):
            return cleaned_str

        # 移除模型可能添加的前后 Markdown 标记，如 ```json ... ``` 或 ```python ... ```
        if cleaned_str.startswith('```'):
            lines = cleaned_str.split('\n')
            # 移除第一行（```python 或 ```json 等）
            if lines[0].strip().startswith('```'):
                lines = lines[1:]
            # 移除最后一行的 ```
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            cleaned_str = '\n'.join(lines)
        
        return cleaned_str.strip()

    model = models[gpu_id]
    device = f"cuda:{gpu_id}"
    
    # 理论上这里的 prompts 长度现在恒为 1
    prompt = prompts[0]
    
    # 1. Tokenize (保持与 MVP 一致)
    # 注意：model.generate 内部可能需要 input_ids 是 Tensor 且带 batch 维
    input_ids = tokenizer(prompt)['input_ids']
    input_ids_tensor = torch.tensor(input_ids).to(device).unsqueeze(0)

    # 2. 推理
    with torch.no_grad():
        # 注意：这里返回的是原始 Tensor，不是带有 .sequences 属性的对象
        output = model.generate(
            input_ids_tensor, 
            steps=512, 
            gen_length=512, 
            block_length=4, 
            temperature=0., 
            remasking='low_confidence', 
            tokenizer=tokenizer, 
            shift=False, 
            threshold=None, 
            eos_id=tokenizer.eos_token_id
        )
        
    # 3. 解码 (根据 MVP 逻辑，取生成的后半部分)
    # output[0] 取出 batch 中的第一个序列
    generated_ids = output[0][input_ids_tensor.shape[1]:]
    clean_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    clean_text = clean_and_normalize_output(clean_text)

    return [clean_text]

async def batch_processor(gpu_id: int):
    loop = asyncio.get_event_loop()
    while True:
        # 因为 BATCH_SIZE=1，这里其实就是简单的先进先出
        item = await REQUEST_QUEUE.get()
        req_id = item['id']
        
        try:
            results = await loop.run_in_executor(
                gpu_executor, _run_inference, gpu_id, [item['prompt']]
            )
            RESULT_STORE[req_id] = results[0]
        except Exception as e:
            print(f"GPU {gpu_id} Error: {e}")
            RESULT_STORE[req_id] = f"Internal Error: {str(e)}"
        finally:
            if req_id in PENDING_EVENTS:
                PENDING_EVENTS[req_id].set()

@app.on_event("startup")
async def startup_event():
    for gpu_id in range(NUM_GPUS):
        asyncio.create_task(batch_processor(gpu_id))

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    data = await request.json()
    messages = data.get("messages", [])

    # 使用官方模板构建 Prompt (参考 MVP)
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    req_id = str(uuid.uuid4())
    event = asyncio.Event()
    PENDING_EVENTS[req_id] = event

    await REQUEST_QUEUE.put({'id': req_id, 'prompt': prompt})
    await event.wait()

    response_text = RESULT_STORE.pop(req_id)
    del PENDING_EVENTS[req_id]

    return {
        "choices": [{"message": {"role": "assistant", "content": response_text}}]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)