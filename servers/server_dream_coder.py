"""
python server_pro_v2_dream_coder.py
server_pro_v2.py — 针对 Dream-Coder 量身定制的数据并行优化版

主要改动：
1. 采用官方 tokenizer.apply_chat_template 处理对话模板。
2. diffusion_generate 默认使用 alg="entropy" 算法。
3. 优化了解码截断逻辑，使用 eos_token 进行精确切割。
"""

import asyncio
import uuid
import time
import concurrent.futures
import re
from fastapi import FastAPI, Request
from transformers import AutoModel, AutoTokenizer
import torch
import uvicorn

app = FastAPI()

# ═══════════════════════════════════════════
# 模型加载：数据并行配置
# ═══════════════════════════════════════════
base_path = "./models--Dream-org--Dream-Coder-v0-Instruct-7B/snapshots/5d9e88c723af9045f362748b5284bdf43d9c501e"
NUM_GPUS = torch.cuda.device_count()
print(f"检测到 {NUM_GPUS} 张 GPU，正在启动数据并行加载...")

tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

models = []
for gpu_id in range(NUM_GPUS):
    print(f"正在加载副本到 cuda:{gpu_id}...")
    m = AutoModel.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(f"cuda:{gpu_id}").eval()
    models.append(m)

# ═══════════════════════════════════════════
# 异步批处理与推理核心
# ═══════════════════════════════════════════
REQUEST_QUEUE = asyncio.Queue()
BATCH_SIZE = 8
MAX_WAIT_TIME = 0.15

RESULT_STORE = {}
PENDING_EVENTS = {}

gpu_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=NUM_GPUS,
    thread_name_prefix="gpu-worker"
)

def _run_inference(gpu_id: int, batch_data: list[dict], max_new_tokens: int = 512) -> list[str]:
    """
    执行推理并执行签名拼接
    """
    model = models[gpu_id]
    device = f"cuda:{gpu_id}"
    
    prompts = [d['prompt'] for d in batch_data]
    signatures = [d['signature'] for d in batch_data]

    # Tokenize
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    input_length = inputs.input_ids.shape[1]

    with torch.no_grad():
        output = model.diffusion_generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            steps=512,  # 扩散步数设为128通常能在质量与速度间取得平衡
            temperature=0.2,
            top_p=0.95,
            alg="entropy",
            alg_temp=0.,
            return_dict_in_generate=True
        )

    results = []
    for i in range(len(prompts)):
        # 提取生成的部分
        gen_seq = output.sequences[i][input_length:]
        full_text = tokenizer.decode(gen_seq, skip_special_tokens=False)
        
        # 1. 截断 EOS 之后的内容
        clean_text = full_text.split(tokenizer.eos_token)[0]
        
        # 2. 扩散模型有时会在开头产生噪声字符（如 BCEF），进行正则清理
        clean_text = re.sub(r'^[A-Z]{1,10}\s*', '', clean_text)
        
        # 3. 拼接回原始签名，形成完整函数
        final_code = signatures[i] + clean_text
        results.append(final_code)
    
    return results

async def collect_batch():
    items = []
    first_item = await REQUEST_QUEUE.get()
    items.append(first_item)
    loop = asyncio.get_event_loop()
    deadline = loop.time() + MAX_WAIT_TIME
    while len(items) < BATCH_SIZE:
        remaining = deadline - loop.time()
        if remaining <= 0: break
        try:
            item = await asyncio.wait_for(REQUEST_QUEUE.get(), timeout=remaining)
            items.append(item)
        except asyncio.TimeoutError: break
    return items

async def batch_processor(gpu_id: int):
    loop = asyncio.get_event_loop()
    while True:
        items = await collect_batch()
        try:
            # 传递整个字典列表到推理函数
            results = await loop.run_in_executor(
                gpu_executor, _run_inference, gpu_id, items
            )
            for item, res in zip(items, results):
                RESULT_STORE[item['id']] = res
                PENDING_EVENTS[item['id']].set()
        except Exception as e:
            print(f"GPU {gpu_id} 错误: {e}")
            for item in items:
                RESULT_STORE[item['id']] = f"Error: {str(e)}"
                PENDING_EVENTS[item['id']].set()

@app.on_event("startup")
async def startup_event():
    for i in range(NUM_GPUS):
        asyncio.create_task(batch_processor(i))

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    data = await request.json()
    messages = data.get("messages", [])
    signature = data.get("function_signature", "")

    # 1. 生成模板
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # 2. 预填充：确保签名后有换行引导模型进入函数体
    if signature:
        if not signature.endswith('\n'):
            formatted_prompt += signature + "\n"
        else:
            formatted_prompt += signature

    req_id = str(uuid.uuid4())
    event = asyncio.Event()
    PENDING_EVENTS[req_id] = event

    # 3. 入队（带上 signature 以便后续拼接）
    await REQUEST_QUEUE.put({
        'id': req_id, 
        'prompt': formatted_prompt, 
        'signature': signature
    })

    await event.wait()
    response_text = RESULT_STORE.pop(req_id)
    del PENDING_EVENTS[req_id]

    return {
        "id": req_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "dream-coder-7b",
        "choices": [{
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop"
        }]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=600)