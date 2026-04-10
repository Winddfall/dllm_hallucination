"""
python server_pro_v2_llada.py
server_pro_v2.py — 数据并行优化版 DiffuCoder 推理服务器

核心优化：
1. 数据并行：每张 A100 各加载一个完整的 7B 模型副本，互不干扰
2. asyncio.Event：零延迟结果通知，替换 sleep 轮询
3. 智能 Batch 组装：主动等待凑大 batch，充分利用 GPU 算力
4. 线程池推理：blocking 的 PyTorch 推理放入独立线程，不阻塞 FastAPI 事件循环
"""

import asyncio
import uuid
import time
import concurrent.futures
import torch
import numpy as np
import torch.nn.functional as F
from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel
import uvicorn

app = FastAPI()

# ═══════════════════════════════════════════
# 1. LLaDA 核心扩散逻辑 (来自 generate.py)
# ═══════════════════════════════════════════

def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens

@torch.no_grad()
def llada_generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0., 
                   cfg_scale=0., remasking='low_confidence', mask_id=126336):
    """适配 Batch 处理的 LLaDA 扩散采样函数"""
    batch_size = prompt.shape[0]
    device = model.device
    
    # 初始化输出张量 [Batch, Prompt_Len + Gen_Len]
    x = torch.full((batch_size, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        
        for i in range(steps_per_block):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_combined = torch.cat([x, un_x], dim=0)
                logits = model(x_combined).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            else:
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=device)

            x0_p[:, block_end:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=device)
            for j in range(batch_size):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x[:, prompt.shape[1]:]

# ═══════════════════════════════════════════
# 2. 后端并发架构 (来自 server_pro_v2.py)
# ═══════════════════════════════════════════

# 配置路径
MODEL_PATH = "./models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07"
NUM_GPUS = torch.cuda.device_count()
BATCH_SIZE = 8  # 扩散模型推理较慢，Batch 不宜过大
MAX_WAIT_TIME = 0.1

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
# 维持 LLaDA 特有的模板逻辑
tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n' }}"

models = []
for i in range(NUM_GPUS):
    print(f"Loading LLaDA to GPU {i}...")
    m = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16).to(f"cuda:{i}").eval()
    models.append(m)

REQUEST_QUEUE = asyncio.Queue()
RESULT_STORE = {}
PENDING_EVENTS = {}
gpu_executor = concurrent.futures.ThreadPoolExecutor(max_workers=NUM_GPUS)

def _run_inference(gpu_id, input_ids_batch):
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

    """在指定 GPU 上运行扩散推理"""
    device = f"cuda:{gpu_id}"
    # 简单的左 Padding 逻辑
    max_len = max(t.size(1) for t in input_ids_batch)
    padded_inputs = [F.pad(t, (max_len - t.size(1), 0), value=tokenizer.pad_token_id or 0) for t in input_ids_batch]
    batch_tensor = torch.cat(padded_inputs, dim=0).to(device)
    
    # 调用扩散生成
    out_tokens = llada_generate(
        models[gpu_id], 
        batch_tensor, 
        steps=128, 
        gen_length=128, 
        temperature=0.0, 
        cfg_scale=0.0
    )
    return [clean_and_normalize_output(tokenizer.decode(t, skip_special_tokens=True)) for t in out_tokens]

async def worker(gpu_id):
    loop = asyncio.get_event_loop()
    while True:
        items = [await REQUEST_QUEUE.get()]
        deadline = loop.time() + MAX_WAIT_TIME
        while len(items) < BATCH_SIZE:
            try:
                item = await asyncio.wait_for(REQUEST_QUEUE.get(), timeout=deadline - loop.time())
                items.append(item)
            except asyncio.TimeoutError: break

        req_ids = [it['id'] for it in items]
        inputs = [it['input_ids'] for it in items]
        
        try:
            results = await loop.run_in_executor(gpu_executor, _run_inference, gpu_id, inputs)
            for rid, res in zip(req_ids, results):
                RESULT_STORE[rid] = res
                PENDING_EVENTS[rid].set()
        except Exception as e:
            print(f"Inference error on GPU {gpu_id}: {e}")
            for rid in req_ids:
                RESULT_STORE[rid] = f"Error: {str(e)}"
                PENDING_EVENTS[rid].set()

@app.on_event("startup")
async def startup():
    for i in range(NUM_GPUS):
        asyncio.create_task(worker(i))

@app.post("/v1/chat/completions")
async def chat(request: Request):
    data = await request.json()
    msgs = data.get("messages", [])
    prompt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(prompt, return_tensors="pt")['input_ids']
    
    rid = str(uuid.uuid4())
    event = asyncio.Event()
    PENDING_EVENTS[rid] = event
    
    await REQUEST_QUEUE.put({'id': rid, 'input_ids': input_ids})
    await event.wait()
    
    ans = RESULT_STORE.pop(rid)
    del PENDING_EVENTS[rid]
    return {"choices": [{"message": {"role": "assistant", "content": ans}}]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)