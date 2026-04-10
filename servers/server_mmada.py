# python server_pro_v2_mmada.py
import asyncio
import uuid
import time
import concurrent.futures
import torch
import numpy as np
import torch.nn.functional as F
from fastapi import FastAPI, Request
from transformers import AutoTokenizer
from models import MMadaModelLM  # 确保 models 文件夹在路径下
import uvicorn

app = FastAPI()

# ═══════════════════════════════════════════
# MMaDA 核心扩散采样逻辑 (来自 test_MMaDA.py)
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
def mmada_generate_batch(model, prompt_ids, steps=128, gen_length=128, block_length=128, 
                         temperature=1.0, cfg_scale=0., mask_id=126336):
    """
    针对 Batch 处理优化的 MMaDA 生成函数
    """
    device = model.device
    batch_size = prompt_ids.shape[0]
    prompt_len = prompt_ids.shape[1]
    
    # 初始化画布：[Prompt + Mask]
    x = torch.full((batch_size, prompt_len + gen_length), mask_id, dtype=torch.long).to(device)
    x[:, :prompt_len] = prompt_ids.clone()
    
    prompt_index = (x != mask_id)
    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        # 当前 Block 的 Mask 范围
        block_start = prompt_len + num_block * block_length
        block_end = prompt_len + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        
        # 计算每步转移多少个 token
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        
        for i in range(steps_per_block):
            mask_index = (x == mask_id)
            
            # Classifier-Free Guidance 逻辑
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_combined = torch.cat([x, un_x], dim=0)
                logits = model(x_combined).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            # Gumbel 采样与重掩码逻辑 (Low Confidence 策略)
            logits_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_noise, dim=-1)
            
            p = F.softmax(logits.to(torch.float64), dim=-1)
            x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            
            # 屏蔽未来 Block
            x0_p[:, block_end:] = -np.inf
            
            # 确定哪些位置从 Mask 变为预测值
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=device)
            for j in range(batch_size):
                _, select_idx = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_idx] = True
            
            x[transfer_index] = x0[transfer_index]
            
    return x[:, prompt_len:]

# ═══════════════════════════════════════════
# 模型加载与并行配置
# ═══════════════════════════════════════════
model_path = "./models--Gen-Verse--MMaDA-8B-MixCoT/snapshots/3ee0085f0c42541f1134aae30482954451952406"
NUM_GPUS = torch.cuda.device_count()

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# 设置 MMaDA 专用的 Chat Template
tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n' }}"

models = []
for gpu_id in range(NUM_GPUS):
    print(f"Loading MMaDA to cuda:{gpu_id}...")
    m = MMadaModelLM.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16
    ).to(f"cuda:{gpu_id}").eval()
    models.append(m)

REQUEST_QUEUE = asyncio.Queue()
BATCH_SIZE = 4      # MMaDA 推理步数多，Batch 不宜过大
MAX_WAIT_TIME = 0.2
RESULT_STORE = {}
PENDING_EVENTS = {}

gpu_executor = concurrent.futures.ThreadPoolExecutor(max_workers=NUM_GPUS)

# ═══════════════════════════════════════════
# 推理与服务逻辑
# ═══════════════════════════════════════════

def _run_mmada_inference(gpu_id: int, input_ids_list: list[torch.Tensor]) -> list[str]:
    """同步阻塞推理函数"""
    model = models[gpu_id]
    device = f"cuda:{gpu_id}"
    
    # 将 list 转换为 padding 后的张量
    # 注意：MMaDA 通常需要左对齐或特定处理，此处简单 padding
    max_len = max(ids.shape[1] for ids in input_ids_list)
    padded_ids = []
    for ids in input_ids_list:
        pad_len = max_len - ids.shape[1]
        # 假设 0 为 padding (根据 tokenizer 调整)
        padded = F.pad(ids, (pad_len, 0), value=tokenizer.pad_token_id or 0)
        padded_ids.append(padded)
    
    batch_input = torch.cat(padded_ids, dim=0).to(device)
    
    t1 = time.time()
    # 调用合并后的扩散采样函数
    output_tokens = mmada_generate_batch(
        model, 
        batch_input, 
        steps=128, 
        gen_length=128, 
        block_length=128, 
        temperature=1.0
    )
    print(f"[GPU {gpu_id}] 推理耗时: {time.time() - t1:.2f}s")
    
    return [tokenizer.decode(t, skip_special_tokens=True) for t in output_tokens]

async def batch_processor(gpu_id: int):
    loop = asyncio.get_event_loop()
    while True:
        # 收集 Batch
        items = []
        first_item = await REQUEST_QUEUE.get()
        items.append(first_item)
        
        deadline = loop.time() + MAX_WAIT_TIME
        while len(items) < BATCH_SIZE:
            remaining = deadline - loop.time()
            if remaining <= 0: break
            try:
                item = await asyncio.wait_for(REQUEST_QUEUE.get(), timeout=remaining)
                items.append(item)
            except asyncio.TimeoutError: break

        # 执行推理
        input_ids_list = [item['input_ids'] for item in items]
        req_ids = [item['id'] for item in items]
        
        try:
            results = await loop.run_in_executor(gpu_executor, _run_mmada_inference, gpu_id, input_ids_list)
            for rid, res in zip(req_ids, results):
                RESULT_STORE[rid] = res
                PENDING_EVENTS[rid].set()
        except Exception as e:
            print(f"Error in GPU {gpu_id}: {e}")
            for rid in req_ids:
                RESULT_STORE[rid] = f"Inference Error: {e}"
                PENDING_EVENTS[rid].set()

@app.on_event("startup")
async def startup():
    for i in range(NUM_GPUS):
        asyncio.create_task(batch_processor(i))

@app.post("/v1/chat/completions")
async def chat(request: Request):
    data = await request.json()
    msgs = data.get("messages", [])
    
    # 构造 Prompt
    prompt_str = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(prompt_str, return_tensors="pt")['input_ids']
    
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