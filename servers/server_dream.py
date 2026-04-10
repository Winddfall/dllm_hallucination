"""
python server_pro_v2_dream.py
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
from fastapi import FastAPI, Request
from transformers import AutoModel, AutoTokenizer
import torch
import uvicorn

app = FastAPI()

# ═══════════════════════════════════════════
# 模型加载：数据并行，每张 GPU 各一个副本
# ═══════════════════════════════════════════
base_path = "./models--Dream-org--Dream-v0-Instruct-7B/snapshots/05334cb9faaf763692dcf9d8737c642be2b2a6ae"
NUM_GPUS = torch.cuda.device_count()
print(f"检测到 {NUM_GPUS} 张 GPU，将进行数据并行加载")

# tokenizer 全局共享（只读操作，线程安全）
tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 每张 GPU 加载一个独立的模型副本
models = []
for gpu_id in range(NUM_GPUS):
    print(f"正在将模型加载到 cuda:{gpu_id} ...")
    m = AutoModel.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(f"cuda:{gpu_id}").eval()
    models.append(m)
    print(f"cuda:{gpu_id} 加载完成，显存占用: {torch.cuda.memory_allocated(gpu_id) / 1e9:.1f} GB")

print(f"所有 {NUM_GPUS} 个模型副本加载完毕！")

# ═══════════════════════════════════════════
# 异步批处理核心配置
# ═══════════════════════════════════════════
REQUEST_QUEUE = asyncio.Queue()

# 80GB A100 跑 7B bf16 模型 (≈14GB)，剩余空间足够大 batch
BATCH_SIZE = 8
# 等待凑 batch 的最长时间（秒），权衡延迟与吞吐
MAX_WAIT_TIME = 0.15

# 结果存储：req_id -> result_text
RESULT_STORE = {}
# 事件通知：req_id -> asyncio.Event（零延迟通知）
PENDING_EVENTS = {}

# 专用线程池：每个 GPU 一个线程，避免多线程竞争同一 GPU
gpu_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=NUM_GPUS,
    thread_name_prefix="gpu-inference"
)


# 统计信息
batch_count = [0] * max(NUM_GPUS, 1)  # 每个 GPU 处理的 batch 计数
total_start_time = [None]              # 记录第一个 batch 开始的时间

def _run_inference(gpu_id: int, prompts: list[str], max_new_tokens: int = 256) -> list[str]:
    """
    在指定 GPU 上执行扩散推理（同步阻塞函数，由线程池调用）。
    
    这个函数在独立线程中运行，不会阻塞 FastAPI 的事件循环。
    """
    model = models[gpu_id]
    device = f"cuda:{gpu_id}"

    t0 = time.time()

    # Tokenize + 移到对应 GPU
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    t_tokenize = time.time() - t0

    t1 = time.time()
    with torch.no_grad():
        output = model.diffusion_generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            steps=max_new_tokens,       # 保持每 token 一步
            temperature=0.2,
            top_p=0.95,
            alg="entropy",
            alg_temp=0.,
            return_dict_in_generate=True
        )
    t_infer = time.time() - t1

    # 解码
    t2 = time.time()
    input_length = inputs.input_ids.shape[1]
    results = []
    for i in range(len(prompts)):
        gen_seq = output.sequences[i][input_length:]
        full_text = tokenizer.decode(gen_seq, skip_special_tokens=False)
        clean_text = full_text.split('<|dlm_pad|>')[0].replace('<|im_end|>', '').strip()
        results.append(clean_text)
    t_decode = time.time() - t2

    batch_count[gpu_id] += 1
    if total_start_time[0] is None:
        total_start_time[0] = time.time()
    elapsed = time.time() - total_start_time[0] if total_start_time[0] else 0

    print(
        f"  [GPU {gpu_id}] Batch #{batch_count[gpu_id]} 完成 | "
        f"batch_size={len(prompts)} | "
        f"tokenize={t_tokenize:.2f}s | 推理={t_infer:.1f}s | decode={t_decode:.2f}s | "
        f"总耗时={elapsed:.0f}s"
    )

    return results


async def collect_batch() -> list[dict]:
    """
    智能 Batch 组装：先等第一个请求到达，然后在 MAX_WAIT_TIME 内尽量多收集。
    这比原来的 get_nowait() 策略更优：
    - 原策略：队列瞬间为空就启动推理 → 大量 batch_size=1 的小 batch
    - 新策略：主动等待凑大 batch → GPU 利用率更高
    """
    items = []

    # 阻塞等待第一个请求（不浪费 CPU）
    first_item = await REQUEST_QUEUE.get()
    items.append(first_item)

    # 在 MAX_WAIT_TIME 内继续收集，直到 batch 满或超时
    loop = asyncio.get_event_loop()
    deadline = loop.time() + MAX_WAIT_TIME

    while len(items) < BATCH_SIZE:
        remaining = deadline - loop.time()
        if remaining <= 0:
            break
        try:
            item = await asyncio.wait_for(REQUEST_QUEUE.get(), timeout=remaining)
            items.append(item)
        except asyncio.TimeoutError:
            break

    return items


async def batch_processor(gpu_id: int):
    """
    每张 GPU 一个独立的 batch processor 协程。
    两个 processor 从同一个队列取任务，自然实现负载均衡。
    """
    loop = asyncio.get_event_loop()
    print(f"[GPU {gpu_id}] batch_processor 启动")

    while True:
        # 1. 收集一批任务
        items = await collect_batch()
        batch_size = len(items)
        print(f"[GPU {gpu_id}] 处理 Batch，大小: {batch_size}")

        prompts = [item['prompt'] for item in items]
        request_ids = [item['id'] for item in items]

        try:
            # 2. 提交到线程池执行推理（不阻塞事件循环）
            results = await loop.run_in_executor(
                gpu_executor,
                _run_inference,
                gpu_id,
                prompts
            )

            # 3. 存储结果并通知等待的请求
            for req_id, result_text in zip(request_ids, results):
                RESULT_STORE[req_id] = result_text
                PENDING_EVENTS[req_id].set()  # 零延迟通知

        except Exception as e:
            print(f"[GPU {gpu_id}] 推理出错: {e}")
            for req_id in request_ids:
                RESULT_STORE[req_id] = f"Error: {str(e)}"
                PENDING_EVENTS[req_id].set()


@app.on_event("startup")
async def startup_event():
    """为每张 GPU 启动一个独立的 batch_processor"""
    for gpu_id in range(NUM_GPUS):
        asyncio.create_task(batch_processor(gpu_id))
    print(f"已启动 {NUM_GPUS} 个 batch_processor（数据并行）")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    data = await request.json()
    messages = data.get("messages", [])
    user_query = messages[-1]["content"] if messages else ""

    # 构造 Prompt
    prompt = (
        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{user_query.strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    # 生成唯一请求 ID，创建事件通知对象
    req_id = str(uuid.uuid4())
    event = asyncio.Event()
    PENDING_EVENTS[req_id] = event

    # 入队
    await REQUEST_QUEUE.put({'id': req_id, 'prompt': prompt})

    # 零延迟等待结果（替换原来的 sleep(0.1) 轮询）
    await event.wait()

    # 取出结果并清理
    response_text = RESULT_STORE.pop(req_id)
    del PENDING_EVENTS[req_id]

    return {
        "choices": [{
            "message": {"role": "assistant", "content": response_text}
        }]
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=600)
