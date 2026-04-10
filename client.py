"""
python client_async.py
client_async.py — 全异步高并发客户端

核心优化：
1. aiohttp 替代 requests：全异步 I/O，无 GIL 瓶颈
2. asyncio.Semaphore 控制并发：64 并发足够喂满双 GPU
3. 直接调用 HTTP API：去掉 langchain 中间层，代码反而更简洁
4. 异步进度条：实时显示完成进度
"""

import os
import json
import asyncio
import aiohttp
import aiofiles
from tqdm.asyncio import tqdm_asyncio

# --- 全局配置 ---
LLM_MODEL_NAME = 'Dream-org--Dream-Coder-v0-Instruct-7B'
SAMPLE_SUM = 1              # 每个任务生成次数
MAX_CONCURRENCY = 64        # 并发请求数（足够喂满双 A100）
API_URL = 'http://localhost:8000/v1/chat/completions'
REQUEST_TIMEOUT = 1800      # 30 分钟超时（扩散推理很慢）

PROMPT_TEMPLATE = '''
Objective: Complete the Python code based on the provided task information.
Scenario:
Extract key context from the given code snippets and complete the function logic to ensure it is fully functional and syntactically correct. 
Each task includes a Task ID, Function Signature, Docstring (summary), and specific Input/Output requirements. You must strictly adhere to these specifications.

Expected Output:
Output the completed Python function ONLY. Do not include any natural language explanations, markdown code blocks, or internal comments.
Example Format: "def index(self, key):\\n    return self.index(key)"

Steps:
1. Analyze the provided code snippets and extract core logic constraints.
2. Complete the code by integrating the function signature, docstring, and I/O requirements.

# Function Signature
{signature}

# Docstring
{docstring}

# Code to Complete
{input_text}
'''

# 写文件锁（asyncio 版本）
file_locks: dict[str, asyncio.Lock] = {}


def get_file_lock(filepath: str) -> asyncio.Lock:
    """为每个输出文件创建独立的异步锁"""
    if filepath not in file_locks:
        file_locks[filepath] = asyncio.Lock()
    return file_locks[filepath]


async def call_api(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    prompt: str,
    function_signature: str
) -> str:
    """
    调用推理 API（带并发控制）。
    
    对比 langchain 版本：
      langchain:  response = llm.invoke(prompt_instruction)
      直接调用:   async with session.post(url, json=payload) as resp: ...
    
    实际代码量差不多，但直接调用避免了 langchain 的同步阻塞和额外开销。
    """
    payload = {
        "messages": [
            {"role": "system", "content": "You are a code generator. You MUST ONLY output the raw function code block. No explanation, no thought process, no preamble, and no postamble. Directly start with the function code and end with the function code."},
            {"role": "user", "content": prompt}
        ],
        "function_signature": function_signature
    }

    async with semaphore:
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        async with session.post(API_URL, json=payload, timeout=timeout) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise RuntimeError(f"API 返回 {resp.status}: {error_text}")
            data = await resp.json()
            return data["choices"][0]["message"]["content"]


async def worker_unit(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    task_data: dict,
    sample_idx: int,
    output_file: str,
) -> str:
    """单次生成任务的最小单元"""
    question_id = task_data.get("question_id", "N/A")

    prompt_instruction = PROMPT_TEMPLATE.format(
        input_text=task_data.get("input", ""),
        signature=task_data.get("signature", ""),
        docstring=task_data.get("docstring", ""),
    )

    try:
        content = await call_api(session, semaphore, prompt_instruction, task_data.get("signature", ""))

        if not content:
            return f"Error: Task {question_id} output is empty."

        result_entry = {
            "_id": question_id,
            "generate_results": [content]
        }

        # 异步写入文件（按文件加锁，不同文件可并行写入）
        lock = get_file_lock(output_file)
        async with lock:
            async with aiofiles.open(output_file, 'a', encoding='utf-8') as f:
                await f.write(json.dumps(result_entry, ensure_ascii=False) + '\n')

        return "Success"
    except Exception as e:
        return f"Error: Task {question_id} (Sample {sample_idx}) -> {str(e)}"


async def main():
    tasks_path = 'CEPythonRaw.jsonl'
    output_folder = f'model_output/{LLM_MODEL_NAME}'
    os.makedirs(output_folder, exist_ok=True)

    # 1. 预载所有任务
    print(f"正在加载任务文件: {tasks_path}")
    all_tasks = []
    with open(tasks_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    all_tasks.append(json.loads(line.strip()))
                except Exception:
                    continue

    # 2. 构建任务队列
    job_queue = []
    for task in all_tasks:
        for i in range(1, SAMPLE_SUM + 1):
            out_file = f'{output_folder}/output{i}.jsonl'
            job_queue.append((task, i, out_file))

    total_jobs = len(job_queue)
    print(f"待处理总任务量: {total_jobs}")
    print(f"并发请求数: {MAX_CONCURRENCY} | 超时设置: {REQUEST_TIMEOUT}s")

    # 3. 创建连接池 + 信号量
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    connector = aiohttp.TCPConnector(
        limit=MAX_CONCURRENCY,       # 连接池上限 = 并发数
        keepalive_timeout=600,       # 长连接复用
    )

    error_count = 0
    success_count = 0

    async with aiohttp.ClientSession(connector=connector) as session:
        # 4. 创建所有异步任务
        tasks = [
            worker_unit(session, semaphore, task_data, sample_idx, out_file)
            for task_data, sample_idx, out_file in job_queue
        ]

        # 5. 使用异步 tqdm 监控进度
        results = await tqdm_asyncio.gather(
            *tasks,
            desc="DLLM Inference (Async)",
            total=total_jobs
        )

    # 6. 汇总结果
    for res in results:
        if "Error" in res:
            error_count += 1
            print(f"[!] {res}")
        else:
            success_count += 1

    print(f"\n{'='*50}")
    print(f"任务完成！成功: {success_count} | 失败: {error_count} | 总计: {total_jobs}")
    print(f"{'='*50}")


if __name__ == "__main__":
    asyncio.run(main())
