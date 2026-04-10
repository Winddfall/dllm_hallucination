client.py 是客户端，包含向服务端发送的每个任务的提示词和函数签名

servers/ 里是各个 dllm 的推理框架，大体都差不多。模型推理参数均参考 huggingface 上的官方标准实现

提示词模版
```python
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
```