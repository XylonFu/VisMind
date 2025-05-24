import os

from dotenv import load_dotenv

load_dotenv()


def get_agent_config(prompt, model, base_url, api_key):
    return {
        "prompt": prompt,
        "model": model,
        "base_url": base_url,
        "api_key": api_key,
        "max_tokens": 8192,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0,
    }


def get_graph_config(question, solution, image):
    return {
        "reference": {
            "question": question,
            "solution": solution,
            "image": image,
        },
        "session_turn": 5,
    }


event_config = {
    "recursion_limit": 100
}

CONCURRENCY = 25

CONDA_ENV_PATH = "/gpfs/work/int/qiufengwang/xinlong_fu/conda/env/vllm"

STUDENT_MODEL_BASE = os.getenv("STUDENT_MODEL_BASE")
STUDENT_MODEL_KEYS = os.getenv("STUDENT_MODEL_KEYS")
STUDENT_MODEL_NAME = "qwen2.5-vl-3b-instruct"
STUDENT_MODEL_PATH = "/gpfs/work/int/qiufengwang/xinlong_fu/downloads/models/Qwen/Qwen2.5-VL-3B-Instruct"

TEACHER_MODEL_BASE = os.getenv("TEACHER_MODEL_BASE")
TEACHER_MODEL_KEYS = os.getenv("TEACHER_MODEL_KEYS")
TEACHER_MODEL_NAME = "qwen2.5-vl-72b-instruct"
TEACHER_MODEL_PATH = "/gpfs/work/int/qiufengwang/xinlong_fu/downloads/models/Qwen/Qwen2.5-VL-72B-Instruct"
