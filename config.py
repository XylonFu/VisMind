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
    }


def get_graph_config(question, solution, images):
    return {
        "reference": {
            "question": question,
            "solution": solution,
            "images": images,
        },
        "session_turn": 5,
    }


event_config = {
    "recursion_limit": 100
}

CONCURRENCY = 100

CONDA_ENV_PATH = "/gpfs/work/int/qiufengwang/xinlong_fu/conda/env/vllm"

STUDENT_MODEL_BASE = os.getenv("STUDENT_MODEL_BASE")
STUDENT_MODEL_KEYS = os.getenv("STUDENT_MODEL_KEYS")
STUDENT_MODEL_NAME = "InternVL3-14B"
STUDENT_MODEL_PATH = "/gpfs/work/int/qiufengwang/xinlong_fu/downloads/models/OpenGVLab/InternVL3-14B"

TEACHER_MODEL_BASE = os.getenv("TEACHER_MODEL_BASE")
TEACHER_MODEL_KEYS = os.getenv("TEACHER_MODEL_KEYS")
TEACHER_MODEL_NAME = "InternVL3-14B"
TEACHER_MODEL_PATH = "/gpfs/work/int/qiufengwang/xinlong_fu/downloads/models/OpenGVLab/InternVL3-14B"
