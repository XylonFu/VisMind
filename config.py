from pathlib import Path


def get_agent_config(prompt):
    return {
        "prompt": prompt,
        "model": "qwen2.5-vl-72b-instruct",
        "temperature": 1.0,
        "max_tokens": 8192,
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

DEFAULT_INPUT_DIR = Path(__file__).parent / "input/GeoQAPlus"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "output/GeoQAPlus/event-0510"
CONCURRENCY = 25

CONDA_ENV_PATH = "/gpfs/work/int/qiufengwang/xinlong_fu/conda/env/vllm"
MODEL_PATH = "/gpfs/work/int/qiufengwang/xinlong_fu/downloads/models/Qwen/Qwen2.5-VL-72B-Instruct"
SERVED_MODEL_NAME = "qwen2.5-vl-72b-instruct"
