from pathlib import Path


def get_agent_config(prompt):
    return {
        "prompt": prompt,
        "model": "qwen2.5-vl-72b-instruct",
        "temperature": 1.0,
        "max_tokens": 8192,
    }


graph_config = {
    "session_turn": 5
}

event_config = {
    "recursion_limit": 100
}

INPUT_DIR = Path(__file__).parent / "input/GeoQAPlus"
OUTPUT_DIR = Path(__file__).parent / "output/GeoQAPlus/event-0419"

CONCURRENCY = 25
