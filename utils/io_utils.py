import base64
import json
from pathlib import Path
from threading import Lock
from typing import Any, List, Dict

from langchain_core.messages import HumanMessage

write_lock = Lock()


def load_image(image_path: str, input_dir: Path) -> str:
    full_path = input_dir / image_path
    with open(full_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def output_exists(file_stem: str, output_dir: Path) -> bool:
    return (output_dir / f"{file_stem}.json").exists()


def save_output(file_stem: str, message: HumanMessage, events: List[Dict[str, Any]], output_dir: Path,
                encoder_cls=None):
    output_data = {"id": file_stem, "message": message, "events": events}
    output_path = output_dir / f"{file_stem}.json"
    with write_lock, open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2, cls=encoder_cls)
