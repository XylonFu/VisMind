import json
from pathlib import Path
from typing import Dict, Any, List

from langchain_core.messages import HumanMessage

from agents import students_teacher
from agents.utils.prompts import get_student_alpha_prompt, get_student_beta_prompt, get_teacher_system_prompt
from config import (get_agent_config, get_graph_config, event_config,
                    TEACHER_MODEL_NAME, TEACHER_MODEL_BASE, TEACHER_MODEL_KEYS,
                    STUDENT_MODEL_NAME, STUDENT_MODEL_BASE, STUDENT_MODEL_KEYS)
from utils.io_utils import load_image, output_exists, save_output
from utils.text_utils import process_answer, MessageEncoder


def process_single_file(json_file: Path, input_dir: Path, output_dir: Path):
    file_stem = json_file.stem
    if output_exists(file_stem, output_dir):
        print(f"⏭️ {file_stem} 已存在，跳过处理。")
        return

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)

        question = data["question"]
        solution = process_answer(data["answer"])
        image_paths = data["image"]

        if isinstance(image_paths, str):
            image_paths = [image_paths]

        images = []
        for path in image_paths:
            image = load_image(path, input_dir)
            images.append(f"data:image/png;base64,{image}")

        content = [{"type": "text", "text": question}]
        for img in images:
            content.append({"type": "image_url", "image_url": {"url": img}})

        message = HumanMessage(content=content)

        student_alpha_config = get_agent_config(get_student_alpha_prompt(), STUDENT_MODEL_NAME,
                                                STUDENT_MODEL_BASE, STUDENT_MODEL_KEYS)
        student_beta_config = get_agent_config(get_student_beta_prompt(), STUDENT_MODEL_NAME,
                                               STUDENT_MODEL_BASE, STUDENT_MODEL_KEYS)
        teacher_config = get_agent_config(get_teacher_system_prompt(), TEACHER_MODEL_NAME,
                                          TEACHER_MODEL_BASE, TEACHER_MODEL_KEYS)
        graph_config = get_graph_config(question, solution, images)

        app = students_teacher.graph(student_alpha_config, student_beta_config, teacher_config, graph_config)

        print(f"🚀 处理中: {json_file.name}...")
        event_list: List[Dict[str, Any]] = []

        for event in app.stream({"messages": [message]}, config=event_config):
            event_list.append(event)

        save_output(file_stem, image_paths, solution, message, event_list, output_dir, encoder_cls=MessageEncoder)
        print(f"✅ {json_file.name} 处理完成，保存成功。")
    except Exception as e:
        print(f"❌ {json_file.name} 处理失败: {str(e)}")
