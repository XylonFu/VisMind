import json
from pathlib import Path
from typing import Dict, Any, List

from langchain_core.messages import HumanMessage

from agents import students_teacher
from agents.utils.prompts import get_student_alpha_prompt, get_student_beta_prompt, get_teacher_system_prompt
from config import get_agent_config, get_graph_config, event_config
from utils.io_utils import load_image, output_exists, save_output
from utils.text_utils import process_answer, MessageEncoder


def process_single_file(json_file: Path):
    file_stem = json_file.stem
    if output_exists(file_stem):
        print(f"⏭️ {file_stem} 已存在，跳过处理。")
        return

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)

        question = data["subject"]
        solution = process_answer(data["answer"])
        image_path = data["image_path"]

        image = load_image(image_path)
        message = HumanMessage(
            content=[
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}}]
        )

        student_alpha_config = get_agent_config(get_student_alpha_prompt())
        student_beta_config = get_agent_config(get_student_beta_prompt())
        teacher_config = get_agent_config(get_teacher_system_prompt())
        graph_config = get_graph_config(question, solution, image)

        app = students_teacher.graph(student_alpha_config, student_beta_config, teacher_config, graph_config)

        print(f"🚀 处理中: {json_file.name}...")
        event_list: List[Dict[str, Any]] = []

        for event in app.stream({"messages": [message]}, config=event_config):
            event_list.append(event)

        save_output(file_stem, message, event_list, encoder_cls=MessageEncoder)
        print(f"✅ {json_file.name} 处理完成，保存成功。")
    except Exception as e:
        print(f"❌ {json_file.name} 处理失败: {str(e)}")
