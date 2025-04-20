import json
import os
from typing import List, Tuple

import tiktoken


def get_json_files(directory: str) -> List[str]:
    """获取指定目录下所有json文件路径"""
    return [
        os.path.join(directory, file)
        for file in os.listdir(directory)
        if file.endswith(".json")
    ]


def extract_contents_from_file(file_path: str) -> Tuple[str, List[str]]:
    """从JSON文件中提取所需的文本内容"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 提取 message.content[0].text
    try:
        message_text = data["message"]["content"][0]["text"]
    except (KeyError, IndexError, TypeError):
        message_text = ""

    # 提取 events[*][role]['messages'][*]['content']
    event_contents = []
    try:
        for event in data.get("events", []):
            if not isinstance(event, dict):
                continue
            for role_data in event.values():
                if not isinstance(role_data, dict):
                    continue
                messages = role_data.get("messages", [])
                for msg in messages:
                    if isinstance(msg, dict):
                        content = msg.get("content", "")
                        event_contents.append(content)
    except Exception as e:
        print(f"Error parsing events in {file_path}: {e}")

    return message_text, event_contents


def count_tokens(texts: List[str], encoder_name: str = "cl100k_base") -> int:
    """统计一组文本的总token数"""
    encoder = tiktoken.get_encoding(encoder_name)
    return sum(len(encoder.encode(text)) for text in texts)


def process_directory(directory: str):
    """主处理逻辑：读取所有json，提取内容，统计token"""
    total_tokens = 0
    json_files = get_json_files(directory)

    for file in json_files:
        message_text, event_contents = extract_contents_from_file(file)
        token_count = count_tokens([message_text] + event_contents)
        print(f"{os.path.basename(file)}: {token_count} tokens")
        total_tokens += token_count

    print(f"\nTotal tokens: {total_tokens}")


if __name__ == "__main__":
    directory_path = "output/GeoQAPlus/event-0419"
    process_directory(directory_path)
