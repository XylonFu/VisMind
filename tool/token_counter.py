import json
import os
from typing import List, Tuple

import tiktoken


def get_json_files(directory: str) -> List[str]:
    """获取指定目录下所有 json 文件路径"""
    return [
        os.path.join(directory, file)
        for file in os.listdir(directory)
        if file.endswith(".json")
    ]


def extract_contents_from_file(file_path: str) -> Tuple[str, List[str]]:
    """从单个 JSON 文件中提取 message_text 与 events 内容"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    try:
        message_text = data["message"]["content"][0]["text"]
    except (KeyError, IndexError, TypeError):
        message_text = ""

    event_contents: List[str] = []
    for event in data.get("events", []):
        if not isinstance(event, dict):
            continue
        for role_data in event.values():
            if not isinstance(role_data, dict):
                continue
            for msg in role_data.get("messages", []):
                if isinstance(msg, dict):
                    event_contents.append(msg.get("content", ""))
    return message_text, event_contents


def count_tokens(texts: List[str], encoder_name: str = "cl100k_base") -> int:
    """统计一组文本的总 token 数"""
    encoder = tiktoken.get_encoding(encoder_name)
    return sum(len(encoder.encode(text)) for text in texts)


def count_tokens_in_combined_file(file_path: str, encoder_name: str = "cl100k_base") -> int:
    """读取合并后的 JSON 文件，统计所有 records 中 content 字段的 token 总数"""
    with open(file_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    total = 0
    for rec in records:
        for msg in rec.get("messages", []):
            total += count_tokens([msg.get("content", "")], encoder_name)
    return total


def count_individual_total(directory: str, encoder_name: str = "cl100k_base") -> int:
    """统计原始目录下所有 JSON 文件内容的 token 总数"""
    total = 0
    for file in get_json_files(directory):
        message_text, event_contents = extract_contents_from_file(file)
        total += count_tokens([message_text] + event_contents, encoder_name)
    return total


if __name__ == "__main__":
    # 原始文件目录（Individual files）
    directory_path = "../output/GeoQAPlus/event-0419"

    # 只输出总 token 数，不展示每个文件的明细
    individual_total = count_individual_total(directory_path)
    print(f"Total tokens in individual files: {individual_total} tokens")

    # 推断合并后文件路径：与目录同名的 .json
    base_name = os.path.basename(directory_path.rstrip('/\\'))
    parent_dir = os.path.dirname(directory_path)
    combined_path = os.path.join(parent_dir, f"{base_name}.json")

    if os.path.exists(combined_path):
        combined_tokens = count_tokens_in_combined_file(combined_path)
        print(f"Combined file {os.path.basename(combined_path)}: {combined_tokens} tokens")
    else:
        print(f"Combined file not found at {combined_path}")
