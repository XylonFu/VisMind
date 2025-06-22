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
    return sum(len(encoder.encode(text, allowed_special={"<|endoftext|>"})) for text in texts)


def count_tokens_in_combined_file(file_path: str, encoder_name: str = "cl100k_base") -> int:
    """读取合并后的 JSON 文件或 JSONL 文件，统计所有 records 中 content 字段的 token 总数"""
    total = 0
    # 支持 JSONL 格式：每行一个 JSON 对象
    open_func = open
    with open_func(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            for msg in rec.get("messages", []):
                total += count_tokens([msg.get("content", "")], encoder_name)
    return total


def count_average_tokens_in_combined_file(file_path: str, encoder_name: str = "cl100k_base") -> float:
    """计算合并 JSONL 文件中每个样本的平均 token 数"""
    total_tokens = 0
    num_records = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            num_records += 1
            # 统计该条记录所有消息的 token
            for msg in rec.get("messages", []):
                total_tokens += count_tokens([msg.get("content", "")], encoder_name)
    if num_records == 0:
        return 0.0
    return total_tokens / num_records


def count_individual_total(directory: str, encoder_name: str = "cl100k_base") -> int:
    """统计原始目录下所有 JSON 文件内容的 token 总数"""
    total = 0
    for file in get_json_files(directory):
        message_text, event_contents = extract_contents_from_file(file)
        total += count_tokens([message_text] + event_contents, encoder_name)
    return total


if __name__ == "__main__":
    for i in range(1, 2):
        directory_path = f"../output/VisualWebInstruct118K/event-0608-0{i}"

        # individual_total = count_individual_total(directory_path)
        # print(f"Total tokens in individual files: {individual_total} tokens")

        base_name = os.path.basename(directory_path.rstrip('/\\'))
        parent_dir = os.path.dirname(directory_path)
        combined_path = os.path.join(parent_dir, f"pt-{base_name}.jsonl")

        if os.path.exists(combined_path):
            combined_tokens = count_tokens_in_combined_file(combined_path)
            print(f"Combined JSONL file {os.path.basename(combined_path)}: {combined_tokens} tokens")
            avg_tokens = count_average_tokens_in_combined_file(combined_path)
            print(f"Average tokens per record: {avg_tokens:.2f} tokens")
        else:
            print(f"Combined file not found at {combined_path}")
