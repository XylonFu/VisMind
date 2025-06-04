import json
from typing import List

import tiktoken


def count_tokens(texts: List[str], encoder_name: str = "cl100k_base") -> int:
    """统计一组文本的总token数"""
    encoder = tiktoken.get_encoding(encoder_name)
    total = 0
    for text in texts:
        tokens = encoder.encode(text, allowed_special={"<|endoftext|>"})
        total += len(tokens)
    return total


def truncate_jsonl_by_tokens(
        input_file: str,
        output_file: str,
        max_tokens: int,
        encoder_name: str = "cl100k_base"
) -> None:
    """
    截取JSONL文件前N个token的内容
    包括刚好使总token数超过max_tokens的最后一行
    """
    encoder = tiktoken.get_encoding(encoder_name)
    total_tokens = 0
    selected_lines = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            # 计算当前行的token数
            contents = []
            for msg in record.get("messages", []):
                content = msg.get("content", "")
                if content:  # 只处理非空内容
                    contents.append(content)

            if not contents:  # 如果没有有效内容，token数为0
                tokens_in_line = 0
            else:
                tokens_in_line = count_tokens(contents, encoder_name)

            # 添加当前行（无论是否会超过）
            selected_lines.append(line)
            total_tokens += tokens_in_line

            # 检查是否达到或超过token限制
            if total_tokens >= max_tokens:
                break

    # 写入截断后的文件
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for line in selected_lines:
            f_out.write(line + '\n')

    print(f"截断完成！总token数: {total_tokens}, 保留行数: {len(selected_lines)}")


if __name__ == "__main__":
    input_jsonl = "input.jsonl"
    output_jsonl = "output.jsonl"
    max_tokens = 1000000

    truncate_jsonl_by_tokens(input_jsonl, output_jsonl, max_tokens)
