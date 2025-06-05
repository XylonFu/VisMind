import json
import random

import tiktoken


def count_tokens(texts, encoder_name="cl100k_base"):
    encoder = tiktoken.get_encoding(encoder_name)
    total = 0
    for text in texts:
        tokens = encoder.encode(text, allowed_special={"<|endoftext|>"})
        total += len(tokens)
    return total


def random_sample_jsonl_by_tokens(
        input_file,
        output_file,
        max_tokens,
        encoder_name="cl100k_base",
        random_seed=42
):
    records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError:
                continue

    record_tokens = []
    for record in records:
        contents = []
        for msg in record.get("messages", []):
            content = msg.get("content", "")
            if content:
                contents.append(content)
        token_count = count_tokens(contents, encoder_name) if contents else 0
        record_tokens.append(token_count)

    indices = list(range(len(records)))
    random.seed(random_seed)
    random.shuffle(indices)

    total_tokens = 0
    selected_indices = []
    for idx in indices:
        token_count = record_tokens[idx]
        if total_tokens + token_count > max_tokens and total_tokens > 0:
            break
        selected_indices.append(idx)
        total_tokens += token_count
        if total_tokens >= max_tokens:
            break

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for idx in selected_indices:
            json_line = json.dumps(records[idx], ensure_ascii=False)
            f_out.write(json_line + '\n')

    print(f"Random sampling completed! Total tokens: {total_tokens}, Lines kept: {len(selected_indices)}")


if __name__ == "__main__":
    input_jsonl = "input.jsonl"
    output_jsonl = "output.jsonl"
    max_tokens = 1000000
    random_seed = 42

    random_sample_jsonl_by_tokens(input_jsonl, output_jsonl, max_tokens, random_seed=random_seed)
