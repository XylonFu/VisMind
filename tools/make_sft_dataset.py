import json
import os
from typing import List

import tiktoken


def count_tokens(texts: List[str], encoder_name: str = "cl100k_base") -> int:
    """Count the total number of tokens in a list of texts"""
    encoder = tiktoken.get_encoding(encoder_name)
    return sum(len(encoder.encode(text, allowed_special={"<|endoftext|>", "<image>"})) for text in texts)


def process_files(jsonl_path, json_dir, output_path, max_length=4096):
    # Read all IDs from the jsonl file
    ids = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                ids.append(str(data['id']))
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line: {line.strip()}")
                continue

    # Collect all required json file paths
    json_files = []
    for filename in os.listdir(json_dir):
        if filename.endswith('.json') and os.path.splitext(filename)[0] in ids:
            json_files.append(os.path.join(json_dir, filename))

    # Process and merge json files
    output_data = []
    skipped_count = 0

    for json_file in json_files:
        with open(json_file, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON file: {json_file}")
                continue

            # Construct user message content
            user_content = " ".join(["<image>"] * len(data["image"])) + " " + data["question"]
            assistant_content = data["answer"]

            # Calculate token count
            try:
                total_tokens = count_tokens([user_content, assistant_content])
            except Exception as e:
                print(f"Error counting tokens ({json_file}): {str(e)}")
                continue

            # Check if token count exceeds limit
            if total_tokens > max_length:
                skipped_count += 1
                continue

            # Construct new data structure
            new_entry = {
                "id": data["idx"],
                "images": data["image"],
                "messages": [
                    {
                        "role": "user",
                        "content": user_content
                    },
                    {
                        "role": "assistant",
                        "content": assistant_content
                    }
                ]
            }
            output_data.append(new_entry)

    # Write to new jsonl file
    with open(output_path, 'w') as f:
        for entry in output_data:
            f.write(json.dumps(entry) + '\n')

    print(
        f"Processing completed: Processed {len(output_data)} entries, skipped {skipped_count} entries exceeding length limit")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    jsonl_path = '../output/VisualWebInstruct118K/pt-event-0608-01-2300.jsonl'
    json_dir = '../input/VisualWebInstruct118K/json'
    output_path = '../output/VisualWebInstruct118K/sft-pairs-0608-01-2300.jsonl'

    process_files(jsonl_path, json_dir, output_path, max_length=4096)
