import json
import os
import random


def deduplicate_jsonl(input_file, random_seed=42):
    base_name = os.path.splitext(input_file)[0]
    output_file = f"{base_name}-deduped.jsonl"

    random.seed(random_seed)
    records_by_key = {}
    key_order = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                images = record.get('images', [])
                key = images[0] if images else None

                if key is None:
                    continue

                if key not in records_by_key:
                    records_by_key[key] = []
                    key_order.append(key)

                records_by_key[key].append(record)
            except (json.JSONDecodeError, IndexError, KeyError):
                continue

    deduplicated_records = []
    for key in key_order:
        if records := records_by_key[key]:
            deduplicated_records.append(random.choice(records))

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for record in deduplicated_records:
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')

    return output_file


if __name__ == "__main__":
    input_filename = "your_file.jsonl"
    output_filename = deduplicate_jsonl(input_filename)
    print(f"Deduplication completed! Results saved to: {output_filename}")
