import json
import os


def deduplicate_jsonl(input_file):
    # 生成输出文件名
    base_name = os.path.splitext(input_file)[0]
    output_file = f"{base_name}-deduped.jsonl"

    seen_keys = set()  # 使用set存储已见键值
    deduplicated_records = []  # 存储去重后的记录

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                images = record.get('images', [])
                key = images[0] if images else None

                if key not in seen_keys:
                    seen_keys.add(key)
                    deduplicated_records.append(record)
            except (json.JSONDecodeError, IndexError, KeyError) as e:
                print(f"警告: 跳过格式错误的行 - {e}")

    # 写入去重后的数据
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for record in deduplicated_records:
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')

    return output_file


if __name__ == "__main__":
    input_filename = "your_file.jsonl"
    output_filename = deduplicate_jsonl(input_filename)
    print(f"去重完成! 结果已保存至: {output_filename}")
