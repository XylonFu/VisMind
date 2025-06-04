import concurrent.futures
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Pattern, Tuple

# Special markers to clean from text
MARKERS: List[Pattern] = [
    re.compile(r"#TO_STUDENT_ALPHA#"),
    re.compile(r"#TO_STUDENT_BETA#"),
    re.compile(r"#TO_TEACHER#"),
    re.compile(r"#END_CONVERSATION#"),
]

# Pattern to detect role-only lines (e.g., "system:")
ROLE_ONLY_PATTERN: Pattern = re.compile(r'^(?:system|student_alpha|student_beta|teacher):\s*$')

# Disallowed language character ranges (non-English/math/Greek/punctuation)
DISALLOWED_PATTERN = re.compile(
    r'['
    r'\u0400-\u04FF'  # 西里尔字母
    r'\u0590-\u05FF'  # 希伯来语
    r'\u0600-\u06FF'  # 阿拉伯语
    # 印度语系
    r'\u0900-\u097F'  # 天城文
    r'\u0980-\u09FF'  # 孟加拉语
    r'\u0A80-\u0AFF'  # 古吉拉特语
    r'\u0B00-\u0B7F'  # 奥里亚语
    r'\u0B80-\u0BFF'  # 泰米尔语
    r'\u0C00-\u0C7F'  # 泰卢固语
    r'\u0C80-\u0CFF'  # 卡纳达语
    r'\u0D00-\u0D7F'  # 马拉雅拉姆语
    r'\u0D80-\u0DFF'  # 僧伽罗语
    # 东南亚语系
    r'\u0E00-\u0E7F'  # 泰语
    r'\u0E80-\u0EFF'  # 老挝语
    r'\u0F00-\u0FFF'  # 藏语
    r'\u1000-\u109F'  # 缅甸语
    r'\u1780-\u17FF'  # 高棉语
    # 其他文字
    r'\u10A0-\u10FF'  # 格鲁吉亚语
    r'\u1100-\u11FF'  # 韩文字母
    r'\u1200-\u137F'  # 埃塞俄比亚语
    r'\u1400-\u167F'  # 加拿大原住民文字
    r'\u1800-\u18AF'  # 蒙古语
    # 日语
    r'\u3040-\u309F'  # 平假名
    r'\u30A0-\u30FF'  # 片假名
    # 中文
    r'\u3400-\u4DBF'  # 中文扩展
    r'\u4E00-\u9FFF'  # 中文基本汉字
    # 其他
    r'\uA980-\uA9DF'  # 爪哇语
    r'\uAC00-\uD7AF'  # 韩文字节
    r']'
)


def clean_text(text: str) -> str:
    """Remove special markers and trim whitespace"""
    for marker in MARKERS:
        text = marker.sub('', text)
    return text.strip()


def extract_system_message(message_block: List[Dict[str, Any]]) -> str:
    """Extract system message from content block"""
    for item in message_block:
        if item.get('type') == 'text':
            text = clean_text(item.get('text', ''))
            # Add period if ending with parenthesis
            if text.endswith((')', '）')):
                text += '.'
            return f"system: <image> {text}"
    raise ValueError("No valid text content found")


def contains_disallowed_language(text: str) -> bool:
    """Check if text contains disallowed language characters"""
    return bool(DISALLOWED_PATTERN.search(text))


def should_keep_events(events: List[Dict[str, Any]]) -> bool:
    """Determine if event sequence meets retention criteria"""
    if not events:
        return False

    # Check for disallowed languages in all events
    for event in events:
        for role_info in event.values():
            for msg in role_info.get('messages', []):
                if contains_disallowed_language(msg.get('content', '')):
                    return False

    # Check for end marker in last event
    last_event = events[-1]
    for role_info in last_event.values():
        for msg in role_info.get('messages', []):
            if '#END_CONVERSATION#' in msg.get('content', ''):
                return True

    return False


def extract_event_texts(events: List[Dict[str, Any]]) -> List[str]:
    """Extract and clean text content from event sequence"""
    texts: List[str] = []
    for evt in events:
        for role, info in evt.items():
            for msg in info.get('messages', []):
                cleaned = clean_text(msg.get('content', ''))
                # Skip role-only lines (e.g., "teacher:")
                if cleaned and not ROLE_ONLY_PATTERN.match(cleaned):
                    texts.append(cleaned)
    return texts


def process_file(filepath: Path) -> Dict[str, Any]:
    """Process single JSON file and extract required data"""
    # Load JSON data
    data = json.loads(filepath.read_text(encoding='utf-8'))

    # Extract system and ground truth messages
    system_msg = extract_system_message(data.get('message', {}).get('content', []))

    # Validate and process events
    events = data.get('events', [])
    if not should_keep_events(events):
        raise ValueError("File rejected by filter rules")

    event_texts = extract_event_texts(events)
    if not event_texts:
        raise ValueError("No valid conversation after extraction")

    # Combine all messages
    combined = '\n'.join([system_msg] + event_texts)

    # Process image path
    image_path = data.get('image_path', '')
    if image_path.startswith('image/'):
        image_path = image_path[6:]

    # Return formatted result
    return {
        "id": data.get('id', ''),
        "messages": [{"role": "assistant", "content": combined.strip()}],
        "images": [image_path]
    }


def prepare_event_dataset(input_dir: str, output_file: str, concurrency: int = 256) -> Tuple[int, int]:
    """Process all JSON files in directory and generate JSONL dataset"""
    input_path = Path(input_dir)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    dropped_count = 0
    filepaths = list(input_path.glob('*.json'))

    print(f"Processing {len(filepaths)} files from: {input_dir}")
    print(f"Writing output to: {output_file}")

    with output_path.open('w', encoding='utf-8') as fout:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(process_file, fp): fp for fp in filepaths}
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                fp = futures[future]
                try:
                    result = future.result()
                    fout.write(json.dumps(result, ensure_ascii=False) + '\n')
                    processed_count += 1
                    if i % 1000 == 0:
                        print(f"Processed {i}/{len(filepaths)} files...")
                except Exception as e:
                    dropped_count += 1
                    # Uncomment for detailed error logging
                    # print(f"Error processing {fp.name}: {str(e)}")

    print(f"Processing complete: {processed_count} files processed, {dropped_count} files dropped")
    print(f"Output saved to: {output_path}")
    return processed_count, dropped_count


if __name__ == '__main__':
    # Process specified data partitions
    for i in range(1, 2):
        input_dir = f'../output/GLLaVA70K/event-0603-0{i}'
        output_file = f'../output/GLLaVA70K/pt-event-0603-0{i}.jsonl'

        print(f"\n{'=' * 50}")
        print(f"Processing partition {i}: {input_dir}")
        processed, dropped = prepare_event_dataset(
            input_dir=input_dir,
            output_file=output_file,
            concurrency=1024
        )
        print(f"Partition {i} result: {processed} processed, {dropped} dropped")
        print(f"{'=' * 50}\n")
