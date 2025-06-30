import concurrent.futures
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Pattern, Tuple, Optional

# ======================
# CONSTANT DEFINITIONS
# ======================

# Special markers to clean from text
MARKERS: List[Pattern] = [
    re.compile(r"#TO_STUDENT_ALPHA#"),
    re.compile(r"#TO_STUDENT_BETA#"),
    re.compile(r"#TO_TEACHER#"),
    re.compile(r"#END_CONVERSATION#"),
]

# Pattern to detect role-only lines (e.g., "system:")
ROLE_ONLY_PATTERN: Pattern = re.compile(r'^(?:system|student_alpha|student_beta|teacher):\s*$')

# Pattern to detect triple word repetition (case-insensitive)
TRIPLE_WORD_PATTERN = re.compile(
    r'\b(?:'
    r'(\w+)(?:\W+\1\W+){2,}'
    r'|'
    r'(\w+\s+\w+)(?:\W+\2\W+){2,}'
    r'|'
    r'(\w+\s+\w+\s+\w+)(?:\W+\3\W+){2,}'
    r')\b',
    re.IGNORECASE
)

# Disallowed language character ranges (non-English/math/Greek/punctuation)
DISALLOWED_PATTERN = re.compile(
    r'['
    # Indian scripts
    r'\u0900-\u097F'  # Devanagari
    r'\u0980-\u09FF'  # Bengali
    r'\u0A80-\u0AFF'  # Gujarati
    r'\u0B00-\u0B7F'  # Oriya
    r'\u0B80-\u0BFF'  # Tamil
    r'\u0C00-\u0C7F'  # Telugu
    r'\u0C80-\u0CFF'  # Kannada
    r'\u0D00-\u0D7F'  # Malayalam
    r'\u0D80-\u0DFF'  # Sinhala
    # Southeast Asian scripts
    r'\u0E00-\u0E7F'  # Thai
    r'\u0E80-\u0EFF'  # Lao
    r'\u0F00-\u0FFF'  # Tibetan
    r'\u1000-\u109F'  # Burmese
    r'\u1780-\u17FF'  # Khmer
    # Japanese
    r'\u3040-\u309F'  # Hiragana
    r'\u30A0-\u30FF'  # Katakana
    # Chinese
    r'\u3400-\u4DBF'  # CJK Unified Ideographs Extension A
    r'\u4E00-\u9FFF'  # CJK Unified Ideographs
    # Others
    r'\uA980-\uA9DF'  # Javanese
    r'\uAC00-\uD7AF'  # Hangul Syllables
    r'\u0400-\u04FF'  # Cyrillic
    r'\u0590-\u05FF'  # Hebrew
    r'\u0600-\u06FF'  # Arabic
    r'\u10A0-\u10FF'  # Georgian
    r'\u1100-\u11FF'  # Hangul Jamo
    r'\u1200-\u137F'  # Ethiopic
    r'\u1400-\u167F'  # Unified Canadian Aboriginal Syllabics
    r'\u1800-\u18AF'  # Mongolian
    r']'
)


# ======================
# UTILITY FUNCTIONS
# ======================

def clean_text(text: str) -> str:
    """Remove special markers and trim whitespace"""
    for marker in MARKERS:
        text = marker.sub('', text)
    return text.strip()


def contains_disallowed_language(text: str) -> bool:
    """Check if text contains disallowed language characters"""
    return bool(DISALLOWED_PATTERN.search(text))


def contains_triple_repetition(text: str) -> bool:
    """Check if text contains triple word repetition"""
    return bool(TRIPLE_WORD_PATTERN.search(text))


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


# ======================
# CORE PROCESSING FUNCTIONS
# ======================

def extract_system_message(message_block: List[Dict[str, Any]], image_count: int) -> str:
    """Extract system message from content block with image tags"""
    for item in message_block:
        if item.get('type') == 'text':
            text = clean_text(item.get('text', ''))
            # Add period if ending with parenthesis
            if text.endswith((')', '）')):
                text += '.'

            image_tags = " ".join(["<image>"] * image_count)
            return f"system: {image_tags} {text}" if image_tags else f"system: {text}"
    raise ValueError("No valid text content found")


def should_keep_events(events: List[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    """Determine if event sequence meets retention criteria"""
    if not events:
        return False, "empty events"

    # Check for disallowed languages in all events
    for event in events:
        for role_info in event.values():
            for msg in role_info.get('messages', []):
                if contains_disallowed_language(msg.get('content', '')):
                    return False, "disallowed language"

    # Check for end marker in last event
    last_event = events[-1]
    end_marker_found = False
    for role_info in last_event.values():
        for msg in role_info.get('messages', []):
            if '#END_CONVERSATION#' in msg.get('content', ''):
                end_marker_found = True
                break
        if end_marker_found:
            break

    if not end_marker_found:
        return False, "missing end marker"

    # Extract event texts for repetition check
    texts = extract_event_texts(events)
    combined_text = " ".join(texts)

    # Check for triple word repetition
    if contains_triple_repetition(combined_text):
        return False, "triple word repetition"

    return True, None


def process_file(filepath: Path) -> Dict[str, Any]:
    """Process single JSON file and extract required data"""
    # Load JSON data
    data = json.loads(filepath.read_text(encoding='utf-8'))

    # Handle image paths (prioritize image_paths, fallback to image_path)
    image_paths: List[str] = []

    # Case 1: image_paths field exists (array)
    if "image_paths" in data and isinstance(data["image_paths"], list):
        image_paths = [p[6:] if p.startswith("image/") else p
                       for p in data["image_paths"] if p.strip()]

    # Case 2: no image_paths but has image_path field (string)
    elif "image_path" in data and data["image_path"]:
        path_str = data["image_path"]
        if isinstance(path_str, list):
            # Handle unexpected array case
            image_paths = [p[6:] if p.startswith("image/") else p
                           for p in path_str if p.strip()]
        else:
            # Handle single path
            p = path_str[6:] if path_str.startswith("image/") else path_str
            if p.strip():
                image_paths = [p]

    # Extract system message (pass image count)
    system_msg = extract_system_message(
        data.get('message', {}).get('content', []),
        image_count=len(image_paths)
    )

    # Validate and process events
    events = data.get('events', [])
    keep, reason = should_keep_events(events)
    if not keep:
        raise ValueError(f"rejected: {reason}")

    event_texts = extract_event_texts(events)
    if not event_texts:
        raise ValueError("no valid conversation after extraction")

    # Combine all messages
    combined = '\n'.join([system_msg] + event_texts)

    # Return formatted result
    return {
        "id": data.get('id', ''),
        "images": image_paths,
        "messages": [{"role": "assistant", "content": combined.strip()}]
    }


# ======================
# MAIN PROCESSING FUNCTION
# ======================

def prepare_event_dataset(input_dir: str, output_file: str, concurrency: int = 256) -> Tuple[int, int, Dict[str, int]]:
    """Process all JSON files in directory and generate JSONL dataset"""
    input_path = Path(input_dir)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    dropped_count = 0
    drop_reasons = defaultdict(int)
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
                    error_msg = str(e)
                    # Extract reason from error message
                    if error_msg.startswith("rejected: "):
                        reason = error_msg.split("rejected: ", 1)[1]
                        drop_reasons[reason] += 1
                    else:
                        drop_reasons["other"] += 1
                    # Uncomment for detailed error logging
                    # print(f"Error processing {fp.name}: {error_msg}")

    print(f"Processing complete: {processed_count} files processed, {dropped_count} files dropped")
    print("Drop reasons:")
    for reason, count in drop_reasons.items():
        print(f"  - {reason}: {count}")
    print(f"Output saved to: {output_path}")
    return processed_count, dropped_count, drop_reasons


# ======================
# MAIN ENTRY POINT
# ======================

if __name__ == '__main__':
    # Process specified data partitions
    for i in range(1, 2):
        input_dir = f'../output/VisualWebInstruct118K/pt-event-0608-0{i}'
        output_file = f'../output/VisualWebInstruct118K/pt-event-0608-0{i}.jsonl'

        print(f"\n{'=' * 50}")
        print(f"Processing partition {i}: {input_dir}")
        processed, dropped, drop_reasons = prepare_event_dataset(
            input_dir=input_dir,
            output_file=output_file,
            concurrency=1024
        )
        print(f"Partition {i} result: {processed} processed, {dropped} dropped")
        print(f"Drop reasons: {dict(drop_reasons)}")
        print(f"{'=' * 50}\n")
