import json
import re
from pathlib import Path
from typing import List, Dict, Any, Pattern

# 全局常量：对话标记列表
MARKERS: List[Pattern] = [
    re.compile(r"#TO_STUDENT_ALPHA#"),
    re.compile(r"#TO_STUDENT_BETA#"),
    re.compile(r"#TO_TEACHER#"),
    re.compile(r"#END_CONVERSATION#"),
]
# 全局常量：仅角色标签的正则，用于过滤空对话
ROLE_ONLY_PATTERN: Pattern = re.compile(r'^(?:system|student_alpha|student_beta|teacher):\s*$')


def clean_text(text: str) -> str:
    """
    去除文本中的特殊对话标记，并去除前后空白。

    :param text: 原始文本
    :return: 清洗后的文本
    """
    for marker in MARKERS:
        text = marker.sub('', text)
    return text.strip()


def extract_system_message(message_block: List[Dict[str, Any]]) -> str:
    """
    从 message.content 列表中提取第一条文本，并添加图片占位符。

    :param message_block: message['content'] 列表
    :return: 格式化后的系统消息字符串
    :raises ValueError: 当找不到文本项时抛出
    """
    for item in message_block:
        if item.get('type') == 'text':
            return f"system: <image> {clean_text(item.get('text', ''))}"
    raise ValueError("未找到 type='text' 的 content 项")


def should_keep_events(events: List[Dict[str, Any]]) -> bool:
    """
    判断 events 列表最后一条是否包含结束对话标记。

    :param events: 原始 events 列表
    :return: True 保留，False 丢弃
    """
    if not events:
        return False
    last_event = events[-1]
    for role_info in last_event.values():
        for msg in role_info.get('messages', []):
            if '#END_CONVERSATION#' in msg.get('content', ''):
                return True
    return False


def extract_event_texts(events: List[Dict[str, Any]]) -> List[str]:
    """
    提取并清洗所有事件消息，过滤空或仅标签的行。

    :param events: 原始 events 列表
    :return: 清洗后有效的消息列表
    """
    texts: List[str] = []
    for evt in events:
        for role, info in evt.items():
            for msg in info.get('messages', []):
                cleaned = clean_text(msg.get('content', ''))
                if cleaned and not ROLE_ONLY_PATTERN.match(cleaned):
                    texts.append(cleaned)
    return texts


def process_file(filepath: Path) -> Dict[str, Any]:
    """
    处理单个 JSON 文件，生成符合要求的记录。

    :param filepath: 文件路径
    :return: 格式化后的记录字典
    :raises ValueError: 当文件不满足条件时抛出
    """
    data = json.loads(filepath.read_text(encoding='utf-8'))

    # 提取系统消息
    system_msg = extract_system_message(data.get('message', {}).get('content', []))

    # 校验并提取事件对话
    events = data.get('events', [])
    if not should_keep_events(events):
        raise ValueError("不满足 END_CONVERSATION 规则，丢弃文件")
    event_texts = extract_event_texts(events)
    if not event_texts:
        raise ValueError("提取后无有效对话，丢弃文件")

    # 合并内容
    combined = ''.join([system_msg] + event_texts)

    # 构造返回记录
    return {
        "messages": [{"role": "assistant", "content": combined}],
        "images": [f"/image/{data.get('id')}.png"]
    }


def prepare_event_dataset(input_dir: str, output_file: str) -> None:
    """
    批量处理目录下所有 JSON 文件，将合格记录写入输出文件，并打印统计信息。

    :param input_dir: 输入目录路径
    :param output_file: 输出 JSON 文件路径
    """
    input_path = Path(input_dir)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    dropped_count = 0

    for filepath in input_path.glob('*.json'):
        try:
            record = process_file(filepath)
            records.append(record)
        except Exception:
            dropped_count += 1

    # 写入文件
    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')

    # 打印处理结果
    print(f"Wrote {len(records)} records to {output_path}")
    print(f"Dropped {dropped_count} files due to filtering rules.")


if __name__ == '__main__':
    prepare_event_dataset(
        input_dir='../output/GeoQAPlus/event-0419',
        output_file='../output/GeoQAPlus/event-0419.json'
    )
