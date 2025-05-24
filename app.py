import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import sleep

from config import (CONCURRENCY, CONDA_ENV_PATH,
                    TEACHER_MODEL_PATH, TEACHER_MODEL_NAME, TEACHER_MODEL_KEYS)
from processor import process_single_file
from server import start_vllm_server, stop_server, wait_server


def main(input_dir: Path, output_dir: Path):
    json_path = input_dir / "json"
    json_files = list(json_path.glob("*.json"))
    print(f"📂 共发现 {len(json_files)} 个待处理文件")

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = {executor.submit(process_single_file, file, input_dir, output_dir): file.name for file in json_files}
        for future in as_completed(futures):
            file_name = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"❌ {file_name} 执行出错: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process GeoQAPlus data.")
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    teacher_server = start_vllm_server(CONDA_ENV_PATH, TEACHER_MODEL_PATH, TEACHER_MODEL_NAME,
                                       api_key=TEACHER_MODEL_KEYS)
    try:
        wait_server()
        main(input_dir, output_dir)
    finally:
        stop_server(teacher_server)
        sleep(60)
