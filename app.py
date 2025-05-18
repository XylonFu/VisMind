import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

from config import CONCURRENCY, DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR, CONDA_ENV_PATH, MODEL_PATH, SERVED_MODEL_NAME
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
    load_dotenv()

    parser = argparse.ArgumentParser(description="Process GeoQAPlus data.")
    parser.add_argument("--input_dir", type=str, default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    vllm_server = start_vllm_server(CONDA_ENV_PATH, MODEL_PATH, SERVED_MODEL_NAME, os.getenv("OPENAI_API_KEY"))
    wait_server()
    main(input_dir, output_dir)
    stop_server(vllm_server)
