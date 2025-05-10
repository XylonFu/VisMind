import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

from config import CONCURRENCY
from processor import process_single_file

load_dotenv()


def main():
    json_path = INPUT_DIR / "json"
    json_files = list(json_path.glob("*.json"))
    print(f"📂 共发现 {len(json_files)} 个待处理文件")

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = {executor.submit(process_single_file, file): file.name for file in json_files}
        for future in as_completed(futures):
            file_name = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"❌ {file_name} 执行出错: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process GeoQAPlus data.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(Path(__file__).parent / "input/GeoQAPlus"),
        help="Path to input directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).parent / "output/GeoQAPlus/event-0510"),
        help="Path to output directory"
    )
    args = parser.parse_args()

    INPUT_DIR = Path(args.input_dir)
    OUTPUT_DIR = Path(args.output_dir)

    main()
