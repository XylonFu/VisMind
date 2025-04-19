from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

from config import CONCURRENCY
from processor import process_single_file
from utils.io_utils import INPUT_DIR

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
    main()
