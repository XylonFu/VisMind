import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import sleep

from config import (CONCURRENCY, CONDA_ENV_PATH,
                    TEACHER_MODEL_PATH, TEACHER_MODEL_NAME, TEACHER_MODEL_KEYS)
from processor import process_single_file
from server import start_vllm_server, stop_server, wait_server
from utils.io_utils import output_exists

logger = logging.getLogger(__name__)


def main(input_dir: Path, output_dir: Path, json_folder: str):
    json_path = input_dir / json_folder
    json_files = list(json_path.glob("*.json"))
    logger.info(f"Found {len(json_files)} files to process")

    unprocessed_files = [f for f in json_files if not output_exists(f.stem, output_dir)]
    logger.info(f"Processing {len(unprocessed_files)} unprocessed files")

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = {
            executor.submit(process_single_file, file, input_dir, output_dir): file.name
            for file in unprocessed_files
        }

        for future in as_completed(futures):
            file_name = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error processing {file_name}: {str(e)}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

    my_modules = ["__main__", "app", "processor", "agents", "utils", "server"]
    for module in my_modules:
        logging.getLogger(module).setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--json_folder", type=str, required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    teacher_log = "teacher_vllm_server.log"
    teacher_server = start_vllm_server(CONDA_ENV_PATH, TEACHER_MODEL_PATH, TEACHER_MODEL_NAME,
                                       devices=[0, 1], tensor_parallel_size=2,
                                       api_key=TEACHER_MODEL_KEYS, log_file=teacher_log)

    try:
        wait_server()
        main(input_dir, output_dir, args.json_folder)
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
    finally:
        stop_server(teacher_server)
        sleep(30)
