import logging
import os
import signal
import subprocess
import time
from datetime import datetime

import requests

logger = logging.getLogger(__name__)


def start_vllm_server(conda_env_path, model_path, served_model_name,
                      devices=None, tensor_parallel_size=4, max_model_len=16384, max_num_seqs=256,
                      host="127.0.0.1", port=8000, api_key="EMPTY", log_file=None):
    if devices is None:
        devices = [0, 1, 2, 3]
    devices_str = ",".join(str(d) for d in devices)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = devices_str

    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"vllm_server_{timestamp}.log"

    cmd = [
        "conda", "run", "--prefix", os.path.expandvars(conda_env_path), "--no-capture-output",
        "vllm", "serve", model_path,
        "--served-model-name", served_model_name,
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--max-model-len", str(max_model_len),
        "--max-num-seqs", str(max_num_seqs),
        "--host", host,
        "--port", str(port),
        "--api-key", api_key,
    ]

    log_fd = open(log_file, "w")
    logger.info(f"VLLM server logs will be saved to: {os.path.abspath(log_file)}")

    return subprocess.Popen(cmd, env=env, stdout=log_fd, stderr=log_fd)


def wait_server(host="127.0.0.1", port=8000, timeout=600):
    url = f"http://{host}:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url)
            if r.status_code == 200:
                logger.info("Server is up and ready")
                return
        except requests.ConnectionError:
            pass
        time.sleep(10)
    raise RuntimeError(f"Server didn't become ready within {timeout}s")


def stop_server(process):
    logger.info("Stopping server...")
    process.send_signal(signal.SIGTERM)
    process.wait()
