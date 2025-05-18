import os
import signal
import subprocess
import time

import requests


def start_vllm_server(conda_env_path, model_path, served_model_name,
                      api_key, host="127.0.0.1", port=8000,
                      devices=None, tensor_parallel_size=4, max_model_len=16384):
    if devices is None:
        devices = [0, 1, 2, 3]
    devices_str = ",".join(str(d) for d in devices)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = devices_str
    cmd = [
        "conda", "run", "--prefix", os.path.expandvars(conda_env_path), "--no-capture-output",
        "vllm", "serve", model_path,
        "--served-model-name", served_model_name,
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--max-model-len", str(max_model_len),
        "--host", host,
        "--port", str(port),
        "--api-key", api_key,
    ]
    return subprocess.Popen(cmd, env=env)


def wait_server(host="127.0.0.1", port=8000, timeout=600):
    url = f"http://{host}:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url)
            if r.status_code == 200:
                print("Server is up!")
                return
        except requests.ConnectionError:
            pass
        time.sleep(10)
    raise RuntimeError(f"Server didn’t become ready within {timeout}s")


def stop_server(process):
    process.send_signal(signal.SIGTERM)
    process.wait()
