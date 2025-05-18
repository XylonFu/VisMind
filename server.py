import os
import signal
import subprocess


def start_vllm_server(model_path, served_model_name,
                      devices=None, tensor_parallel_size=4, max_model_len=16384):
    if devices is None:
        devices = [0, 1, 2, 3]
    devices_str = ",".join(str(d) for d in devices)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = devices_str
    cmd = [
        "vllm", "serve", model_path,
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--max-model-len", str(max_model_len),
        "--served-model-name", served_model_name
    ]
    return subprocess.Popen(cmd, env=env)


def stop_vllm_server(process):
    process.send_signal(signal.SIGTERM)
    process.wait()
