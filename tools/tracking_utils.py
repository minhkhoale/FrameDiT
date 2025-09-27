from pathlib import Path
import time
import os
from typing import Optional
import subprocess
import torch


def file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return -1

def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def ensure_dirs(root: str) -> None:
    # mkdir root
    Path(root).mkdir(parents=True, exist_ok=True)
    # state ilfes
    Path(root, 'processed_checkpoints.txt').touch()


def wait_for_stable_file(path: Path, stable_wait: int = 15) -> None:
    last = -1
    while True:
        cur = file_size(path)
        if cur >= 0 and cur == last:
            return
        last = cur
        time.sleep(stable_wait)


def extract_step_from_ckpt(ckpt_path: Path) -> int:
    """Extract consecutive digits from the stem; returns -1 if not found."""
    digits = "".join(ch for ch in ckpt_path.stem if ch.isdigit())
    try:
        return int(digits)
    except ValueError:
        return -1
    

def already_processed(path: Path, state_file: Path) -> bool:
    try:
        with state_file.open("r") as f:
            for line in f:
                if line.strip() == str(path):
                    return True
    except FileNotFoundError:
        pass
    return False

def mark_processed(path: Path, state_file: Path) -> None:
    with state_file.open("a") as f:
        f.write(str(path) + "\n")



def run_cmd(cmd: list[str], log_file: Path, env: Optional[dict] = None) -> int:
    with log_file.open("a", buffering=1) as lf:  # line-buffered
        lf.write(f"\n[cmd] {' '.join(map(str, cmd))}\n")
        lf.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=lf,
            stderr=subprocess.STDOUT,
            env=env or os.environ.copy(),
        )
        return proc.wait()
    

def get_real_data_path(dataset_name: str) -> str:
    match dataset_name:
        case 'taichi128':
            return '/scratch/s224075134/temporal_diffusion/datasets/video_for_metrics/taichi128_reconstruction/train'
        case _:
            raise ValueError(f"Unknown dataset name: {dataset_name}")
        

def get_available_gpus():
    gpu_list = []
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            gpu_list.append(i)
    return gpu_list