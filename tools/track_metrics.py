#!/usr/bin/env python3
import argparse
import os
import time
import watchdog
import sys
import signal
import subprocess
from pathlib import Path
from typing import Optional
from omegaconf import OmegaConf
from dataclasses import dataclass
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from .tracking_utils import (
    run_cmd,
    wait_for_stable_file,
    already_processed,
    mark_processed,
    extract_step_from_ckpt,
    get_real_data_path,
    get_available_gpus,
    get_real_data_sample_factor
)
import wandb
# ==============================
# CONFIG — EDIT THESE
# ==============================
SAMPLE_DIFFERENCE_ENTRY = "sample/sample_difference_ddp.py"
SAMPLE_ENTRY = "sample/sample_ddp.py"
METRICS_SCRIPT = "tools/my_cal_metrics_for_dataset.py"
STATE_FILE = 'processed_checkpoints.txt'  # tracks processed checkpoints
# ==============================
@dataclass(frozen=True)
class RunEnv:
    cuda_visible_devices: str
    master_port: Optional[int] = None

    def as_environ(self) -> dict:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        if self.master_port is not None:
            env["MASTER_PORT"] = str(self.master_port)
        return env


@dataclass(frozen=True)
class Paths:
    experiment_dir: Path               # path to experiment dir (contains checkpoints/ and config.yaml)
    generation_dir: Path               # base output for generations/logs/state (we create <dataset>/<exp>/<method_steps>)
    config_path: Path                  # experiment_dir / "config.yaml"
    ckpt_dir: Path                     # experiment_dir / "checkpoints"
    logs_dir: Path                     # generation_dir / "logs"
    metrics_dir: Path                   # generation_dir / "metrics"
    state_file: Path                   # generation_dir / STATE_FILE_NAME
    dataset_name: str = ""            # derived from config.yaml

    @staticmethod
    def build(experiment_dir: Path, generation_root: Path, sample_method: str, num_sampling_steps: int) -> "Paths":
        config_path = experiment_dir / "config.yaml"
        # Load config to decide dataset folder layout
        cfg = OmegaConf.load(config_path)
        dataset_name = f"{cfg.dataset}{cfg.image_size}"

        experiment_name = experiment_dir.name
        generation_dir = generation_root / dataset_name / experiment_name / f"{sample_method}_{num_sampling_steps}"
        ckpt_dir = experiment_dir / "checkpoints"
        logs_dir = generation_dir / "logs"
        metrics_dir = generation_dir / "metrics"
        state_file = generation_dir / STATE_FILE
        return Paths(experiment_dir, generation_dir, config_path, ckpt_dir, logs_dir, metrics_dir, state_file, dataset_name)
    
    def args(self) -> list[str]:
        cfg = OmegaConf.load(self.config_path)
        return cfg

    def ensure_dirs(self) -> None:
        self.generation_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.state_file.touch(exist_ok=True)


@dataclass(frozen=True)
class SamplerArgs:
    is_difference: bool
    config_path: Path
    use_fp16: bool
    seed: int
    sample_method: str
    num_sampling_steps: int
    cfg_scale: float
    negative_name: str
    batch_size: int
    num_fvd_samples: int
    fps: int
    video_quality: int
    real_sample_factor: int


def run_sampling(
    env: RunEnv,
    log_file: Path,
    ckpt_path: Path,
    save_video_path: Path,
    args: SamplerArgs,
):
    sampler = SAMPLE_DIFFERENCE_ENTRY if args.is_difference else SAMPLE_ENTRY

    # check if output dir already exists, and num_fvd_samples videos are present, then skip
    existing_videos = list(save_video_path.glob("*.mp4"))
    if len(existing_videos) >= args.num_fvd_samples:
        with log_file.open("a") as lf:
            lf.write(f"[info] Output already exists, skipping: {save_video_path} with {len(existing_videos)} videos\n")
        return 0

    cmd = [
        "torchrun",
        "--nnodes=1",
        f"--nproc_per_node={env.cuda_visible_devices.count(',') + 1}",
        f"--master_port={env.master_port}" if env.master_port is not None else "",
        sampler,
        "--config", str(args.config_path),
        "--ckpt", str(ckpt_path),
        "--save_video_path", str(save_video_path),
        "--use-fp16", str(args.use_fp16),
        "--seed", str(args.seed),
        "--sample-method", args.sample_method,
        "--num-sampling-steps", str(args.num_sampling_steps),
        "--cfg-scale", str(args.cfg_scale),
        "--negative-name", args.negative_name,
        "--batch-size", str(args.batch_size),
        "--num-fvd-samples", str(args.num_fvd_samples),
        "--fps", str(args.fps),
        "--video-quality", str(args.video_quality),
    ]

    rc = run_cmd(cmd, log_file, env=env.as_environ())
    if rc != 0:
        with log_file.open("a") as lf:
            lf.write(f"[error] Sampler failed (rc={rc}) for {ckpt_path.name}\n")
    else:
        with log_file.open("a") as lf:
            lf.write(f"[ok] Completed {ckpt_path.name} | outputs: {save_video_path} | logs: {log_file}\n")
    return rc

def run_metrics(
    env: RunEnv,
    log_file: Path,
    real_data_path: Path,
    fake_data_path: Path,
    result_file: Path,
    resolution: int = 128,
    real_sample_factor: int = 1,
):
    metrics_cmd = [
        sys.executable, METRICS_SCRIPT,
        "--real_data_path", real_data_path,
        "--fake_data_path", fake_data_path,
        "--resolution", str(resolution),
        "--result_file", result_file,
        '--verbose',
        '--real-sample-factor', str(real_sample_factor)
    ]
    # check if result_file exists, return immediately if so
    if result_file.exists():
        with log_file.open("a") as lf:
            lf.write(f"[info] Metrics already exist, skipping: {result_file}\n")
        return 0

    rc = run_cmd(metrics_cmd, log_file, env=env.as_environ())
    if rc != 0:
        with log_file.open("a") as lf:
            lf.write(f"[error] Metrics failed (rc={rc}) for {real_data_path}\n")
    else:
        with log_file.open("a") as lf:
            lf.write(f"[ok] Completed {real_data_path} | outputs: {fake_data_path} | logs: {log_file}\n")
    return rc

def write_metrics_to_wandb(metrics_file: Path, step: int) -> None:
    if not metrics_file.exists():
        print(f"[warn] Metrics file does not exist: {metrics_file}")
        return
    import json
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    if not wandb.run:
        print("[warn] No active W&B run. Skipping logging metrics.")
        return

    final_metrics = {}
    for k, v in metrics.items():
        k = k.replace('final/VideoMetricType.', '').replace('VBENCH/VBenchDimensionType.', 'VBENCH_')
        final_metrics[f'metrics/{k}'] = v

    print('final_metrics', final_metrics)
    # define metrics if not already defined, with custom step since default steps must be monotonically increasing, and I want to log to a existing run
    for k, v in final_metrics.items():
        wandb.define_metric(k, step_metric="n_step")
        wandb.log({k: v, "n_step": step})

    print(f"[wandb] Logged metrics for step {step} from {metrics_file}")

def process_checkpoint(
    env: RunEnv,
    paths: Paths,
    ckpt_path: Path,
    real_data_path: Path,
    sampler_args: SamplerArgs,
    metrics_resolution: int = 128,
) -> None:
    step = int(extract_step_from_ckpt(ckpt_path))
    # step with 7 digits
    out_dir = paths.generation_dir / f"{step:07d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = paths.logs_dir / f"{step:07d}.log"
    metrics_file = paths.metrics_dir / f"metrics_{step:07d}.json"

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with log_file.open("a") as lf:
        lf.write(f"===== [{ts}] Processing {ckpt_path.name} (step={step}) =====\n")
        lf.flush()

    # Ensure file finished writing
    with log_file.open("a") as lf:
        lf.write(f"[info] Waiting for file to become stable: {ckpt_path}\n")
    wait_for_stable_file(ckpt_path)

    rc = run_sampling(
        env=env,
        log_file=log_file,
        ckpt_path=ckpt_path,
        save_video_path=out_dir,
        args=sampler_args,
    )

    if rc != 0:
        return
    
    rc = run_metrics(
        env=env,
        log_file=log_file,
        real_data_path=real_data_path,
        fake_data_path=out_dir,
        resolution=metrics_resolution,
        result_file=metrics_file,
        real_sample_factor=sampler_args.real_sample_factor,
    )
    if rc != 0:
        return
    
    # write to wandb
    # write_metrics_to_wandb(metrics_file, step)

    mark_processed(ckpt_path, state_file=paths.state_file)
    with log_file.open("a") as lf:
        lf.write(f"[ok] Completed end-to-end for {ckpt_path.name}\n")


def initial_scan(env: RunEnv, paths: Paths, real_data_path: Path, sampler_args: SamplerArgs, resolution: int, reverse=False, frequency=10000) -> None:
    if not paths.ckpt_dir.exists():
        print(f"[warn] Checkpoints dir does not exist: {paths.ckpt_dir}")
        return

    ckpts = sorted(paths.ckpt_dir.glob("*.pt"), reverse=reverse)
    print('[info] Found checkpoints:', [f.name for f in ckpts])
    for f in ckpts:
        ckpt_step = extract_step_from_ckpt(f)
        if ckpt_step is None or ckpt_step % frequency != 0:
            print(f"[info] Skipping checkpoint (invalid step or frequency): {f.name}")
            continue
        if not already_processed(f, paths.state_file):
            print(f"[info] Processing existing checkpoint: {f.name}")
            process_checkpoint(env, paths, f, real_data_path, sampler_args, metrics_resolution=resolution)


def watch_checkpoints(env: RunEnv, paths: Paths, real_data_path: Path, sampler_args: SamplerArgs) -> None:
    print(f"[watch] Using watchdog on {paths.ckpt_dir}")

    class Handler(FileSystemEventHandler):
        def _try_process(self, p: Path) -> None:
            if p.suffix == ".pt" and not already_processed(p, paths.state_file):
                process_checkpoint(env, paths, p, real_data_path, sampler_args)
                
        def on_created(self, event):
            if not event.is_directory:
                self._try_process(Path(event.src_path))

        # Catch moved/closed files as well (some trainers write temp then move)
        def on_moved(self, event):
            if not event.is_directory:
                self._try_process(Path(event.dest_path))

        def on_modified(self, event):
            if not event.is_directory:
                # Let process_checkpoint's stability check handle partial writes.
                self._try_process(Path(event.src_path))

    observer = Observer()
    observer.schedule(Handler(), str(paths.ckpt_dir), recursive=False)
    observer.start()

    stop = False

    def _signal_handler(sig, frame):
        nonlocal stop
        print(f"\n[info] Caught signal {sig}. Stopping watcher...")
        stop = True

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        while not stop:
            time.sleep(1.0)
    finally:
        observer.stop()
        observer.join()
        print("[info] Watcher exited.")

def prepare_dir(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / 'logs').mkdir(parents=True, exist_ok=True)
    (root / STATE_FILE).touch()



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-dir', type=str, help='Path to the experiment directory containing checkpoints')
    parser.add_argument('--generation-dir', type=str, help='Path to the generation directory containing outputs', default='./generation')
    parser.add_argument('--use-fp16', type=bool, help='Whether to use fp16 sampling', default=False)
    parser.add_argument('--seed', type=int, help='Random seed for sampling', default=0)
    parser.add_argument('--sample-method', type=str, help='Sampling method', default='ddpm')
    parser.add_argument('--num-sampling-steps', type=int, help='Number of sampling steps', default=50)
    parser.add_argument('--cfg-scale', type=float, help='Classifier-free guidance scale', default=1.0)
    parser.add_argument('--negative-name', type=str, help='Negative prompt name', default='')
    parser.add_argument('--batch-size', type=int, help='Batch size for sampling', default=16)
    parser.add_argument('--num-fvd-samples', type=int, help='Number of samples for FVD', default=2048)
    parser.add_argument('--ckpt', type=str, help='Path to the checkpoint file', default=None)
    parser.add_argument('--fps', type=int, help='Frames per second for video', default=8)
    parser.add_argument('--video-quality', type=int, help='Quality for video encoding (1-10)', default=9)
    parser.add_argument('--wandb-run-id', type=str, help='W&B run ID for logging', default='')
    parser.add_argument('--reverse', action='store_true', help='Process existing checkpoints in reverse order')
    parser.add_argument('--frequency', type=int, help='Checkpoint frequency (unused)', default=10000)
    parser.add_argument('--resolution', type=int, help='Resolution for metrics computation', default=128)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    paths = Paths.build(
        experiment_dir=Path(args.experiment_dir),
        generation_root=Path(args.generation_dir),
        sample_method=args.sample_method,
        num_sampling_steps=args.num_sampling_steps,
    )
    paths.ensure_dirs()
    
    cfg = paths.args()

    # Print W&B attach info
    run_id = args.wandb_run_id or os.getenv("WANDB_RUN_ID", "")
    if run_id:
        print(f"[wandb] Attaching to existing run: {run_id}")
        run_id = run_id.split("/")
        wandb.init(project=run_id[0], id=run_id[1], resume="must")
    else:
        print("[warn] [wandb] run_id is not set. Your sampler/metrics scripts must handle W&B init on their own.")

    real_data_path = Path(get_real_data_path(dataset_name=paths.dataset_name))
    real_sample_factor = get_real_data_sample_factor(paths.dataset_name, cfg.num_frames)

    devices = get_available_gpus()
    cuda_available_devices = ",".join(str(d) for d in devices)
    master_port = 29500 + (os.getpid() % 1000)  # Randomize port a bit to avoid collisions
    env = RunEnv(cuda_visible_devices=cuda_available_devices, master_port=master_port)

    sampler_args = SamplerArgs(
        is_difference=("difflatte" in paths.experiment_dir.name.lower()),
        config_path=paths.config_path,
        use_fp16=args.use_fp16,
        seed=args.seed,
        sample_method=args.sample_method,
        num_sampling_steps=args.num_sampling_steps,
        cfg_scale=args.cfg_scale,
        negative_name=args.negative_name,
        batch_size=args.batch_size,
        num_fvd_samples=args.num_fvd_samples,
        fps=args.fps,
        video_quality=args.video_quality,
        real_sample_factor=real_sample_factor
    )
    print("sampler_args", sampler_args)

    if args.ckpt:
        if not args.ckpt.exists():
            print(f"[error] --ckpt not found: {args.ckpt}")
            sys.exit(2)
        if already_processed(args.ckpt, paths.state_file):
            print(f"[info] Already processed: {args.ckpt}")
            sys.exit(0)
        process_checkpoint(env, paths, args.ckpt, real_data_path, sampler_args)
        sys.exit(0)

    # Initial pass over existing files
    initial_scan(env, paths, real_data_path, sampler_args, reverse=args.reverse, frequency=args.frequency, resolution=args.resolution)

    # Watch for new checkpoints
    watch_checkpoints(env, paths, real_data_path, sampler_args)



if __name__ == "__main__":
    main()
