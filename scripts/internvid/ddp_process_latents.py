import json
from operator import __getitem__
import os
from os.path import join as opj
import subprocess
import torchvision
import torch
import argparse
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from diffusers import AutoencoderKL
from huggingface_hub import snapshot_download
import importlib
from einops import rearrange
import inspect
import decord
from decord import VideoReader, cpu
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time
import asyncio
import aiohttp
import aiofiles
import async_timeout
import threading
from typing import Optional
import pandas as pd
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

MAX_TOTAL_BYTES = 4 * 1024**4  # 4 TB
CONCURRENCY = 32               # try 16/32/64 depending on your network
TIMEOUT_SECS = 60
CHUNK_SIZE = 4 * 1024 * 1024   # 4MB chunks is typically faster


# def print(*args, **kwargs):
#     #if dist.get_rank() == 0:
#     __builtins__.print(flush=True, *args, **kwargs)


def setup_distributed(backend="nccl", port=None):
    """Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    see torch.distributed.init_process_group() for more details
    """
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        master_addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if port is not None:
            master_port = int(port)
        elif "MASTER_PORT" in os.environ:
            master_port = int(os.environ["MASTER_PORT"])
        else:
            job_id = int(os.environ["SLURM_JOB_ID"].split(".")[0])
            master_port = 10000 + (job_id % 50000)

        os.environ["MASTER_PORT"] = str(master_port)
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = master_addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    # torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )


# def is_image_vae(vae):
#     # check if the vae is an image vae by checking if it has a 'decode' method that takes in (B, C, H, W) and outputs (B, C, H, W)
#     if hasattr(vae, "decode"):
#         decode_signature = inspect.signature(vae.decode)
#         parameters = decode_signature.parameters
#         if len(parameters) == 1:
#             param = next(iter(parameters.values()))
#             if param.annotation == torch.Tensor:
#                 return True
#     return False


def _is_tensor_video_clip(clip):
    if not torch.is_tensor(clip):
        raise TypeError("clip should be Tensor. Got %s" % type(clip))

    if not clip.ndimension() == 4:
        raise ValueError("clip should be 4D. Got %dD" % clip.dim())

    return True

def to_tensor(clip):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (T, C, H, W)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
    """
    _is_tensor_video_clip(clip)
    if not clip.dtype == torch.uint8:
        raise TypeError("clip tensor should have data type uint8. Got %s" % str(clip.dtype))
    # return clip.float().permute(3, 0, 1, 2) / 255.0
    return clip.float() / 255.0

def crop(clip, i, j, h, w):
    """
    Args:
        clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
    """
    if len(clip.size()) != 4:
        raise ValueError("clip should be a 4D tensor")
    return clip[..., i : i + h, j : j + w]

def center_crop_using_short_edge(clip):
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    h, w = clip.size(-2), clip.size(-1)
    if h < w:
        th, tw = h, h
        i = 0
        j = int(round((w - tw) / 2.0))
    else:
        th, tw = w, w
        i = int(round((h - th) / 2.0))
        j = 0
    return crop(clip, i, j, th, tw)

def resize(clip, target_size, interpolation_mode):
    if len(target_size) != 2:
        raise ValueError(f"target size should be tuple (height, width), instead got {target_size}")
    return torch.nn.functional.interpolate(clip, size=target_size, mode=interpolation_mode, align_corners=False)


class CenterCropResizeVideo:
    '''
    First use the short side for cropping length, 
    center crop video, then resize to the specified size
    '''
    def __init__(
        self,
        size,
        interpolation_mode="bilinear",
    ):
        if isinstance(size, tuple):
            if len(size) != 2:
                raise ValueError(f"size should be tuple (height, width), instead got {size}")
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation_mode = interpolation_mode
       

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: scale resized / center cropped video clip.
                size is (T, C, crop_size, crop_size)
        """
        clip_center_crop = center_crop_using_short_edge(clip)
        clip_center_crop_resize = resize(clip_center_crop, target_size=self.size, interpolation_mode=self.interpolation_mode)
        return clip_center_crop_resize

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, interpolation_mode={self.interpolation_mode})"

class ToTensorVideo:
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    """

    def __init__(self):
        pass

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor, dtype=torch.uint8): Size is (T, C, H, W)
        Return:
            clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
        """
        return to_tensor(clip)

    def __repr__(self) -> str:
        return self.__class__.__name__

class VideoCaptionDataset(Dataset):
    def __init__(self, parquet_path, output_path, shard_size, first_n_videos=None):
        super().__init__()
        self.parquet_path = parquet_path
        self.output_path = output_path
        self.shard_size = shard_size
        self.n_frames = 16
        self.train_fps = 8
        self.height = 512
        self.width = 512
        self.text_max_length = 120
        self.first_n_videos = first_n_videos
        self.raw_data = self.load_raw_data()


    def __getitem__(self, index):
        return self.raw_data.iloc[index].to_dict()

    def __len__(self):
        return len(self.raw_data)

    def load_raw_data(self):
        import pandas as pd
        df = pd.read_parquet(self.parquet_path)
        df['video_id'] = df.index.astype(int)

        # filter existing pt files in output path to avoid re-processing
        existing_video_ids = set()
        for root, dirs, files in os.walk(self.output_path):
            for file in files:
                if file.endswith(".pt"):
                    video_id = int(file.split(".")[0])
                    existing_video_ids.add(video_id)

        print(f"Found {len(existing_video_ids)} existing processed videos in {self.output_path}, skipping them...")
        df = df[~df["video_id"].isin(existing_video_ids)]

        df = df.sort_values("Aesthetic_Score", ascending=False)
        if self.first_n_videos is not None:
            df = df.head(self.first_n_videos)
        return df

def get_class(module_name: str, class_name: str):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def load_models(model_path, modules=['text_encoder', 'tokenizer', 'vae'], eval=True, device='cuda:0'):
    # if model_path not exist
    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist, downloading from HuggingFace Hub...")
        model_path = snapshot_download(repo_id=model_path)
        print(f"Model downloaded to {model_path}")

    model_config_path = opj(model_path, "model_index.json")
    model_configs = json.load(open(model_config_path))

    models = {}
    for module in modules:
        if module not in model_configs:
            raise ValueError(f"Module {module} not found in model config")
    
        package, model_name = model_configs[module]
        model_class = get_class(package, model_name)
        model = model_class.from_pretrained(model_path, subfolder=module)

        if hasattr(model, "to"):
            model = model.to(device)
        models[module] = model
        if eval and hasattr(model, "eval"):
            model.eval()
    
    return models

def download_video(youtube_id, start, end, save_path):
    if os.path.exists(save_path):
        return save_path

    url = f"https://www.youtube.com/watch?v={youtube_id}"
    try:
        # subprocess.run(
        #     ["wget", "-q", "-O", save_path, url],
        #     check=True,
        # )
        subprocess.run([
            "yt-dlp",
            url,
            "--download-sections", f"*{start}-{end}",
            "--force-keyframes-at-cuts",
            "-t", "mp4",
            "-o", save_path,
            "--quiet",
            "-N", "8"
        ], check=True)
        return save_path
    except Exception as e:
        print(f"Download failed: {youtube_id}, {e}")
        return None


# async def download_one(session, url, save_path, semaphore):
#     if os.path.exists(save_path):
#         return save_path

#     async with semaphore:
#         try:
#             async with async_timeout.timeout(TIMEOUT_SECS):
#                 async with session.get(url) as response:
#                     if response.status != 200:
#                         return None
#                     async with aiofiles.open(save_path, "wb") as f:
#                         async for chunk in response.content.iter_chunked(CHUNK_SIZE):
#                             await f.write(chunk)
#             return save_path
#         except Exception:
#             return None

# async def download_one(session, url, start, end, save_path, semaphore

# async def download_batch(urls, starts, ends, save_paths):
#     semaphore = asyncio.Semaphore(CONCURRENCY)

#     connector = aiohttp.TCPConnector(
#         limit=CONCURRENCY,
#         ttl_dns_cache=300,
#         ssl=False
#     )

#     async with aiohttp.ClientSession(connector=connector) as session:
#         tasks = [
#             download_one(session, url, path, semaphore)
#             for url, path in zip(urls, save_paths)
#         ]
#         results = await asyncio.gather(*tasks)

#     return results

def download_batch(youtube_ids, starts, ends, save_paths):
    results = []
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = [
            executor.submit(download_video, youtube_id, start, end, path)
            for youtube_id, start, end, path in zip(youtube_ids, starts, ends, save_paths)
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
    return results

def read_video(path, n_frames=16, train_fps=8):
    decord.bridge.set_bridge('torch')
    if path is None or not os.path.exists(path):
        print(f"Video path {path} does not exist, skipping...")
        return None
    try:
        vr = VideoReader(path)
        video_length = len(vr)
        fps = vr.get_avg_fps()
        frame_interval = max(1, int(round(fps / train_fps)))
        sample_frame_index = list(range(0, len(vr), frame_interval))[:n_frames]
        if len(sample_frame_index) < n_frames:
            print(f"Video {path} has only {len(sample_frame_index)} frames sampled at {train_fps} fps, which is less than the required {n_frames} frames, skipping this video.")
            return None
        if sample_frame_index[-1] >= video_length:
            print(f"Video {path} has only {video_length} frames, but trying to sample frame index {sample_frame_index[-1]}, skipping this video.")
            return None
        video = vr.get_batch(sample_frame_index).permute(0, 3, 1, 2)  # (T, C, H, W)
        return video
    except Exception as e:
        print(f"Failed to read video {path}: {e}")
        return None


@torch.inference_mode()
def main():
    decord.bridge.set_bridge('torch')
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)   # put at top-level under if __name__ == "__main__": ideally

    parser = argparse.ArgumentParser(description="Process raw videos into latents for training")
    parser.add_argument("--parquet_path", type=str, required=True, help="Path to the parquet file containing video metadata")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the directory containing video files")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the processed latents")
    parser.add_argument("--model_path", type=str, default="maxin-cn/Latte-1", help="Path to the pretrained model for VAE")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for processing")
    parser.add_argument("--shard-size", type=int, default=1000, help="Number of videos per shard for pexels dataset")
    parser.add_argument("--height", type=int, default=512, help="Height of the processed video frames")
    parser.add_argument("--width", type=int, default=512, help="Width of the processed video frames")
    parser.add_argument("--n-frames", type=int, default=16, help="Number of frames to sample from each video")
    parser.add_argument("--train-fps", type=int, default=8, help="FPS to sample frames from videos for training")
    parser.add_argument("--first_n_videos", type=int, default=None, help="Only process the first n videos with highest aesthetic score")
    args = parser.parse_args()

    setup_distributed()
    world_size = dist.get_world_size()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    print('device:', device, flush=True)
    print('device:', device, flush=True)
    print('device:', device, flush=True)
    print('device:', device, flush=True)
    # gpu mem before loading models
    print(f"GPU memory usage before loading models: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB", flush=True)
    print(f"GPU memory usage before loading models: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB", flush=True)
    print(f"GPU memory usage before loading models: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB", flush=True)
    print(f"GPU memory usage before loading models: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB", flush=True)
    print(f"GPU memory usage before loading models: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB", flush=True)

    models = load_models(args.model_path, modules=['vae', 'text_encoder', 'tokenizer'], eval=True, device=device)
    # models['vae'] = models['vae']
    # models['text_encoder'] = models['text_encoder'], device_ids=[local_rank])

    dataset = VideoCaptionDataset(args.parquet_path, args.output_path, args.shard_size)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=False, num_workers=args.num_workers, prefetch_factor=None, pin_memory=False)

    transform = torchvision.transforms.Compose([
            ToTensorVideo(),
            CenterCropResizeVideo(size=(args.height, args.width), interpolation_mode="bilinear"),
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])

    # create output directory if not exist
    if rank == 0:
        os.makedirs(args.video_path, exist_ok=True)
        os.makedirs(args.output_path, exist_ok=True)
        num_processed_samples = 0
        print('total batch:', len(dataloader), flush=True)

    for i, batch in enumerate(dataloader):
        if batch is None:
            continue

        prompts = batch['Caption'] # list of str

        # download video, decode frames, apply transforms, and move to device
        youtube_ids = batch["YoutubeID"]
        video_idxs = batch["video_id"].tolist()
        video_names = [f"{video_idx}.mp4" for video_idx in video_idxs]

        save_paths = [opj(args.video_path, name) for name in video_names]

        start_time = time()
        # local_video_paths = asyncio.run(
        #     download_batch(video_urls, save_paths)
        # )
        local_video_paths = download_batch(youtube_ids, batch["Start_timestamp"], batch["End_timestamp"], save_paths)
        end_time = time()
        print(f"Downloaded batch {i} in {end_time - start_time:.2f} seconds", flush=True)

        # start_time = time()
        # TODO: using decord to read videos
        skip_items = []
        videos = []


        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(read_video, local_video_paths, [args.n_frames]*len(local_video_paths), [args.train_fps]*len(local_video_paths)))

        videos = []
        for idx, video in enumerate(results):
            if video is None:
                skip_items.append(idx)
                continue
            try:
                video = transform(video.to(device))  # (T, C, H, W)
                videos.append(video)
            except Exception as e:
                print(f"Failed to transform video {local_video_paths[idx]}: {e}")
                skip_items.append(idx)

        if len(videos) == 0:
            print(f"No valid videos in batch {i}, skipping...")
            continue

        pixel_values = torch.stack(videos)  # (B, T, C, H, W)
        # pixel_values = transform(pixel_values).to(device)  # (B, T, C, H, W)
        
        # remove ' · Free Stock Video' in the end of each prompt
        # prompts = [p.replace(" · Free Stock Video", "") for idx, p in enumerate(prompts) if idx not in skip_items]

        start_time = time()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            # if image vae, then reshape B, T, C, H, W to (B*T, C, H, W)
            pixel_values_vae = rearrange(pixel_values, "B T C H W -> (B T) C H W")
            latent = models['vae'].encode(pixel_values_vae).latent_dist.mean
            latent = rearrange(latent, "(B T) C H W -> B T C H W", B=pixel_values.size(0))
        end_time = time()
        print(f"Encoded batch {i} into latents in {end_time - start_time:.2f} seconds", flush=True)

        start_time = time()
        with torch.no_grad():
            text_inputs = models['tokenizer'](
                prompts,
                padding="max_length",      # or "longest"
                truncation=True,
                max_length=120,
                return_attention_mask=True,
                return_tensors="pt",
            )
            input_ids = text_inputs.input_ids.to(device)
            attention_mask = text_inputs.attention_mask.to(device)

            text_embeddings = models['text_encoder'](
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state.cpu() # (B, seq_len, hidden_size)
        
        end_time = time()
        print(f"Encoded batch {i} into text embeddings in {end_time - start_time:.2f} seconds")
        for j in range(latent.size(0)):
            video_id = video_idxs[j]
            output_item = {
                "video_name": video_names[j],
                "video_id": video_id,
                "latent": latent[j].cpu(),
                'text': prompts[j],
                # "input_ids": input_ids[j].cpu(),
                # "attention_mask": attention_mask[j].cpu(),
                "text_embedding": text_embeddings[j],
            }
            shard_idx = video_id // args.shard_size
            shard_dir = opj(args.output_path, f"{shard_idx:04d}")
            os.makedirs(shard_dir, exist_ok=True)
            output_file = opj(shard_dir, f"{video_id:06d}.pt")

            # print(f"Saving processed latent for (video_id: {video_id}) to {output_file}")
            torch.save(output_item, output_file)

        if rank == 0:
            # get num processed distributedly
            num_processed_samples += latent.size(0)

            if num_processed_samples % 100 == 0:
                print(f"Processed {num_processed_samples} samples so far...", flush=True)

        # remove local video files to save disk space
        for path in local_video_paths:
            if path is not None and os.path.exists(path):
                os.remove(path)

if __name__ == "__main__":
    main()


"""
    parser = argparse.ArgumentParser(description="Process raw videos into latents for training")
    parser.add_argument("--parquet_path", type=str, required=True, help="Path to the parquet file containing video metadata")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the directory containing video files")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the processed latents")
    parser.add_argument("--model_path", type=str, default="maxin-cn/Latte-1", help="Path to the pretrained model for VAE")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the processing on")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for processing")
    parser.add_argument("--shard-size", type=int, default=1000, help="Number of videos per shard for pexels dataset")
    parser.add_argument("--height", type=int, default=512, help="Height of the processed video frames")
    parser.add_argument("--width", type=int, default=512, help="Width of the processed video frames")
    parser.add_argument("--n-frames", type=int, default=16, help="Number of frames to sample from each video")
    parser.add_argument("--train-fps", type=int, default=8, help="FPS to sample frames from videos for training")
    """


"""
srun python scripts/internvid/ddp_process_latents.py \
    --parquet_path /scratch/s224075134/temporal_diffusion/datasets/video/internvid/InternVid-18M-aes.parquet \
    --model_path maxin-cn/Latte-1 \
    --output_path /scratch/s224075134/temporal_diffusion/datasets/video/internvid/autoencoder_kl_latents \
    --video_path /scratch/s224075134/temporal_diffusion/datasets/video/internvid/videos \
    --num_workers 4
"""