import os
import argparse
import subprocess
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from diffusers import AutoencoderKL
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from einops import rearrange
from decord import VideoReader
import decord

# -------------------------------------------------
# Dataset
# -------------------------------------------------

class InternVidAESDataset(Dataset):
    def __init__(self, parquet_path, output_path,
                 topk=None, aes_threshold=None):
        self.parquet_path = parquet_path
        self.output_path = output_path
        self.topk = topk
        self.aes_threshold = aes_threshold
        self.data = self.load_data()

    def load_data(self):
        df = pd.read_parquet(self.parquet_path)
        df = df.sort_values("Aesthetic_Score", ascending=False)

        if self.aes_threshold is not None:
            df = df[df["Aesthetic_Score"] >= self.aes_threshold]

        if self.topk is not None:
            df = df.head(self.topk)

        df["clip_id"] = (
            df["YoutubeID"]
            + "_"
            + df["Start_timestamp"].str.replace(":", "")
            + "_"
            + df["End_timestamp"].str.replace(":", "")
        )

        # Skip already processed
        existing = set()
        for f in os.listdir(self.output_path):
            if f.endswith(".pt"):
                existing.add(f.replace(".pt", ""))

        df = df[~df["clip_id"].isin(existing)]

        print("Final dataset size:", len(df))
        return df.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx].to_dict()


# -------------------------------------------------
# Segment Downloader (ONLY needed segment)
# -------------------------------------------------

def download_segment(youtube_id, start, end, save_path):
    if os.path.exists(save_path):
        return save_path

    url = f"https://www.youtube.com/watch?v={youtube_id}"

    try:
        # print([
        #     "yt-dlp",
        #     url,
        #     "--download-sections", f"*{start}-{end}",
        #     "--force-keyframes-at-cuts",
        #     "-t", "mp4[height<=720]",
        #     "-o", save_path,
        #     "--quiet"
        # ])
        # exit(0)
        subprocess.run([
            "yt-dlp",
            url,
            "--download-sections", f"*{start}-{end}",
            "--force-keyframes-at-cuts",
            "-t", "mp4",
            "-o", save_path,
            "--quiet"
        ], check=True)
        return save_path
    except Exception as e:
        print("Download failed:", youtube_id, e)
        return None


# -------------------------------------------------
# Video Reader
# -------------------------------------------------

def read_video(path, n_frames=16):
    print('reading video:', path)
    try:
        vr = VideoReader(path)
        total = len(vr)
        if total < n_frames:
            return None
        idx = torch.linspace(0, total - 1, n_frames).long()
        frames = vr.get_batch(idx).permute(0, 3, 1, 2)
        return frames
    except Exception as e:
        print("Video read failed:", path, e)
        return None


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--segment_cache", required=True)
    parser.add_argument("--model_path", default="maxin-cn/Latte-1")
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--aes_threshold", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--n_frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.segment_cache, exist_ok=True)

    # Load model
    if not os.path.exists(args.model_path):
        args.model_path = snapshot_download(args.model_path)

    vae = AutoencoderKL.from_pretrained(
        args.model_path, subfolder="vae"
    ).to(device).eval()

    dataset = InternVidAESDataset(
        args.parquet_path,
        args.output_path,
        topk=args.topk,
        aes_threshold=args.aes_threshold
    )

    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=False)

    transform = torchvision.transforms.Compose([
        lambda x: x.float() / 255.0,
        torchvision.transforms.Resize((args.height, args.width)),
        torchvision.transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    for batch in dataloader:
        for i in range(len(batch["YoutubeID"])):

            yt = batch["YoutubeID"][i]
            start = batch["Start_timestamp"][i]
            end = batch["End_timestamp"][i]
            caption = batch["Caption"][i]
            clip_id = batch["clip_id"][i]

            clip_path = os.path.join(
                args.segment_cache,
                clip_id + ".mp4"
            )

            video_path = download_segment(
                yt, start, end, clip_path
            )

            if video_path is None:
                print("Skipping due to download failure:", clip_id)
                continue

            frames = read_video(video_path, args.n_frames)
            if frames is None:
                print("Skipping due to video read failure:", clip_id)
                continue

            frames = transform(frames).to(device)
            frames = rearrange(frames, "T C H W -> (T) C H W")

            with torch.no_grad(), torch.autocast("cuda", torch.float16):
                latent = vae.encode(frames).latent_dist.mean

            output = {
                "clip_id": clip_id,
                "latent": latent.cpu(),
                "caption": caption,
            }

            torch.save(output,
                       os.path.join(args.output_path,
                                    clip_id + ".pt"))

            # rm video to save space
            os.remove(video_path)

        print("Batch processed")



    print("All done!")


if __name__ == "__main__":
    main()

"""
    parser.add_argument("--parquet_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--segment_cache", required=True)
    parser.add_argument("--model_path", default="maxin-cn/Latte-1")
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--aes_threshold", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--n_frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)

python scripts/internvid/ddp_process_latents.py \
    --parquet_path /scratch/s224075134/temporal_diffusion/datasets/video/internvid/InternVid-18M-aes.parquet \
    --output_path /scratch/s224075134/temporal_diffusion/datasets/video/internvid/autoencoder_kl_latent \
    --segment_cache /scratch/s224075134/temporal_diffusion/datasets/video/internvid/segments \
    --model_path maxin-cn/Latte-1 \
    --topk 10000 \
    --aes_threshold 5.0 \
    --batch_size 4 \
    --n_frames 16 \
    --height 512 \
    --width 512

"""