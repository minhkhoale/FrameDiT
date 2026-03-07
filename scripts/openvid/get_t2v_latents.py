"""
Process OpenVid-HD zips ONE AT A TIME (sequentially), while still using a PyTorch DataLoader
to parallelize CPU decode inside that single zip.

Workflow:
1) For each zip shard:
   - download zip
   - build an IterableDataset that streams ONLY that zip’s matching videos
   - DataLoader(num_workers>0) decodes/preprocesses in parallel
   - main process batches -> VAE encode on GPU -> save .pt
   - delete zip
2) Move to next zip

This matches your “one zip at a time on disk” requirement.
"""
import time
from tqdm import tqdm
import os
import subprocess
import io
import shutil
import zipfile
import argparse
from typing import List, Set, Tuple, Optional
import re
import zstandard as zstd
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import pandas as pd
import torch
from torch.utils.data import IterableDataset, DataLoader, Dataset
import torchvision.transforms as T
import decord
from diffusers import AutoencoderKL
from huggingface_hub import list_repo_files, hf_hub_download

compressor = zstd.ZstdCompressor(level=6)

def print(*args, **kwargs):
    """Print with flush=True by default."""
    kwargs.setdefault("flush", True)
    __builtins__.print(*args, **kwargs)
# decord.bridge.set_bridge("native")

# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
RE_CAT_GROUP_1 = re.compile(r"^(?P<base>.+)_part_[a-z]+$")
RE_CAT_GROUP_2 = re.compile(r"^(?P<base>.+\.zip)\.\d+$")

def save_tensor(tensor, output_path):
    try:
        torch.save(tensor.clone(), output_path)
    except Exception as e:
        pass

def load_zip_index(json_path: str) -> dict:
    with open(json_path, "r") as f:
        zip_idxs = json.load(f)

    final_index = {}
    for z in zip_idxs:
        key = list(z.keys())[0][4:]
        final_index[f"OpenVidHD_part_{key}.zip"] = z[f"part{key}"]
    return final_index

def pick_needed_zip_keys(zip_index: dict, targets: Set[str]) -> Set[str]:
    """
    zip_index: {zip_key: [mp4_basename,...]}
    targets: set of mp4 basenames
    returns: set of zip_keys that contain at least one target
    """
    needed = set()
    # Fast membership tests: targets is a set
    for zip_key, files in zip_index.items():
        # files are basenames like "...mp4"
        for bn in files:
            if bn in targets:
                needed.add(zip_key)
                break
    return needed

def zip_key_to_repo_match_substring(zip_key: str) -> str:
    # zip_key like "part15" -> "OpenVidHD_part_15"
    m = re.match(r"part(\d+)$", zip_key)
    if not m:
        return zip_key  # fallback
    return f"OpenVidHD_part_{int(m.group(1))}"

# -----------------------------
# Dataset that streams from ONE zip only
# -----------------------------
class OneZipVideoStream(IterableDataset):
    """
    Streams (basename, frames[T,C,H,W]) for videos inside a single local zip file.

    Sharding across DataLoader workers:
    - We shard by index over the filtered members list, so workers don't duplicate work.
    - This requires building the members list once in the main process and passing it in.
    """

    def __init__(
        self,
        local_zip_path: str,
        members: List[str],              # full member paths inside zip to read
        save_dir: str,
        video_length: int,
        target_h: int,
        target_w: int,
        skip_existing: bool = True,
    ):
        super().__init__()
        self.local_zip_path = local_zip_path
        self.members = members
        self.save_dir = save_dir
        self.video_length = video_length
        self.skip_existing = skip_existing

        self.transform = T.Compose(
            [
                T.Resize(target_h),
                T.CenterCrop((target_h, target_w)),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def _decode_and_preprocess(self, video_bytes: bytes) -> Optional[torch.Tensor]:
        try:
            file_obj = io.BytesIO(video_bytes)
            vr = decord.VideoReader(file_obj)
            if len(vr) < self.video_length:
                return None

            idx = torch.linspace(0, len(vr) - 1, self.video_length).long().tolist()
            frames = vr.get_batch(idx)
            frames = torch.from_numpy(frames.asnumpy())  # (T,H,W,C) uint8
            frames = frames.permute(0, 3, 1, 2).contiguous().float() / 255.0  # (T,C,H,W)
            frames = self.transform(frames)  # [-1,1]
            return frames
        except Exception:
            return None

    def __iter__(self):
        # Shard members across workers
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start, end, step = 0, len(self.members), 1
        else:
            # "strided" sharding is simple and robust
            start, end, step = worker_info.id, len(self.members), worker_info.num_workers

        os.makedirs(self.save_dir, exist_ok=True)

        # Each worker opens the zip independently (read-only)
        with zipfile.ZipFile(self.local_zip_path, "r") as zf:
            for i in range(start, end, step):
                member = self.members[i]
                basename = os.path.basename(member)
                out_path = os.path.join(self.save_dir, basename.replace(".mp4", ".pt"))
                if self.skip_existing and os.path.exists(out_path):
                    continue

                try:
                    video_bytes = zf.read(member)
                except Exception:
                    continue

                frames = self._decode_and_preprocess(video_bytes)
                if frames is None:
                    continue

                yield basename, frames

class ExtractedMP4Dataset(Dataset):
    """
    Map-style dataset over extracted mp4 paths.
    Returns (basename, frames[T,C,H,W]).
    """

    def __init__(
        self,
        mp4_paths: List[str],
        video_length: int,
        target_h: int,
        target_w: int,
    ):
        self.mp4_paths = mp4_paths
        self.video_length = video_length

        self.transform = T.Compose([
            T.Resize(target_h),
            T.CenterCrop((target_h, target_w)),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.mp4_paths)

    def __getitem__(self, idx: int):
        mp4_path = self.mp4_paths[idx]
        name = os.path.basename(mp4_path)

        try:
            vr = decord.VideoReader(mp4_path)
            #indices = torch.linspace(0, len(vr) - 1, self.video_length).long().tolist()
            # sample first n frames
            indices = list(range(min(len(vr), self.video_length)))
            frames = vr.get_batch(indices)  # NDArray
            frames = torch.from_numpy(frames.asnumpy())  # (T,H,W,C) uint8

            frames = frames.permute(0, 3, 1, 2).contiguous().float() / 255.0  # (T,C,H,W)
            frames = self.transform(frames)  # (T,C,H,W) in [-1,1]
            return name, frames
        except Exception as e:
            print(f"Error decoding {mp4_path}: {e}")
            # signal DataLoader to drop this sample

            return None



def collate_batch(batch: List[Tuple[str, torch.Tensor]]) -> Tuple[List[str], torch.Tensor]:
    names = [x[0] for x in batch]
    frames = torch.stack([x[1] for x in batch], dim=0)  # (B,T,C,H,W)
    return names, frames


def cat_to_zip(part_paths: List[str], out_zip: str):
    with open(out_zip, "wb") as w:
        for p in part_paths:
            with open(p, "rb") as r:
                shutil.copyfileobj(r, w, length=16 * 1024 * 1024)


def extract_selected_members(local_zip: str, members: List[str], out_dir: str) -> List[str]:
    """
    Extract selected zip members to out_dir using streaming copy (low RAM).
    Returns list of extracted mp4 paths (flat names).
    """
    os.makedirs(out_dir, exist_ok=True)
    extracted_paths = []

    with zipfile.ZipFile(local_zip, "r") as zf:
        for member in tqdm(members, desc=f"Extracting from {os.path.basename(local_zip)}"):
            bn = os.path.basename(member)
            out_path = os.path.join(out_dir, bn)  # flatten
            if os.path.exists(out_path):
                extracted_paths.append(out_path)
                continue
            try:
                with zf.open(member) as src, open(out_path, "wb") as dst:
                    shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
                extracted_paths.append(out_path)
            except Exception:
                # skip broken member
                continue

    return extracted_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--n_videos", type=int, default=200_000)

    parser.add_argument("--repo_id", type=str, default="nkp37/OpenVid-1M")
    parser.add_argument("--zip_contains", type=str, default="HD")

    parser.add_argument("--video_col", type=str, default="video")
    parser.add_argument("--score_col", type=str, default="aesthetic score")

    parser.add_argument("--video_length", type=int, default=16)
    parser.add_argument("--target_h", type=int, default=512)
    parser.add_argument("--target_w", type=int, default=512)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)

    parser.add_argument("--local_dir", type=str, default=".")
    parser.add_argument("--keep_zip", action="store_true")

    # Latent saving choices
    parser.add_argument("--use_sample", action="store_true", help="Use posterior.sample(). Default is mean().")
    parser.add_argument("--latent_scale", type=float, default=1.0)
    parser.add_argument("--save_layout", type=str, default="T_C", choices=["T_C", "C_T"])

    parser.add_argument("--zip_index_json", type=str, required=True, help="JSON mapping {zip_key: [mp4_basename,...]}")
    parser.add_argument("--zip_key_prefix", type=str, default="OpenVidHD_part_", help="How to map JSON keys -> actual repo filenames (optional).")


    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    @torch.no_grad()
    def encode_and_save(names: List[str], frames: torch.Tensor):
        # frames: (B,T,C,H,W)
        B, T, C, H, W = frames.shape
        x = frames.view(B * T, C, H, W).to(device, non_blocking=True).half()

        # torch.cuda.synchronize()

        start_time = time.time()
        dist = vae.encode(x).latent_dist
        # torch.cuda.synchronize()
        end_time = time.time()
        # print(f"VAE encode in {end_time - start_time:.2f} seconds")
        lat = dist.parameters


        # (B*T,4,h,w) -> (B,T,4,h,w)
        lat = lat.view(B, T, lat.shape[1], lat.shape[2], lat.shape[3])
        # start_time = time.time()
        for i, name in enumerate(names):
            out_path = os.path.join(args.save_dir, name.replace(".mp4", ".pt"))
            if args.save_layout == "C_T":
                to_save = lat[i].permute(1, 0, 2, 3).contiguous().cpu()  # (4,T,h,w)
            else:
                to_save = lat[i].contiguous().cpu()  # (T,4,h,w)
            compressed = compressor.compress(to_save.numpy().tobytes())
            with open(out_path + ".zst", "wb") as f:
                f.write(compressed)
        # end_time = time.time()
        # print(f"Saved batch in {end_time - start_time:.2f} seconds")

    # Load VAE
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained("maxin-cn/Latte-1", subfolder="vae").to(device).half()
    vae.eval()

    # Load CSV + choose targets
    print(f"Loading CSV: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    if args.video_col not in df.columns:
        raise ValueError(f"Missing video_col='{args.video_col}'. Columns={df.columns.tolist()}")

    if args.score_col in df.columns:
        df = df.sort_values(by=args.score_col, ascending=False)
        print(f"Sorted by '{args.score_col}'.")
    else:
        print(f"No '{args.score_col}' column; using row order.")

    filtered_df = df.head(args.n_videos)
    # save only the top n_videos
    filtered_df.to_csv(args.csv_path.replace('.csv', f'_top{args.n_videos}.csv'), index=False)

    target_filenames: Set[str] = set(filtered_df[args.video_col].astype(str).values)

    # filter out processed files
    existing_files = os.listdir(args.save_dir)
    for ef in existing_files:
        if ef.endswith(".pt"):
            target_filenames.discard(ef.replace(".pt", ".mp4"))
    print(f"Target filenames: {len(target_filenames)}")
    
    zip_index = load_zip_index(args.zip_index_json)
    # print('zip_index', list(zip_index.keys()))
    # for z in zip_index:
    #     print(z.keys())
    # exit(0)
    needed_zip_keys = pick_needed_zip_keys(zip_index, target_filenames)
    # print('needed_zip_keys', needed_zip_keys)
    needed_substrings = {zip_key_to_repo_match_substring(k) for k in needed_zip_keys}
    # List zip shards
    all_files = list_repo_files(repo_id=args.repo_id, repo_type="dataset")

    groups = {}
    normal_files = []

    for f in all_files:
        if 'OpenVidHD' not in f:
            continue

        if f.endswith('.zip'):
            if os.path.basename(f) not in needed_substrings:
                continue
            normal_files.append(f)
            continue

        m1 = RE_CAT_GROUP_1.match(f)
        if m1:
            base = m1.group("base") + ".zip"
            if os.path.basename(base) not in needed_substrings:
                continue
            groups.setdefault(base, []).append(f)

    # # zip_files = []
    # splited_zip_files = []
    # zip_files = sorted([f for f in all_files if f.endswith(".zip") and (args.zip_contains in f)])
    # splited_zip_files = sorted([f for f in all_files if (args.zip_contains in f) and ("part_" in f) and not f.endswith(".zip")])
    # print(f"Found {len(zip_files)} zip shards matching contains='{args.zip_contains}'.")
    zip_jobs = []
    for zf in normal_files:
        zip_jobs.append({
            'type': 'zip',
            'base_zip': zf,
            'parts': [zf]
        })

    for base_zip, parts in groups.items():
        zip_jobs.append({
            'type': 'cat',
            'base_zip': base_zip,
            'parts': sorted(parts)
        })    
    
    zip_jobs = sorted(zip_jobs, key=lambda x: x["base_zip"])
    print('zip_jobs', zip_jobs)
    print(f"Total zip jobs: {len(zip_jobs)}")

    total_processed = 0

    #for zip_name in zip_files:
    for job in zip_jobs:
        base_zip = job["base_zip"]

        print(f"\n=== Processing {base_zip} ===")
        dl_paths = []
        for part in job["parts"]:
            print(f"  part: {part}")
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
            # lp = hf_hub_download(
            #     repo_id=args.repo_id,
            #     filename=part,
            #     repo_type="dataset",
            #     local_dir=args.local_dir,
            # )
            # call bash command to download file
            cmd = f"HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download {args.repo_id} {part} --repo-type dataset --local-dir {args.local_dir}"
            print(f"    running command: {cmd} and get stdout...")
            subprocess.run(cmd, shell=True, check=True)
            #os.system(cmd)
            lp = os.path.join(args.local_dir, part)
            dl_paths.append(lp)
        
        # local_zip = hf_hub_download(
        #     repo_id=args.repo_id,
        #     filename=zip_name,
        #     repo_type="dataset",
        #     local_dir=args.local_dir,
        # )
        if job["type"] == "cat":
            local_zip = os.path.join(args.local_dir, base_zip)
            print(f"Reconstructing via cat -> {local_zip}")
            cat_to_zip(dl_paths, local_zip)
        else:
            local_zip = dl_paths[0]

        try:
            # Build member list for THIS zip only (main process)
            with zipfile.ZipFile(local_zip, "r") as zf:
                members = []
                for member in zf.namelist():
                    basename = os.path.basename(member)
                    if basename in target_filenames:
                        # also skip if already exists to reduce worker overhead
                        out_path = os.path.join(args.save_dir, basename.replace(".mp4", ".pt"))
                        if os.path.exists(out_path):
                            continue
                        members.append(member)

            if not members:
                print(f"No target videos found in {base_zip}.")
                continue

            print(f"{base_zip}: matched {len(members)} videos. Starting DataLoader decode...")

            extract_dir = os.path.join(args.local_dir, base_zip[:-4])
            os.makedirs(extract_dir, exist_ok=True)
            mp4_paths = extract_selected_members(local_zip, members, extract_dir)
            mp4_paths = [p for p in mp4_paths if os.path.exists(p)]


            ds = ExtractedMP4Dataset(
                mp4_paths=mp4_paths,
                video_length=args.video_length,
                target_h=args.target_h,
                target_w=args.target_w,
            )

            loader = DataLoader(
                ds,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=False,
                persistent_workers=False,   # IMPORTANT: dataset changes per zip
                prefetch_factor=4,
                collate_fn=collate_batch,
            )

            processed_this_zip = 0
            try:
                start_time = time.time()

                for names, frames in loader:
                    # measure load time
                    end_time = time.time()
                    print(f"Batch loaded in {end_time - start_time:.2f} seconds")
                    encode_and_save(names, frames)
                    #print(f"Batch processed in {end_time - start_time:.2f} seconds")
                    processed_this_zip += len(names)
                    total_processed += len(names)
                    if processed_this_zip % 256 == 0:
                        print(f"{base_zip}: processed={processed_this_zip}/{len(members)}  total={total_processed}", end="\r")

                    start_time = time.time()

                print(f"\n{base_zip}: done. processed={processed_this_zip}")
            except Exception as e:
                print(f"\nError during DataLoader processing of {base_zip}: {e}")
                raise e
                # del loader
                # del ds
                # torch.cuda.empty_cache()

        finally:
            # wait for e
            shutil.rmtree(extract_dir, ignore_errors=True)
            if not args.keep_zip:
                print(f"Deleting {local_zip}")
                try:
                    if job["type"] == "cat":
                        os.remove(local_zip)
                    for p in dl_paths:
                        os.remove(p)
                except OSError:
                    pass
            
            del loader
            del ds
            torch.cuda.empty_cache()

    print(f"\nAll done. Total processed={total_processed}")


if __name__ == "__main__":
    main()

"""
python scripts/get_t2v_latents.py \
 --csv_path /scratch/s224075134/temporal_diffusion/datasets/video/OpenVid-0.4M/OpenVidHD.csv \
  --save_dir /scratch/s224075134/temporal_diffusion/datasets/video/OpenVid-0.4M/latents \
  --n_videos 200000 \
  --repo_id nkp37/OpenVid-1M \
  --zip_contains HD \
  --batch_size 16 \
  --num_workers 8 \
  --video_length 48 \
  --target_h 512 --target_w 512 \
  --latent_scale 1.0 \
  --save_layout T_C \
  --zip_index_json /scratch/s224075134/temporal_diffusion/datasets/video/OpenVid-0.4M/OpenVidHD.json
"""