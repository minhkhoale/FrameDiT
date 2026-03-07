import os, json, subprocess, shlex
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

PARQUET_PATH = "/scratch/s224075134/temporal_diffusion/datasets/video/pexels400k/pexels_400k.parquet"
JSON_PATH = "/scratch/s224075134/temporal_diffusion/datasets/video/pexels400k/pexels_400k_v2.json"
VIDEO_PATH = "/scratch/s224075134/temporal_diffusion/datasets/video/pexels400k/videos"
SHARD_SIZE = 1000
MIN_FRAMES = 16
WORKERS = 8  # tune: 8/16/32/64
SKIP_IDS = []

df = pd.read_parquet(PARQUET_PATH, columns=["video", "duration", "title"])

def ffprobe_meta(path: str):
    # width,height,avg_frame_rate,nb_frames (nb_frames can be "N/A" for some files)
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames",
        "-of", "json",
        path
    ]
    out = subprocess.check_output(cmd)
    j = json.loads(out)
    s = j["streams"][0]
    w = int(s.get("width", 0) or 0)
    h = int(s.get("height", 0) or 0)

    afr = s.get("avg_frame_rate", "0/0")
    num, den = afr.split("/")
    fps = (float(num) / float(den)) if float(den) != 0 else 0.0

    nb = s.get("nb_read_frames", "0")
    num_frames = int(nb) if str(nb).isdigit() else -1  # -1 means unknown
    return w, h, fps, num_frames

def build_item(idx: int, row):
    video_path = os.path.join(VIDEO_PATH, f"{(idx//SHARD_SIZE):04d}", f"pexels_{idx:06d}.mp4")
    if not os.path.exists(video_path):
        return None, "missing"

    try:
        if idx in SKIP_IDS:
            return None, "skipped_id"

        w, h, fps, nframes = ffprobe_meta(video_path)

        # if nb_frames unknown, you can either:
        #  - accept it (store -1)
        #  - or fall back to decord length only in that case (see below)
        if nframes != -1 and nframes < MIN_FRAMES:
            return None, "too_short"

        return {
            "video_id": idx,
            #"url": row["video"],
            "path": video_path,
            "duration": float(row["duration"]) if not pd.isna(row["duration"]) else None,
            "fps": fps,
            "num_frames": nframes,
            "resolution": {"width": w, "height": h},
            "cap": row["title"],
        }, None
    except Exception as e:
        return None, f"error:{e}"

data_items = []
skipped = 0

with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    futures = {ex.submit(build_item, int(idx), row): int(idx) for idx, row in df.iterrows()}
    for fut in tqdm(as_completed(futures), total=len(futures)):
        item, reason = fut.result()
        if item is None:
            # print(f"Skipping idx {futures[fut]}: {reason}")
            skipped += 1
        else:
            data_items.append(item)

with open(JSON_PATH, "w") as f:
    json.dump(data_items, f)

print(f"Total kept: {len(data_items)}, skipped: {skipped}")
