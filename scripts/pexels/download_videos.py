import os
import asyncio
import threading
from typing import Optional

import pandas as pd
import aiohttp
import aiofiles
from tqdm import tqdm

# =========================
# Config
# =========================
CSV_PATH = "/scratch/s224075134/temporal_diffusion/datasets/video/pexels400k/pexels_400k.parquet"
OUTPUT_DIR = "/scratch/s224075134/temporal_diffusion/datasets/video/pexels400k/videos"

MAX_TOTAL_BYTES = 4 * 1024**4  # 4 TB
CONCURRENCY = 32               # try 16/32/64 depending on your network
TIMEOUT_SECS = 60
CHUNK_SIZE = 4 * 1024 * 1024   # 4MB chunks is typically faster

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Disk usage
# =========================
def get_dir_size_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.isfile(fp) and not fp.endswith(".part"):
                total += os.path.getsize(fp)
    return total

total_bytes = get_dir_size_bytes(OUTPUT_DIR)
print(f"Initial usage: {total_bytes / 1024**4:.2f} TB")
if total_bytes >= MAX_TOTAL_BYTES:
    raise RuntimeError("Storage already >= 4TB. Abort.")

# Shared counters/locks (thread-safe enough for asyncio loop)
size_lock = asyncio.Lock()
stop_event = asyncio.Event()

# =========================
# Path helpers
# =========================
def make_out_paths(idx: int):
    shard = idx // 1000
    shard_dir = os.path.join(OUTPUT_DIR, f"{shard:04d}")
    os.makedirs(shard_dir, exist_ok=True)
    out_path = os.path.join(shard_dir, f"pexels_{idx:06d}.mp4")
    part_path = out_path + ".part"
    return out_path, part_path

# =========================
# Download one
# =========================
async def fetch_content_length(session: aiohttp.ClientSession, url: str) -> int:
    try:
        async with session.head(url, allow_redirects=True) as r:
            cl = r.headers.get("Content-Length")
            return int(cl) if cl and cl.isdigit() else 0
    except Exception:
        return 0

async def download_one(session: aiohttp.ClientSession, sem: asyncio.Semaphore, idx: int, url: str):
    global total_bytes

    if stop_event.is_set():
        return "stopped"

    out_path, part_path = make_out_paths(idx)

    if os.path.exists(out_path):
        return "exists"

    async with sem:
        if stop_event.is_set():
            return "stopped"

        # Reserve based on Content-Length if available
        reserved = await fetch_content_length(session, url)

        async with size_lock:
            if total_bytes + reserved >= MAX_TOTAL_BYTES:
                stop_event.set()
                return "cap_reached"
            total_bytes += reserved  # reserve

        written = 0
        try:
            timeout = aiohttp.ClientTimeout(total=TIMEOUT_SECS)
            async with session.get(url, timeout=timeout, allow_redirects=True) as r:
                r.raise_for_status()
                async with aiofiles.open(part_path, "wb") as f:
                    async for chunk in r.content.iter_chunked(CHUNK_SIZE):
                        if stop_event.is_set():
                            raise RuntimeError("Stopped by cap")
                        if chunk:
                            await f.write(chunk)
                            written += len(chunk)

            os.replace(part_path, out_path)

            # Adjust reservation vs actual
            async with size_lock:
                total_bytes += (written - reserved)
                if total_bytes >= MAX_TOTAL_BYTES:
                    stop_event.set()

            return "ok"

        except Exception as e:
            # Release reservation on failure
            async with size_lock:
                total_bytes -= reserved

            try:
                if os.path.exists(part_path):
                    os.remove(part_path)
            except Exception:
                pass

            return f"error: {e}"

# =========================
# Main
# =========================
async def main():
    df = pd.read_parquet(CSV_PATH)
    df = df[df["sfw"] == True]
    df = df[df["video"].notna()]

    jobs = [(int(idx), row["video"]) for idx, row in df.iterrows()]

    sem = asyncio.Semaphore(CONCURRENCY)

    counts = {"ok": 0, "exists": 0, "cap_reached": 0, "stopped": 0, "error": 0}

    connector = aiohttp.TCPConnector(limit=CONCURRENCY * 2, ttl_dns_cache=300)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [asyncio.create_task(download_one(session, sem, idx, url)) for idx, url in jobs]

        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            status = await fut
            if status in counts:
                counts[status] += 1
            elif status.startswith("error:"):
                counts["error"] += 1

            if stop_event.is_set():
                # We can't instantly stop in-flight tasks; they will notice stop_event on next chunk.
                pass

    print("Counts:", counts)
    print(f"Final usage: {total_bytes / 1024**4:.2f} TB")

if __name__ == "__main__":
    asyncio.run(main())