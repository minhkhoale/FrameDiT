import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import hf_hub_download

# Faster transfer backend
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

local_dir = "/scratch/s224075134/temporal_diffusion/datasets/video/videoufo/raw"
os.makedirs(local_dir, exist_ok=True)

REPO_ID = "WenhaoWang/VideoUFO"
MAX_WORKERS = 8   # adjust: 4–16 depending on network

def download_one(i):
    file_name = f"VideoUFO_tar/VideoUFO_{i}.tar"
    local_path = os.path.join(local_dir, file_name)

    if os.path.exists(local_path):
        return f"✓ Skip {file_name}"

    try:
        hf_hub_download(
            repo_id=REPO_ID,
            filename=file_name,
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # IMPORTANT for HPC
        )
        return f"⬇ Done {file_name}"

    except Exception as e:
        print('Error:', e)
        return f"✗ Failed {file_name}: {e}"


def main():
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(download_one, i) for i in range(1, 201)]

        for future in as_completed(futures):
            print(future.result())


if __name__ == "__main__":
    main()