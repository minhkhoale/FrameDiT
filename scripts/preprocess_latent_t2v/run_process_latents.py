import subprocess
import time

"""
python scripts/process_latents.py     --data_root /scratch/s224075134/temporal_diffusion/datasets/video/pexels400k     --model_path maxin-cn/Latte-1     --output_path /scratch/s224075134/temporal_diffusion/datasets/video/pexels400k/pexels400k_processed_t2v     --device cuda:0
"""
cmd = ["python", "scripts/process_latents.py",
    "--data_root", "/scratch/s224075134/temporal_diffusion/datasets/video/pexels400k",
    "--model_path", "maxin-cn/Latte-1",
    "--output_path", "/scratch/s224075134/temporal_diffusion/datasets/video/pexels400k/pexels400k_processed_t2v",
    "--device", "cuda:0"
]

while True:
    print("Running script...")
    result = subprocess.run(
        cmd,
        
    )

    if result.returncode == 0:
        print("✅ Success")
        break
    else:
        print("❌ Failed, stderr:")
        print(result.stderr)
        time.sleep(10)
