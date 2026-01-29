import os
import json

log_dir = '/scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/generation/taichi128/019-Latte-M-2-F16S3-taichi128/ddpm_250/logs'
metric_dir = '/scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/generation/taichi128/019-Latte-M-2-F16S3-taichi128/ddpm_250/metrics'

for filename in os.listdir(log_dir):
    metric_file = os.path.join(metric_dir, f'metrics_{filename.replace('.log', '.json')}')
    if os.path.exists(metric_file):
        continue

    log_file = os.path.join(log_dir, filename)
    with open(log_file, 'r') as f:
        lines = f.readlines()
        start_line = -1
        end_line = -1
        for line in lines:
            if "{'final/VideoMetricType.FID'" in line:
                start_line = lines.index(line)
            
            if "}" in line and start_line != -1:
                end_line = lines.index(line)
                break

        if start_line == -1:
            continue

        metrics = {}
        for line in lines[start_line:end_line]:
            if line.startswith("}"):
                break
            key, value = line.split(": ")
            key = key.strip().strip("'").replace("'", "").replace("{", '')
            metrics[key] = float(value[:-2])

        print(metrics)
        with open(metric_file, 'w') as f:
            json.dump(metrics, f, indent=4)