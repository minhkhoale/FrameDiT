import wandb

log_path = '/scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/results/bair64/005-Latte-B-2-F16S1-bair64/log.txt'

wandb.init(project="vdm", name="005-Latte-B-2-F16S1-bair64")

"each line in the log file has the template: [2025-09-04 12:21:11] (step=0000500/epoch=0000) Train Loss: 0.1158, Gradient Norm: 0.1939, Train Steps/Sec: 17.58"
with open(log_path, 'r') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        if 'Train Loss' in line:
            parts = line.split('Train Loss: ')[1].split(', ')
            loss = float(parts[0])
            step = int(line.split('step=')[1].split('/')[0])
            norm = float(parts[1].split('Gradient Norm: ')[1])
            wandb.log({"train/loss": loss, "train/xs_loss": loss, "train/grad_norm": norm}, step=step)

            print(f"Step: {step}, train/loss: {loss}, train/grad_norm: {norm}")

