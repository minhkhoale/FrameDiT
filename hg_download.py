from huggingface_hub import snapshot_download, hf_hub_download




# snapshot_download(
#     repo_id="maxin-cn/Latte",
#     local_dir='/scratch/s224075134/temporal_diffusion/datasets/video/taichi-hd',
#     repo_type="dataset",
#     local_dir_use_symlinks=False
# )

hf_hub_download(
    repo_id="maxin-cn/Latte",
    local_dir='pretrained',
    filename='ucf101.pt',
    local_dir_use_symlinks=False
)