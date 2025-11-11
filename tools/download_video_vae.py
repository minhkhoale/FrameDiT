from huggingface_hub import snapshot_download, hf_hub_download


hf_hub_download(
    repo_id="kiwhansong/DFoT",
    local_dir='pretrained',
    filename='pretrained_models/VideoVAE_K600.ckpt',
    local_dir_use_symlinks=False
)