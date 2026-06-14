import os
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple, Union
from einops import rearrange
from omegaconf.listconfig import ListConfig
import torch
from torch.utils.data import Dataset
import pandas as pd
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution


def _safe_id(x: str) -> str:
    """Must match how you saved prompt-latent per-sample files."""
    x = str(x)
    x = x.replace("/", "_").replace("\\", "_")
    return re.sub(r"[^0-9a-zA-Z._-]+", "_", x)


@dataclass
class LatentTextSample:
    video_id: str
    prompt: str
    video_latent: torch.Tensor          # e.g. (T,4,h,w) or (4,T,h,w)
    text_embed: torch.Tensor            # e.g. (L,D) or (D,) depending on pooling
    # optional extra fields
    input_ids: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None


class LatentTextVideoDataset(Dataset):
    """
    Loads (video latent, prompt string, prompt latent) per row from a CSV.

    Expected files:
      - video latent:        latent_dir/<video_basename>.pt
        where <video_basename> is video filename without extension, or with .mp4 replaced by .pt
      - prompt latent:       prompt_latent_dir/<safe_id(video_id)>.pt   (per-sample mode)
        saved object can be either:
          * dict with keys: {"id","prompt","embeds", ...}
          * or a raw tensor (then we treat it as embeds)
    CSV:
      - must contain at least [id_col, prompt_col]
      - id_col is typically "video" (e.g. "abc.mp4")

    Notes:
      - This is map-style Dataset (works with shuffle=True).
      - It does NOT do any GPU work; keep tensors on CPU and move in training step.
    """

    def __init__(
        self,
        latent_path: Union[str, List[str]],
        **kwargs
    ):
        super().__init__()
        self.data_root = list(latent_path) if isinstance(latent_path, list) or isinstance(latent_path, ListConfig) else [latent_path]
        # recursively list all .pt files in latent_path
        self.files = []
        def load_latent(path, idx):
            for root, _, filenames in os.walk(path):
                for filename in filenames:
                    if filename.endswith(".pt"):
                        self.files.append((idx, os.path.join(root, filename)))
        
        for i, ltn_path in enumerate(self.data_root):
            load_latent(ltn_path, i)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        idx, pt_file = self.files[idx]
        pt_path = os.path.join(self.data_root[idx], pt_file)
        data = torch.load(pt_path)

        assert "text_embedding" in data or "prompt_embedding" in data, f"Missing text embedding in {pt_path}"

        video_latent = rearrange(data.get("latent"), 'T C H W -> C T H W')
        if "text_embedding" in data:
            prompt_embedding = data.get("text_embedding")
        elif "prompt_embedding" in data:
            prompt_embedding = data.get("prompt_embedding")

        # if shape is (1,L,D), squeeze to (L,D)
        if len(prompt_embedding.shape) == 3 and prompt_embedding.shape[0] == 1:
            prompt_embedding = prompt_embedding.squeeze(0)

        sample = {
            'video_id': str(data.get("video_id")),
            'prompt': data.get("text"),
            'prompt_embedding': prompt_embedding,
            'video_latent': video_latent
        }
        return sample

# ----------------------------
# Example collate (optional)
# ----------------------------
# def collate_latent_text(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
#     """
#     Collate function that stacks fixed-shape tensors.
#     If your tensors are variable-length (e.g., prompt_latent is (L,D) with different L),
#     you should pad instead of stack.
#     """
#     out = {
#         "video_id": [b["video_id"] for b in batch],
#         "prompt": [b["prompt"] for b in batch],
#         "video_latent": torch.stack([b["video_latent"] for b in batch], dim=0),
#         "prompt_latent": torch.stack([b["prompt_latent"] for b in batch], dim=0),
#     }
#     if "input_ids" in batch[0]:
#         # these may be None if not saved
#         if batch[0]["input_ids"] is not None:
#             out["input_ids"] = torch.stack([b["input_ids"] for b in batch], dim=0)
#         if batch[0]["attention_mask"] is not None:
#             out["attention_mask"] = torch.stack([b["attention_mask"] for b in batch], dim=0)
#     return out

# if __name__ == '__main__':
#     # simple test
#     dataset = OpenVidLatentTextDataset(
#         csv_path="/scratch/s224075134/temporal_diffusion/datasets/video/OpenVid-0.4M/OpenVidHD_top200000.csv",
#         latent_path="/scratch/s224075134/temporal_diffusion/datasets/video/OpenVid-0.4M/latents",
#         prompt_latent_path="/scratch/s224075134/temporal_diffusion/datasets/video/OpenVid-0.4M/prompts_latents",
#         drop_missing=True,
#         return_tokens_if_available=False,
#     )
#     print(f"Dataset length: {len(dataset)}")
#     sample = dataset[0]
#     print("Sample keys:", sample.keys())
#     print("Video latent shape:", sample["video_latent"].shape)
#     print("Prompt latent shape:", sample["prompt_latent"].shape)
#     if "input_ids" in sample:
#         print("Input IDs shape:", sample["input_ids"].shape)
#     if "attention_mask" in sample:
#         print("Attention mask shape:", sample["attention_mask"].shape)