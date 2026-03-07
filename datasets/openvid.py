import os
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from einops import rearrange

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


class OpenVidLatentTextDataset(Dataset):
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
        csv_path: str,
        latent_path: str,
        prompt_latent_path: str,
        id_col: str = "video",
        prompt_col: str = "caption",
        # if your latent saving uses .pt with same basename
        video_ext: str = ".mp4",
        video_latent_ext: str = ".pt",
        # filtering
        drop_missing: bool = True,
        # prompt-latent file naming
        prompt_latent_naming: str = "safe_id",  # "safe_id" or "video_stem"
        # optionally return tokens if present in prompt latent dict
        return_tokens_if_available: bool = False,
        # optionally memory-map-ish cache (small dict cache)
        cache_video_latents: bool = False,
        cache_text_latents: bool = False,
        **kwargs
    ):
        super().__init__()
        self.csv_path = csv_path
        self.latent_path = latent_path
        self.prompt_latent_path = prompt_latent_path
        self.id_col = id_col
        self.prompt_col = prompt_col
        self.video_ext = video_ext
        self.video_latent_ext = video_latent_ext
        self.drop_missing = drop_missing
        self.prompt_latent_naming = prompt_latent_naming
        self.return_tokens_if_available = return_tokens_if_available

        self._video_cache: Dict[str, torch.Tensor] = {} if cache_video_latents else None
        self._text_cache: Dict[str, Any] = {} if cache_text_latents else None

        df = pd.read_csv(csv_path)[:75]
        # df = df[:10000]
        print('warning: only load 75 samples for test')
        print('warning: only load 75 samples for test')
        print('warning: only load 75 samples for test')
        print('warning: only load 75 samples for test')
        print('warning: only load 75 samples for test')
        if id_col not in df.columns:
            raise ValueError(f"CSV missing id_col='{id_col}'. Columns={df.columns.tolist()}")
        if prompt_col not in df.columns:
            raise ValueError(f"CSV missing prompt_col='{prompt_col}'. Columns={df.columns.tolist()}")

        # Normalize types
        df[id_col] = df[id_col].astype(str)
        df[prompt_col] = df[prompt_col].astype(str)

        # Build file paths; optionally drop missing rows
        rows = []
        missing_video = 0
        missing_text = 0

        for _, r in df.iterrows():
            vid = r[id_col]
            prompt = r[prompt_col]

            video_latent_path = self._video_latent_path(vid)
            text_latent_path = self._text_latent_path(vid)

            ok = True
            if not os.path.exists(video_latent_path):
                missing_video += 1
                ok = False
            if not os.path.exists(text_latent_path):
                missing_text += 1
                ok = False

            if ok or not drop_missing:
                rows.append({
                    "video_id": vid,
                    "prompt": prompt,
                    "video_latent_path": video_latent_path,
                    "text_latent_path": text_latent_path,
                })

        if drop_missing:
            print(
                f"[OpenVidLatentTextDataset] Loaded {len(rows)} rows from {len(df)} "
                f"(dropped missing: video_latent={missing_video}, text_latent={missing_text})."
            )
        else:
            print(
                f"[OpenVidLatentTextDataset] Loaded {len(rows)} rows from {len(df)} "
                f"(missing counts: video_latent={missing_video}, text_latent={missing_text})."
            )

        self.rows = rows

    def _video_latent_path(self, video_id: str) -> str:
        # video_id like "abc.mp4" -> "abc.pt"
        bn = os.path.basename(video_id)
        if bn.endswith(self.video_ext):
            bn = bn[: -len(self.video_ext)] + self.video_latent_ext
        else:
            # if already no .mp4, just append/replace extension
            stem, _ = os.path.splitext(bn)
            bn = stem + self.video_latent_ext
        return os.path.join(self.latent_path, bn)

    def _text_latent_path(self, video_id: str) -> str:
        bn = os.path.basename(video_id)
        if self.prompt_latent_naming == "video_stem":
            # expects prompt latent file named by video stem (abc.pt)
            stem, _ = os.path.splitext(bn)
            fn = stem + ".pt"
        else:
            # expects prompt latent file named by safe_id(video_id)
            fn = _safe_id(video_id) + ".pt"
        return os.path.join(self.prompt_latent_path, fn)

    def __len__(self) -> int:
        return len(self.rows)

    def _load_video_latent(self, path: str) -> torch.Tensor:
        if self._video_cache is not None and path in self._video_cache:
            return self._video_cache[path]
        parameters = torch.load(path, map_location="cpu", weights_only=True)
        x = DiagonalGaussianDistribution(parameters=parameters).sample()
        # if not torch.is_tensor(x):
        #     # Some people save dicts; if so, try common keys
        #     if isinstance(x, dict):
        #         for k in ["latents", "latent", "video_latent"]:
        #             if k in x and torch.is_tensor(x[k]):
        #                 x = x[k]
        #                 break
        #     if not torch.is_tensor(x):
        #         raise ValueError(f"Video latent at {path} is not a Tensor (type={type(x)}).")
        # if self._video_cache is not None:
        #     self._video_cache[path] = x
        return x

    def _load_text_latent(self, path: str) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # max length: 120
        if self._text_cache is not None and path in self._text_cache:
            obj = self._text_cache[path]
        else:
            obj = torch.load(path, map_location="cpu", weights_only=True)
            if self._text_cache is not None:
                self._text_cache[path] = obj

        input_ids = None
        attn = None

        if torch.is_tensor(obj):
            embeds = obj
        elif isinstance(obj, dict):
            # support your encoding script format: {"embeds": ..., "input_ids":..., "attention_mask":...}
            if "embeds" in obj and torch.is_tensor(obj["embeds"]):
                embeds = obj["embeds"]
            elif "text_embeds" in obj and torch.is_tensor(obj["text_embeds"]):
                embeds = obj["text_embeds"]
            else:
                raise ValueError(f"Text latent dict at {path} missing 'embeds' tensor keys. Keys={list(obj.keys())}")

            if self.return_tokens_if_available:
                if "input_ids" in obj and torch.is_tensor(obj["input_ids"]):
                    input_ids = obj["input_ids"]
                if "attention_mask" in obj and torch.is_tensor(obj["attention_mask"]):
                    attn = obj["attention_mask"]
        else:
            raise ValueError(f"Text latent at {path} has unsupported type={type(obj)}")

        return embeds, input_ids, attn

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        vid = r["video_id"]
        prompt = r["prompt"]

        vlat = self._load_video_latent(r["video_latent_path"])
        tlat, input_ids, attn = self._load_text_latent(r["text_latent_path"])

        sample = {
            "video_id": vid,
            "prompt": prompt,
            "video_latent": rearrange(vlat, "T C H W -> C T H W"),     # Tensor: C, T, H, W
            "prompt_embedding": tlat,    # Tensor
            "is_latent": True,
        }
        if self.return_tokens_if_available:
            sample["input_ids"] = input_ids
            sample["attention_mask"] = attn
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

if __name__ == '__main__':
    # simple test
    dataset = OpenVidLatentTextDataset(
        csv_path="/scratch/s224075134/temporal_diffusion/datasets/video/OpenVid-0.4M/OpenVidHD_top200000.csv",
        latent_path="/scratch/s224075134/temporal_diffusion/datasets/video/OpenVid-0.4M/latents",
        prompt_latent_path="/scratch/s224075134/temporal_diffusion/datasets/video/OpenVid-0.4M/prompts_latents",
        drop_missing=True,
        return_tokens_if_available=False,
    )
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print("Sample keys:", sample.keys())
    print("Video latent shape:", sample["video_latent"].shape)
    print("Prompt latent shape:", sample["prompt_latent"].shape)
    if "input_ids" in sample:
        print("Input IDs shape:", sample["input_ids"].shape)
    if "attention_mask" in sample:
        print("Attention mask shape:", sample["attention_mask"].shape)