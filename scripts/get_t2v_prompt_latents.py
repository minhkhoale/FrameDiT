"""
Preprocess prompts (text) and encode them with a T5 text encoder.

Typical use for T2V training:
- Read your metadata CSV (e.g., OpenVidHD.csv)
- Select top-N rows (optionally sorted by aesthetic score)
- Clean/normalize the prompt string
- Tokenize with T5Tokenizer
- Encode with T5EncoderModel to get hidden states
- Save per-sample embeddings (and optionally input_ids/attn_mask)

Outputs (choose one):
A) One file per sample:   SAVE_DIR/<id>.pt  (good for sharded dataloading)
B) One big shard file:    SAVE_DIR/text_embeds_shard_000.pt ... (good for faster FS)

This script does NOT depend on your video zip pipeline; you can run it independently.
"""

import os
import re
import math
import argparse
from typing import Optional, List, Dict

import torch
import pandas as pd
from transformers import T5Tokenizer, T5EncoderModel


# ----------------------------
# Prompt preprocessing
# ----------------------------
_ws_re = re.compile(r"\s+")

def clean_prompt(
    s: str,
    lower: bool = False,
    strip: bool = True,
    collapse_ws: bool = True,
    max_chars: Optional[int] = None,
) -> str:
    if s is None:
        return ""
    s = str(s)
    if strip:
        s = s.strip()
    if collapse_ws:
        s = _ws_re.sub(" ", s)
    if lower:
        s = s.lower()
    if max_chars is not None and max_chars > 0:
        s = s[:max_chars]
    return s


# ----------------------------
# Saving helpers
# ----------------------------
def safe_id(x: str) -> str:
    # make filename-safe id (for per-sample saving)
    x = str(x)
    x = x.replace("/", "_").replace("\\", "_")
    return re.sub(r"[^0-9a-zA-Z._-]+", "_", x)

def save_per_sample(
    out_dir: str,
    sample_ids: List[str],
    prompts: List[str],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    embeds: torch.Tensor,
    save_tokens: bool,
    save_pool: str,
):
    os.makedirs(out_dir, exist_ok=True)

    # embeds: (B, L, D)
    # pool: "none" -> store full seq; "cls" -> first token; "mean" -> masked mean
    if save_pool == "cls":
        pooled = embeds[:, 0]  # (B,D)
    elif save_pool == "mean":
        mask = attention_mask.unsqueeze(-1).to(embeds.dtype)  # (B,L,1)
        pooled = (embeds * mask).sum(dim=1) / (mask.sum(dim=1).clamp(min=1.0))
    else:
        pooled = None

    for i, sid in enumerate(sample_ids):
        path = os.path.join(out_dir, f"{safe_id(sid)}.pt")
        obj = {
            "id": sid,
            "prompt": prompts[i],
        }
        if save_pool == "none":
            obj["embeds"] = embeds[i].cpu()
        else:
            obj["embeds"] = pooled[i].cpu()

        if save_tokens:
            obj["input_ids"] = input_ids[i].cpu()
            obj["attention_mask"] = attention_mask[i].cpu()

        torch.save(obj, path)

def save_sharded(
    out_dir: str,
    shard_idx: int,
    sample_ids: List[str],
    prompts: List[str],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    embeds: torch.Tensor,
    save_tokens: bool,
    save_pool: str,
):
    os.makedirs(out_dir, exist_ok=True)

    if save_pool == "cls":
        embeds = embeds[:, 0]  # (B,D)
    elif save_pool == "mean":
        mask = attention_mask.unsqueeze(-1).to(embeds.dtype)
        embeds = (embeds * mask).sum(dim=1) / (mask.sum(dim=1).clamp(min=1.0))
    # else keep (B,L,D)

    obj = {
        "ids": sample_ids,
        "prompts": prompts,
        "embeds": embeds.cpu(),
    }
    if save_tokens:
        obj["input_ids"] = input_ids.cpu()
        obj["attention_mask"] = attention_mask.cpu()

    path = os.path.join(out_dir, f"text_embeds_shard_{shard_idx:04d}.pt")
    torch.save(obj, path)


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    # columns
    parser.add_argument("--prompt_col", type=str, default="caption", help="Column name containing prompt text.")
    parser.add_argument("--id_col", type=str, default="video", help="Unique id per row (e.g., video filename).")
    parser.add_argument("--score_col", type=str, default="aesthetic score")
    parser.add_argument("--n_rows", type=int, default=200_000)

    # prompt preprocessing
    parser.add_argument("--lower", action="store_true")
    parser.add_argument("--max_chars", type=int, default=0)

    # T5 config
    parser.add_argument("--t5_name", type=str, default="google/t5-v1_1-base")
    parser.add_argument("--max_length", type=int, default=128)

    # batching / device
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true")

    # output mode
    parser.add_argument("--save_mode", type=str, default="per_sample", choices=["per_sample", "sharded"])
    parser.add_argument("--shard_size", type=int, default=50_000, help="Rows per shard when save_mode=sharded.")
    parser.add_argument("--save_tokens", action="store_true", help="Also save input_ids & attention_mask.")
    parser.add_argument("--pool", type=str, default="none", choices=["none", "cls", "mean"],
                        help="How to store embeddings: none=(L,D), cls=(D), mean=(D).")

    args = parser.parse_args()

    assert args.max_length > 0
    os.makedirs(args.out_dir, exist_ok=True)

    # Load CSV
    df = pd.read_csv(args.csv_path)
    if args.prompt_col not in df.columns:
        raise ValueError(f"Missing prompt_col='{args.prompt_col}'. Columns={df.columns.tolist()}")
    if args.id_col not in df.columns:
        raise ValueError(f"Missing id_col='{args.id_col}'. Columns={df.columns.tolist()}")

    if args.score_col in df.columns:
        df = df.sort_values(by=args.score_col, ascending=False)

    df = df.head(args.n_rows).copy()

    # Clean prompt
    max_chars = args.max_chars if args.max_chars > 0 else None
    df[args.prompt_col] = df[args.prompt_col].apply(
        lambda x: clean_prompt(x, lower=args.lower, max_chars=max_chars)
    )

    # Prepare model/tokenizer
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Loading T5 tokenizer/model: {args.t5_name}  device={device} fp16={args.fp16}")
    # tokenizer = T5Tokenizer.from_pretrained(args.t5_name)
    # model = T5EncoderModel.from_pretrained(args.t5_name).to(device)
    # from transformers import T5EncoderModel, T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained("maxin-cn/Latte-1", subfolder='tokenizer')
    model = T5EncoderModel.from_pretrained("maxin-cn/Latte-1", subfolder='text_encoder').to(device)
    model.eval()
    if args.fp16:
        model = model.half()

    # Iterate in batches
    total = len(df)
    print(f"Encoding {total} prompts from {args.csv_path}")

    shard_idx = 0
    shard_buf = {
        "ids": [],
        "prompts": [],
        "input_ids": [],
        "attention_mask": [],
        "embeds": [],
    }

    @torch.no_grad()
    def encode_batch(prompts: List[str]):
        toks = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        input_ids = toks["input_ids"].to(device)
        attention_mask = toks["attention_mask"].to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        # last_hidden_state: (B, L, D)
        embeds = out.last_hidden_state
        return input_ids, attention_mask, embeds

    for start in range(0, total, args.batch_size):
        end = min(start + args.batch_size, total)
        batch = df.iloc[start:end]
        ids = batch[args.id_col].astype(str).tolist()
        prompts = batch[args.prompt_col].astype(str).tolist()

        input_ids, attention_mask, embeds = encode_batch(prompts)

        if args.save_mode == "per_sample":
            save_per_sample(
                out_dir=args.out_dir,
                sample_ids=ids,
                prompts=prompts,
                input_ids=input_ids,
                attention_mask=attention_mask,
                embeds=embeds,
                save_tokens=args.save_tokens,
                save_pool=args.pool,
            )
        else:
            # accumulate into shard buffer
            shard_buf["ids"].extend(ids)
            shard_buf["prompts"].extend(prompts)

            # store on CPU to keep VRAM low
            shard_buf["embeds"].append(embeds.cpu())
            if args.save_tokens:
                shard_buf["input_ids"].append(input_ids.cpu())
                shard_buf["attention_mask"].append(attention_mask.cpu())

            # flush shard if large enough
            if len(shard_buf["ids"]) >= args.shard_size:
                all_embeds = torch.cat(shard_buf["embeds"], dim=0)
                if args.save_tokens:
                    all_input_ids = torch.cat(shard_buf["input_ids"], dim=0)
                    all_attn = torch.cat(shard_buf["attention_mask"], dim=0)
                else:
                    all_input_ids = torch.empty(0)
                    all_attn = torch.empty(0)

                save_sharded(
                    out_dir=args.out_dir,
                    shard_idx=shard_idx,
                    sample_ids=shard_buf["ids"],
                    prompts=shard_buf["prompts"],
                    input_ids=all_input_ids,
                    attention_mask=all_attn,
                    embeds=all_embeds,
                    save_tokens=args.save_tokens,
                    save_pool=args.pool,
                )
                shard_idx += 1
                shard_buf = {"ids": [], "prompts": [], "input_ids": [], "attention_mask": [], "embeds": []}

        if (start // args.batch_size) % 20 == 0:
            print(f"progress: {end}/{total}", end="\r")

    # Flush remaining shard buffer
    if args.save_mode == "sharded" and len(shard_buf["ids"]) > 0:
        all_embeds = torch.cat(shard_buf["embeds"], dim=0)
        if args.save_tokens:
            all_input_ids = torch.cat(shard_buf["input_ids"], dim=0)
            all_attn = torch.cat(shard_buf["attention_mask"], dim=0)
        else:
            all_input_ids = torch.empty(0)
            all_attn = torch.empty(0)

        save_sharded(
            out_dir=args.out_dir,
            shard_idx=shard_idx,
            sample_ids=shard_buf["ids"],
            prompts=shard_buf["prompts"],
            input_ids=all_input_ids,
            attention_mask=all_attn,
            embeds=all_embeds,
            save_tokens=args.save_tokens,
            save_pool=args.pool,
        )

    print(f"\nDone. Outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()


"""
python scripts/get_t2v_prompt_latents.py \
  --csv_path /scratch/s224075134/temporal_diffusion/datasets/video/OpenVid-0.4M/OpenVidHD.csv \
  --out_dir /scratch/s224075134/temporal_diffusion/datasets/video/OpenVid-0.4M/prompts_latents \
  --prompt_col caption \
  --id_col video \
  --max_length 128 \
  --batch_size 128 \
  --fp16 \
  --save_mode per_sample \
  --pool none
"""