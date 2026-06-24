import json
from operator import __getitem__
import os
from os.path import join as opj
import subprocess
import torchvision
import torch
import argparse
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from diffusers import AutoencoderKL
from huggingface_hub import snapshot_download
import importlib
from einops import rearrange
import inspect
import decord
from decord import VideoReader, cpu
from concurrent.futures import ThreadPoolExecutor, as_completed
decord.bridge.set_bridge('torch')


# def is_image_vae(vae):
#     # check if the vae is an image vae by checking if it has a 'decode' method that takes in (B, C, H, W) and outputs (B, C, H, W)
#     if hasattr(vae, "decode"):
#         decode_signature = inspect.signature(vae.decode)
#         parameters = decode_signature.parameters
#         if len(parameters) == 1:
#             param = next(iter(parameters.values()))
#             if param.annotation == torch.Tensor:
#                 return True
#     return False


def _is_tensor_video_clip(clip):
    if not torch.is_tensor(clip):
        raise TypeError("clip should be Tensor. Got %s" % type(clip))

    if not clip.ndimension() == 4:
        raise ValueError("clip should be 4D. Got %dD" % clip.dim())

    return True

def to_tensor(clip):
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (T, C, H, W)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
    """
    _is_tensor_video_clip(clip)
    if not clip.dtype == torch.uint8:
        raise TypeError("clip tensor should have data type uint8. Got %s" % str(clip.dtype))
    # return clip.float().permute(3, 0, 1, 2) / 255.0
    return clip.float() / 255.0

def crop(clip, i, j, h, w):
    """
    Args:
        clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
    """
    if len(clip.size()) != 4:
        raise ValueError("clip should be a 4D tensor")
    return clip[..., i : i + h, j : j + w]

def center_crop_using_short_edge(clip):
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    h, w = clip.size(-2), clip.size(-1)
    if h < w:
        th, tw = h, h
        i = 0
        j = int(round((w - tw) / 2.0))
    else:
        th, tw = w, w
        i = int(round((h - th) / 2.0))
        j = 0
    return crop(clip, i, j, th, tw)

def resize(clip, target_size, interpolation_mode):
    if len(target_size) != 2:
        raise ValueError(f"target size should be tuple (height, width), instead got {target_size}")
    return torch.nn.functional.interpolate(clip, size=target_size, mode=interpolation_mode, align_corners=False)


class CenterCropResizeVideo:
    '''
    First use the short side for cropping length, 
    center crop video, then resize to the specified size
    '''
    def __init__(
        self,
        size,
        interpolation_mode="bilinear",
    ):
        if isinstance(size, tuple):
            if len(size) != 2:
                raise ValueError(f"size should be tuple (height, width), instead got {size}")
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation_mode = interpolation_mode
       

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: scale resized / center cropped video clip.
                size is (T, C, crop_size, crop_size)
        """
        clip_center_crop = center_crop_using_short_edge(clip)
        clip_center_crop_resize = resize(clip_center_crop, target_size=self.size, interpolation_mode=self.interpolation_mode)
        return clip_center_crop_resize

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, interpolation_mode={self.interpolation_mode})"

class ToTensorVideo:
    """
    Convert tensor data type from uint8 to float, divide value by 255.0 and
    permute the dimensions of clip tensor
    """

    def __init__(self):
        pass

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor, dtype=torch.uint8): Size is (T, C, H, W)
        Return:
            clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
        """
        return to_tensor(clip)

    def __repr__(self) -> str:
        return self.__class__.__name__

class VideoCaptionDataset(Dataset):
    def __init__(self, data_root, shard_size, output_path, tokenizer=None):
        super().__init__()
        self.data_root = data_root
        self.data_merge_path = opj(data_root, "merge.txt")
        self.data_video_path = opj(data_root, "videos")
        self.shard_size = shard_size
        self.output_path = output_path
        self.tokenizer = tokenizer
        self.n_frames = 16
        self.train_fps = 8
        self.height = 512
        self.width = 512
        self.text_max_length = 120

        # resize by the shorter edge to 512, and then center crop to 512x512
        self.transform = torchvision.transforms.Compose([
            ToTensorVideo(),
            CenterCropResizeVideo(size=(self.height, self.width), interpolation_mode="bilinear"),
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        self.raw_data = self.load_raw_data()


    def __getitem__(self, index):
        item = self.raw_data[index]
        video_path = item["path"]
        print(f"[idx={index}] Processing video: {video_path}, num frames: {item['num_frames']}, sample_frame_index: {item['sample_frame_index']}")

        # read video
        try:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            video = vr.get_batch(item['sample_frame_index']).permute(0, 3, 1, 2) # (T, C, H, W)
            video = self.transform(video)
            item['pixel_values'] = video

            # remove  · Free Stock Video from text
            if '· Free Stock Video' in item['prompt']:
                item['prompt'] = item['prompt'].replace('· Free Stock Video', '').strip()
            item['video_path'] = video_path
            item['video_name'] = os.path.basename(video_path)[:-4]


            final_item = {
                'video_id': item['video_id'],
                'video_name': item['video_name'],
                'pixel_values': item['pixel_values'],
                'prompt': item['prompt'],
            }
            # release video reader resources
            del vr
            assert all(value is not None for value in final_item.values()), "Some values in final_item are None"

            return final_item
        except Exception as e:
            print(f"[idx={index}] Error processing video {video_path}: {repr(e)}")
            return None

    def __len__(self):
        return len(self.raw_data)

    def load_raw_data(self):
        with open(self.data_merge_path) as f:
            folder_anno_pairs = [
                line.strip().split(",") for line in f if line.strip()
            ]
        assert len(
            folder_anno_pairs) == 1, "Only support one folder-annotation pair"
        assert len(folder_anno_pairs[0]
                   ) == 2, "Folder-annotation pair should have two elements"
        folder, annotation_file = folder_anno_pairs[0]

        annotation_file = opj(self.data_root, os.path.basename(annotation_file))

        data_items: list[dict] = []
        with open(annotation_file) as f:
            data_items = json.load(f)

        # Update paths with folder prefix
        filtered_items = []
        total_discarded = {
            'no_cap': 0,
            'no_resolution': 0,
            'invalid_resolution': 0,
            'too_long_video': 0,
            'too_short_video': 0,
            'not_exist_video': 0,
            'existing_latent': 0,
        }
        for idx, item in enumerate(data_items):
            # if absolute path, keep it; if relative path, add folder prefix
            if not os.path.isabs(item["path"]):
                item["path"] = opj(self.data_video_path, item["path"])

            if "action_path" in item and item["action_path"] and not os.path.isabs(item["action_path"]):
                item["action_path"] = opj(self.data_video_path, item["action_path"])

            # check if video path exists
            if not os.path.exists(item["path"]):
                total_discarded['not_exist_video'] += 1
                continue
            
            # filtering out
            if 'cap' not in item:
                total_discarded['no_cap'] += 1
                continue
            if 'resolution' not in item:
                total_discarded['no_resolution'] += 1
                continue

            if 'height' not in item['resolution'] or 'width' not in item['resolution']:
                total_discarded['invalid_resolution'] += 1
                continue

            if os.path.exists(get_output_path_pt(self.output_path, self.shard_size, item)):
                total_discarded['existing_latent'] += 1
                continue

            # sample first n_frames frames
            fps = item.get('fps', self.train_fps)
            frame_interval = max(1, int(round(fps / self.train_fps)))
            item['sample_frame_index'] = list(range(0, item['num_frames'], frame_interval))[:self.n_frames]
            
            # discard too long video
            if item['duration'] > 5*item['num_frames']/self.train_fps*1.0:
                total_discarded['too_long_video'] += 1
                continue

            # discard too short video
            if len(item['sample_frame_index']) < self.n_frames or item['num_frames'] < item['sample_frame_index'][-1] + 1:
                total_discarded['too_short_video'] += 1
                continue
            
            if 'video_id' not in item:
                item['video_id'] = idx
                
            filtered_items.append({
                'video_id': item['video_id'],
                'path': item['path'],
                'duration': item['duration'],
                'num_frames': item['num_frames'],
                'prompt': item['cap'],
                'resolution': item['resolution'],
                'sample_frame_index': item['sample_frame_index'],
                'fps': fps,
                'action_path': item.get('action_path', None),
            })

        print(f"Discarded {total_discarded['no_cap']} items without captions, {total_discarded['no_resolution']} items without resolution, {total_discarded['invalid_resolution']} items with invalid resolution, and {total_discarded['too_long_video']} items with too long duration, {total_discarded['too_short_video']} items with too short duration, {total_discarded['not_exist_video']} items with non-existing video path, {total_discarded['existing_latent']} items with existing latent file. Total discarded: {sum(total_discarded.values())}. Total remaining: {len(filtered_items)}.")

        return filtered_items

def collate_fn(batch):
    # filter out None items
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.default_collate(batch)    


def get_class(module_name: str, class_name: str):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def load_models(model_path, modules=['text_encoder', 'tokenizer', 'vae'], eval=True, device='cuda:0'):
    # if model_path not exist
    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist, downloading from HuggingFace Hub...")
        model_path = snapshot_download(repo_id=model_path)
        print(f"Model downloaded to {model_path}")

    model_config_path = opj(model_path, "model_index.json")
    model_configs = json.load(open(model_config_path))

    models = {}
    for module in modules:
        if module not in model_configs:
            raise ValueError(f"Module {module} not found in model config")
    
        package, model_name = model_configs[module]
        model_class = get_class(package, model_name)
        model = model_class.from_pretrained(model_path, subfolder=module)
        if hasattr(model, "to"):
            model = model.to(device)
        models[module] = model
        if eval and hasattr(model, "eval"):
            model.eval()
    
    return models

def atomic_torch_save(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)  # atomic rename

def save_many(items, max_workers=8, max_pending=None):
    """
    items: iterable of (obj, path)
    """
    if max_pending is None:
        max_pending = max_workers * 4

    pending = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for obj, path in items:
            pending.append(ex.submit(atomic_torch_save, obj, path))

            # throttle
            if len(pending) >= max_pending:
                for fut in pending[:max_workers]:
                    fut.result()
                pending = pending[max_workers:]

        # flush remaining + raise exceptions
        for fut in as_completed(pending):
            fut.result()

def get_output_path_pt(output_path, shard_size, item):
    video_id = int(item['video_id'])
    video_name = os.path.basename(item['path'])[:-4]
    shard_idx = video_id // shard_size
    shard_dir = opj(output_path, f"{shard_idx:04d}")
    # os.makedirs(shard_dir, exist_ok=True)
    output_file = opj(shard_dir, video_name + ".pt")
    return output_file


@torch.inference_mode()
def main():
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)   # put at top-level under if __name__ == "__main__": ideally

    parser = argparse.ArgumentParser(description="Process raw videos into latents for training")
    parser.add_argument("--data_root", type=str, required=True, help="Path to the data root directory")
    parser.add_argument("--model_path", type=str, default="maxin-cn/Latte-1", help="Path to the pretrained model for VAE")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the processing on")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the processed latents")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for processing")
    parser.add_argument("--shard-size", type=int, default=1000, help="Number of videos per shard for pexels dataset")
    args = parser.parse_args()

    device = torch.device(args.device)
    models = load_models(args.model_path, modules=['vae', 'text_encoder', 'tokenizer'], eval=True, device=device)

    dataset = VideoCaptionDataset(args.data_root, args.shard_size, args.output_path, tokenizer=models['tokenizer'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, prefetch_factor=None, pin_memory=False, collate_fn=collate_fn)

    # create output directory if not exist
    os.makedirs(args.output_path, exist_ok=True)

    num_processed_samples = 0
    from tqdm import tqdm
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        if batch is None:
            continue

        pixel_values = batch['pixel_values'].to(device) # (T, C, H, W)
        # input_ids = batch['input_ids'].to(device) # (B, seq_len)
        # attention_mask = batch['attention_mask'].to(device) # (B, seq_len)
        prompts = batch['prompt'] # list of str

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            # if image vae, then reshape B, T, C, H, W to (B*T, C, H, W)
            pixel_values_vae = rearrange(pixel_values, "B T C H W -> (B T) C H W")
            latent = models['vae'].encode(pixel_values_vae).latent_dist.mean
            latent = rearrange(latent, "(B T) C H W -> B T C H W", B=pixel_values.size(0))

            text_inputs = models['tokenizer'](
                prompts,
                padding="max_length",      # or "longest"
                truncation=True,
                max_length=120,
                return_attention_mask=True,
                return_tensors="pt",
            )
            input_ids = text_inputs.input_ids.to(device)
            attention_mask = text_inputs.attention_mask.to(device)
            text_embeddings = models['text_encoder'](
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state.cpu() # (B, seq_len, hidden_size)
        
        for j in range(latent.size(0)):
            video_id = batch['video_id'][j]
            output_item = {
                "video_name": batch['video_name'][j],
                "video_id": video_id,
                "latent": latent[j].cpu(),
                'text': batch['prompt'][j],
                "input_ids": input_ids[j].cpu(),
                "attention_mask": attention_mask[j].cpu(),
                "text_embedding": text_embeddings[j],
            }
            shard_idx = video_id // args.shard_size
            shard_dir = opj(args.output_path, f"{shard_idx:04d}")
            os.makedirs(shard_dir, exist_ok=True)
            output_file = opj(shard_dir, batch['video_name'][j] + ".pt")


            # print(f"Saving processed latent for video {batch['video_name'][j]} (video_id: {video_id}) to {output_file}")
            torch.save(output_item, output_file)

        num_processed_samples += latent.size(0)

        del pixel_values
        del latent
        del text_embeddings
        del input_ids
        del attention_mask
        del text_inputs

if __name__ == "__main__":
    main()


"""
python scripts/process_latents.py \
    --data_root /scratch/s224075134/temporal_diffusion/datasets/video/crush-smol-merged \
    --model_path maxin-cn/Latte-1 \
    --output_path /scratch/s224075134/temporal_diffusion/datasets/video/crush-smol-merged/crush-smol_processed_t2v \
    --device cuda:0
"""

"""
python scripts/process_latents.py \
    --data_root /scratch/s224075134/temporal_diffusion/datasets/video/pexels400k \
    --model_path maxin-cn/Latte-1 \
    --output_path /scratch/s224075134/temporal_diffusion/datasets/video/pexels400k/pexels400k_processed_t2v \
    --device cuda:0
    --num_workers 4
"""