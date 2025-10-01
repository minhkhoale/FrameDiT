import sys; sys.path.extend(['.', 'tools'])
import torch
from tools import dnnlib
from tools.new_metrics import VideoMetric, SharedVideoMetricModelRegistry
from omegaconf import OmegaConf
import random
from pprint import pprint
from tqdm import tqdm

NUM_FRAMES_IN_BATCH = {64: 256, 128: 128, 256: 128, 512: 64, 1024: 32}

def check_if_video_folder(path: str) -> bool:
    # if folder contains all video files, return True
    import os
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    if not os.path.isdir(path):
        return False
    for fname in os.listdir(path):
        if not any(fname.lower().endswith(ext) for ext in video_extensions):
            return False
    return True


def assert_video(video: torch.Tensor):
    assert isinstance(video, torch.Tensor)
    assert video.ndim == 5
    assert video.shape[2] == 3
    assert video.dtype == torch.uint8
    assert video.min() >= 0 and video.max() <= 255


def cal_metrics(
        metrics, 
        real_data_path, 
        fake_data_path, 
        mirror, 
        resolution, 
        verbose, 
        num_frames=16,
        realdata_subsample_factor=6,
        gendata_subsample_factor=1,
        result_file=None,
    ):
    dnnlib.util.Logger(should_flush=True)
    args = dnnlib.EasyDict(metrics=metrics, verbose=verbose)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    registry = SharedVideoMetricModelRegistry()
    metrics = VideoMetric(
        registry,
        args.metrics,
        split_batch_size=16,
        torchmetrics_kwargs={'sync_on_compute': False}
    )
    metrics = metrics.to(device)
    
    dataset_cfg = OmegaConf.create({'max_num_frames': 10000})

    class_name = 'tools.utils.dataset.VideoDataset' if check_if_video_folder(real_data_path) else 'tools.utils.dataset.VideoFramesFolderDataset'
    args.dataset_kwargs = dnnlib.EasyDict(
        class_name=class_name,
        path=real_data_path,
        cfg=dataset_cfg,
        load_n_consecutive=num_frames,
        subsample_factor=realdata_subsample_factor,
        discard_short_videos=True,
        xflip=mirror,
        resolution=resolution,
        use_labels=False,
    )
    class_name = 'tools.utils.dataset.VideoDataset' if check_if_video_folder(fake_data_path) else 'tools.utils.dataset.VideoFramesFolderDataset'
    args.gen_dataset_kwargs = dnnlib.EasyDict(
        class_name=class_name,
        path=fake_data_path,
        cfg=dataset_cfg,
        load_n_consecutive=num_frames,
        load_n_consecutive_random_offset=False,
        subsample_factor=gendata_subsample_factor,
        xflip=False,
        resolution=resolution,
        use_labels=False,
    )
    args.generator_as_dataset = True

    # Print dataset options.
    if args.verbose:
        print('Real data options:')
        print(args.dataset_kwargs)

        print('Fake data options:')
        print(args.gen_dataset_kwargs)

    real_dataset = dnnlib.util.construct_class_by_name(**args.dataset_kwargs)  # subclass of torch.utils.data.Dataset
    fake_dataset = dnnlib.util.construct_class_by_name(**args.gen_dataset_kwargs)

    # data_length = len(real_dataset)
    num_items = len(fake_dataset)
    # item_subset = random.sample(range(data_length), num_items) # added by xin, randomly selected 2048 videos

    print('num_items:', num_items)
    real_loader = torch.utils.data.DataLoader(
        dataset=real_dataset,
        sampler=random.sample(range(len(real_dataset)), num_items),
        batch_size=NUM_FRAMES_IN_BATCH[resolution],
        num_workers=4
    )
    fake_loader = torch.utils.data.DataLoader(
        dataset=fake_dataset,
        sampler=random.sample(range(len(fake_dataset)), num_items),
        batch_size=NUM_FRAMES_IN_BATCH[resolution],
        num_workers=4
    )

    for real_batch, fake_batch in tqdm(zip(real_loader, fake_loader)):
        assert_video(real_batch['image'])
        assert_video(fake_batch['image'])
    
        real_videos = real_batch['image'].to(device).float() / 255.0
        fake_videos = fake_batch['image'].to(device).float() / 255.0
        metrics(real_videos, fake_videos)
    
    result = metrics.log('final') # dict
    max_key_len = max(len(k) for k in result)
    finals = {}
    for k, v in result.items():
        finals[k] = v.cpu().item() if isinstance(v, torch.Tensor) else v
        print(f"{k:<{max_key_len}} : {finals[k]:.6f}")

    pprint(finals)
    if result_file is not None:
        import json
        with open(result_file, 'w') as f:
            json.dump(finals, f, indent=4)
        print(f'Saved results to {result_file}')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculate metrics between two video datasets.")
    parser.add_argument('--real_data_path', type=str, required=True, help='Path to the real videos')
    parser.add_argument('--fake_data_path', type=str, required=True, help='Path to the fake videos')
    parser.add_argument('--mirror', action='store_true', help='Whether to apply x-flip augmentation')
    parser.add_argument('--resolution', type=int, default=256, help='Resolution of the videos')
    parser.add_argument('--verbose', action='store_true', help='Whether to print out more information')
    parser.add_argument('--result_file', type=str, help='Path to save the results')
    args = parser.parse_args()

    metrics = ['vbench', 'fvd', 'is', 'fid', 'lpips', 'mse', 'ssim', 'psnr']
    cal_metrics(metrics=metrics, real_data_path=args.real_data_path, fake_data_path=args.fake_data_path, mirror=args.mirror, resolution=args.resolution, verbose=args.verbose, result_file=args.result_file)


       
    
