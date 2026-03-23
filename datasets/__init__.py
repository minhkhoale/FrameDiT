from torchvision import transforms
from datasets import video_transforms
from .sky_datasets import Sky
from .sky_latent_datasets import SkyLatent
from .sky_image_datasets import SkyImages
from .sky_image_latent_datasets import SkyImagesLatent
from .ucf101_datasets import UCF101
from .ucf101_image_datasets import UCF101Images
from .ucf101_image_latent_datasets import UCF101ImagesLatent
from .ffs_datasets import FaceForensics
from .ffs_latent_datasets import FaceForensicsLatent
from .ffs_image_datasets import FaceForensicsImages
from .ffs_image_latent_datasets import FaceForensicsImageLatent
from .taichi_datasets import Taichi
from .taichi_latent_datasets import TaichiLatent
from .taichi_image_datasets import TaichiImages
from .taichi_image_latent_datasets import TaichiImagesLatent
from .latent_text_video_dataset import LatentTextVideoDataset

all_dataset_classes = {
    'ffs_latent': FaceForensicsLatent,
    'ffs': FaceForensics,
    'ffs_img_latent': FaceForensicsImageLatent,
    'ffs_img': FaceForensicsImages,
    'ucf101': UCF101,
    'ucf101_img_latent': UCF101ImagesLatent,
    'ucf101_img': UCF101Images,
    'taichi_latent': TaichiLatent,
    'taichi': Taichi,
    'taichi_img_latent': TaichiImagesLatent,
    'taichi_img': TaichiImages,
    'sky': Sky,
    'sky_img': SkyImages,
    'sky_latent': SkyLatent,
    'sky_img_latent': SkyImagesLatent,
}

def get_dataset(args):
    # T2V dataset
    if args.dataset.lower() in ['latent_text_video', 'pexels']:
        return LatentTextVideoDataset(**args)

    # support temporal sampling for video datasets
    if args.num_frames is not None and args.frame_interval is not None:
        temporal_sample = video_transforms.TemporalRandomCrop(args.num_frames * args.frame_interval) # 16 1
    else:
        temporal_sample = None

    dataset_name = args.dataset.lower()
    if args.load_latent:
        dataset_name += '_latent'

    dataset_class = all_dataset_classes[dataset_name]

    if args.load_latent:
        transform = transforms.Compose([])
    else:
        match args.dataset:
            case 'ffs' | 'ffs_img' | 'ucf101' | 'ucf101_img':
                transform = transforms.Compose([
                    video_transforms.ToTensorVideo(), # TCHW
                    video_transforms.RandomHorizontalFlipVideo(),
                    video_transforms.UCFCenterCropVideo(args.image_size),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
                ])
            case 'taichi' | 'taichi_img':
                transform = transforms.Compose([
                    video_transforms.ToTensorVideo(), # TCHW
                    video_transforms.CenterCropResizeVideo(args.image_size),
                    video_transforms.RandomHorizontalFlipVideo(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
                ])
            case 'sky' | 'sky_img':
                transform = transforms.Compose([
                    video_transforms.ToTensorVideo(),
                    video_transforms.CenterCropResizeVideo(args.image_size),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
                ])

    return dataset_class(args, transform=transform, temporal_sample=temporal_sample)