from torchvision import transforms
from datasets import video_transforms
from .sky_datasets import Sky
from .sky_latent_datasets import SkyLatent
from .sky_image_datasets import SkyImages
from .sky_image_latent_datasets import SkyImagesLatent
from .bair_datasets import BAIR
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


def get_dataset(args):
    temporal_sample = video_transforms.TemporalRandomCrop(args.num_frames * args.frame_interval) # 16 1

    match args.dataset:
        case 'ffs':
            if args.load_latent:
                transform = transforms.Compose([])
                datset_class = FaceForensicsLatent
            else:
                transform = transforms.Compose([
                    video_transforms.ToTensorVideo(), # TCHW
                    video_transforms.RandomHorizontalFlipVideo(),
                    video_transforms.UCFCenterCropVideo(args.image_size),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
                ])
                datset_class = FaceForensics
        case 'ffs_img':
            if args.load_latent:
                transform = transforms.Compose([])
                datset_class = FaceForensicsImageLatent
            else:
                transform = transforms.Compose([
                    video_transforms.ToTensorVideo(), # TCHW
                    video_transforms.RandomHorizontalFlipVideo(),
                    video_transforms.UCFCenterCropVideo(args.image_size),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
                ])
                datset_class = FaceForensicsImages
        case 'ffs_whole':
            transform = transforms.Compose([
                video_transforms.ToTensorVideo(), # TCHW
                video_transforms.UCFCenterCropVideo(args.image_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
            temporal_sample = None
            datset_class = FaceForensics
        case 'ucf101':
            if args.load_latent:
                transform = transforms.Compose([])
                datset_class = UCF101Latent
            else:
                transform = transforms.Compose([
                    video_transforms.ToTensorVideo(), # TCHW
                    video_transforms.RandomHorizontalFlipVideo(),
                    video_transforms.UCFCenterCropVideo(args.image_size),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
                ])
                datset_class = UCF101
        case 'ucf101_img':
            if args.load_latent:
                transform = transforms.Compose([])
                datset_class = UCF101ImagesLatent
            else:
                transform = transforms.Compose([
                    video_transforms.ToTensorVideo(), # TCHW
                    video_transforms.RandomHorizontalFlipVideo(),
                    video_transforms.UCFCenterCropVideo(args.image_size),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
                ])
                datset_class = UCF101Images
        case 'ucf101_whole':
            transform = transforms.Compose([
                video_transforms.ToTensorVideo(), # TCHW
                video_transforms.UCFCenterCropVideo(args.image_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
            temporal_sample = None
            datset_class = UCF101
        case 'taichi':
            if args.load_latent:
                transform = transforms.Compose([])
                datset_class = TaichiLatent
            else:
                if args.image_size < 256:
                    transform = transforms.Compose([
                        video_transforms.ToTensorVideo(), # TCHW
                        video_transforms.CenterCropResizeVideo(args.image_size),
                        video_transforms.RandomHorizontalFlipVideo(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
                    ])
                else:
                    transform = transforms.Compose([
                        video_transforms.ToTensorVideo(), # TCHW
                        video_transforms.RandomHorizontalFlipVideo(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
                    ])
                datset_class = Taichi
        case 'taichi_img':
            if args.load_latent:
                transform = transforms.Compose([])
                datset_class = TaichiImagesLatent
            else:
                if hasattr(args, 'flip_aug') and args.flip_aug:
                    transform = transforms.Compose([
                        video_transforms.ToTensorVideo(), # TCHW
                        video_transforms.RandomHorizontalFlipVideo(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
                    ])
                else:
                    transform = transforms.Compose([
                        video_transforms.ToTensorVideo(), # TCHW
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
                    ])
                datset_class = TaichiImages
        case 'taichi_whole':
            if args.image_size < 256:
                transform = transforms.Compose([
                    video_transforms.ToTensorVideo(), # TCHW
                    video_transforms.CenterCropResizeVideo(args.image_size),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
                ])
            else:
                transform = transforms.Compose([
                    video_transforms.ToTensorVideo(), # TCHW
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
                ])
            temporal_sample = None
            datset_class = TaichiPreprocess
        case 'sky':
            if args.load_latent:
                transform = transforms.Compose([])
                datset_class = SkyLatent
            else:
                transform = transforms.Compose([
                    video_transforms.ToTensorVideo(),
                    video_transforms.CenterCropResizeVideo(args.image_size),
                    # video_transforms.RandomHorizontalFlipVideo(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
                ])
                datset_class = Sky
        case 'sky_img':
            if args.load_latent:
                transform = transforms.Compose([])
                datset_class = SkyImagesLatent
            else:
                transform = transforms.Compose([
                    video_transforms.ToTensorVideo(),
                    video_transforms.CenterCropResizeVideo(args.image_size),
                    # video_transforms.RandomHorizontalFlipVideo(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
                ])
                datset_class = SkyImages
        case 'sky_whole':
            transform = transforms.Compose([
                video_transforms.ToTensorVideo(),
                video_transforms.CenterCropResizeVideo(args.image_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
            temporal_sample = None
            datset_class = Sky
        case 'bair':
            if args.load_latent:
                transform = transforms.Compose([])
            else:
                transform = transforms.Compose([
                    video_transforms.ToTensorVideo(), # TCHW
                    video_transforms.RandomHorizontalFlipVideo(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
                ])
            return BAIR(args, transform=transform, temporal_sample=temporal_sample)
        
        case _:
            raise NotImplementedError(args.dataset)

    print('transform', transform)
    return datset_class(args, transform=transform, temporal_sample=temporal_sample)
