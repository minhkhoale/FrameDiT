import os, io
import re
import json
import torch
import decord
import torchvision
import numpy as np
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution


from PIL import Image
from einops import rearrange
from typing import Dict, List, Tuple
from torchvision import transforms
import random


class_labels_map = None
cls_sample_cnt = None

class_labels_map = None
cls_sample_cnt = None


def temporal_sampling(frames, start_idx, end_idx, num_samples):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        frames (tensor): a tensor of video frames, dimension is
            `num video frames` x `channel` x `height` x `width`.
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """
    index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, frames.shape[0] - 1).long()
    frames = torch.index_select(frames, 0, index)
    return frames


def get_filelist(file_path):
    Filelist = []
    for home, dirs, files in os.walk(file_path):
        for filename in files:
            Filelist.append(os.path.join(home, filename))
            # Filelist.append( filename)
    return Filelist


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(num_class, anno_pth='./k400_classmap.json'):
    global class_labels_map, cls_sample_cnt
    
    if class_labels_map is not None:
        return class_labels_map, cls_sample_cnt
    else:
        cls_sample_cnt = {}
        class_labels_map = load_annotation_data(anno_pth)
        for cls in class_labels_map:
            cls_sample_cnt[cls] = 0
        return class_labels_map, cls_sample_cnt


def load_annotations(ann_file, num_class, num_samples_per_cls):
    dataset = []
    class_to_idx, cls_sample_cnt = get_class_labels(num_class)
    with open(ann_file, 'r') as fin:
        for line in fin:
            line_split = line.strip().split('\t')
            sample = {}
            idx = 0
            # idx for frame_dir
            frame_dir = line_split[idx]
            sample['video'] = frame_dir
            idx += 1
                                
            # idx for label[s]
            label = [x for x in line_split[idx:]]
            assert label, f'missing label in line: {line}'
            assert len(label) == 1
            class_name = label[0]
            class_index = int(class_to_idx[class_name])
            
            # choose a class subset of whole dataset
            if class_index < num_class:
                sample['label'] = class_index
                if cls_sample_cnt[class_name] < num_samples_per_cls:
                    dataset.append(sample)
                    cls_sample_cnt[class_name]+=1

    return dataset


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


class DecordInit(object):
    """Using Decord(https://github.com/dmlc/decord) to initialize the video_reader."""

    def __init__(self, num_threads=1):
        self.num_threads = num_threads
        self.ctx = decord.cpu(0)
        
    def __call__(self, filename):
        """Perform the Decord initialization.
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        reader = decord.VideoReader(filename,
                                    ctx=self.ctx,
                                    num_threads=self.num_threads)
        return reader

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'sr={self.sr},'
                    f'num_threads={self.num_threads})')
        return repr_str


class UCF101ImagesLatent(torch.utils.data.Dataset):
    """Load the UCF101 video files
    
    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(self,
                 configs,
                 transform=None,
                 temporal_sample=None):
        self.configs = configs
        self.data_path = configs.data_path
        self.transform = transform
        self.temporal_sample = temporal_sample
        self.target_video_len = self.configs.num_frames
        self.v_decoder = DecordInit()
        self.data_all, self.video_frame_all, self.classes, self.class_to_idx = self.load_video_frames_latent(self.data_path)
        print('self.classes', self.classes)
        self.video_num = len(self.data_all)

        random.shuffle(self.video_frame_all)
        self.use_image_num = configs.use_image_num
        self.video_frame_num = len(self.video_frame_all)

    def __getitem__(self, index):
        video_index = index % self.video_num
        video_path, total_frames = self.data_all[video_index]
        print('\nvideo_path', video_path, 'total_frames', total_frames)

        start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
        assert end_frame_ind - start_frame_ind >= self.target_video_len
        frame_indice = np.linspace(start_frame_ind, end_frame_ind-1, self.target_video_len, dtype=int)        

        class_name = self.filename_to_class_name(os.path.basename(video_path))
        print('class_name', class_name)
        class_index = self.class_to_idx[class_name]

        params = torch.load(video_path, weights_only=False)
        gaussian_dist = DiagonalGaussianDistribution(parameters=params)
        latent = gaussian_dist.sample()
        video_clip = latent[frame_indice]
        
        images = []
        image_names_idx = []
        for i in range(self.use_image_num):
            while True:
                try:
                    video_frame_path, frame_id = self.video_frame_all[index+i]
                    params = torch.load(video_frame_path, weights_only=False)[frame_id].unsqueeze(0)
                    gaussian_dist = DiagonalGaussianDistribution(parameters=params)
                    image = gaussian_dist.sample()
                    images.append(image)
                    image_class_name = self.filename_to_class_name(os.path.basename(video_frame_path))
                    image_class_index = self.class_to_idx[image_class_name]
                    image_names_idx.append(str(image_class_index))
                    print(f'image {i} from {video_frame_path}, frame_id {frame_id}, image_class_name {image_class_name}, image_class_index {image_class_index}')
                    break
                except Exception as e:
                    index = random.randint(0, self.video_frame_num - self.use_image_num)
        images =  torch.cat(images, dim=0)
        assert len(images) == self.use_image_num
        assert len(image_names_idx) == self.use_image_num

        image_names_idx = '====='.join(image_names_idx)
        
        video_cat = torch.cat([video_clip, images], dim=0)
    
        return {'video': video_cat, 
                'video_name': class_index, 
                'image_name': image_names_idx}
    
    def load_video_frames_latent(self, dataroot):
        data_all = []
        frames_all = []
        classes = set()
        # Find all mp4 files in dataroot (recursively)
        latent_files = []
        for root, _, files in os.walk(dataroot):
            for file in files:
                if file.lower().endswith('.pt'):
                    latent_files.append(os.path.join(root, file))

        for latent_path in latent_files:
            latent = torch.load(latent_path, weights_only=False)
            n_frames = latent.shape[0]
            if n_frames > 0:
                data_all.append((latent_path, n_frames))
                for i in range(n_frames):
                    frames_all.append((latent_path, i))
                
            class_name = self.filename_to_class_name(os.path.basename(latent_path))
            classes.add(class_name)
        
        classes = sorted(list(classes))
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        return data_all, frames_all, classes, class_to_idx
    
    def filename_to_class_name(self, filename):
        # filename pattern: v_ApplyEyeMakeup_g07_c02
        pattern = r'v_(.+)_g\d+_c\d+'
        match = re.match(pattern, filename)
        if match:
            return match.group(1)
        else:
            raise ValueError(f"Filename {filename} does not match the expected pattern.")

    def __len__(self):
        return self.video_frame_num


if __name__ == '__main__':

    import argparse
    import video_transforms
    import torch.utils.data as Data
    import torchvision.transforms as transforms
    
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--frame_interval", type=int, default=3)
    parser.add_argument("--use-image-num", type=int, default=5)
    parser.add_argument("--data-path", type=str, default="/path/to/datasets/UCF101/videos/")
    parser.add_argument("--frame-data-path", type=str, default="/path/to/datasets/preprocessed_ffs/train/images/")
    parser.add_argument("--frame-data-txt", type=str, default="/path/to/datasets/UCF101/train_256_list.txt")
    config = parser.parse_args()


    temporal_sample = video_transforms.TemporalRandomCrop(config.num_frames * config.frame_interval)

    transform_ucf101 = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            video_transforms.RandomHorizontalFlipVideo(),
            video_transforms.UCFCenterCropVideo(256),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])


    ffs_dataset = UCF101Images(config, transform=transform_ucf101, temporal_sample=temporal_sample)
    ffs_dataloader = Data.DataLoader(dataset=ffs_dataset, batch_size=6, shuffle=False, num_workers=1)

    # for i, video_data in enumerate(ffs_dataloader):
    for video_data in ffs_dataloader:
        # print(type(video_data))
        video = video_data['video']
        # video_name = video_data['video_name']
        print(video.shape)
        print(video_data['image_name'])
        image_name = video_data['image_name']
        image_names = []
        for caption in image_name:
            single_caption = [int(item) for item in caption.split('=====')]
            image_names.append(torch.as_tensor(single_caption))
        print(image_names)
        # print(video_name)
        # print(video_data[2])

        # for i in range(16):
        #     img0 = rearrange(video_data[0][0][i], 'c h w -> h w c')
        #     print('Label: {}'.format(video_data[1]))
        #     print(img0.shape)
        #     img0 = Image.fromarray(np.uint8(img0 * 255))
        #     img0.save('./img{}.jpg'.format(i))
