import os, io
import re
import json
import torch
import decord
import torchvision
import numpy as np
from datasets import video_transforms


from PIL import Image
from einops import rearrange
from typing import Dict, List, Tuple
from torchvision import transforms
import random


class_labels_map = None
cls_sample_cnt = None

class_labels_map = None
cls_sample_cnt = None


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class UCF101Images(torch.utils.data.Dataset):
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
        self.image_path = configs.image_path
        self.transform = transform
        self.temporal_sample = temporal_sample
        self.target_video_len = self.configs.num_frames
        # self.v_decoder = DecordInit()
        self.data_all, self.video_frame_all, self.classes, self.class_to_idx = self.load_video_frames(self.data_path, self.image_path)
        print('self.class_to_idx', self.class_to_idx)
        self.video_num = len(self.data_all)

        random.shuffle(self.video_frame_all)
        self.use_image_num = configs.use_image_num
        self.image_tranform = transforms.Compose([
                video_transforms.ToTensorVideo(), # TCHW
                video_transforms.UCFCenterCropVideo(configs.image_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        self.video_frame_num = len(self.video_frame_all)

    def __getitem__(self, index):
        video_index = index % self.video_num
        path, total_frames = self.data_all[video_index]
        class_name = path.split('/')[-2].lower()
        class_index = self.class_to_idx[class_name]

        vr = decord.VideoReader(path, ctx=decord.cpu(0))
        
        # Sampling video frames
        start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
        assert end_frame_ind - start_frame_ind >= self.target_video_len
        frame_indice = np.linspace(start_frame_ind, end_frame_ind-1, self.target_video_len, dtype=int)
        video = vr.get_batch(frame_indice).asnumpy()
        video = torch.from_numpy(video).permute(0, 3, 1, 2)

        # videotransformer data proprecess
        video = self.transform(video) # T C H W
        images = []
        image_names = []
        for i in range(self.use_image_num):
            while True:
                try:      
                    video_frame_path = self.video_frame_all[index+i]
                    image = Image.open(video_frame_path).convert('RGB')
                    image = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0)  # 1 H W C
                    image = self.image_tranform(image)  # 1 C H W
                    images.append(image)

                    image_class_name = video_frame_path.split('/')[-3].lower()
                    image_names.append(str(self.class_to_idx[image_class_name]))
                    break
                except Exception as e:
                    # print(f"Error loading image frame: {e}. Retrying with next frame.")
                    index = random.randint(0, self.video_frame_num - self.use_image_num)

        images =  torch.cat(images, dim=0)
        assert len(images) == self.use_image_num
        assert len(image_names) == self.use_image_num

        image_names = '====='.join(image_names)
        
        video_cat = torch.cat([video, images], dim=0)
    
        return {'video': video_cat, 
                'video_name': class_index, 
                'image_name': image_names}
    
    def load_video_frames(self, dataroot, imageroot):
        data_all = []
        frames_all = []
        
        video_files = []

        classes = set()

        for root, _, files in os.walk(dataroot):
            for file in files:
                if file.lower().endswith('.mp4'):
                    video_files.append(os.path.join(root, file))

        for video_file in video_files:
            vr = decord.VideoReader(video_file, ctx=decord.cpu(0))

            n_frames = len(vr)
            if n_frames > 0:
                data_all.append((video_file, n_frames))

            class_name = self.filename_to_class_name(os.path.basename(video_file))
            classes.add(class_name.lower())

        for root, _, files in os.walk(imageroot):
            for file in files:
                if is_image_file(file):
                    frames_all.append(os.path.join(root, file))
        
        classes = sorted(list(classes))
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        return data_all, frames_all, classes, class_to_idx

    def filename_to_class_name(self, filename):
        # filename pattern: v_ApplyEyeMakeup_g07_c02
        pattern = r'v_(.+)_g\d+_c\d+'
        match = re.match(pattern, filename)
        if match:
            return match.group(1).lower()
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
