import os
import torch
import random
import torch.utils.data as data

import numpy as np
import io
import json
from PIL import Image
from decord import VideoReader
from decord import cpu, gpu
from tqdm import tqdm


class DMLab(data.Dataset):
    def __init__(self, configs, transform, temporal_sample=None, train=True):

        self.configs = configs
        self.data_path = configs.data_path
        self.transform = transform

        self.temporal_sample = temporal_sample
        self.target_video_len = self.configs.num_frames
        self.frame_interval = self.configs.frame_interval
        # Preprocessed latents

        self.data_all = self.load_video_frames(self.data_path)
        self.video_num = len(self.data_all)

    def __getitem__(self, index):
        video_path, total_frames = self.data_all[index]
        # Sampling video frames
        if self.temporal_sample is not None:
            start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
            assert end_frame_ind - start_frame_ind >= self.target_video_len
            frame_indice = np.linspace(start_frame_ind, end_frame_ind-1, self.target_video_len, dtype=int).tolist()
        # Load whole video frames
        else:
            frame_indice = np.linspace(0, total_frames-1, total_frames, dtype=int).tolist()

        data = np.load(video_path)
        video_clip = torch.from_numpy(data['video'][frame_indice]).permute(0, 3, 1, 2)
        video_clip = self.transform(video_clip)

        actions = torch.from_numpy(data['actions'][frame_indice])

        return {'video': video_clip, 'action': actions, 'video_path': video_path}

    def __len__(self):
        return self.video_num
    
    def load_video_frames(self, dataroot):
        data_all = []
        # Find all npz files in dataroot (recursively)
        video_files = []
        for root, _, files in os.walk(dataroot):
            for file in files:
                if file.lower().endswith('.npz'):
                    video_files.append(os.path.join(root, file))

        for video_path in video_files:
            data = np.load(video_path)
            n_frames = data['frames'].shape[0]
            if n_frames > 0:
                data_all.append((video_path, n_frames))
            del data
        return data_all    


class DMLabLatent(data.Dataset):
    def __init__(self, configs, transform, temporal_sample=None, train=True):

        self.configs = configs
        self.latent_path = configs.latent_path
        self.action_path = configs.action_path
        self.transform = transform

        self.temporal_sample = temporal_sample
        self.target_video_len = self.configs.num_frames
        self.frame_interval = self.configs.frame_interval
        # Preprocessed latents

        self.data_all = self.load_video_frames(self.latent_path)
        self.video_num = len(self.data_all)

    def __getitem__(self, index):
        latent_path, total_frames = self.data_all[index]
        action_path = self.get_actionpath(latent_path)
        # Sampling video frames
        if self.temporal_sample is not None:
            start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
            assert end_frame_ind - start_frame_ind >= self.target_video_len
            frame_indice = np.linspace(start_frame_ind, end_frame_ind-1, self.target_video_len, dtype=int).tolist()
        # Load whole video frames
        else:
            frame_indice = np.linspace(0, total_frames-1, total_frames, dtype=int).tolist()

        video_clip = torch.load(latent_path, weights_only=True)[frame_indice]
        video_clip = self.transform(video_clip)
        actions = torch.from_numpy(np.load(action_path)['actions'][frame_indice])

        return {'video': video_clip, 'action': actions, 'video_path': latent_path}

    def __len__(self):
        return self.video_num
    
    def load_video_frames(self, dataroot):
        data_all = []
        # Find all npz files in dataroot (recursively)
        video_files = []
        for root, _, files in os.walk(dataroot):
            for file in files:
                if file.lower().endswith('.pth') or file.lower().endswith('.pt'):
                    video_files.append(os.path.join(root, file))

        for video_path in tqdm(video_files, desc="Loading latent files"):
            data = torch.load(video_path, mmap=True, weights_only=True)
            n_frames = data.shape[0]
            if n_frames > 0:
                data_all.append((video_path, n_frames))
            del data
        return data_all

    def get_actionpath(self, latent_path):
        """
        Docstring for get_actionpath
        
        :param self: Description
        :param latent_path: format <root>/folder/video_name.pt
        return <action_root>/folder/video_name.npz
        """
        path_parts = latent_path.split(os.sep)
        video_name = os.path.splitext(path_parts[-1])[0]
        action_path = os.path.join(self.action_path, path_parts[-2], video_name + '.npz')
        return action_path


if __name__ == '__main__':

    import argparse
    import torchvision
    import video_transforms
    import torch.utils.data as data

    from torchvision import transforms
    from torchvision.utils import save_image
    
    