# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import copy
from typing import List, Dict, Optional
import zipfile
import json
import random
from typing import Tuple

import numpy as np
import PIL.Image
import torch
from tools import dnnlib
from omegaconf import DictConfig, OmegaConf

from tools.utils.layers import sample_frames

try:
    import pyspng
except ImportError:
    pyspng = None

try:
    import decord
    _HAVE_DECORD = True
except Exception:
    _HAVE_DECORD = False

try:
    import torchvision.io as tvio
    _HAVE_TVIO = True
except Exception:
    _HAVE_TVIO = False

try:
    import cv2
    _HAVE_CV2 = True
except Exception:
    _HAVE_CV2 = False

#----------------------------------------------------------------------------

NUMPY_INTEGER_TYPES = [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]
NUMPY_FLOAT_TYPES = [np.float16, np.float32, np.float64, np.single, np.double]

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]

        return {
            'image': image.copy(),
            'label': self.get_label(idx),
        }

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1, f"Labels must be 1-dimensional: {self.label_shape} to use `.label_dim`"
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError(f'Image files do not match the specified resolution. Resolution is {resolution}, shape is {raw_shape}')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]

        with self._open_file(fname) as f:
            use_pyspng = pyspng is not None and self._file_ext(fname) == '.png'
            image = load_image_from_buffer(f, use_pyspng=use_pyspng)

        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        labels_files = [f for f in self._all_fnames if f.endswith(fname)]
        if len(labels_files) == 0:
            return None
        assert len(labels_files) == 1, f"There can be only a single {fname} file"
        with self._open_file(labels_files[0]) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[remove_root(fname, self._name).replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)

        if labels.dtype in NUMPY_INTEGER_TYPES:
            labels = labels.astype(np.int64)
        elif labels.dtype in NUMPY_FLOAT_TYPES:
            labels = labels.astype(np.float32)
        else:
            raise NotImplementedError(f"Unsupported label dtype: {labels.dtype}")

        return labels

#----------------------------------------------------------------------------

class VideoFramesFolderDataset(Dataset):
    def __init__(self,
        path,                                           # Path to directory or zip.
        cfg: DictConfig,                                # Config
        resolution=None,                                # Unused arg for backward compatibility
        load_n_consecutive: int=None,                   # Should we load first N frames for each video?
        load_n_consecutive_random_offset: bool=True,    # Should we use a random offset when loading consecutive frames?
        subsample_factor: int=1,                        # Sampling factor, i.e. decreasing the temporal resolution
        discard_short_videos: bool=False,               # Should we discard videos that are shorter than `load_n_consecutive`?
        **super_kwargs,                                 # Additional arguments for the Dataset base class.
    ):
        self.sampling_dict = OmegaConf.to_container(OmegaConf.create({**cfg.sampling})) if 'sampling' in cfg else None
        self.max_num_frames = cfg.max_num_frames
        self._path = path
        self._zipfile = None
        self.load_n_consecutive = load_n_consecutive
        self.load_n_consecutive_random_offset = load_n_consecutive_random_offset
        self.subsample_factor = subsample_factor
        print(subsample_factor)
        self.discard_short_videos = discard_short_videos

        if self.subsample_factor > 1 and self.load_n_consecutive is None:
            raise NotImplementedError("Can do subsampling only when loading consecutive frames.")

        listdir_full_paths = lambda d: sorted([os.path.join(d, x) for x in os.listdir(d)])
        name = os.path.splitext(os.path.basename(self._path))[0]

        if os.path.isdir(self._path):
            self._type = 'dir'
            # We assume that the depth is 2
            self._all_objects = {o for d in listdir_full_paths(self._path) for o in (([d] + listdir_full_paths(d)) if os.path.isdir(d) else [d])}
            self._all_objects = {os.path.relpath(o, start=os.path.dirname(self._path)) for o in {self._path}.union(self._all_objects)}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_objects = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must be either a directory or point to a zip archive')

        PIL.Image.init()
        self._video_dir2frames = {}
        objects = sorted([d for d in self._all_objects])
        root_path_depth = len(os.path.normpath(objects[0]).split(os.path.sep))
        curr_d = objects[1] # Root path is the first element

        for o in objects[1:]:
            curr_obj_depth = len(os.path.normpath(o).split(os.path.sep))

            if self._file_ext(o) in PIL.Image.EXTENSION:
                assert o.startswith(curr_d), f"Object {o} is out of sync. It should lie inside {curr_d}"
                assert curr_obj_depth == root_path_depth + 2, "Frame images should be inside directories"
                if not curr_d in self._video_dir2frames:
                    self._video_dir2frames[curr_d] = []
                self._video_dir2frames[curr_d].append(o)
            elif self._file_ext(o) == 'json':
                assert curr_obj_depth == root_path_depth + 1, "Classes info file should be inside the root dir"
                pass
            else:
                # We encountered a new directory
                assert curr_obj_depth == root_path_depth + 1, f"Video directories should be inside the root dir. {o} is not."
                if curr_d in self._video_dir2frames:
                    sorted_files = sorted(self._video_dir2frames[curr_d])
                    self._video_dir2frames[curr_d] = sorted_files
                curr_d = o

        if self.discard_short_videos:
            self._video_dir2frames = {d: fs for d, fs in self._video_dir2frames.items() if len(fs) >= self.load_n_consecutive * self.subsample_factor}

        self._video_idx2frames = [frames for frames in self._video_dir2frames.values()]

        if len(self._video_idx2frames) == 0:
            raise IOError('No videos found in the specified archive')

        raw_shape = [len(self._video_idx2frames)] + list(self._load_raw_frames(0, [0])[0][0].shape)

        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(os.path.dirname(self._path), fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_labels(self):
        """
        We leave the `dataset.json` file in the same format as in the original SG2-ADA repo:
        it's `labels` field is a hashmap of filename-label pairs.
        """
        fname = 'dataset.json'
        labels_files = [f for f in self._all_objects if f.endswith(fname)]
        if len(labels_files) == 0:
            return None
        assert len(labels_files) == 1, f"There can be only a single {fname} file"
        with self._open_file(labels_files[0]) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None

        labels = dict(labels)
        # The `dataset.json` file defines a label for each image and
        # For the video dataset, this is both inconvenient and redundant.
        # So let's redefine this
        video_labels = {}
        for filename, label in labels.items():
            dirname = os.path.dirname(filename)
            if dirname in video_labels:
                assert video_labels[dirname] == label
            else:
                video_labels[dirname] = label
        labels = video_labels
        labels = [labels[os.path.normpath(dname).split(os.path.sep)[-1]] for dname in self._video_dir2frames]
        labels = np.array(labels)

        if labels.dtype in NUMPY_INTEGER_TYPES:
            labels = labels.astype(np.int64)
        elif labels.dtype in NUMPY_FLOAT_TYPES:
            labels = labels.astype(np.float32)
        else:
            raise NotImplementedError(f"Unsupported label dtype: {labels.dtype}")

        return labels

    def __getitem__(self, idx: int) -> Dict:
        if self.load_n_consecutive:
            num_frames_available = len(self._video_idx2frames[self._raw_idx[idx]])
            assert num_frames_available - self.load_n_consecutive * self.subsample_factor >= 0, f"We have only {num_frames_available} frames available, cannot load {self.load_n_consecutive} frames."

            if self.load_n_consecutive_random_offset:
                random_offset = random.randint(0, num_frames_available - self.load_n_consecutive * self.subsample_factor + self.subsample_factor - 1)
            else:
                random_offset = 0
            frames_idx = np.arange(0, self.load_n_consecutive * self.subsample_factor, self.subsample_factor) + random_offset
        else:
            frames_idx = None

        frames, times = self._load_raw_frames(self._raw_idx[idx], frames_idx=frames_idx)

        assert isinstance(frames, np.ndarray)
        assert list(frames[0].shape) == self.image_shape
        assert frames.dtype == np.uint8
        assert len(frames) == len(times)

        if self._xflip[idx]:
            assert frames.ndim == 4 # TCHW
            frames = frames[:, :, :, ::-1]

        return {
            'image': frames.copy(),
            'label': self.get_label(idx),
            'times': times,
            'video_len': self.get_video_len(idx),
        }

    def get_video_len(self, idx: int) -> int:
        return min(self.max_num_frames, len(self._video_idx2frames[self._raw_idx[idx]]))

    def _load_raw_frames(self, raw_idx: int, frames_idx: List[int]=None) -> Tuple[np.ndarray, np.ndarray]:
        frame_paths = self._video_idx2frames[raw_idx]
        total_len = len(frame_paths)
        offset = 0
        images = []

        if frames_idx is None:
            assert not self.sampling_dict is None, f"The dataset was created without `cfg.sampling` config and cannot sample frames on its own."
            if total_len > self.max_num_frames:
                offset = random.randint(0, total_len - self.max_num_frames)
            frames_idx = sample_frames(self.sampling_dict, total_video_len=min(total_len, self.max_num_frames)) + offset
        else:
            frames_idx = np.array(frames_idx)

        for frame_idx in frames_idx:
            with self._open_file(frame_paths[frame_idx]) as f:
                images.append(load_image_from_buffer(f))

        return np.array(images), frames_idx - offset

    def compute_max_num_frames(self) -> int:
        return max(len(frames) for frames in self._video_idx2frames)
    
class _VideoDecoder:
    """
    Thin abstraction over decord / torchvision / OpenCV for frame-accurate-ish reads.
    """
    def __init__(self, path: str):
        self.path = path
        self.backend = None
        self.length = None  # number of frames
        self.fps = None
        self._init_backend()

    def _init_backend(self):
        # Prefer decord
        if _HAVE_DECORD:
            try:
                vr = decord.VideoReader(self.path, ctx=decord.cpu(0))
                self.length = len(vr)
                self.fps = vr.get_avg_fps()
                self.backend = 'decord'
                return
            except Exception as e:
                pass

        # Fallback: OpenCV (reasonably fast random access via CAP_PROP_POS_FRAMES)
        if _HAVE_CV2:
            cap = cv2.VideoCapture(self.path)
            if cap.isOpened():
                self.length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.fps = cap.get(cv2.CAP_PROP_FPS) or None
                self.cap = cap
                self.backend = 'cv2'
                return

        # Last resort: torchvision (reads entire video to tensor once per call)
        if _HAVE_TVIO:
            # We won’t read here; we’ll lazily read to discover length when needed
            self.backend = 'torchvision'
            # try to get metadata (not always available); length will be probed on first read
            self.length = None
            self.fps = None
            return

        raise RuntimeError(
            "No video backend available. Please install one of: decord, opencv-python, torchvision."
        )

    def get_length(self) -> int:
        if self.length is not None:
            return self.length
        # use decord
        vr = decord.VideoReader(self.path, ctx=decord.cpu(0))
        self.length = len(vr)
        return self.length

    def get_batch(self, frame_indices: np.ndarray) -> np.ndarray:
        """
        Returns frames as uint8 HWC for the given frame indices (sorted not required).
        """
        if self.backend == 'decord':
            # Decord accepts NDArray of indices; returns batch in HWC
            vr = decord.VideoReader(self.path, ctx=decord.cpu(0))  # re-init to avoid issues with multithreading
            batch = vr.get_batch(frame_indices).asnumpy()  # T,H,W,C uint8 (RGB)
            return batch

        elif self.backend == 'cv2':
            # Read frames by seeking; cv2 returns BGR; convert to RGB
            frames = []
            # For fewer seeks, process in ascending order but restore order after
            order = np.argsort(frame_indices)
            sorted_idx = frame_indices[order]
            last_pos = -1
            for i, fidx in zip(order, sorted_idx):
                if fidx != last_pos:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
                ok, frame = self.cap.read()
                if not ok:
                    # If read fails, create a black frame on the fly (rare)
                    if len(frames) > 0:
                        h, w, c = frames[-1].shape
                    else:
                        # One more try: jump to 0
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok2, fr2 = self.cap.read()
                        if ok2: 
                            h, w, c = fr2.shape
                        else:
                            raise RuntimeError(f"Failed to read any frame from {self.path}")
                    frame = np.zeros((h, w, 3), dtype=np.uint8)
                last_pos = fidx
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append((i, frame))
            # restore original order
            frames_sorted = sorted(frames, key=lambda x: x[0])
            batch = np.stack([fr for _, fr in frames_sorted], axis=0)
            return batch

        elif self.backend == 'torchvision':
            # Load entire video, then gather indices
            vid, _, _ = tvio.read_video(self.path, pts_unit='sec')  # T,H,W,C uint8
            t = vid.shape[0]
            safe_idx = np.clip(frame_indices, 0, max(0, t-1)).astype(np.int64)
            batch = vid[safe_idx].numpy()
            return batch

        else:
            raise RuntimeError("Unknown backend.")

    def close(self):
        if self.backend == 'cv2':
            try:
                self.cap.release()
            except Exception:
                pass

class VideoDataset(torch.utils.data.Dataset):
    """
    MP4-based video dataset, API aligned with VideoFramesFolderDataset.

    Returns dict:
      - 'image': np.uint8 array [T, C, H, W]
      - 'label': per-video label (one-hot float32 or continuous)
      - 'times': np.ndarray of frame indices (int), relative to the (possibly offset) window
      - 'video_len': int, min(max_num_frames, actual length)
    """
    def __init__(
        self,
        path: str,                             # Root directory containing .mp4 files (no zip support here)
        cfg: DictConfig,                       # Config; expects cfg.max_num_frames and optionally cfg.sampling
        resolution: Optional[int] = None,      # Unused, kept for API compatibility
        load_n_consecutive: Optional[int] = None,
        load_n_consecutive_random_offset: bool = True,
        subsample_factor: int = 1,
        discard_short_videos: bool = False,
        use_labels: bool = False,
        xflip: bool = False,
        max_size: Optional[int] = None,
        random_seed: int = 0,
    ):
        assert os.path.isdir(path), "VideoMP4Dataset currently supports directory roots only."
        self._path = path
        print('self._path', self._path)
        self.load_n_consecutive = load_n_consecutive
        self.load_n_consecutive_random_offset = load_n_consecutive_random_offset
        self.subsample_factor = subsample_factor
        self.discard_short_videos = discard_short_videos
        self.max_num_frames = cfg.max_num_frames
        self.sampling_dict = OmegaConf.to_container(OmegaConf.create({**cfg.sampling})) if 'sampling' in cfg else None

        if self.subsample_factor > 1 and self.load_n_consecutive is None:
            raise NotImplementedError("Subsampling requires load_n_consecutive to be set.")

        # Discover mp4 files (one file = one video)
        all_files = sorted(
            f for f in os.listdir(self._path)
            if os.path.isfile(os.path.join(self._path, f)) and os.path.splitext(f)[1].lower() in ('.mp4', '.mov', '.mkv', '.webm', '.avi')
        )
        if len(all_files) == 0:
            raise IOError("No video files found (mp4/mov/mkv/webm).")

        # Build per-video decoders + lengths
        self._video_files = [os.path.join(self._path, f) for f in all_files]
        self._decoders: List[_VideoDecoder] = []
        self._lengths: List[int] = []

        # Initialize decoders and compute lengths (cheap with decord; might be slower with cv2/torchvision once)
        for vf in self._video_files:
            vd = _VideoDecoder(vf)
            L = vd.get_length()
            self._decoders.append(vd)
            self._lengths.append(L)

        # Optionally discard short videos
        if self.discard_short_videos and self.load_n_consecutive:
            min_len = self.load_n_consecutive * self.subsample_factor
            keep_mask = [L >= min_len for L in self._lengths]
            self._video_files = [v for v, k in zip(self._video_files, keep_mask) if k]
            self._decoders    = [d for d, k in zip(self._decoders, keep_mask) if k]
            self._lengths     = [L for L, k in zip(self._lengths, keep_mask) if k]

        if len(self._video_files) == 0:
            raise IOError("No valid videos remain after filtering.")

        # Infer a representative frame shape by reading frame 0 from the first video
        probe = self._decoders[0].get_batch(np.array([0], dtype=np.int64))[0]  # HWC uint8
        C, H, W = probe.shape[2], probe.shape[0], probe.shape[1]
        name = os.path.splitext(os.path.basename(self._path))[0]

        # We mimic the base Dataset's contract: raw_shape is [N, C, H, W] (per-frame shape)
        raw_shape = [len(self._video_files), C, H, W]

        # Build the Dataset base indexing/state by hand (to align with your Dataset semantics)
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Create raw index with optional max_size + xflip (exactly as in your Dataset)
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

        # Preload labels lazily (same behavior as in your Dataset)
        # → _load_raw_labels() implemented below.

    # ----------------------------- Labeling -----------------------------

    @staticmethod
    def _file_ext(fname):  # match your base helper
        return os.path.splitext(fname)[1].lower()

    def _open_file(self, fname, mode='rb'):
        return open(fname, mode)

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def _load_raw_labels(self):
        """
        Same format as ImageFolderDataset:
          dataset.json: { "labels": { "video_filename.mp4": label_value, ... } }
        One label per video file (by base name).
        """
        fname = os.path.join(self._path, 'dataset.json')
        if not os.path.isfile(fname):
            return None
        with self._open_file(fname, 'r') as f:
            labels = json.load(f).get('labels', None)
        if labels is None:
            return None

        labels = dict(labels)
        # Construct label vector aligned to self._video_files order
        keys = [os.path.basename(vf) for vf in self._video_files]
        try:
            values = [labels[k] for k in keys]
        except KeyError as e:
            missing = str(e).strip("'")
            raise KeyError(f"Video {missing} missing from dataset.json labels.") from None

        labels_arr = np.array(values)
        if labels_arr.dtype in NUMPY_INTEGER_TYPES:
            labels_arr = labels_arr.astype(np.int64)
        elif labels_arr.dtype in NUMPY_FLOAT_TYPES:
            labels_arr = labels_arr.astype(np.float32)
        else:
            raise NotImplementedError(f"Unsupported label dtype: {labels_arr.dtype}")

        return labels_arr

    # Public label helpers (mirroring your base Dataset)
    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

    # --------------------------- Basic metadata -------------------------

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])  # C,H,W

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1, f"Labels must be 1-D to use label_dim, got {self.label_shape}"
        return self.label_shape[0]

    # ------------------------------ Core I/O ----------------------------

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx: int) -> Dict:
        raw_idx = int(self._raw_idx[idx])
        vd = self._decoders[raw_idx]
        total_len = vd.get_length()

        # Decide which frames to load
        if self.load_n_consecutive is not None:
            need = self.load_n_consecutive * self.subsample_factor
            assert total_len - need >= 0, f"Video has {total_len} frames; need {need} for your settings."
            if self.load_n_consecutive_random_offset:
                # offset in [0, total_len - need], but allow last stride window to include last frame
                max_off = total_len - need
                random_offset = random.randint(0, max(0, max_off))
            else:
                random_offset = 0
            frames_idx = np.arange(0, need, self.subsample_factor, dtype=np.int64) + random_offset
            offset_used = random_offset
        else:
            assert self.sampling_dict is not None, \
                "Dataset created without cfg.sampling; either provide it or set load_n_consecutive."
            # If longer than max_num_frames, pick a random offset window
            if total_len > self.max_num_frames:
                offset_used = random.randint(0, total_len - self.max_num_frames)
                effective = self.max_num_frames
            else:
                offset_used = 0
                effective = total_len
            # sample within [0, effective)
            local_idx = sample_frames(self.sampling_dict, total_video_len=effective)
            frames_idx = np.asarray(local_idx, dtype=np.int64) + offset_used

        # Fetch frames HWC uint8
        frames_hwc = vd.get_batch(frames_idx)
        # Convert to TCHW uint8
        frames = np.transpose(frames_hwc, (0, 3, 1, 2))  # T,C,H,W
        assert frames.dtype == np.uint8

        # Apply xflip if needed (mirroring your video variant’s behavior)
        if int(self._xflip[idx]) != 0:
            frames = frames[:, :, :, ::-1]

        return {
            'image': frames.copy(),
            'label': self.get_label(idx),
            'times': (frames_idx - offset_used),
            'video_len': self.get_video_len(idx),
        }

    def get_video_len(self, idx: int) -> int:
        raw_idx = int(self._raw_idx[idx])
        L = self._lengths[raw_idx]
        return min(self.max_num_frames, L)

    def compute_max_num_frames(self) -> int:
        return max(self._lengths)

    def close(self):
        for d in getattr(self, "_decoders", []):
            try:
                d.close()
            except Exception:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
#----------------------------------------------------------------------------

def load_image_from_buffer(f, use_pyspng: bool=False) -> np.ndarray:
    if use_pyspng:
        image = pyspng.load(f.read())
    else:
        image = np.array(PIL.Image.open(f))
    if image.ndim == 2:
        image = image[:, :, np.newaxis] # HW => HWC
    image = image.transpose(2, 0, 1) # HWC => CHW

    return image

#----------------------------------------------------------------------------

def video_to_image_dataset_kwargs(video_dataset_kwargs: dnnlib.EasyDict) -> dnnlib.EasyDict:
    """Converts video dataset kwargs to image dataset kwargs"""
    return dnnlib.EasyDict(
        class_name='training.dataset.ImageFolderDataset',
        path=video_dataset_kwargs.path,
        use_labels=video_dataset_kwargs.use_labels,
        xflip=video_dataset_kwargs.xflip,
        resolution=video_dataset_kwargs.resolution,
        random_seed=video_dataset_kwargs.get('random_seed'),
        # Explicitly ignoring the max size, since we are now interested
        # in the number of images instead of the number of videos
        # max_size=video_dataset_kwargs.max_size,
    )

#----------------------------------------------------------------------------

def remove_root(fname: os.PathLike, root_name: os.PathLike):
    """`root_name` should NOT start with '/'"""
    if fname == root_name or fname == ('/' + root_name):
        return ''
    elif fname.startswith(root_name + '/'):
        return fname[len(root_name) + 1:]
    elif fname.startswith('/' + root_name + '/'):
        return fname[len(root_name) + 2:]
    else:
        return fname

#----------------------------------------------------------------------------
