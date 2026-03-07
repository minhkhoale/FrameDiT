from vbench import VBench
import torch

device = torch.device('cuda:0')
my_VBench = VBench(device, '/scratch/s224075134/temporal_diffusion/VBench/vbench/VBench_full_info.json', '/scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/vbench')
my_VBench.evaluate(
    videos_path = '/scratch/s224075134/temporal_diffusion/video-diffusion-model-v2/sample_videos/vbench/framedith_concat_lora',
    name = 'framedit-h-lora',
    dimension_list = ['subject_consistency', 'background_consistency', 'temporal_flickering', 'motion_smoothness', 'dynamic_degree', 'aesthetic_quality', 'imaging_quality', 'object_class', 'multiple_objects', 'human_action', 'color', 'spatial_relationship', 'scene', 'temporal_style', 'appearance_style', 'overall_consistency']
)