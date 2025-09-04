import torch
from einops import rearrange
    

def get_difference(frames):
    return frames[:, 1:] - frames[:, :-1]

def combine_frames_and_difference(frames, diff, combine_type='interleave'):
    assert diff.shape[1] == frames.shape[1] - 1, f"diff shape {diff.shape} must have one less time step than frames {frames.shape}"
    match combine_type:
        case 'concat':
            return torch.cat([frames, diff], dim=1)
        case 'interleave':
            xy = rearrange(torch.stack([diff, frames[:,1:]], dim=-1), "b t ... two -> b (t two) ...")
            return torch.cat([frames[:, :1], xy], dim=1)
        case _:
            raise NotImplementedError(f"Unknown combine_type: {combine_type}")

def uncombine_frames_and_difference(x, combine_type='interleave'):
    match combine_type:
        case 'concat':
            return x.chunk(2, dim=1)
        case 'interleave':
            return x[:,::2,...], x[:,1::2,...]
        case _:
            raise NotImplementedError(f"Unknown combine_type: {combine_type}")