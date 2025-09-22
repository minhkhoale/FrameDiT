import torch
from einops import rearrange
    

def get_difference(frames, padding=False):
    if padding:
        # pad the difference to have the same number of frames as the original
        return torch.diff(frames, dim=1, prepend=frames[:, :1])
    return frames[:, 1:] - frames[:, :-1]

def get_mean(frames):
    return (frames[:, 1:] + frames[:, :-1]) / 2

def get_sum(frames):
    return frames[:, 1:] + frames[:, :-1]

def combine_frames_and_difference(frames, diff, combine_type='interleave'):
    assert diff.shape[1] == frames.shape[1] - 1, f"diff shape {diff.shape} must have one less time step than frames {frames.shape}"
    match combine_type:
        case 'concat':
            return torch.cat([frames, diff], dim=1)
        case 'interleave':
            xy = rearrange(torch.stack([diff, frames[:,1:]], dim=-1), "b t ... two -> b (t two) ...")
            return torch.cat([frames[:, :1], xy], dim=1)
        case 'token_interleave':
            xy = torch.stack([frames, diff], dim=-3) # b, f, 2, n, d
            return rearrange(xy, "... two n d -> ... (n two) d") # b, f, n*2, d
        case _:
            raise NotImplementedError(f"Unknown combine_type: {combine_type}")

def interleave_frames(x, dx, frame_dim=1):
    # x, dx: B, F, C, H, W
    xy = torch.stack([x, dx], dim=frame_dim+1) # ... F, 2, ...
    return xy.flatten(frame_dim, frame_dim+1) # ... F*2, ...

def uninterleave_frames(xy, frame_dim=1):
    # xy: B, F*2, C, H, W
    f = xy.shape[frame_dim] // 2
    x = xy.index_select(frame_dim, torch.arange(0, 2*f, 2, device=xy.device))
    dx = xy.index_select(frame_dim, torch.arange(1, 2*f, 2, device=xy.device))
    return x, dx
        

def uncombine_frames_and_difference(x, combine_type='interleave'):
    match combine_type:
        case 'concat':
            return x.chunk(2, dim=1)
        case 'interleave':
            return x[:,::2,...], x[:,1::2,...]
        case 'token_interleave':
            return x[..., ::2, :], x[..., 1::2, :]
        case _:
            raise NotImplementedError(f"Unknown combine_type: {combine_type}")
        

