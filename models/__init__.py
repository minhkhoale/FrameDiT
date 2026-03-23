import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])
import torch

from .latte import Latte_models
from .latte_img import LatteIMG_models
from .framedit_g import FrameDiTG_models
from .framedit_g_img import FrameDiTGIMG_models
from .dit3d import DiT3D_models
from .framedit_h import FrameDiTH_models
from .framedit_h_img import FrameDiTHIMG_models
from .framedit_h_t2v import FrameDiTHT2V_models
from .latte_t2v import LatteT2V

    
def get_models(args) -> torch.nn.Module:
    if args.model == 'LatteT2V':
        return LatteT2V.from_pretrained(args.pretrained_model_path, subfolder="transformer", video_length=args.video_length)

    model_name = args.model.split('-')[0]
    if model_name == 'FrameDiTHT2V':
        return FrameDiTHT2V_models[args.model](in_channels=args.in_channels, out_channels=args.in_channels*2 if args.learn_sigma else args.in_channels)

    model_class = {
        'Latte': Latte_models,
        'LatteIMG': LatteIMG_models,
        'DiT3D': DiT3D_models,
        'FrameDiTG': FrameDiTG_models,
        'FrameDiTGIMG': FrameDiTGIMG_models,
        'FrameDiTH': FrameDiTH_models,
        'FrameDiTHIMG': FrameDiTHIMG_models,
        'FusedMatLatteIMG': FrameDiTHIMG_models,  # use the same architecture as FrameDiT_H_IMG for FusedMatLatteIMG
    }[args.model.split('-')[0]]
    return model_class[args.model](
        input_size=args.latent_size,
        num_classes=args.num_classes,
        num_frames=args.num_frames,
        learn_sigma=args.learn_sigma,
        in_channels=args.in_channels,
        extras=args.extras,
        gradient_checkpointing=args.get('gradient_checkpointing', False),
        attention_mode=args.get('attention_mode', 'math'),
    )
