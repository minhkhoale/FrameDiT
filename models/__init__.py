import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])

from .latte import Latte_models
from .latte_img import LatteIMG_models
# from .latte_t2v import LatteT2V
from .mat_latte import MatLatte_models
from .mat_lattev2 import MatLatteV2_models
from .mat_latte_img import MatLatteIMG_models
from .dit3d import DiT3D_models
from .diff_latte import DiffLatte_models
from .diff_lattev2 import DiffLatteV2_models
from .latte_v2 import LatteV2_models
from .spatial_diff_lattev2 import SpatialDiffLatteV2_models
from .temporal_diff_lattev2 import TemporalDiffLatteV2_models
from .fused_mat_latte import FusedMatLatte_models
from .fused_mat_latte_unsqueeze import UnsqueezedFusedMatLatte_models
from .fused_mat_latte_img import FusedMatLatteIMG_models

from torch.optim.lr_scheduler import LambdaLR


def customized_lr_scheduler(optimizer, warmup_steps=5000): # 5000 from u-vit
    from torch.optim.lr_scheduler import LambdaLR
    def fn(step):
        if warmup_steps > 0:
            return min(step / warmup_steps, 1)
        else:
            return 1
    return LambdaLR(optimizer, fn)


def get_lr_scheduler(optimizer, name, **kwargs):
    if name == 'warmup':
        return customized_lr_scheduler(optimizer, **kwargs)
    elif name == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, **kwargs)
    else:
        raise NotImplementedError(name)
    
def get_models(args):
    model_class = {
        'LatteIMG': LatteIMG_models,
        'Latte': Latte_models,
        'LatteV2': LatteV2_models,
        'DiT3D': DiT3D_models,
        'DiffLatte': DiffLatte_models,
        'DiffLatteV2': DiffLatteV2_models,
        'SpatialDiffLatteV2': SpatialDiffLatteV2_models,
        'TemporalDiffLatteV2': TemporalDiffLatteV2_models,
        'MatLatte': MatLatte_models,
        'MatLatteV2': MatLatteV2_models,
        'FusedMatLatte': FusedMatLatte_models,
        'MatLatteIMG': MatLatteIMG_models,
        'UnsqueezedFusedMatLatte': UnsqueezedFusedMatLatte_models,
        'FusedMatLatteIMG': FusedMatLatteIMG_models,
    }[args.model.split('-')[0]]
    return model_class[args.model](
        input_size=args.latent_size,
        num_classes=args.num_classes,
        num_frames=args.num_frames,
        learn_sigma=args.learn_sigma,
        in_channels=args.in_channels,
        extras=args.extras,
        gradient_checkpointing=args.gradient_checkpointing,
        attention_mode=args.get('attention_mode', 'math'),
    )
