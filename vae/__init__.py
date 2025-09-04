from .autoencoder_kl import AutoencoderKLWrapper
from .dc_ae import MyAutoencoderDC
import torch
from einops import rearrange


def get_vae(args):
    match args.name:
        case 'autoencoder_kl':
            return AutoencoderKLWrapper.from_pretrained(args.pretrained_model_path)
        case 'dc_ae':
            return MyAutoencoderDC.from_pretrained(cfg=args)
        case _:
            raise NotImplementedError(f"VAE {args.name} not implemented")
        

def encode_video(model: torch.nn.Module, x: torch.Tensor):
    """
    x: (B,F,C,H,W).
    Input range: [-1,1]
    """
    B,F,C,H,W = x.shape
    x = rearrange(x, 'b f c h w -> (b f) c h w')
    with torch.no_grad():
        match model:
            case AutoencoderKLWrapper():
                latents = model.encode(x).latent_dist.sample()
            case MyAutoencoderDC():
                latents = model.encode(x)
            case _:
                raise NotImplementedError(f"VAE {type(model)} not implemented")
        return rearrange(latents, '(b f) c h w -> b f c h w', b=B, f=F)
    

def decode_video(model: torch.nn.Module, latents: torch.Tensor):
    """
    x: (B,F,C,H,W).
    """
    print('latents', latents.shape)
    B,F,C,H,W = latents.shape
    latents = rearrange(latents, 'b f c h w -> (b f) c h w')
    with torch.no_grad():
        match model:
            case AutoencoderKLWrapper():
                x = model.decode(latents).sample
            case MyAutoencoderDC():
                x = model.decode(latents)
            case _:
                raise NotImplementedError(f"VAE {type(model)} not implemented")
        return rearrange(x, '(b f) c h w -> b f c h w', b=B, f=F)
