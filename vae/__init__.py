import torch
from einops import rearrange

from .autoencoder_kl import AutoencoderKLWrapper


def get_vae(args):
    match args.name:
        case 'autoencoder_kl':
            vae = AutoencoderKLWrapper.from_pretrained(args.pretrained_model_path)
        case 'lattet2v_autoencoder_kl':
            vae = AutoencoderKLWrapper.from_pretrained("maxin-cn/Latte-1", subfolder="vae")
        case _:
            raise NotImplementedError(f"VAE {args.name} not implemented")
    return vae
        

def scale_latents(model: torch.nn.Module, latents: torch.Tensor):
    match model:
        case AutoencoderKLWrapper():
            return latents.mul_(model.scaler)
        case _:
            raise NotImplementedError(f"VAE {type(model)} not implemented")
        

def encode_video(model: torch.nn.Module, x: torch.Tensor):
    """
    x: (B,F,C,H,W).
    Input range: [-1,1]
    """
    B,F,C,H,W = x.shape
    match model:
        case AutoencoderKLWrapper():
            x = rearrange(x, 'b f c h w -> (b f) c h w').contiguous()
        case _:
            raise NotImplementedError(f"VAE {type(model)} not implemented")

    with torch.no_grad():
        match model:
            case AutoencoderKLWrapper():
                latents = model.encode(x).latent_dist.sample()
            case _:
                raise NotImplementedError(f"VAE {type(model)} not implemented")
            
    match model:
        case _:
            latents = rearrange(latents, '(b f) c h w -> b f c h w', b=B, f=F).contiguous()

    return latents
    

def decode_video(model: torch.nn.Module, latents: torch.Tensor):
    """
    x: (B,F,C,H,W).
    """
    B,F,C,H,W = latents.shape
    match model:
        case _:
            latents = rearrange(latents, 'b f c h w -> (b f) c h w').contiguous()

    with torch.no_grad():
        match model:
            case AutoencoderKLWrapper():
                x = model.decode(latents).sample
            case _:
                raise NotImplementedError(f"VAE {type(model)} not implemented")
    
    x = rearrange(x, '(b f) c h w -> b f c h w', b=B, f=F)
    
    return x
