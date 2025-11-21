from .autoencoder_kl import AutoencoderKLWrapper
from .dc_ae import MyAutoencoderDC
from .videovae import VideoVAE
import torch
from einops import rearrange


def get_vae(args):
    match args.name:
        case 'autoencoder_kl':
            vae = AutoencoderKLWrapper.from_pretrained(args.pretrained_model_path)
            vae.is_video_vae = False
        case 'dc_ae':
            vae = MyAutoencoderDC.from_pretrained(cfg=args)
            vae.is_video_vae = False
        case 'video_vae':
            vae = VideoVAE.from_pretrained(args.pretrained_model_path)
            vae.register_buffer('data_mean', torch.tensor(args.data_mean))
            vae.register_buffer('data_std', torch.tensor(args.data_std))
            vae.is_video_vae = True
        case _:
            raise NotImplementedError(f"VAE {args.name} not implemented")
    return vae
        

def scale_latents(model: torch.nn.Module, latents: torch.Tensor):
    match model:
        case AutoencoderKLWrapper():
            return latents.mul_(model.scaler)
        case MyAutoencoderDC():
            return latents
        case VideoVAE():
            data_mean = model.data_mean
            data_std = model.data_std
            shape = [1] * (latents.ndim - data_mean.ndim) + list(data_mean.shape)
            mean = data_mean.reshape(shape)
            std = data_std.reshape(shape)
            return (latents - mean) / std
        case _:
            raise NotImplementedError(f"VAE {type(model)} not implemented")
        

def encode_video(model: torch.nn.Module, x: torch.Tensor):
    """
    x: (B,F,C,H,W).
    Input range: [-1,1]
    """
    B,F,C,H,W = x.shape
    match model:
        case AutoencoderKLWrapper() | MyAutoencoderDC():
            x = rearrange(x, 'b f c h w -> (b f) c h w').contiguous()
        case VideoVAE():
            x = rearrange(x, 'b f c h w -> b c f h w').contiguous()
        case _:
            raise NotImplementedError(f"VAE {type(model)} not implemented")

    with torch.no_grad():
        match model:
            case AutoencoderKLWrapper():
                latents = model.encode(x).latent_dist.sample()
            case MyAutoencoderDC():
                latents = model.encode(x)
            case VideoVAE():
                latents = model.encode(x).sample()
            case _:
                raise NotImplementedError(f"VAE {type(model)} not implemented")
            
    match model:
        case VideoVAE():
            latents = rearrange(latents, 'b c f h w -> b f c h w').contiguous()
        case _:
            latents = rearrange(latents, '(b f) c h w -> b f c h w', b=B, f=F).contiguous()
    
    return latents
    

def decode_video(model: torch.nn.Module, latents: torch.Tensor):
    """
    x: (B,F,C,H,W).
    """
    B,F,C,H,W = latents.shape
    latents = rearrange(latents, 'b f c h w -> (b f) c h w').contiguous()
    with torch.no_grad():
        match model:
            case AutoencoderKLWrapper():
                x = model.decode(latents).sample
            case MyAutoencoderDC():
                x = model.decode(latents)
            case _:
                raise NotImplementedError(f"VAE {type(model)} not implemented")
        return rearrange(x, '(b f) c h w -> b f c h w', b=B, f=F)
