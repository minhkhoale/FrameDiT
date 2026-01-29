from .autoencoder_kl import AutoencoderKLWrapper
<<<<<<< HEAD
from .dc_ae import MyAutoencoderDC
from .videovae import VideoVAE
import torch
from einops import rearrange
=======
#from .dc_ae import MyAutoencoderDC
from .videovae import VideoVAE
from .titok_kl import TiTok_KL
import torch
from einops import rearrange
import safetensors
>>>>>>> 55f319d (code1)


def get_vae(args):
    match args.name:
        case 'autoencoder_kl':
            vae = AutoencoderKLWrapper.from_pretrained(args.pretrained_model_path)
<<<<<<< HEAD
            vae.is_video_vae = False
=======
        case 'lattet2v_autoencoder_kl':
            vae = AutoencoderKLWrapper.from_pretrained("maxin-cn/Latte-1", subfolder="vae")
>>>>>>> 55f319d (code1)
        case 'dc_ae':
            vae = MyAutoencoderDC.from_pretrained(cfg=args)
            vae.is_video_vae = False
        case 'video_vae':
            vae = VideoVAE.from_pretrained(args.pretrained_model_path)
            vae.register_buffer('data_mean', torch.tensor(args.data_mean))
            vae.register_buffer('data_std', torch.tensor(args.data_std))
            vae.is_video_vae = True
<<<<<<< HEAD
=======
        case 'titok_kl':
            vae = TiTok_KL(**args)
            state_dict = safetensors.torch.load_file(args.pretrained_model_path, device='cpu')
            vae.load_state_dict(state_dict, strict=True)
            vae.scaler = args.scaler
            vae.is_video_vae = False
            for n, p in vae.named_parameters():
                p.requires_grad = False
>>>>>>> 55f319d (code1)
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
<<<<<<< HEAD
=======
        case TiTok_KL():
            return latents.mul_(model.scaler)
>>>>>>> 55f319d (code1)
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
<<<<<<< HEAD
=======
        case TiTok_KL():
            pass
>>>>>>> 55f319d (code1)
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
<<<<<<< HEAD
=======
            case TiTok_KL():
                latents = model.encode(x, sample_posterior=True) #TODO: should we sample mean or posterior.
>>>>>>> 55f319d (code1)
            case _:
                raise NotImplementedError(f"VAE {type(model)} not implemented")
            
    match model:
        case VideoVAE():
            latents = rearrange(latents, 'b c f h w -> b f c h w').contiguous()
<<<<<<< HEAD
        case _:
            latents = rearrange(latents, '(b f) c h w -> b f c h w', b=B, f=F).contiguous()
    
=======
        case TiTok_KL():
            pass
        case _:
            latents = rearrange(latents, '(b f) c h w -> b f c h w', b=B, f=F).contiguous()

>>>>>>> 55f319d (code1)
    return latents
    

def decode_video(model: torch.nn.Module, latents: torch.Tensor):
    """
    x: (B,F,C,H,W).
    """
    B,F,C,H,W = latents.shape
<<<<<<< HEAD
    latents = rearrange(latents, 'b f c h w -> (b f) c h w').contiguous()
=======
    match model:
        case TiTok_KL():
            latents = rearrange(latents, 'b t (h w) c -> b t c h w', h=1)
        case _:
            latents = rearrange(latents, 'b f c h w -> (b f) c h w').contiguous()

>>>>>>> 55f319d (code1)
    with torch.no_grad():
        match model:
            case AutoencoderKLWrapper():
                x = model.decode(latents).sample
            case MyAutoencoderDC():
                x = model.decode(latents)
<<<<<<< HEAD
            case _:
                raise NotImplementedError(f"VAE {type(model)} not implemented")
        return rearrange(x, '(b f) c h w -> b f c h w', b=B, f=F)
=======
            case TiTok_KL():
                x = x / model.scaler
                x = model.decode(latents)
            case _:
                raise NotImplementedError(f"VAE {type(model)} not implemented")
    
    if model != TiTok_KL():
        x = rearrange(x, '(b f) c h w -> b f c h w', b=B, f=F)
    
    return x
>>>>>>> 55f319d (code1)
