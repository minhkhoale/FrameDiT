from diffusers.models import AutoencoderKL as _AutoencoderKL

class AutoencoderKLWrapper(_AutoencoderKL):
    scaler = 0.18215
    is_video_vae = False

