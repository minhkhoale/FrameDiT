from diffusers.models import AutoencoderKL as _AutoencoderKL

class AutoencoderKLWrapper(_AutoencoderKL):
    scaler = 0.18215

