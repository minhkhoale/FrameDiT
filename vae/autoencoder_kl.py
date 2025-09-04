from diffusers.models import AutoencoderKL as _AutoencoderKL

class AutoencoderKLWrapper:
    def __init__(self, *args, **kwargs):
        self.model = _AutoencoderKL(*args, **kwargs)
        self.scaler = 0.18215
