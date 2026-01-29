from diffusers.models import AutoencoderKLTemporalDecoder


class AutoencoderKLTemporalDecoderWrapper(AutoencoderKLTemporalDecoder):
    scaler = 0.18215