from transformers import T5EncoderModel, T5Tokenizer
from .tokenizer_utils import _text_preprocessing


def get_tokenizer(args):
    match args.name:
        case 't5':
            tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model_path, subfolder='tokenizer')
            text_encoder = T5EncoderModel.from_pretrained(args.pretrained_model_path, subfolder='text_encoder')
        case _:
            raise NotImplementedError(f"Tokenizer {args.name} not implemented")
    return tokenizer, text_encoder

