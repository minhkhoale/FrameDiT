import torch
from models.latte_t2v import LatteT2V

device = 'cuda:0'

model = LatteT2V.from_pretrained('maxin-cn/Latte-1', subfolder="transformer", video_length=16).to(device, dtype=torch.float16)
print('total params:', sum(p.numel() for p in model.parameters()))
exit(0)

from transformers import T5EncoderModel, T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained("maxin-cn/Latte-1", subfolder='tokenizer')
text_encoder = T5EncoderModel.from_pretrained("maxin-cn/Latte-1", subfolder='text_encoder', torch_dtype=torch.float16).to(device)

prompt = "A dog playing with a ball"
text_inputs = tokenizer(
    prompt,
    padding="max_length",
    max_length=120,
    truncation=True,
    return_tensors="pt",
)
with torch.no_grad():
    text_encoder_outputs = text_encoder(
        input_ids=text_inputs.input_ids.to(device),
        #attention_mask=text_inputs.attention_mask,
    )
encoder_hidden_states = text_encoder_outputs[0]

inputs = torch.randn(1, 4, 16, 32, 32).to(device, dtype=torch.float16)  # (batch_size, channels, frames, height, width)
timesteps = torch.randint(0, 1000, (1,)).long().to(device)
print('inpu')
outputs = model(inputs, encoder_hidden_states=encoder_hidden_states, timestep=timesteps)
print(outputs.sample.shape)  # expected output shape: (1, 4, 16, 64, 64)