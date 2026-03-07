
def inject_lora_for_lattet2v(model, lora_r=8, lora_alpha=16, lora_dropout=0.1):
    """
    Inject LoRA layers into the LatteT2V model for efficient fine-tuning.

    Args:
        model: The LatteT2V model to modify.
        lora_r: Rank of the LoRA layers.
        lora_alpha: Scaling factor for the LoRA layers.
        lora_dropout: Dropout rate for the LoRA layers.