from transformers import AutoConfig


def get_model_max_len(model_name: str):
    """
    Retrieves model configuration from Hugging Face and infers:
      - Maximum sequence length (from max_position_embeddings)
      - Recommended VRAM (in GB) using a simple heuristic for LLaMA models.
    """
    config = AutoConfig.from_pretrained(model_name)

    # Extract the maximum sequence length.
    max_len = config.max_position_embeddings
    return max_len
