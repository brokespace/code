import requests
import json


def get_model_config(model_id, access_token=None):
    """
    Downloads the model's config.json from Hugging Face and optionally attempts to load
    model.safetensors.index.json to compute the number of parameters.

    If the index file is missing and "num_params" is not already in the config,
    num_params will be set to None.

    Returns a dictionary with the model configuration.
    """
    base_url = f"https://huggingface.co/{model_id}/raw/main/"
    config_url = base_url + "config.json"
    headers = {}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"

    # Fetch config.json
    r_config = requests.get(config_url, headers=headers)
    r_config.raise_for_status()
    config = r_config.json()

    # Attempt to fetch the index file to compute num_params.
    try:
        index_url = base_url + "model.safetensors.index.json"
        r_index = requests.get(index_url, headers=headers)
        r_index.raise_for_status()
        index = r_index.json()
        # Ruby logic: total_size / 2 / 1e9 gives billions of parameters.
        total_size = index["metadata"]["total_size"]
        config["num_params"] = total_size / 2 / 1e9
    except Exception as e:
        # Index file might not exist.
        if "num_params" not in config:
            config["num_params"] = None
    return config


def calculate_vram_raw(
    num_params,
    bpw,
    lm_head_bpw,
    kv_cache_bpw,
    context,
    num_hidden_layers,
    hidden_size,
    num_key_value_heads,
    num_attention_heads,
    intermediate_size,
    vocab_size,
    num_gpus=1,
    gqa=True,
):
    """
    Implements the VRAM estimation logic.

    Parameters:
      - num_params: Number of parameters in billions.
      - bpw: Bits per weight.
      - lm_head_bpw: Bits per weight for LM head.
      - kv_cache_bpw: Bits per weight for the KV cache.
      - context: Sequence length (context length).
      - num_hidden_layers: Number of transformer layers.
      - hidden_size: Size of hidden layers.
      - num_key_value_heads: Number of key/value heads.
      - num_attention_heads: Total number of attention heads.
      - intermediate_size: Size of the intermediate layer in the MLP.
      - vocab_size: Vocabulary size.
      - num_gpus: Number of GPUs (default: 1).
      - gqa: Whether to adjust KV cache size for Grouped Query Attention (default: True).

    Returns:
      Estimated VRAM usage in gigabytes (GB), rounded to two decimals.
    """
    # CUDA kernel overhead per GPU (~500MB per GPU)
    cuda_size = 500 * (2**20) * num_gpus  # in bytes

    # VRAM for model parameters (convert billions to absolute number and then bits to bytes)
    params_size = num_params * 1e9 * (bpw / 8)

    # VRAM for KV cache (depends on context and model layers)
    kv_cache_size = (context * 2 * num_hidden_layers * hidden_size) * (kv_cache_bpw / 8)
    if gqa:
        kv_cache_size *= num_key_value_heads / num_attention_heads

    # Activation bytes per parameter
    bytes_per_param = bpw / 8
    lm_head_bytes_per_param = lm_head_bpw / 8

    head_dim = hidden_size / num_attention_heads
    attention_input = bytes_per_param * context * hidden_size

    q = bytes_per_param * context * head_dim * num_attention_heads
    k = bytes_per_param * context * head_dim * num_key_value_heads
    v = bytes_per_param * context * head_dim * num_key_value_heads

    softmax_output = lm_head_bytes_per_param * num_attention_heads * context
    softmax_dropout_mask = num_attention_heads * context
    dropout_output = lm_head_bytes_per_param * num_attention_heads * context

    out_proj_input = lm_head_bytes_per_param * context * num_attention_heads * head_dim
    attention_dropout = context * hidden_size

    attention_block = (
        attention_input
        + q
        + k
        + softmax_output
        + v
        + out_proj_input
        + softmax_dropout_mask
        + dropout_output
        + attention_dropout
    )

    mlp_input = bytes_per_param * context * hidden_size
    activation_input = bytes_per_param * context * intermediate_size
    down_proj_input = bytes_per_param * context * intermediate_size
    dropout_mask = context * hidden_size
    mlp_block = mlp_input + activation_input + down_proj_input + dropout_mask

    layer_norms = bytes_per_param * context * hidden_size * 2
    activations_size = attention_block + mlp_block + layer_norms

    # VRAM for the output layer
    output_size = lm_head_bytes_per_param * context * vocab_size

    total_bytes = (
        cuda_size + params_size + activations_size + output_size + kv_cache_size
    )
    vram_gb = total_bytes / (2**30)
    return round(vram_gb, 2)


def estimate_vram_usage(
    model_id, bpw=None, context=None, fp8=True, access_token=None, num_params=None
):
    """
    Estimates the VRAM usage (in GB) required to run a model from Hugging Face.

    If the model's index file is unavailable and config.json does not provide a "num_params"
    value, you can supply num_params manually (in billions).

    The function also checks for a quantization_config in config.json. If present and bpw is not
    explicitly provided, the "bits" value from the quantization configuration is used.

    Parameters:
      - model_id (str): The Hugging Face model ID (e.g., "meta-llama/Meta-Llama-3-8B").
      - bpw (float, optional): Bits per weight. If not provided, and if the model config includes a
                               quantization_config, that value is used; otherwise defaults to 5.0.
      - context (int, optional): Sequence length to use. Defaults to the model's max_position_embeddings.
      - fp8 (bool): Whether to use FP8 KV cache (default: True).
      - access_token (str, optional): Hugging Face access token if needed.
      - num_params (float, optional): If known, provide the number of parameters (in billions) in case
                                      index.json is missing.

    Returns:
      float: Estimated VRAM usage in GB.
    """
    config = get_model_config(model_id, access_token=access_token)

    # Use quantization config's "bits" if bpw is not provided.
    if bpw is None:
        bpw = config.get("quantization_config", {}).get("bits", 8.0)
    else:
        bpw = float(bpw)

    # Use the model's maximum context if not specified.
    if context is None:
        context = config.get("max_position_embeddings", 2048)

    # Determine number of parameters.
    if config.get("num_params") is None:
        if num_params is not None:
            config["num_params"] = num_params
        else:
            # Try to extract parameter count from model name (e.g. llama-3-8B -> 8)
            import re

            param_match = re.search(r"(\d+)B", model_id, re.IGNORECASE)
            if param_match:
                config["num_params"] = float(param_match.group(1))
            else:
                raise ValueError(
                    "Number of parameters could not be determined from the index file, config, or model name. "
                    "Please supply the 'num_params' argument (in billions)."
                )
    # Determine lm_head_bpw: use 8.0 if bpw > 6.0, else 6.0.
    lm_head_bpw = 8.0 if bpw > 6.0 else 6.0

    # KV cache bits per weight: 8 if using FP8, else 16.
    kv_cache_bpw = 8 if fp8 else 16

    return calculate_vram_raw(
        num_params=config["num_params"],
        bpw=bpw,
        lm_head_bpw=lm_head_bpw,
        kv_cache_bpw=kv_cache_bpw,
        context=context,
        num_hidden_layers=config["num_hidden_layers"],
        hidden_size=config["hidden_size"],
        num_key_value_heads=config["num_key_value_heads"],
        num_attention_heads=config["num_attention_heads"],
        intermediate_size=config["intermediate_size"],
        vocab_size=config["vocab_size"],
        num_gpus=1,
        gqa=True,
    )


def calculate_gpu_config(required_vram):
    """
    Calculate the optimal GPU configuration to load an ML model.

    Parameters:
        required_vram (float): Total VRAM required for the model (in GB).

    Returns:
        tuple: A tuple (num_gpus, vram_per_gpu) where:
            - num_gpus is the number of GPUs needed.
            - vram_per_gpu is the GPU memory (in GB) that each GPU should have.

    The function uses the available GPU memory options: 24, 48, 80, and 96 GB.
    It finds the smallest number of GPUs such that when the required VRAM is split evenly,
    there is a common GPU option that is large enough for the per-GPU share.
    """
    available_vrams = [24, 48, 80, 96]
    num_gpus = 1

    while True:
        required_per_gpu = required_vram / num_gpus
        # Find the smallest available GPU VRAM that is at least the per-GPU requirement.
        for gpu_vram in available_vrams:
            if gpu_vram >= required_per_gpu:
                return num_gpus, gpu_vram
        num_gpus += 1


def calculate_model_gpu_vram(model_id) -> tuple[int, int]:
    vram_estimate = estimate_vram_usage(model_id, fp8=True)
    num_gpus, vram_per_gpu = calculate_gpu_config(vram_estimate)
    return num_gpus, vram_per_gpu


if __name__ == "__main__":
    model_id = "TechxGenus/Meta-Llama-3-8B-GPTQ"
    try:
        vram_estimate = estimate_vram_usage(model_id, fp8=True)
        print(f"Estimated VRAM usage for {model_id}: {vram_estimate} GB")
    except Exception as e:
        print(f"Error estimating VRAM usage: {e}")
