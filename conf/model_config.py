# Copyright (c) 2025 Huawei. All rights reserved.

"""
Model Configuration Module

This module provides configuration management for various large language models (LLMs).
It defines model types, their corresponding tokenizers, and detailed configurations
including architectural parameters, file paths, and model-specific settings.

The module supports multiple model families:
- LLaMA 2 (7B)
- Qwen 1.5 (7B, 72B)
- Qwen 2.5 (7B, 72B)
- DeepSeek V3

Key features:
- Enum-based model type definitions for type safety
- Automatic tokenizer selection based on model type
- Centralized model configuration with architectural parameters
- File path generation for model checkpoints and configurations
- Factory pattern for creating model configurations

Usage:
    from model_config import ModelType, ModelConfig

    model_type = ModelType.QWEN2_5_7B
    config = ModelConfig.create(model_type)
    tokenizer = get_tokenizer_for_model(model_type)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional

from fastchat.conversation import (
    get_conv_template,
)
from transformers import AutoTokenizer


class ModelType(Enum):
    """
    Enumeration of supported model types with their HuggingFace model identifiers.

    This enum provides a centralized way to reference different LLM models
    and ensures type safety when working with model configurations.
    """
    LLAMA2_7B = "meta-llama/Llama-2-7b"
    QWEN1_5_7B = "Qwen/Qwen1.5-7B"
    QWEN1_5_72B = "Qwen/Qwen1.5-72B"
    QWEN2_5_7B = "QWEN/QWEN2.5-7B"
    QWEN2_5_72B = "Qwen/Qwen2.5-72B"
    DEEPSEEK_V3 = "deepseek-ai/DeepSeek-V3"


class TokenizerForModelType(Enum):
    """
    Enumeration mapping model types to their corresponding tokenizer identifiers.

    Each model type has a specific tokenizer that should be used for text processing.
    This enum ensures the correct tokenizer is selected for each model.
    """
    LLAMA2_7B = "meta-llama/Llama-2-7b-chat-hf"
    QWEN1_5_7B = "Qwen/Qwen1.5-7B-Chat"
    QWEN1_5_72B = "Qwen/Qwen1.5-72B-Chat"
    QWEN2_5_7B = "Qwen/Qwen2.5-7B-Instruct"
    QWEN2_5_72B = "Qwen/Qwen2.5-72B-Instruct"
    DEEPSEEK_V3 = "deepseek-ai/deepseek-vl-7b-chat"


def get_model_type_name_from_string(model_string: str) -> ModelType:
    """
    Convert a model string identifier to its corresponding ModelType enum value.

    Args:
        model_string (str): The string identifier of the model (e.g., "meta-llama/Llama-2-7b")

    Returns:
        ModelType: The corresponding ModelType enum value

    Raises:
        ValueError: If no matching ModelType is found for the given string

    Example:
        >>> model_type = get_model_type_name_from_string("meta-llama/Llama-2-7b")
        >>> print(model_type)
        ModelType.LLAMA2_7B
    """
    try:
        return next(m for m in ModelType if m.value == model_string)
    except StopIteration as e:
        raise ValueError("No ModelType found for model string: {},{} ".format(model_string, e))


def get_tokenizer_for_model(model_type: ModelType, trust_remote_code=True):
    """
    Get the appropriate tokenizer for a given model type.

    Args:
        model_type (ModelType): The model type for which to get the tokenizer
        trust_remote_code (bool): Whether to trust remote code when loading the tokenizer

    Returns:
        AutoTokenizer: The loaded tokenizer instance for the specified model

    Raises:
        ValueError: If no tokenizer is defined for the given model type

    Example:
        >>> tokenizer = get_tokenizer_for_model(ModelType.QWEN2_5_7B)
        >>> tokens = tokenizer.encode("Hello world")
    """
    try:
        tokenizer_model_name = TokenizerForModelType[model_type.name].value
        return AutoTokenizer.from_pretrained(tokenizer_model_name, trust_remote_code=trust_remote_code)
    except KeyError as e:
        raise ValueError('No tokenizer defined for model type: {}, {}'.format(model_type, e))


def get_ckpt_path(model_name: str, tp: int, dtype: str):
    """
    Generate the checkpoint file path for a given model configuration.

    Args:
        model_name (str): The name/identifier of the model
        tp (int): Tensor parallelism degree (number of devices)
        dtype (str): Data type of the model (currently unused, defaults to .bin)

    Returns:
        str: The file path pattern for the model checkpoint

    Example:
        >>> path = get_ckpt_path("meta-llama/Llama-2-7b", 4, "fp16")
        >>> print(path)
        /mnt/disk0/meta-llama/llama-2-7b/4_device/*/model.bin
    """
    prefix_path = "/mnt/disk0"
    model_type = ModelType(model_name).value.lower()
    return (f"{prefix_path}/{model_type}/"
            f"{tp}_device/*/model.bin")


def get_yaml_path(model_name: str):
    """
    Generate the YAML configuration file path for a given model.

    Args:
        model_name (str): The name/identifier of the model

    Returns:
        str: The file path to the model's YAML configuration file

    Example:
        >>> yaml_path = get_yaml_path("meta-llama/Llama-2-7b")
        >>> print(yaml_path)
        /mnt/disk0/meta-llama/llama-2-7b/config.yaml
    """
    prefix_path = "/mnt/disk0"
    model_type = ModelType(model_name).value.lower()
    return (f"{prefix_path}/{model_type}/config.yaml")

@dataclass
class _torch_dtype:
    itemsize: int

@dataclass
class ModelConfig:
    """
    Configuration dataclass containing all architectural parameters for supported models.

    This class stores comprehensive model specifications including:
    - Basic architecture parameters (hidden_size, num_layers, etc.)
    - Attention mechanism configuration (num_heads, kv_heads, head_size)
    - Vocabulary and sequence length settings
    - Model-specific parameters (especially for MoE models like DeepSeek V3)
    - Memory and quantization settings

    Attributes:
        model_type (ModelType): The type of the model
        hidden_size (int): Hidden dimension size of the model
        num_layers (int): Number of transformer layers
        max_seq_length (int): Maximum sequence length the model can handle
        num_heads (int): Number of attention heads
        kv_heads (int): Number of key-value heads (for GQA/MQA)
        vocab_size (int): Size of the vocabulary
        head_size (int): Size of each attention head
        model_size_b (float): Model size in billions of parameters
        intermediate_size (int): Size of the feed-forward network intermediate layer
        chat_template (Optional[Any]): Template for chat-based interactions

        # DeepSeek V3 specific parameters:
        kv_lora_rank (int): Rank for key-value LoRA decomposition
        max_position_embeddings (int): Maximum position embeddings
        moe_intermediate_size (int): MoE intermediate layer size
        n_routed_experts (int): Number of routed experts in MoE
        n_shared_experts (int): Number of shared experts in MoE
        num_attention_heads (int): Number of attention heads (DeepSeek naming)
        num_experts_per_tok (int): Number of experts activated per token
        num_key_value_heads (int): Number of key-value heads (DeepSeek naming)
        first_k_dense_replace (int): Number of initial layers that are dense (not MoE)
        q_lora_rank (int): Rank for query LoRA decomposition
        qk_nope_head_dim (int): Query-key head dimension without positional encoding
        qk_rope_head_dim (int): Query-key head dimension with RoPE
        v_head_dim (int): Value head dimension

        # Memory/quantization parameters:
        size_of_w (int): Size multiplier for weights
        size_of_a (int): Size multiplier for activations
        size_of_kv (int): Size multiplier for key-value cache
    """
    model_type: ModelType
    hidden_size: int = 0
    num_layers: int = 0
    max_seq_length: int = 0
    num_heads: int = 0
    kv_heads: int = 0
    vocab_size: int = 0
    head_size: int = 0
    model_size_b: float = 0.0
    intermediate_size: int = 0
    chat_template: Optional[Any] = None
    kv_lora_rank: int = 0
    max_position_embeddings: int = 0
    moe_intermediate_size: int = 0
    n_routed_experts: int = 0
    n_shared_experts: int = 0
    num_attention_heads: int = 0
    num_experts_per_tok: int = 0
    num_key_value_heads: int = 0
    first_k_dense_replace: int = 0
    q_lora_rank: int = 0
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0
    v_head_dim: int = 0
    size_of_w: int = 2
    size_of_a: int = 2
    size_of_kv: int = 2
    torch_dtype: _torch_dtype = field(default_factory=lambda: _torch_dtype(itemsize=2))

    @classmethod
    def create_model_config(cls, model_type: ModelType) -> 'ModelConfig':
        def cfg(**kwargs):
            return kwargs

        conv = get_conv_template
        configs = {
            ModelType.DEEPSEEK_V3: cfg(
                model_size_b=671, hidden_size=7168, intermediate_size=18432,
                kv_lora_rank=512, max_position_embeddings=163840, moe_intermediate_size=2048,
                first_k_dense_replace=3, n_routed_experts=256, n_shared_experts=1,
                num_attention_heads=128, num_experts_per_tok=8, num_layers=61,
                num_key_value_heads=128, q_lora_rank=1536, qk_nope_head_dim=128,
                qk_rope_head_dim=64, v_head_dim=128, vocab_size=129280, torch_dtype=_torch_dtype(itemsize=1),
                size_of_w=1, size_of_a=1, size_of_kv=1),
            ModelType.LLAMA2_7B: cfg(
                hidden_size=4096, num_layers=32, max_seq_length=4096, num_heads=32,
                kv_heads=32, vocab_size=32000, head_size=128, model_size_b=7,
                intermediate_size=11008, chat_template=conv("llama-2")),
            ModelType.QWEN1_5_72B: cfg(
                hidden_size=8192, num_layers=80, max_seq_length=8192, num_heads=64,
                kv_heads=8, vocab_size=151936, head_size=128, model_size_b=72,
                intermediate_size=24576, chat_template=conv("qwen-7b-chat")),
            ModelType.QWEN2_5_72B: cfg(
                hidden_size=8192, num_layers=80, max_seq_length=32768, num_heads=64,
                kv_heads=8, vocab_size=152064, head_size=128, model_size_b=72,
                intermediate_size=29568, chat_template=conv("qwen-7b-chat")),
            ModelType.QWEN1_5_7B: cfg(
                hidden_size=4096, num_layers=32, max_seq_length=32768, num_heads=32,
                kv_heads=8, vocab_size=151936, head_size=128, model_size_b=7,
                intermediate_size=22016, chat_template=conv("qwen-7b-chat")),
            ModelType.QWEN2_5_7B: cfg(
                hidden_size=3584, num_layers=28, max_seq_length=131072, num_heads=28,
                kv_heads=4, vocab_size=151646, head_size=128, model_size_b=7.61,
                intermediate_size=18944, chat_template=conv("qwen-7b-chat")),
        }

        try:
            return cls(model_type=model_type, **configs[model_type])
        except KeyError as e:
            raise ValueError(f'Unsupported model_type: {model_type}') from e
