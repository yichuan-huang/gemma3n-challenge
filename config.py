import os
import torch
from dataclasses import dataclass


@dataclass
class Config:
    # Gemma3n model configuration
    MODEL_NAME: str = "google/gemma-3n-E2B-it"

    # Generation parameters
    MAX_NEW_TOKENS: int = 512

    # Device configuration
    TORCH_DTYPE: str = torch.bfloat16
    if torch.cuda.is_available():
        DEVICE_MAP: str = "cuda:0"  # Use first GPU if available
    else:
        DEVICE_MAP: str = "cpu"

    # Image preprocessing
    IMAGE_SIZE: int = 512

    # Hugging Face token
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
