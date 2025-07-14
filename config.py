import os
import torch
from dataclasses import dataclass


@dataclass
class Config:
    # Gemma3n model configuration
    MODEL_NAME: str = "google/gemma-3n-E2B-it"

    # Generation parameters
    MAX_NEW_TOKENS: int = 256
    TEMPERATURE: float = 0.3
    DO_SAMPLE: bool = True
    TOP_P: float = 0.8
    TOP_K: int = 40

    # Device configuration
    TORCH_DTYPE: str = torch.bfloat16
    DEVICE_MAP: str = "auto"

    # Image preprocessing
    IMAGE_SIZE: int = 512 

    # Hugging Face token
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")

    # Gradio configuration
    GRADIO_SHARE: bool = False
    GRADIO_PORT: int = 7860
    GRADIO_SERVER_NAME: str = "0.0.0.0"
