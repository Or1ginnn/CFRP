"""Minimal, lazily-loaded Qwen3-VL policies for CFRP experiments."""

from .stage1 import (
    DEFAULT_QWEN3_VL_MODEL,
    Qwen3VLDependencyError,
    Qwen3VLStage1Policy,
    Stage1ModelRequest,
    build_stage1_messages,
)
from .worker import run_file_worker
from .vllm_client import VLLMRequestError, VLLMStage1Client, make_openai_messages
from .vision import (
    HABITAT_RGB_HEIGHT,
    HABITAT_RGB_HFOV,
    HABITAT_RGB_WIDTH,
    QWEN3_VL_IMAGE_HEIGHT,
    QWEN3_VL_IMAGE_WIDTH,
    QWEN3_VL_MAX_PIXELS,
    QWEN3_VL_MIN_PIXELS,
    prepare_qwen3vl_image,
    qwen3vl_image_size,
    qwen3vl_processor_kwargs,
)

__all__ = [
    "DEFAULT_QWEN3_VL_MODEL",
    "Qwen3VLDependencyError",
    "Qwen3VLStage1Policy",
    "Stage1ModelRequest",
    "build_stage1_messages",
    "run_file_worker",
    "VLLMRequestError",
    "VLLMStage1Client",
    "HABITAT_RGB_HEIGHT",
    "HABITAT_RGB_HFOV",
    "HABITAT_RGB_WIDTH",
    "QWEN3_VL_IMAGE_HEIGHT",
    "QWEN3_VL_IMAGE_WIDTH",
    "QWEN3_VL_MAX_PIXELS",
    "QWEN3_VL_MIN_PIXELS",
    "prepare_qwen3vl_image",
    "qwen3vl_image_size",
    "qwen3vl_processor_kwargs",
    "make_openai_messages",
]
