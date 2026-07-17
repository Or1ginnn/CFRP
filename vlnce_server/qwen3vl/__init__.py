"""Minimal, lazily-loaded Qwen3-VL policies for CFRP experiments."""

from .stage1 import (
    DEFAULT_QWEN3_VL_MODEL,
    DEFAULT_STAGE1_STREAMING_TURNS,
    Qwen3VLDependencyError,
    Qwen3VLStage1Policy,
    Stage1ModelRequest,
    build_stage1_messages,
    build_stage1_turn_content,
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
from .action_sft import (
    ACTION_SFT_MAX_FRAMES,
    ACTION_SFT_SCHEMA,
    JANUS_ACTION_COLLECTION_SCHEMA,
    load_action_sft_jsonl,
    make_action_sft_example,
    validate_janus_action_sft_manifest,
    validate_action_sft_example,
)
from .action_policy import ActionModelRequest, build_action_messages, parse_action_xml
from .action_vllm_client import VLLMActionClient

__all__ = [
    "DEFAULT_QWEN3_VL_MODEL",
    "DEFAULT_STAGE1_STREAMING_TURNS",
    "Qwen3VLDependencyError",
    "Qwen3VLStage1Policy",
    "Stage1ModelRequest",
    "build_stage1_messages",
    "build_stage1_turn_content",
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
    "ACTION_SFT_MAX_FRAMES",
    "ACTION_SFT_SCHEMA",
    "JANUS_ACTION_COLLECTION_SCHEMA",
    "load_action_sft_jsonl",
    "make_action_sft_example",
    "validate_janus_action_sft_manifest",
    "validate_action_sft_example",
    "ActionModelRequest",
    "build_action_messages",
    "parse_action_xml",
    "VLLMActionClient",
]
