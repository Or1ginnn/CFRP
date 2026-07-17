"""Shared visual contract for CFRP's Qwen3-VL Stage 1 pipeline."""

from __future__ import annotations

from typing import Any, Dict, Tuple


# Match JanusVLN's R2R camera contract rather than the old 128px smoke setting.
HABITAT_RGB_WIDTH = 640
HABITAT_RGB_HEIGHT = 480
HABITAT_RGB_HFOV = 79

# Qwen3-VL resizes to multiples of its 32px visual merge unit.  384x288 is an
# exact 4:3 camera frame whose two sides are both aligned, so Qwen does not
# silently distort it during its own smart-resize step.
QWEN3_VL_IMAGE_WIDTH = 384
QWEN3_VL_IMAGE_HEIGHT = 288
QWEN3_VL_MIN_PIXELS = 65_536
QWEN3_VL_MAX_PIXELS = QWEN3_VL_IMAGE_WIDTH * QWEN3_VL_IMAGE_HEIGHT


def qwen3vl_processor_kwargs() -> Dict[str, int]:
    """Return the processor settings used by Transformers and vLLM."""

    return {"min_pixels": QWEN3_VL_MIN_PIXELS, "max_pixels": QWEN3_VL_MAX_PIXELS}


def qwen3vl_image_size() -> Tuple[int, int]:
    """Return the exported SFT image size as ``(width, height)``."""

    return (QWEN3_VL_IMAGE_WIDTH, QWEN3_VL_IMAGE_HEIGHT)


def prepare_qwen3vl_image(image: Any) -> Any:
    """Convert an in-memory RGB frame to CFRP's fixed 4:3 model size.

    String image paths are deliberately left unchanged: converted SFT PNGs are
    already materialized at this size, and keeping paths portable matters for
    the JSONL manifest.
    """

    if isinstance(image, str):
        return image
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - exercised in model runtime
        raise RuntimeError("Pillow is required to prepare Qwen3-VL images") from exc
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    return image.convert("RGB").resize(qwen3vl_image_size(), resample=Image.Resampling.LANCZOS)
