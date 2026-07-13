"""Shared visual contract for CFRP's Qwen3-VL Stage 1 pipeline."""

from __future__ import annotations

from typing import Any, Dict, Tuple


# Match the established ActiveVLN / InternNav Habitat camera rather than the
# old lightweight 128px smoke setting.
HABITAT_RGB_WIDTH = 640
HABITAT_RGB_HEIGHT = 480
HABITAT_RGB_HFOV = 90

# Qwen3-VL resizes to multiples of its 32px visual merge unit.  448x336 keeps
# the original 4:3 camera geometry while spending almost the same pixel budget
# as InternNav's 384x384 square input.
QWEN3_VL_IMAGE_WIDTH = 448
QWEN3_VL_IMAGE_HEIGHT = 336
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
