"""Verify CFRP's shared Qwen3-VL image contract without loading model weights."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlnce_server.cfrp import PlanPoint, PlanState
from vlnce_server.qwen3vl import (
    QWEN3_VL_IMAGE_HEIGHT,
    QWEN3_VL_IMAGE_WIDTH,
    QWEN3_VL_MAX_PIXELS,
    Stage1ModelRequest,
    build_stage1_messages,
    qwen3vl_processor_kwargs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-npy", required=True)
    parser.add_argument("--model", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import numpy as np
        from PIL import Image
        from transformers import AutoProcessor
    except ImportError as exc:
        raise RuntimeError("this smoke requires numpy, Pillow, and transformers") from exc

    rgb = np.load(args.rgb_npy)
    request = Stage1ModelRequest(
        instruction="Walk toward the destination.",
        current_plan=PlanState(
            global_goal="reach the destination",
            points=(PlanPoint(id="p1", status="current", text="move forward"),),
        ),
        visual_history=(Image.fromarray(rgb),),
        action_history=tuple(),
        allowed_actions=("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"),
    )
    processor = AutoProcessor.from_pretrained(args.model, **qwen3vl_processor_kwargs())
    inputs = processor.apply_chat_template(
        build_stage1_messages(request),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    pixel_values = inputs["pixel_values"]
    grid = inputs["image_grid_thw"]
    print("raw_rgb_shape={}".format(tuple(rgb.shape)))
    print("prepared_image_size=({}, {})".format(QWEN3_VL_IMAGE_WIDTH, QWEN3_VL_IMAGE_HEIGHT))
    print("processor_kwargs={}".format(qwen3vl_processor_kwargs()))
    print("pixel_values_shape={}".format(tuple(pixel_values.shape)))
    print("image_grid_thw={}".format(grid.tolist()))
    print("image_grid_cells={}".format(int(grid.prod().item())))
    if tuple(rgb.shape[:2]) != (480, 640):
        raise RuntimeError("expected a 480x640 Habitat RGB frame")
    print("qwen3vl_visual_contract_smoke: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
