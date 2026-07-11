"""Run one Qwen3-VL Stage 1 decision from an exported Habitat RGB frame."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlnce_server.cfrp import PlanPoint, PlanState, parse_cfrp_output, validate_output
from vlnce_server.qwen3vl import Qwen3VLStage1Policy, Stage1ModelRequest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--manifest",
        help="JSON manifest written by habitat030_export_r2r_rgb.py",
    )
    source.add_argument("--rgb-npy", help="RGB .npy exported by the Habitat process")
    parser.add_argument("--instruction", help="Required when --rgb-npy is used")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def smoke_plan() -> PlanState:
    return PlanState(
        global_goal="follow the navigation instruction",
        points=(
            PlanPoint(id="p1", status="current", text="inspect the next navigable direction"),
            PlanPoint(id="p2", status="todo", text="continue toward the destination"),
        ),
    )


def main() -> int:
    args = parse_args()
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("The Qwen3-VL model environment must provide numpy and Pillow") from exc

    rgb_path, instruction = _resolve_input(args)
    rgb_array = np.load(rgb_path)
    rgb = Image.fromarray(rgb_array)
    request = Stage1ModelRequest(
        instruction=instruction,
        current_plan=smoke_plan(),
        visual_history=(rgb,),
        action_history=tuple(),
        allowed_actions=("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"),
    )
    policy = Qwen3VLStage1Policy.from_pretrained(args.model, max_new_tokens=args.max_new_tokens)
    raw_xml = policy.generate_xml(request)
    output = parse_cfrp_output(raw_xml)
    validate_output(output, request.allowed_actions, previous_plan=request.current_plan, mode="stage1")

    print(f"model={args.model}")
    print(f"rgb_npy={rgb_path}")
    print(f"rgb_shape={tuple(rgb_array.shape)}")
    print(f"raw_xml={raw_xml}")
    print(f"progress={output.progress} action={output.action}")
    print("qwen3vl_stage1_inference_smoke: OK")
    return 0


def _resolve_input(args: argparse.Namespace) -> tuple[str, str]:
    if args.manifest:
        manifest_path = Path(args.manifest)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rgb_path = manifest.get("rgb_npy")
        instruction = manifest.get("instruction")
        if not isinstance(rgb_path, str) or not isinstance(instruction, str):
            raise RuntimeError("manifest must contain string rgb_npy and instruction fields")
        return rgb_path, instruction
    if not args.instruction:
        raise RuntimeError("--instruction is required when --rgb-npy is used")
    return args.rgb_npy, args.instruction


if __name__ == "__main__":
    raise SystemExit(main())
