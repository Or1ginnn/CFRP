"""Validate a portable Qwen3-VL Stage 1 SFT manifest before training."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlnce_server.cfrp import parse_cfrp_output
from vlnce_server.qwen3vl.sft_manifest import load_stage1_sft_jsonl, validate_stage1_sft_example


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--check-images", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    examples = load_stage1_sft_jsonl(args.input_jsonl)
    for example in examples:
        validate_stage1_sft_example(example, check_images=args.check_images)
    actions = Counter(parse_cfrp_output(item["target_xml"]).action for item in examples)
    print(f"examples={len(examples)}")
    print(f"actions={dict(sorted(actions.items()))}")
    print(f"images_checked={args.check_images}")
    print("validate_stage1_sft_manifest: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
