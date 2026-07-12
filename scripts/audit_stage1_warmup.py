"""Audit Stage 1 warm-up trajectory semantics before SFT training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlnce_server.cfrp.warmup_audit import audit_sft_alignment, audit_stage1_warmup
from vlnce_server.qwen3vl.sft_manifest import load_stage1_sft_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup-jsonl", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sft-jsonl")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--check-frames", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = _load_jsonl(Path(args.warmup_jsonl))
    manifest = _load_json(Path(args.manifest))
    summary = audit_stage1_warmup(records, manifest, check_frames=args.check_frames)
    if args.sft_jsonl:
        sft_examples = load_stage1_sft_jsonl(args.sft_jsonl)
        audit_sft_alignment(records, sft_examples)
        summary["sft_alignment"] = "passed"
    Path(args.output_json).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    print("audit_stage1_warmup: OK")
    return 0


def _load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
