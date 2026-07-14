"""Validate a Stage 1 SFT manifest against the real Qwen3-VL processor.

This performs no model forward pass.  It verifies that every multimodal chat
template has a terminal XML suffix whose token offsets can receive the CFRP
action-weighted loss, before an expensive multi-GPU training job starts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_qwen3vl_stage1_sft import _supervised_inputs
from vlnce_server.cfrp import parse_cfrp_output
from vlnce_server.qwen3vl.loss_weights import (
    DEFAULT_ACTION_LOSS_WEIGHT,
    DEFAULT_PROGRESS_LOSS_WEIGHT,
    DEFAULT_SUBGOAL_LOSS_WEIGHT,
)
from vlnce_server.qwen3vl.sft_manifest import iter_stage1_targets, load_stage1_sft_jsonl
from vlnce_server.qwen3vl.stage1 import DEFAULT_QWEN3_VL_MODEL
from vlnce_server.qwen3vl.vision import qwen3vl_processor_kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--model", default=DEFAULT_QWEN3_VL_MODEL)
    parser.add_argument("--report", required=True)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--require-action-chunks", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_examples is not None and args.max_examples < 1:
        raise ValueError("max-examples must be positive")
    examples = load_stage1_sft_jsonl(args.train_jsonl)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
    if args.require_action_chunks and not any(
        len(parse_cfrp_output(target).actions) > 1
        for item in examples
        for target in iter_stage1_targets(item)
    ):
        raise ValueError("manifest contains no multi-action Stage 1 targets; recollect short-chunk warm-up data")
    try:
        import torch
        from transformers import AutoProcessor
    except ImportError as exc:
        raise RuntimeError("preflight requires torch and transformers in the Qwen training environment") from exc

    processor = AutoProcessor.from_pretrained(args.model, **qwen3vl_processor_kwargs())
    weight_args = SimpleNamespace(
        action_loss_weight=DEFAULT_ACTION_LOSS_WEIGHT,
        progress_loss_weight=DEFAULT_PROGRESS_LOSS_WEIGHT,
        subgoal_loss_weight=DEFAULT_SUBGOAL_LOSS_WEIGHT,
    )
    action_weighted_tokens = 0
    for index, example in enumerate(examples, start=1):
        # This first pass also checks that every declared image is readable by
        # Qwen's real multimodal chat template.
        _supervised_inputs(processor, example, torch.device("cpu"), weight_args)
        for target in iter_stage1_targets(example):
            action_weighted_tokens += sum(
                1
                for value in _target_weights(processor, target)
                if value == DEFAULT_ACTION_LOSS_WEIGHT
            )
        if index % 100 == 0:
            print(f"checked={index}")
    if action_weighted_tokens == 0:
        raise RuntimeError("preflight found no action-weighted target tokens")
    report = {
        "schema": "cfrp.qwen3vl.stage1_sft_preflight.v1",
        "status": "passed",
        "conversation_windows": len(examples),
        "supervised_turns": sum(len(example["targets"]) for example in examples),
        "model": args.model,
        "processor_kwargs": qwen3vl_processor_kwargs(),
        "require_action_chunks": args.require_action_chunks,
        "action_weighted_tokens": action_weighted_tokens,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"conversation_windows={len(examples)}")
    print(f"supervised_turns={sum(len(example['targets']) for example in examples)}")
    print(f"report={report_path}")
    print("preflight_qwen3vl_stage1_sft: OK")
    return 0


def _target_weights(processor: object, target_xml: str) -> list[float]:
    """Re-use the production token mapping without requiring a model forward."""

    from vlnce_server.qwen3vl.loss_weights import locate_target_token_weights

    encoded = processor.tokenizer(target_xml, add_special_tokens=False, return_offsets_mapping=True)
    _, weights = locate_target_token_weights(
        target_xml,
        encoded["input_ids"],
        processor.tokenizer,
    )
    return weights


if __name__ == "__main__":
    raise SystemExit(main())
