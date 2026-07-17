"""Validate Stage 1 or Janus action-only SFT before training.

The processor gate checks real multimodal chat-template construction.  The
optional model-forward gate additionally runs a small no-grad Qwen3-VL forward
pass, without creating an optimizer or starting training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_qwen3vl_stage1_sft import _supervised_inputs, _weighted_causal_lm_loss
from vlnce_server.cfrp import parse_cfrp_output
from vlnce_server.qwen3vl.action_sft import load_action_sft_jsonl
from vlnce_server.qwen3vl.loss_weights import (
    DEFAULT_ACTION_LOSS_WEIGHT,
    DEFAULT_PROGRESS_LOSS_WEIGHT,
    DEFAULT_SUBGOAL_LOSS_WEIGHT,
    locate_target_action_token_mask,
)
from vlnce_server.qwen3vl.sft_manifest import (
    iter_stage1_targets,
    load_stage1_sft_jsonl,
    local_image_path,
)
from vlnce_server.qwen3vl.stage1 import DEFAULT_QWEN3_VL_MODEL
from vlnce_server.qwen3vl.vision import qwen3vl_processor_kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--contract", choices=("stage1", "action-only"), default="stage1")
    parser.add_argument("--model", default=DEFAULT_QWEN3_VL_MODEL)
    parser.add_argument("--report", required=True)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument(
        "--processor-sample-examples",
        type=int,
        help="Deterministically sample this many windows for real multimodal processing.",
    )
    parser.add_argument("--require-action-chunks", action="store_true")
    parser.add_argument(
        "--check-all-images",
        action="store_true",
        help="Decode every selected action-only JPEG and verify RGB 384x288 storage.",
    )
    parser.add_argument(
        "--model-forward-examples",
        type=int,
        default=0,
        help="Run this many no-grad Qwen3-VL forwards after processor validation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_examples is not None and args.max_examples < 1:
        raise ValueError("max-examples must be positive")
    if args.processor_sample_examples is not None and args.processor_sample_examples < 1:
        raise ValueError("processor-sample-examples must be positive")
    if args.model_forward_examples < 0:
        raise ValueError("model-forward-examples must not be negative")
    if args.contract == "action-only":
        if args.require_action_chunks:
            raise ValueError("action-only preflight cannot require multi-action chunks")
        examples = load_action_sft_jsonl(args.train_jsonl, require_janus_contract=True)
    else:
        examples = load_stage1_sft_jsonl(args.train_jsonl)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
    if args.require_action_chunks and not any(
        len(parse_cfrp_output(target).actions) > 1
        for item in examples
        for target in iter_stage1_targets(item)
    ):
        raise ValueError("manifest contains no multi-action Stage 1 targets; recollect short-chunk warm-up data")
    image_report = (
        _validate_action_images(examples) if args.contract == "action-only" and args.check_all_images else None
    )
    processor_examples = _processor_sample(examples, args.processor_sample_examples)
    try:
        import torch
        from transformers import AutoProcessor
    except ImportError as exc:
        raise RuntimeError("preflight requires torch and transformers in the Qwen training environment") from exc

    processor = AutoProcessor.from_pretrained(args.model, **qwen3vl_processor_kwargs())
    weight_args = SimpleNamespace(
        action_loss_weight=(1.0 if args.contract == "action-only" else DEFAULT_ACTION_LOSS_WEIGHT),
        stop_action_loss_weight=None,
        progress_loss_weight=DEFAULT_PROGRESS_LOSS_WEIGHT,
        subgoal_loss_weight=DEFAULT_SUBGOAL_LOSS_WEIGHT,
    )
    action_payload_tokens = 0
    for index, example in enumerate(processor_examples, start=1):
        _supervised_inputs(processor, example, torch.device("cpu"), weight_args)
        if index % 100 == 0:
            print(f"processor_checked={index}")
    for index, example in enumerate(examples, start=1):
        for target in iter_stage1_targets(example):
            action_payload_tokens += _target_action_token_count(processor, target)
        if index % 10_000 == 0:
            print(f"targets_checked={index}")
    if action_payload_tokens == 0:
        raise RuntimeError("preflight found no supervised action payload tokens")
    model_forward_report = None
    if args.model_forward_examples:
        model_forward_report = _run_model_forward_gate(
            processor,
            _model_forward_sample(examples, args.model_forward_examples),
            weight_args,
            args.model,
            torch,
        )
    report = {
        "schema": "cfrp.qwen3vl.sft_preflight.v2",
        "status": "passed",
        "contract": args.contract,
        "examples": len(examples),
        "supervised_turns": sum(len(example["targets"]) for example in examples),
        "processor_examples_checked": len(processor_examples),
        "processor_sampling": (
            "all" if len(processor_examples) == len(examples) else "stable_hash"
        ),
        "model": args.model,
        "processor_kwargs": qwen3vl_processor_kwargs(),
        "require_action_chunks": args.require_action_chunks,
        "action_payload_tokens": action_payload_tokens,
        "image_validation": image_report,
        "model_forward": model_forward_report,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"examples={len(examples)}")
    print(f"supervised_turns={sum(len(example['targets']) for example in examples)}")
    print(f"processor_examples_checked={len(processor_examples)}")
    print(f"report={report_path}")
    print("preflight_qwen3vl_stage1_sft: OK")
    return 0


def _target_action_token_count(processor: object, target_xml: str) -> int:
    """Count only action-payload token pieces using the production mapper."""

    encoded = processor.tokenizer(target_xml, add_special_tokens=False, return_offsets_mapping=True)
    _, action_mask = locate_target_action_token_mask(
        target_xml,
        encoded["input_ids"],
        processor.tokenizer,
    )
    return sum(action_mask)


def _processor_sample(
    examples: list[dict], requested: int | None
) -> list[dict]:
    if requested is None or requested >= len(examples):
        return examples
    return sorted(
        examples,
        key=lambda item: hashlib.sha256(
            f"{item['episode_id']}:{item['window_index']}".encode("utf-8")
        ).digest(),
    )[:requested]


def _model_forward_sample(examples: list[dict], requested: int) -> list[dict]:
    """Prefer the longest visual contexts so the gate exercises the 9-frame path."""

    return sorted(
        examples,
        key=lambda item: (
            -len(item.get("images", ())),
            hashlib.sha256(
                f"{item['episode_id']}:{item['window_index']}".encode("utf-8")
            ).digest(),
        ),
    )[:requested]


def _validate_action_images(examples: list[dict]) -> dict[str, int]:
    """Decode every unique collected JPEG instead of merely checking path existence."""

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for action-only image validation") from exc

    from vlnce_server.qwen3vl.vision import qwen3vl_image_size

    paths = {local_image_path(str(image)) for example in examples for image in example["images"]}
    expected_size = qwen3vl_image_size()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            if image.format != "JPEG" or image.mode != "RGB" or image.size != expected_size:
                raise ValueError(
                    f"invalid action-only image {path}: "
                    f"format={image.format!r} mode={image.mode!r} size={image.size!r}"
                )
            image.load()
    return {"unique_images_checked": len(paths), "width": expected_size[0], "height": expected_size[1]}


def _run_model_forward_gate(
    processor: object,
    examples: list[dict],
    weight_args: object,
    model_name: str,
    torch: object,
) -> dict[str, object]:
    """Run a small no-grad vision-to-logits check on the single visible GPU."""

    if not examples:
        raise ValueError("model forward gate requires at least one example")
    if not torch.cuda.is_available():
        raise RuntimeError("model forward gate requires one visible CUDA GPU")
    try:
        from transformers import AutoModelForImageTextToText
    except ImportError as exc:
        raise RuntimeError("model forward gate requires transformers") from exc

    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device
    losses: list[float] = []
    with torch.no_grad():
        for example in examples:
            inputs, labels, weights, _action_mask = _supervised_inputs(
                processor, example, device, weight_args
            )
            logits = model(**inputs).logits
            if not bool(torch.isfinite(logits[:, -1, :]).all().item()):
                raise RuntimeError("Qwen3-VL forward gate produced non-finite logits")
            losses.append(float(_weighted_causal_lm_loss(logits, labels, weights).item()))
    del model
    torch.cuda.empty_cache()
    return {
        "examples_checked": len(examples),
        "device": str(device),
        "mean_teacher_forced_loss": sum(losses) / len(losses),
        "training_started": False,
    }


if __name__ == "__main__":
    raise SystemExit(main())
