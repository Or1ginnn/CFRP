"""Action-weighted LoRA SFT for the normal Stage 1 Qwen3-VL contract.

This script deliberately trains only ``progress/subgoal/action``.  It has no
risk head, recovery tool, plan update, oracle field, or CFRP branch logic.
The controller plan is model input and is therefore prompt-masked.  Within the
terminal XML, primitive actions receive the highest supervised weight.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlnce_server.qwen3vl.sft_manifest import load_stage1_sft_jsonl, local_image_path
from vlnce_server.qwen3vl.stage1 import DEFAULT_QWEN3_VL_MODEL
from vlnce_server.qwen3vl.vision import qwen3vl_processor_kwargs
from vlnce_server.qwen3vl.loss_weights import (
    DEFAULT_ACTION_LOSS_WEIGHT,
    DEFAULT_PROGRESS_LOSS_WEIGHT,
    DEFAULT_SUBGOAL_LOSS_WEIGHT,
    locate_target_token_weights,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--model", default=DEFAULT_QWEN3_VL_MODEL)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--action-loss-weight", type=float, default=DEFAULT_ACTION_LOSS_WEIGHT)
    parser.add_argument("--progress-loss-weight", type=float, default=DEFAULT_PROGRESS_LOSS_WEIGHT)
    parser.add_argument("--subgoal-loss-weight", type=float, default=DEFAULT_SUBGOAL_LOSS_WEIGHT)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.gradient_accumulation < 1 or args.lora_rank < 1 or args.lora_alpha < 1:
        raise ValueError("epochs, gradient-accumulation, lora-rank, and lora-alpha must be positive")
    if min(args.action_loss_weight, args.progress_loss_weight, args.subgoal_loss_weight) <= 0:
        raise ValueError("all Stage 1 loss weights must be positive")
    examples = load_stage1_sft_jsonl(args.train_jsonl)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
    if not examples:
        raise ValueError("no SFT examples selected")
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    if args.dry_run:
        _write_run_manifest(output_dir, args, examples, status="dry_run")
        print(f"examples={len(examples)}")
        print("qwen3vl_stage1_sft_dry_run: OK")
        return 0

    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "training requires torch, transformers, and peft in the dedicated cfrp-qwen3vl environment"
        ) from exc

    torch.manual_seed(args.seed)
    processor = AutoProcessor.from_pretrained(args.model, **qwen3vl_processor_kwargs())
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto"
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            task_type="CAUSAL_LM",
        ),
    )
    model.train()
    optimizer = torch.optim.AdamW((item for item in model.parameters() if item.requires_grad), lr=args.learning_rate)
    device = model.get_input_embeddings().weight.device
    step = 0
    optimizer.zero_grad(set_to_none=True)
    for _epoch in range(args.epochs):
        for example in examples:
            inputs, labels, token_weights = _supervised_inputs(processor, example, device, args)
            outputs = model(**inputs)
            loss = _weighted_causal_lm_loss(outputs.logits, labels, token_weights) / args.gradient_accumulation
            loss.backward()
            if (step + 1) % args.gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            step += 1
            if args.max_steps is not None and step >= args.max_steps:
                break
        if args.max_steps is not None and step >= args.max_steps:
            break
    if step % args.gradient_accumulation:
        optimizer.step()
    model.save_pretrained(output_dir / "adapter")
    processor.save_pretrained(output_dir / "processor")
    _write_run_manifest(output_dir, args, examples, status="completed", optimizer_steps=(step + args.gradient_accumulation - 1) // args.gradient_accumulation)
    print(f"examples={len(examples)}")
    print(f"micro_steps={step}")
    print(f"output_dir={output_dir}")
    print("qwen3vl_stage1_sft: OK")
    return 0


def _supervised_inputs(
    processor: Any, example: dict[str, Any], device: Any, args: argparse.Namespace
) -> tuple[Any, Any, Any]:
    """Mask every prompt token and supervise only the terminal XML response."""

    messages = _messages_with_processor_image_paths(example["messages"])
    prompt = processor.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    full = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors="pt"
    )
    prompt_ids = prompt["input_ids"]
    full_ids = full["input_ids"]
    prefix_length = prompt_ids.shape[1]
    if full_ids.shape[1] <= prefix_length or not torch_equal_prefix(full_ids, prompt_ids):
        raise RuntimeError("Qwen chat template changed: assistant target is not a prefix extension")
    labels = full_ids.clone()
    labels[:, :prefix_length] = -100
    token_weights = full_ids.new_ones(full_ids.shape, dtype=getattr(full_ids, "dtype", None)).float()
    token_weights[:, :prefix_length] = 0.0
    target_suffix = full_ids[0, prefix_length:].tolist()
    target_start, target_region_weights = locate_target_token_weights(
        example["target_xml"],
        target_suffix,
        processor.tokenizer,
        action_weight=args.action_loss_weight,
        progress_weight=args.progress_loss_weight,
        subgoal_weight=args.subgoal_loss_weight,
    )
    start = prefix_length + target_start
    token_weights[0, start : start + len(target_region_weights)] = token_weights.new_tensor(target_region_weights)
    return full.to(device), labels.to(device), token_weights.to(device)


def _weighted_causal_lm_loss(logits: Any, labels: Any, token_weights: Any) -> Any:
    """Causal CE with prompt masking and CFRP terminal-region weights."""

    import torch.nn.functional as functional

    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    shifted_weights = token_weights[:, 1:].contiguous()
    per_token = functional.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        shifted_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view_as(shifted_labels)
    mask = shifted_labels.ne(-100)
    effective_weights = shifted_weights * mask
    return (per_token * effective_weights).sum() / effective_weights.sum().clamp_min(1.0)


def torch_equal_prefix(full_ids: Any, prompt_ids: Any) -> bool:
    return bool((full_ids[:, : prompt_ids.shape[1]] == prompt_ids).all().item())


def _messages_with_processor_image_paths(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert portable file URIs to paths accepted by the Qwen processor."""

    normalized = []
    for message in messages:
        content = message["content"]
        if not isinstance(content, list):
            normalized.append(dict(message))
            continue
        blocks = []
        for block in content:
            copied = dict(block)
            if copied.get("type") == "image":
                copied["image"] = str(local_image_path(copied["image"]))
            blocks.append(copied)
        normalized.append({**message, "content": blocks})
    return normalized


def _write_run_manifest(output_dir: Path, args: argparse.Namespace, examples: list[dict[str, Any]], *, status: str, optimizer_steps: int = 0) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema": "cfrp.qwen3vl.stage1_sft_run.v1",
                "status": status,
                "model": args.model,
                "examples": len(examples),
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "gradient_accumulation": args.gradient_accumulation,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "loss_weights": {
                    "action": args.action_loss_weight,
                    "progress": args.progress_loss_weight,
                    "subgoal": args.subgoal_loss_weight,
                    "xml": 1.0,
                },
                "optimizer_steps": optimizer_steps,
                "seed": args.seed,
                "processor_kwargs": qwen3vl_processor_kwargs(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
