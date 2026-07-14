"""Action-weighted LoRA SFT for the normal Stage 1 Qwen3-VL contract.

This script deliberately trains only ``progress/subgoal/action``.  It has no
risk head, recovery tool, plan update, oracle field, or CFRP branch logic.
The controller plan is model input and is therefore prompt-masked.  Within the
terminal XML, primitive actions receive the highest supervised weight.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
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
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--action-loss-weight", type=float, default=DEFAULT_ACTION_LOSS_WEIGHT)
    parser.add_argument("--progress-loss-weight", type=float, default=DEFAULT_PROGRESS_LOSS_WEIGHT)
    parser.add_argument("--subgoal-loss-weight", type=float, default=DEFAULT_SUBGOAL_LOSS_WEIGHT)
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--save-every-optimizer-steps", type=int, default=100)
    parser.add_argument("--wandb-project")
    parser.add_argument("--run-name")
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
    if not 0 <= args.validation_fraction < 1:
        raise ValueError("validation-fraction must be in [0, 1)")
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError("warmup-ratio must be in [0, 1)")
    if args.save_every_optimizer_steps < 1:
        raise ValueError("save-every-optimizer-steps must be positive")
    examples = load_stage1_sft_jsonl(args.train_jsonl)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
    if not examples:
        raise ValueError("no SFT examples selected")
    train_examples, validation_examples = _split_examples_by_episode(
        examples, args.validation_fraction, args.seed
    )
    if not train_examples:
        raise ValueError("validation split left no training examples")
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {output_dir}")
    if args.dry_run:
        _write_run_manifest(
            output_dir,
            args,
            examples,
            status="dry_run",
            train_examples=len(train_examples),
            validation_examples=len(validation_examples),
        )
        print(f"examples={len(examples)} train={len(train_examples)} validation={len(validation_examples)}")
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

    runtime = _initialize_runtime(torch)
    torch.manual_seed(args.seed + runtime.rank)
    processor = AutoProcessor.from_pretrained(args.model, **qwen3vl_processor_kwargs())
    model_kwargs: dict[str, Any] = {"dtype": torch.bfloat16}
    if runtime.distributed:
        # Each DDP process owns one complete model replica on its local GPU.
        model_kwargs["device_map"] = runtime.local_rank
    else:
        model_kwargs["device_map"] = "auto"
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, **model_kwargs
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
    if runtime.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[runtime.local_rank],
            output_device=runtime.local_rank,
            find_unused_parameters=False,
        )
    model.train()
    local_train_examples = _equal_train_shard(train_examples, runtime.rank, runtime.world_size)
    local_validation_examples = validation_examples[runtime.rank :: runtime.world_size]
    optimizer = torch.optim.AdamW((item for item in model.parameters() if item.requires_grad), lr=args.learning_rate)
    scheduler = _cosine_scheduler(
        optimizer,
        total_steps=_optimizer_step_count(len(local_train_examples), args),
        warmup_ratio=args.warmup_ratio,
    )
    wandb_run = _start_wandb(args, runtime, examples, train_examples, validation_examples)
    try:
        micro_steps, optimizer_steps = _train(
            model,
            processor,
            optimizer,
            scheduler,
            local_train_examples,
            local_validation_examples,
            args,
            runtime,
            wandb_run,
        )
        validation_loss = _evaluate(
            model, processor, local_validation_examples, args, runtime
        )
        if wandb_run is not None:
            wandb_run.log({"validation/loss": validation_loss, "train/optimizer_steps": optimizer_steps})
        _save_adapter(model, processor, output_dir / "adapter", runtime)
        if runtime.is_main:
            _write_run_manifest(
                output_dir,
                args,
                examples,
                status="completed",
                optimizer_steps=optimizer_steps,
                train_examples=len(train_examples),
                validation_examples=len(validation_examples),
                validation_loss=validation_loss,
                distributed_world_size=runtime.world_size,
            )
            print(f"examples={len(examples)} train={len(train_examples)} validation={len(validation_examples)}")
            print(f"micro_steps_per_rank={micro_steps}")
            print(f"optimizer_steps={optimizer_steps}")
            print(f"validation_loss={validation_loss:.6f}")
            print(f"output_dir={output_dir}")
            print("qwen3vl_stage1_sft: OK")
        return 0
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        _shutdown_runtime(runtime, torch)


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


@dataclass(frozen=True)
class _Runtime:
    rank: int
    local_rank: int
    world_size: int
    distributed: bool
    device: Any

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def _initialize_runtime(torch: Any) -> _Runtime:
    """Initialize one full Qwen model per GPU when launched with torchrun."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("multi-process Stage 1 SFT requires CUDA/NCCL")
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    return _Runtime(rank, local_rank, world_size, distributed, device)


def _shutdown_runtime(runtime: _Runtime, torch: Any) -> None:
    if runtime.distributed and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def _split_examples_by_episode(
    examples: list[dict[str, Any]], validation_fraction: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split by episode, never by individual decision, to prevent path leakage."""

    if validation_fraction == 0 or len({str(item["episode_id"]) for item in examples}) < 2:
        return list(examples), []
    by_episode: dict[str, list[dict[str, Any]]] = {}
    for example in examples:
        by_episode.setdefault(str(example["episode_id"]), []).append(example)
    episode_ids = sorted(by_episode)
    validation_ids = {
        episode_id
        for episode_id in episode_ids
        if _stable_unit_interval(f"{seed}:{episode_id}") < validation_fraction
    }
    # Keep the split useful for small smoke manifests too.
    if not validation_ids:
        validation_ids.add(episode_ids[-1])
    if len(validation_ids) == len(episode_ids):
        validation_ids.remove(episode_ids[0])
    train = [item for item in examples if str(item["episode_id"]) not in validation_ids]
    validation = [item for item in examples if str(item["episode_id"]) in validation_ids]
    return train, validation


def _stable_unit_interval(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _equal_train_shard(
    examples: list[dict[str, Any]], rank: int, world_size: int
) -> list[dict[str, Any]]:
    """Shard uniformly and pad deterministically so every DDP rank backprops equally."""

    per_rank = (len(examples) + world_size - 1) // world_size
    total = per_rank * world_size
    padded = examples + [examples[index % len(examples)] for index in range(total - len(examples))]
    return padded[rank:total:world_size]


def _optimizer_step_count(local_examples: int, args: argparse.Namespace) -> int:
    remaining = args.max_steps
    total = 0
    for _epoch in range(args.epochs):
        micro_steps = local_examples if remaining is None else min(local_examples, remaining)
        total += math.ceil(micro_steps / args.gradient_accumulation)
        if remaining is not None:
            remaining -= micro_steps
            if remaining == 0:
                break
    return max(total, 1)


def _cosine_scheduler(optimizer: Any, *, total_steps: int, warmup_ratio: float) -> Any:
    import torch

    warmup_steps = int(total_steps * warmup_ratio)

    def scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _train(
    model: Any,
    processor: Any,
    optimizer: Any,
    scheduler: Any,
    examples: list[dict[str, Any]],
    validation_examples: list[dict[str, Any]],
    args: argparse.Namespace,
    runtime: _Runtime,
    wandb_run: Any,
) -> tuple[int, int]:
    optimizer.zero_grad(set_to_none=True)
    remaining_micro_steps = args.max_steps
    total_micro_steps = 0
    optimizer_steps = 0
    for epoch in range(args.epochs):
        max_micro_steps = (
            len(examples)
            if remaining_micro_steps is None
            else min(len(examples), remaining_micro_steps)
        )
        if max_micro_steps == 0:
            break
        epoch_examples = list(examples)
        random.Random(args.seed + runtime.rank + epoch * 10_007).shuffle(epoch_examples)
        for micro_index, example in enumerate(epoch_examples[:max_micro_steps]):
            group_start = (micro_index // args.gradient_accumulation) * args.gradient_accumulation
            group_end = min(group_start + args.gradient_accumulation, max_micro_steps)
            should_step = micro_index + 1 == group_end
            divisor = group_end - group_start
            sync_context = (
                model.no_sync() if runtime.distributed and not should_step else contextlib.nullcontext()
            )
            with sync_context:
                inputs, labels, token_weights = _supervised_inputs(processor, example, runtime.device, args)
                loss = _weighted_causal_lm_loss(model(**inputs).logits, labels, token_weights)
                (loss / divisor).backward()
            total_micro_steps += 1
            if not should_step:
                continue
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            mean_loss = _mean_across_ranks(loss.detach(), runtime)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": mean_loss,
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "train/epoch": epoch + (micro_index + 1) / max_micro_steps,
                    },
                    step=optimizer_steps,
                )
            if optimizer_steps % args.save_every_optimizer_steps == 0:
                _save_adapter(model, processor, Path(args.output_dir) / f"checkpoint-{optimizer_steps}", runtime)
        validation_loss = _evaluate(model, processor, validation_examples, args, runtime)
        if wandb_run is not None and validation_loss == validation_loss:
            wandb_run.log({"validation/loss": validation_loss, "train/epoch": epoch + 1}, step=optimizer_steps)
        if remaining_micro_steps is not None:
            remaining_micro_steps -= max_micro_steps
    return total_micro_steps, optimizer_steps


def _evaluate(
    model: Any,
    processor: Any,
    examples: list[dict[str, Any]],
    args: argparse.Namespace,
    runtime: _Runtime,
) -> float:
    if not examples and not runtime.distributed:
        return float("nan")
    import torch

    model.eval()
    loss_sum = torch.zeros((), device=runtime.device, dtype=torch.float32)
    count = torch.zeros((), device=runtime.device, dtype=torch.float32)
    with torch.no_grad():
        for example in examples:
            inputs, labels, token_weights = _supervised_inputs(processor, example, runtime.device, args)
            loss_sum += _weighted_causal_lm_loss(model(**inputs).logits, labels, token_weights).float()
            count += 1
    if runtime.distributed:
        torch.distributed.all_reduce(loss_sum)
        torch.distributed.all_reduce(count)
    model.train()
    return float((loss_sum / count.clamp_min(1)).item()) if count.item() else float("nan")


def _mean_across_ranks(loss: Any, runtime: _Runtime) -> float:
    if runtime.distributed:
        import torch

        torch.distributed.all_reduce(loss)
        loss /= runtime.world_size
    return float(loss.item())


def _save_adapter(model: Any, processor: Any, path: Path, runtime: _Runtime) -> None:
    if runtime.distributed:
        import torch as imported_torch

        imported_torch.distributed.barrier()
    if runtime.is_main:
        path.mkdir(parents=True, exist_ok=True)
        target_model = model.module if hasattr(model, "module") else model
        target_model.save_pretrained(path)
        processor.save_pretrained(path / "processor")
    if runtime.distributed:
        import torch as imported_torch

        imported_torch.distributed.barrier()


def _start_wandb(
    args: argparse.Namespace,
    runtime: _Runtime,
    examples: list[dict[str, Any]],
    train_examples: list[dict[str, Any]],
    validation_examples: list[dict[str, Any]],
) -> Any:
    if not args.wandb_project or not runtime.is_main:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("--wandb-project requires wandb in the training environment") from exc
    return wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        config={
            "examples": len(examples),
            "train_examples": len(train_examples),
            "validation_examples": len(validation_examples),
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "action_loss_weight": args.action_loss_weight,
            "progress_loss_weight": args.progress_loss_weight,
            "subgoal_loss_weight": args.subgoal_loss_weight,
            "warmup_ratio": args.warmup_ratio,
            "world_size": runtime.world_size,
        },
    )


def torch_equal_prefix(full_ids: Any, prompt_ids: Any) -> bool:
    return bool((full_ids[:, : prompt_ids.shape[1]] == prompt_ids).all().item())


def _messages_with_processor_image_paths(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize portable messages for old and new Qwen processor APIs.

    The manifest keeps system and assistant text compact as strings, while
    Transformers 5.6 expects every role to expose a list of content blocks.
    Newer processors accept the block form too, so normalize at this runtime
    boundary rather than version-gating the training environment.
    """

    normalized = []
    for message in messages:
        content = message["content"]
        if not isinstance(content, list):
            normalized.append(
                {
                    **message,
                    "content": [{"type": "text", "text": str(content)}],
                }
            )
            continue
        blocks = []
        for block in content:
            copied = dict(block)
            if copied.get("type") == "image":
                copied["image"] = str(local_image_path(copied["image"]))
            blocks.append(copied)
        normalized.append({**message, "content": blocks})
    return normalized


def _write_run_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    examples: list[dict[str, Any]],
    *,
    status: str,
    optimizer_steps: int = 0,
    train_examples: int | None = None,
    validation_examples: int | None = None,
    validation_loss: float | None = None,
    distributed_world_size: int = 1,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema": "cfrp.qwen3vl.stage1_sft_run.v1",
                "objective": "action_weighted_causal_cross_entropy",
                "status": status,
                "model": args.model,
                "examples": len(examples),
                "train_examples": len(examples) if train_examples is None else train_examples,
                "validation_examples": 0 if validation_examples is None else validation_examples,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "warmup_ratio": args.warmup_ratio,
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
                "validation_loss": validation_loss,
                "distributed_world_size": distributed_world_size,
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
