"""LoRA SFT for Qwen3-VL Stage 1 or the Phase 0 action-only contract.

This script trains every assistant turn in a bounded streaming conversation.
The first episode turn also learns compact plan initialization; later turns
learn ``progress/subgoal/action`` without repeating the controller-owned plan.
It has no risk head, recovery tool, plan update, oracle field, or branch logic.
Within each assistant XML response, primitive actions receive the highest
supervised weight.
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
from vlnce_server.qwen3vl.action_sft import load_action_sft_jsonl
from vlnce_server.qwen3vl.stage1 import DEFAULT_QWEN3_VL_MODEL
from vlnce_server.qwen3vl.vision import prepare_qwen3vl_image, qwen3vl_processor_kwargs
from vlnce_server.qwen3vl.loss_weights import (
    DEFAULT_ACTION_LOSS_WEIGHT,
    DEFAULT_PROGRESS_LOSS_WEIGHT,
    DEFAULT_SUBGOAL_LOSS_WEIGHT,
    locate_target_token_weights,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument(
        "--contract",
        choices=("stage1", "action-only"),
        default="stage1",
        help="Supervise the full Stage 1 XML or one JanusVLN-style primitive action.",
    )
    parser.add_argument("--model", default=DEFAULT_QWEN3_VL_MODEL)
    parser.add_argument(
        "--initial-adapter",
        help="Optional LoRA adapter whose trainable weights initialize this continuation run",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument(
        "--action-loss-weight",
        type=float,
        help="Defaults to 5 for Stage 1 and ordinary weight 1 for action-only SFT.",
    )
    parser.add_argument("--stop-action-loss-weight", type=float)
    parser.add_argument("--progress-loss-weight", type=float, default=DEFAULT_PROGRESS_LOSS_WEIGHT)
    parser.add_argument("--subgoal-loss-weight", type=float, default=DEFAULT_SUBGOAL_LOSS_WEIGHT)
    parser.add_argument("--validation-fraction", type=float, default=0.02)
    parser.add_argument("--save-every-optimizer-steps", type=int, default=100)
    parser.add_argument("--validation-runs", type=int, default=0)
    parser.add_argument("--validation-max-examples", type=int)
    parser.add_argument("--checkpoint-count", type=int, default=0)
    parser.add_argument("--wandb-project")
    parser.add_argument("--run-name")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.action_loss_weight is None:
        args.action_loss_weight = (
            1.0 if args.contract == "action-only" else DEFAULT_ACTION_LOSS_WEIGHT
        )
    if args.epochs < 1 or args.gradient_accumulation < 1 or args.lora_rank < 1 or args.lora_alpha < 1:
        raise ValueError("epochs, gradient-accumulation, lora-rank, and lora-alpha must be positive")
    configured_weights = [
        args.action_loss_weight,
        args.progress_loss_weight,
        args.subgoal_loss_weight,
    ]
    if args.stop_action_loss_weight is not None:
        configured_weights.append(args.stop_action_loss_weight)
    if min(configured_weights) <= 0:
        raise ValueError("all Stage 1 loss weights must be positive")
    if args.initial_adapter is not None:
        _validate_initial_adapter(Path(args.initial_adapter), args)
    if not 0 <= args.validation_fraction < 1:
        raise ValueError("validation-fraction must be in [0, 1)")
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError("warmup-ratio must be in [0, 1)")
    if args.save_every_optimizer_steps < 1:
        raise ValueError("save-every-optimizer-steps must be positive")
    if args.validation_runs < 0 or args.checkpoint_count < 0:
        raise ValueError("validation-runs and checkpoint-count must not be negative")
    if args.validation_max_examples is not None and args.validation_max_examples < 1:
        raise ValueError("validation-max-examples must be positive")
    loader = load_action_sft_jsonl if args.contract == "action-only" else load_stage1_sft_jsonl
    examples = loader(args.train_jsonl)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
    if not examples:
        raise ValueError("no SFT examples selected")
    train_examples, validation_examples = _split_examples_by_episode(
        examples, args.validation_fraction, args.seed
    )
    if not train_examples:
        raise ValueError("validation split left no training examples")
    validation_examples = _fixed_validation_examples(
        validation_examples,
        args.validation_max_examples,
        args.seed,
    )
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
        print(f"qwen3vl_{args.contract.replace('-', '_')}_sft_dry_run: OK")
        return 0

    try:
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
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
    if args.initial_adapter is not None:
        model = PeftModel.from_pretrained(
            model,
            args.initial_adapter,
            is_trainable=True,
        )
    else:
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
    total_optimizer_steps = _optimizer_step_count(len(local_train_examples), args)
    validation_steps = _milestone_steps(total_optimizer_steps, args.validation_runs)
    checkpoint_steps = _milestone_steps(total_optimizer_steps, args.checkpoint_count)
    optimizer = torch.optim.AdamW((item for item in model.parameters() if item.requires_grad), lr=args.learning_rate)
    scheduler = _cosine_scheduler(
        optimizer,
        total_steps=total_optimizer_steps,
        warmup_ratio=args.warmup_ratio,
    )
    wandb_run = _start_wandb(args, runtime, examples, train_examples, validation_examples)
    try:
        micro_steps, optimizer_steps, validation_loss = _train(
            model,
            processor,
            optimizer,
            scheduler,
            local_train_examples,
            local_validation_examples,
            args,
            runtime,
            wandb_run,
            validation_steps=validation_steps,
            checkpoint_steps=checkpoint_steps,
        )
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
                validation_steps=validation_steps,
                checkpoint_steps=checkpoint_steps,
            )
            print(f"examples={len(examples)} train={len(train_examples)} validation={len(validation_examples)}")
            print(f"micro_steps_per_rank={micro_steps}")
            print(f"optimizer_steps={optimizer_steps}")
            print(f"validation_loss={validation_loss:.6f}")
            print(f"output_dir={output_dir}")
            print(f"qwen3vl_{args.contract.replace('-', '_')}_sft: OK")
        return 0
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        _shutdown_runtime(runtime, torch)


def _validate_initial_adapter(path: Path, args: argparse.Namespace) -> None:
    config_path = path / "adapter_config.json"
    weights_path = path / "adapter_model.safetensors"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rank = int(config.get("r", 0))
    alpha = int(config.get("lora_alpha", 0))
    if rank != args.lora_rank or alpha != args.lora_alpha:
        raise ValueError(
            "initial adapter LoRA config does not match requested continuation config: "
            f"adapter r/alpha={rank}/{alpha}, requested={args.lora_rank}/{args.lora_alpha}"
        )


def _supervised_inputs(
    processor: Any, example: dict[str, Any], device: Any, args: argparse.Namespace
) -> tuple[Any, Any, Any]:
    """Mask user/system tokens and supervise every assistant XML response."""

    messages = _messages_with_processor_image_paths(example["messages"])
    full = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors="pt"
    )
    _require_visual_tensors(full, "full supervised example")
    full_ids = full["input_ids"]
    labels = full_ids.new_full(full_ids.shape, -100)
    token_weights = full_ids.new_ones(full_ids.shape, dtype=getattr(full_ids, "dtype", None)).float()
    token_weights.zero_()

    for target in example["targets"]:
        message_index = int(target["message_index"])
        target_xml = str(target["target_xml"])
        prompt = processor.apply_chat_template(
            messages[:message_index],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        through_target = processor.apply_chat_template(
            messages[: message_index + 1],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )
        _require_visual_tensors(prompt, f"prompt before assistant message {message_index}")
        _require_visual_tensors(through_target, f"assistant message {message_index}")
        prompt_ids = prompt["input_ids"]
        through_ids = through_target["input_ids"]
        prefix_length = prompt_ids.shape[1]
        target_end = through_ids.shape[1]
        if (
            target_end <= prefix_length
            or not torch_equal_prefix(full_ids, prompt_ids)
            or not torch_equal_prefix(full_ids, through_ids)
        ):
            raise RuntimeError(
                "Qwen chat template changed: a multi-turn assistant target is not a prefix extension"
            )
        labels[:, prefix_length:target_end] = full_ids[:, prefix_length:target_end]
        token_weights[:, prefix_length:target_end] = 1.0
        target_suffix = full_ids[0, prefix_length:target_end].tolist()
        target_start, target_region_weights = locate_target_token_weights(
            target_xml,
            target_suffix,
            processor.tokenizer,
            action_weight=args.action_loss_weight,
            stop_action_weight=args.stop_action_loss_weight,
            progress_weight=args.progress_loss_weight,
            subgoal_weight=args.subgoal_loss_weight,
        )
        start = prefix_length + target_start
        token_weights[0, start : start + len(target_region_weights)] = token_weights.new_tensor(
            target_region_weights
        )
    if not bool(labels.ne(-100).any().item()):
        raise RuntimeError("multi-turn SFT example contains no supervised assistant tokens")
    return full.to(device), labels.to(device), token_weights.to(device)


def _require_visual_tensors(inputs: Any, source: str) -> None:
    """Fail loudly if a multimodal SFT sample was reduced to text only."""

    missing = [key for key in ("pixel_values", "image_grid_thw") if key not in inputs]
    if missing:
        raise RuntimeError(
            f"Qwen3-VL {source} is missing visual tensors: {', '.join(missing)}"
        )


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


def _fixed_validation_examples(
    examples: list[dict[str, Any]], maximum: int | None, seed: int
) -> list[dict[str, Any]]:
    """Choose a stable validation window subset without changing episode isolation."""

    if maximum is None or maximum >= len(examples):
        return list(examples)
    return sorted(
        examples,
        key=lambda item: hashlib.sha256(
            f"{seed}:{item['episode_id']}:{item['window_index']}".encode("utf-8")
        ).digest(),
    )[:maximum]


def _milestone_steps(total_steps: int, count: int) -> tuple[int, ...]:
    """Return evenly spaced one-based optimizer steps, always including the final step."""

    if count == 0:
        return tuple()
    if total_steps < 1 or count > total_steps:
        raise ValueError("milestone count must not exceed total optimizer steps")
    steps = tuple(round(total_steps * index / count) for index in range(1, count + 1))
    if len(set(steps)) != count or steps[-1] != total_steps:
        raise RuntimeError("failed to construct unique optimizer milestones")
    return steps


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
    *,
    validation_steps: tuple[int, ...] = tuple(),
    checkpoint_steps: tuple[int, ...] = tuple(),
) -> tuple[int, int, float]:
    optimizer.zero_grad(set_to_none=True)
    remaining_micro_steps = args.max_steps
    total_micro_steps = 0
    optimizer_steps = 0
    validation_loss = float("nan")
    validation_step_set = set(validation_steps)
    checkpoint_step_set = set(checkpoint_steps)
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
            should_save = (
                optimizer_steps in checkpoint_step_set
                if checkpoint_step_set
                else optimizer_steps % args.save_every_optimizer_steps == 0
            )
            if should_save:
                _save_adapter(model, processor, Path(args.output_dir) / f"checkpoint-{optimizer_steps}", runtime)
            if optimizer_steps in validation_step_set:
                validation_loss = _evaluate(
                    model, processor, validation_examples, args, runtime
                )
                if wandb_run is not None and validation_loss == validation_loss:
                    wandb_run.log(
                        {"validation/loss": validation_loss}, step=optimizer_steps
                    )
        if not validation_step_set:
            validation_loss = _evaluate(model, processor, validation_examples, args, runtime)
            if wandb_run is not None and validation_loss == validation_loss:
                wandb_run.log(
                    {"validation/loss": validation_loss, "train/epoch": epoch + 1},
                    step=optimizer_steps,
                )
        if remaining_micro_steps is not None:
            remaining_micro_steps -= max_micro_steps
    if validation_step_set and optimizer_steps not in validation_step_set:
        validation_loss = _evaluate(model, processor, validation_examples, args, runtime)
    return total_micro_steps, optimizer_steps, validation_loss


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
    evaluation_model = _unwrap_distributed_model(model)
    loss_sum = torch.zeros((), device=runtime.device, dtype=torch.float32)
    count = torch.zeros((), device=runtime.device, dtype=torch.float32)
    with torch.no_grad():
        for example in examples:
            inputs, labels, token_weights = _supervised_inputs(processor, example, runtime.device, args)
            loss_sum += _weighted_causal_lm_loss(
                evaluation_model(**inputs).logits, labels, token_weights
            ).float()
            count += 1
    if runtime.distributed:
        torch.distributed.all_reduce(loss_sum)
        torch.distributed.all_reduce(count)
    model.train()
    return float((loss_sum / count.clamp_min(1)).item()) if count.item() else float("nan")


def _unwrap_distributed_model(model: Any) -> Any:
    """Bypass DDP forward collectives for uneven per-rank evaluation shards."""
    return model.module if hasattr(model, "module") else model


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
        target_model = _unwrap_distributed_model(model)
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
            "conversation_windows": len(examples),
            "supervised_turns": _supervised_turn_count(examples),
            "train_windows": len(train_examples),
            "validation_windows": len(validation_examples),
            "validation_runs": args.validation_runs,
            "validation_max_examples": args.validation_max_examples,
            "checkpoint_count": args.checkpoint_count,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "initial_adapter": args.initial_adapter,
            "action_loss_weight": args.action_loss_weight,
            "stop_action_loss_weight": args.stop_action_loss_weight,
            "progress_loss_weight": args.progress_loss_weight,
            "subgoal_loss_weight": args.subgoal_loss_weight,
            "warmup_ratio": args.warmup_ratio,
            "world_size": runtime.world_size,
        },
    )


def torch_equal_prefix(full_ids: Any, prompt_ids: Any) -> bool:
    return bool((full_ids[:, : prompt_ids.shape[1]] == prompt_ids).all().item())


def _supervised_turn_count(examples: list[dict[str, Any]]) -> int:
    return sum(len(example.get("targets", ())) for example in examples)


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
                copied["image"] = _processor_image(copied["image"])
            blocks.append(copied)
        normalized.append({**message, "content": blocks})
    return normalized


def _processor_image(source: str) -> Any:
    """Load collected NPY frames once per window; keep portable images as paths."""

    path = local_image_path(source)
    if path.suffix.lower() != ".npy":
        return str(path)
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required to load referenced Stage 1 RGB frames") from exc
    return prepare_qwen3vl_image(np.load(path))


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
    validation_steps: tuple[int, ...] = tuple(),
    checkpoint_steps: tuple[int, ...] = tuple(),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema": "cfrp.qwen3vl.sft_run.v2",
                "contract": args.contract,
                "objective": _objective_name(args),
                "status": status,
                "model": args.model,
                "initial_adapter": args.initial_adapter,
                "examples": len(examples),
                "conversation_windows": len(examples),
                "supervised_turns": _supervised_turn_count(examples),
                "train_examples": len(examples) if train_examples is None else train_examples,
                "validation_examples": 0 if validation_examples is None else validation_examples,
                "validation_runs": args.validation_runs,
                "validation_max_examples": args.validation_max_examples,
                "checkpoint_count": args.checkpoint_count,
                "validation_steps": list(validation_steps),
                "checkpoint_steps": list(checkpoint_steps),
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "warmup_ratio": args.warmup_ratio,
                "gradient_accumulation": args.gradient_accumulation,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "loss_weights": _manifest_loss_weights(args),
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


def _manifest_loss_weights(args: argparse.Namespace) -> dict[str, float]:
    weights = {
        "action": args.action_loss_weight,
        "stop_action": (
            args.action_loss_weight
            if args.stop_action_loss_weight is None
            else args.stop_action_loss_weight
        ),
        "xml": 1.0,
    }
    if args.contract == "stage1":
        weights.update(
            progress=args.progress_loss_weight,
            subgoal=args.subgoal_loss_weight,
        )
    return weights


def _objective_name(args: argparse.Namespace) -> str:
    weights = _manifest_loss_weights(args)
    if args.contract == "action-only" and all(value == 1.0 for value in weights.values()):
        return "assistant_only_causal_cross_entropy"
    return "action_weighted_causal_cross_entropy"


if __name__ == "__main__":
    raise SystemExit(main())
