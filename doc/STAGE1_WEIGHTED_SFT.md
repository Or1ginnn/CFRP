# Stage 1 Weighted SFT

This is the formal Phase 0 cold-start training path. It trains the normal
Stage 1 controller only: progress, subgoal, and a chunk of one to four
primitive actions. The controller-owned plan, instruction, action history, and
visual frames are prompt input and are never supervised as output tokens.

## Objective

The terminal XML uses weighted causal cross entropy:

| Target region | Weight |
| --- | ---: |
| Action primitive text | 5.0 |
| Progress text | 2.0 |
| XML tags | 1.0 |
| Free-form subgoal text | 0.25 |

The action chunks must come from the current collector. A pre-chunk SFT
manifest is not valid formal training data because it cannot teach the
multi-action execution policy.

## Required Gates

1. Recollect high-resolution R2R warm-up data after commit `ceec317` or later,
   then merge, convert, validate, and audit it.
2. Run `preflight_qwen3vl_stage1_sft.py --require-action-chunks` over the full
   converted JSONL using the actual Qwen3-VL processor. This checks every image
   and asserts that the assistant XML token alignment receives action weights.
3. Train with `torchrun`; do not use the generated LLaMA-Factory YAML for this
   run. That YAML remains an unweighted CE reference baseline.

## Four-GPU Training Shape

The weighted trainer starts one full LoRA-wrapped Qwen3-VL model per GPU under
DDP. It groups SFT records by episode for a deterministic held-out validation
split, pads only the final DDP shard deterministically, logs rank-zero metrics
to W&B, and writes rank-zero PEFT checkpoints every configured optimizer-step
interval.

Typical formal settings are `r=32`, `alpha=64`, three epochs, gradient
accumulation eight, and a two-percent episode-level validation split. The
visual contract remains 640x480 Habitat render, Qwen input 384x288, six slow
memory anchors plus three recent consecutive frames.
