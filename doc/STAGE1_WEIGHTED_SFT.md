# Stage 1 Weighted SFT

This is the formal Phase 0 cold-start training path. Each source episode is a
complete shortest-path expert trajectory. It is converted into bounded
multi-turn conversation windows instead of independent per-step examples.
The first episode turn supervises compact `<plan>` initialization together
with progress, subgoal, and one to three comma-separated primitive actions.
Later turns receive the controller-owned plan and must not repeat it.

The default window contains at most four user/assistant navigation turns. Its
first turn receives the full visible slow-fast context; later turns append the
one to three newly arrived contiguous frames produced by the preceding action
chunk. Frames are stored once per episode and referenced from windows.
For the formal full-split run, the SFT manifest references the collected NPY
frames directly; the weighted trainer loads each referenced frame and resizes it
to 384x288 in memory before the Qwen3-VL processor. Portable PNG export remains
available for small smoke manifests, but must not duplicate the full raw corpus.

## Objective

Every assistant XML response in the conversation uses weighted causal cross
entropy; system and user tokens are prompt-masked:

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

1. Collect the complete R2R-CE `train` split, then merge, convert, validate,
   and audit it. `val_seen` and `val_unseen` must never enter SFT.
2. Run `validate_stage1_sft_manifest.py --check-images` over every converted
   window, then run `preflight_qwen3vl_stage1_sft.py --require-action-chunks`
   over the full JSONL. The preflight checks token weights for every assistant
   target and uses a deterministic whole-dataset sample for the expensive real
   Qwen3-VL multimodal processor path. The report must state both counts.
3. Train with `torchrun`; do not use the generated LLaMA-Factory YAML for this
   run. That YAML remains an unweighted CE reference baseline.

## Four-GPU Training Shape

The weighted trainer starts one full LoRA-wrapped Qwen3-VL model per GPU under
DDP. It groups conversation windows by episode for a deterministic held-out validation
split, pads only the final DDP shard deterministically, logs rank-zero metrics
to W&B, and writes rank-zero PEFT checkpoints every configured optimizer-step
interval.

Typical formal settings are `r=32`, `alpha=64`, three epochs, gradient
accumulation eight, and a two-percent episode-level validation split. The
visual contract remains 640x480 Habitat render, Qwen input 384x288, six slow
memory anchors plus three recent consecutive frames.

The full R2R run uses two epochs, ten evenly spaced validations over one fixed
200-window subset, and five evenly spaced LoRA checkpoints. Milestones are
derived from the actual DDP optimizer-step count instead of being hard-coded,
and the final optimizer step is always both validated and checkpointed.

Recovery tools are deliberately absent here. Valid `continue/replan` labels
require model-error states and counterfactual evidence, so they remain in the
later recovery warm-up rather than being fabricated from successful expert
trajectories.
