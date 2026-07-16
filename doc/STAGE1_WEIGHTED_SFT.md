# Stage 1 Weighted SFT

This is the formal Phase 0 cold-start training path. Each source episode is a
complete shortest-path expert trajectory. It is converted into bounded
multi-turn conversation windows instead of independent per-step examples.
The first episode turn supervises compact `<plan>` initialization together
with progress, subgoal, and one to three comma-separated primitive actions.
Later turns receive the controller-owned plan and must not repeat it.

The default window contains at most eight user/assistant navigation turns. Its
first turn receives up to eight uniformly sampled historical route anchors
followed by the current observation. Later turns append exactly one current
observation after the preceding action chunk. A window therefore contains at
most sixteen images, matching StreamVLN's eight-turn slow-fast organization
without importing its additional datasets or LLaVA-specific token plumbing.
Frames are stored once per episode and referenced from windows.
For the formal full-split run, the SFT manifest references the collected NPY
frames directly; the weighted trainer loads each referenced frame and resizes it
to 384x288 in memory before the Qwen3-VL processor. Portable PNG export remains
available for small smoke manifests, but must not duplicate the full raw corpus.

Previously collected complete expert episodes remain valid raw data. Run
`resample_stage1_warmup_visual_history.py` to rewrite only their JSON frame
references to the 8+1 contract, then reconvert them with the default eight-turn
window. The migration must not rerun Habitat or copy the retained NPY frames.

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

Formal settings are `r=32`, `alpha=64`, two epochs, gradient
accumulation eight, and a two-percent episode-level validation split. The
visual contract remains 640x480 Habitat render and Qwen input 384x288. The
streaming context uses eight route anchors plus one current frame at a window
boundary, then one new current frame per dialogue turn.

The full R2R run uses two epochs, ten evenly spaced validations over one fixed
200-window subset, and five evenly spaced LoRA checkpoints. Milestones are
derived from the actual DDP optimizer-step count instead of being hard-coded,
and the final optimizer step is always both validated and checkpointed.

Evaluation keeps one bounded conversation per episode. It appends the latest
observation and previous assistant response for eight turns, then opens a new
window from the controller-owned plan and refreshed 8+1 context. vLLM prefix
caching is enabled explicitly; correctness does not depend on a cache hit.

Recovery tools are deliberately absent here. Valid `continue/replan` labels
require model-error states and counterfactual evidence, so they remain in the
later recovery warm-up rather than being fabricated from successful expert
trajectories.
