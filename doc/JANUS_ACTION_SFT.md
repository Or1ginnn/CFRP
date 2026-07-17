# Phase 0: JanusVLN-style action imitation

## Scope

Phase 0 now has one narrow objective: initialize a Qwen3-VL-4B navigation
policy from true Habitat expert actions. It does not teach CFRP planning,
progress tracking, tool selection, risk prediction, counterfactual branching,
or recovery.

## Upstream baseline

The local reference clone at `third_party/JanusVLN` preserves
MIV-XJTU/JanusVLN at commit
`64ac8373c1e3c4a810a999cad536f633f2277d68`. We reuse its expert-imitation
sample organization, not its Qwen2.5-VL-7B model, VGGT module, Habitat version,
or full-parameter training runtime. The upstream snapshot is ignored by the
CFRP repository because it does not include a root license file; this document
and the CFRP implementation retain the exact provenance without republishing
the upstream source tree.

## Data contract

One complete expert episode remains the source trajectory. Each primitive
decision becomes one independent SFT example:

```xml
<action>MOVE_FORWARD</action>
```

For decision `t`, the visual input contains every frame when `t < 9`; later
decisions contain nine uniformly sampled frames from episode start through the
current frame. The current frame is always last. Only one of
`MOVE_FORWARD`, `TURN_LEFT`, `TURN_RIGHT`, or `STOP` is legal.

The converter reconstructs primitive actions from `oracle_only.oracle_actions`
in the existing warmup JSONL. It deliberately discards old plan, progress,
subgoal, and multi-action chunk labels, so the already collected Habitat frames
do not need to be regenerated.

## Training contract

- Model: Qwen3-VL-4B-Instruct.
- Adaptation: LoRA on language attention projections; visual tower frozen.
- Loss: assistant-only causal cross entropy over one `<action>` response. All
  supervised response tokens use ordinary weight 1 by default.
- Split: by episode, never by individual decisions.
- Evaluation: execute exactly one predicted primitive action, observe the next
  RGB frame, and request the next action.

Use `scripts/train_qwen3vl_stage1_sft.py --contract action-only`. The existing
`stage1` contract remains available for later experiments but is not the Phase
0 baseline.

## Commands

Convert an existing complete expert warmup collection without rerunning
Habitat:

```bash
python scripts/convert_stage1_warmup_to_action_sft.py \
  --input-jsonl /path/to/merged/stage1_warmup.jsonl \
  --output-dir /path/to/action_sft \
  --image-storage source
```

Audit the resulting primitive trajectories before training:

```bash
python scripts/audit_action_sft.py \
  --input-jsonl /path/to/action_sft/action_sft.jsonl \
  --output-json /path/to/action_sft/audit.json \
  --check-images
```

Run the mandatory training preflight before a GPU launch:

```bash
python scripts/train_qwen3vl_stage1_sft.py \
  --contract action-only \
  --train-jsonl /path/to/action_sft/action_sft.jsonl \
  --output-dir /path/to/action_sft_dry_run \
  --dry-run
```

The closed-loop evaluator is `scripts/habitat030_r2r_action_eval.py`. It sends
one request, executes one parsed action, appends the new observation, and sends
the next request. Invalid XML is recorded as an error and cannot receive SR or
SPL credit.

## Execution order

1. Convert a small R2R expert shard and validate every image/action pair.
2. Run a two-step LoRA smoke and a small closed-loop Habitat evaluation.
3. Convert and train on all R2R train expert episodes.
4. Add EnvDrop expert trajectories after the R2R-only baseline is measurable.
5. Add DAgger/on-policy correction only if closed-loop drift remains the main
   failure mode.
6. Enter CFRP risk and counterfactual recovery phases after this action policy
   is a credible navigation baseline.
