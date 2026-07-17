# Phase 0: JanusVLN-style action imitation

## Scope

Phase 0 now has one narrow objective: initialize a Qwen3-VL-4B navigation
policy from true Habitat expert actions. It does not teach CFRP planning,
progress tracking, tool selection, risk prediction, counterfactual branching,
or recovery.

## Upstream baseline

The local reference clone at `third_party/JanusVLN` preserves
MIV-XJTU/JanusVLN at commit
`64ac8373c1e3c4a810a999cad536f633f2277d68`. We reuse its R2R simulator
geometry, force-expert reference-path follower, temporal sampling, and
one-action imitation target. We do not reuse its Qwen2.5-VL-7B model, VGGT
module, Habitat version, or full-parameter training runtime. The upstream snapshot is ignored by the
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

The formal collection contract is fixed to JanusVLN's R2R settings:

- Habitat render: 640x480, RGB/depth HFOV 79 degrees;
- primitive geometry: 0.25m forward and 15-degree turns;
- episode cap and success: 500 primitive actions and 3.0m;
- force-expert follower: 1.8m for intermediate reference waypoints and 0.25m
  for the final waypoint;
- stored model frame: 384x288 JPEG, produced once at collection time.

The old 10-degree, 0.2m-waypoint warmup collection is not a formal JanusVLN
baseline. Its future observations were rendered from different headings, so it
cannot be repaired by relabeling turns. The action-only trainer rejects that
legacy manifest.

## Training contract

- Model: Qwen3-VL-4B-Instruct.
- Adaptation: LoRA on language attention projections; visual tower frozen.
- Loss: assistant-only causal cross entropy over one `<action>` response. All
  supervised response tokens use ordinary weight 1 by default.
- Logging: each W&B training metric is aggregated over the complete gradient
  accumulation group and all DDP ranks. In addition to `train/loss`, the run
  records `train/action_token_accuracy` and `train/action_exact_match` against
  the expert action payload. Validation records the corresponding
  `validation/*` metrics over the complete fixed validation subset.
- Metric scope: action accuracy is teacher-forced and reuses the training
  forward logits; XML tags and other assistant fields do not enter it. It is a
  fast imitation diagnostic, not closed-loop SR or SPL.
- Split: by episode, never by individual decisions.
- Evaluation: execute exactly one predicted primitive action, observe the next
  RGB frame, and request the next action.

Use `scripts/train_qwen3vl_stage1_sft.py --contract action-only`. The existing
`stage1` contract remains available for later experiments but is not the Phase
0 baseline.

## Commands

Collect a small direct action-SFT gate with the Habitat 0.3 environment:

```bash
python scripts/habitat030_collect_janus_action_sft.py \
  --dataset-root /path/to/R2R_VLNCE_v1-3_preprocessed \
  --scenes-dir /path/to/scene_datasets \
  --config /path/to/pointnav_habitat_test.yaml \
  --split train \
  --episode-count 3 \
  --episode-offset 0 \
  --output-dir /path/to/janus_action_sft_gate
```

For the full split, use `scripts/launch_janus_action_sft_collection.py`, then
merge only complete shards with `scripts/merge_janus_action_sft_shards.py`.

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

The measured four-GPU 4090 starting point is micro-batch 2, accumulation 4,
gradient checkpointing enabled, for a global batch of 32:

```bash
torchrun --standalone --nproc-per-node 4 scripts/train_qwen3vl_stage1_sft.py \
  --contract action-only \
  --train-jsonl /path/to/action_sft.jsonl \
  --model /path/to/Qwen3-VL-4B-Instruct \
  --output-dir /path/to/run \
  --per-device-batch-size 2 \
  --gradient-accumulation 4 \
  --gradient-checkpointing \
  --stop-file /path/to/run.STOP
```

Create the configured stop file to request a synchronized stop. Every rank
finishes the current optimizer step, then the run writes
`interrupt-checkpoint/` with the LoRA adapter, optimizer, scheduler, trainer
position, W&B run id, and rank-local random state. Continue with the identical
training contract and output directory:

```bash
torchrun --standalone --nproc-per-node 4 scripts/train_qwen3vl_stage1_sft.py \
  --contract action-only \
  --train-jsonl /path/to/action_sft.jsonl \
  --model /path/to/Qwen3-VL-4B-Instruct \
  --output-dir /path/to/run \
  --per-device-batch-size 2 \
  --gradient-accumulation 4 \
  --gradient-checkpointing \
  --stop-file /path/to/run.STOP \
  --resume-from-checkpoint /path/to/run/interrupt-checkpoint
```

Do not change the data split, seed, LoRA rank/alpha, micro-batch, accumulation,
or activation-checkpointing setting when resuming. The script rejects a
mismatched resume contract instead of silently changing optimizer semantics.

The closed-loop evaluator is `scripts/habitat030_r2r_action_eval.py`. It sends
one request, executes one parsed action, appends the new observation, and sends
the next request. Invalid XML is recorded as an error and cannot receive SR or
SPL credit.

## Execution order

1. Collect a small R2R expert shard and validate every image/action pair and
   every simulator/oracle manifest field.
2. Run a two-step LoRA smoke and a small closed-loop Habitat evaluation.
3. Convert and train on all R2R train expert episodes.
4. Add EnvDrop expert trajectories after the R2R-only baseline is measurable.
5. Add DAgger/on-policy correction only if closed-loop drift remains the main
   failure mode.
6. Enter CFRP risk and counterfactual recovery phases after this action policy
   is a credible navigation baseline.
