# Phase 0.2 Baseline Migration Audit

## 1. Decision

Phase 0.2 audits the copied ActiveVLN baseline and fixes the migration boundary
for CFRP. The implementation target is:

```text
Qwen3-VL-4B policy process (Python 3.10+, modern Transformers/vLLM/veRL)
                         |
                         | HTTP batch API
                         v
Habitat-Lab 0.3 process (Python 3.9, Habitat task and simulator)
```

The two environments must remain separate. Do not install Qwen, Transformers,
vLLM, or veRL into the Habitat 0.3 environment, and do not install Habitat into
the model-training environment.

R2R-CE is the first benchmark. RxR-CE is deferred until one R2R-CE episode and
the R2R validation path work end to end.

## 2. Baseline Definition

Phase 0 uses three explicit gates instead of treating every copied ActiveVLN
module as one baseline:

1. **B0 Environment parity**: Habitat 0.3 resets a real R2R-CE episode, exposes
   instruction/RGB, executes primitive actions and reports task metrics.
2. **B1 ActiveVLN parity reference**: when the published checkpoint is
   available, run the original Qwen2.5-VL-3B SFT policy as a migration
   comparison. This is a reference, not the final CFRP backbone.
3. **B2 CFRP base policy**: Qwen3-VL-4B-Instruct uses the Stage 1
   `progress/subgoal/action` protocol with a controller-owned plan and fixed
   visual/action history. B2 is the baseline used by Stage 1 data collection.

The method may proceed to Phase 1 only after B2 produces complete episodes and
repeatable SR/SPL on a fixed small R2R-CE validation subset.

## 3. Actual ActiveVLN Call Path

The useful execution path is:

```text
examples/vlnce/r2r.sh
  -> verl.trainer.main_ppo
  -> verl/workers/agent/parallel_env_vlnce.py
  -> vlnce_server.client.BatchEnvClient
  -> HTTP /environments, /batch/reset, /batch/step, /batch/close
  -> vlnce_server.server.BatchEnvServer
  -> vlnce_server.service.VLNCEService / VLNCEActor
  -> vlnce_server.env.VLNCEEnv / Simulator
  -> habitat.Env
```

The model side already transports PIL images through `multi_modal_data`, uses
an HF processor, and supports batched multimodal vLLM rollout. This boundary is
worth preserving.

## 4. Keep, Adapt, Rewrite, Retire

### 4.1 Keep

Keep these interfaces with narrow changes:

- `vlnce_server/client.py`: HTTP batch client contract.
- `vlnce_server/server.py`: process boundary and batch routes.
- `vlnce_server/service.py`: Ray actor pooling concept.
- `vlnce_server/utils/serial_utils.py`: image serialization boundary.
- `verl/workers/agent/parallel_env_vlnce.py`: multimodal rollout bridge.
- `vlnce_server/cfrp/`: protocol, plan cursor, adapter, checkpoint and branch
  records.
- `data/r2r_4000_train.parquet` and `data/r2r_val_tiny.parquet`: lightweight
  rollout metadata inputs, subject to schema validation.

### 4.2 Adapt

Adapt these modules after the Habitat 0.3 environment wrapper exists:

- `vlnce_server/env_config.py`: add benchmark paths, split, sensor shape,
  primitive action settings, Stage 1 history budgets and plan source.
- `vlnce_server/prompt.py`: replace legacy free-form/multi-action instructions
  with the Stage 1 XML prompt.
- `vlnce_server/utils/parse_utils.py`: route Stage 1 XML through
  `vlnce_server.cfrp.protocol`; keep the old parser only for parity tests.
- `vlnce_server/service.py`: construct the new Habitat 0.3 wrapper behind the
  existing batch API.
- `examples/vlnce/train_vlnce_4gpus.yaml`: create a two-GPU CFRP baseline
  configuration and use one primitive action per model turn.

### 4.3 Rewrite

Rewrite the Habitat-facing implementation rather than patching it in place:

- `vlnce_server/env.py::VLNDataset`
- `vlnce_server/env.py::Simulator`
- the Habitat-specific portions of `vlnce_server/env.py::VLNCEEnv`
- evaluation setup in `eval/vlnce/eval_vlnce.py`
- old YACS task/config loading under
  `vlnce_server/VLN_CE/*/config/default.py`

The new implementation should live beside the legacy path first, for example:

```text
vlnce_server/habitat030/
  config.py
  dataset.py
  environment.py
  metrics.py
  episode_runner.py
```

Do not turn `env.py` into a mixed 0.1/0.3 compatibility file.

### 4.4 Retire From The CFRP Main Path

Do not port the following old IL baseline stack unless a later baseline
experiment specifically requires it:

- `vlnce_server/VLN_CE/vlnce_baselines/common/`
- `vlnce_server/VLN_CE/vlnce_baselines/models/`
- waypoint predictors and old DAgger/DDPPO trainers
- old `habitat_baselines.common.baseline_registry` wrappers
- old observation transformers

These modules trained classical VLN policies. CFRP uses Qwen3-VL through the
veRL multimodal rollout path, so porting them would add migration work without
supporting the selected baseline.

## 5. Habitat 0.1 Coupling That Must Be Removed

The current `vlnce_server/env.py` directly uses private Habitat state:

```text
env._reset_stats()
env._current_episode = ...
env.reconfigure(env._config)
env._task.measurements...
env._elapsed_steps = ...
env._episode_over = ...
measure._metric = ...
```

It also loads old YACS configs through `habitat.config.default.Config` and
assumes numeric action IDs `{stop: 0, forward: 1, left: 2, right: 3}`.

Migration rules:

1. Create an environment from a selected one-episode dataset instead of
   mutating `_current_episode`.
2. Execute named Habitat-Lab task actions through the existing CFRP
   `HabitatActionAdapter`.
3. Execute `STOP` through `habitat.Env.step("stop")`; never emulate task STOP
   solely in the outer wrapper.
4. Read public `env.current_episode`, `env.get_metrics()` and
   `env.sim.get_agent_state()` APIs.
5. Never restore a branch by assigning task measure internals.
6. Restore pose and CFRP-owned policy state, then compute branch-local deltas
   from the saved baseline metrics and suffix trace.
7. Keep privileged goal/expert/pose fields out of the model prompt.

## 6. Habitat 0.3 Wrapper Contract

The first wrapper must expose a small version-independent contract:

```python
class NavigationEnvironment:
    def reset(self, episode_id: str | None = None) -> NavigationObservation: ...
    def step(self, primitive_action: str) -> NavigationStep: ...
    def metrics(self) -> dict[str, float]: ...
    def agent_pose(self) -> NavigationPose: ...
    def capture_checkpoint(self, policy_state) -> CFRPCheckpoint: ...
    def restore_checkpoint(self, checkpoint) -> RestoredCFRPState: ...
    def close(self) -> None: ...
```

`NavigationObservation` must contain only inference-visible fields:

```text
RGB frame
instruction
episode id
allowed primitive actions
```

Training-only metadata is stored separately:

```text
goal positions
expert/reference path
distance to goal
success/SPL/nDTW/path length
agent pose
```

This separation is required to prevent oracle leakage into Stage 1 prompts.

## 7. Qwen3-VL Integration Boundary

The copied install scripts pin `transformers==4.51.3` and
`vllm==0.8.5.post1`, which target the previous model stack. They must not be
reused for Qwen3-VL.

The official Qwen3-VL repository currently specifies:

```text
transformers >= 4.57.0
vllm >= 0.11.0
qwen-vl-utils == 0.0.14 (official deployment example)
```

Source:

- https://github.com/QwenLM/Qwen3-VL

These versions belong in a new model environment and must be compatibility
tested with the selected veRL revision before installation. Do not upgrade the
Habitat environment or the whole repository in place.

The minimum model-side integration point is the existing HF processor and
`multi_modal_data` path in `verl/workers/agent/parallel_env_vlnce.py`. Phase 0
does not add the risk head yet. It only verifies:

```text
current RGB + fixed history + instruction + read-only plan
  -> Qwen3-VL generation
  -> Stage 1 XML parser
  -> one primitive Habitat action
```

## 8. Required External Artifacts

The repository currently contains the small parquet rollout metadata, but not
the complete benchmark assets. The server must later provide, on the data disk:

- R2R-CE/VLN-CE episode JSON for train and validation splits.
- R2R ground-truth path/metric JSON used only for labels and evaluation.
- Matterport3D scene assets referenced by those episodes.
- Qwen3-VL-4B-Instruct model weights.
- optionally the ActiveVLN Qwen2.5-VL-3B SFT checkpoint for B1 parity.

Before any download, record expected paths, source, license/access requirement
and estimated size. Use the configured HF mirror or ModelScope for public model
weights; Matterport3D must follow its own authorized dataset access process.

## 9. Implementation Order After This Audit

### Phase 0.3A: One Real R2R Episode

1. Add typed `vlnce_server/habitat030` records and wrapper interface.
2. Load one R2R-CE episode without private episode mutation.
3. Return instruction, RGB and allowed named actions.
4. Execute `move_forward`, `turn_left`, `turn_right`, `stop`.
5. Return `distance_to_goal`, `success`, SPL and path length when configured.

### Phase 0.3B: Stage 1 Loop Integration

1. Initialize a controller-owned compact plan.
2. Keep fixed visual/action windows.
3. Generate and parse exactly one `progress/subgoal/action` output per turn.
4. Advance the plan cursor only on `progress=advance`.
5. Run one complete episode and persist a lightweight trajectory record.

### Phase 0.3C: Baseline Evaluation

1. Freeze a deterministic small R2R validation subset and seeds.
2. Run B1 when its checkpoint is available.
3. Run B2 Qwen3-VL-4B with the same episode IDs and action settings.
4. Report SR, SPL, navigation error, oracle success, invalid XML/action rate,
   average steps and STOP correctness.
5. Repeat the subset to establish deterministic/repeatable behavior before
   scaling.

## 10. Go/No-Go Criteria

Phase 0 is complete only when all items below pass:

```text
[ ] Habitat 0.3 resets a real R2R-CE episode by ID
[ ] instruction and RGB are correct
[ ] named primitive actions and task-level STOP execute correctly
[ ] no model prompt contains pose, goal distance or expert path
[ ] fixed visual/action history does not grow without bound
[ ] Stage 1 XML valid rate >= 99% on the smoke subset
[ ] complete episodes produce standard metrics
[ ] fixed subset SR/SPL is repeatable across two runs
[ ] Qwen3-VL and Habitat remain in separate environments
```

Checkpoint/restore and branch records are already validated infrastructure,
but they do not substitute for these baseline gates.

## 11. Phase 0.2 Result

Phase 0.2 is complete when this audit is accepted. Its concrete decisions are:

- preserve the ActiveVLN service boundary;
- replace the Habitat implementation behind it;
- do not port the classical VLN baseline stack;
- use R2R-CE before RxR-CE;
- keep Habitat and Qwen/veRL dependency environments separate;
- treat Qwen2.5-VL ActiveVLN as an optional parity reference;
- use Qwen3-VL-4B Stage 1 episodes as the actual CFRP baseline.
