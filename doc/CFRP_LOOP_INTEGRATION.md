# CFRP Loop Integration Notes

This note records the Phase 1 integration target. It intentionally avoids
Habitat 0.2.4 migration work.

## Current Scope

Phase 1 only proves the control loop:

```text
instruction + observation
    -> CFRP prompt
    -> model XML output
    -> parse + validate
    -> persistent plan controller
    -> primitive action
```

The first implementation uses scripted model XML outputs and string
observations. It does not import Habitat, torch, transformers, or Qwen.

## Implemented Mock Path

- `vlnce_server/cfrp/loop.py`
  - `run_scripted_cfrp_loop(...)`
  - Builds one prompt per turn.
  - Parses XML with the Phase 0 protocol parser.
  - Updates `CFRPController`.
  - Returns a turn trace containing tool, subgoal, action, prompt, and plan.

- `scripts/run_cfrp_mock_loop.py`
  - Runs a deterministic five-turn mock trajectory.
  - Covers initialization, continue, recovery replan, and the `STOP` primitive
    action under the `continue` tool.

## ActiveVLN Service Path

The most natural first integration point is the service-style environment path:

- `vlnce_server/env.py`
  - `VLNCEEnv.reset(...)` returns initial rendered observations.
  - `VLNCEEnv.step(response: str)` currently receives a raw model response.
  - It parses that response through `self.parse_func(...)`.
  - The parsed action strings are converted by `parse_action(...)`.
  - Each valid action is executed by `_execute_action(...)`.

- `vlnce_server/service.py`
  - `VLNCEActor.step(action)` forwards the raw action/response to
    `VLNCEEnv.step(...)`.
  - `VLNCEService.step_batch(ids2actions)` batches those calls.

For CFRP, this path should eventually become:

```text
model XML response
    -> parse_cfrp_output(...)
    -> CFRPController.step(...)
    -> primitive action string
    -> existing VLNCEEnv.step(...) action execution path
```

The key change is that `VLNCEEnv.step(...)` should not treat the full XML as a
comma-separated action list. Instead, CFRP should extract exactly one validated
`<action>` before the environment parser sees it.

## VLN-CE Baseline Path

The legacy baseline evaluation/inference loop is in:

- `vlnce_server/VLN_CE/vlnce_baselines/common/base_il_trainer.py`

Important call sites:

- Evaluation:
  - `self.policy.act(...)`
  - `envs.step([a[0].item() for a in actions])`

- Inference:
  - `self.policy.act(...)`
  - `envs.step([a[0].item() for a in actions])`

This path is lower priority for CFRP Phase 1 because it is tensor-policy based.
It is still useful as a reference for where primitive actions enter Habitat.

## Next Integration Step

After the mock loop is stable, add an adapter that accepts a CFRP XML response
and returns the primitive action string expected by the existing service path.

The adapter should stay independent of Habitat. It should own:

- one `CFRPController` per environment id or episode id
- allowed action names for the current action space
- XML parse/validate errors
- compact trace fields for logging and training data

Habitat 0.2.4 migration remains a separate phase.
