# CFRP

**CFRP-VLN**: Counterfactual Recovery Planning for Continuous Vision-and-Language Navigation.

This repository is a working codebase for exploring recovery-oriented planning in
continuous VLN. The main idea is to maintain a sparse persistent plan state and
train a VLM agent to decide when to continue its current plan or replan after
recovering from a wrong route.

## Project Goal

CFRP-VLN focuses on navigation states where the agent has drifted away from the
intended route. Instead of blindly following the old plan, the agent should
produce structured XML:

```xml
<plan>...</plan>   <!-- only when initializing or full replanning -->
<tool>continue / replan</tool>
<subgoal>short executable local instruction</subgoal>
<action>one allowed action</action>
```

`STOP` is a primitive action, not a tool. A compact `<plan_update>` may replace
the full `<plan>` during replanning.

The long-term training target is counterfactual branch comparison from the same
critical state:

```text
continue
replan
```

Each branch is rolled out in the environment and ranked by navigation outcome.

## Current Status

This repository currently contains:

- CFRP method notes in `doc/`
- an ActiveVLN-derived engineering base for multi-turn VLN rollout and training
- VLN-CE server/client code under `vlnce_server/`
- verl-based training infrastructure under `verl/`
- migration notes for moving the environment layer from Habitat 0.1.7 to 0.2.4

The CFRP-specific protocol, parser, controller, prompt builder, and mock loop
are implemented and covered by lightweight tests. Habitat integration and branch
rollout remain under development.

## Engineering Base

The initial codebase is copied from ActiveVLN because it already provides useful
infrastructure:

- a separate VLN-CE environment server
- HTTP batch environment interaction
- multi-turn VLM rollout
- verl / GRPO training integration
- simulator state save/restore hooks useful for counterfactual rollout

The copied code is a starting point, not the final CFRP implementation.

## Habitat Version Warning

The current environment layer is still coupled to Habitat / Habitat-Lab 0.1.7.
The intended target is Habitat 0.2.4, so environment files will need substantial
refactoring before serious experiments.

See:

```text
doc/HABITAT_024_MIGRATION.md
```

## Planned Phases

```text
Phase 0: CFRP protocol, XML parser, plan/action validators, trajectory schema
Phase 1: XML / action / plan format SFT
Phase 2: normal VLN action tuning
Phase 3: short recovery SFT
Phase 4: counterfactual branch rollout
Phase 5: preference / RL optimization
Phase 6: evaluation and ablation
```

## Repository Layout

```text
doc/                     CFRP method, action-interface, and migration notes
vlnce_server/            VLN-CE environment server base
verl/                    training and rollout infrastructure
examples/vlnce/          VLN training/evaluation entry examples
eval/vlnce/              evaluation scripts
```

## Data And Artifacts

Large or machine-specific artifacts should not be committed:

- Matterport3D scenes
- VLN-CE datasets
- parquet trajectory files
- checkpoints and model weights
- rollout videos
- experiment logs

The `.gitignore` is configured to keep these out of the repository.

## Attribution

This project uses ActiveVLN as an initial engineering base and keeps the
upstream license and notice files in the repository. CFRP-specific method design
and future modules are developed separately on top of that base.
