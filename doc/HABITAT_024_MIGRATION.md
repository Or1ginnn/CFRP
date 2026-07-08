# Habitat 0.2.4 Migration Notes

This project uses ActiveVLN as the initial engineering base, but the copied
upstream code targets Habitat 0.1.7. CFRP-VLN should target the more common
Habitat / Habitat-Lab 0.2.4 stack, so the environment layer must be treated as
legacy until migrated.

## Current Base

- `vlnce_server/` is copied from ActiveVLN.
- The server/training split is useful and should be preserved.
- The HTTP batch environment interface is useful and should be preserved.
- `Simulator.save_current_state()` / `Simulator.load_state()` are especially
  important for CFRP counterfactual branch rollout.

## Known 0.1.7 Coupling

The following areas are likely coupled to Habitat 0.1.7 APIs:

- `vlnce_server/env.py`
  - direct `habitat.Env(...)` construction
  - manual `env._current_episode` mutation
  - `env.reconfigure(...)`
  - private fields such as `_task`, `_elapsed_steps`, `_episode_over`
  - measurement reset/update internals
- `vlnce_server/VLN_CE/habitat_extensions/`
  - task registration
  - sensors
  - measures
  - action definitions
  - config defaults
- `vlnce_server/VLN_CE/vlnce_baselines/config/`
  - old YAML/config-node structure
  - R2R/RxR task config paths
- `vlnce_server/VLN_CE/vlnce_baselines/common/`
  - old baseline helpers and environment wrappers

## Migration Direction

Keep the high-level ActiveVLN architecture:

```text
training process
-> HTTP batch client
-> VLN-CE environment server
-> simulator workers
```

Refactor the Habitat-specific layer behind the same server API:

```text
create_environments_batch
reset_batch
step_batch
close_batch
warmup_batch
```

This lets CFRP protocol, prompt, controller, reward, and branch rollout code
remain mostly independent from the Habitat version.

## CFRP-Specific Requirements

The migrated environment must support:

- reset to a given episode and history prefix
- render current RGB observation as PIL/image payload
- execute one primitive action per model turn for CFRP first version
- return metrics including success, SPL/nDTW when available, path length, and
  distance-to-goal for training only
- save and restore simulator state from a critical state
- preserve state restore correctness across multiple branch rollouts

## First Migration Checkpoints

1. Run one R2R episode reset under Habitat 0.2.4.
2. Execute a short fixed action sequence.
3. Verify RGB observation shape and instruction text.
4. Verify success / distance / path metrics are available.
5. Save state, execute actions, restore state, and verify pose/metrics match.
6. Re-run the same action suffix from the restored state and confirm identical
   or acceptably close metrics.

## Boundary For Phase 0

Phase 0 should not depend on Habitat 0.2.4 yet. It should only build:

- CFRP XML parser and validators
- prompt builders
- compact plan/controller data structures
- trajectory record schema
- tests using hand-written XML examples

Habitat 0.2.4 migration belongs to the environment integration step after the
CFRP protocol is stable.
