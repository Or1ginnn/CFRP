# Habitat 0.3.0 Branching Reference

## Scope

This note records the Habitat 0.3.0 APIs relevant to CFRP counterfactual
branching. It was checked against the official `v0.3.0` source tags:

- Habitat-Sim: `dfb388e29e5e1f25da4b576305e85bdc0be140b8`
- Habitat-Lab: `afe4058a7f8aa5ab71a133575cdaa79f0308af6a`

The official Habitat-Sim README snapshot is stored next to this file as
`HABITAT_SIM_030_OFFICIAL_README.md`.

## What Habitat Provides For Navigation

For ordinary static navigation, Habitat-Lab's `HabitatSim` exposes:

```python
state = env.sim.get_agent_state(agent_id=0)
ok = env.sim.set_agent_state(
    state.position,
    state.rotation,
    agent_id=0,
    reset_sensors=False,
)
observations = env.sim.get_observations_at(position, rotation)
metrics = env.get_metrics()
```

`set_agent_state` restores the agent body pose and clears absolute sensor
states so sensors follow the restored body. It returns whether placement
succeeds. `get_observations_at` can render a pose and optionally restore the
previous pose automatically.

Useful built-in sources for CFRP are:

```text
env.current_episode        episode id, scene, goals, dataset metadata
env.sim.get_agent_state()  position and rotation
env.get_metrics()          task metrics such as distance_to_goal, success, SPL
env.sim.geodesic_distance  geodesic navigation distance
env.sim.pathfinder         navigability and collision-aware motion support
```

## What Habitat Does Not Provide For Standard Navigation

The v0.3.0 navigation environment has no public generic API equivalent to:

```python
branch_env = env.clone()
env.restore_full_state(snapshot)
```

In particular, ordinary navigation does not expose a public complete snapshot
for task measures, elapsed steps, controller memory, VLM history, or the VLN
expert route. Gfx replay keyframes are rendering/replay artifacts; they are not
branchable RL environment checkpoints.

Therefore CFRP must retain its own branch context in addition to Habitat's
agent state.

## The RearrangeSim Exception

Habitat-Lab v0.3.0 does provide:

```python
state = rearrange_sim.capture_state()
rearrange_sim.set_state(state)
```

That API records articulated-agent transforms, rigid-object transforms and
velocities, articulated-object joint states, and grasp state. It is designed
for dynamic Rearrange tasks, not classic R2R-CE/RxR-CE navigation. Do not use
it as the primary CFRP checkpoint abstraction unless the project later moves
to interactive or dynamic-object tasks.

## CFRP Recommendation

For static R2R-CE/RxR-CE, use Habitat's pose restore plus a CFRP-owned
`BranchContext`:

```text
Habitat snapshot
- episode id
- agent position and rotation
- metrics observed at the critical state

Shared episode reference
- instruction
- goal and success rule
- expert trajectory
- expert progress/alignment at the critical state

CFRP control snapshot
- current plan and plan statuses
- controller action history
- recent visual/action history
- turn index and replan cooldown

Trajectory prefix
- agent poses and primitive actions before the critical state
- accumulated path length, collision count, and elapsed step count

Branch-local suffix
- action/pose/observation trace after the forced continue or replan decision
- branch-local reward components
```

The two branches share the episode reference and trajectory prefix. They differ
only after the same restored critical state. Branch reward should be computed
from Habitat task signals plus CFRP recovery signals, without trying to mutate
private Habitat measure internals.

## Source Locations

- Habitat-Lab v0.3.0 `HabitatSim.get_agent_state`, `set_agent_state`, and
  `get_observations_at`:
  https://github.com/facebookresearch/habitat-lab/blob/v0.3.0/habitat-lab/habitat/sims/habitat_simulator/habitat_simulator.py
- Habitat-Lab v0.3.0 `RearrangeSim.capture_state` and `set_state`:
  https://github.com/facebookresearch/habitat-lab/blob/v0.3.0/habitat-lab/habitat/tasks/rearrange/rearrange_sim.py
- Habitat-Sim v0.3.0 simulator implementation:
  https://github.com/facebookresearch/habitat-sim/blob/v0.3.0/src_python/habitat_sim/simulator.py
