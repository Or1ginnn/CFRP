"""CFRP-VLN protocol and loop helpers."""

from .adapter import CFRPActionAdapterError, HabitatActionAdapter, HabitatActionCommand
from .branching import (
    BranchContext,
    BranchStep,
    BranchTrace,
    BranchTraceRecorder,
    CFRPBranchingError,
    CounterfactualGroup,
    CriticalStateBaseline,
    EpisodeReference,
    MetricSnapshot,
    TrajectoryPrefix,
    make_trajectory_prefix,
)
from .checkpoint import (
    CFRPCheckpoint,
    CFRPCheckpointError,
    RestoredCFRPState,
    capture_cfrp_checkpoint,
    restore_cfrp_checkpoint,
)
from .controller import CFRPController, ControllerStepResult
from .loop import CFRPLoopTurn, run_scripted_cfrp_loop
from .protocol import (
    CFRPOutput,
    CFRPProtocolError,
    PlanPoint,
    PlanState,
    PlanUpdate,
    apply_plan_update,
    parse_cfrp_output,
    validate_output,
)

__all__ = [
    "CFRPController",
    "CFRPActionAdapterError",
    "CFRPCheckpoint",
    "CFRPCheckpointError",
    "CFRPBranchingError",
    "CFRPOutput",
    "CFRPProtocolError",
    "CFRPLoopTurn",
    "BranchContext",
    "BranchStep",
    "BranchTrace",
    "BranchTraceRecorder",
    "ControllerStepResult",
    "CounterfactualGroup",
    "CriticalStateBaseline",
    "EpisodeReference",
    "HabitatActionAdapter",
    "HabitatActionCommand",
    "MetricSnapshot",
    "PlanPoint",
    "PlanState",
    "PlanUpdate",
    "RestoredCFRPState",
    "TrajectoryPrefix",
    "apply_plan_update",
    "capture_cfrp_checkpoint",
    "make_trajectory_prefix",
    "parse_cfrp_output",
    "run_scripted_cfrp_loop",
    "restore_cfrp_checkpoint",
    "validate_output",
]
