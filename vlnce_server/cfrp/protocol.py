"""CFRP XML protocol parsing and validation.

This module is intentionally independent of Habitat. It turns model XML into a
small structured object and checks CFRP control-state invariants before any
environment action is executed.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
import xml.etree.ElementTree as ET


VALID_TOOLS = {"continue", "replan"}
VALID_PROGRESS = {"hold", "advance"}
VALID_PROTOCOL_MODES = {"stage1", "stage2"}
VALID_PLAN_STATUSES = {"done", "current", "todo", "abandoned"}
MAX_STAGE1_ACTION_CHUNK = 4


class CFRPProtocolError(ValueError):
    """Raised when CFRP XML cannot be parsed or violates protocol rules."""


@dataclass(frozen=True)
class PlanPoint:
    id: str
    status: str
    text: str


@dataclass(frozen=True)
class PlanState:
    global_goal: str
    points: tuple[PlanPoint, ...]

    def current_points(self) -> tuple[PlanPoint, ...]:
        return tuple(point for point in self.points if point.status == "current")

    def done_points_by_id(self) -> dict[str, PlanPoint]:
        return {point.id: point for point in self.points if point.status == "done"}

    @property
    def current_index(self) -> int:
        current_indices = [
            index for index, point in enumerate(self.points) if point.status == "current"
        ]
        if len(current_indices) != 1:
            raise CFRPProtocolError("plan must contain exactly one current point")
        return current_indices[0]

    def advance_current(self) -> PlanState:
        """Advance the normal execution cursor without rewriting plan content."""

        current_index = self.current_index
        next_index = next(
            (
                index
                for index in range(current_index + 1, len(self.points))
                if self.points[index].status == "todo"
            ),
            None,
        )
        if next_index is None:
            raise CFRPProtocolError("cannot advance plan without a following todo point")

        advanced_points = tuple(
            PlanPoint(
                id=point.id,
                status=(
                    "done"
                    if index == current_index
                    else "current"
                    if index == next_index
                    else point.status
                ),
                text=point.text,
            )
            for index, point in enumerate(self.points)
        )
        advanced = PlanState(global_goal=self.global_goal, points=advanced_points)
        validate_plan(advanced)
        validate_done_points_immutable(self, advanced)
        return advanced

    def to_xml(self) -> str:
        lines = ["<plan>", f"  <global>{escape(self.global_goal)}</global>", "  <local>"]
        for point in self.points:
            lines.append(
                f'    <p id="{escape(point.id, quote=True)}" '
                f'status="{escape(point.status, quote=True)}">{escape(point.text)}</p>'
            )
        lines.extend(["  </local>", "</plan>"])
        return "\n".join(lines)


@dataclass(frozen=True)
class PlanUpdate:
    """Compact replan patch applied by the controller to a plan state."""

    abandon_id: str
    current: str
    future: str


@dataclass(frozen=True)
class CFRPOutput:
    subgoal: str
    action: str
    actions: tuple[str, ...] = tuple()
    tool: str | None = None
    progress: str | None = None
    plan: PlanState | None = None
    plan_update: PlanUpdate | None = None
    raw_xml: str = ""


def parse_cfrp_output(text: str) -> CFRPOutput:
    """Parse model XML output into a :class:`CFRPOutput`.

    The model may emit sibling top-level fields, so the parser wraps the text in
    a synthetic root before XML parsing.
    """

    raw_text = text.strip()
    if not raw_text:
        raise CFRPProtocolError("empty CFRP output")

    try:
        root = ET.fromstring(f"<cfrp_output>{raw_text}</cfrp_output>")
    except ET.ParseError as exc:
        raise CFRPProtocolError(f"invalid XML: {exc}") from exc

    plan_nodes = root.findall("plan")
    if len(plan_nodes) > 1:
        raise CFRPProtocolError("output contains multiple <plan> fields")
    plan_update_nodes = root.findall("plan_update")
    if len(plan_update_nodes) > 1:
        raise CFRPProtocolError("output contains multiple <plan_update> fields")
    if plan_nodes and plan_update_nodes:
        raise CFRPProtocolError("output cannot contain both <plan> and <plan_update>")

    plan = _parse_plan(plan_nodes[0]) if plan_nodes else None
    plan_update = _parse_plan_update(plan_update_nodes[0]) if plan_update_nodes else None
    tool = _optional_text(root, "tool")
    progress = _optional_text(root, "progress")
    subgoal = _required_text(root, "subgoal")
    actions = _parse_actions(root)

    return CFRPOutput(
        tool=tool,
        progress=progress,
        subgoal=subgoal,
        action=actions[0],
        actions=actions,
        plan=plan,
        plan_update=plan_update,
        raw_xml=raw_text,
    )


def validate_output(
    output: CFRPOutput,
    allowed_actions: set[str] | list[str] | tuple[str, ...],
    previous_plan: PlanState | None = None,
    mode: str = "stage2",
) -> None:
    """Validate a parsed CFRP output.

    Args:
        output: Parsed model output.
        allowed_actions: Primitive actions exposed by the current environment.
        previous_plan: Existing controller plan, used to enforce immutable done
            points when replanning.
    """

    if mode not in VALID_PROTOCOL_MODES:
        raise ValueError(f"invalid protocol mode: {mode}")

    allowed_action_set = set(allowed_actions)
    if mode == "stage1":
        _validate_stage1_output(output, previous_plan)
    else:
        _validate_stage2_output(output, previous_plan)

    if output.tool not in VALID_TOOLS:
        if mode == "stage2":
            raise CFRPProtocolError(f"invalid tool: {output.tool}")
    actions = output.actions or (output.action,)
    if mode != "stage1" and len(actions) != 1:
        raise CFRPProtocolError("Stage 2 output must contain exactly one action")
    if mode == "stage1" and len(actions) > MAX_STAGE1_ACTION_CHUNK:
        raise CFRPProtocolError(
            f"Stage 1 action chunk exceeds {MAX_STAGE1_ACTION_CHUNK} primitive actions"
        )
    if "STOP" in actions and actions != ("STOP",):
        raise CFRPProtocolError("STOP must be the only action in a chunk")
    for action in actions:
        if action not in allowed_action_set:
            raise CFRPProtocolError(f"invalid action: {action}")
    if not output.subgoal:
        raise CFRPProtocolError("missing subgoal")

    if mode == "stage1":
        return

    if output.tool == "continue":
        if output.plan_update is not None:
            raise CFRPProtocolError("continue must not output <plan_update>")
        if output.plan is not None:
            if previous_plan is not None:
                raise CFRPProtocolError("continue must not output <plan> after initialization")
            validate_plan(output.plan)
    elif output.tool == "replan":
        if output.plan is None and output.plan_update is None:
            raise CFRPProtocolError("replan must output <plan> or <plan_update>")
        if output.plan is not None:
            validate_plan(output.plan)
        if output.plan is not None and previous_plan is not None:
            validate_done_points_immutable(previous_plan, output.plan)
        if output.plan_update is not None:
            if previous_plan is None:
                raise CFRPProtocolError("replan <plan_update> requires an existing plan")
            validate_plan_update(previous_plan, output.plan_update)


def _validate_stage1_output(output: CFRPOutput, previous_plan: PlanState | None) -> None:
    if output.tool is not None:
        raise CFRPProtocolError("Stage 1 output must not contain <tool>")
    if output.progress not in VALID_PROGRESS:
        raise CFRPProtocolError(f"invalid progress: {output.progress}")
    if output.plan is not None or output.plan_update is not None:
        raise CFRPProtocolError("Stage 1 output must not contain plan updates")
    if previous_plan is None:
        raise CFRPProtocolError("Stage 1 requires a controller-owned current plan")


def _validate_stage2_output(output: CFRPOutput, previous_plan: PlanState | None) -> None:
    if output.progress is not None:
        raise CFRPProtocolError("Stage 2 output must not contain <progress>")


def _parse_actions(root: ET.Element) -> tuple[str, ...]:
    """Accept a legacy primitive or a bounded Stage 1 action chunk."""

    primitive_nodes = root.findall("action")
    chunk_nodes = root.findall("actions")
    if primitive_nodes and chunk_nodes:
        raise CFRPProtocolError("output cannot contain both <action> and <actions>")
    if len(primitive_nodes) > 1:
        raise CFRPProtocolError("output contains multiple top-level <action> fields")
    if primitive_nodes:
        return (_node_text(primitive_nodes[0]),)
    if len(chunk_nodes) != 1:
        raise CFRPProtocolError("output requires exactly one <action> or <actions> field")
    actions = tuple(_node_text(node) for node in chunk_nodes[0].findall("action"))
    if not actions:
        raise CFRPProtocolError("<actions> must contain at least one <action>")
    if any(not action for action in actions):
        raise CFRPProtocolError("<actions> must not contain an empty <action>")
    return actions


def validate_plan(plan: PlanState) -> None:
    if not plan.global_goal:
        raise CFRPProtocolError("plan missing <global>")
    if not plan.points:
        raise CFRPProtocolError("plan missing local points")

    seen_ids: set[str] = set()
    for point in plan.points:
        if not point.id:
            raise CFRPProtocolError("plan point missing id")
        if point.id in seen_ids:
            raise CFRPProtocolError(f"duplicate plan point id: {point.id}")
        seen_ids.add(point.id)
        if point.status not in VALID_PLAN_STATUSES:
            raise CFRPProtocolError(f"invalid plan status for {point.id}: {point.status}")
        if not point.text:
            raise CFRPProtocolError(f"empty plan point text: {point.id}")

    current_points = plan.current_points()
    if len(current_points) != 1:
        raise CFRPProtocolError("plan must contain exactly one current point")


def validate_done_points_immutable(previous_plan: PlanState, new_plan: PlanState) -> None:
    previous_done = previous_plan.done_points_by_id()
    new_by_id = {point.id: point for point in new_plan.points}

    for point_id, old_point in previous_done.items():
        if point_id not in new_by_id:
            raise CFRPProtocolError(f"done point removed during replan: {point_id}")
        new_point = new_by_id[point_id]
        if new_point.status != "done":
            raise CFRPProtocolError(f"done point status changed during replan: {point_id}")
        if new_point.text != old_point.text:
            raise CFRPProtocolError(f"done point text changed during replan: {point_id}")


def validate_plan_update(previous_plan: PlanState, update: PlanUpdate) -> None:
    point_by_id = {point.id: point for point in previous_plan.points}
    target = point_by_id.get(update.abandon_id)
    if target is None:
        raise CFRPProtocolError(f"plan update references unknown point: {update.abandon_id}")
    if target.status != "current":
        raise CFRPProtocolError("plan update must abandon the current point")
    if not update.current or not update.future:
        raise CFRPProtocolError("plan update fields must not be empty")


def apply_plan_update(previous_plan: PlanState, update: PlanUpdate) -> PlanState:
    """Apply a compact recovery patch without changing completed plan points."""

    validate_plan_update(previous_plan, update)
    existing_ids = {point.id for point in previous_plan.points}
    recovery_id = _next_generated_id("r", existing_ids)
    future_id = _next_generated_id("f", existing_ids | {recovery_id})
    updated_points = tuple(
        PlanPoint(
            id=point.id,
            status="abandoned" if point.id == update.abandon_id else point.status,
            text=point.text,
        )
        for point in previous_plan.points
    ) + (
        PlanPoint(id=recovery_id, status="current", text=update.current),
        PlanPoint(id=future_id, status="todo", text=update.future),
    )
    new_plan = PlanState(global_goal=previous_plan.global_goal, points=updated_points)
    validate_plan(new_plan)
    validate_done_points_immutable(previous_plan, new_plan)
    return new_plan


def _parse_plan(plan_node: ET.Element) -> PlanState:
    global_goal = _required_text(plan_node, "global")
    local_node = plan_node.find("local")
    if local_node is None:
        raise CFRPProtocolError("plan missing <local>")

    points: list[PlanPoint] = []
    for point_node in local_node.findall("p"):
        points.append(
            PlanPoint(
                id=(point_node.attrib.get("id") or "").strip(),
                status=(point_node.attrib.get("status") or "").strip(),
                text=_node_text(point_node),
            )
        )

    plan = PlanState(global_goal=global_goal, points=tuple(points))
    validate_plan(plan)
    return plan


def _parse_plan_update(update_node: ET.Element) -> PlanUpdate:
    return PlanUpdate(
        abandon_id=_required_text(update_node, "abandon"),
        current=_required_text(update_node, "current"),
        future=_required_text(update_node, "future"),
    )


def _next_generated_id(prefix: str, existing_ids: set[str]) -> str:
    index = 1
    while f"{prefix}{index}" in existing_ids:
        index += 1
    return f"{prefix}{index}"


def _required_text(root: ET.Element, tag: str) -> str:
    nodes = root.findall(tag)
    if len(nodes) != 1:
        raise CFRPProtocolError(f"expected exactly one <{tag}> field")
    text = _node_text(nodes[0])
    if not text:
        raise CFRPProtocolError(f"empty <{tag}> field")
    return text


def _optional_text(root: ET.Element, tag: str) -> str | None:
    nodes = root.findall(tag)
    if len(nodes) > 1:
        raise CFRPProtocolError(f"expected at most one <{tag}> field")
    if not nodes:
        return None
    text = _node_text(nodes[0])
    if not text:
        raise CFRPProtocolError(f"empty <{tag}> field")
    return text


def _node_text(node: ET.Element) -> str:
    return "".join(node.itertext()).strip()
