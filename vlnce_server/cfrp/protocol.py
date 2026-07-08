"""CFRP XML protocol parsing and validation.

This module is intentionally independent of Habitat. It turns model XML into a
small structured object and checks CFRP control-state invariants before any
environment action is executed.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
import xml.etree.ElementTree as ET


VALID_TOOLS = {"continue", "replan", "stop"}
VALID_PLAN_STATUSES = {"done", "current", "todo", "abandoned"}


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
class CFRPOutput:
    tool: str
    subgoal: str
    action: str
    plan: PlanState | None = None
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

    plan = _parse_plan(plan_nodes[0]) if plan_nodes else None
    tool = _required_text(root, "tool")
    subgoal = _required_text(root, "subgoal")
    action = _required_text(root, "action")

    return CFRPOutput(
        tool=tool,
        subgoal=subgoal,
        action=action,
        plan=plan,
        raw_xml=raw_text,
    )


def validate_output(
    output: CFRPOutput,
    allowed_actions: set[str] | list[str] | tuple[str, ...],
    previous_plan: PlanState | None = None,
) -> None:
    """Validate a parsed CFRP output.

    Args:
        output: Parsed model output.
        allowed_actions: Primitive actions exposed by the current environment.
        previous_plan: Existing controller plan, used to enforce immutable done
            points when replanning.
    """

    allowed_action_set = set(allowed_actions)
    if output.tool not in VALID_TOOLS:
        raise CFRPProtocolError(f"invalid tool: {output.tool}")
    if output.action not in allowed_action_set:
        raise CFRPProtocolError(f"invalid action: {output.action}")
    if not output.subgoal:
        raise CFRPProtocolError("missing subgoal")

    if output.tool == "continue":
        if output.plan is not None:
            if previous_plan is not None:
                raise CFRPProtocolError("continue must not output <plan> after initialization")
            validate_plan(output.plan)
    elif output.tool == "replan":
        if output.plan is None:
            raise CFRPProtocolError("replan must output <plan>")
        validate_plan(output.plan)
        if previous_plan is not None:
            validate_done_points_immutable(previous_plan, output.plan)
    elif output.tool == "stop":
        if output.action != "STOP":
            raise CFRPProtocolError("stop must use action STOP")
        if output.plan is not None:
            raise CFRPProtocolError("stop must not output <plan>")


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


def _required_text(root: ET.Element, tag: str) -> str:
    nodes = root.findall(tag)
    if len(nodes) != 1:
        raise CFRPProtocolError(f"expected exactly one <{tag}> field")
    text = _node_text(nodes[0])
    if not text:
        raise CFRPProtocolError(f"empty <{tag}> field")
    return text


def _node_text(node: ET.Element) -> str:
    return "".join(node.itertext()).strip()
