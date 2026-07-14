"""Token-region weights for the CFRP Stage 1 supervised XML response."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence


DEFAULT_ACTION_LOSS_WEIGHT = 5.0
DEFAULT_PROGRESS_LOSS_WEIGHT = 2.0
DEFAULT_SUBGOAL_LOSS_WEIGHT = 0.25
DEFAULT_XML_LOSS_WEIGHT = 1.0


def target_xml_region_weights(
    target_xml: str,
    offsets: Sequence[tuple[int, int]],
    *,
    action_weight: float = DEFAULT_ACTION_LOSS_WEIGHT,
    progress_weight: float = DEFAULT_PROGRESS_LOSS_WEIGHT,
    subgoal_weight: float = DEFAULT_SUBGOAL_LOSS_WEIGHT,
    xml_weight: float = DEFAULT_XML_LOSS_WEIGHT,
) -> list[float]:
    """Return one loss weight per token offset in a Stage 1 XML response.

    XML tags retain ordinary cross-entropy weight.  Only their textual payloads
    are reweighted: action primitives are emphasized, while the free-form
    subgoal remains useful context but cannot dominate the navigation signal.
    """

    _validate_weights(action_weight, progress_weight, subgoal_weight, xml_weight)
    regions = _payload_regions(target_xml, action_weight, progress_weight, subgoal_weight)
    weights: list[float] = []
    for start, end in offsets:
        weight = xml_weight
        for region_start, region_end, region_weight in regions:
            if start < region_end and end > region_start:
                # Payload weights deliberately override the default tag weight.
                # In particular, subgoal text must be able to be downweighted.
                weight = region_weight
        weights.append(weight)
    return weights


def locate_target_token_weights(
    target_xml: str,
    target_token_ids: Sequence[int],
    tokenizer: Callable[..., object],
    *,
    action_weight: float = DEFAULT_ACTION_LOSS_WEIGHT,
    progress_weight: float = DEFAULT_PROGRESS_LOSS_WEIGHT,
    subgoal_weight: float = DEFAULT_SUBGOAL_LOSS_WEIGHT,
    xml_weight: float = DEFAULT_XML_LOSS_WEIGHT,
) -> tuple[int, list[float]]:
    """Locate target XML inside the chat-template suffix and weight its tokens.

    ``target_token_ids`` is the supervised suffix produced by the Qwen chat
    template, which can include an assistant end token after the XML.  The
    function deliberately checks the token subsequence instead of assuming a
    particular Qwen template implementation.  A template change therefore
    fails loudly rather than silently applying weights to the wrong tokens.
    """

    encoded = tokenizer(target_xml, add_special_tokens=False, return_offsets_mapping=True)
    token_ids = list(encoded["input_ids"])
    offsets = [tuple(item) for item in encoded["offset_mapping"]]
    if len(token_ids) != len(offsets):
        raise RuntimeError("tokenizer returned mismatched target ids and offsets")
    start = _find_subsequence(list(target_token_ids), token_ids)
    if start is None:
        raise RuntimeError("Qwen chat template suffix does not contain the terminal target XML")
    return (
        start,
        target_xml_region_weights(
            target_xml,
            offsets,
            action_weight=action_weight,
            progress_weight=progress_weight,
            subgoal_weight=subgoal_weight,
            xml_weight=xml_weight,
        ),
    )


def _payload_regions(target_xml: str, action_weight: float, progress_weight: float, subgoal_weight: float) -> list[tuple[int, int, float]]:
    regions: list[tuple[int, int, float]] = []
    for tag, weight in (
        ("progress", progress_weight),
        ("subgoal", subgoal_weight),
        ("action", action_weight),
    ):
        for match in re.finditer(rf"<{tag}>(.*?)</{tag}>", target_xml, flags=re.DOTALL):
            regions.append((match.start(1), match.end(1), weight))
    return regions


def _find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> int | None:
    if not needle:
        return None
    width = len(needle)
    for start in range(len(haystack) - width + 1):
        if list(haystack[start : start + width]) == list(needle):
            return start
    return None


def _validate_weights(*weights: float) -> None:
    if any(weight <= 0 for weight in weights):
        raise ValueError("all Stage 1 loss weights must be positive")
