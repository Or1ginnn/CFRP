"""R2R-CE JSON loader utilities for Habitat 0.3 smoke tests."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Optional, Sequence, Tuple


@dataclass(frozen=True)
class R2REpisodeRecord:
    episode_id: str
    scene_id: str
    scene_path: str
    start_position: Tuple[float, ...]
    start_rotation: Tuple[float, ...]
    goals: Tuple[Tuple[float, ...], ...]
    goal_radii: Tuple[Optional[float], ...]
    instruction_text: str
    reference_path: Tuple[Tuple[float, ...], ...]


class R2RSceneNotFoundError(FileNotFoundError):
    pass


class R2REpisodeNotFoundError(LookupError):
    pass


def load_r2r_episode(
    dataset_root: str,
    scenes_dir: str,
    split: str = "val_seen",
    episode_id: Optional[str] = None,
) -> R2REpisodeRecord:
    records = load_r2r_dataset(
        dataset_root=dataset_root,
        split=split,
        episode_ids=(episode_id,) if episode_id is not None else None,
        limit=1,
        scenes_dir=scenes_dir,
    )
    if not records:
        raise R2REpisodeNotFoundError(f"R2R episode not found: split={split} episode_id={episode_id}")
    return records[0]


def load_r2r_dataset(
    dataset_root: str,
    split: str,
    scenes_dir: str,
    episode_ids: Optional[Iterable[str]] = None,
    limit: Optional[int] = None,
) -> Tuple[R2REpisodeRecord, ...]:
    path = Path(dataset_root) / split / f"{split}.json.gz"
    if not path.exists():
        raise FileNotFoundError(f"R2R split JSON not found: {path}")
    selected_ids = None if episode_ids is None else {str(item) for item in episode_ids}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    episodes = payload.get("episodes", payload if isinstance(payload, list) else [])
    records = []
    for raw_episode in episodes:
        current_id = str(raw_episode.get("episode_id"))
        if selected_ids is not None and current_id not in selected_ids:
            continue
        records.append(_episode_from_json(raw_episode, scenes_dir=scenes_dir))
        if limit is not None and len(records) >= limit:
            break
    if selected_ids is not None and not records:
        raise R2REpisodeNotFoundError(f"No matching R2R episodes in {path}: {sorted(selected_ids)}")
    return tuple(records)


def make_habitat_episode(record: R2REpisodeRecord) -> Any:
    from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal

    goals = [
        NavigationGoal(position=list(position), radius=radius)
        for position, radius in zip(record.goals, record.goal_radii)
    ]
    episode = NavigationEpisode(
        episode_id=record.episode_id,
        scene_id=record.scene_path,
        start_position=list(record.start_position),
        start_rotation=list(record.start_rotation),
        goals=goals,
    )
    episode.instruction = SimpleNamespace(instruction_text=record.instruction_text)
    episode.reference_path = tuple(record.reference_path)
    return episode


def make_habitat_dataset(records: Sequence[R2REpisodeRecord]) -> Any:
    from habitat.core.dataset import Dataset

    class R2RHabitatDataset(Dataset):
        def __init__(self, episodes: Sequence[Any]) -> None:
            self.episodes = list(episodes)

    return R2RHabitatDataset([make_habitat_episode(record) for record in records])


def _episode_from_json(raw_episode: dict[str, Any], scenes_dir: str) -> R2REpisodeRecord:
    scene_path = _resolve_mp3d_scene(raw_episode.get("scene_id"), scenes_dir=scenes_dir)
    goals, radii = _goals(raw_episode.get("goals") or [])
    instruction = raw_episode.get("instruction") or {}
    if isinstance(instruction, dict):
        instruction_text = str(instruction.get("instruction_text") or instruction.get("text") or "")
    else:
        instruction_text = str(instruction)
    return R2REpisodeRecord(
        episode_id=str(raw_episode.get("episode_id")),
        scene_id=str(raw_episode.get("scene_id")),
        scene_path=str(scene_path),
        start_position=_float_tuple(raw_episode.get("start_position") or []),
        start_rotation=_float_tuple(raw_episode.get("start_rotation") or []),
        goals=goals,
        goal_radii=radii,
        instruction_text=instruction_text,
        reference_path=tuple(_float_tuple(point) for point in raw_episode.get("reference_path") or []),
    )


def _resolve_mp3d_scene(scene_id: Any, scenes_dir: str) -> Path:
    if not scene_id:
        raise ValueError("R2R episode is missing scene_id")
    raw_scene = str(scene_id)
    scene_path = Path(raw_scene)
    if scene_path.is_absolute():
        resolved = scene_path
    else:
        parts = scene_path.parts
        if parts and parts[0] == "data":
            parts = parts[2:] if len(parts) > 1 and parts[1] == "scene_datasets" else parts[1:]
        if parts and parts[0] == "scene_datasets":
            parts = parts[1:]
        if not parts or parts[0] != "mp3d":
            parts = ("mp3d",) + tuple(parts)
        if len(parts) == 2:
            house_id = parts[1]
            parts = ("mp3d", house_id, f"{house_id}.glb")
        resolved = Path(scenes_dir).joinpath(*parts)
    if resolved.name != f"{resolved.parent.name}.glb":
        house_id = resolved.parent.name if resolved.suffix else resolved.name
        resolved = Path(scenes_dir) / "mp3d" / house_id / f"{house_id}.glb"
    if not resolved.exists():
        raise R2RSceneNotFoundError(f"MP3D scene not found for scene_id={raw_scene}: {resolved}")
    return resolved


def _goals(raw_goals: Sequence[dict[str, Any]]) -> Tuple[Tuple[Tuple[float, ...], ...], Tuple[Optional[float], ...]]:
    positions = []
    radii = []
    for goal in raw_goals:
        positions.append(_float_tuple(goal.get("position") or []))
        radius = goal.get("radius")
        radii.append(None if radius is None else float(radius))
    return tuple(positions), tuple(radii)


def _float_tuple(values: Iterable[Any]) -> Tuple[float, ...]:
    return tuple(float(value) for value in values)
