"""Render deterministic success/failure video clips from a Phase 0 evaluation run."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="qwen-baseline-* directory containing summary.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples-per-outcome", type=int, default=3)
    parser.add_argument(
        "--all-episodes",
        action="store_true",
        help="Render every evaluated episode instead of a deterministic outcome sample",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--frame-duration-ms", type=int, default=250)
    parser.add_argument("--video-format", choices=("mp4", "gif"), default="mp4")
    parser.add_argument("--select-only", action="store_true", help="Write a reproducible episode selection manifest without rendering")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples_per_outcome < 1 or args.frame_duration_ms < 1:
        raise ValueError("samples-per-outcome and frame-duration-ms must be positive")
    run_dir = Path(args.run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    episodes = summary["repetitions"][0]["episodes"]
    selected = (
        group_all_outcome_episodes(episodes)
        if args.all_episodes
        else select_outcome_episodes(episodes, args.samples_per_outcome, args.seed)
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    rendered: dict[str, list[dict[str, Any]]] = {"success": [], "failure": []}
    for outcome, items in selected.items():
        if args.select_only:
            rendered[outcome] = [{"episode_id": episode["episode_id"]} for episode in items]
            continue
        outcome_dir = output_dir / outcome
        for index, episode in enumerate(items, start=1):
            destination = outcome_dir / f"{index:02d}_episode-{episode['episode_id']}.{args.video_format}"
            frame_paths = episode_frame_paths(episode)
            if not frame_paths:
                continue
            render_video(frame_paths, destination, args.frame_duration_ms, args.video_format)
            rendered[outcome].append(
                {
                    "episode_id": episode["episode_id"],
                    "path": str(destination),
                    "frames": len(frame_paths),
                }
            )
    (output_dir / "video_selection.json").write_text(
        json.dumps(
            {
                "schema": "cfrp.phase0.video_selection.v1",
                "seed": args.seed,
                "all_episodes": args.all_episodes,
                "select_only": args.select_only,
                "rendered": rendered,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"success_videos={len(rendered['success'])}")
    print(f"failure_videos={len(rendered['failure'])}")
    print(f"video_dir={output_dir}")
    print("render_phase0_videos: OK")
    return 0


def select_outcome_episodes(
    episodes: Iterable[dict[str, Any]], samples_per_outcome: int, seed: int
) -> dict[str, list[dict[str, Any]]]:
    outcomes = {"success": [], "failure": []}
    for episode in episodes:
        success = float(episode.get("final_metrics", {}).get("success") or 0.0) >= 1.0
        outcomes["success" if success else "failure"].append(episode)
    rng = random.Random(seed)
    selected = {}
    for outcome, items in outcomes.items():
        candidates = sorted(items, key=lambda item: str(item["episode_id"]))
        selected[outcome] = rng.sample(candidates, min(samples_per_outcome, len(candidates)))
    return selected


def group_all_outcome_episodes(episodes: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Return every episode grouped by task success in a stable order."""
    grouped = {"success": [], "failure": []}
    for episode in episodes:
        success = float(episode.get("final_metrics", {}).get("success") or 0.0) >= 1.0
        grouped["success" if success else "failure"].append(episode)
    for items in grouped.values():
        items.sort(key=lambda item: str(item["episode_id"]))
    return grouped


def episode_frame_paths(episode: dict[str, Any]) -> list[str]:
    paths = []
    for step in episode.get("steps", []):
        history = step.get("history") or {}
        rgb_paths = history.get("rgb_paths") or []
        if rgb_paths:
            latest = str(rgb_paths[-1])
            if not paths or paths[-1] != latest:
                paths.append(latest)
    return paths


def render_video(frame_paths: list[str], destination: Path, duration_ms: int, video_format: str) -> None:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("video rendering requires numpy and Pillow") from exc
    frames = [Image.fromarray(np.load(path)).convert("RGB") for path in frame_paths]
    if not frames:
        raise ValueError("cannot render an empty frame sequence")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if video_format == "gif":
        frames[0].save(destination, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
        return
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError("MP4 rendering requires imageio and imageio-ffmpeg") from exc
    imageio.mimsave(destination, [np.asarray(frame) for frame in frames], fps=1000.0 / duration_ms)


if __name__ == "__main__":
    raise SystemExit(main())
