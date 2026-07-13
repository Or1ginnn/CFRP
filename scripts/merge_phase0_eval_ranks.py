"""Merge per-rank CFRP artifacts into InternNav-compatible result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    records = list(_read_rank_progress(results_dir))
    if len(records) != args.expected_episodes:
        raise RuntimeError(
            "expected {} episode records, found {}".format(args.expected_episodes, len(records))
        )
    records.sort(key=lambda item: (str(item["scene_id"]), int(item["episode_id"])))
    _write_jsonl(results_dir / "progress.json", records)

    trajectories = list(
        _read_jsonl_files(sorted(results_dir.glob("trajectories_rank*.jsonl")))
    )
    if len(trajectories) != args.expected_episodes:
        raise RuntimeError(
            "expected {} trajectories, found {}".format(args.expected_episodes, len(trajectories))
        )
    trajectories.sort(
        key=lambda item: (str(item["scene_id"]), int(item["episode_id"]))
    )
    _write_jsonl(results_dir / "trajectories.jsonl", trajectories)

    aggregate = _aggregate(records)
    (results_dir / "result.json").write_text(
        json.dumps(aggregate) + "\n", encoding="utf-8"
    )
    print(json.dumps(aggregate, sort_keys=True))
    print("merge_phase0_eval_ranks: OK")
    return 0


def _read_rank_progress(results_dir: Path) -> Iterable[Dict[str, Any]]:
    return _read_jsonl_files(sorted(results_dir.glob("progress_rank*.json")))


def _read_jsonl_files(paths: Iterable[Path]) -> Iterable[Dict[str, Any]]:
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        raise ValueError("cannot aggregate an empty evaluation")
    length = len(records)
    return {
        "sucs_all": sum(float(item["success"]) for item in records) / length,
        "spls_all": sum(float(item["spl"]) for item in records) / length,
        "oss_all": sum(float(item["os"]) for item in records) / length,
        "nes_all": sum(float(item["ne"]) for item in records) / length,
        "length": length,
    }


if __name__ == "__main__":
    raise SystemExit(main())
