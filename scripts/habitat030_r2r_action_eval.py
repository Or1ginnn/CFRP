"""Evaluate the Phase 0 action-only policy in a one-action closed loop."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlnce_server.habitat030.action_runner import ActionOnlyEpisodeRunner
from vlnce_server.habitat030.environment import Habitat030NavigationEnvironment
from vlnce_server.habitat030.r2r_dataset import load_r2r_dataset, make_habitat_dataset
from vlnce_server.habitat030.r2r_environment import (
    R2R_MAX_EPISODE_STEPS,
    R2R_SUCCESS_DISTANCE,
    r2r_simulator_overrides,
)
from vlnce_server.qwen3vl import VLLMActionClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--scenes-dir", required=True)
    parser.add_argument("--split", default="val_unseen")
    parser.add_argument("--episode-count", type=int, default=200)
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=R2R_MAX_EPISODE_STEPS)
    parser.add_argument("--success-distance", type=float, default=R2R_SUCCESS_DISTANCE)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="cfrp-action")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.episode_count < 1 or args.episode_offset < 0 or args.max_steps < 1:
        raise ValueError("episode-count/max-steps must be positive and offset non-negative")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    records = load_r2r_dataset(
        args.dataset_root,
        args.split,
        args.scenes_dir,
        limit=args.episode_offset + args.episode_count,
    )[args.episode_offset :]
    if len(records) != args.episode_count:
        raise ValueError("requested evaluation range exceeds the selected R2R split")

    env = _make_env(args, records)
    wrapper = Habitat030NavigationEnvironment(env)
    client = VLLMActionClient(args.base_url, args.model, seed=args.seed)
    trajectories_path = output_dir / "trajectories.jsonl"
    results = []
    try:
        with trajectories_path.open("w", encoding="utf-8") as destination:
            for expected in records:
                runner = ActionOnlyEpisodeRunner(wrapper)
                observation = runner.reset()
                if observation.episode_id != expected.episode_id:
                    raise RuntimeError("Habitat episode order differs from the selected evaluation records")
                model_errors = 0
                raw_outputs = []
                for _ in range(args.max_steps):
                    raw_xml = client.generate_xml(runner.model_request())
                    raw_outputs.append(raw_xml)
                    try:
                        step = runner.step(raw_xml)
                    except ValueError:
                        model_errors += 1
                        step = runner.step("<action>STOP</action>")
                    if step.episode_over:
                        break
                metrics = wrapper.metrics()
                effective_success = 0.0 if model_errors else float(metrics.success or 0.0)
                effective_spl = 0.0 if model_errors else float(metrics.spl or 0.0)
                result = {
                    "episode_id": observation.episode_id,
                    "steps": len(runner.trajectory),
                    "model_errors": model_errors,
                    "success": effective_success,
                    "spl": effective_spl,
                    "raw_habitat_success": metrics.success,
                    "raw_habitat_spl": metrics.spl,
                    "distance_to_goal": metrics.distance_to_goal,
                    "path_length": metrics.path_length,
                    "raw_outputs": raw_outputs,
                    "executed_actions": [item.action for item in runner.trajectory],
                }
                destination.write(json.dumps(result, ensure_ascii=False) + "\n")
                results.append(result)
    finally:
        wrapper.close()

    summary = _summary(results)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    print("habitat030_r2r_action_eval: OK")
    return 0


def _make_env(args, records):
    import habitat
    from habitat.config.default import get_config

    overrides = [
        f"habitat.dataset.data_path={Path(args.dataset_root) / args.split / (args.split + '.json.gz')}",
        f"habitat.dataset.scenes_dir={args.scenes_dir}",
        f"habitat.dataset.split={args.split}",
        "habitat.environment.iterator_options.shuffle=False",
        f"habitat.environment.max_episode_steps={args.max_steps}",
        f"habitat.task.measurements.success.success_distance={args.success_distance}",
        f"habitat.seed={args.seed}",
    ]
    overrides.extend(r2r_simulator_overrides())
    config = get_config(args.config, overrides=overrides)
    return habitat.Env(config=config, dataset=make_habitat_dataset(records))


def _summary(results):
    count = len(results)
    return {
        "episodes": count,
        "sr": sum(float(item["success"] or 0.0) for item in results) / count,
        "spl": sum(float(item["spl"] or 0.0) for item in results) / count,
        "navigation_error": sum(float(item["distance_to_goal"] or 0.0) for item in results) / count,
        "invalid_output_rate": sum(item["model_errors"] > 0 for item in results) / count,
        "average_steps": sum(item["steps"] for item in results) / count,
    }


if __name__ == "__main__":
    raise SystemExit(main())
