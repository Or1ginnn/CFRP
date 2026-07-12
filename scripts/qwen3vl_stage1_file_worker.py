"""Serve Qwen3-VL Stage 1 requests from the Habitat rollout process."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlnce_server.qwen3vl import DEFAULT_QWEN3_VL_MODEL, run_file_worker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exchange-dir", required=True)
    parser.add_argument("--model", default=DEFAULT_QWEN3_VL_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--poll-seconds", type=float, default=0.1)
    parser.add_argument("--max-requests", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    handled = run_file_worker(
        args.exchange_dir,
        model_name_or_path=args.model,
        max_new_tokens=args.max_new_tokens,
        poll_seconds=args.poll_seconds,
        max_requests=args.max_requests,
    )
    print(f"handled_requests={handled}")
    print("qwen3vl_stage1_file_worker: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
