"""Export canonical CFRP Stage 1 SFT JSONL to LLaMA-Factory ShareGPT data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vlnce_server.qwen3vl.llamafactory_data import make_llamafactory_stage1_example
from vlnce_server.qwen3vl.sft_manifest import load_stage1_sft_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", default="cfrp_stage1")
    parser.add_argument("--lora-output-dir", required=True)
    parser.add_argument("--run-name", default="cfrp-stage1-lora")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    examples = load_stage1_sft_jsonl(args.input_jsonl)
    data_file = output_dir / "stage1_sharegpt.jsonl"
    with data_file.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(make_llamafactory_stage1_example(example), ensure_ascii=False) + "\n")
    _write_json(output_dir / "dataset_info.json", _dataset_info(args.dataset_name, data_file.name))
    _write_train_config(
        output_dir / "train_lora.yaml",
        output_dir,
        args.dataset_name,
        Path(args.lora_output_dir),
        args.run_name,
    )
    _write_json(
        output_dir / "manifest.json",
        {
            "schema": "cfrp.llamafactory.stage1_export.v1",
            "source_jsonl": str(Path(args.input_jsonl).resolve()),
            "examples": len(examples),
            "dataset_name": args.dataset_name,
        },
    )
    print(f"examples={len(examples)}")
    print(f"dataset_dir={output_dir}")
    print(f"train_config={output_dir / 'train_lora.yaml'}")
    print("convert_stage1_sft_to_llamafactory: OK")
    return 0


def _dataset_info(dataset_name: str, data_file_name: str) -> dict:
    return {
        dataset_name: {
            "file_name": data_file_name,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system", "images": "images"},
        }
    }


def _write_train_config(
    path: Path,
    dataset_dir: Path,
    dataset_name: str,
    lora_output_dir: Path,
    run_name: str,
) -> None:
    # The no-thinking template prevents the SFT target from gaining an empty
    # CoT; CFRP supervises only its normal action XML.
    lines = [
        "### model",
        "model_name_or_path: Qwen/Qwen3-VL-4B-Instruct",
        "trust_remote_code: true",
        "",
        "### method",
        "stage: sft",
        "do_train: true",
        "finetuning_type: lora",
        "lora_rank: 16",
        "lora_alpha: 32",
        "lora_dropout: 0.05",
        "lora_target: q_proj,k_proj,v_proj,o_proj",
        "",
        "### dataset",
        f"dataset_dir: {dataset_dir.resolve()}",
        f"dataset: {dataset_name}",
        "template: qwen3_vl_nothink",
        "cutoff_len: 4096",
        "preprocessing_num_workers: 8",
        "dataloader_num_workers: 4",
        "",
        "### output",
        f"output_dir: {lora_output_dir.resolve()}",
        "logging_steps: 10",
        "save_strategy: epoch",
        "save_total_limit: 3",
        "plot_loss: true",
        "overwrite_output_dir: false",
        "report_to: wandb",
        f"run_name: {run_name}",
        "",
        "### train",
        "per_device_train_batch_size: 1",
        "gradient_accumulation_steps: 8",
        "learning_rate: 1.0e-4",
        "num_train_epochs: 3.0",
        "lr_scheduler_type: cosine",
        "warmup_ratio: 0.1",
        "bf16: true",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
