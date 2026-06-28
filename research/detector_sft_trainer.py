from __future__ import annotations

import json
import hashlib
import argparse
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol


CANONICAL_SFT_JSONL = Path("data/detector_sft/synthetic_detector_dataset/detector_sft_chat.jsonl")
CANONICAL_ARTIFACT_ROOT = Path("artifacts/detector_sft/qwen3_5_4b_unsloth_lora_v1")
ADAPTER_DIRNAME = "adapter"
BASE_MODEL_ID = "unsloth/Qwen3.5-4B"
ADAPTER_ID = "vinacount-qwen-lora-detector-v1"
DISPLAY_LABEL = "Qwen3.5-4B-Detector-LoRA"
MAX_SEQ_LENGTH = 8192
MAX_EPOCHS = 1
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_SAVE_TOTAL_LIMIT = 2
DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE = 4
DEFAULT_PER_DEVICE_EVAL_BATCH_SIZE = 1
TRAINING_RUNTIME = "unsloth_bf16_lora"
SYNTHETIC_SOURCE_TYPE = "synthetic_injected_filtered"
SUPPORTED_SPLITS = {"train", "validation", "test"}
MESSAGE_ROLES = ("system", "user", "assistant")


class ChatTemplateTokenizer(Protocol):
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs: Any,
    ) -> Any:
        ...


@dataclass(frozen=True)
class Wave4DetectorSftRow:
    line_number: int
    example_id: str
    split: str
    source_type: str
    messages: list[dict[str, str]]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Wave4DetectorSftDataset:
    source_jsonl: Path
    train_rows: list[Wave4DetectorSftRow]
    validation_rows: list[Wave4DetectorSftRow]
    test_rows: list[Wave4DetectorSftRow]
    split_counts: dict[str, int]


@dataclass(frozen=True)
class Wave4DetectorSftTokenPreflight:
    max_seq_length: int
    rows_checked: int
    rows_checked_by_split: dict[str, int]
    max_token_count: int
    row_token_counts: list[dict[str, Any]]


@dataclass(frozen=True)
class Wave4DetectorSftTrainingResult:
    status: str
    artifact_root: Path
    adapter_dir: Path
    training_config_path: Path
    data_manifest_path: Path
    metrics_path: Path
    validation_selection_summary_path: Path
    eval_summary_path: Path
    errors: list[str]


def load_detector_sft_dataset(
    sft_jsonl: Path | str = CANONICAL_SFT_JSONL,
    *,
    allow_noncanonical_sft_jsonl: bool = False,
) -> Wave4DetectorSftDataset:
    source_path = Path(sft_jsonl)
    _enforce_canonical_sft_source(source_path, allow_noncanonical_sft_jsonl=allow_noncanonical_sft_jsonl)
    rows = _read_sft_rows(source_path)
    train_rows = [row for row in rows if row.split == "train"]
    validation_rows = [row for row in rows if row.split == "validation"]
    test_rows = [row for row in rows if row.split == "test"]
    split_counts = dict(sorted(Counter(row.split for row in rows).items()))
    if not train_rows:
        raise ValueError("detector SFT export must contain at least one metadata.split == 'train' row")
    if not validation_rows:
        raise ValueError("detector SFT export must contain at least one metadata.split == 'validation' row")
    return Wave4DetectorSftDataset(
        source_jsonl=source_path,
        train_rows=train_rows,
        validation_rows=validation_rows,
        test_rows=test_rows,
        split_counts=split_counts,
    )


def preflight_detector_sft_token_lengths(
    dataset: Wave4DetectorSftDataset,
    *,
    tokenizer: ChatTemplateTokenizer,
    max_seq_length: int = MAX_SEQ_LENGTH,
) -> Wave4DetectorSftTokenPreflight:
    row_token_counts: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    rows_to_check = [*dataset.train_rows, *dataset.validation_rows]
    for row in rows_to_check:
        tokenized = _apply_chat_template_for_qwen35(
            tokenizer,
            row.messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        token_count = _token_count(tokenized)
        record = {
            "example_id": row.example_id,
            "split": row.split,
            "line_number": row.line_number,
            "token_count": token_count,
        }
        row_token_counts.append(record)
        if token_count > max_seq_length:
            violations.append(record)
    if violations:
        details = "; ".join(
            f"{violation['example_id']} split={violation['split']} line={violation['line_number']} "
            f"has {violation['token_count']} tokens over max_seq_length {max_seq_length}"
            for violation in violations
        )
        raise ValueError(
            "Wave 4 #120 tokenizer preflight failed; rows must not be silently truncated or dropped: "
            f"{details}"
        )
    return Wave4DetectorSftTokenPreflight(
        max_seq_length=max_seq_length,
        rows_checked=len(row_token_counts),
        rows_checked_by_split=dict(sorted(Counter(row["split"] for row in row_token_counts).items())),
        max_token_count=max((row["token_count"] for row in row_token_counts), default=0),
        row_token_counts=row_token_counts,
    )


def run_detector_sft_training(
    *,
    sft_jsonl: Path | str = CANONICAL_SFT_JSONL,
    artifact_root: Path | str = CANONICAL_ARTIFACT_ROOT,
    mode: Literal["dry_run", "train"],
    tokenizer: ChatTemplateTokenizer | None = None,
    allow_noncanonical_sft_jsonl: bool = False,
    max_epochs: int = MAX_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    eval_steps: int | None = None,
    save_total_limit: int = DEFAULT_SAVE_TOTAL_LIMIT,
    weakly_supported_train_multiplier: int = 1,
    per_device_train_batch_size: int = DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size: int = DEFAULT_PER_DEVICE_EVAL_BATCH_SIZE,
) -> Wave4DetectorSftTrainingResult:
    if mode not in {"dry_run", "train"}:
        raise ValueError("mode must be dry_run or train")
    _validate_training_hparams(
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        eval_steps=eval_steps,
        save_total_limit=save_total_limit,
        weakly_supported_train_multiplier=weakly_supported_train_multiplier,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
    )
    output_root = Path(artifact_root)
    adapter_dir = output_root / ADAPTER_DIRNAME
    training_config_path = output_root / "training_config.json"
    data_manifest_path = output_root / "data_manifest.json"
    metrics_path = output_root / "metrics.json"
    validation_selection_summary_path = output_root / "validation_selection_summary.json"
    eval_summary_path = output_root / "eval_summary.json"

    dataset = load_detector_sft_dataset(
        sft_jsonl,
        allow_noncanonical_sft_jsonl=allow_noncanonical_sft_jsonl,
    )
    live_assets: dict[str, Any] | None = None
    if tokenizer is None:
        if mode == "dry_run":
            tokenizer = ApproximateDryRunTokenizer()
        else:
            live_assets = _load_live_model_and_tokenizer()
            tokenizer = live_assets["tokenizer"]
    preflight = preflight_detector_sft_token_lengths(
        dataset,
        tokenizer=tokenizer,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    training_config = _training_config(
        artifact_root=output_root,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        eval_steps=eval_steps,
        save_total_limit=save_total_limit,
        weakly_supported_train_multiplier=weakly_supported_train_multiplier,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
    )
    training_rows = _training_rows_for_sft(
        dataset,
        weakly_supported_train_multiplier=weakly_supported_train_multiplier,
    )
    data_manifest = _data_manifest(
        dataset,
        preflight,
        training_rows=training_rows,
        weakly_supported_train_multiplier=weakly_supported_train_multiplier,
    )
    if mode == "dry_run":
        (adapter_dir / "DRY_RUN_ADAPTER.txt").write_text(
            "Dry-run marker only. Run --mode train with Unsloth to create the LoRA adapter.\n",
            encoding="utf-8",
        )
        metrics = _dry_run_metrics(
            dataset,
            preflight,
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            eval_steps=eval_steps,
            effective_train_rows=len(training_rows),
            weakly_supported_train_multiplier=weakly_supported_train_multiplier,
        )
        validation_selection_summary = _validation_selection_summary(mode="dry_run", max_epochs=max_epochs)
        eval_summary = _eval_summary(dataset)
    else:
        try:
            live_summary = _run_live_unsloth_training(
                dataset,
                output_root=output_root,
                adapter_dir=adapter_dir,
                live_assets=live_assets,
                max_epochs=max_epochs,
                learning_rate=learning_rate,
                eval_steps=eval_steps,
                save_total_limit=save_total_limit,
                training_rows=training_rows,
                per_device_train_batch_size=per_device_train_batch_size,
                per_device_eval_batch_size=per_device_eval_batch_size,
            )
        except Exception as error:
            metrics = _failed_train_metrics(
                dataset,
                preflight,
                error,
                max_epochs=max_epochs,
                learning_rate=learning_rate,
                eval_steps=eval_steps,
                effective_train_rows=len(training_rows),
                weakly_supported_train_multiplier=weakly_supported_train_multiplier,
            )
            validation_selection_summary = _validation_selection_summary(
                mode="train",
                max_epochs=max_epochs,
                error=str(error),
            )
            eval_summary = _eval_summary(dataset)
            _write_json(training_config_path, training_config)
            _write_json(data_manifest_path, data_manifest)
            _write_json(metrics_path, metrics)
            _write_json(validation_selection_summary_path, validation_selection_summary)
            _write_json(eval_summary_path, eval_summary)
            return Wave4DetectorSftTrainingResult(
                status="failed",
                artifact_root=output_root,
                adapter_dir=adapter_dir,
                training_config_path=training_config_path,
                data_manifest_path=data_manifest_path,
                metrics_path=metrics_path,
                validation_selection_summary_path=validation_selection_summary_path,
                eval_summary_path=eval_summary_path,
                errors=[str(error)],
            )
        metrics = _train_metrics(
            dataset,
            preflight,
            live_summary,
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            eval_steps=eval_steps,
            effective_train_rows=len(training_rows),
            weakly_supported_train_multiplier=weakly_supported_train_multiplier,
        )
        validation_selection_summary = _validation_selection_summary(
            mode="train",
            max_epochs=max_epochs,
            live_summary=live_summary,
        )
        eval_summary = _eval_summary(dataset)

    _write_json(training_config_path, training_config)
    _write_json(data_manifest_path, data_manifest)
    _write_json(metrics_path, metrics)
    _write_json(validation_selection_summary_path, validation_selection_summary)
    _write_json(eval_summary_path, eval_summary)
    return Wave4DetectorSftTrainingResult(
        status="passed",
        artifact_root=output_root,
        adapter_dir=adapter_dir,
        training_config_path=training_config_path,
        data_manifest_path=data_manifest_path,
        metrics_path=metrics_path,
        validation_selection_summary_path=validation_selection_summary_path,
        eval_summary_path=eval_summary_path,
        errors=[],
    )


def _validate_training_hparams(
    *,
    max_epochs: int,
    learning_rate: float,
    eval_steps: int | None,
    save_total_limit: int,
    weakly_supported_train_multiplier: int,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
) -> None:
    if max_epochs <= 0:
        raise ValueError("max_epochs must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if eval_steps is not None and eval_steps <= 0:
        raise ValueError("eval_steps must be positive when provided")
    if save_total_limit <= 0:
        raise ValueError("save_total_limit must be positive")
    if weakly_supported_train_multiplier <= 0:
        raise ValueError("weakly_supported_train_multiplier must be positive")
    if per_device_train_batch_size <= 0:
        raise ValueError("per_device_train_batch_size must be positive")
    if per_device_eval_batch_size <= 0:
        raise ValueError("per_device_eval_batch_size must be positive")


def _enforce_canonical_sft_source(source_path: Path, *, allow_noncanonical_sft_jsonl: bool) -> None:
    if allow_noncanonical_sft_jsonl:
        return
    if source_path != CANONICAL_SFT_JSONL:
        raise ValueError(
            "Wave 4 #120 must train from the official detector_sft_chat.jsonl export only: "
            f"{CANONICAL_SFT_JSONL}"
        )


def _read_sft_rows(source_path: Path) -> list[Wave4DetectorSftRow]:
    rows: list[Wave4DetectorSftRow] = []
    for line_number, line in enumerate(source_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{source_path}:{line_number}: invalid JSONL row: {error.msg}") from error
        rows.append(_parse_sft_row(payload, line_number=line_number, source_path=source_path))
    if not rows:
        raise ValueError(f"{source_path} contains no SFT chat rows")
    return rows


def _parse_sft_row(payload: dict[str, Any], *, line_number: int, source_path: Path) -> Wave4DetectorSftRow:
    if not isinstance(payload, dict):
        raise ValueError(f"{source_path}:{line_number}: SFT row must be a JSON object")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{source_path}:{line_number}: SFT row metadata must be an object")
    split = metadata.get("split")
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"{source_path}:{line_number}: unsupported metadata.split for #120 SFT training: {split!r}")
    source_type = metadata.get("source_type")
    if source_type != SYNTHETIC_SOURCE_TYPE:
        raise ValueError(
            f"{source_path}:{line_number}: #120 SFT training may only consume {SYNTHETIC_SOURCE_TYPE} rows"
        )
    example_id = metadata.get("example_id")
    if not isinstance(example_id, str) or not example_id:
        raise ValueError(f"{source_path}:{line_number}: metadata.example_id must be a non-empty string")
    messages = _parse_messages(payload.get("messages"), source_path=source_path, line_number=line_number)
    return Wave4DetectorSftRow(
        line_number=line_number,
        example_id=example_id,
        split=split,
        source_type=source_type,
        messages=messages,
        metadata=dict(metadata),
    )


def _parse_messages(raw_messages: Any, *, source_path: Path, line_number: int) -> list[dict[str, str]]:
    if not isinstance(raw_messages, list) or len(raw_messages) != len(MESSAGE_ROLES):
        raise ValueError(f"{source_path}:{line_number}: messages must contain system, user, assistant entries")
    messages: list[dict[str, str]] = []
    for expected_role, message in zip(MESSAGE_ROLES, raw_messages, strict=True):
        if not isinstance(message, dict):
            raise ValueError(f"{source_path}:{line_number}: message must be an object")
        role = message.get("role")
        content = message.get("content")
        if role != expected_role:
            raise ValueError(
                f"{source_path}:{line_number}: expected message role {expected_role!r}, found {role!r}"
            )
        if not isinstance(content, str) or not content:
            raise ValueError(f"{source_path}:{line_number}: {expected_role} message content must be a non-empty string")
        messages.append({"role": role, "content": content})
    return messages


class ApproximateDryRunTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        if not tokenize:
            raise ValueError("ApproximateDryRunTokenizer only supports tokenize=True")
        text = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
        return list(range(max(1, (len(text) + 3) // 4)))


def _training_config(
    *,
    artifact_root: Path,
    max_epochs: int,
    learning_rate: float,
    eval_steps: int | None,
    save_total_limit: int,
    weakly_supported_train_multiplier: int,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
) -> dict[str, Any]:
    eval_strategy = "steps" if eval_steps is not None else "epoch"
    trainer_config: dict[str, Any] = {
        "num_train_epochs": max_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "gradient_accumulation_steps": 2,
        "learning_rate": learning_rate,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "weight_decay": 0.01,
        "bf16": True,
        "save_strategy": eval_strategy,
        "eval_strategy": eval_strategy,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "remove_unused_columns": False,
        "save_total_limit": save_total_limit,
    }
    if eval_steps is not None:
        trainer_config["eval_steps"] = eval_steps
        trainer_config["save_steps"] = eval_steps
    return {
        "base_model": BASE_MODEL_ID,
        "training_runtime": TRAINING_RUNTIME,
        "precision": "bf16",
        "adapter_id": ADAPTER_ID,
        "display_label": DISPLAY_LABEL,
        "artifact_root": str(artifact_root),
        "adapter_output": ADAPTER_DIRNAME,
        "max_seq_length": MAX_SEQ_LENGTH,
        "max_epochs": max_epochs,
        "selection_metric": "eval_loss",
        "selection_split": "validation",
        "train_split": "train",
        "test_split_role": "post_training_synthetic_regression_only",
        "train_sampling_policy": {
            "weakly_supported_train_multiplier": weakly_supported_train_multiplier,
            "validation_and_test_rows_duplicated": False,
            "purpose": "calibrate partial-support boundary without touching validation or test labels",
        },
        "use_official_chat_messages_exactly": True,
        "implements_sft_vllm_runtime": False,
        "chat_template_adapter": {
            "purpose": "Qwen3.5 processor compatibility only",
            "stored_sft_messages_modified": False,
            "plain_string_content_wrapped_as_text_blocks_for_processor": True,
        },
        "model_load_config": {
            "load_in_4bit": False,
            "load_in_16bit": True,
            "full_finetuning": False,
        },
        "excluded_paths": {
            "sweep": False,
            "second_base_model": False,
            "managed_provider": False,
            "deepseek_detector_finetune": False,
            "gemma_fallback": False,
        },
        "lora_config": {
            "r": 16,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "bias": "none",
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        },
        "trainer_config": trainer_config,
        "source_of_truth": [
            "CONTEXT.md:452-459",
            "CONTEXT.md:733",
            "data/detector_sft/synthetic_detector_dataset/README.md:11-33",
            "data/detector_sft/synthetic_detector_dataset/detector_sft_chat.jsonl",
            "docs/03_data/TrainingDataFormat.md:1092-1142",
            "docs/prd_waves/WAVE_04_DETECTOR_SFT_PRD.md:104-112",
            "docs/04_evaluation/API_LLM_Detector_Baseline.md:27-39",
            "docs/04_evaluation/API_LLM_Detector_Baseline.md:205-218",
        ],
    }


def _data_manifest(
    dataset: Wave4DetectorSftDataset,
    preflight: Wave4DetectorSftTokenPreflight,
    *,
    training_rows: list[Wave4DetectorSftRow],
    weakly_supported_train_multiplier: int,
) -> dict[str, Any]:
    original_train_support_counts = _support_level_counts(dataset.train_rows)
    effective_train_support_counts = _support_level_counts(training_rows)
    return {
        "source_jsonl": str(dataset.source_jsonl),
        "source_jsonl_sha256": _sha256(dataset.source_jsonl),
        "split_counts": dataset.split_counts,
        "trainer_splits": {
            "train": len(dataset.train_rows),
            "validation": len(dataset.validation_rows),
        },
        "effective_trainer_splits": {
            "train": len(training_rows),
            "validation": len(dataset.validation_rows),
        },
        "train_support_level_counts": original_train_support_counts,
        "effective_train_support_level_counts": effective_train_support_counts,
        "train_sampling_policy": {
            "weakly_supported_train_multiplier": weakly_supported_train_multiplier,
            "duplicated_support_level": "weakly_supported"
            if weakly_supported_train_multiplier > 1
            else None,
            "validation_and_test_rows_duplicated": False,
        },
        "reserved_splits": {"test": len(dataset.test_rows)},
        "tokenizer_preflight": {
            "max_seq_length": preflight.max_seq_length,
            "rows_checked": preflight.rows_checked,
            "rows_checked_by_split": preflight.rows_checked_by_split,
            "max_token_count": preflight.max_token_count,
        },
        "source_policy": {
            "only_training_export": True,
            "official_chat_messages_used_exactly": True,
            "qwen35_processor_text_block_adapter_only": True,
            "sft_export_modified": False,
            "synthetic_rows_only": True,
            "real_manual_rows_excluded": True,
            "api_baseline_predictions_used_as_labels": False,
            "silent_truncation_or_dropping": False,
            "oversampling_changes_source_jsonl": False,
        },
    }


def _training_rows_for_sft(
    dataset: Wave4DetectorSftDataset,
    *,
    weakly_supported_train_multiplier: int,
) -> list[Wave4DetectorSftRow]:
    if weakly_supported_train_multiplier == 1:
        return list(dataset.train_rows)
    rows: list[Wave4DetectorSftRow] = []
    for row in dataset.train_rows:
        rows.append(row)
        if _gold_support_level(row) == "weakly_supported":
            rows.extend([row] * (weakly_supported_train_multiplier - 1))
    return rows


def _support_level_counts(rows: list[Wave4DetectorSftRow]) -> dict[str, int]:
    return dict(sorted(Counter(_gold_support_level(row) for row in rows).items()))


def _gold_support_level(row: Wave4DetectorSftRow) -> str:
    try:
        assistant_payload = json.loads(row.messages[2]["content"])
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{row.example_id}: assistant message must be valid DetectorAssessment JSON for training sampling"
        ) from error
    support_level = assistant_payload.get("support_level")
    if not isinstance(support_level, str):
        return "unknown"
    return support_level


def _dry_run_metrics(
    dataset: Wave4DetectorSftDataset,
    preflight: Wave4DetectorSftTokenPreflight,
    *,
    max_epochs: int,
    learning_rate: float,
    eval_steps: int | None,
    effective_train_rows: int,
    weakly_supported_train_multiplier: int,
) -> dict[str, Any]:
    return {
        "status": "passed",
        "mode": "dry_run",
        "max_epochs": max_epochs,
        "learning_rate": learning_rate,
        "eval_steps": eval_steps,
        "train_rows": len(dataset.train_rows),
        "effective_train_rows": effective_train_rows,
        "weakly_supported_train_multiplier": weakly_supported_train_multiplier,
        "validation_rows": len(dataset.validation_rows),
        "test_rows_reserved": len(dataset.test_rows),
        "tokenizer_preflight_rows_checked": preflight.rows_checked,
        "max_observed_token_count": preflight.max_token_count,
        "train_loss": None,
        "eval_loss": None,
        "cuda_checked": False,
        "cuda_available": None,
        "environment_note": "Dry-run does not check CUDA and must not be used to conclude GPU availability.",
    }


def _failed_train_metrics(
    dataset: Wave4DetectorSftDataset,
    preflight: Wave4DetectorSftTokenPreflight,
    error: Exception,
    *,
    max_epochs: int,
    learning_rate: float,
    eval_steps: int | None,
    effective_train_rows: int,
    weakly_supported_train_multiplier: int,
) -> dict[str, Any]:
    return {
        "status": "failed",
        "mode": "train",
        "max_epochs": max_epochs,
        "learning_rate": learning_rate,
        "eval_steps": eval_steps,
        "train_rows": len(dataset.train_rows),
        "effective_train_rows": effective_train_rows,
        "weakly_supported_train_multiplier": weakly_supported_train_multiplier,
        "validation_rows": len(dataset.validation_rows),
        "test_rows_reserved": len(dataset.test_rows),
        "tokenizer_preflight_rows_checked": preflight.rows_checked,
        "max_observed_token_count": preflight.max_token_count,
        "failure": str(error),
        "technical_fit_failure_policy": "stop_and_report_blocker_without_changing_model_or_provider_path",
    }


def _train_metrics(
    dataset: Wave4DetectorSftDataset,
    preflight: Wave4DetectorSftTokenPreflight,
    live_summary: dict[str, Any],
    *,
    max_epochs: int,
    learning_rate: float,
    eval_steps: int | None,
    effective_train_rows: int,
    weakly_supported_train_multiplier: int,
) -> dict[str, Any]:
    return {
        "status": "passed",
        "mode": "train",
        "max_epochs": max_epochs,
        "learning_rate": learning_rate,
        "eval_steps": eval_steps,
        "train_rows": len(dataset.train_rows),
        "effective_train_rows": effective_train_rows,
        "weakly_supported_train_multiplier": weakly_supported_train_multiplier,
        "validation_rows": len(dataset.validation_rows),
        "test_rows_reserved": len(dataset.test_rows),
        "tokenizer_preflight_rows_checked": preflight.rows_checked,
        "max_observed_token_count": preflight.max_token_count,
        **live_summary.get("metrics", {}),
    }


def _validation_selection_summary(
    *,
    mode: Literal["dry_run", "train"],
    max_epochs: int,
    live_summary: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    summary = {
        "selection_split": "validation",
        "selection_metric": "eval_loss",
        "max_epochs": max_epochs,
        "test_split_used_for_selection": False,
        "selected_checkpoint_source": "dry_run_no_checkpoint" if mode == "dry_run" else None,
    }
    if live_summary:
        summary["selected_checkpoint_source"] = live_summary.get("best_model_checkpoint")
        summary["best_eval_loss"] = live_summary.get("best_eval_loss")
    if error:
        summary["selection_status"] = "not_available_training_failed"
        summary["failure"] = error
    return summary


def _eval_summary(dataset: Wave4DetectorSftDataset) -> dict[str, Any]:
    return {
        "synthetic_test_rows_reserved": len(dataset.test_rows),
        "test_split_role": "post_training_synthetic_regression_only",
        "runtime_inference_integration_implemented": False,
        "sft_vllm_implemented": False,
        "api_baseline_predictions_used_as_labels": False,
        "real_manual_rows_trained_on": False,
        "post_training_eval_status": "not_run_in_dry_run_or_training_harness",
    }


def _load_live_model_and_tokenizer() -> dict[str, Any]:
    try:
        from unsloth import FastLanguageModel
    except Exception as error:
        raise RuntimeError(
            "Unsloth FastLanguageModel is required for --mode train tokenizer preflight; install the Wave 4 SFT training "
            "environment before running live training. This sandbox result does not prove CUDA is unavailable."
        ) from error
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_ID,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )
    return {"model": model, "tokenizer": tokenizer, "fast_language_model": FastLanguageModel}


def _run_live_unsloth_training(
    dataset: Wave4DetectorSftDataset,
    *,
    output_root: Path,
    adapter_dir: Path,
    live_assets: dict[str, Any] | None,
    max_epochs: int,
    learning_rate: float,
    eval_steps: int | None,
    save_total_limit: int,
    training_rows: list[Wave4DetectorSftRow] | None = None,
    per_device_train_batch_size: int = DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size: int = DEFAULT_PER_DEVICE_EVAL_BATCH_SIZE,
) -> dict[str, Any]:
    try:
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer
    except Exception as error:
        raise RuntimeError(
            "Live Wave 4 #120 training requires unsloth, torch, transformers, datasets, and trl. "
            "Install those dependencies in the training environment; do not switch base model or provider."
        ) from error

    if live_assets is None:
        live_assets = _load_live_model_and_tokenizer()
    model = live_assets["model"]
    tokenizer = live_assets["tokenizer"]
    FastLanguageModel = live_assets["fast_language_model"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=120,
        max_seq_length=MAX_SEQ_LENGTH,
    )
    if training_rows is None:
        training_rows = list(dataset.train_rows)
    train_dataset = Dataset.from_list([_trainer_text_row(row, tokenizer) for row in training_rows])
    eval_dataset = Dataset.from_list([_trainer_text_row(row, tokenizer) for row in dataset.validation_rows])
    eval_strategy = "steps" if eval_steps is not None else "epoch"
    training_arg_values: dict[str, Any] = {
        "output_dir": str(output_root / "checkpoints"),
        "num_train_epochs": max_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "gradient_accumulation_steps": 2,
        "learning_rate": learning_rate,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "weight_decay": 0.01,
        "bf16": True,
        "logging_steps": 10,
        "save_strategy": eval_strategy,
        "eval_strategy": eval_strategy,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "save_total_limit": save_total_limit,
        "report_to": [],
        "dataset_text_field": "text",
        "max_length": MAX_SEQ_LENGTH,
        "max_seq_length": MAX_SEQ_LENGTH,
        "remove_unused_columns": False,
        "packing": False,
    }
    if eval_steps is not None:
        training_arg_values["eval_steps"] = eval_steps
        training_arg_values["save_steps"] = eval_steps
    training_args = SFTConfig(**training_arg_values)
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )
    train_result = trainer.train()
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    eval_metrics = trainer.evaluate()
    eval_loss = _finite_metric_or_fallback(eval_metrics.get("eval_loss"), trainer.state.best_metric)
    return {
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_eval_loss": trainer.state.best_metric,
        "metrics": {
            "train_loss": getattr(train_result, "training_loss", None),
            "eval_loss": eval_loss,
        },
    }


def _finite_metric_or_fallback(value: Any, fallback: Any) -> Any:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return value
    return fallback


def _trainer_text_row(row: Wave4DetectorSftRow, tokenizer: ChatTemplateTokenizer) -> dict[str, str]:
    text = _apply_chat_template_for_qwen35(tokenizer, row.messages, tokenize=False, add_generation_prompt=False)
    return {"text": text, "example_id": row.example_id}


def _apply_chat_template_for_qwen35(
    tokenizer: ChatTemplateTokenizer,
    messages: list[dict[str, str]],
    *,
    tokenize: bool,
    add_generation_prompt: bool,
    **kwargs: Any,
) -> Any:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )
    except TypeError as error:
        if "string indices must be integers" not in str(error):
            raise
        return tokenizer.apply_chat_template(
            _messages_as_qwen35_text_blocks(messages),
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )


def _messages_as_qwen35_text_blocks(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "role": message["role"],
            "content": [{"type": "text", "text": message["content"]}],
        }
        for message in messages
    ]


def _token_count(tokenized: Any) -> int:
    if isinstance(tokenized, dict) and "input_ids" in tokenized:
        return _token_count(tokenized["input_ids"])
    shape = getattr(tokenized, "shape", None)
    if shape is not None:
        return int(shape[-1])
    try:
        if tokenized and isinstance(tokenized[0], (list, tuple)):
            return len(tokenized[0])
        return len(tokenized)
    except TypeError as error:
        raise ValueError("tokenizer.apply_chat_template(..., tokenize=True) must return a sized token sequence") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Wave 4 #120 detector SFT training path.")
    parser.add_argument("--sft-jsonl", default=str(CANONICAL_SFT_JSONL))
    parser.add_argument("--artifact-root", default=str(CANONICAL_ARTIFACT_ROOT))
    parser.add_argument("--mode", choices=["dry_run", "train"], required=True)
    parser.add_argument("--allow-noncanonical-sft-jsonl", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--per-device-train-batch-size", type=int, default=DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=DEFAULT_PER_DEVICE_EVAL_BATCH_SIZE)
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=None,
        help="Evaluate and save every N optimizer steps instead of every epoch.",
    )
    parser.add_argument("--save-total-limit", type=int, default=DEFAULT_SAVE_TOTAL_LIMIT)
    parser.add_argument(
        "--weakly-supported-train-multiplier",
        type=int,
        default=1,
        help="Duplicate weakly_supported train rows N times; validation and test rows are never duplicated.",
    )
    args = parser.parse_args(argv)
    result = run_detector_sft_training(
        sft_jsonl=args.sft_jsonl,
        artifact_root=args.artifact_root,
        mode=args.mode,
        allow_noncanonical_sft_jsonl=args.allow_noncanonical_sft_jsonl,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        weakly_supported_train_multiplier=args.weakly_supported_train_multiplier,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
    )
    print(
        json.dumps(
            {
                "status": result.status,
                "artifact_root": str(result.artifact_root),
                "errors": result.errors,
            },
            sort_keys=True,
        )
    )
    return 0 if result.status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
