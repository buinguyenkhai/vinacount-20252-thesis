from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from research.detector_contract_validation import (
    ALLOWED_SEVERITIES,
    ALLOWED_SUPPORT_LEVELS,
    enrich_detector_packet_evidence_roles,
    parse_and_validate_detector_assessment,
)
from research.detector_sft_trainer import (
    BASE_MODEL_ID,
    CANONICAL_SFT_JSONL,
    Wave4DetectorSftRow,
    _apply_chat_template_for_qwen35,
    load_detector_sft_dataset,
)
from research.sft_exporter import (
    DEFAULT_DETECTOR_SFT_SYSTEM_PROMPT_VERSION,
    DETECTOR_SFT_SYSTEM_PROMPTS,
    detector_sft_system_prompt,
)


DEFAULT_ADAPTER_DIR = Path("artifacts/detector_sft/qwen3_5_4b_unsloth_lora_v1_lr1e4_e6_eval180/adapter")
DEFAULT_OUTPUT_ROOT = Path("artifacts/detector_sft/qwen3_5_4b_unsloth_lora_v1_lr1e4_e6_eval180/eval/validation")
SUPPORTED_EVAL_SPLITS = {"validation", "test"}
SUPPORTED_SOURCE_FORMATS = {"sft_chat", "detector_record"}


@dataclass(frozen=True)
class Wave4DetectorSftEvalResult:
    status: str
    output_root: Path
    predictions_path: Path
    metrics_path: Path
    manifest_path: Path
    errors: list[str]


def run_detector_sft_evaluation(
    *,
    adapter_dir: Path | str | None = DEFAULT_ADAPTER_DIR,
    sft_jsonl: Path | str = CANONICAL_SFT_JSONL,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    split: Literal["validation", "test"] = "validation",
    source_format: Literal["sft_chat", "detector_record"] = "sft_chat",
    limit: int | None = None,
    max_new_tokens: int = 1024,
    batch_size: int = 1,
    device: str = "cuda",
    base_model_only: bool = False,
    allow_noncanonical_sft_jsonl: bool = False,
    system_prompt_version: str = DEFAULT_DETECTOR_SFT_SYSTEM_PROMPT_VERSION,
    packet_evidence_role_enrichment: bool = True,
) -> Wave4DetectorSftEvalResult:
    if split not in SUPPORTED_EVAL_SPLITS:
        raise ValueError(f"split must be one of {sorted(SUPPORTED_EVAL_SPLITS)}")
    if source_format not in SUPPORTED_SOURCE_FORMATS:
        raise ValueError(f"source_format must be one of {sorted(SUPPORTED_SOURCE_FORMATS)}")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    adapter_path = None if base_model_only else Path(adapter_dir or DEFAULT_ADAPTER_DIR)
    output_path = Path(output_root)
    predictions_path = output_path / "predictions.jsonl"
    metrics_path = output_path / "metrics.json"
    manifest_path = output_path / "manifest.json"
    rows, source_split_counts = _load_evaluation_rows(
        sft_jsonl=Path(sft_jsonl),
        split=split,
        source_format=source_format,
        allow_noncanonical_sft_jsonl=allow_noncanonical_sft_jsonl,
        system_prompt_version=system_prompt_version,
        packet_evidence_role_enrichment=packet_evidence_role_enrichment,
    )
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"no rows available for split={split}")

    model_bundle = _load_model_bundle(
        adapter_dir=adapter_path,
        device=device,
        base_model_only=base_model_only,
    )
    predictions: list[dict[str, Any]] = []
    next_progress_report = 25
    for batch_start in range(0, len(rows), batch_size):
        batch_rows = rows[batch_start : batch_start + batch_size]
        batch_records = [_prediction_record_for_row(row, split=split) for row in batch_rows]
        try:
            completions = _generate_detector_assessments(
                model_bundle,
                [row.messages[:2] for row in batch_rows],
                max_new_tokens=max_new_tokens,
            )
        except Exception as error:
            completions = [None] * len(batch_rows)
            batch_error = error
        else:
            batch_error = None
        for prediction_record, completion in zip(batch_records, completions, strict=True):
            if completion is None:
                _mark_prediction_invalid(prediction_record, batch_error or RuntimeError("batch generation failed"))
            else:
                _apply_completion_to_prediction(prediction_record, completion)
            prediction_record.pop("_gold_assessment", None)
            predictions.append(prediction_record)
        evaluated = len(predictions)
        if evaluated >= next_progress_report or evaluated == len(rows):
            print(json.dumps({"evaluated": evaluated, "total": len(rows)}, sort_keys=True), flush=True)
            while next_progress_report <= evaluated:
                next_progress_report += 25

    metrics = _score_predictions(predictions, split=split)
    manifest = {
        "status": "passed",
        "model_variant": "qwen3_5_4b_base" if base_model_only else "qwen3_5_4b_detector_lora",
        "adapter_dir": None if adapter_path is None else str(adapter_path),
        "base_model": BASE_MODEL_ID,
        "source_jsonl": str(sft_jsonl),
        "source_format": source_format,
        "source_split_counts": source_split_counts,
        "source_type_counts": dict(sorted(Counter(row.source_type for row in rows).items())),
        "split": split,
        "limit": limit,
        "max_new_tokens": max_new_tokens,
        "batch_size": batch_size,
        "device": model_bundle["device"],
        "enable_thinking": False,
        "system_prompt_version": system_prompt_version,
        "packet_evidence_role_enrichment": {
            "enabled_for_detector_record_source": source_format == "detector_record"
            and packet_evidence_role_enrichment,
            "risk_categories": [
                "earnings_cashflow_mismatch",
                "earnings_cashflow_quality_risk",
            ],
            "uses_gold_labels": False,
        },
        "predictions": "predictions.jsonl",
        "metrics": "metrics.json",
        "evaluation_policy": {
            "selection_split": split,
            "gold_source": _gold_source_for_format(source_format),
            "test_split_used_for_training_or_selection": False,
            "predictions_are_evaluation_outputs_only": True,
            "not_training_data": source_format == "detector_record",
        },
    }
    output_path.mkdir(parents=True, exist_ok=True)
    _write_jsonl(predictions_path, predictions)
    _write_json(metrics_path, metrics)
    _write_json(manifest_path, manifest)
    return Wave4DetectorSftEvalResult(
        status="passed",
        output_root=output_path,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        manifest_path=manifest_path,
        errors=[],
    )


def _load_evaluation_rows(
    *,
    sft_jsonl: Path,
    split: str,
    source_format: str,
    allow_noncanonical_sft_jsonl: bool,
    system_prompt_version: str,
    packet_evidence_role_enrichment: bool,
) -> tuple[list[Wave4DetectorSftRow], dict[str, int]]:
    if source_format == "sft_chat":
        dataset = load_detector_sft_dataset(
            sft_jsonl,
            allow_noncanonical_sft_jsonl=allow_noncanonical_sft_jsonl,
        )
        rows = dataset.validation_rows if split == "validation" else dataset.test_rows
        return rows, dataset.split_counts
    if source_format == "detector_record":
        rows = _read_detector_record_eval_rows(
            sft_jsonl,
            system_prompt_version=system_prompt_version,
            packet_evidence_role_enrichment=packet_evidence_role_enrichment,
        )
        split_counts = dict(sorted(Counter(row.split for row in rows).items()))
        return [row for row in rows if row.split == split], split_counts
    raise ValueError(f"source_format must be one of {sorted(SUPPORTED_SOURCE_FORMATS)}")


def _read_detector_record_eval_rows(
    source_path: Path,
    *,
    system_prompt_version: str = DEFAULT_DETECTOR_SFT_SYSTEM_PROMPT_VERSION,
    packet_evidence_role_enrichment: bool = True,
) -> list[Wave4DetectorSftRow]:
    rows: list[Wave4DetectorSftRow] = []
    for line_number, line in enumerate(source_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{source_path}:{line_number}: invalid JSONL row: {error.msg}") from error
        rows.append(
            _parse_detector_record_eval_row(
                payload,
                line_number=line_number,
                source_path=source_path,
                system_prompt_version=system_prompt_version,
                packet_evidence_role_enrichment=packet_evidence_role_enrichment,
            )
        )
    if not rows:
        raise ValueError(f"{source_path} contains no detector records")
    return rows


def _parse_detector_record_eval_row(
    payload: dict[str, Any],
    *,
    line_number: int,
    source_path: Path,
    system_prompt_version: str = DEFAULT_DETECTOR_SFT_SYSTEM_PROMPT_VERSION,
    packet_evidence_role_enrichment: bool = True,
) -> Wave4DetectorSftRow:
    if not isinstance(payload, dict):
        raise ValueError(f"{source_path}:{line_number}: detector record row must be a JSON object")
    split = _detector_record_split(payload, source_path=source_path, line_number=line_number)
    if split not in SUPPORTED_EVAL_SPLITS:
        raise ValueError(f"{source_path}:{line_number}: unsupported evaluation split: {split!r}")
    input_wrapper = payload.get("input")
    output_wrapper = payload.get("output")
    if not isinstance(input_wrapper, dict) or input_wrapper.get("type") != "DetectorPacket":
        raise ValueError(f"{source_path}:{line_number}: input wrapper type must be DetectorPacket")
    if not isinstance(output_wrapper, dict) or output_wrapper.get("type") != "DetectorAssessment":
        raise ValueError(f"{source_path}:{line_number}: output wrapper type must be DetectorAssessment")
    packet = input_wrapper.get("data")
    gold = output_wrapper.get("data")
    if not isinstance(packet, dict):
        raise ValueError(f"{source_path}:{line_number}: input.data must be a DetectorPacket object")
    if not isinstance(gold, dict):
        raise ValueError(f"{source_path}:{line_number}: output.data must be a DetectorAssessment object")
    if packet_evidence_role_enrichment:
        packet = enrich_detector_packet_evidence_roles(packet)
    try:
        gold = parse_and_validate_detector_assessment(_canonical_json(gold), packet)
    except ValueError as error:
        raise ValueError(f"{source_path}:{line_number}: output.data must validate against input.data: {error}") from error

    example_id = payload.get("example_id")
    if not isinstance(example_id, str) or not example_id:
        raise ValueError(f"{source_path}:{line_number}: example_id must be a non-empty string")
    source_type = payload.get("source_type")
    if not isinstance(source_type, str) or not source_type:
        source_type = "detector_record"
    metadata = _detector_record_metadata(payload, packet=packet, gold=gold, split=split)
    messages = [
        {"role": "system", "content": detector_sft_system_prompt(system_prompt_version)},
        {"role": "user", "content": _canonical_json(packet)},
        {"role": "assistant", "content": _canonical_json(gold)},
    ]
    return Wave4DetectorSftRow(
        line_number=line_number,
        example_id=example_id,
        split=split,
        source_type=source_type,
        messages=messages,
        metadata=metadata,
    )


def _detector_record_split(payload: dict[str, Any], *, source_path: Path, line_number: int) -> str:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{source_path}:{line_number}: detector record metadata must be an object")
    split_metadata = metadata.get("split_metadata", {})
    if isinstance(split_metadata, dict) and isinstance(split_metadata.get("split"), str):
        return split_metadata["split"]
    if isinstance(metadata.get("split"), str):
        return metadata["split"]
    raise ValueError(f"{source_path}:{line_number}: metadata split must be present")


def _detector_record_metadata(
    payload: dict[str, Any],
    *,
    packet: dict[str, Any],
    gold: dict[str, Any],
    split: str,
) -> dict[str, Any]:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    split_metadata = metadata.get("split_metadata") if isinstance(metadata.get("split_metadata"), dict) else {}
    eval_metadata = {
        "example_id": payload.get("example_id"),
        "dataset_version": payload.get("dataset_version"),
        "source_type": payload.get("source_type", "detector_record"),
        "split": split,
        "risk_category": gold.get("risk_category"),
        "support_level": gold.get("support_level"),
        "severity": gold.get("severity"),
        "report_profile": metadata.get("report_profile"),
        "report_id": packet.get("report_id"),
    }
    for field in [
        "company_key",
        "period_key",
        "group_key",
        "source_file_sha256",
        "normalized_text_hash",
        "table_content_hash",
        "derived_from_report_artifact_id",
        "derived_from_source_document_id",
    ]:
        if split_metadata.get(field) is not None:
            eval_metadata[field] = split_metadata[field]
    return eval_metadata


def _gold_source_for_format(source_format: str) -> str:
    if source_format == "detector_record":
        return "output.data in detector record JSONL"
    return "assistant message in detector_sft_chat.jsonl"


def _load_model_bundle(
    *,
    adapter_dir: Path | None,
    device: str,
    base_model_only: bool,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_device = _resolve_device(torch, device)
    tokenizer_source = BASE_MODEL_ID if base_model_only else adapter_dir
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        dtype=torch.bfloat16 if resolved_device.type == "cuda" else torch.float32,
        local_files_only=True,
    )
    base_model.to(resolved_device)
    if base_model_only:
        model = base_model
    else:
        from peft import PeftModel

        if adapter_dir is None:
            raise ValueError("adapter_dir is required unless base_model_only=True")
        model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.to(resolved_device)
    model.eval()
    if resolved_device.type == "cuda":
        _assert_model_materialized_on_cuda(model)
    return {"model": model, "tokenizer": tokenizer, "torch": torch, "device": str(resolved_device)}


def _resolve_device(torch: Any, device: str) -> Any:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA evaluation requested with --device {device!r}, but torch.cuda.is_available() is false. "
            "Run outside the sandbox or pass --device cpu for a debug-only smoke run."
        )
    return torch.device(device)


def _assert_model_materialized_on_cuda(model: Any) -> None:
    meta_parameters: list[str] = []
    cpu_parameters: list[str] = []
    for name, parameter in model.named_parameters():
        if parameter.device.type == "meta":
            meta_parameters.append(name)
        elif parameter.device.type != "cuda":
            cpu_parameters.append(f"{name}:{parameter.device}")
    if not meta_parameters and not cpu_parameters:
        return
    example_meta = ", ".join(meta_parameters[:3])
    example_cpu = ", ".join(cpu_parameters[:3])
    raise RuntimeError(
        "CUDA evaluator load left model parameters off GPU; refusing to run slow or partial CPU evaluation. "
        f"meta_parameters={len(meta_parameters)} [{example_meta}], "
        f"non_cuda_parameters={len(cpu_parameters)} [{example_cpu}]"
    )


def _generate_detector_assessment(
    model_bundle: dict[str, Any],
    prompt_messages: list[dict[str, str]],
    *,
    max_new_tokens: int,
) -> str:
    model = model_bundle["model"]
    tokenizer = model_bundle["tokenizer"]
    torch = model_bundle["torch"]
    tokenized = _apply_chat_template_for_qwen35(
        tokenizer,
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    model_device = torch.device(model_bundle["device"])
    if isinstance(tokenized, Mapping):
        model_inputs = {key: value.to(model_device) for key, value in tokenized.items()}
        input_length = model_inputs["input_ids"].shape[-1]
    elif hasattr(tokenized, "data") and isinstance(tokenized.data, Mapping):
        model_inputs = {key: value.to(model_device) for key, value in tokenized.data.items()}
        input_length = model_inputs["input_ids"].shape[-1]
    else:
        input_ids = tokenized.to(model_device)
        model_inputs = {"input_ids": input_ids}
        input_length = input_ids.shape[-1]
    with torch.inference_mode():
        generate_kwargs = {
            **model_inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }
        if tokenizer.eos_token_id is not None:
            generate_kwargs["pad_token_id"] = tokenizer.eos_token_id
        output_ids = model.generate(
            **generate_kwargs,
        )
    generated_ids = output_ids[0][input_length:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def _generate_detector_assessments(
    model_bundle: dict[str, Any],
    prompt_messages_batch: list[list[dict[str, str]]],
    *,
    max_new_tokens: int,
) -> list[str]:
    if not prompt_messages_batch:
        return []
    if len(prompt_messages_batch) == 1:
        return [
            _generate_detector_assessment(
                model_bundle,
                prompt_messages_batch[0],
                max_new_tokens=max_new_tokens,
            )
        ]
    model = model_bundle["model"]
    tokenizer = model_bundle["tokenizer"]
    torch = model_bundle["torch"]
    model_device = torch.device(model_bundle["device"])
    prompt_texts = [
        _apply_chat_template_for_qwen35(
            tokenizer,
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt_messages in prompt_messages_batch
    ]
    _ensure_tokenizer_can_pad(tokenizer)
    original_padding_side = getattr(tokenizer, "padding_side", None)
    tokenizer.padding_side = "left"
    try:
        model_inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True)
    finally:
        if original_padding_side is not None:
            tokenizer.padding_side = original_padding_side
    model_inputs = {key: value.to(model_device) for key, value in model_inputs.items()}
    input_width = model_inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        generate_kwargs = {
            **model_inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if pad_token_id is not None:
            generate_kwargs["pad_token_id"] = pad_token_id
        output_ids = model.generate(**generate_kwargs)
    completions = []
    for row_output_ids in output_ids:
        generated_ids = row_output_ids[input_width:]
        completions.append(tokenizer.decode(generated_ids, skip_special_tokens=True).strip())
    return completions


def _ensure_tokenizer_can_pad(tokenizer: Any) -> None:
    if tokenizer.pad_token_id is not None:
        return
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        return
    if tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        return
    raise ValueError("tokenizer must define a pad token or eos token for batched evaluation")


def _prediction_record_for_row(row: Wave4DetectorSftRow, *, split: str) -> dict[str, Any]:
    packet = json.loads(row.messages[1]["content"])
    gold = json.loads(row.messages[2]["content"])
    prediction_record = _base_prediction_record(row.example_id, split, packet, gold)
    prediction_record["_gold_assessment"] = gold
    return prediction_record


def _apply_completion_to_prediction(prediction_record: dict[str, Any], completion: str) -> None:
    packet = prediction_record["model_visible_input"]["data"]
    gold = prediction_record["_gold_assessment"]
    try:
        prediction_record["raw_completion"] = completion
        assessment = parse_and_validate_detector_assessment(completion, packet)
    except Exception as error:
        _mark_prediction_invalid(prediction_record, error)
        return
    prediction_record["prediction_status"] = "accepted"
    prediction_record["schema_valid"] = True
    prediction_record["evidence_valid"] = True
    prediction_record["prediction"] = assessment
    prediction_record["predicted_support_level"] = assessment["support_level"]
    prediction_record["predicted_severity"] = assessment["severity"]
    prediction_record["support_level_exact_match"] = assessment["support_level"] == gold["support_level"]
    prediction_record["severity_exact_match"] = assessment["severity"] == gold["severity"]
    prediction_record["assessment_exact_match"] = _canonical_json(assessment) == _canonical_json(gold)


def _mark_prediction_invalid(prediction_record: dict[str, Any], error: Exception) -> None:
    prediction_record["prediction_status"] = "invalid"
    prediction_record["schema_valid"] = False
    prediction_record["evidence_valid"] = False
    prediction_record["invalid_reason_codes"] = [_error_reason_code(error)]


def _base_prediction_record(
    example_id: str,
    split: str,
    packet: dict[str, Any],
    gold: dict[str, Any],
) -> dict[str, Any]:
    return {
        "example_id": example_id,
        "split": split,
        "packet_id": packet["packet_id"],
        "risk_category": packet["task"]["risk_category"],
        "gold_support_level": gold["support_level"],
        "gold_severity": gold["severity"],
        "prediction_status": "not_attempted",
        "schema_valid": False,
        "evidence_valid": False,
        "model_visible_input": {"type": "DetectorPacket", "data": packet},
    }


def _score_predictions(predictions: list[dict[str, Any]], *, split: str) -> dict[str, Any]:
    valid_predictions = [
        prediction
        for prediction in predictions
        if prediction.get("prediction_status") == "accepted"
    ]
    invalid_predictions = [
        prediction
        for prediction in predictions
        if prediction.get("prediction_status") == "invalid"
    ]
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    severity_confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    invalid_reason_counts: Counter[str] = Counter()
    by_support_level: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_risk_category: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    false_positive_total = 0
    false_positive_correct_or_conservative = 0
    insufficient_gold_total = 0
    insufficient_predicted_total = 0
    insufficient_true_positive = 0

    for prediction in predictions:
        gold_support = prediction["gold_support_level"]
        gold_severity = prediction["gold_severity"]
        by_support_level[gold_support]["total"] += 1
        by_risk_category[prediction["risk_category"]]["total"] += 1
        if gold_support == "insufficient_evidence":
            insufficient_gold_total += 1
        if gold_support in {"not_supported", "insufficient_evidence"}:
            false_positive_total += 1
        if prediction.get("prediction_status") == "invalid":
            by_support_level[gold_support]["invalid"] += 1
            by_risk_category[prediction["risk_category"]]["invalid"] += 1
            invalid_reason_counts.update(prediction.get("invalid_reason_codes", []))
            continue
        if prediction.get("prediction_status") != "accepted":
            continue
        predicted_support = prediction["predicted_support_level"]
        predicted_severity = prediction["predicted_severity"]
        confusion[gold_support][predicted_support] += 1
        severity_confusion[gold_severity][predicted_severity] += 1
        if predicted_support == gold_support:
            by_support_level[gold_support]["support_level_exact_match"] += 1
            by_risk_category[prediction["risk_category"]]["support_level_exact_match"] += 1
        if predicted_severity == gold_severity:
            by_support_level[gold_support]["severity_exact_match"] += 1
            by_risk_category[prediction["risk_category"]]["severity_exact_match"] += 1
        if prediction.get("assessment_exact_match"):
            by_support_level[gold_support]["assessment_exact_match"] += 1
            by_risk_category[prediction["risk_category"]]["assessment_exact_match"] += 1
        if gold_support in {"not_supported", "insufficient_evidence"} and predicted_support in {
            "not_supported",
            "insufficient_evidence",
            "weakly_supported",
        }:
            false_positive_correct_or_conservative += 1
        if predicted_support == "insufficient_evidence":
            insufficient_predicted_total += 1
        if gold_support == "insufficient_evidence" and predicted_support == "insufficient_evidence":
            insufficient_true_positive += 1

    support_exact = sum(
        1 for prediction in valid_predictions if prediction.get("support_level_exact_match")
    )
    severity_exact = sum(1 for prediction in valid_predictions if prediction.get("severity_exact_match"))
    assessment_exact = sum(1 for prediction in valid_predictions if prediction.get("assessment_exact_match"))
    support_class_metrics = _classification_metrics(
        predictions,
        labels=sorted(ALLOWED_SUPPORT_LEVELS),
        gold_key="gold_support_level",
        predicted_key="predicted_support_level",
    )
    severity_class_metrics = _classification_metrics(
        predictions,
        labels=sorted(ALLOWED_SEVERITIES),
        gold_key="gold_severity",
        predicted_key="predicted_severity",
    )
    return {
        "split": split,
        "num_examples": len(predictions),
        "num_attempted": len(predictions),
        "num_valid_predictions": len(valid_predictions),
        "num_invalid_responses": len(invalid_predictions),
        "schema_valid_count": sum(1 for prediction in predictions if prediction.get("schema_valid")),
        "evidence_id_valid_count": sum(1 for prediction in predictions if prediction.get("evidence_valid")),
        "support_level_exact_match_count": support_exact,
        "support_level_exact_match_rate": _rate(support_exact, len(predictions)),
        "support_level_exact_match_rate_on_valid": _rate(support_exact, len(valid_predictions)),
        "severity_exact_match_count": severity_exact,
        "severity_exact_match_rate": _rate(severity_exact, len(predictions)),
        "severity_exact_match_rate_on_valid": _rate(severity_exact, len(valid_predictions)),
        "severity_macro_f1": _macro_f1(severity_class_metrics),
        "severity_class_metrics": severity_class_metrics,
        "assessment_exact_match_count": assessment_exact,
        "assessment_exact_match_rate": _rate(assessment_exact, len(predictions)),
        "false_positive_rejection": {
            "correct_or_conservative": false_positive_correct_or_conservative,
            "total": false_positive_total,
            "rate": _rate(false_positive_correct_or_conservative, false_positive_total),
        },
        "insufficient_evidence_detection": {
            "true_positive": insufficient_true_positive,
            "predicted_total": insufficient_predicted_total,
            "gold_total": insufficient_gold_total,
            "recall": _rate(insufficient_true_positive, insufficient_gold_total),
            "precision": _rate(insufficient_true_positive, insufficient_predicted_total),
        },
        "support_level_macro_f1": _macro_f1(support_class_metrics),
        "support_level_class_metrics": support_class_metrics,
        "weakly_supported_detection": support_class_metrics["weakly_supported"],
        "support_level_confusion_matrix_counts": {
            gold: dict(predicted_counts)
            for gold, predicted_counts in sorted(confusion.items())
        },
        "severity_confusion_matrix_counts": {
            gold: dict(predicted_counts)
            for gold, predicted_counts in sorted(severity_confusion.items())
        },
        "invalid_reason_counts": dict(sorted(invalid_reason_counts.items())),
        "by_gold_support_level": _summarize_groups(by_support_level),
        "by_risk_category": _summarize_groups(by_risk_category),
        "support_metrics_by_risk_category": _support_metrics_by_risk_category(predictions),
    }


def _support_metrics_by_risk_category(
    predictions: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    categories = sorted({prediction["risk_category"] for prediction in predictions})
    summaries: dict[str, dict[str, Any]] = {}
    for category in categories:
        category_predictions = [
            prediction
            for prediction in predictions
            if prediction["risk_category"] == category
        ]
        class_metrics = _classification_metrics(
            category_predictions,
            labels=sorted(ALLOWED_SUPPORT_LEVELS),
            gold_key="gold_support_level",
            predicted_key="predicted_support_level",
        )
        confusion: dict[str, Counter[str]] = defaultdict(Counter)
        for prediction in category_predictions:
            if prediction.get("prediction_status") == "accepted":
                confusion[prediction["gold_support_level"]][prediction["predicted_support_level"]] += 1
        exact = sum(
            1
            for prediction in category_predictions
            if prediction.get("support_level_exact_match")
        )
        summaries[category] = {
            "num_examples": len(category_predictions),
            "support_level_exact_match_count": exact,
            "support_level_exact_match_rate": _rate(exact, len(category_predictions)),
            "support_level_macro_f1": _macro_f1(class_metrics),
            "support_level_class_metrics": class_metrics,
            "support_level_confusion_matrix_counts": {
                gold: dict(sorted(predicted.items()))
                for gold, predicted in sorted(confusion.items())
            },
        }
    return summaries


def _classification_metrics(
    predictions: list[dict[str, Any]],
    *,
    labels: list[str],
    gold_key: str,
    predicted_key: str,
) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for label in labels:
        true_positive = 0
        predicted_total = 0
        gold_total = 0
        for prediction in predictions:
            gold = prediction[gold_key]
            predicted = prediction.get(predicted_key) if prediction.get("prediction_status") == "accepted" else None
            if gold == label:
                gold_total += 1
            if predicted == label:
                predicted_total += 1
            if gold == label and predicted == label:
                true_positive += 1
        precision = _rate(true_positive, predicted_total)
        recall = _rate(true_positive, gold_total)
        metrics[label] = {
            "true_positive": true_positive,
            "predicted_total": predicted_total,
            "gold_total": gold_total,
            "precision": precision,
            "recall": recall,
            "f1": _f1(
                precision,
                recall,
                predicted_total=predicted_total,
                gold_total=gold_total,
            ),
        }
    return metrics


def _f1(
    precision: float | None,
    recall: float | None,
    *,
    predicted_total: int,
    gold_total: int,
) -> float | None:
    if predicted_total == 0 and gold_total == 0:
        return None
    if precision is None or recall is None:
        return 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _macro_f1(class_metrics: dict[str, dict[str, Any]]) -> float | None:
    f1_values = [metrics["f1"] for metrics in class_metrics.values() if metrics["f1"] is not None]
    if not f1_values:
        return None
    return sum(f1_values) / len(f1_values)


def _summarize_groups(groups: dict[str, dict[str, int]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for name, counts in sorted(groups.items()):
        total = counts.get("total", 0)
        support_matches = counts.get("support_level_exact_match", 0)
        severity_matches = counts.get("severity_exact_match", 0)
        summary[name] = {
            **dict(sorted(counts.items())),
            "support_level_exact_match_rate": _rate(support_matches, total),
            "severity_exact_match_rate": _rate(severity_matches, total),
        }
    return summary


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _error_reason_code(error: Exception) -> str:
    message = str(error)
    if message:
        return message
    return error.__class__.__name__


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a Wave 4 detector SFT adapter on SFT validation/test rows.")
    parser.add_argument("--adapter-dir", default=str(DEFAULT_ADAPTER_DIR))
    parser.add_argument("--sft-jsonl", default=str(CANONICAL_SFT_JSONL))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--split", choices=sorted(SUPPORTED_EVAL_SPLITS), default="validation")
    parser.add_argument("--source-format", choices=sorted(SUPPORTED_SOURCE_FORMATS), default="sft_chat")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--base-model-only",
        action="store_true",
        help="Evaluate the base Qwen3.5-4B model without loading a LoRA adapter.",
    )
    parser.add_argument("--allow-noncanonical-sft-jsonl", action="store_true")
    parser.add_argument(
        "--system-prompt-version",
        choices=sorted(DETECTOR_SFT_SYSTEM_PROMPTS),
        default=DEFAULT_DETECTOR_SFT_SYSTEM_PROMPT_VERSION,
    )
    parser.add_argument(
        "--disable-packet-evidence-role-enrichment",
        action="store_true",
        help="Evaluate detector_record JSONL using the packet surface exactly as stored.",
    )
    args = parser.parse_args(argv)
    result = run_detector_sft_evaluation(
        adapter_dir=None if args.base_model_only else args.adapter_dir,
        sft_jsonl=args.sft_jsonl,
        output_root=args.output_root,
        split=args.split,
        source_format=args.source_format,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        device=args.device,
        base_model_only=args.base_model_only,
        allow_noncanonical_sft_jsonl=args.allow_noncanonical_sft_jsonl,
        system_prompt_version=args.system_prompt_version,
        packet_evidence_role_enrichment=not args.disable_packet_evidence_role_enrichment,
    )
    print(
        json.dumps(
            {
                "status": result.status,
                "output_root": str(result.output_root),
                "errors": result.errors,
            },
            sort_keys=True,
        )
    )
    return 0 if result.status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
