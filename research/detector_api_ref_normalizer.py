from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research.api_llm_detector_baseline import _visible_packet_evidence_ids
from research.detector_contract_validation import (
    normalize_json_content,
    parse_and_validate_detector_assessment,
)
from research.detector_sft_evaluator import (
    _error_reason_code,
    _score_predictions,
)


DEFAULT_SOURCE_EVAL_ROOT = Path("artifacts/detector_ablation/deepseek_v4_pro_api/validation")
DEFAULT_OUTPUT_ROOT = Path("artifacts/detector_ablation/deepseek_v4_pro_api_ref_normalized/validation")


@dataclass(frozen=True)
class Wave4DetectorApiRefNormalizerResult:
    status: str
    output_root: Path
    predictions_path: Path
    metrics_path: Path
    manifest_path: Path
    invalid_responses_path: Path
    repair_records_path: Path
    errors: list[str]


def run_detector_api_ref_normalizer(
    *,
    source_eval_root: Path | str = DEFAULT_SOURCE_EVAL_ROOT,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    variant: str = "deepseek_v4_pro_api_ref_normalized",
) -> Wave4DetectorApiRefNormalizerResult:
    source_path = Path(source_eval_root)
    output_path = Path(output_root)
    predictions_path = output_path / "predictions.jsonl"
    metrics_path = output_path / "metrics.json"
    manifest_path = output_path / "manifest.json"
    invalid_responses_path = output_path / "invalid_responses.jsonl"
    repair_records_path = output_path / "repair_records.jsonl"
    raw_invalid_responses_path = output_path / "debug" / "raw_invalid_responses.jsonl"

    source_manifest = _read_json(source_path / "manifest.json")
    source_predictions = _read_jsonl(source_path / "predictions.jsonl")
    source_invalid_by_example_id = {
        record["example_id"]: record
        for record in _read_jsonl(source_path / "invalid_responses.jsonl")
    }
    source_raw_invalid_by_example_id = {
        record["example_id"]: record
        for record in _read_jsonl(source_path / "debug" / "raw_invalid_responses.jsonl")
    }
    if not source_predictions:
        raise ValueError(f"{source_path / 'predictions.jsonl'} contains no predictions")

    predictions: list[dict[str, Any]] = []
    invalid_responses: list[dict[str, Any]] = []
    raw_invalid_responses: list[dict[str, Any]] = []
    repair_records: list[dict[str, Any]] = []
    for source_record in source_predictions:
        prediction, invalid_response, raw_invalid_response, repair_record = _normalized_prediction(
            source_record,
            source_invalid_by_example_id=source_invalid_by_example_id,
            source_raw_invalid_by_example_id=source_raw_invalid_by_example_id,
        )
        predictions.append(prediction)
        if invalid_response is not None:
            invalid_responses.append(invalid_response)
        if raw_invalid_response is not None:
            raw_invalid_responses.append(raw_invalid_response)
        if repair_record is not None:
            repair_records.append(repair_record)

    metrics = _score_predictions(predictions, split=source_manifest.get("split", source_predictions[0]["split"]))
    metrics["ref_normalizer_counts"] = _ref_normalizer_counts(repair_records)
    manifest = {
        "status": "passed",
        "variant": variant,
        "source_eval_root": str(source_path),
        "source_manifest": source_manifest,
        "split": metrics["split"],
        "predictions": "predictions.jsonl",
        "metrics": "metrics.json",
        "invalid_responses": "invalid_responses.jsonl",
        "repair_records": "repair_records.jsonl",
        "ref_normalizer_policy": {
            "semantic_prediction_source": "raw invalid API response from source eval debug artifact",
            "eligible_source_invalid_reason_codes": ["invalid_evidence_ids"],
            "repair_rule": "replace an out-of-contract ref_id only when exactly one visible packet evidence ID has ':<ref_id>' as suffix",
            "ambiguous_or_unrepairable_outputs_remain_invalid": True,
            "support_level_or_severity_overrides": False,
            "rationale_overrides": False,
            "labels_used_for_repair": False,
            "schema_validation_after_repair": True,
            "evidence_reference_validation_after_repair": True,
        },
        "artifact_policy": {
            "canonical_prediction_artifact": "predictions.jsonl",
            "canonical_metrics_artifact": "metrics.json",
            "canonical_repair_log_artifact": "repair_records.jsonl",
            "canonical_invalid_response_artifact": "invalid_responses.jsonl",
            "raw_invalid_responses_debug_artifact": "debug/raw_invalid_responses.jsonl",
            "raw_valid_responses_canonical": False,
            "raw_responses_are_labels": False,
            "predictions_are_evaluation_outputs_only": True,
            "not_gold_labels": True,
            "not_training_data": True,
            "usable_for_training": False,
        },
    }
    output_path.mkdir(parents=True, exist_ok=True)
    raw_invalid_responses_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(predictions_path, predictions)
    _write_json(metrics_path, metrics)
    _write_json(manifest_path, manifest)
    _write_jsonl(invalid_responses_path, invalid_responses)
    _write_jsonl(repair_records_path, repair_records)
    _write_jsonl(raw_invalid_responses_path, raw_invalid_responses)
    return Wave4DetectorApiRefNormalizerResult(
        status="passed",
        output_root=output_path,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        manifest_path=manifest_path,
        invalid_responses_path=invalid_responses_path,
        repair_records_path=repair_records_path,
        errors=[],
    )


def _normalized_prediction(
    source_record: dict[str, Any],
    *,
    source_invalid_by_example_id: dict[str, dict[str, Any]],
    source_raw_invalid_by_example_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    prediction = copy.deepcopy(source_record)
    prediction["source_prediction_status"] = source_record.get("prediction_status")
    prediction["source_invalid_reason_codes"] = source_record.get("invalid_reason_codes", [])
    prediction["ref_normalizer_actions"] = []
    if source_record.get("prediction_status") != "invalid":
        return prediction, None, None, None
    raw_invalid = source_raw_invalid_by_example_id.get(source_record["example_id"])
    if source_record.get("invalid_reason_codes") != ["invalid_evidence_ids"] or raw_invalid is None:
        return _remaining_invalid_tuple(prediction, source_invalid_by_example_id, raw_invalid)

    packet = source_record["model_visible_input"]["data"]
    try:
        assessment = _parse_assessment(raw_invalid["raw_response_text"])
        changes = _repair_unambiguous_suffix_ref_ids(assessment, packet)
        completion = json.dumps(assessment, ensure_ascii=False, sort_keys=True)
        validated = parse_and_validate_detector_assessment(completion, packet)
    except Exception as error:
        prediction["invalid_reason_codes"] = [_error_reason_code(error)]
        return _remaining_invalid_tuple(prediction, source_invalid_by_example_id, raw_invalid)

    prediction["prediction_status"] = "accepted"
    prediction["schema_valid"] = True
    prediction["evidence_valid"] = True
    prediction["prediction"] = validated
    prediction["predicted_support_level"] = validated["support_level"]
    prediction["predicted_severity"] = validated["severity"]
    prediction["support_level_exact_match"] = validated["support_level"] == source_record["gold_support_level"]
    prediction["severity_exact_match"] = validated["severity"] == source_record["gold_severity"]
    prediction["assessment_exact_match"] = False
    prediction["ref_normalizer_actions"] = ["repaired_evidence_ref_id"] if changes else []
    prediction["ref_normalizer_repaired_ref_count"] = len(changes)
    prediction.pop("invalid_reason_codes", None)
    repair_record = {
        "example_id": source_record["example_id"],
        "packet_id": source_record["packet_id"],
        "candidate_id": packet["candidate_id"],
        "source_invalid_reason_codes": source_record.get("invalid_reason_codes", []),
        "actions": prediction["ref_normalizer_actions"],
        "repaired_ref_count": len(changes),
        "changes": changes,
        "support_level_or_severity_overrides": False,
        "labels_used_for_repair": False,
    }
    return prediction, None, None, repair_record


def _remaining_invalid_tuple(
    prediction: dict[str, Any],
    source_invalid_by_example_id: dict[str, dict[str, Any]],
    raw_invalid: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None, None]:
    invalid_response = copy.deepcopy(source_invalid_by_example_id.get(prediction["example_id"], {}))
    if not invalid_response:
        invalid_response = {
            "example_id": prediction["example_id"],
            "packet_id": prediction["packet_id"],
            "candidate_id": prediction["model_visible_input"]["data"]["candidate_id"],
            "failure_stage": "response_validation",
            "raw_response_debug_path": "debug/raw_invalid_responses.jsonl",
        }
    invalid_response["reason_codes"] = prediction.get("invalid_reason_codes", [])
    copied_raw_invalid = copy.deepcopy(raw_invalid) if raw_invalid is not None else None
    if copied_raw_invalid is not None:
        copied_raw_invalid["reason_codes"] = prediction.get("invalid_reason_codes", [])
    return prediction, invalid_response, copied_raw_invalid, None


def _parse_assessment(raw_response_text: str) -> dict[str, Any]:
    parsed = json.loads(normalize_json_content(raw_response_text))
    if not isinstance(parsed, dict):
        raise ValueError("wrong_top_level_structure")
    if parsed.get("type") == "DetectorAssessment" and isinstance(parsed.get("data"), dict):
        parsed = parsed["data"]
    return parsed


def _repair_unambiguous_suffix_ref_ids(assessment: dict[str, Any], packet: dict[str, Any]) -> list[dict[str, Any]]:
    visible_ids = _visible_packet_evidence_ids(packet)
    changes: list[dict[str, Any]] = []
    for path, evidence_ref in _iter_assessment_evidence_refs(assessment):
        old_ref_id = evidence_ref.get("ref_id")
        if not isinstance(old_ref_id, str) or old_ref_id in visible_ids:
            continue
        suffix_matches = sorted(ref_id for ref_id in visible_ids if ref_id.endswith(f":{old_ref_id}"))
        if len(suffix_matches) != 1:
            continue
        evidence_ref["ref_id"] = suffix_matches[0]
        changes.append(
            {
                "path": path,
                "old_ref_id": old_ref_id,
                "new_ref_id": suffix_matches[0],
                "evidence_ref_type": evidence_ref.get("evidence_ref_type"),
                "role": evidence_ref.get("role"),
            }
        )
    return changes


def _iter_assessment_evidence_refs(assessment: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    refs: list[tuple[str, dict[str, Any]]] = []
    for index, ref in enumerate(assessment.get("cited_evidence_refs", [])):
        if isinstance(ref, dict):
            refs.append((f"cited_evidence_refs[{index}]", ref))
    for signal_index, signal in enumerate(assessment.get("validated_signals", [])):
        if not isinstance(signal, dict):
            continue
        for ref_index, ref in enumerate(signal.get("cited_evidence_refs", [])):
            if isinstance(ref, dict):
                refs.append((f"validated_signals[{signal_index}].cited_evidence_refs[{ref_index}]", ref))
    return refs


def _ref_normalizer_counts(repair_records: list[dict[str, Any]]) -> dict[str, Any]:
    action_counts = Counter(action for record in repair_records for action in record.get("actions", []))
    return {
        "repaired_records": len(repair_records),
        "repaired_ref_count": sum(record.get("repaired_ref_count", 0) for record in repair_records),
        "actions": dict(sorted(action_counts.items())),
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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
    parser = argparse.ArgumentParser(description="Apply deterministic ref_id normalization to DeepSeek API detector eval artifacts.")
    parser.add_argument("--source-eval-root", default=str(DEFAULT_SOURCE_EVAL_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--variant", default="deepseek_v4_pro_api_ref_normalized")
    args = parser.parse_args(argv)
    result = run_detector_api_ref_normalizer(
        source_eval_root=args.source_eval_root,
        output_root=args.output_root,
        variant=args.variant,
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
