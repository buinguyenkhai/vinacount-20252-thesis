from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research.detector_contract_validation import (
    normalize_json_content,
    parse_and_validate_detector_assessment,
)
from research.detector_sft_evaluator import (
    _error_reason_code,
    _score_predictions,
)
from vinacount.runtime_detector_modes import apply_sft_detector_contract_guard


DEFAULT_SOURCE_EVAL_ROOT = Path(
    "artifacts/detector_sft/qwen3_5_4b_unsloth_lora_v1_lr1e4_e6_eval180/eval/validation_fastpath_t1024"
)
DEFAULT_OUTPUT_ROOT = Path("artifacts/detector_ablation/hybrid_first_lora/validation")


@dataclass(frozen=True)
class Wave4DetectorHybridEvalResult:
    status: str
    output_root: Path
    predictions_path: Path
    metrics_path: Path
    manifest_path: Path
    errors: list[str]


def run_detector_hybrid_evaluation(
    *,
    source_eval_root: Path | str = DEFAULT_SOURCE_EVAL_ROOT,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    variant: str = "hybrid_first_lora",
) -> Wave4DetectorHybridEvalResult:
    source_path = Path(source_eval_root)
    output_path = Path(output_root)
    predictions_path = output_path / "predictions.jsonl"
    metrics_path = output_path / "metrics.json"
    manifest_path = output_path / "manifest.json"
    source_manifest = _read_json(source_path / "manifest.json")
    source_predictions = _read_jsonl(source_path / "predictions.jsonl")
    if not source_predictions:
        raise ValueError(f"{source_path / 'predictions.jsonl'} contains no predictions")

    predictions = [_hybrid_prediction(record) for record in source_predictions]
    metrics = _score_predictions(predictions, split=source_manifest.get("split", source_predictions[0]["split"]))
    metrics["hybrid_guard_counts"] = dict(
        sorted(Counter(action for prediction in predictions for action in prediction.get("guard_actions", [])).items())
    )
    manifest = {
        "status": "passed",
        "variant": variant,
        "source_eval_root": str(source_path),
        "source_manifest": source_manifest,
        "split": metrics["split"],
        "predictions": "predictions.jsonl",
        "metrics": "metrics.json",
        "hybrid_policy": {
            "semantic_prediction_source": "model_raw_completion_from_source_eval",
            "schema_validation": True,
            "evidence_reference_validation": True,
            "evidence_ref_aux_field_stripping": True,
            "signal_aux_field_stripping": True,
            "unambiguous_evidence_ref_type_repair": True,
            "unambiguous_evidence_ref_id_suffix_repair": True,
            "invalid_optional_tool_result_id_removal": True,
            "ambiguous_or_unrepairable_outputs_remain_invalid": True,
            "support_level_or_severity_overrides": False,
        },
    }
    output_path.mkdir(parents=True, exist_ok=True)
    _write_jsonl(predictions_path, predictions)
    _write_json(metrics_path, metrics)
    _write_json(manifest_path, manifest)
    return Wave4DetectorHybridEvalResult(
        status="passed",
        output_root=output_path,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        manifest_path=manifest_path,
        errors=[],
    )


def _hybrid_prediction(source_record: dict[str, Any]) -> dict[str, Any]:
    prediction = _base_hybrid_record(source_record)
    packet = source_record["model_visible_input"]["data"]
    raw_completion = source_record.get("raw_completion")
    if not isinstance(raw_completion, str) or not raw_completion.strip():
        prediction["prediction_status"] = "invalid"
        prediction["invalid_reason_codes"] = ["missing_raw_completion"]
        return prediction
    try:
        assessment = _parse_assessment(raw_completion)
        guard_actions = _apply_contract_guard(assessment, packet)
        completion = json.dumps(assessment, ensure_ascii=False, sort_keys=True)
        validated = parse_and_validate_detector_assessment(completion, packet)
    except Exception as error:
        prediction["prediction_status"] = "invalid"
        prediction["invalid_reason_codes"] = [_error_reason_code(error)]
        return prediction
    prediction["prediction_status"] = "accepted"
    prediction["schema_valid"] = True
    prediction["evidence_valid"] = True
    prediction["prediction"] = validated
    prediction["predicted_support_level"] = validated["support_level"]
    prediction["predicted_severity"] = validated["severity"]
    prediction["support_level_exact_match"] = validated["support_level"] == source_record["gold_support_level"]
    prediction["severity_exact_match"] = validated["severity"] == source_record["gold_severity"]
    prediction["assessment_exact_match"] = False
    prediction["guard_actions"] = guard_actions
    return prediction


def _base_hybrid_record(source_record: dict[str, Any]) -> dict[str, Any]:
    return {
        "example_id": source_record["example_id"],
        "split": source_record["split"],
        "packet_id": source_record["packet_id"],
        "risk_category": source_record["risk_category"],
        "gold_support_level": source_record["gold_support_level"],
        "gold_severity": source_record["gold_severity"],
        "prediction_status": "not_attempted",
        "schema_valid": False,
        "evidence_valid": False,
        "model_visible_input": source_record["model_visible_input"],
        "source_prediction_status": source_record.get("prediction_status"),
        "source_invalid_reason_codes": source_record.get("invalid_reason_codes", []),
        "raw_completion": source_record.get("raw_completion"),
        "guard_actions": [],
    }


def _parse_assessment(raw_completion: str) -> dict[str, Any]:
    parsed = json.loads(normalize_json_content(raw_completion))
    if not isinstance(parsed, dict):
        raise ValueError("wrong_top_level_structure")
    if parsed.get("type") == "DetectorAssessment" and isinstance(parsed.get("data"), dict):
        parsed = parsed["data"]
    return parsed


def _apply_contract_guard(assessment: dict[str, Any], packet: dict[str, Any]) -> list[str]:
    return apply_sft_detector_contract_guard(assessment, packet)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
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
    parser = argparse.ArgumentParser(description="Apply deterministic hybrid guards to detector prediction artifacts.")
    parser.add_argument("--source-eval-root", default=str(DEFAULT_SOURCE_EVAL_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--variant", default="hybrid_first_lora")
    args = parser.parse_args(argv)
    result = run_detector_hybrid_evaluation(
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
