from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from research.detector_contract_validation import (
    parse_and_validate_detector_assessment,
    validate_detector_packet,
)
from research.detector_sft_evaluator import (
    SUPPORTED_EVAL_SPLITS,
    _base_prediction_record,
    _canonical_json,
    _error_reason_code,
    _score_predictions,
)
from research.detector_sft_trainer import (
    CANONICAL_SFT_JSONL,
    load_detector_sft_dataset,
)


DEFAULT_OUTPUT_ROOT = Path("artifacts/detector_ablation/rule_only_detector_baseline/validation")


@dataclass(frozen=True)
class Wave4DetectorRuleBaselineResult:
    status: str
    output_root: Path
    predictions_path: Path
    metrics_path: Path
    manifest_path: Path
    errors: list[str]


def run_detector_rule_baseline(
    *,
    sft_jsonl: Path | str = CANONICAL_SFT_JSONL,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    split: Literal["validation", "test"] = "validation",
    limit: int | None = None,
    allow_noncanonical_sft_jsonl: bool = False,
) -> Wave4DetectorRuleBaselineResult:
    if split not in SUPPORTED_EVAL_SPLITS:
        raise ValueError(f"split must be one of {sorted(SUPPORTED_EVAL_SPLITS)}")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided")

    output_path = Path(output_root)
    predictions_path = output_path / "predictions.jsonl"
    metrics_path = output_path / "metrics.json"
    manifest_path = output_path / "manifest.json"
    dataset = load_detector_sft_dataset(
        sft_jsonl,
        allow_noncanonical_sft_jsonl=allow_noncanonical_sft_jsonl,
    )
    rows = dataset.validation_rows if split == "validation" else dataset.test_rows
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"no rows available for split={split}")

    predictions: list[dict[str, Any]] = []
    for row in rows:
        packet = json.loads(row.messages[1]["content"])
        gold = json.loads(row.messages[2]["content"])
        prediction_record = _base_prediction_record(row.example_id, split, packet, gold)
        try:
            validate_detector_packet(packet)
            assessment = _rule_only_assessment(packet)
            completion = json.dumps(assessment, ensure_ascii=False, sort_keys=True)
            prediction_record["raw_completion"] = completion
            validated = parse_and_validate_detector_assessment(completion, packet)
        except Exception as error:
            prediction_record["prediction_status"] = "invalid"
            prediction_record["schema_valid"] = False
            prediction_record["evidence_valid"] = False
            prediction_record["invalid_reason_codes"] = [_error_reason_code(error)]
        else:
            prediction_record["prediction_status"] = "accepted"
            prediction_record["schema_valid"] = True
            prediction_record["evidence_valid"] = True
            prediction_record["prediction"] = validated
            prediction_record["predicted_support_level"] = validated["support_level"]
            prediction_record["predicted_severity"] = validated["severity"]
            prediction_record["support_level_exact_match"] = validated["support_level"] == gold["support_level"]
            prediction_record["severity_exact_match"] = validated["severity"] == gold["severity"]
            prediction_record["assessment_exact_match"] = _canonical_json(validated) == _canonical_json(gold)
        predictions.append(prediction_record)

    metrics = _score_predictions(predictions, split=split)
    manifest = {
        "status": "passed",
        "variant": "rule_only_detector_baseline",
        "source_jsonl": str(sft_jsonl),
        "split": split,
        "limit": limit,
        "metric_version": "detector_ablation_metrics_v1",
        "predictions": "predictions.jsonl",
        "metrics": "metrics.json",
        "rule_policy": {
            "semantic_model_used": False,
            "schema_validation": True,
            "evidence_reference_validation": True,
            "supported_when_flagged_tool_signal_has_visible_evidence": True,
            "insufficient_evidence_when_no_visible_candidate_evidence": True,
            "weakly_supported_fallback_for_partial_visible_evidence": True,
        },
        "evaluation_policy": {
            "gold_source": "assistant message in official detector_sft_chat.jsonl",
            "test_split_used_for_training_or_selection": False,
            "predictions_are_evaluation_outputs_only": True,
        },
    }
    output_path.mkdir(parents=True, exist_ok=True)
    _write_jsonl(predictions_path, predictions)
    _write_json(metrics_path, metrics)
    _write_json(manifest_path, manifest)
    return Wave4DetectorRuleBaselineResult(
        status="passed",
        output_root=output_path,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        manifest_path=manifest_path,
        errors=[],
    )


def _rule_only_assessment(packet: dict[str, Any]) -> dict[str, Any]:
    support_level = _rule_support_level(packet)
    cited_evidence_refs = _rule_cited_evidence_refs(packet, support_level=support_level)
    signal_id = _primary_signal_id(packet)
    return {
        "assessment_id": f"RULE_ASSESS_{packet['packet_id']}",
        "packet_id": packet["packet_id"],
        "candidate_id": packet["candidate_id"],
        "report_id": packet["report_id"],
        "risk_category": packet["task"]["risk_category"],
        "support_level": support_level,
        "confidence": _rule_confidence(support_level),
        "severity": _rule_severity(packet, support_level=support_level),
        "validated_signals": [
            {
                "signal_id": signal_id,
                "status": _signal_status(support_level),
                "support_level": support_level,
                "cited_evidence_refs": cited_evidence_refs,
            }
        ],
        "cited_evidence_refs": cited_evidence_refs,
        "rationale_short": _rule_rationale(support_level),
    }


def _rule_support_level(packet: dict[str, Any]) -> str:
    if not _candidate_visible_evidence_refs(packet):
        return "insufficient_evidence"
    if any(finding.get("flag") and finding.get("evidence_refs") for finding in packet.get("tool_findings", [])):
        return "supported"
    if packet.get("tool_findings"):
        return "not_supported"
    return "weakly_supported"


def _rule_cited_evidence_refs(packet: dict[str, Any], *, support_level: str) -> list[dict[str, str]]:
    if support_level == "insufficient_evidence":
        rule_id = packet["rules"][0]["rule_id"]
        return [{"evidence_ref_type": "rule", "ref_id": rule_id, "role": "missing_required_context"}]
    for finding in packet.get("tool_findings", []):
        for evidence_ref in finding.get("evidence_refs", []):
            return [dict(evidence_ref)]
    return [dict(packet["candidate_summary"]["evidence_refs"][0])]


def _candidate_visible_evidence_refs(packet: dict[str, Any]) -> list[dict[str, str]]:
    refs = packet.get("candidate_summary", {}).get("evidence_refs", [])
    return [ref for ref in refs if isinstance(ref, dict) and ref.get("ref_id")]


def _primary_signal_id(packet: dict[str, Any]) -> str:
    for finding in packet.get("tool_findings", []):
        if finding.get("signal_id"):
            return finding["signal_id"]
    for signal_id in packet.get("candidate_summary", {}).get("supporting_signal_ids", []):
        if signal_id:
            return signal_id
    return packet["rules"][0]["related_signal_ids"][0]


def _rule_severity(packet: dict[str, Any], *, support_level: str) -> str:
    if support_level in {"not_supported", "insufficient_evidence"}:
        return "unknown"
    priority = packet.get("candidate_summary", {}).get("priority")
    if priority in {"high", "medium", "low"}:
        return priority
    return "medium"


def _rule_confidence(support_level: str) -> float:
    if support_level == "supported":
        return 0.65
    if support_level == "weakly_supported":
        return 0.45
    return 0.5


def _signal_status(support_level: str) -> str:
    return {
        "supported": "validated",
        "weakly_supported": "partially_validated",
        "not_supported": "rejected",
        "insufficient_evidence": "not_assessable",
    }[support_level]


def _rule_rationale(support_level: str) -> str:
    return {
        "supported": "Rule baseline found a flagged tool signal with visible packet evidence.",
        "weakly_supported": "Rule baseline found visible evidence but no deterministic confirming signal.",
        "not_supported": "Rule baseline found visible evidence but no flagged supporting signal.",
        "insufficient_evidence": "Rule baseline found no visible candidate evidence to assess the risk.",
    }[support_level]


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
    parser = argparse.ArgumentParser(description="Evaluate the Wave 4 rule-only detector baseline.")
    parser.add_argument("--sft-jsonl", default=str(CANONICAL_SFT_JSONL))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--split", choices=sorted(SUPPORTED_EVAL_SPLITS), default="validation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow-noncanonical-sft-jsonl", action="store_true")
    args = parser.parse_args(argv)
    result = run_detector_rule_baseline(
        sft_jsonl=args.sft_jsonl,
        output_root=args.output_root,
        split=args.split,
        limit=args.limit,
        allow_noncanonical_sft_jsonl=args.allow_noncanonical_sft_jsonl,
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
