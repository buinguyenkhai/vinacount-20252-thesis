from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_RISK_CATEGORY = "earnings_cashflow_mismatch"


@dataclass(frozen=True)
class DetectorCorroborationDecisionResult:
    status: str
    output_dir: Path
    decision_json_path: Path
    decision_markdown_path: Path
    errors: list[str]


def run_detector_corroboration_decision(
    *,
    baseline_validation_root: Path | str,
    candidate_validation_root: Path | str,
    output_dir: Path | str,
    risk_category: str = DEFAULT_RISK_CATEGORY,
    min_weak_recall: float = 0.70,
    min_supported_recall: float = 0.70,
    min_boundary_macro_f1: float = 0.70,
    max_overall_macro_f1_drop: float = 0.01,
    max_valid_output_rate_drop: float = 0.01,
    max_fp_rejection_drop: float = 0.01,
    max_invalid_evidence_ref_rate_increase: float = 0.01,
) -> DetectorCorroborationDecisionResult:
    output_path = Path(output_dir)
    baseline = _summary(Path(baseline_validation_root), risk_category=risk_category)
    candidate = _summary(Path(candidate_validation_root), risk_category=risk_category)
    checks = [
        _minimum_check("weak_recall", candidate["weak_recall"], min_weak_recall),
        _minimum_check("supported_recall", candidate["supported_recall"], min_supported_recall),
        _minimum_check("boundary_macro_f1", candidate["boundary_macro_f1"], min_boundary_macro_f1),
        _drop_check(
            "overall_support_macro_f1_drop",
            candidate["overall_support_macro_f1"],
            baseline["overall_support_macro_f1"],
            max_overall_macro_f1_drop,
        ),
        _drop_check(
            "valid_output_rate_drop",
            candidate["valid_output_rate"],
            baseline["valid_output_rate"],
            max_valid_output_rate_drop,
        ),
        _drop_check(
            "fp_rejection_drop",
            candidate["fp_rejection_rate"],
            baseline["fp_rejection_rate"],
            max_fp_rejection_drop,
        ),
        _increase_check(
            "invalid_evidence_ref_rate_increase",
            candidate["invalid_evidence_ref_rate"],
            baseline["invalid_evidence_ref_rate"],
            max_invalid_evidence_ref_rate_increase,
        ),
    ]
    recommendation = "candidate_requires_reserved_test_confirmation" if all(
        check["passed"] for check in checks
    ) else "keep_baseline"
    decision = {
        "status": "passed",
        "decision_type": "detector_corroboration_calibration_gate",
        "risk_category": risk_category,
        "recommendation": recommendation,
        "baseline_validation": baseline,
        "candidate_validation": candidate,
        "checks": checks,
        "selection_boundary": {
            "combined_synthetic_validation_used_for_selection": True,
            "real_manual_metrics_used_for_gate": False,
            "reserved_synthetic_test_used": False,
        },
    }
    output_path.mkdir(parents=True, exist_ok=True)
    decision_json_path = output_path / "decision.json"
    decision_markdown_path = output_path / "decision.md"
    _write_json(decision_json_path, decision)
    decision_markdown_path.write_text(_render_markdown(decision), encoding="utf-8")
    return DetectorCorroborationDecisionResult(
        "passed",
        output_path,
        decision_json_path,
        decision_markdown_path,
        [],
    )


def _summary(root: Path, *, risk_category: str) -> dict[str, Any]:
    metrics = _read_json(root / "metrics.json")
    category = metrics.get("support_metrics_by_risk_category", {}).get(risk_category)
    if not isinstance(category, dict):
        raise ValueError(f"metrics do not contain support metrics for risk category {risk_category}: {root}")
    class_metrics = category["support_level_class_metrics"]
    weak = class_metrics["weakly_supported"]
    supported = class_metrics["supported"]
    boundary_f1_values = [
        value
        for value in [weak.get("f1"), supported.get("f1")]
        if value is not None
    ]
    num_examples = metrics["num_examples"]
    invalid_evidence_refs = sum(
        count
        for reason, count in metrics.get("invalid_reason_counts", {}).items()
        if reason.startswith("invalid_evidence")
    )
    return {
        "artifact_root": str(root),
        "num_examples": num_examples,
        "category_examples": category["num_examples"],
        "weak_recall": weak.get("recall"),
        "weak_precision": weak.get("precision"),
        "weak_f1": weak.get("f1"),
        "supported_recall": supported.get("recall"),
        "supported_precision": supported.get("precision"),
        "supported_f1": supported.get("f1"),
        "boundary_macro_f1": (
            sum(boundary_f1_values) / len(boundary_f1_values)
            if boundary_f1_values
            else None
        ),
        "overall_support_macro_f1": metrics.get("support_level_macro_f1"),
        "valid_output_rate": _rate(metrics.get("num_valid_predictions", 0), num_examples),
        "fp_rejection_rate": metrics.get("false_positive_rejection", {}).get("rate"),
        "invalid_evidence_ref_rate": _rate(invalid_evidence_refs, num_examples),
    }


def _minimum_check(name: str, value: float | None, threshold: float) -> dict[str, Any]:
    return {
        "name": name,
        "value": value,
        "threshold": threshold,
        "passed": value is not None and value >= threshold,
    }


def _drop_check(
    name: str,
    candidate: float | None,
    baseline: float | None,
    threshold: float,
) -> dict[str, Any]:
    drop = None if candidate is None or baseline is None else baseline - candidate
    return {
        "name": name,
        "candidate": candidate,
        "baseline": baseline,
        "drop": drop,
        "threshold": threshold,
        "passed": drop is not None and drop <= threshold,
    }


def _increase_check(
    name: str,
    candidate: float | None,
    baseline: float | None,
    threshold: float,
) -> dict[str, Any]:
    increase = None if candidate is None or baseline is None else candidate - baseline
    return {
        "name": name,
        "candidate": candidate,
        "baseline": baseline,
        "increase": increase,
        "threshold": threshold,
        "passed": increase is not None and increase <= threshold,
    }


def _render_markdown(decision: dict[str, Any]) -> str:
    baseline = decision["baseline_validation"]
    candidate = decision["candidate_validation"]
    lines = [
        "# Detector Corroboration Calibration Decision",
        "",
        f"Recommendation: **{decision['recommendation']}**.",
        "",
        "| Model | Weak recall | Weak F1 | Supported recall | Supported F1 | Boundary macro F1 | Overall macro F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        _summary_row("Baseline", baseline),
        _summary_row("Candidate", candidate),
        "",
        "Checks:",
    ]
    lines.extend(
        f"- {check['name']}: {'pass' if check['passed'] else 'fail'}"
        for check in decision["checks"]
    )
    lines.extend(
        [
            "",
            "Boundary:",
            "- Selection uses the combined synthetic validation corpus.",
            "- Real/manual diagnostics are not used for replacement.",
            "- The reserved synthetic test remains untouched until every validation check passes.",
        ]
    )
    return "\n".join(lines) + "\n"


def _summary_row(label: str, summary: dict[str, Any]) -> str:
    return (
        f"| {label} | {_fmt(summary['weak_recall'])} | {_fmt(summary['weak_f1'])} | "
        f"{_fmt(summary['supported_recall'])} | {_fmt(summary['supported_f1'])} | "
        f"{_fmt(summary['boundary_macro_f1'])} | {_fmt(summary['overall_support_macro_f1'])} |"
    )


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the detector corroboration calibration replacement gate.")
    parser.add_argument("--baseline-validation-root", required=True)
    parser.add_argument("--candidate-validation-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--risk-category", default=DEFAULT_RISK_CATEGORY)
    args = parser.parse_args()
    result = run_detector_corroboration_decision(
        baseline_validation_root=args.baseline_validation_root,
        candidate_validation_root=args.candidate_validation_root,
        output_dir=args.output_dir,
        risk_category=args.risk_category,
    )
    print(
        json.dumps(
            {
                "status": result.status,
                "decision_json_path": str(result.decision_json_path),
                "errors": result.errors,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
