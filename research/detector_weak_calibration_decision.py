from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_ROOT = Path("artifacts/detector_weak_calibration/decision")


@dataclass(frozen=True)
class WeakCalibrationDecisionResult:
    status: str
    output_root: Path
    decision_json_path: Path
    decision_markdown_path: Path
    errors: list[str]


def run_detector_weak_calibration_decision(
    *,
    baseline_validation_root: Path | str,
    candidate_validation_root: Path | str,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    baseline_label: str = "Qwen3.5-4B-Detector-LoRA-Guard",
    candidate_label: str = "Qwen3.5-4B-Detector-LoRA-WeakCal",
    baseline_test_root: Path | str | None = None,
    candidate_test_root: Path | str | None = None,
    min_weak_f1_delta: float = 0.05,
    max_support_macro_f1_drop: float = 0.01,
    max_valid_output_rate_drop: float = 0.01,
    max_fp_rejection_drop: float = 0.01,
    max_invalid_evidence_ref_rate_increase: float = 0.01,
    test_weak_f1_tolerance: float = 0.03,
) -> WeakCalibrationDecisionResult:
    output_path = Path(output_root)
    baseline_validation = _metrics_summary(Path(baseline_validation_root), split_role="validation")
    candidate_validation = _metrics_summary(Path(candidate_validation_root), split_role="validation")
    validation_checks = _validation_checks(
        baseline=baseline_validation,
        candidate=candidate_validation,
        min_weak_f1_delta=min_weak_f1_delta,
        max_support_macro_f1_drop=max_support_macro_f1_drop,
        max_valid_output_rate_drop=max_valid_output_rate_drop,
        max_fp_rejection_drop=max_fp_rejection_drop,
        max_invalid_evidence_ref_rate_increase=max_invalid_evidence_ref_rate_increase,
    )

    test_checks: list[dict[str, Any]] = []
    baseline_test = None
    candidate_test = None
    if baseline_test_root is not None or candidate_test_root is not None:
        if baseline_test_root is None or candidate_test_root is None:
            raise ValueError("baseline_test_root and candidate_test_root must be provided together")
        baseline_test = _metrics_summary(Path(baseline_test_root), split_role="test")
        candidate_test = _metrics_summary(Path(candidate_test_root), split_role="test")
        test_checks = _test_checks(
            baseline=baseline_test,
            candidate=candidate_test,
            candidate_validation=candidate_validation,
            max_support_macro_f1_drop=max_support_macro_f1_drop,
            max_valid_output_rate_drop=max_valid_output_rate_drop,
            max_fp_rejection_drop=max_fp_rejection_drop,
            max_invalid_evidence_ref_rate_increase=max_invalid_evidence_ref_rate_increase,
            test_weak_f1_tolerance=test_weak_f1_tolerance,
        )

    validation_passed = all(check["passed"] for check in validation_checks)
    test_passed = all(check["passed"] for check in test_checks) if test_checks else None
    if not validation_passed:
        recommendation = "keep_baseline"
    elif test_passed is None:
        recommendation = "candidate_requires_synthetic_test_confirmation"
    elif test_passed:
        recommendation = "replace_baseline"
    else:
        recommendation = "keep_baseline"

    decision = {
        "status": "passed",
        "decision_type": "detector_weak_calibration_gate",
        "baseline_label": baseline_label,
        "candidate_label": candidate_label,
        "recommendation": recommendation,
        "baseline_validation": baseline_validation,
        "candidate_validation": candidate_validation,
        "baseline_test": baseline_test,
        "candidate_test": candidate_test,
        "validation_checks": validation_checks,
        "test_checks": test_checks,
        "thresholds": {
            "min_weak_f1_delta": min_weak_f1_delta,
            "max_support_macro_f1_drop": max_support_macro_f1_drop,
            "max_valid_output_rate_drop": max_valid_output_rate_drop,
            "max_fp_rejection_drop": max_fp_rejection_drop,
            "max_invalid_evidence_ref_rate_increase": max_invalid_evidence_ref_rate_increase,
            "test_weak_f1_tolerance": test_weak_f1_tolerance,
        },
        "selection_boundary": {
            "real_manual_metrics_used_for_gate": False,
            "synthetic_validation_used_for_selection": True,
            "synthetic_test_used_only_after_validation_gate": baseline_test is not None,
        },
    }
    output_path.mkdir(parents=True, exist_ok=True)
    decision_json_path = output_path / "decision.json"
    decision_markdown_path = output_path / "decision.md"
    _write_json(decision_json_path, decision)
    decision_markdown_path.write_text(_render_markdown(decision), encoding="utf-8")
    return WeakCalibrationDecisionResult(
        status="passed",
        output_root=output_path,
        decision_json_path=decision_json_path,
        decision_markdown_path=decision_markdown_path,
        errors=[],
    )


def _metrics_summary(root: Path, *, split_role: str) -> dict[str, Any]:
    metrics = _read_json(root / "metrics.json")
    num_examples = metrics["num_examples"]
    invalid_evidence_ref_count = sum(
        count
        for reason, count in metrics.get("invalid_reason_counts", {}).items()
        if reason.startswith("invalid_evidence")
    )
    return {
        "artifact_root": str(root),
        "split_role": split_role,
        "num_examples": num_examples,
        "valid_output_rate": _rate(metrics.get("num_valid_predictions", 0), num_examples),
        "support_accuracy": metrics.get("support_level_exact_match_rate"),
        "support_macro_f1": metrics.get("support_level_macro_f1"),
        "weak_f1": _weak_metric(metrics, "f1"),
        "weak_precision": _weak_metric(metrics, "precision"),
        "weak_recall": _weak_metric(metrics, "recall"),
        "fp_rejection_rate": metrics.get("false_positive_rejection", {}).get("rate"),
        "invalid_evidence_ref_rate": _rate(invalid_evidence_ref_count, num_examples),
    }


def _validation_checks(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    min_weak_f1_delta: float,
    max_support_macro_f1_drop: float,
    max_valid_output_rate_drop: float,
    max_fp_rejection_drop: float,
    max_invalid_evidence_ref_rate_increase: float,
) -> list[dict[str, Any]]:
    return [
        _check_delta_at_least(
            "weak_f1_delta",
            candidate.get("weak_f1"),
            baseline.get("weak_f1"),
            min_weak_f1_delta,
        ),
        _check_drop_at_most(
            "support_macro_f1_drop",
            candidate.get("support_macro_f1"),
            baseline.get("support_macro_f1"),
            max_support_macro_f1_drop,
        ),
        _check_drop_at_most(
            "valid_output_rate_drop",
            candidate.get("valid_output_rate"),
            baseline.get("valid_output_rate"),
            max_valid_output_rate_drop,
        ),
        _check_drop_at_most(
            "fp_rejection_drop",
            candidate.get("fp_rejection_rate"),
            baseline.get("fp_rejection_rate"),
            max_fp_rejection_drop,
        ),
        _check_increase_at_most(
            "invalid_evidence_ref_rate_increase",
            candidate.get("invalid_evidence_ref_rate"),
            baseline.get("invalid_evidence_ref_rate"),
            max_invalid_evidence_ref_rate_increase,
        ),
    ]


def _test_checks(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    candidate_validation: dict[str, Any],
    max_support_macro_f1_drop: float,
    max_valid_output_rate_drop: float,
    max_fp_rejection_drop: float,
    max_invalid_evidence_ref_rate_increase: float,
    test_weak_f1_tolerance: float,
) -> list[dict[str, Any]]:
    return [
        _check_drop_at_most(
            "test_weak_f1_vs_candidate_validation",
            candidate.get("weak_f1"),
            candidate_validation.get("weak_f1"),
            test_weak_f1_tolerance,
        ),
        _check_drop_at_most(
            "test_support_macro_f1_drop_vs_baseline",
            candidate.get("support_macro_f1"),
            baseline.get("support_macro_f1"),
            max_support_macro_f1_drop,
        ),
        _check_drop_at_most(
            "test_valid_output_rate_drop_vs_baseline",
            candidate.get("valid_output_rate"),
            baseline.get("valid_output_rate"),
            max_valid_output_rate_drop,
        ),
        _check_drop_at_most(
            "test_fp_rejection_drop_vs_baseline",
            candidate.get("fp_rejection_rate"),
            baseline.get("fp_rejection_rate"),
            max_fp_rejection_drop,
        ),
        _check_increase_at_most(
            "test_invalid_evidence_ref_rate_increase_vs_baseline",
            candidate.get("invalid_evidence_ref_rate"),
            baseline.get("invalid_evidence_ref_rate"),
            max_invalid_evidence_ref_rate_increase,
        ),
    ]


def _check_delta_at_least(name: str, candidate: float | None, baseline: float | None, threshold: float) -> dict[str, Any]:
    delta = _delta(candidate, baseline)
    return {
        "name": name,
        "candidate": candidate,
        "baseline": baseline,
        "delta": delta,
        "threshold": threshold,
        "passed": delta is not None and delta >= threshold,
    }


def _check_drop_at_most(name: str, candidate: float | None, baseline: float | None, threshold: float) -> dict[str, Any]:
    delta = _delta(candidate, baseline)
    drop = None if delta is None else -delta
    return {
        "name": name,
        "candidate": candidate,
        "baseline": baseline,
        "drop": drop,
        "threshold": threshold,
        "passed": drop is not None and drop <= threshold,
    }


def _check_increase_at_most(name: str, candidate: float | None, baseline: float | None, threshold: float) -> dict[str, Any]:
    increase = _delta(candidate, baseline)
    return {
        "name": name,
        "candidate": candidate,
        "baseline": baseline,
        "increase": increase,
        "threshold": threshold,
        "passed": increase is not None and increase <= threshold,
    }


def _weak_metric(metrics: dict[str, Any], field: str) -> float | None:
    if "weakly_supported_detection" in metrics:
        return metrics["weakly_supported_detection"].get(field)
    return metrics.get("support_level_class_metrics", {}).get("weakly_supported", {}).get(field)


def _delta(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None:
        return None
    return candidate - baseline


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _render_markdown(decision: dict[str, Any]) -> str:
    lines = [
        "# Weak Calibration Decision",
        "",
        f"Recommendation: **{decision['recommendation']}**.",
        "",
        "| Split | Model | Valid output | Support macro F1 | Weak F1 | Weak precision | Weak recall | FP rejection | Invalid evidence refs |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        _summary_row("validation", decision["baseline_label"], decision["baseline_validation"]),
        _summary_row("validation", decision["candidate_label"], decision["candidate_validation"]),
    ]
    if decision.get("baseline_test") and decision.get("candidate_test"):
        lines.extend(
            [
                _summary_row("test", decision["baseline_label"], decision["baseline_test"]),
                _summary_row("test", decision["candidate_label"], decision["candidate_test"]),
            ]
        )
    lines.extend(["", "Validation Checks:"])
    for check in decision["validation_checks"]:
        lines.append(f"- {check['name']}: {'pass' if check['passed'] else 'fail'}")
    if decision["test_checks"]:
        lines.extend(["", "Test Checks:"])
        for check in decision["test_checks"]:
            lines.append(f"- {check['name']}: {'pass' if check['passed'] else 'fail'}")
    lines.extend(
        [
            "",
            "Boundary:",
            "- Real/manual diagnostics are not used for this gate.",
            "- Synthetic validation selects the candidate; synthetic test is used only after validation passes.",
        ]
    )
    return "\n".join(lines) + "\n"


def _summary_row(split: str, label: str, metrics: dict[str, Any]) -> str:
    return (
        f"| {split} | {label} | {_fmt(metrics.get('valid_output_rate'))} | "
        f"{_fmt(metrics.get('support_macro_f1'))} | {_fmt(metrics.get('weak_f1'))} | "
        f"{_fmt(metrics.get('weak_precision'))} | {_fmt(metrics.get('weak_recall'))} | "
        f"{_fmt(metrics.get('fp_rejection_rate'))} | {_fmt(metrics.get('invalid_evidence_ref_rate'))} |"
    )


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply the Wave 4 weak-calibration replacement gate.")
    parser.add_argument("--baseline-validation-root", required=True)
    parser.add_argument("--candidate-validation-root", required=True)
    parser.add_argument("--baseline-test-root")
    parser.add_argument("--candidate-test-root")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--baseline-label", default="Qwen3.5-4B-Detector-LoRA-Guard")
    parser.add_argument("--candidate-label", default="Qwen3.5-4B-Detector-LoRA-WeakCal")
    args = parser.parse_args(argv)
    result = run_detector_weak_calibration_decision(
        baseline_validation_root=args.baseline_validation_root,
        candidate_validation_root=args.candidate_validation_root,
        baseline_test_root=args.baseline_test_root,
        candidate_test_root=args.candidate_test_root,
        output_root=args.output_root,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
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
