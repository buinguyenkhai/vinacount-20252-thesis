import json
import tempfile
import unittest
from pathlib import Path

from research.detector_corroboration_decision import (
    run_detector_corroboration_decision,
)


class DetectorCorroborationDecisionTest(unittest.TestCase):
    def test_candidate_requires_test_confirmation_only_after_boundary_and_regression_gates_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            baseline = root / "baseline"
            candidate = root / "candidate"
            _write_metrics(baseline, weak_recall=0.20, weak_f1=0.30, supported_recall=0.90, supported_f1=0.85)
            _write_metrics(candidate, weak_recall=0.80, weak_f1=0.78, supported_recall=0.80, supported_f1=0.79)

            result = run_detector_corroboration_decision(
                baseline_validation_root=baseline,
                candidate_validation_root=candidate,
                output_dir=root / "decision",
            )

            decision = _read_json(result.decision_json_path)
            self.assertEqual(decision["recommendation"], "candidate_requires_reserved_test_confirmation")
            self.assertTrue(all(check["passed"] for check in decision["checks"]))

    def test_candidate_is_rejected_when_supported_recall_collapses(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            baseline = root / "baseline"
            candidate = root / "candidate"
            _write_metrics(baseline, weak_recall=0.20, weak_f1=0.30, supported_recall=0.90, supported_f1=0.85)
            _write_metrics(candidate, weak_recall=0.90, weak_f1=0.82, supported_recall=0.40, supported_f1=0.50)

            result = run_detector_corroboration_decision(
                baseline_validation_root=baseline,
                candidate_validation_root=candidate,
                output_dir=root / "decision",
            )

            decision = _read_json(result.decision_json_path)
            self.assertEqual(decision["recommendation"], "keep_baseline")
            supported_recall = next(check for check in decision["checks"] if check["name"] == "supported_recall")
            self.assertFalse(supported_recall["passed"])


def _write_metrics(
    root: Path,
    *,
    weak_recall: float,
    weak_f1: float,
    supported_recall: float,
    supported_f1: float,
) -> None:
    root.mkdir()
    payload = {
        "num_examples": 100,
        "num_valid_predictions": 100,
        "support_level_macro_f1": 0.90,
        "false_positive_rejection": {"rate": 1.0},
        "invalid_reason_counts": {},
        "support_metrics_by_risk_category": {
            "earnings_cashflow_mismatch": {
                "num_examples": 30,
                "support_level_class_metrics": {
                    "weakly_supported": {
                        "precision": weak_f1,
                        "recall": weak_recall,
                        "f1": weak_f1,
                    },
                    "supported": {
                        "precision": supported_f1,
                        "recall": supported_recall,
                        "f1": supported_f1,
                    },
                    "not_supported": {"precision": None, "recall": None, "f1": None},
                    "insufficient_evidence": {"precision": None, "recall": None, "f1": None},
                },
            }
        },
    }
    (root / "metrics.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
