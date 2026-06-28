import json
import tempfile
import unittest
from pathlib import Path

from research.detector_weak_calibration_decision import (
    run_detector_weak_calibration_decision,
)


class Wave4DetectorWeakCalibrationDecisionTest(unittest.TestCase):
    def test_validation_gate_keeps_baseline_when_weak_f1_gain_is_too_small(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            baseline = root / "baseline"
            candidate = root / "candidate"
            _write_metrics(baseline, weak_f1=0.71, support_macro_f1=0.897)
            _write_metrics(candidate, weak_f1=0.74, support_macro_f1=0.900)

            result = run_detector_weak_calibration_decision(
                baseline_validation_root=baseline,
                candidate_validation_root=candidate,
                output_root=root / "decision",
            )

            decision = _read_json(result.decision_json_path)
            self.assertEqual(decision["recommendation"], "keep_baseline")
            self.assertFalse(decision["validation_checks"][0]["passed"])
            self.assertTrue((root / "decision" / "decision.md").exists())

    def test_validation_gate_requires_test_confirmation_after_validation_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            baseline = root / "baseline"
            candidate = root / "candidate"
            _write_metrics(baseline, weak_f1=0.71, support_macro_f1=0.897)
            _write_metrics(candidate, weak_f1=0.77, support_macro_f1=0.892)

            result = run_detector_weak_calibration_decision(
                baseline_validation_root=baseline,
                candidate_validation_root=candidate,
                output_root=root / "decision",
            )

            decision = _read_json(result.decision_json_path)
            self.assertEqual(decision["recommendation"], "candidate_requires_synthetic_test_confirmation")
            self.assertTrue(all(check["passed"] for check in decision["validation_checks"]))
            self.assertEqual(decision["test_checks"], [])

    def test_test_confirmation_can_replace_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            baseline_val = root / "baseline_val"
            candidate_val = root / "candidate_val"
            baseline_test = root / "baseline_test"
            candidate_test = root / "candidate_test"
            _write_metrics(baseline_val, weak_f1=0.71, support_macro_f1=0.897)
            _write_metrics(candidate_val, weak_f1=0.77, support_macro_f1=0.892)
            _write_metrics(baseline_test, weak_f1=0.70, support_macro_f1=0.890)
            _write_metrics(candidate_test, weak_f1=0.75, support_macro_f1=0.885)

            result = run_detector_weak_calibration_decision(
                baseline_validation_root=baseline_val,
                candidate_validation_root=candidate_val,
                baseline_test_root=baseline_test,
                candidate_test_root=candidate_test,
                output_root=root / "decision",
            )

            decision = _read_json(result.decision_json_path)
            self.assertEqual(decision["recommendation"], "replace_baseline")
            self.assertTrue(all(check["passed"] for check in decision["test_checks"]))


def _write_metrics(
    root: Path,
    *,
    weak_f1: float,
    support_macro_f1: float,
    valid: int = 100,
    num_examples: int = 100,
    fp_rejection: float = 1.0,
    invalid_evidence_refs: int = 0,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "metrics.json").write_text(
        json.dumps(
            {
                "num_examples": num_examples,
                "num_valid_predictions": valid,
                "support_level_exact_match_rate": 0.9,
                "support_level_macro_f1": support_macro_f1,
                "weakly_supported_detection": {
                    "f1": weak_f1,
                    "precision": 0.8,
                    "recall": 0.7,
                    "true_positive": 7,
                    "predicted_total": 9,
                    "gold_total": 10,
                },
                "false_positive_rejection": {"rate": fp_rejection, "correct_or_conservative": 10, "total": 10},
                "invalid_reason_counts": {"invalid_evidence_ids": invalid_evidence_refs},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
