import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from research.detector_hybrid_evaluator import run_detector_hybrid_evaluation


class Wave4DetectorHybridEvaluatorTest(unittest.TestCase):
    def test_hybrid_repairs_unambiguous_evidence_ref_type_and_counts_guard_action(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_root = root / "source_eval"
            output_root = root / "hybrid_eval"
            packet = _packet("repair-1")
            _write_source_eval(
                source_root,
                predictions=[
                    _invalid_prediction(
                        "repair-1",
                        packet=packet,
                        gold_support_level="weakly_supported",
                        gold_severity="medium",
                        raw_completion=_assessment_json(
                            packet,
                            support_level="weakly_supported",
                            severity="medium",
                            evidence_ref_type="span",
                            ref_id="REPORT_1:span-1",
                        ),
                    )
                ],
            )

            result = run_detector_hybrid_evaluation(
                source_eval_root=source_root,
                output_root=output_root,
                variant="hybrid_fixture",
            )

            self.assertEqual(result.status, "passed", result.errors)
            predictions = _read_jsonl(result.predictions_path)
            metrics = _read_json(result.metrics_path)
            manifest = _read_json(result.manifest_path)
            self.assertEqual(predictions[0]["prediction_status"], "accepted")
            self.assertEqual(predictions[0]["guard_actions"], ["repaired_evidence_ref_type"])
            self.assertEqual(
                predictions[0]["prediction"]["cited_evidence_refs"],
                [{"evidence_ref_type": "variance_explanation_span", "ref_id": "REPORT_1:span-1", "role": "context"}],
            )
            self.assertEqual(metrics["num_valid_predictions"], 1)
            self.assertEqual(metrics["num_invalid_responses"], 0)
            self.assertEqual(metrics["hybrid_guard_counts"], {"repaired_evidence_ref_type": 1})
            self.assertEqual(metrics["support_level_exact_match_rate"], 1.0)
            self.assertEqual(metrics["assessment_exact_match_count"], 0)
            self.assertEqual(manifest["variant"], "hybrid_fixture")
            self.assertEqual(manifest["source_eval_root"], str(source_root))

    def test_hybrid_strips_aux_ref_fields_and_repairs_unique_local_ref_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_root = root / "source_eval"
            output_root = root / "hybrid_eval"
            packet = _packet("repair-local-id")
            _write_source_eval(
                source_root,
                predictions=[
                    _invalid_prediction(
                        "repair-local-id",
                        packet=packet,
                        gold_support_level="supported",
                        gold_severity="medium",
                        raw_completion=json.dumps(
                            {
                                "assessment_id": f"ASSESS_{packet['packet_id']}",
                                "packet_id": packet["packet_id"],
                                "candidate_id": packet["candidate_id"],
                                "report_id": packet["report_id"],
                                "risk_category": packet["task"]["risk_category"],
                                "support_level": "supported",
                                "confidence": 0.8,
                                "severity": "medium",
                                "validated_signals": [
                                    {
                                        "signal_id": "signal-1",
                                        "status": "validated",
                                        "support_level": "supported",
                                        "tool_result_id": "not-visible-tool",
                                        "value": 27.83,
                                        "cited_evidence_refs": [
                                            {
                                                "evidence_ref_type": "variance_explanation_span",
                                                "local_evidence_id": "span-1",
                                                "ref_id": "REPORT_1:TABLE_1:span-1",
                                                "report_id": "REPORT_1",
                                                "role": "context",
                                            }
                                        ],
                                    }
                                ],
                                "cited_evidence_refs": [
                                    {
                                        "evidence_ref_type": "variance_explanation_span",
                                        "local_evidence_id": "span-1",
                                        "ref_id": "REPORT_1:TABLE_1:span-1",
                                        "report_id": "REPORT_1",
                                        "role": "context",
                                    }
                                ],
                                "rationale_short": "Fixture assessment.",
                            },
                            sort_keys=True,
                        ),
                    )
                ],
            )

            result = run_detector_hybrid_evaluation(
                source_eval_root=source_root,
                output_root=output_root,
                variant="hybrid_fixture",
            )

            self.assertEqual(result.status, "passed", result.errors)
            predictions = _read_jsonl(result.predictions_path)
            metrics = _read_json(result.metrics_path)
            self.assertEqual(predictions[0]["prediction_status"], "accepted")
            self.assertEqual(
                predictions[0]["prediction"]["cited_evidence_refs"],
                [{"evidence_ref_type": "variance_explanation_span", "ref_id": "REPORT_1:span-1", "role": "context"}],
            )
            self.assertNotIn("tool_result_id", predictions[0]["prediction"]["validated_signals"][0])
            self.assertEqual(
                metrics["hybrid_guard_counts"],
                {
                    "removed_invalid_optional_tool_result_id": 1,
                    "repaired_evidence_ref_id": 1,
                    "stripped_signal_aux_fields": 1,
                    "stripped_evidence_ref_aux_fields": 1,
                },
            )
            self.assertEqual(metrics["support_level_exact_match_rate"], 1.0)

    def test_public_command_writes_hybrid_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_root = root / "source_eval"
            output_root = root / "hybrid_eval"
            packet = _packet("accepted-1")
            _write_source_eval(
                source_root,
                predictions=[
                    _accepted_prediction(
                        "accepted-1",
                        packet=packet,
                        gold_support_level="supported",
                        gold_severity="medium",
                        raw_completion=_assessment_json(
                            packet,
                            support_level="supported",
                            severity="medium",
                            evidence_ref_type="variance_explanation_span",
                            ref_id="REPORT_1:span-1",
                        ),
                    )
                ],
            )

            completed = _run_hybrid_command(
                "--source-eval-root",
                str(source_root),
                "--output-root",
                str(output_root),
                "--variant",
                "hybrid_fixture",
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(completed.stdout)["status"], "passed")
            self.assertTrue((output_root / "predictions.jsonl").exists())
            self.assertTrue((output_root / "metrics.json").exists())
            self.assertTrue((output_root / "manifest.json").exists())


def _write_source_eval(path: Path, *, predictions: list[dict]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "manifest.json").write_text(
        json.dumps({"status": "passed", "split": "validation", "predictions": "predictions.jsonl"}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (path / "metrics.json").write_text(
        json.dumps({"status": "source_fixture"}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (path / "predictions.jsonl").write_text(
        "".join(json.dumps(prediction, sort_keys=True) + "\n" for prediction in predictions),
        encoding="utf-8",
    )


def _packet(example_id: str) -> dict:
    return {
        "packet_id": f"PACKET_{example_id}",
        "candidate_id": f"CAND_{example_id}",
        "report_id": "REPORT_1",
        "task": {"risk_category": "asset_quality_valuation_risk"},
        "metadata": {},
        "candidate_summary": {
            "priority": "medium",
            "reason_for_candidate": "Fixture candidate.",
            "supporting_signal_ids": ["signal-1"],
            "evidence_refs": [{"evidence_ref_type": "variance_explanation_span", "ref_id": "REPORT_1:span-1", "role": "context"}],
        },
        "relevant_table_rows": [],
        "relevant_notes": [],
        "relevant_variance_explanations": [
            {"report_id": "REPORT_1", "span_id": "span-1", "text": "Management context."}
        ],
        "tool_findings": [
            {
                "tool_result_id": "tool-1",
                "report_id": "REPORT_1",
                "tool_name": "fixture_tool",
                "risk_category": "asset_quality_valuation_risk",
                "signal_id": "signal-1",
                "flag": True,
                "finding_summary": "Fixture signal.",
                "evidence_refs": [{"evidence_ref_type": "variance_explanation_span", "ref_id": "REPORT_1:span-1", "role": "context"}],
            }
        ],
        "rules": [{"rule_id": "rule-1", "related_signal_ids": ["signal-1"]}],
        "constraints": {},
    }


def _accepted_prediction(
    example_id: str,
    *,
    packet: dict,
    gold_support_level: str,
    gold_severity: str,
    raw_completion: str,
) -> dict:
    return {
        **_prediction_base(example_id, packet, gold_support_level=gold_support_level, gold_severity=gold_severity),
        "prediction_status": "accepted",
        "raw_completion": raw_completion,
    }


def _invalid_prediction(
    example_id: str,
    *,
    packet: dict,
    gold_support_level: str,
    gold_severity: str,
    raw_completion: str,
) -> dict:
    return {
        **_prediction_base(example_id, packet, gold_support_level=gold_support_level, gold_severity=gold_severity),
        "prediction_status": "invalid",
        "raw_completion": raw_completion,
        "invalid_reason_codes": ["invalid_evidence_ref_type"],
    }


def _prediction_base(example_id: str, packet: dict, *, gold_support_level: str, gold_severity: str) -> dict:
    return {
        "example_id": example_id,
        "split": "validation",
        "packet_id": packet["packet_id"],
        "risk_category": packet["task"]["risk_category"],
        "gold_support_level": gold_support_level,
        "gold_severity": gold_severity,
        "model_visible_input": {"type": "DetectorPacket", "data": packet},
    }


def _assessment_json(
    packet: dict,
    *,
    support_level: str,
    severity: str,
    evidence_ref_type: str,
    ref_id: str,
) -> str:
    return json.dumps(
        {
            "assessment_id": f"ASSESS_{packet['packet_id']}",
            "packet_id": packet["packet_id"],
            "candidate_id": packet["candidate_id"],
            "report_id": packet["report_id"],
            "risk_category": packet["task"]["risk_category"],
            "support_level": support_level,
            "confidence": 0.8,
            "severity": severity,
            "validated_signals": [
                {
                    "signal_id": "signal-1",
                    "status": "partially_validated" if support_level == "weakly_supported" else "validated",
                    "support_level": support_level,
                    "tool_result_id": "tool-1",
                    "cited_evidence_refs": [{"evidence_ref_type": evidence_ref_type, "ref_id": ref_id, "role": "context"}],
                }
            ],
            "cited_evidence_refs": [{"evidence_ref_type": evidence_ref_type, "ref_id": ref_id, "role": "context"}],
            "rationale_short": "Fixture assessment.",
        },
        sort_keys=True,
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _run_hybrid_command(*args: str) -> subprocess.CompletedProcess:
    repo_root = Path(__file__).resolve().parents[3]
    pythonpath = os.pathsep.join([str(repo_root / "src"), str(repo_root)])
    return subprocess.run(
        [sys.executable, "-m", "research.detector_hybrid_evaluator", *args],
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": pythonpath},
        check=False,
    )
