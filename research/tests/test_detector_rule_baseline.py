import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from research.detector_rule_baseline import run_detector_rule_baseline


class Wave4DetectorRuleBaselineTest(unittest.TestCase):
    def test_rule_only_baseline_writes_metrics_manifest_and_predictions_for_fixture_split(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sft_jsonl = root / "detector_sft_chat.jsonl"
            _write_jsonl(
                sft_jsonl,
                [
                    _sft_row(
                        "supported-1",
                        split="validation",
                        packet=_packet("supported-1", flagged_tool_finding=True),
                        gold_support_level="supported",
                        gold_severity="medium",
                    ),
                    _sft_row(
                        "insufficient-1",
                        split="validation",
                        packet=_packet("insufficient-1", include_evidence=False),
                        gold_support_level="insufficient_evidence",
                        gold_severity="unknown",
                    ),
                    _sft_row(
                        "train-1",
                        split="train",
                        packet=_packet("train-1", flagged_tool_finding=True),
                        gold_support_level="supported",
                        gold_severity="medium",
                    ),
                ],
            )
            output_root = root / "rule_eval"

            result = run_detector_rule_baseline(
                sft_jsonl=sft_jsonl,
                output_root=output_root,
                split="validation",
                allow_noncanonical_sft_jsonl=True,
            )

            self.assertEqual(result.status, "passed", result.errors)
            predictions = _read_jsonl(output_root / "predictions.jsonl")
            metrics = _read_json(output_root / "metrics.json")
            manifest = _read_json(output_root / "manifest.json")
            self.assertEqual([prediction["prediction_status"] for prediction in predictions], ["accepted", "accepted"])
            self.assertEqual(metrics["num_examples"], 2)
            self.assertEqual(metrics["support_level_exact_match_count"], 2)
            self.assertEqual(metrics["support_level_macro_f1"], 1.0)
            self.assertEqual(metrics["invalid_reason_counts"], {})
            self.assertEqual(manifest["variant"], "rule_only_detector_baseline")
            self.assertEqual(manifest["split"], "validation")
            self.assertEqual(manifest["metric_version"], "detector_ablation_metrics_v1")
            self.assertEqual(manifest["rule_policy"]["semantic_model_used"], False)

    def test_rule_only_baseline_reports_weak_support_and_invalid_evidence_refs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sft_jsonl = root / "detector_sft_chat.jsonl"
            weak_packet = _packet("weak-1", flagged_tool_finding=False)
            weak_packet["tool_findings"] = []
            invalid_packet = _packet("invalid-ref-1", flagged_tool_finding=True)
            invalid_packet["tool_findings"][0]["evidence_refs"][0]["evidence_ref_type"] = "spreadsheet_cell"
            _write_jsonl(
                sft_jsonl,
                [
                    _sft_row(
                        "weak-1",
                        split="validation",
                        packet=weak_packet,
                        gold_support_level="weakly_supported",
                        gold_severity="medium",
                    ),
                    _sft_row(
                        "invalid-ref-1",
                        split="validation",
                        packet=invalid_packet,
                        gold_support_level="supported",
                        gold_severity="medium",
                    ),
                    _sft_row(
                        "train-1",
                        split="train",
                        packet=_packet("train-1", flagged_tool_finding=True),
                        gold_support_level="supported",
                        gold_severity="medium",
                    ),
                ],
            )

            result = run_detector_rule_baseline(
                sft_jsonl=sft_jsonl,
                output_root=root / "rule_eval",
                split="validation",
                allow_noncanonical_sft_jsonl=True,
            )

            self.assertEqual(result.status, "passed", result.errors)
            metrics = _read_json(result.metrics_path)
            predictions = _read_jsonl(result.predictions_path)
            self.assertEqual(predictions[0]["predicted_support_level"], "weakly_supported")
            self.assertEqual(predictions[1]["prediction_status"], "invalid")
            self.assertEqual(metrics["num_valid_predictions"], 1)
            self.assertEqual(metrics["num_invalid_responses"], 1)
            self.assertEqual(metrics["invalid_reason_counts"], {"invalid_evidence_ref_type": 1})
            self.assertEqual(metrics["weakly_supported_detection"]["true_positive"], 1)
            self.assertEqual(metrics["weakly_supported_detection"]["recall"], 1.0)

    def test_public_command_runs_rule_only_smoke_with_unsupported_case(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sft_jsonl = root / "detector_sft_chat.jsonl"
            _write_jsonl(
                sft_jsonl,
                [
                    _sft_row(
                        "unsupported-1",
                        split="validation",
                        packet=_packet("unsupported-1", flagged_tool_finding=False),
                        gold_support_level="not_supported",
                        gold_severity="unknown",
                    ),
                    _sft_row(
                        "train-1",
                        split="train",
                        packet=_packet("train-1", flagged_tool_finding=True),
                        gold_support_level="supported",
                        gold_severity="medium",
                    ),
                ],
            )
            output_root = root / "rule_eval"

            completed = _run_rule_baseline_command(
                "--sft-jsonl",
                str(sft_jsonl),
                "--output-root",
                str(output_root),
                "--split",
                "validation",
                "--allow-noncanonical-sft-jsonl",
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(completed.stdout)["status"], "passed")
            predictions = _read_jsonl(output_root / "predictions.jsonl")
            metrics = _read_json(output_root / "metrics.json")
            self.assertEqual(predictions[0]["predicted_support_level"], "not_supported")
            self.assertEqual(metrics["false_positive_rejection"]["rate"], 1.0)


def _sft_row(
    example_id: str,
    *,
    split: str,
    packet: dict,
    gold_support_level: str,
    gold_severity: str,
) -> dict:
    assessment = {
        "assessment_id": f"ASSESS_{example_id}",
        "packet_id": packet["packet_id"],
        "candidate_id": packet["candidate_id"],
        "report_id": packet["report_id"],
        "risk_category": packet["task"]["risk_category"],
        "support_level": gold_support_level,
        "confidence": 0.9,
        "severity": gold_severity,
        "validated_signals": [
            {
                "signal_id": "signal-1",
                "status": "validated" if gold_support_level == "supported" else "not_assessable",
                "support_level": gold_support_level,
                "cited_evidence_refs": _gold_refs(packet),
            }
        ],
        "cited_evidence_refs": _gold_refs(packet),
        "rationale_short": "Fixture gold assessment.",
    }
    return {
        "messages": [
            {"role": "system", "content": "locked system prompt"},
            {"role": "user", "content": json.dumps(packet, sort_keys=True)},
            {"role": "assistant", "content": json.dumps(assessment, sort_keys=True)},
        ],
        "metadata": {
            "example_id": example_id,
            "source_type": "synthetic_injected_filtered",
            "split": split,
        },
    }


def _packet(
    example_id: str,
    *,
    flagged_tool_finding: bool = False,
    include_evidence: bool = True,
) -> dict:
    evidence_refs = [{"evidence_ref_type": "table_cell", "ref_id": "REPORT_1:cell-1", "role": "supporting"}]
    tool_findings = []
    if include_evidence:
        tool_findings.append(
            {
                "tool_result_id": "tool-1",
                "report_id": "REPORT_1",
                "tool_name": "fixture_rule",
                "risk_category": "earnings_cashflow_quality_risk",
                "signal_id": "signal-1",
                "flag": flagged_tool_finding,
                "finding_summary": "Fixture signal.",
                "evidence_refs": evidence_refs,
            }
        )
    return {
        "packet_id": f"PACKET_{example_id}",
        "candidate_id": f"CAND_{example_id}",
        "report_id": "REPORT_1",
        "task": {"risk_category": "earnings_cashflow_quality_risk"},
        "metadata": {},
        "candidate_summary": {
            "priority": "medium",
            "reason_for_candidate": "Fixture candidate.",
            "supporting_signal_ids": ["signal-1"],
            "evidence_refs": evidence_refs if include_evidence else [],
        },
        "relevant_table_rows": [
            {
                "report_id": "REPORT_1",
                "row_id": "row-1",
                "values": {"current": {"cell_id": "cell-1", "value": 1}},
            }
        ]
        if include_evidence
        else [],
        "relevant_notes": [],
        "relevant_variance_explanations": [],
        "tool_findings": tool_findings,
        "rules": [{"rule_id": "rule-1", "related_signal_ids": ["signal-1"]}],
        "constraints": {},
    }


def _gold_refs(packet: dict) -> list[dict]:
    refs = packet["candidate_summary"]["evidence_refs"]
    if refs:
        return refs
    return [{"evidence_ref_type": "rule", "ref_id": "rule-1", "role": "missing_required_context"}]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _run_rule_baseline_command(*args: str) -> subprocess.CompletedProcess:
    repo_root = Path(__file__).resolve().parents[3]
    pythonpath = os.pathsep.join([str(repo_root / "src"), str(repo_root)])
    return subprocess.run(
        [sys.executable, "-m", "research.detector_rule_baseline", *args],
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": pythonpath},
        check=False,
    )
