import json
import tempfile
import unittest
from pathlib import Path

from research.detector_contract_validation import (
    validate_detector_assessment,
    validate_detector_packet,
)
from research.detector_real_shape_alignment_dataset import (
    run_detector_real_shape_alignment_raw_builder,
    run_detector_real_shape_alignment_sft_composer,
)
from research.synthetic_detector_assessment_gate import (
    CORROBORATION_BALANCED_JUDGE_PROMPT_VERSION,
    CORROBORATION_BALANCED_TEACHER_PROMPT_VERSION,
)
from research.tests.test_grounded_synthetic_packet_generator import (
    _clean_structured_report,
)


class DetectorRealShapeAlignmentRawBuilderTest(unittest.TestCase):
    def test_builder_creates_real_surface_matched_pairs_without_evidence_roles(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            training_report = root / "train.json"
            development_report = root / "development.json"
            _write_json(training_report, _report("TRAIN", "2025_Q1"))
            _write_json(development_report, _report("DEV", "2025_Q2"))

            result = run_detector_real_shape_alignment_raw_builder(
                training_clean_reports=[training_report],
                development_clean_reports=[development_report],
                output_dir=root / "raw",
            )

            self.assertEqual(result.status, "passed", result.errors)
            records = _read_jsonl(result.raw_jsonl_path)
            self.assertEqual(len(records), 8)
            pairs: dict[str, list[dict]] = {}
            for record in records:
                packet = record["input"]["data"]
                validate_detector_packet(packet)
                self.assertEqual(packet["task"]["risk_category"], "earnings_cashflow_mismatch")
                self.assertEqual(packet["relevant_notes"], [])
                self.assertEqual(packet["relevant_variance_explanations"], [])
                self.assertNotIn("evidence_role", json.dumps(packet))
                self.assertNotIn("trigger_strength", json.dumps(packet))
                self.assertNotIn("evidence_bundle_semantics", packet["constraints"])
                self.assertEqual(packet["candidate_summary"]["review_mode"], "required")
                pair_id = record["metadata"]["generation_metadata"]["matched_pair_id"]
                pairs.setdefault(pair_id, []).append(record)

            self.assertEqual(len(pairs), 4)
            for pair_records in pairs.values():
                self.assertEqual(len(pair_records), 2)
                by_target = {
                    record["metadata"]["generation_metadata"]["target_support_level"]: record
                    for record in pair_records
                }
                self.assertEqual(set(by_target), {"supported", "weakly_supported"})
                weak_packet = by_target["weakly_supported"]["input"]["data"]
                supported_packet = by_target["supported"]["input"]["data"]
                self.assertEqual(len(weak_packet["relevant_table_rows"]), 4)
                self.assertEqual(len(weak_packet["tool_findings"]), 1)
                self.assertTrue(
                    all(
                        ref["evidence_ref_type"] == "table_cell"
                        for ref in weak_packet["tool_findings"][0]["evidence_refs"]
                    )
                )
                self.assertEqual(weak_packet["tool_findings"][0]["signal_id"], "positive_profit_negative_cfo")
                self.assertEqual(weak_packet["tool_findings"][0]["strength"], "strong")
                self.assertEqual(len(supported_packet["relevant_table_rows"]), 6)
                self.assertEqual(len(supported_packet["tool_findings"]), 2)
                self.assertNotEqual(
                    supported_packet["tool_findings"][0]["signal_id"],
                    supported_packet["tool_findings"][1]["signal_id"],
                )

            manifest = _read_json(result.manifest_path)
            self.assertEqual(manifest["records_written"], 8)
            self.assertEqual(manifest["counts"]["calibration_roles"], {"development": 4, "train": 4})
            self.assertEqual(manifest["counts"]["target_support_levels"], {"supported": 4, "weakly_supported": 4})
            self.assertEqual(
                manifest["packet_label_leakage_policy"],
                {
                    "corroboration_boundary_encoded_as_evidence_roles": False,
                    "real_manual_packet_surface_used": True,
                    "support_ceiling_visible_to_detector": False,
                    "target_support_level_visible_to_detector": False,
                },
            )


class DetectorRealShapeAlignmentSftComposerTest(unittest.TestCase):
    def test_composer_preserves_test_rows_and_does_not_enrich_packets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_sft = root / "base.jsonl"
            _write_jsonl(
                base_sft,
                [
                    _base_earnings_cashflow_chat_row("base-earnings", "train"),
                    _base_earnings_cashflow_chat_row("base-test", "test"),
                ],
            )

            training_report = root / "train.json"
            development_report = root / "development.json"
            _write_json(training_report, _report("TRAIN", "2025_Q1"))
            _write_json(development_report, _report("DEV", "2025_Q2"))
            raw_result = run_detector_real_shape_alignment_raw_builder(
                training_clean_reports=[training_report],
                development_clean_reports=[development_report],
                output_dir=root / "raw",
            )
            raw_records = _read_jsonl(raw_result.raw_jsonl_path)

            gate_dir = root / "gate"
            gate_dir.mkdir()
            filtered = [_promoted_record(record) for record in raw_records]
            _write_jsonl(gate_dir / "filtered.jsonl", filtered)
            _write_json(
                gate_dir / "manifest.json",
                {
                    "mode": "live",
                    "run_purpose": "approved_corroboration_calibration",
                    "trainable_labels_approved": True,
                    "teacher": {"prompt_version": CORROBORATION_BALANCED_TEACHER_PROMPT_VERSION},
                    "judge": {"prompt_version": CORROBORATION_BALANCED_JUDGE_PROMPT_VERSION},
                },
            )
            _write_json(gate_dir / "metrics.json", {"promoted_records": len(filtered)})

            result = run_detector_real_shape_alignment_sft_composer(
                base_sft_jsonl=base_sft,
                gate_run_dirs=[gate_dir],
                output_dir=root / "composed",
            )

            self.assertEqual(result.status, "passed", result.errors)
            rows = _read_jsonl(result.sft_jsonl_path)
            self.assertEqual(sum(row["metadata"]["split"] == "test" for row in rows), 1)
            self.assertEqual(
                [row["metadata"]["example_id"] for row in rows if row["metadata"]["split"] == "test"],
                ["base-test"],
            )
            enriched_base = next(row for row in rows if row["metadata"]["example_id"] == "base-earnings")
            base_packet = json.loads(enriched_base["messages"][1]["content"])
            self.assertEqual(base_packet["task"]["risk_category"], "earnings_cashflow_mismatch")
            self.assertNotIn("evidence_role", json.dumps(base_packet))
            self.assertNotIn("evidence_bundle_semantics", base_packet["constraints"])
            alignment_rows = [row for row in rows if row["metadata"].get("real_shape_alignment")]
            self.assertEqual(len(alignment_rows), 8)

            manifest = _read_json(result.manifest_path)
            self.assertEqual(manifest["reserved_test_rows_preserved"], 1)
            self.assertFalse(manifest["experiment_boundary"]["packet_evidence_role_enrichment_applied"])


def _report(ticker: str, period_key: str) -> dict:
    report = _clean_structured_report()
    report["company_key"] = ticker
    report["ticker"] = ticker
    report["company_name"] = f"{ticker} Company"
    report["period_key"] = period_key
    report["period"] = period_key.replace("_", "-")
    report["report_id"] = f"{ticker}_{period_key}_CLEAN"
    report["artifact_id"] = f"RPT_{ticker}_{period_key}_CLEAN"
    report["source_document_id"] = f"DOC_{ticker}_{period_key}"
    report["traceability"]["source_file_sha256"] = f"source-{ticker}-{period_key}"
    report["traceability"]["normalized_text_hash"] = f"normalized-{ticker}-{period_key}"
    report["traceability"]["table_content_hash"] = f"table-{ticker}-{period_key}"
    report["traceability"]["source_group_key"] = f"test:{ticker}:{period_key}"
    return report


def _promoted_record(raw_record: dict) -> dict:
    record = json.loads(json.dumps(raw_record))
    packet = record["input"]["data"]
    target = record["metadata"]["generation_metadata"]["target_support_level"]
    assessment = {
        "assessment_id": f"ASSESS_{packet['packet_id']}",
        "packet_id": packet["packet_id"],
        "candidate_id": packet["candidate_id"],
        "report_id": packet["report_id"],
        "risk_category": packet["task"]["risk_category"],
        "support_level": target,
        "confidence": 0.9 if target == "supported" else 0.7,
        "severity": "high" if target == "supported" else "medium",
        "validated_signals": [
            {
                "signal_id": finding["signal_id"],
                "status": "validated" if target == "supported" else "partially_validated",
                "support_level": target,
                "tool_result_id": finding["tool_result_id"],
                "cited_evidence_refs": [
                    {
                        "evidence_ref_type": "tool_result",
                        "ref_id": finding["tool_result_id"],
                        "role": "supporting",
                    }
                ],
            }
            for finding in packet["tool_findings"]
        ],
        "cited_evidence_refs": [
            {
                "evidence_ref_type": "tool_result",
                "ref_id": finding["tool_result_id"],
                "role": "supporting",
            }
            for finding in packet["tool_findings"]
        ],
        "rationale_short": (
            "The packet contains a separate corroborating signal in addition to the earnings-cash-flow trigger."
            if target == "supported"
            else "The packet contains one strong earnings-cash-flow trigger but no separate corroborating signal."
        ),
    }
    validate_detector_assessment(assessment, packet)
    record["source_type"] = "synthetic_injected_filtered"
    record["metadata"]["support_level"] = target
    record["metadata"]["severity"] = assessment["severity"]
    record["output"] = {"type": "DetectorAssessment", "data": assessment}
    return record


def _base_earnings_cashflow_chat_row(example_id: str, split: str) -> dict:
    packet = {
        "packet_id": f"PACKET_{example_id}",
        "candidate_id": f"CAND_{example_id}",
        "report_id": "REPORT_BASE",
        "task": {"risk_category": "earnings_cashflow_quality_risk"},
        "metadata": {},
        "candidate_summary": {
            "priority": "high",
            "reason_for_candidate": "Profit increased while operating cash flow lagged.",
            "supporting_signal_ids": ["profit_growth_outpaces_operating_cash_flow"],
        },
        "relevant_table_rows": [{"report_id": "REPORT_BASE", "row_id": "row-profit", "values": {}}],
        "relevant_notes": [],
        "relevant_variance_explanations": [],
        "tool_findings": [
            {
                "tool_result_id": "TOOL_BASE_EARN_CASH",
                "tool_name": "earnings_vs_operating_cash_flow_tool",
                "risk_category": "earnings_cashflow_quality_risk",
                "signal_id": "profit_growth_outpaces_operating_cash_flow",
                "flag": True,
                "metric": "operating_cash_flow_to_profit_ratio",
                "metric_value": 0.35,
                "strength": "strong",
                "threshold": "flag if operating cash flow is below 80 percent of profit while profit grows",
                "finding_summary": "Operating cash flow was 35.0% of current profit while profit grew.",
                "evidence_refs": [
                    {
                        "evidence_ref_type": "table_row",
                        "ref_id": "REPORT_BASE:row-profit",
                    }
                ],
            }
        ],
        "rules": [
            {
                "rule_id": "RULE_BASE_EARN_CASH",
                "risk_category": "earnings_cashflow_quality_risk",
                "related_signal_ids": ["profit_growth_outpaces_operating_cash_flow"],
            }
        ],
        "constraints": {},
    }
    assessment = {
        "assessment_id": f"ASSESS_{example_id}",
        "packet_id": packet["packet_id"],
        "candidate_id": packet["candidate_id"],
        "report_id": packet["report_id"],
        "risk_category": "earnings_cashflow_quality_risk",
        "support_level": "supported",
        "confidence": 0.8,
        "severity": "medium",
        "validated_signals": [
            {
                "signal_id": "profit_growth_outpaces_operating_cash_flow",
                "status": "validated",
                "support_level": "supported",
                "tool_result_id": "TOOL_BASE_EARN_CASH",
                "cited_evidence_refs": [
                    {
                        "evidence_ref_type": "tool_result",
                        "ref_id": "TOOL_BASE_EARN_CASH",
                        "role": "supporting",
                    }
                ],
            }
        ],
        "cited_evidence_refs": [
            {
                "evidence_ref_type": "tool_result",
                "ref_id": "TOOL_BASE_EARN_CASH",
                "role": "supporting",
            }
        ],
        "rationale_short": "The base row is a fixture.",
    }
    return {
        "messages": [
            {"role": "system", "content": "locked system prompt"},
            {"role": "user", "content": json.dumps(packet, sort_keys=True)},
            {"role": "assistant", "content": json.dumps(assessment, sort_keys=True)},
        ],
        "metadata": {
            "example_id": example_id,
            "risk_category": "earnings_cashflow_quality_risk",
            "source_type": "synthetic_injected_filtered",
            "split": split,
            "support_level": "supported",
        },
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
