import json
import tempfile
import unittest
from pathlib import Path

from research.detector_contract_validation import (
    validate_detector_assessment,
    validate_detector_packet,
)
from research.detector_corroboration_dataset import (
    run_detector_corroboration_raw_builder,
    run_detector_corroboration_sft_composer,
)
from research.synthetic_detector_assessment_gate import (
    CORROBORATION_BALANCED_JUDGE_PROMPT_VERSION,
    CORROBORATION_BALANCED_TEACHER_PROMPT_VERSION,
    CORROBORATION_STRICT_JUDGE_PROMPT_VERSION,
    CORROBORATION_STRICT_TEACHER_PROMPT_VERSION,
)
from research.tests.test_grounded_synthetic_packet_generator import (
    _clean_structured_report,
)
from research.tests.test_detector_sft_trainer import _chat_row


class DetectorCorroborationRawBuilderTest(unittest.TestCase):
    def test_builder_creates_matched_isolated_and_corroborated_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            training_report = root / "train.json"
            development_report = root / "development.json"
            _write_json(training_report, _report("TRAIN", "2025_Q1"))
            _write_json(development_report, _report("DEV", "2025_Q2"))

            result = run_detector_corroboration_raw_builder(
                training_clean_reports=[training_report],
                development_clean_reports=[development_report],
                output_dir=root / "raw",
            )

            self.assertEqual(result.status, "passed", result.errors)
            records = _read_jsonl(result.raw_jsonl_path)
            self.assertEqual(len(records), 8)
            self.assertEqual(
                {
                    record["metadata"]["generation_metadata"]["calibration_role"]
                    for record in records
                },
                {"train", "development"},
            )

            pairs = {}
            for record in records:
                packet = record["input"]["data"]
                validate_detector_packet(packet)
                self.assertEqual(packet["task"]["risk_category"], "earnings_cashflow_mismatch")
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
                self.assertEqual(len(weak_packet["tool_findings"]), 1)
                self.assertEqual(len(supported_packet["tool_findings"]), 2)
                self.assertEqual(weak_packet["relevant_notes"], [])
                self.assertEqual(
                    weak_packet["tool_findings"][0]["metric_value"],
                    supported_packet["tool_findings"][0]["metric_value"],
                )
                self.assertEqual(weak_packet["tool_findings"][0]["strength"], "strong")
                self.assertEqual(supported_packet["tool_findings"][0]["strength"], "strong")
                self.assertEqual(weak_packet["tool_findings"][0]["evidence_role"], "primary_trigger")
                self.assertEqual(weak_packet["tool_findings"][0]["trigger_strength"], "strong")
                self.assertFalse(weak_packet["tool_findings"][0]["independent_corroboration_present"])
                self.assertEqual(weak_packet["tool_findings"][0]["corroboration_evidence_refs"], [])
                self.assertNotIn("support_calibration", weak_packet["constraints"])
                self.assertNotIn("strong_isolated_trigger_support_level", json.dumps(weak_packet))
                self.assertNotIn("weakly supported", weak_packet["rules"][0]["description"].lower())
                self.assertNotIn("weakly_supported", weak_packet["rules"][0]["description"].lower())
                self.assertEqual(
                    supported_packet["tool_findings"][1]["evidence_role"],
                    "independent_corroboration",
                )
                self.assertEqual(
                    supported_packet["tool_findings"][1]["corroborates_signal_ids"],
                    ["positive_profit_negative_operating_cash_flow"],
                )
                self.assertTrue(supported_packet["tool_findings"][0]["independent_corroboration_present"])
                self.assertEqual(
                    supported_packet["tool_findings"][0]["corroboration_evidence_refs"][0]["ref_id"],
                    supported_packet["tool_findings"][1]["tool_result_id"],
                )

            manifest = _read_json(result.manifest_path)
            self.assertEqual(manifest["records_written"], 8)
            self.assertEqual(manifest["counts"]["calibration_roles"], {"development": 4, "train": 4})
            self.assertEqual(manifest["counts"]["target_support_levels"], {"supported": 4, "weakly_supported": 4})
            self.assertEqual(
                manifest["packet_label_leakage_policy"],
                {
                    "corroboration_boundary_encoded_as_evidence_roles": True,
                    "support_ceiling_visible_to_detector": False,
                    "target_support_level_visible_to_detector": False,
                },
            )


class DetectorCorroborationSftComposerTest(unittest.TestCase):
    def test_composer_keeps_only_complete_pairs_and_preserves_reserved_test_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_sft = root / "base.jsonl"
            _write_jsonl(
                base_sft,
                [
                    _base_earnings_cashflow_chat_row("base-earnings", "train"),
                    _chat_row("base-train", "train"),
                    _chat_row("base-validation", "validation"),
                    _chat_row("base-test", "test"),
                ],
            )

            training_report = root / "train.json"
            development_report = root / "development.json"
            _write_json(training_report, _report("TRAIN", "2025_Q1"))
            _write_json(development_report, _report("DEV", "2025_Q2"))
            raw_result = run_detector_corroboration_raw_builder(
                training_clean_reports=[training_report],
                development_clean_reports=[development_report],
                output_dir=root / "raw",
            )
            raw_records = _read_jsonl(raw_result.raw_jsonl_path)

            gate_dir = root / "gate"
            gate_dir.mkdir()
            filtered = [_promoted_record(record) for record in raw_records]
            incomplete_pair_id = filtered[-1]["metadata"]["generation_metadata"]["matched_pair_id"]
            filtered = [
                record
                for record in filtered
                if not (
                    record["metadata"]["generation_metadata"]["matched_pair_id"] == incomplete_pair_id
                    and record["metadata"]["support_level"] == "supported"
                )
            ]
            _write_jsonl(gate_dir / "filtered.jsonl", filtered)
            _write_json(
                gate_dir / "manifest.json",
                {
                    "mode": "live",
                    "run_purpose": "approved_corroboration_calibration",
                    "trainable_labels_approved": True,
                    "teacher": {"prompt_version": CORROBORATION_STRICT_TEACHER_PROMPT_VERSION},
                    "judge": {"prompt_version": CORROBORATION_STRICT_JUDGE_PROMPT_VERSION},
                },
            )
            _write_json(gate_dir / "metrics.json", {"promoted_records": len(filtered)})

            result = run_detector_corroboration_sft_composer(
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
            enriched_packet = json.loads(enriched_base["messages"][1]["content"])
            self.assertEqual(enriched_packet["task"]["risk_category"], "earnings_cashflow_mismatch")
            self.assertEqual(enriched_base["metadata"]["risk_category"], "earnings_cashflow_mismatch")
            self.assertEqual(enriched_packet["tool_findings"][0]["evidence_role"], "primary_trigger")
            self.assertEqual(enriched_packet["tool_findings"][0]["trigger_strength"], "strong")
            self.assertFalse(enriched_packet["tool_findings"][0]["independent_corroboration_present"])
            self.assertEqual(
                enriched_packet["constraints"]["evidence_bundle_semantics"]["tool_finding_strength_scope"],
                "trigger_magnitude_only",
            )
            calibration_rows = [row for row in rows if row["metadata"].get("calibration_matched_pair_id")]
            self.assertEqual(len(calibration_rows), 6)
            self.assertTrue(
                all("complete visible evidence bundle" in row["messages"][0]["content"] for row in rows)
            )

            manifest = _read_json(result.manifest_path)
            self.assertEqual(manifest["pair_filter"]["incomplete_pairs_excluded"], 1)
            self.assertEqual(manifest["reserved_test_rows_preserved"], 1)

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
            balanced_result = run_detector_corroboration_sft_composer(
                base_sft_jsonl=base_sft,
                gate_run_dirs=[gate_dir],
                output_dir=root / "composed_balanced",
            )

            self.assertEqual(balanced_result.status, "passed", balanced_result.errors)


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
    signal_refs = [
        {
            "evidence_ref_type": "tool_result",
            "ref_id": finding["tool_result_id"],
            "role": "supporting",
        }
        for finding in packet["tool_findings"]
    ]
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
        "cited_evidence_refs": signal_refs,
        "rationale_short": (
            "The packet contains two aligned signals that form a complete visible evidence bundle."
            if target == "supported"
            else "The packet contains one strong quantitative trigger but no independent corroborating signal."
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
        "rules": [{"rule_id": "RULE_BASE_EARN_CASH", "related_signal_ids": ["profit_growth_outpaces_operating_cash_flow"]}],
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
