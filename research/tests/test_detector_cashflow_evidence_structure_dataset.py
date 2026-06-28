import json
import tempfile
import unittest
from pathlib import Path

from research.detector_contract_validation import (
    validate_detector_assessment,
    validate_detector_packet,
)
from research.detector_cashflow_evidence_structure_dataset import (
    run_detector_cashflow_evidence_structure_raw_builder,
    run_detector_cashflow_evidence_structure_sft_composer,
)
from research.synthetic_detector_assessment_gate import (
    CORROBORATION_BALANCED_JUDGE_PROMPT_VERSION,
    CORROBORATION_BALANCED_TEACHER_PROMPT_VERSION,
)
from research.tests.test_grounded_synthetic_packet_generator import (
    _clean_structured_report,
)


class DetectorCashflowEvidenceStructureRawBuilderTest(unittest.TestCase):
    def test_v1_builder_creates_cashflow_audit_groups_with_disagreement_cases(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            training_report = root / "train.json"
            development_report = root / "development.json"
            _write_json(training_report, _report("TRAIN", "2025_Q1"))
            _write_json(development_report, _report("DEV", "2025_Q2"))

            result = run_detector_cashflow_evidence_structure_raw_builder(
                training_clean_reports=[training_report],
                development_clean_reports=[development_report],
                output_dir=root / "raw",
                case_set="v1",
            )

            self.assertEqual(result.status, "passed", result.errors)
            records = _read_jsonl(result.raw_jsonl_path)
            self.assertEqual(len(records), 8)
            groups: dict[str, list[dict]] = {}
            for record in records:
                packet = record["input"]["data"]
                validate_detector_packet(packet)
                self.assertEqual(packet["task"]["risk_category"], "earnings_cashflow_mismatch")
                self.assertIn("evidence_bundle_semantics", packet["constraints"])
                self.assertEqual(packet["candidate_summary"]["review_mode"], "required")
                self.assertNotIn("target_support_level", json.dumps(packet))
                primary = packet["tool_findings"][0]
                audit = packet["tool_findings"][-1]
                self.assertEqual(primary["evidence_role"], "primary_trigger")
                self.assertEqual(primary["trigger_strength"], "strong")
                self.assertEqual(audit["tool_name"], "cashflow_corroboration_audit_tool")
                self.assertEqual(audit["evidence_role"], "context")
                self.assertFalse(audit["flag"])
                group_id = record["metadata"]["generation_metadata"]["matched_group_id"]
                groups.setdefault(group_id, []).append(record)

            self.assertEqual(len(groups), 2)
            for group_records in groups.values():
                self.assertEqual(len(group_records), 4)
                by_case = {
                    record["metadata"]["generation_metadata"]["variant_pattern_id"]: record
                    for record in group_records
                }
                self.assertEqual(
                    set(by_case),
                    {
                        "isolated_trigger_weak",
                        "collection_delay_supported",
                        "balance_growth_supported",
                        "corroborated_mitigated_weak",
                    },
                )
                isolated = by_case["isolated_trigger_weak"]
                isolated_packet = isolated["input"]["data"]
                self.assertEqual(isolated["metadata"]["generation_metadata"]["target_support_level"], "weakly_supported")
                self.assertFalse(isolated_packet["tool_findings"][0]["independent_corroboration_present"])
                self.assertEqual(len(isolated_packet["tool_findings"]), 2)
                self.assertEqual(isolated_packet["relevant_notes"], [])
                self.assertEqual(isolated_packet["relevant_variance_explanations"], [])

                disclosure = by_case["collection_delay_supported"]
                disclosure_packet = disclosure["input"]["data"]
                self.assertEqual(disclosure["metadata"]["generation_metadata"]["target_support_level"], "supported")
                self.assertTrue(disclosure["metadata"]["generation_metadata"]["audit_disagreement_case"])
                self.assertFalse(disclosure_packet["tool_findings"][0]["independent_corroboration_present"])
                self.assertEqual(len(disclosure_packet["relevant_notes"]), 1)

                supported = by_case["balance_growth_supported"]
                supported_packet = supported["input"]["data"]
                self.assertTrue(supported_packet["tool_findings"][0]["independent_corroboration_present"])
                self.assertEqual(len(supported_packet["relevant_table_rows"]), 6)
                self.assertEqual(len(supported_packet["tool_findings"]), 3)

                mitigated = by_case["corroborated_mitigated_weak"]
                mitigated_packet = mitigated["input"]["data"]
                self.assertEqual(mitigated["metadata"]["generation_metadata"]["target_support_level"], "weakly_supported")
                self.assertTrue(mitigated["metadata"]["generation_metadata"]["audit_disagreement_case"])
                self.assertTrue(mitigated_packet["tool_findings"][0]["independent_corroboration_present"])
                self.assertEqual(len(mitigated_packet["relevant_variance_explanations"]), 1)

            manifest = _read_json(result.manifest_path)
            self.assertEqual(manifest["records_written"], 8)
            self.assertEqual(manifest["counts"]["calibration_roles"], {"development": 4, "train": 4})
            self.assertEqual(manifest["counts"]["target_support_levels"], {"supported": 4, "weakly_supported": 4})
            self.assertEqual(manifest["counts"]["audit_disagreement_cases"], {"False": 4, "True": 4})
            self.assertEqual(
                manifest["packet_label_leakage_policy"],
                {
                    "audit_status_deterministically_maps_to_label": False,
                    "audit_status_visible_to_detector": True,
                    "real_manual_packet_surface_used": True,
                    "support_ceiling_visible_to_detector": False,
                    "target_support_level_visible_to_detector": False,
                },
            )

    def test_v2_builder_creates_natural_four_way_groups_without_label_shortcut(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            training_report = root / "train.json"
            development_report = root / "development.json"
            _write_json(training_report, _report("TRAIN", "2025_Q1"))
            _write_json(development_report, _report("DEV", "2025_Q2"))

            result = run_detector_cashflow_evidence_structure_raw_builder(
                training_clean_reports=[training_report],
                development_clean_reports=[development_report],
                output_dir=root / "raw",
            )

            self.assertEqual(result.status, "passed", result.errors)
            records = _read_jsonl(result.raw_jsonl_path)
            self.assertEqual(len(records), 8)
            groups: dict[str, list[dict]] = {}
            for record in records:
                packet = record["input"]["data"]
                validate_detector_packet(packet)
                self.assertEqual(packet["task"]["risk_category"], "earnings_cashflow_mismatch")
                self.assertNotIn("target_support_level", json.dumps(packet))
                group_id = record["metadata"]["generation_metadata"]["matched_group_id"]
                groups.setdefault(group_id, []).append(record)

            self.assertEqual(len(groups), 2)
            for group_records in groups.values():
                self.assertEqual(len(group_records), 4)
                by_case = {
                    record["metadata"]["generation_metadata"]["variant_pattern_id"]: record
                    for record in group_records
                }
                self.assertEqual(
                    set(by_case),
                    {
                        "isolated_trigger_weak",
                        "balance_growth_supported",
                        "adequate_cashflow_not_supported",
                        "missing_cashflow_insufficient",
                    },
                )

                isolated = by_case["isolated_trigger_weak"]
                isolated_packet = isolated["input"]["data"]
                self.assertEqual(isolated["metadata"]["generation_metadata"]["target_support_level"], "weakly_supported")
                self.assertTrue(isolated_packet["tool_findings"][0]["flag"])
                self.assertEqual(isolated_packet["tool_findings"][0]["value"], -2.5)
                self.assertEqual(isolated_packet["tool_findings"][-1]["value"], "isolated")
                self.assertFalse(isolated_packet["tool_findings"][-1]["independent_corroboration_present"])

                supported = by_case["balance_growth_supported"]
                supported_packet = supported["input"]["data"]
                self.assertEqual(supported["metadata"]["generation_metadata"]["target_support_level"], "supported")
                self.assertTrue(supported_packet["tool_findings"][0]["flag"])
                self.assertEqual(len(supported_packet["tool_findings"]), 3)
                self.assertEqual(supported_packet["tool_findings"][-1]["value"], "corroborated")

                adequate = by_case["adequate_cashflow_not_supported"]
                adequate_packet = adequate["input"]["data"]
                self.assertEqual(adequate["metadata"]["generation_metadata"]["target_support_level"], "not_supported")
                self.assertFalse(adequate_packet["tool_findings"][0]["flag"])
                self.assertEqual(adequate_packet["tool_findings"][0]["value"], 0.9)
                self.assertEqual(adequate_packet["tool_findings"][-1]["value"], "isolated")

                missing = by_case["missing_cashflow_insufficient"]
                missing_packet = missing["input"]["data"]
                self.assertEqual(
                    missing["metadata"]["generation_metadata"]["target_support_level"],
                    "insufficient_evidence",
                )
                self.assertFalse(missing_packet["tool_findings"][0]["flag"])
                self.assertIsNone(missing_packet["tool_findings"][0]["value"])
                self.assertEqual(missing_packet["tool_findings"][-1]["value"], "not_assessable")
                current_rows = [
                    row for row in missing_packet["relevant_table_rows"]
                    if row["report_id"] == missing_packet["report_id"]
                ]
                self.assertEqual(len(current_rows), 1)
                self.assertTrue(current_rows[0]["row_id"].startswith("ROW_PROFIT_"))

            manifest = _read_json(result.manifest_path)
            self.assertEqual(manifest["case_set"], "v2")
            self.assertEqual(manifest["scenario_id"], "earnings_cashflow_evidence_structure_v2")
            self.assertEqual(manifest["records_written"], 8)
            self.assertEqual(manifest["counts"]["calibration_roles"], {"development": 4, "train": 4})
            self.assertEqual(
                manifest["counts"]["target_support_levels"],
                {
                    "insufficient_evidence": 2,
                    "not_supported": 2,
                    "supported": 2,
                    "weakly_supported": 2,
                },
            )
            self.assertEqual(manifest["counts"]["audit_disagreement_cases"], {"False": 6, "True": 2})


class DetectorCashflowEvidenceStructureSftComposerTest(unittest.TestCase):
    def test_composer_preserves_test_rows_and_marks_evidence_structure_rows(self) -> None:
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
            raw_result = run_detector_cashflow_evidence_structure_raw_builder(
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

            result = run_detector_cashflow_evidence_structure_sft_composer(
                base_sft_jsonl=base_sft,
                gate_run_dirs=[gate_dir],
                output_dir=root / "composed",
                case_set="v2",
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
            self.assertIn("evidence_role", json.dumps(base_packet))
            self.assertIn("evidence_bundle_semantics", base_packet["constraints"])
            evidence_structure_rows = [row for row in rows if row["metadata"].get("cashflow_evidence_structure")]
            self.assertEqual(len(evidence_structure_rows), 8)
            self.assertEqual(
                sum(row["metadata"].get("audit_disagreement_case") is True for row in evidence_structure_rows),
                2,
            )

            manifest = _read_json(result.manifest_path)
            self.assertEqual(manifest["reserved_test_rows_preserved"], 1)
            self.assertEqual(manifest["group_filter"]["complete_groups_included"], 2)
            self.assertTrue(manifest["experiment_boundary"]["base_packet_evidence_role_enrichment_applied"])
            self.assertFalse(manifest["experiment_boundary"]["audit_status_deterministically_maps_to_label"])


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
    role_by_support = {
        "supported": "supporting",
        "weakly_supported": "context",
        "not_supported": "refuting",
        "insufficient_evidence": "missing_required_context",
    }
    status_by_support = {
        "supported": "validated",
        "weakly_supported": "partially_validated",
        "not_supported": "rejected",
        "insufficient_evidence": "not_assessable",
    }
    severity_by_support = {
        "supported": "high",
        "weakly_supported": "medium",
        "not_supported": "low",
        "insufficient_evidence": "unknown",
    }
    confidence_by_support = {
        "supported": 0.9,
        "weakly_supported": 0.7,
        "not_supported": 0.85,
        "insufficient_evidence": 0.6,
    }
    role = role_by_support[target]
    cited_refs = [
        {
            "evidence_ref_type": "tool_result",
            "ref_id": finding["tool_result_id"],
            "role": role,
        }
        for finding in packet["tool_findings"]
        if finding["tool_name"] != "cashflow_corroboration_audit_tool"
    ]
    cited_refs.extend(
        {
            "evidence_ref_type": "note",
            "ref_id": f"{note['report_id']}:{note['note_id']}",
            "role": role,
        }
        for note in packet["relevant_notes"]
    )
    cited_refs.extend(
        {
            "evidence_ref_type": "variance_explanation_span",
            "ref_id": f"{span['report_id']}:{span['span_id']}",
            "role": role,
        }
        for span in packet["relevant_variance_explanations"]
    )
    validated_signals = [
        {
            "signal_id": finding["signal_id"],
            "status": status_by_support[target],
            "support_level": target,
            "tool_result_id": finding["tool_result_id"],
            "cited_evidence_refs": [
                {
                    "evidence_ref_type": "tool_result",
                    "ref_id": finding["tool_result_id"],
                    "role": role,
                }
            ],
        }
        for finding in packet["tool_findings"]
        if finding["tool_name"] != "cashflow_corroboration_audit_tool"
    ]
    assessment = {
        "assessment_id": f"ASSESS_{packet['packet_id']}",
        "packet_id": packet["packet_id"],
        "candidate_id": packet["candidate_id"],
        "report_id": packet["report_id"],
        "risk_category": packet["task"]["risk_category"],
        "support_level": target,
        "confidence": confidence_by_support[target],
        "severity": severity_by_support[target],
        "validated_signals": validated_signals,
        "cited_evidence_refs": cited_refs,
        "rationale_short": _rationale_for_support(target),
    }
    validate_detector_assessment(assessment, packet)
    record["source_type"] = "synthetic_injected_filtered"
    record["metadata"]["support_level"] = target
    record["metadata"]["severity"] = assessment["severity"]
    record["output"] = {"type": "DetectorAssessment", "data": assessment}
    return record


def _rationale_for_support(support_level: str) -> str:
    if support_level == "supported":
        return "The packet contains enough non-audit evidence for the earnings-cash-flow risk signal."
    if support_level == "weakly_supported":
        return "The packet contains a strong earnings-cash-flow trigger, but the complete evidence remains partial."
    if support_level == "not_supported":
        return "The packet-visible cash-flow comparison does not cross the risk threshold."
    return "The packet lacks the current-period cash-flow input needed to assess the signal."


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
