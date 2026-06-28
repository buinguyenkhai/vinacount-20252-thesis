import json
import copy
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from research.fixture_spine import (
    _securities_fixture_case,
    assess_detector_packets,
    build_final_report,
    build_detector_packets,
    run_fixture_spine,
    validate_detector_packet,
    validate_detector_assessment,
    validate_final_report,
)


class FixtureSpineTest(unittest.TestCase):
    def _fixture_case(self, case_id="signal_present_standard_corporate"):
        fixture_path = (
            Path(__file__).resolve().parents[1]
            / "fixtures"
            / "wave1"
            / "fixture_spine_cases.json"
        )
        raw = json.loads(fixture_path.read_text(encoding="utf-8"))
        return copy.deepcopy(next(case for case in raw["report_sets"] if case["case_id"] == case_id))

    def test_unresolved_amendment_context_stops_before_tools_and_detector_packets(self):
        fixture_case = self._fixture_case()
        fixture_case["case_id"] = "GAS_2025_Q1_CONSOLIDATED_UNRESOLVED_ADJUSTED_NOTES"
        fixture_case["target_report_memory"]["metadata"]["company_name"] = "GAS Issuer"
        fixture_case["prior_year_report_memory"]["metadata"]["company_name"] = "GAS Issuer"
        fixture_case["target_report_memory"]["metadata"]["canonical_source_document_id"] = "GAS_Q1_2025_FS"
        fixture_case["target_report_memory"]["metadata"]["amendment_context_attachments"] = [
            {
                "source_document_id": "GAS_Q1_2025_ADJUSTED_NOTES",
                "attachment_type": "adjusted_notes",
                "affects_operative_values": True,
                "corrected_value_resolution_status": "unresolved",
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_fixture_spine(Path(temp_dir), fixture_cases=[fixture_case])
            audit_text = result.audit_log_path.read_text(encoding="utf-8")

        self.assertEqual(result.loaded_report_sets, ["GAS_2025_Q1_CONSOLIDATED_UNRESOLVED_ADJUSTED_NOTES"])
        self.assertEqual(result.tool_gating_records, [])
        self.assertEqual(result.tool_findings, [])
        self.assertEqual(result.candidate_risks, [])
        self.assertEqual(result.detector_packets, [])
        self.assertIn("amendment_context_corrected_value_resolution_required", audit_text)

    def test_failed_quality_gate_path_does_not_produce_candidates_or_detector_packets(self):
        fixture_case = self._fixture_case()
        fixture_case["case_id"] = "real_extraction_failed_gate"
        fixture_case["target_report_memory"]["quality_records"] = [
            {
                "record_type": "note_reference_match_failure",
                "severity": "blocking",
                "row_id": "ROW_TRADE_RECEIVABLES_2025_Q3",
                "message": "Required Note Reference Match is ambiguous or missing.",
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch(
            "research.fixture_spine._load_fixture_cases",
            return_value=[fixture_case],
        ):
            result = run_fixture_spine(Path(temp_dir))

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.tool_gating_records, [])
        self.assertEqual(result.tool_findings, [])
        self.assertEqual(result.candidate_risks, [])
        self.assertEqual(result.detector_packets, [])
        self.assertEqual(result.final_reports, [])

    def test_full_report_amendment_supersession_without_context_attachment_does_not_require_gate(self):
        fixture_case = self._fixture_case()
        fixture_case["case_id"] = "FULL_REPORT_AMENDMENT_SUPERSESSION"
        fixture_case["target_report_memory"]["metadata"]["filing_status"] = "amended_or_replacement"

        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_fixture_spine(Path(temp_dir), fixture_cases=[fixture_case])

        self.assertTrue(result.tool_gating_records)
        self.assertTrue(result.tool_findings)
        self.assertTrue(result.detector_packets)

    def test_resolved_amendment_context_detector_packets_emit_only_operative_corrected_value(self):
        fixture_case = self._fixture_case()
        fixture_case["case_id"] = "SSI_2025_Q3_CONSOLIDATED_RESOLVED_CORRECTION"
        fixture_case["target_report_memory"]["metadata"]["company_name"] = "SSI Issuer"
        fixture_case["prior_year_report_memory"]["metadata"]["company_name"] = "SSI Issuer"
        fixture_case["target_report_memory"]["metadata"]["amendment_context_attachments"] = [
            {
                "source_document_id": "SSI_2025_Q3_CORRECTION_PAGES",
                "attachment_type": "correction_packet",
                "affects_operative_values": True,
                "corrected_value_resolution_status": "resolved",
            }
        ]
        original_revenue = None
        for table in fixture_case["target_report_memory"]["tables"]:
            if table["table_id"] != "TBL_IS_2025_Q3":
                continue
            table["source_document_id"] = "SSI_2025_Q3_CORRECTION_PAGES"
            for row in table["rows"]:
                if row.get("row_id") != "ROW_REVENUE_2025_Q3":
                    continue
                row["source_document_id"] = "SSI_2025_Q3_CORRECTION_PAGES"
                cell = row["cells"][0]
                original_revenue = cell["value"]
                cell["value"] = 2222222
                cell["source_document_id"] = "SSI_2025_Q3_CORRECTION_PAGES"
                cell["amendment_resolution"] = {
                    "basis": "deterministic_one_to_one",
                    "amendment_context_attachment_id": "SSI_2025_Q3_CORRECTION_PAGES",
                    "superseded_source_document_id": "SSI_2025_Q3_FS",
                    "superseded_value": original_revenue,
                }
                fixture_case["target_report_memory"]["cell_index"][cell["cell_id"]]["value"] = 2222222
                fixture_case["target_report_memory"]["cell_index"][cell["cell_id"]][
                    "source_document_id"
                ] = "SSI_2025_Q3_CORRECTION_PAGES"
                fixture_case["target_report_memory"]["cell_index"][cell["cell_id"]][
                    "amendment_resolution"
                ] = cell["amendment_resolution"]
        self.assertEqual(original_revenue, 1850000)

        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_fixture_spine(Path(temp_dir), fixture_cases=[fixture_case])

        packet_payload = json.dumps(result.detector_packets)
        self.assertIn("2222222", packet_payload)
        self.assertNotIn("1850000", packet_payload)
        self.assertIn("SSI_2025_Q3_CORRECTION_PAGES", packet_payload)

    def test_wave2_insurance_fixture_track_runs_through_existing_spine(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            result = run_fixture_spine(output_dir)

            self.assertIn("signal_present_insurance", result.loaded_report_sets)
            insurance_set = next(
                report_set
                for report_set in result.report_sets
                if report_set.case_id == "signal_present_insurance"
            )
            self.assertEqual(insurance_set.target.metadata["report_profile"], "insurance")
            self.assertEqual(insurance_set.prior_year.metadata["report_profile"], "insurance")
            self.assertEqual(insurance_set.target.metadata["insurance_subprofile"], "non_life")
            self.assertEqual(insurance_set.prior_year.metadata["insurance_subprofile"], "non_life")
            self.assertEqual(insurance_set.target.metadata["report_basis"], "consolidated")
            self.assertEqual(insurance_set.prior_year.metadata["report_basis"], "consolidated")
            self.assertEqual(
                insurance_set.target.metadata["company_name"],
                insurance_set.prior_year.metadata["company_name"],
            )
            self.assertEqual(insurance_set.target.quarter, insurance_set.prior_year.quarter)
            self.assertEqual(insurance_set.target.year, insurance_set.prior_year.year + 1)

            insurance_gating = [
                record
                for record in result.tool_gating_records
                if record["case_id"] == "signal_present_insurance"
            ]
            enabled_tool_names = {
                record["tool_name"] for record in insurance_gating if record["status"] == "enabled"
            }
            self.assertGreaterEqual(
                enabled_tool_names,
                {
                    "insurance_premium_receivable_coherence_tool",
                    "insurance_reserve_movement_tool",
                    "insurance_reinsurance_balance_tool",
                },
            )
            self.assertTrue(
                all(
                    record["status"] == "disabled_not_applicable"
                    for record in insurance_gating
                    if record["tool_name"].startswith("standard_corporate_")
                )
            )
            self.assertTrue(
                all(
                    record["metadata"]["insurance_subprofile"] == "non_life"
                    for record in insurance_gating
                    if record["tool_name"].startswith("insurance_")
                )
            )

            insurance_findings = [
                finding
                for finding in result.tool_findings
                if finding["report_id"] == insurance_set.target.report_id
            ]
            findings_by_tool = {finding["tool_name"]: finding for finding in insurance_findings}
            self.assertEqual(
                findings_by_tool["insurance_premium_receivable_coherence_tool"]["signal_id"],
                "premium_receivables_growth_outpaces_premium_growth",
            )
            self.assertEqual(
                findings_by_tool["insurance_reserve_movement_tool"]["signal_id"],
                "claims_reserve_growth_lags_claims_and_premium_exposure",
            )
            self.assertEqual(
                findings_by_tool["insurance_reinsurance_balance_tool"]["signal_id"],
                "reinsurance_recoverables_expand_with_weak_cash_support",
            )
            self.assertIn(
                "reserve_movement_ratio",
                findings_by_tool["insurance_reserve_movement_tool"]["metric"]["secondary_values"],
            )

            insurance_candidates = [
                candidate
                for candidate in result.candidate_risks
                if candidate["report_id"] == insurance_set.target.report_id
            ]
            self.assertEqual(
                {candidate["risk_category"] for candidate in insurance_candidates},
                {
                    "receivables_credit_quality_risk",
                    "expense_liability_understatement_risk",
                    "earnings_cashflow_mismatch",
                },
            )
            self.assertFalse(
                any(
                    candidate["supporting_signal_ids"] == ["reserve_note_absent"]
                    for candidate in insurance_candidates
                )
            )

            insurance_packets = [
                packet
                for packet in result.detector_packets
                if packet["report_id"] == insurance_set.target.report_id
            ]
            self.assertTrue(insurance_packets)
            for packet in insurance_packets:
                self.assertEqual(packet["metadata"]["report_profile"], "insurance")
                self.assertEqual(packet["metadata"]["insurance_subprofile"], "non_life")
                self.assertEqual(packet["report_set_id"], "signal_present_insurance")
                self.assertEqual(
                    packet["task"]["risk_category"],
                    next(
                        candidate["risk_category"]
                        for candidate in insurance_candidates
                        if candidate["candidate_id"] == packet["candidate_id"]
                    ),
                )
                self.assertLessEqual(len(packet["tool_findings"]), 5)
                self.assertLessEqual(len(packet["rules"]), 3)
                self.assertTrue(packet["relevant_table_rows"] or packet["relevant_notes"])

            insurance_report = next(
                report for report in result.final_reports if report["target_report_id"] == insurance_set.target.report_id
            )
            self.assertEqual(insurance_report["metadata"]["report_profile"], "insurance")
            self.assertEqual(insurance_report["metadata"]["insurance_subprofile"], "non_life")
            self.assertTrue(insurance_report["reviewed_candidate_audit"])
            self.assertTrue((output_dir / f"FINAL_{insurance_set.target.report_id}.json").exists())
            self.assertTrue((output_dir / f"FINAL_{insurance_set.target.report_id}.md").exists())

            all_insurance_text = json.dumps(
                {
                    "findings": insurance_findings,
                    "candidates": insurance_candidates,
                    "packets": insurance_packets,
                    "report": insurance_report,
                }
            ).lower()
            for prohibited in [
                "fraud",
                "manipulat",
                "conceal",
                "intent",
                "illegal",
                "legal misstatement",
                "actuarial adequacy",
                "reserve adequacy",
            ]:
                self.assertNotIn(prohibited, all_insurance_text)

    def test_wave2_securities_fixture_track_runs_through_existing_spine(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            result = run_fixture_spine(output_dir)

            self.assertIn("signal_present_securities", result.loaded_report_sets)
            securities_set = next(
                report_set
                for report_set in result.report_sets
                if report_set.case_id == "signal_present_securities"
            )
            self.assertEqual(securities_set.target.metadata["report_profile"], "securities")
            self.assertEqual(securities_set.prior_year.metadata["report_profile"], "securities")
            self.assertEqual(securities_set.target.metadata["report_basis"], "consolidated")
            self.assertEqual(securities_set.prior_year.metadata["report_basis"], "consolidated")
            self.assertEqual(
                securities_set.target.metadata["company_name"],
                securities_set.prior_year.metadata["company_name"],
            )
            self.assertEqual(securities_set.target.quarter, securities_set.prior_year.quarter)
            self.assertEqual(securities_set.target.year, securities_set.prior_year.year + 1)

            securities_gating = [
                record
                for record in result.tool_gating_records
                if record["case_id"] == "signal_present_securities"
            ]
            enabled_tool_names = {
                record["tool_name"] for record in securities_gating if record["status"] == "enabled"
            }
            self.assertGreaterEqual(
                enabled_tool_names,
                {
                    "ctck_margin_book_quality_tool",
                    "ctck_trading_book_valuation_bridge_tool",
                    "ctck_earnings_cash_bridge_tool",
                    "ctck_disclosure_consistency_tool",
                },
            )
            self.assertTrue(
                all(
                    record["status"] == "disabled_not_applicable"
                    for record in securities_gating
                    if record["tool_name"].startswith("standard_corporate_")
                )
            )

            securities_findings = [
                finding
                for finding in result.tool_findings
                if finding["report_id"] == securities_set.target.report_id
            ]
            findings_by_tool = {finding["tool_name"]: finding for finding in securities_findings}
            self.assertEqual(
                findings_by_tool["ctck_margin_book_quality_tool"]["signal_id"],
                "ctck_margin_book_provision_gap",
            )
            self.assertEqual(
                findings_by_tool["ctck_trading_book_valuation_bridge_tool"]["signal_id"],
                "ctck_trading_book_valuation_concentration_with_weak_disclosure",
            )
            self.assertEqual(
                findings_by_tool["ctck_earnings_cash_bridge_tool"]["signal_id"],
                "ctck_profit_supported_by_noncash_fvtpl_with_weak_cash_support",
            )
            self.assertEqual(
                findings_by_tool["ctck_disclosure_consistency_tool"]["signal_id"],
                "ctck_fvtpl_volatility_context_only",
            )
            self.assertFalse(findings_by_tool["ctck_disclosure_consistency_tool"]["flag"])

            securities_candidates = [
                candidate
                for candidate in result.candidate_risks
                if candidate["report_id"] == securities_set.target.report_id
            ]
            self.assertEqual(
                {candidate["risk_category"] for candidate in securities_candidates},
                {
                    "receivables_credit_quality_risk",
                    "asset_quality_valuation_risk",
                    "earnings_cashflow_mismatch",
                },
            )
            self.assertFalse(
                any(
                    candidate["supporting_signal_ids"] == ["ctck_fvtpl_volatility_context_only"]
                    for candidate in securities_candidates
                )
            )

            securities_packets = [
                packet
                for packet in result.detector_packets
                if packet["report_id"] == securities_set.target.report_id
            ]
            self.assertTrue(securities_packets)
            for packet in securities_packets:
                self.assertEqual(packet["metadata"]["report_profile"], "securities")
                self.assertEqual(packet["report_set_id"], "signal_present_securities")
                self.assertEqual(
                    packet["task"]["risk_category"],
                    next(
                        candidate["risk_category"]
                        for candidate in securities_candidates
                        if candidate["candidate_id"] == packet["candidate_id"]
                    ),
                )
                self.assertLessEqual(len(packet["tool_findings"]), 5)
                self.assertLessEqual(len(packet["rules"]), 3)
                self.assertTrue(packet["relevant_table_rows"])

            securities_report = next(
                report for report in result.final_reports if report["target_report_id"] == securities_set.target.report_id
            )
            self.assertEqual(securities_report["metadata"]["report_profile"], "securities")
            self.assertTrue(securities_report["reviewed_candidate_audit"])
            self.assertTrue((output_dir / f"FINAL_{securities_set.target.report_id}.json").exists())
            self.assertTrue((output_dir / f"FINAL_{securities_set.target.report_id}.md").exists())

            all_securities_text = json.dumps(
                {
                    "findings": securities_findings,
                    "candidates": securities_candidates,
                    "packets": securities_packets,
                    "report": securities_report,
                }
            ).lower()
            for prohibited in ["fraud", "manipulat", "conceal", "intent", "illegal", "legal misstatement"]:
                self.assertNotIn(prohibited, all_securities_text)

    def test_ctck_margin_book_tool_handles_zero_comparison_impairment_without_crashing(self):
        fixture_case = _securities_fixture_case()
        fixture_case["case_id"] = "ctck_zero_comparison_impairment"
        for table in fixture_case["prior_year_report_memory"]["tables"]:
            for row in table["rows"]:
                if row.get("standard_account") != "margin_impairment":
                    continue
                row["cells"][0]["value"] = 0

        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_fixture_spine(Path(temp_dir), fixture_cases=[fixture_case])

        margin_finding = next(
            finding
            for finding in result.tool_findings
            if finding["tool_name"] == "ctck_margin_book_quality_tool"
        )
        self.assertEqual(margin_finding["signal_id"], "ctck_margin_book_growth_not_computable")
        self.assertIsNone(margin_finding["metric"]["value"])
        self.assertFalse(margin_finding["flag"])
        self.assertNotIn(
            "receivables_credit_quality_risk",
            {candidate["risk_category"] for candidate in result.candidate_risks},
        )

    def test_wave2_credit_institution_fixture_track_runs_through_existing_spine(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            result = run_fixture_spine(output_dir)

            self.assertIn("signal_present_credit_institution", result.loaded_report_sets)
            credit_set = next(
                report_set
                for report_set in result.report_sets
                if report_set.case_id == "signal_present_credit_institution"
            )
            self.assertEqual(credit_set.target.metadata["report_profile"], "credit_institution")
            self.assertEqual(credit_set.prior_year.metadata["report_profile"], "credit_institution")
            self.assertEqual(credit_set.target.metadata["report_basis"], "consolidated")
            self.assertEqual(credit_set.prior_year.metadata["report_basis"], "consolidated")
            self.assertEqual(credit_set.target.metadata["company_name"], credit_set.prior_year.metadata["company_name"])
            self.assertEqual(credit_set.target.quarter, credit_set.prior_year.quarter)
            self.assertEqual(credit_set.target.year, credit_set.prior_year.year + 1)

            credit_gating = [
                record
                for record in result.tool_gating_records
                if record["case_id"] == "signal_present_credit_institution"
            ]
            enabled_tool_names = {
                record["tool_name"] for record in credit_gating if record["status"] == "enabled"
            }
            self.assertGreaterEqual(
                enabled_tool_names,
                {
                    "credit_institution_loan_quality_tool",
                    "credit_institution_provision_movement_tool",
                },
            )
            self.assertTrue(
                all(
                    record["status"] == "disabled_not_applicable"
                    for record in credit_gating
                    if record["tool_name"].startswith("standard_corporate_")
                )
            )

            credit_findings = [
                finding
                for finding in result.tool_findings
                if finding["report_id"] == credit_set.target.report_id
            ]
            findings_by_tool = {finding["tool_name"]: finding for finding in credit_findings}
            loan_quality = findings_by_tool["credit_institution_loan_quality_tool"]
            provision = findings_by_tool["credit_institution_provision_movement_tool"]
            self.assertEqual(loan_quality["signal_id"], "loan_quality_deterioration")
            self.assertEqual(provision["signal_id"], "provision_growth_lags_risk_assets")
            self.assertIn("npl_ratio_current", loan_quality["metric"]["secondary_values"])
            self.assertIn("general_provision_growth_pct", provision["metric"]["secondary_values"])
            self.assertIn("specific_provision_growth_pct", provision["metric"]["secondary_values"])

            credit_candidates = [
                candidate
                for candidate in result.candidate_risks
                if candidate["report_id"] == credit_set.target.report_id
            ]
            self.assertEqual(
                {candidate["risk_category"] for candidate in credit_candidates},
                {"receivables_credit_quality_risk", "expense_liability_understatement_risk"},
            )
            self.assertFalse(
                any("restructuring_context_present" in candidate["supporting_signal_ids"] for candidate in credit_candidates)
            )

            credit_packets = [
                packet
                for packet in result.detector_packets
                if packet["report_id"] == credit_set.target.report_id
            ]
            self.assertTrue(credit_packets)
            for packet in credit_packets:
                self.assertEqual(packet["metadata"]["report_profile"], "credit_institution")
                self.assertEqual(packet["report_set_id"], "signal_present_credit_institution")
                self.assertIn(packet["task"]["risk_category"], {candidate["risk_category"] for candidate in credit_candidates})
                self.assertLessEqual(len(packet["tool_findings"]), 5)
                self.assertEqual(len({packet["candidate_id"]}), 1)

            credit_report = next(
                report for report in result.final_reports if report["target_report_id"] == credit_set.target.report_id
            )
            self.assertEqual(credit_report["metadata"]["report_profile"], "credit_institution")
            self.assertTrue(credit_report["reviewed_candidate_audit"])
            self.assertTrue((output_dir / f"FINAL_{credit_set.target.report_id}.json").exists())
            self.assertTrue((output_dir / f"FINAL_{credit_set.target.report_id}.md").exists())

            all_credit_text = json.dumps(
                {
                    "findings": credit_findings,
                    "candidates": credit_candidates,
                    "packets": credit_packets,
                    "report": credit_report,
                }
            ).lower()
            for prohibited in [
                "fraud",
                "manipulat",
                "conceal",
                "intent",
                "illegal",
                "legal misstatement",
                "compliance",
                "required provision rate",
            ]:
                self.assertNotIn(prohibited, all_credit_text)

    def test_wave1_exit_demo_public_command_regenerates_complete_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            env = {**os.environ, "PYTHONPATH": str(Path.cwd() / "src")}
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "research.fixture_spine",
                    "--output-dir",
                    str(output_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )

            command_summary = json.loads(completed.stdout)
            manifest_path = output_dir / "wave1_exit_demo_manifest.json"
            self.assertEqual(command_summary["status"], "completed")
            self.assertEqual(command_summary["exit_demo_manifest_path"], str(manifest_path))

            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["demo_id"], "wave1_exit_demo")
            self.assertGreaterEqual(
                manifest["fixture_cases"],
                ["signal_present_standard_corporate", "clean_standard_corporate"],
            )
            self.assertIn("signal_present_credit_institution", manifest["fixture_cases"])
            self.assertEqual(
                set(manifest["scope"]["report_profiles"]),
                {"standard_corporate", "credit_institution", "securities", "insurance"},
            )
            self.assertEqual(manifest["scope"]["input_mode"], "fixture_only")
            self.assertFalse(manifest["scope"]["requires_optional_api_detector"])
            for excluded in [
                "real_pdf_or_ocr_extraction",
                "source_fetching",
                "external_data",
                "detector_sft_or_training_data",
                "full_thesis_evaluation_experiments",
                "current_filing_prior_comparison_fallback",
            ]:
                self.assertIn(excluded, manifest["scope"]["excluded"])

            required_artifacts = {
                "report_memories",
                "company_report_sets",
                "tool_gating_records",
                "tool_findings",
                "candidate_risks",
                "detector_packets",
                "detector_assessments",
                "aggregation_output",
                "final_json_reports",
                "final_markdown_reports",
                "append_only_logs",
                "thesis_architecture_notes",
                "thesis_pipeline_flow",
                "thesis_contract_rationale",
                "language_compliance_examples",
            }
            self.assertLessEqual(required_artifacts, set(manifest["artifacts"]))
            for artifact_name in required_artifacts:
                artifact_value = manifest["artifacts"][artifact_name]
                if isinstance(artifact_value, str):
                    artifact_paths = [artifact_value]
                elif isinstance(artifact_value, list):
                    artifact_paths = artifact_value
                else:
                    artifact_paths = list(artifact_value.values())
                for artifact_path in artifact_paths:
                    self.assertTrue((output_dir / artifact_path).exists(), artifact_path)

            report_memories = json.loads((output_dir / manifest["artifacts"]["report_memories"]).read_text())
            self.assertEqual(len(report_memories), 10)
            self.assertEqual(
                {report["metadata"]["report_profile"] for report in report_memories},
                {"standard_corporate", "credit_institution", "securities", "insurance"},
            )

            company_report_sets = json.loads((output_dir / manifest["artifacts"]["company_report_sets"]).read_text())
            self.assertGreaterEqual(
                [report_set["case_id"] for report_set in company_report_sets],
                ["signal_present_standard_corporate", "clean_standard_corporate"],
            )

            detector_assessments = json.loads(
                (output_dir / manifest["artifacts"]["detector_assessments"]).read_text()
            )
            self.assertGreaterEqual(
                {assessment["support_level"] for assessment in detector_assessments},
                {"supported", "weakly_supported"},
            )

            aggregation_output = json.loads(
                (output_dir / manifest["artifacts"]["aggregation_output"]).read_text()
            )
            self.assertEqual(
                [report["target_report_id"] for report in aggregation_output["final_reports"]],
                [
                    "VINACOUNT_SIGNAL_2025_Q3",
                    "VINACOUNT_CLEAN_2025_Q3",
                    "VINABANK_SIGNAL_2025_Q3",
                    "VINASEC_SIGNAL_2025_Q3",
                    "VINAINS_SIGNAL_2025_Q3",
                ],
            )

            all_demo_text = "\n".join(
                path.read_text(encoding="utf-8")
                for path in output_dir.rglob("*")
                if path.is_file() and path.suffix in {".json", ".jsonl", ".md"}
            ).lower()
            for prohibited in ["fraud", "manipulat", "conceal", "intent", "illegal", "legal misstatement"]:
                self.assertNotIn(prohibited, all_demo_text)

    def test_api_detector_adapter_is_disabled_by_default_and_deterministic_detector_remains_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_fixture_spine(Path(temp_dir))

            self.assertEqual(len(result.detector_assessments), len(result.detector_packets))
            self.assertEqual(len(result.detector_assessment_audit_records), len(result.detector_packets))
            for record in result.detector_assessment_audit_records:
                self.assertEqual(record["detector_name"], "deterministic_wave1_detector")
                self.assertEqual(record["api_detector"]["provider"], "openrouter")
                self.assertEqual(record["api_detector"]["model"], "deepseek/deepseek-v4-flash")
                self.assertEqual(record["api_detector"]["status"], "disabled")
                self.assertEqual(record["api_detector"]["reason_code"], "api_detector_not_explicitly_enabled")
                self.assertFalse(record["api_detector"]["network_call_attempted"])
                self.assertFalse(record["api_detector"]["credentials_logged"])
                self.assertFalse(record["raw_prompt_stored"])
                self.assertFalse(record["raw_response_stored"])
                self.assertFalse(record["long_reasoning_stored"])

            method_scope = result.final_reports[0]["method_and_scope"]
            self.assertTrue(
                any(
                    "Optional API LLM detector adapter" in scope_item
                    for scope_item in method_scope["excluded_scope"]
                )
            )

    def test_configured_api_detector_uses_same_packet_and_validates_same_assessment_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_fixture_spine(Path(temp_dir))

        packet = result.detector_packets[0]
        api_assessment = dict(result.detector_assessments[0])
        api_assessment["assessment_id"] = f"API_{api_assessment['assessment_id']}"
        response_payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(api_assessment),
                    }
                }
            ]
        }

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self):
                return json.dumps(response_payload).encode("utf-8")

        audit_records = []
        with mock.patch.dict(
            os.environ,
            {
                "VINACOUNT_API_DETECTOR_ENABLED": "true",
                "OPENROUTER_API_KEY": "test-secret-not-logged",
                "VINACOUNT_API_DETECTOR_DEBUG_RAW": "false",
            },
            clear=False,
        ), mock.patch("urllib.request.urlopen", return_value=Response()) as urlopen:
            assessments = assess_detector_packets([packet], audit_records)

        self.assertEqual(assessments, [api_assessment])
        self.assertEqual(audit_records[0]["detector_name"], "openrouter_api_detector")
        self.assertEqual(audit_records[0]["api_detector"]["status"], "enabled")
        self.assertEqual(audit_records[0]["api_detector"]["provider"], "openrouter")
        self.assertEqual(audit_records[0]["api_detector"]["model"], "deepseek/deepseek-v4-flash")
        self.assertTrue(audit_records[0]["api_detector"]["network_call_attempted"])
        self.assertFalse(audit_records[0]["api_detector"]["credentials_logged"])
        self.assertFalse(audit_records[0]["raw_prompt_stored"])
        self.assertFalse(audit_records[0]["raw_response_stored"])
        self.assertFalse(audit_records[0]["long_reasoning_stored"])
        self.assertEqual(urlopen.call_count, 1)
        request = urlopen.call_args.args[0]
        self.assertNotIn("test-secret-not-logged", json.dumps(audit_records))
        self.assertIn("Bearer test-secret-not-logged", request.headers["Authorization"])

    def test_configured_api_detector_rejects_malformed_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_fixture_spine(Path(temp_dir))

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self):
                return json.dumps({"choices": [{"message": {"content": "not json"}}]}).encode("utf-8")

        with mock.patch.dict(
            os.environ,
            {
                "VINACOUNT_API_DETECTOR_ENABLED": "true",
                "OPENROUTER_API_KEY": "test-secret-not-logged",
            },
            clear=False,
        ), mock.patch("urllib.request.urlopen", return_value=Response()):
            with self.assertRaisesRegex(ValueError, "malformed assessment output"):
                assess_detector_packets([result.detector_packets[0]], [])

    def test_fixture_spine_produces_final_json_and_markdown_reports_from_detector_assessments(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_fixture_spine(Path(temp_dir))

            self.assertTrue(result.final_report_paths)
            self.assertTrue(result.final_reports)

            report = result.final_reports[0]
            self.assertEqual(report["metadata"]["report_basis"], "consolidated")
            self.assertEqual(report["metadata"]["filing_status"], "original")
            self.assertEqual(
                report["metadata"]["canonical_source_document_id"],
                "DOC_VCF_2025_Q3_FS_001",
            )
            self.assertEqual(report["metadata"]["business_context_tags"], ["manufacturing_inventory"])

            finding_assessment_ids = {
                assessment_id
                for finding in report["grouped_findings"]
                for assessment_id in finding["assessment_ids"]
            }
            weak_assessment_ids = {
                item["assessment_id"] for item in report["weak_or_limited_signals"]
            }
            audit_by_assessment_id = {
                item["assessment_id"]: item for item in report["reviewed_candidate_audit"]
            }
            first_report_assessment_ids = set(audit_by_assessment_id)
            for assessment in [
                assessment
                for assessment in result.detector_assessments
                if assessment["assessment_id"] in first_report_assessment_ids
            ]:
                audit_item = audit_by_assessment_id[assessment["assessment_id"]]
                self.assertEqual(audit_item["candidate_id"], assessment["candidate_id"])
                self.assertEqual(audit_item["risk_category"], assessment["risk_category"])
                self.assertEqual(audit_item["support_level"], assessment["support_level"])
                self.assertEqual(audit_item["evidence_refs"], assessment["cited_evidence_refs"])

                if assessment["support_level"] == "supported":
                    self.assertIn(assessment["assessment_id"], finding_assessment_ids)
                elif assessment["support_level"] == "weakly_supported":
                    self.assertIn(assessment["assessment_id"], weak_assessment_ids)
                else:
                    self.assertNotIn(assessment["assessment_id"], finding_assessment_ids)
                    self.assertNotIn(assessment["assessment_id"], weak_assessment_ids)

            for finding in report["grouped_findings"]:
                self.assertTrue(finding["assessment_ids"])
                self.assertTrue(finding["candidate_ids"])
                self.assertTrue(finding["risk_categories"])
                self.assertTrue(finding["support_levels"])
                self.assertTrue(finding["tool_refs"])
                self.assertTrue(finding["evidence_refs"])

            json_report_path = result.final_report_paths[0]["json"]
            markdown_report_path = result.final_report_paths[0]["markdown"]
            self.assertEqual(json.loads(json_report_path.read_text()), report)

            markdown = markdown_report_path.read_text()
            self.assertIn("# Accounting Irregularity Risk-Signal Review", markdown)
            self.assertIn("## Reviewed Candidate Audit Section", markdown)
            self.assertIn("## Limitations and Run Status Notes", markdown)
            final_text = (json.dumps(report) + markdown).lower()
            for prohibited in ["fraud", "manipulat", "conceal", "intent", "illegal", "legal misstatement"]:
                self.assertNotIn(prohibited, final_text)

    def test_final_report_keeps_not_supported_and_insufficient_evidence_out_of_findings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_fixture_spine(Path(temp_dir))

            signal_report_set = result.report_sets[0]
            signal_candidates = [
                candidate
                for candidate in result.candidate_risks
                if candidate["report_id"] == signal_report_set.target.report_id
            ]
            assessments = json.loads(json.dumps(result.detector_assessments[:2]))
            assessments[0]["support_level"] = "not_supported"
            assessments[0]["severity"] = "unknown"
            assessments[0]["validated_signals"] = []
            assessments[0]["cited_evidence_refs"] = []
            assessments[0]["rationale_short"] = "The provided packet evidence does not support the candidate risk signal."
            assessments[1]["support_level"] = "insufficient_evidence"
            assessments[1]["severity"] = "unknown"
            assessments[1]["validated_signals"] = []
            assessments[1]["cited_evidence_refs"] = []
            assessments[1]["rationale_short"] = "The packet lacks enough visible evidence to assess the candidate risk signal."

            report = build_final_report(
                signal_report_set,
                signal_candidates,
                result.tool_findings,
                assessments,
                result.tool_gating_records,
            )

            self.assertEqual(report["grouped_findings"], [])
            self.assertEqual(report["weak_or_limited_signals"], [])
            self.assertEqual(
                {item["support_level"] for item in report["reviewed_candidate_audit"]},
                {"not_supported", "insufficient_evidence"},
            )
            self.assertEqual(
                [item["support_level"] for item in report["insufficient_evidence_or_data_gaps"]],
                ["insufficient_evidence"],
            )
            self.assertEqual(
                report["overall_assessment"]["overall_review_status"],
                "insufficient_evidence_for_overall_assessment",
            )

    def test_final_report_validation_rejects_prohibited_user_facing_language(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_fixture_spine(Path(temp_dir))

            report = json.loads(json.dumps(result.final_reports[0]))
            report["executive_summary"].append("This text uses prohibited fraud wording.")

            with self.assertRaisesRegex(ValueError, "prohibited"):
                validate_final_report(report)

    def test_fixture_spine_produces_grounded_detector_assessments_for_packets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_fixture_spine(Path(temp_dir))

            self.assertEqual(len(result.detector_assessments), len(result.detector_packets))

            packets_by_id = {packet["packet_id"]: packet for packet in result.detector_packets}
            for assessment in result.detector_assessments:
                packet = packets_by_id[assessment["packet_id"]]
                visible_evidence_ids = _detector_visible_evidence_ids(packet)

                self.assertEqual(assessment["candidate_id"], packet["candidate_id"])
                self.assertEqual(assessment["risk_category"], packet["task"]["risk_category"])
                self.assertIn(
                    assessment["support_level"],
                    {"supported", "weakly_supported", "not_supported", "insufficient_evidence"},
                )
                self.assertIn("confidence", assessment)
                self.assertIn("severity", assessment)
                self.assertIn("validated_signals", assessment)
                self.assertIn("cited_evidence_refs", assessment)
                self.assertIn("rationale_short", assessment)
                self.assertLessEqual(len(assessment["rationale_short"].split(".")), 4)
                self.assertLessEqual(
                    {ref["ref_id"] for ref in assessment["cited_evidence_refs"]},
                    visible_evidence_ids,
                )

            events = [json.loads(line) for line in result.audit_log_path.read_text().splitlines()]
            assessment_events = [
                event for event in events if event["stage"] == "detector_assessment_completed"
            ]
            self.assertEqual(len(assessment_events), 5)
            self.assertEqual(
                [
                    detail["assessment_id"]
                    for event in assessment_events
                    for detail in event["details"]["assessments"]
                ],
                [assessment["assessment_id"] for assessment in result.detector_assessments],
            )
            assessment_sidechain = [
                json.loads(line) for line in result.detector_assessment_log_path.read_text().splitlines()
            ]
            self.assertEqual(
                [record["details"]["assessment_id"] for record in assessment_sidechain],
                [assessment["assessment_id"] for assessment in result.detector_assessments],
            )

    def test_deterministic_detector_outputs_compact_fields_and_call_metadata_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_fixture_spine(Path(temp_dir))

            self.assertGreaterEqual(
                {assessment["support_level"] for assessment in result.detector_assessments},
                {"supported", "weakly_supported"},
            )
            for assessment in result.detector_assessments:
                self.assertEqual(
                    set(assessment),
                    {
                        "assessment_id",
                        "packet_id",
                        "candidate_id",
                        "report_id",
                        "risk_category",
                        "support_level",
                        "confidence",
                        "severity",
                        "validated_signals",
                        "cited_evidence_refs",
                        "rationale_short",
                    },
                )
                self.assertTrue(assessment["validated_signals"])
                self.assertTrue(assessment["cited_evidence_refs"])
                self.assertNotIn("reasoning", json.dumps(assessment).lower())
                self.assertNotIn("chain", json.dumps(assessment).lower())

            self.assertEqual(len(result.detector_assessment_audit_records), len(result.detector_packets))
            for record in result.detector_assessment_audit_records:
                self.assertEqual(record["detector_name"], "deterministic_wave1_detector")
                self.assertFalse(record["raw_prompt_stored"])
                self.assertFalse(record["raw_response_stored"])
                self.assertFalse(record["long_reasoning_stored"])

    def test_detector_assessment_validation_rejects_category_mismatch_and_outside_evidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_fixture_spine(Path(temp_dir))

            packet = result.detector_packets[0]
            valid_assessment = result.detector_assessments[0]

            category_mismatch = dict(valid_assessment)
            category_mismatch["risk_category"] = "asset_quality_valuation_risk"
            with self.assertRaisesRegex(ValueError, "risk_category must match"):
                validate_detector_assessment(category_mismatch, packet)

            outside_evidence = dict(valid_assessment)
            outside_evidence["cited_evidence_refs"] = [
                *valid_assessment["cited_evidence_refs"],
                {
                    "evidence_ref_type": "table_cell",
                    "ref_id": "OUTSIDE_REPORT:OUTSIDE_CELL",
                    "role": "supporting",
                },
            ]
            with self.assertRaisesRegex(ValueError, "cited evidence must be visible"):
                validate_detector_assessment(outside_evidence, packet)

            prohibited_language = dict(valid_assessment)
            prohibited_language["rationale_short"] = "This output uses prohibited fraud wording."
            with self.assertRaisesRegex(ValueError, "prohibited"):
                validate_detector_assessment(prohibited_language, packet)

    def test_deterministic_detector_returns_conservative_negative_support_levels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_fixture_spine(Path(temp_dir))

            not_supported_packet = json.loads(json.dumps(result.detector_packets[0]))
            for finding in not_supported_packet["tool_findings"]:
                finding["flag"] = False

            insufficient_packet = json.loads(json.dumps(not_supported_packet))
            insufficient_packet["packet_id"] = "PACKET_EMPTY_VISIBLE_EVIDENCE"
            insufficient_packet["tool_findings"] = []
            insufficient_packet["relevant_table_rows"] = []
            insufficient_packet["relevant_notes"] = []
            insufficient_packet["relevant_variance_explanations"] = []

            assessments = assess_detector_packets([not_supported_packet, insufficient_packet])

            self.assertEqual(
                [assessment["support_level"] for assessment in assessments],
                ["not_supported", "insufficient_evidence"],
            )
            self.assertEqual(assessments[0]["validated_signals"][0]["status"], "rejected")
            self.assertEqual(assessments[1]["validated_signals"][0]["status"], "not_assessable")
            self.assertEqual(assessments[1]["cited_evidence_refs"][0]["evidence_ref_type"], "rule")

    def test_detector_assessment_text_uses_risk_signal_language(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_fixture_spine(Path(temp_dir))

            detector_assessment_text = json.dumps(result.detector_assessments).lower()
            for prohibited in ["fraud", "manipulat", "conceal", "intent", "illegal", "legal misstatement"]:
                self.assertNotIn(prohibited, detector_assessment_text)

    def test_fixture_spine_builds_detector_packets_for_generated_candidates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_fixture_spine(Path(temp_dir))

            self.assertEqual(len(result.detector_packets), len(result.candidate_risks))

            candidates_by_id = {candidate["candidate_id"]: candidate for candidate in result.candidate_risks}
            tool_findings_by_id = {
                finding["tool_result_id"]: finding for finding in result.tool_findings
            }
            for packet in result.detector_packets:
                candidate = candidates_by_id[packet["candidate_id"]]

                self.assertEqual(packet["report_id"], candidate["report_id"])
                self.assertEqual(packet["task"]["risk_category"], candidate["risk_category"])
                self.assertNotIn("secondary_risk_categories", packet["task"])
                self.assertEqual(
                    packet["candidate_summary"]["supporting_signal_ids"],
                    candidate["supporting_signal_ids"],
                )

                self.assertEqual(
                    [finding["tool_result_id"] for finding in packet["tool_findings"]],
                    candidate["linked_tool_result_ids"],
                )
                self.assertTrue(packet["relevant_table_rows"])
                self.assertTrue(
                    all(row["report_id"] and row["local_evidence_id"] for row in packet["relevant_table_rows"])
                )
                self.assertTrue(
                    all(
                        ref["report_id"] and ref["local_evidence_id"]
                        for finding_id in candidate["linked_tool_result_ids"]
                        for ref in tool_findings_by_id[finding_id]["evidence_refs"]
                        if ref["evidence_ref_type"] == "table_cell"
                    )
                )

            events = [json.loads(line) for line in result.audit_log_path.read_text().splitlines()]
            packet_events = [event for event in events if event["stage"] == "detector_packet_build_completed"]
            self.assertEqual(len(packet_events), 5)
            self.assertEqual(
                [
                    detail["packet_id"]
                    for event in packet_events
                    for detail in event["details"]["packets"]
                ],
                [packet["packet_id"] for packet in result.detector_packets],
            )
            packet_sidechain = [
                json.loads(line) for line in result.detector_packet_log_path.read_text().splitlines()
            ]
            self.assertEqual(
                [record["details"]["packet_id"] for record in packet_sidechain],
                [packet["packet_id"] for packet in result.detector_packets],
            )

    def test_detector_packets_enforce_caps_and_keep_omissions_audit_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_fixture_spine(Path(temp_dir))

            for packet in result.detector_packets:
                self.assertLessEqual(len(packet["relevant_table_rows"]), 12)
                self.assertLessEqual(len(packet["relevant_notes"]), 3)
                self.assertLessEqual(len(packet["relevant_variance_explanations"]), 2)
                self.assertLessEqual(len(packet["tool_findings"]), 5)
                self.assertLessEqual(len(packet["rules"]), 3)
                self.assertNotIn("omitted_evidence_records", packet)
                self.assertNotIn("omitted_evidence_ids", json.dumps(packet))

            events = [json.loads(line) for line in result.audit_log_path.read_text().splitlines()]
            for event in events:
                if event["stage"] == "detector_packet_build_completed":
                    self.assertIn("omitted_evidence_records", event["details"])
            self.assertEqual(result.detector_packet_audit_records, [])

    def test_detector_packet_validator_rejects_raw_extraction_payloads(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_fixture_spine(Path(temp_dir))

            packet = json.loads(json.dumps(result.detector_packets[0]))
            packet["relevant_table_rows"][0]["raw_ocr_text"] = "full OCR page text must stay raw artifact only"
            packet["relevant_table_rows"][0]["raw_tables"] = [[["account", "current", "prior"]]]
            packet["relevant_table_rows"][0]["raw_coordinates"] = {
                "x0": 10,
                "y0": 20,
                "x1": 300,
                "y1": 160,
            }

            with self.assertRaisesRegex(ValueError, "raw extraction payload"):
                validate_detector_packet(packet)

    def test_candidate_that_cannot_fit_required_packet_caps_is_discarded_with_audit_reason(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_fixture_spine(Path(temp_dir))

            candidate = dict(result.candidate_risks[0])
            candidate["candidate_id"] = "CAND_TOO_BROAD_FOR_PACKET"
            candidate["required_evidence_refs"] = [
                {
                    "evidence_ref_type": "note_span",
                    "ref_id": f"VINACOUNT_SIGNAL_2025_Q3:NOTE_REQUIRED_{index}",
                    "report_id": "VINACOUNT_SIGNAL_2025_Q3",
                    "local_evidence_id": f"NOTE_REQUIRED_{index}",
                    "role": "required_for_review",
                }
                for index in range(4)
            ]
            audit_records = []

            packets = build_detector_packets(
                result.report_sets[0],
                [candidate],
                result.tool_findings,
                audit_records,
            )

            self.assertEqual(packets, [])
            self.assertEqual(
                audit_records,
                [
                    {
                        "candidate_id": "CAND_TOO_BROAD_FOR_PACKET",
                        "report_id": "VINACOUNT_SIGNAL_2025_Q3",
                        "reason_code": "too_broad_for_detector_packet",
                        "reason": "Required relevant_notes count 4 exceeds hard cap 3.",
                        "candidate_status": "discarded_by_agent",
                    }
                ],
            )

    def test_detector_packet_text_uses_risk_signal_language(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_fixture_spine(Path(temp_dir))

            detector_visible_text = json.dumps(result.detector_packets).lower()
            for prohibited in ["fraud", "manipulat", "conceal", "intent", "illegal", "legal misstatement"]:
                self.assertNotIn(prohibited, detector_visible_text)

    def test_fixture_spine_runs_disclosure_and_variance_explanation_slice(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_fixture_spine(Path(temp_dir))

            signal_findings = [
                finding
                for finding in result.tool_findings
                if finding["report_id"] == "VINACOUNT_SIGNAL_2025_Q3"
            ]
            signal_candidates = [
                candidate
                for candidate in result.candidate_risks
                if candidate["report_id"] == "VINACOUNT_SIGNAL_2025_Q3"
            ]

            gating_records = {
                record["tool_name"]: record
                for record in result.tool_gating_records
                if record["case_id"] == "signal_present_standard_corporate"
            }
            self.assertEqual(
                gating_records["standard_corporate_disclosure_consistency_tool"]["status"],
                "enabled",
            )
            self.assertEqual(
                gating_records["standard_corporate_variance_explanation_quality_tool"]["status"],
                "enabled",
            )

            findings_by_tool = {finding["tool_name"]: finding for finding in signal_findings}
            disclosure = findings_by_tool["standard_corporate_disclosure_consistency_tool"]
            variance = findings_by_tool["standard_corporate_variance_explanation_quality_tool"]

            self.assertEqual(disclosure["status"], "completed")
            self.assertEqual(variance["status"], "completed")
            self.assertEqual(disclosure["tool_category"], "consistency")
            self.assertEqual(variance["tool_category"], "disclosure")
            self.assertTrue(disclosure["flag"])
            self.assertTrue(variance["flag"])
            self.assertEqual(disclosure["risk_category"], "disclosure_inconsistency_or_obfuscation")
            self.assertEqual(variance["risk_category"], "disclosure_inconsistency_or_obfuscation")
            self.assertEqual(variance["metric"]["checks"]["concrete_driver"], True)
            self.assertEqual(variance["metric"]["checks"]["connected_to_changed_metric"], False)
            self.assertEqual(variance["metric"]["checks"]["directionally_consistent"], True)
            self.assertEqual(variance["metric"]["checks"]["non_boilerplate"], True)
            self.assertEqual(variance["metric"]["checks"]["missing_explanation_for_material_change"], False)
            self.assertIn(
                {"evidence_ref_type": "variance_explanation_span", "local_evidence_id": "VAR_PROFIT_2025_Q3"},
                [
                    {
                        "evidence_ref_type": ref["evidence_ref_type"],
                        "local_evidence_id": ref["local_evidence_id"],
                    }
                    for ref in variance["evidence_refs"]
                ],
            )
            self.assertIn(
                {"evidence_ref_type": "note_span", "local_evidence_id": "NOTE_AR_2025_Q3"},
                [
                    {
                        "evidence_ref_type": ref["evidence_ref_type"],
                        "local_evidence_id": ref["local_evidence_id"],
                    }
                    for ref in disclosure["evidence_refs"]
                ],
            )

            disclosure_candidates = [
                candidate
                for candidate in signal_candidates
                if candidate["risk_category"] == "disclosure_inconsistency_or_obfuscation"
            ]
            self.assertEqual(len(disclosure_candidates), 1)
            candidate = disclosure_candidates[0]
            self.assertEqual(candidate["candidate_id"], "CAND_VINACOUNT_SIGNAL_2025_Q3_DISC_VAR_001")
            self.assertEqual(candidate["candidate_status"], "pending_detector_review")
            self.assertEqual(candidate["priority"], "medium")
            self.assertEqual(candidate["review_mode"], "required")
            self.assertEqual(
                candidate["linked_tool_result_ids"],
                [disclosure["tool_result_id"], variance["tool_result_id"]],
            )
            self.assertEqual(
                candidate["supporting_signal_ids"],
                ["disclosure_narrative_tension", "variance_explanation_weak_connection"],
            )
            self.assertTrue(candidate["required_evidence_refs"])
            self.assertEqual(
                {query["query_type"] for query in candidate["required_context_queries"]},
                {"notes", "variance_explanations"},
            )

    def test_generated_tool_and_candidate_text_uses_risk_signal_language(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_fixture_spine(Path(temp_dir))

            generated_text = " ".join(
                [finding["finding_summary"] for finding in result.tool_findings]
                + [
                    limitation
                    for finding in result.tool_findings
                    for limitation in finding["limitations"]
                ]
                + [record["reason"] for record in result.tool_gating_records]
                + [candidate["reason_for_candidate"] for candidate in result.candidate_risks]
            ).lower()

            for prohibited in ["fraud", "manipulat", "conceal", "intent", "illegal", "legal misstatement"]:
                self.assertNotIn(prohibited, generated_text)

    def test_conditional_core_disclosure_capabilities_are_visible_when_skipped(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_fixture_spine(Path(temp_dir))

            skipped_tool_names = {
                record["tool_name"]
                for record in result.tool_gating_records
                if record["status"] == "skipped_missing_disclosure_context"
            }
            self.assertGreaterEqual(
                skipped_tool_names,
                {
                    "standard_corporate_related_party_exposure_tool",
                    "standard_corporate_accounting_policy_change_tool",
                },
            )
            self.assertTrue(
                all(
                    record["reason_code"]
                    for record in result.tool_gating_records
                    if record["tool_name"]
                    in {
                        "standard_corporate_related_party_exposure_tool",
                        "standard_corporate_accounting_policy_change_tool",
                    }
                )
            )
            self.assertFalse(
                {
                    "standard_corporate_related_party_exposure_tool",
                    "standard_corporate_accounting_policy_change_tool",
                }
                & {finding["tool_name"] for finding in result.tool_findings}
            )
            standard_candidate_categories = {
                candidate["risk_category"]
                for candidate in result.candidate_risks
                if candidate["report_id"].startswith("VINACOUNT_")
            }
            self.assertFalse({"related_party_disclosure_risk"} & standard_candidate_categories)

            events = [json.loads(line) for line in result.audit_log_path.read_text().splitlines()]
            gating_events = [
                event for event in events if event["stage"] == "tool_availability_gating_completed"
            ]
            self.assertEqual(len(gating_events), 5)
            logged_tool_names = {
                record["tool_name"]
                for event in gating_events
                for record in event["details"]["records"]
            }
            self.assertGreaterEqual(
                logged_tool_names,
                {
                    "standard_corporate_related_party_exposure_tool",
                    "standard_corporate_accounting_policy_change_tool",
                },
            )

    def test_clean_fixture_does_not_create_candidate_from_disabled_tool_context(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_fixture_spine(Path(temp_dir))

            clean_gating_records = [
                record
                for record in result.tool_gating_records
                if record["case_id"] == "clean_standard_corporate"
            ]
            clean_findings = [
                finding
                for finding in result.tool_findings
                if finding["report_id"] == "VINACOUNT_CLEAN_2025_Q3"
            ]
            clean_candidates = [
                candidate
                for candidate in result.candidate_risks
                if candidate["report_id"] == "VINACOUNT_CLEAN_2025_Q3"
            ]

            disabled_records = [
                record for record in clean_gating_records if record["status"] == "disabled_missing_context"
            ]
            self.assertEqual(
                [record["tool_name"] for record in disabled_records],
                ["standard_corporate_receivables_vs_revenue_growth_tool"],
            )
            self.assertNotIn(
                "standard_corporate_receivables_vs_revenue_growth_tool",
                {finding["tool_name"] for finding in clean_findings},
            )
            self.assertEqual(clean_candidates, [])

    def test_fixture_spine_runs_earnings_cashflow_mismatch_slice(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_fixture_spine(Path(temp_dir))

            signal_tool_findings = [
                finding
                for finding in result.tool_findings
                if finding["report_id"] == "VINACOUNT_SIGNAL_2025_Q3"
            ]
            signal_candidates = [
                candidate
                for candidate in result.candidate_risks
                if candidate["report_id"] == "VINACOUNT_SIGNAL_2025_Q3"
            ]

            gating_records = {
                record["tool_name"]: record
                for record in result.tool_gating_records
                if record["case_id"] == "signal_present_standard_corporate"
            }
            self.assertEqual(
                gating_records["standard_corporate_earnings_cashflow_mismatch_tool"]["status"],
                "enabled",
            )

            findings_by_tool = {finding["tool_name"]: finding for finding in signal_tool_findings}
            mismatch = findings_by_tool["standard_corporate_earnings_cashflow_mismatch_tool"]
            self.assertEqual(mismatch["status"], "completed")
            self.assertEqual(mismatch["period_basis"], "year_to_date")
            self.assertEqual(mismatch["analysis_scope"], "company_report_set")
            self.assertEqual(mismatch["threshold"]["basis"], "configured_default_v1")
            self.assertEqual(mismatch["threshold"]["config_version"], "standard_corporate_v1")
            self.assertEqual(mismatch["strength"], "strong")
            self.assertAlmostEqual(mismatch["metric"]["value"], -0.24, places=2)
            self.assertTrue(mismatch["calculation"]["inputs"])
            self.assertEqual(
                {input_value["period_basis"] for input_value in mismatch["calculation"]["inputs"]},
                {"year_to_date"},
            )
            self.assertNotIn(
                "CELL_CFO_2025_Q3",
                {input_value["local_evidence_id"] for input_value in mismatch["calculation"]["inputs"]},
            )
            self.assertTrue(all(input_value["report_id"] for input_value in mismatch["calculation"]["inputs"]))
            self.assertTrue(mismatch["limitations"])
            self.assertEqual(
                {ref["report_id"] for ref in mismatch["evidence_refs"]},
                {"VINACOUNT_SIGNAL_2025_Q3", "VINACOUNT_SIGNAL_2024_Q3"},
            )
            self.assertTrue(all(ref["local_evidence_id"] for ref in mismatch["evidence_refs"]))

            earnings_candidates = [
                candidate
                for candidate in signal_candidates
                if candidate["risk_category"] == "earnings_cashflow_mismatch"
            ]
            self.assertEqual(len(earnings_candidates), 1)
            candidate = earnings_candidates[0]
            self.assertEqual(candidate["candidate_status"], "pending_detector_review")
            self.assertEqual(candidate["priority"], "high")
            self.assertEqual(candidate["review_mode"], "required")
            self.assertEqual(candidate["linked_tool_result_ids"], [mismatch["tool_result_id"]])
            self.assertEqual(candidate["supporting_signal_ids"], ["positive_profit_negative_cfo"])
            self.assertTrue(candidate["required_evidence_refs"])

    def test_fixture_spine_runs_revenue_receivables_signal_slice(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_fixture_spine(Path(temp_dir))

            signal_tool_findings = [
                finding
                for finding in result.tool_findings
                if finding["report_id"] == "VINACOUNT_SIGNAL_2025_Q3"
            ]
            signal_candidates = [
                candidate
                for candidate in result.candidate_risks
                if candidate["report_id"] == "VINACOUNT_SIGNAL_2025_Q3"
            ]

            gating_records = {
                record["tool_name"]: record
                for record in result.tool_gating_records
                if record["case_id"] == "signal_present_standard_corporate"
            }
            self.assertEqual(
                gating_records["standard_corporate_revenue_growth_tool"]["status"],
                "enabled",
            )
            self.assertEqual(
                gating_records["standard_corporate_receivables_vs_revenue_growth_tool"]["status"],
                "enabled",
            )

            findings_by_tool = {finding["tool_name"]: finding for finding in signal_tool_findings}
            revenue_growth = findings_by_tool["standard_corporate_revenue_growth_tool"]
            receivables_gap = findings_by_tool["standard_corporate_receivables_vs_revenue_growth_tool"]

            self.assertEqual(revenue_growth["status"], "completed")
            self.assertEqual(receivables_gap["status"], "completed")
            self.assertEqual(revenue_growth["analysis_scope"], "company_report_set")
            self.assertEqual(receivables_gap["period_basis"], "quarter")
            self.assertEqual(revenue_growth["threshold"]["basis"], "configured_default_v1")
            self.assertEqual(receivables_gap["threshold"]["basis"], "configured_default_v1")
            self.assertEqual(revenue_growth["threshold"]["config_version"], "standard_corporate_v1")
            self.assertEqual(receivables_gap["threshold"]["config_version"], "standard_corporate_v1")
            self.assertAlmostEqual(revenue_growth["metric"]["value"], 65.18, places=2)
            self.assertAlmostEqual(receivables_gap["metric"]["value"], 51.49, places=2)
            self.assertTrue(revenue_growth["evidence_refs"])
            self.assertTrue(receivables_gap["calculation"]["inputs"])

            self.assertGreaterEqual(len(signal_candidates), 1)
            candidate = signal_candidates[0]
            self.assertEqual(candidate["risk_category"], "revenue_income_recognition_risk")
            self.assertEqual(candidate["candidate_status"], "pending_detector_review")
            self.assertEqual(candidate["priority"], "high")
            self.assertEqual(candidate["review_mode"], "required")
            self.assertEqual(
                candidate["linked_tool_result_ids"],
                [
                    revenue_growth["tool_result_id"],
                    receivables_gap["tool_result_id"],
                ],
            )
            self.assertTrue(candidate["required_evidence_refs"])
            self.assertEqual(
                [
                    candidate["candidate_id"]
                    for candidate in signal_candidates
                    if candidate["risk_category"] == "revenue_income_recognition_risk"
                ],
                ["CAND_VINACOUNT_SIGNAL_2025_Q3_REV_REC_001"],
            )

            user_facing_text = " ".join(
                [
                    revenue_growth["finding_summary"],
                    receivables_gap["finding_summary"],
                    candidate["reason_for_candidate"],
                ]
            ).lower()
            for prohibited in ["fraud", "manipulat", "conceal", "intent", "illegal", "legal misstatement"]:
                self.assertNotIn(prohibited, user_facing_text)

    def test_fixture_spine_validates_standard_corporate_pairs_and_logs_stages(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_fixture_spine(Path(temp_dir))

            self.assertEqual(result.status, "completed")
            self.assertGreaterEqual(
                result.loaded_report_sets,
                [
                    "signal_present_standard_corporate",
                    "clean_standard_corporate",
                ],
            )

            for report_set in [
                report_set
                for report_set in result.report_sets
                if report_set.target.metadata["report_profile"] == "standard_corporate"
            ]:
                self.assertEqual(report_set.target.metadata["report_profile"], "standard_corporate")
                self.assertEqual(report_set.prior_year.metadata["report_profile"], "standard_corporate")
                self.assertEqual(report_set.target.metadata["report_basis"], "consolidated")
                self.assertEqual(report_set.prior_year.metadata["report_basis"], "consolidated")
                self.assertTrue(report_set.target.metadata["business_context_tags"])
                self.assertTrue(report_set.prior_year.metadata["business_context_tags"])
                self.assertEqual(report_set.target.metadata["filing_status"], "original")
                self.assertEqual(report_set.prior_year.metadata["filing_status"], "original")
                self.assertTrue(report_set.target.metadata["canonical_source_document_id"])
                self.assertTrue(report_set.prior_year.metadata["canonical_source_document_id"])

                self.assertEqual(
                    report_set.target.metadata["company_name"],
                    report_set.prior_year.metadata["company_name"],
                )
                self.assertEqual(report_set.target.quarter, report_set.prior_year.quarter)
                self.assertEqual(report_set.target.year, report_set.prior_year.year + 1)

                for report_memory in [report_set.target, report_set.prior_year]:
                    self.assertEqual(report_memory.metadata["report_profile"], "standard_corporate")
                    self.assertFalse(_contains_prior_comparison_columns(report_memory.raw))

            events = [json.loads(line) for line in result.audit_log_path.read_text().splitlines()]
            stages = [event["stage"] for event in events]
            self.assertEqual(stages[0:2], ["run_started", "fixtures_loaded"])
            self.assertEqual(stages[-2:], ["wave1_exit_demo_artifacts_written", "run_completed"])
            self.assertEqual(stages.count("report_memory_validated"), 10)
            for stage in [
                "company_report_set_constructed",
                "real_extraction_quality_gate_completed",
                "tool_availability_gating_completed",
                "tool_execution_completed",
                "candidate_generation_completed",
                "detector_packet_build_completed",
                "detector_assessment_completed",
                "final_report_generation_completed",
            ]:
                self.assertEqual(stages.count(stage), 5)

    def test_fixture_spine_api_writes_append_only_audit_log(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            first = run_fixture_spine(output_dir)
            second = run_fixture_spine(output_dir)

            events = [json.loads(line) for line in second.audit_log_path.read_text().splitlines()]

            self.assertEqual(first.audit_log_path, second.audit_log_path)
            self.assertEqual(len(events), 108)
            self.assertEqual(events[0]["sequence"], 1)
            self.assertEqual(events[-1]["sequence"], 108)


def _contains_prior_comparison_columns(value):
    if isinstance(value, dict):
        return any(
            key in {"prior_period_value", "prior_year_value", "comparison_value"}
            or _contains_prior_comparison_columns(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_prior_comparison_columns(child) for child in value)
    return False


def _detector_visible_evidence_ids(packet):
    ids = {finding["tool_result_id"] for finding in packet["tool_findings"]}
    ids.update(
        f'{row["report_id"]}:{cell["cell_id"]}'
        for row in packet["relevant_table_rows"]
        for cell in row["values"].values()
    )
    ids.update(f'{note["report_id"]}:{note["note_id"]}' for note in packet["relevant_notes"])
    ids.update(
        f'{span["report_id"]}:{span["span_id"]}'
        for span in packet["relevant_variance_explanations"]
    )
    ids.update(rule["rule_id"] for rule in packet["rules"])
    return ids


if __name__ == "__main__":
    unittest.main()
