import json
import os
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from research.detector_contract_validation import validate_detector_packet


ALL_SUPPORT_TARGETS = {"supported", "weakly_supported", "not_supported", "insufficient_evidence"}
THESIS_SCENARIO_FAMILIES = (
    "revenue_receivables",
    "receivables_credit_quality",
    "inventory_cost_asset_flow",
    "expense_liability_understatement",
    "earnings_cashflow",
    "asset_quality_valuation",
    "related_party_disclosure",
    "disclosure_inconsistency",
)


class Wave4GroundedSyntheticPacketGeneratorPublicCommandTest(unittest.TestCase):
    def test_public_command_generates_earnings_cashflow_scenario_family(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clean_artifact = root / "clean_structured_report.json"
            output_dir = root / "generated"
            _write_json(clean_artifact, _clean_structured_report())

            completed = _run_generator(
                "--clean-report-artifact",
                str(clean_artifact),
                "--output-dir",
                str(output_dir),
                "--run-id",
                "grounded-earnings-cashflow",
                "--scenario-family",
                "earnings_cashflow",
            )

            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            records = _read_jsonl(output_dir / "staged_synthetic_packets.jsonl")
            self.assertEqual(len(records), 32)
            for record in records:
                self.assertEqual(record["source_type"], "synthetic_injected_raw")
                self.assertEqual(record["input"]["type"], "DetectorPacket")
                self.assertNotIn("output", record)
                validate_detector_packet(record["input"]["data"])

                packet = record["input"]["data"]
                self.assertEqual(packet["task"]["risk_category"], "earnings_cashflow_quality_risk")
                for finding in packet["tool_findings"]:
                    self.assertEqual(finding["risk_category"], "earnings_cashflow_quality_risk")

                generation = record["metadata"]["generation_metadata"]
                self.assertEqual(generation["injection_scenario_id"], "earnings_cashflow_divergence_v1")
                self.assertEqual(generation["target_risk_category"], "earnings_cashflow_quality_risk")
                self.assertIn(generation["target_support_level"], ALL_SUPPORT_TARGETS)
                if (
                    generation["target_support_level"] != "insufficient_evidence"
                    and generation["variant_slot_id"] != "V4_hard_missing_required_context"
                ):
                    self.assertIn("Profit increased", packet["candidate_summary"]["reason_for_candidate"])

            manifest = _read_json(output_dir / "manifest.json")
            self.assertEqual(manifest["records_written"], 32)
            self.assertEqual(manifest["coverage_counts"]["scenario_ids"], {"earnings_cashflow_divergence_v1": 32})
            self.assertEqual(manifest["coverage_counts"]["risk_categories"], {"earnings_cashflow_quality_risk": 32})

            metrics = _read_json(output_dir / "metrics.json")
            self.assertEqual(metrics["generated_records"], 32)
            self.assertEqual(metrics["scenario_ids"], {"earnings_cashflow_divergence_v1": 32})
            self.assertEqual(metrics["risk_categories"], {"earnings_cashflow_quality_risk": 32})

    def test_public_command_rejects_reserved_real_manual_source_anchor_before_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clean_artifact = root / "clean_structured_report.json"
            output_dir = root / "generated"
            real_release_dir = root / "real"
            _write_json(clean_artifact, _clean_structured_report())
            _write_real_manual_release(real_release_dir, company_key="FPT", period_key="2024_Q3")

            completed = _run_generator(
                "--clean-report-artifact",
                str(clean_artifact),
                "--output-dir",
                str(output_dir),
                "--run-id",
                "reserved-source-rejected",
                "--reserved-real-manual-release-dir",
                str(real_release_dir),
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("Remove reserved anchors from the synthetic source pool before generation", completed.stdout)
            self.assertFalse((output_dir / "staged_synthetic_packets.jsonl").exists())

    def test_earnings_cashflow_records_are_packet_only_without_hidden_metadata_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clean_artifact = root / "clean_structured_report.json"
            output_dir = root / "generated"
            _write_json(clean_artifact, _clean_structured_report(source_group_key="vietstock:FPT:2024_Q3"))

            completed = _run_generator(
                "--clean-report-artifact",
                str(clean_artifact),
                "--output-dir",
                str(output_dir),
                "--scenario-family",
                "earnings_cashflow",
            )

            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            for record in _read_jsonl(output_dir / "staged_synthetic_packets.jsonl"):
                self.assertEqual(record["source_type"], "synthetic_injected_raw")
                self.assertEqual(record["input"]["type"], "DetectorPacket")
                self.assertNotIn("output", record)
                validate_detector_packet(record["input"]["data"])
                detector_visible_text = json.dumps(record["input"]["data"], ensure_ascii=False, sort_keys=True).lower()
                for prohibited_text in [
                    "target_support_level",
                    "hidden_injection_details",
                    "source_file_sha256",
                    "normalized_text_hash",
                    "table_content_hash",
                    "derived_from_report_artifact_id",
                    "derived_from_source_document_id",
                    "raw_ocr",
                    "raw_pdf",
                    "cache",
                    "omitted_evidence",
                    "final_report",
                    "source_group",
                ]:
                    self.assertNotIn(prohibited_text, detector_visible_text)

    def test_earnings_cashflow_tool_finding_matches_visible_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clean_artifact = root / "clean_structured_report.json"
            output_dir = root / "generated"
            _write_json(clean_artifact, _clean_structured_report())

            completed = _run_generator(
                "--clean-report-artifact",
                str(clean_artifact),
                "--output-dir",
                str(output_dir),
                "--scenario-family",
                "earnings_cashflow",
            )

            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            records_by_target = _records_by_target_first_slot(output_dir / "staged_synthetic_packets.jsonl")
            supported_packet = records_by_target["supported"]["input"]["data"]
            weak_packet = records_by_target["weakly_supported"]["input"]["data"]

            supported_growths = _packet_growths(supported_packet)
            weak_growths = _packet_growths(weak_packet)
            self.assertAlmostEqual(supported_growths["net_profit"], 0.40)
            self.assertAlmostEqual(supported_growths["operating_cash_flow"], -0.540625)
            self.assertAlmostEqual(weak_growths["net_profit"], 0.40)
            self.assertAlmostEqual(weak_growths["operating_cash_flow"], 0.025)
            self.assertLess(
                supported_growths["operating_cash_flow"],
                weak_growths["operating_cash_flow"],
            )

            self.assertIn("Profit increased 40.0%", weak_packet["candidate_summary"]["reason_for_candidate"])
            self.assertIn("operating cash flow changed 2.5%", weak_packet["candidate_summary"]["reason_for_candidate"])
            self.assertEqual(weak_packet["candidate_summary"]["supporting_signal_ids"], ["profit_growth_outpaces_operating_cash_flow"])

            finding = weak_packet["tool_findings"][0]
            self.assertEqual(weak_packet["task"]["risk_category"], "earnings_cashflow_quality_risk")
            self.assertEqual(finding["tool_name"], "earnings_vs_operating_cash_flow_tool")
            self.assertEqual(finding["risk_category"], "earnings_cashflow_quality_risk")
            self.assertEqual(finding["signal_id"], "profit_growth_outpaces_operating_cash_flow")
            self.assertIs(finding["flag"], True)
            self.assertEqual(finding["metric"], "operating_cash_flow_to_profit_ratio")
            self.assertEqual(finding["metric_value"], 0.78)
            self.assertEqual(
                finding["threshold"],
                "flag if operating cash flow is below 80 percent of profit while profit grows",
            )
            self.assertIn("78.1% of current profit", finding["finding_summary"])
            self.assertEqual(
                [ref["ref_id"] for ref in finding["evidence_refs"]],
                [
                    "FPT_2024_Q3_SYN_EARN_CASH_002_V01:ROW_NET_PROFIT",
                    "FPT_2024_Q3_SYN_EARN_CASH_002_V01:ROW_OPERATING_CASH_FLOW",
                ],
            )
            self.assertEqual(weak_packet["rules"][0]["risk_category"], "earnings_cashflow_quality_risk")
            self.assertEqual(weak_packet["rules"][0]["related_signal_ids"], ["profit_growth_outpaces_operating_cash_flow"])

    def test_earnings_cashflow_records_include_required_non_detector_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clean_artifact = root / "clean_structured_report.json"
            output_dir = root / "generated"
            _write_json(clean_artifact, _clean_structured_report(source_group_key="vietstock:FPT:2024_Q3"))

            completed = _run_generator(
                "--clean-report-artifact",
                str(clean_artifact),
                "--output-dir",
                str(output_dir),
                "--scenario-family",
                "earnings_cashflow",
                "--run-id",
                "grounded-earnings-metadata",
            )

            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            records_by_target = _records_by_target_first_slot(output_dir / "staged_synthetic_packets.jsonl")
            self.assertEqual(set(records_by_target), ALL_SUPPORT_TARGETS)

            for target_support_level, record in records_by_target.items():
                metadata = record["metadata"]
                generation = metadata["generation_metadata"]
                split = metadata["split_metadata"]
                self.assertEqual(generation["base_report_id"], "FPT_2024_Q3_CLEAN")
                self.assertEqual(generation["injection_scenario_id"], "earnings_cashflow_divergence_v1")
                self.assertEqual(generation["target_risk_category"], "earnings_cashflow_quality_risk")
                self.assertEqual(generation["target_support_level"], target_support_level)
                self.assertEqual(generation["variant_slot_id"], "V1_easy_quantitative_clear")
                self.assertEqual(generation["variant_pattern_id"], "V1_easy_quantitative_clear_standard_corporate_v1")
                self.assertEqual(metadata["report_profile"], "standard_corporate")
                self.assertEqual(split["company_key"], "FPT")
                self.assertEqual(split["period_key"], "2024_Q3")
                self.assertEqual(split["derived_from_group_key"], "FPT_2024_Q3_CLEAN")
                self.assertEqual(split["source_group_key"], "vietstock:FPT:2024_Q3")
                self.assertEqual(split["derived_from_report_artifact_id"], "RPT_FPT_2024_Q3_CLEAN")
                self.assertEqual(split["derived_from_source_document_id"], "DOC_FPT_2024_Q3_FS")
                self.assertTrue(split["source_file_sha256"])
                self.assertTrue(split["normalized_text_hash"])
                self.assertTrue(split["table_content_hash"])

            self.assertEqual(
                records_by_target["supported"]["metadata"]["generation_metadata"]["synthetic_report_id"],
                "FPT_2024_Q3_SYN_EARN_CASH_001_V01",
            )
            self.assertEqual(
                records_by_target["weakly_supported"]["metadata"]["generation_metadata"]["synthetic_report_id"],
                "FPT_2024_Q3_SYN_EARN_CASH_002_V01",
            )
            self.assertEqual(
                records_by_target["not_supported"]["metadata"]["generation_metadata"]["synthetic_report_id"],
                "FPT_2024_Q3_SYN_EARN_CASH_003_V01",
            )
            self.assertEqual(
                records_by_target["insufficient_evidence"]["metadata"]["generation_metadata"]["synthetic_report_id"],
                "FPT_2024_Q3_SYN_EARN_CASH_004_V01",
            )

    def test_public_command_can_generate_revenue_and_earnings_scenarios_in_one_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clean_artifact = root / "clean_structured_report.json"
            output_dir = root / "generated"
            _write_json(clean_artifact, _clean_structured_report(source_group_key="vietstock:FPT:2024_Q3"))

            completed = _run_generator(
                "--clean-report-artifact",
                str(clean_artifact),
                "--output-dir",
                str(output_dir),
                "--scenario-family",
                "revenue_receivables",
                "--scenario-family",
                "earnings_cashflow",
                "--run-id",
                "grounded-combined-scenarios",
            )

            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            records = _read_jsonl(output_dir / "staged_synthetic_packets.jsonl")
            self.assertEqual(len(records), 64)
            self.assertEqual(
                {
                    record["metadata"]["generation_metadata"]["injection_scenario_id"]
                    for record in records
                },
                {"revenue_receivables_divergence_v1", "earnings_cashflow_divergence_v1"},
            )
            for record in records:
                self.assertNotIn("output", record)
                validate_detector_packet(record["input"]["data"])
                detector_visible_text = json.dumps(record["input"]["data"], ensure_ascii=False, sort_keys=True).lower()
                self.assertNotIn("target_support_level", detector_visible_text)
                self.assertNotIn("hidden_injection_details", detector_visible_text)
                self.assertNotIn("source_file_sha256", detector_visible_text)

            manifest = _read_json(output_dir / "manifest.json")
            self.assertEqual(manifest["records_written"], 64)
            self.assertEqual(
                manifest["scenarios"],
                [
                    {"scenario_id": "earnings_cashflow_divergence_v1", "version": "1.0.0"},
                    {"scenario_id": "revenue_receivables_divergence_v1", "version": "1.0.0"},
                ],
            )
            self.assertEqual(
                manifest["risk_categories"],
                ["earnings_cashflow_quality_risk", "revenue_income_recognition_risk"],
            )
            self.assertEqual(manifest["report_profiles"], ["standard_corporate"])
            self.assertEqual(manifest["base_groups"], ["FPT_2024_Q3_CLEAN"])
            self.assertEqual(
                manifest["coverage_counts"]["scenario_ids"],
                {"earnings_cashflow_divergence_v1": 32, "revenue_receivables_divergence_v1": 32},
            )
            self.assertEqual(
                manifest["coverage_counts"]["risk_categories"],
                {"earnings_cashflow_quality_risk": 32, "revenue_income_recognition_risk": 32},
            )

            metrics = _read_json(output_dir / "metrics.json")
            self.assertEqual(metrics["generated_records"], 64)
            self.assertEqual(
                metrics["scenarios"],
                [
                    {"scenario_id": "earnings_cashflow_divergence_v1", "version": "1.0.0"},
                    {"scenario_id": "revenue_receivables_divergence_v1", "version": "1.0.0"},
                ],
            )
            self.assertEqual(
                metrics["scenario_ids"],
                {"earnings_cashflow_divergence_v1": 32, "revenue_receivables_divergence_v1": 32},
            )
            self.assertEqual(metrics["target_support_levels"], _support_counts(16))

    def test_public_command_generates_all_eight_thesis_families_from_one_rich_clean_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clean_artifact = root / "clean_structured_report.json"
            output_dir = root / "generated"
            _write_json(clean_artifact, _clean_structured_report())

            family_args = []
            for family in THESIS_SCENARIO_FAMILIES:
                family_args.extend(["--scenario-family", family])
            completed = _run_generator(
                "--clean-report-artifact",
                str(clean_artifact),
                "--output-dir",
                str(output_dir),
                "--run-id",
                "grounded-all-thesis-families",
                *family_args,
            )

            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            records = _read_jsonl(output_dir / "staged_synthetic_packets.jsonl")
            self.assertEqual(len(records), 256)
            for record in records:
                validate_detector_packet(record["input"]["data"])
                self.assertIn(
                    record["metadata"]["generation_metadata"]["target_support_level"],
                    ALL_SUPPORT_TARGETS,
                )
            metrics = _read_json(output_dir / "metrics.json")
            self.assertEqual(metrics["generated_records"], 256)
            self.assertEqual(metrics["target_support_levels"], _support_counts(64))
            self.assertEqual(set(metrics["scenario_ids"].values()), {32})
            self.assertEqual(len(metrics["scenario_ids"]), 8)

    def test_public_command_generates_eight_variant_slots_per_canonical_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clean_artifact = root / "clean_structured_report.json"
            output_dir = root / "generated"
            _write_json(clean_artifact, _clean_structured_report())

            family_args = []
            for family in THESIS_SCENARIO_FAMILIES:
                family_args.extend(["--scenario-family", family])
            completed = _run_generator(
                "--clean-report-artifact",
                str(clean_artifact),
                "--output-dir",
                str(output_dir),
                "--run-id",
                "grounded-all-variant-slots",
                *family_args,
            )

            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            records = _read_jsonl(output_dir / "staged_synthetic_packets.jsonl")
            self.assertEqual(len(records), 256)

            slots_by_target: dict[tuple[str, str], set[str]] = {}
            patterns_by_target: dict[tuple[str, str], set[str]] = {}
            for record in records:
                packet = record["input"]["data"]
                validate_detector_packet(packet)
                generation = record["metadata"]["generation_metadata"]
                target = (
                    generation["target_risk_category"],
                    generation["target_support_level"],
                )
                slots_by_target.setdefault(target, set()).add(generation["variant_slot_id"])
                patterns_by_target.setdefault(target, set()).add(generation["variant_pattern_id"])

                detector_visible_text = json.dumps(packet, ensure_ascii=False, sort_keys=True).lower()
                self.assertNotIn("variant_slot", detector_visible_text)
                self.assertNotIn("variant_pattern", detector_visible_text)
                self.assertNotIn("canonical_target", detector_visible_text)

            self.assertEqual(len(slots_by_target), 32)
            for slots in slots_by_target.values():
                self.assertEqual(slots, _variant_slots())
            for patterns in patterns_by_target.values():
                self.assertEqual(len(patterns), 8)

            metrics = _read_json(output_dir / "metrics.json")
            self.assertEqual(metrics["generated_records"], 256)
            self.assertEqual(metrics["variant_slots"], {slot: 32 for slot in sorted(_variant_slots())})
            self.assertEqual(metrics["target_support_levels"], _support_counts(64))
            self.assertEqual(set(metrics["scenario_ids"].values()), {32})

    def test_variant_slots_create_distinct_detector_visible_evidence_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clean_artifact = root / "clean_structured_report.json"
            output_dir = root / "generated"
            _write_json(clean_artifact, _clean_structured_report())

            completed = _run_generator(
                "--clean-report-artifact",
                str(clean_artifact),
                "--output-dir",
                str(output_dir),
                "--scenario-family",
                "revenue_receivables",
                "--run-id",
                "grounded-visible-variant-profiles",
            )

            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            supported_records = [
                record
                for record in _read_jsonl(output_dir / "staged_synthetic_packets.jsonl")
                if record["metadata"]["generation_metadata"]["target_support_level"] == "supported"
            ]
            profiles = {
                record["metadata"]["generation_metadata"]["variant_slot_id"]: _packet_profile(record["input"]["data"])
                for record in supported_records
            }

            self.assertEqual(set(profiles), _variant_slots())
            self.assertEqual(len({json.dumps(profile, sort_keys=True) for profile in profiles.values()}), 8)
            self.assertTrue(profiles["V2_easy_quantitative_contradiction"]["has_variance_explanation"])
            self.assertFalse(profiles["V4_hard_missing_required_context"]["has_tool_findings"])
            self.assertGreater(profiles["V5_note_or_disclosure_quality"]["note_text_length"], profiles["V1_easy_quantitative_clear"]["note_text_length"])
            self.assertFalse(profiles["V6_tool_finding_contradiction"]["tool_flag"])
            self.assertTrue(
                any(row_id.endswith("_ALT") for row_id in profiles["V7_profile_specific_mapping"]["row_ids"])
            )
            self.assertIn("Doanh thu", profiles["V8_language_or_account_label_style"]["candidate_reason"])

    def test_public_command_generates_all_support_target_variants_with_visible_packet_differences(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clean_artifact = root / "clean_structured_report.json"
            output_dir = root / "generated"
            _write_json(clean_artifact, _clean_structured_report())

            completed = _run_generator(
                "--clean-report-artifact",
                str(clean_artifact),
                "--output-dir",
                str(output_dir),
                "--scenario-family",
                "revenue_receivables",
                "--scenario-family",
                "earnings_cashflow",
                "--run-id",
                "grounded-all-support-targets",
            )

            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            records = _read_jsonl(output_dir / "staged_synthetic_packets.jsonl")
            self.assertEqual(len(records), 64)

            by_family_and_target = {
                (
                    record["metadata"]["generation_metadata"]["injection_scenario_id"],
                    record["metadata"]["generation_metadata"]["target_support_level"],
                ): record
                for record in records
            }
            self.assertEqual(
                set(by_family_and_target),
                {
                    ("revenue_receivables_divergence_v1", "supported"),
                    ("revenue_receivables_divergence_v1", "weakly_supported"),
                    ("revenue_receivables_divergence_v1", "not_supported"),
                    ("revenue_receivables_divergence_v1", "insufficient_evidence"),
                    ("earnings_cashflow_divergence_v1", "supported"),
                    ("earnings_cashflow_divergence_v1", "weakly_supported"),
                    ("earnings_cashflow_divergence_v1", "not_supported"),
                    ("earnings_cashflow_divergence_v1", "insufficient_evidence"),
                },
            )

            revenue_not_supported = by_family_and_target[
                ("revenue_receivables_divergence_v1", "not_supported")
            ]["input"]["data"]
            self.assertIs(revenue_not_supported["tool_findings"][0]["flag"], False)
            self.assertLessEqual(revenue_not_supported["tool_findings"][0]["metric_value"], 10.0)
            self.assertIn("did not exceed", revenue_not_supported["tool_findings"][0]["finding_summary"])

            earnings_insufficient = by_family_and_target[
                ("earnings_cashflow_divergence_v1", "insufficient_evidence")
            ]
            earnings_packet = earnings_insufficient["input"]["data"]
            self.assertEqual(earnings_packet["tool_findings"], [])
            self.assertEqual(earnings_packet["relevant_notes"], [])
            self.assertEqual(earnings_packet["candidate_summary"]["supporting_signal_ids"], [])
            self.assertTrue(earnings_insufficient["metadata"]["evidence_profile"]["has_missing_required_evidence"])
            self.assertEqual(earnings_insufficient["metadata"]["evidence_profile"]["evidence_types"], ["table_row"])

            detector_visible_text = json.dumps(earnings_packet, ensure_ascii=False, sort_keys=True).lower()
            self.assertNotIn("target_support_level", detector_visible_text)
            self.assertNotIn("hidden_injection_details", detector_visible_text)

    def test_earnings_cashflow_passes_raw_staging_and_gate_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clean_artifact = root / "clean_structured_report.json"
            second_artifact = root / "second_clean_structured_report.json"
            generated_dir = root / "generated"
            raw_dir = root / "raw"
            gate_root = root / "gate"
            _write_json(clean_artifact, _clean_structured_report(source_group_key="vietstock:FPT:2024_Q3"))
            _write_json(second_artifact, _clean_structured_report("VNM", "2024_Q4", source_group_key="vietstock:VNM:2024_Q4"))

            generated = _run_generator(
                "--clean-report-artifact",
                str(clean_artifact),
                "--clean-report-artifact",
                str(second_artifact),
                "--output-dir",
                str(generated_dir),
                "--scenario-family",
                "earnings_cashflow",
                "--run-id",
                "grounded-earnings-chain",
            )
            self.assertEqual(generated.returncode, 0, generated.stdout + generated.stderr)
            staged_path = generated_dir / "staged_synthetic_packets.jsonl"
            staged_records = _read_jsonl(staged_path)
            self.assertEqual(len(staged_records), 64)
            for record in staged_records:
                validate_detector_packet(record["input"]["data"])

            staged = _run_module(
                "research.synthetic_raw_stager",
                "--input-jsonl",
                str(staged_path),
                "--output-dir",
                str(raw_dir),
                "--run-id",
                "raw-earnings-chain",
            )
            self.assertEqual(staged.returncode, 0, staged.stdout + staged.stderr)
            raw_path = raw_dir / "synthetic_injected_raw.jsonl"
            self.assertEqual(len(_read_jsonl(raw_path)), 64)
            raw_metrics = _read_json(raw_dir / "metrics.json")
            self.assertEqual(raw_metrics["records_written"], 64)
            self.assertEqual(raw_metrics["risk_categories"], {"earnings_cashflow_quality_risk": 64})
            self.assertEqual(raw_metrics["target_support_levels"], _support_counts(16))
            self.assertEqual(raw_metrics["report_profiles"], {"standard_corporate": 64})
            self.assertEqual(raw_metrics["base_groups"], {"FPT_2024_Q3_CLEAN": 32, "VNM_2024_Q4_CLEAN": 32})
            self.assertEqual(raw_metrics["synthetic_groups"], _synthetic_group_counts("SYN_EARN_CASH", ["FPT_2024_Q3", "VNM_2024_Q4"]))
            self.assertEqual(raw_metrics["source_groups"], {"vietstock:FPT:2024_Q3": 32, "vietstock:VNM:2024_Q4": 32})
            self.assertEqual(raw_metrics["scenario_ids"], {"earnings_cashflow_divergence_v1": 64})

            gate = _run_module(
                "research.synthetic_detector_assessment_gate",
                "--input-jsonl",
                str(raw_path),
                "--output-root",
                str(gate_root),
                "--mode",
                "dry_run",
                "--run-id",
                "gate-earnings-chain",
            )
            self.assertEqual(gate.returncode, 0, gate.stdout + gate.stderr)
            gate_metrics = _read_json(gate_root / "gate-earnings-chain" / "metrics.json")
            self.assertEqual(gate_metrics["input_records"], 64)
            self.assertEqual(gate_metrics["input_target_support_levels"], _support_counts(16))
            self.assertEqual(gate_metrics["input_risk_categories"], {"earnings_cashflow_quality_risk": 64})
            self.assertEqual(gate_metrics["input_report_profiles"], {"standard_corporate": 64})
            self.assertEqual(gate_metrics["input_base_groups"], {"FPT_2024_Q3_CLEAN": 32, "VNM_2024_Q4_CLEAN": 32})
            self.assertEqual(gate_metrics["input_synthetic_groups"], _synthetic_group_counts("SYN_EARN_CASH", ["FPT_2024_Q3", "VNM_2024_Q4"]))
            self.assertEqual(
                gate_metrics["input_source_groups"],
                {"vietstock:FPT:2024_Q3": 32, "vietstock:VNM:2024_Q4": 32},
            )
            self.assertEqual(gate_metrics["input_scenario_ids"], {"earnings_cashflow_divergence_v1": 64})

    def test_public_command_accepts_two_clean_report_artifacts_in_one_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fpt_artifact = root / "fpt_clean_structured_report.json"
            vnm_artifact = root / "vnm_clean_structured_report.json"
            output_dir = root / "generated"
            _write_json(fpt_artifact, _clean_structured_report())
            _write_json(vnm_artifact, _clean_structured_report("VNM", "2024_Q4"))

            completed = _run_generator(
                "--clean-report-artifact",
                str(fpt_artifact),
                "--clean-report-artifact",
                str(vnm_artifact),
                "--output-dir",
                str(output_dir),
                "--run-id",
                "grounded-batch",
            )

            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            staged_path = output_dir / "staged_synthetic_packets.jsonl"
            self.assertTrue(staged_path.exists())
            records = _read_jsonl(staged_path)
            self.assertEqual(len(records), 64)
            self.assertEqual(
                {
                    record["metadata"]["split_metadata"]["derived_from_group_key"]
                    for record in records
                },
                {"FPT_2024_Q3_CLEAN", "VNM_2024_Q4_CLEAN"},
            )
            self.assertEqual(
                {
                    (
                        record["metadata"]["split_metadata"]["derived_from_group_key"],
                        record["metadata"]["generation_metadata"]["target_support_level"],
                    )
                    for record in records
                },
                {
                    ("FPT_2024_Q3_CLEAN", "supported"),
                    ("FPT_2024_Q3_CLEAN", "weakly_supported"),
                    ("FPT_2024_Q3_CLEAN", "not_supported"),
                    ("FPT_2024_Q3_CLEAN", "insufficient_evidence"),
                    ("VNM_2024_Q4_CLEAN", "supported"),
                    ("VNM_2024_Q4_CLEAN", "weakly_supported"),
                    ("VNM_2024_Q4_CLEAN", "not_supported"),
                    ("VNM_2024_Q4_CLEAN", "insufficient_evidence"),
                },
            )

    def test_public_command_preserves_packet_only_batch_lineage_and_coverage_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fpt_artifact = root / "fpt_clean_structured_report.json"
            vnm_artifact = root / "vnm_clean_structured_report.json"
            output_dir = root / "generated"
            _write_json(fpt_artifact, _clean_structured_report(source_group_key="vietstock:FPT:2024_Q3"))
            _write_json(
                vnm_artifact,
                _clean_structured_report(
                    "VNM",
                    "2024_Q4",
                    report_profile="credit_institution",
                    source_group_key="vietstock:VNM:2024_Q4",
                ),
            )

            completed = _run_generator(
                "--clean-report-artifact",
                str(fpt_artifact),
                "--clean-report-artifact",
                str(vnm_artifact),
                "--output-dir",
                str(output_dir),
                "--run-id",
                "grounded-batch-coverage",
            )

            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            records = _read_jsonl(output_dir / "staged_synthetic_packets.jsonl")
            self.assertEqual(len(records), 64)
            for record in records:
                self.assertEqual(record["source_type"], "synthetic_injected_raw")
                self.assertEqual(record["input"]["type"], "DetectorPacket")
                self.assertNotIn("output", record)
                validate_detector_packet(record["input"]["data"])
                detector_visible_text = json.dumps(record["input"]["data"], ensure_ascii=False, sort_keys=True).lower()
                for prohibited_text in [
                    "target_support_level",
                    "hidden_injection_details",
                    "source_file_sha256",
                    "normalized_text_hash",
                    "table_content_hash",
                    "raw_ocr",
                    "raw_pdf",
                    "cache",
                    "omitted_evidence",
                    "final_report",
                ]:
                    self.assertNotIn(prohibited_text, detector_visible_text)

            metadata_by_group = {
                (
                    record["metadata"]["split_metadata"]["derived_from_group_key"],
                    record["metadata"]["generation_metadata"]["target_support_level"],
                ): record["metadata"]
                for record in records
            }
            self.assertEqual(metadata_by_group[("FPT_2024_Q3_CLEAN", "supported")]["report_profile"], "standard_corporate")
            self.assertEqual(metadata_by_group[("FPT_2024_Q3_CLEAN", "supported")]["split_metadata"]["company_key"], "FPT")
            self.assertEqual(metadata_by_group[("FPT_2024_Q3_CLEAN", "supported")]["split_metadata"]["period_key"], "2024_Q3")
            self.assertEqual(
                metadata_by_group[("FPT_2024_Q3_CLEAN", "supported")]["split_metadata"]["source_group_key"],
                "vietstock:FPT:2024_Q3",
            )
            self.assertEqual(metadata_by_group[("VNM_2024_Q4_CLEAN", "weakly_supported")]["report_profile"], "credit_institution")
            self.assertEqual(metadata_by_group[("VNM_2024_Q4_CLEAN", "weakly_supported")]["split_metadata"]["company_key"], "VNM")
            self.assertEqual(metadata_by_group[("VNM_2024_Q4_CLEAN", "weakly_supported")]["split_metadata"]["period_key"], "2024_Q4")
            self.assertEqual(
                metadata_by_group[("VNM_2024_Q4_CLEAN", "weakly_supported")]["split_metadata"]["source_group_key"],
                "vietstock:VNM:2024_Q4",
            )

            manifest = _read_json(output_dir / "manifest.json")
            self.assertEqual(manifest["records_written"], 64)
            self.assertEqual(manifest["coverage_counts"]["risk_categories"], {"revenue_income_recognition_risk": 64})
            self.assertEqual(manifest["coverage_counts"]["target_support_levels"], _support_counts(16))
            self.assertEqual(manifest["coverage_counts"]["report_profiles"], {"credit_institution": 32, "standard_corporate": 32})
            self.assertEqual(
                manifest["coverage_counts"]["source_groups"],
                {"vietstock:FPT:2024_Q3": 32, "vietstock:VNM:2024_Q4": 32},
            )
            self.assertEqual(manifest["coverage_counts"]["scenario_ids"], {"revenue_receivables_divergence_v1": 64})
            self.assertEqual(manifest["coverage_counts"]["rejected_invalid_reasons"], {})

            metrics = _read_json(output_dir / "metrics.json")
            self.assertEqual(metrics["generated_records"], 64)
            self.assertEqual(metrics["risk_categories"], {"revenue_income_recognition_risk": 64})
            self.assertEqual(metrics["target_support_levels"], _support_counts(16))
            self.assertEqual(metrics["report_profiles"], {"credit_institution": 32, "standard_corporate": 32})
            self.assertEqual(metrics["base_groups"], {"FPT_2024_Q3_CLEAN": 32, "VNM_2024_Q4_CLEAN": 32})
            self.assertEqual(metrics["synthetic_groups"], _synthetic_group_counts("SYN_REV_REC", ["FPT_2024_Q3", "VNM_2024_Q4"]))
            self.assertEqual(metrics["source_groups"], {"vietstock:FPT:2024_Q3": 32, "vietstock:VNM:2024_Q4": 32})
            self.assertEqual(metrics["scenario_ids"], {"revenue_receivables_divergence_v1": 64})
            self.assertEqual(metrics["rejected_invalid_reasons"], {})

    def test_public_command_generates_insurance_packets_from_profile_native_accounts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clean_artifact = root / "insurance_clean_structured_report.json"
            output_dir = root / "generated"
            _write_json(
                clean_artifact,
                _clean_structured_report("BIC", "2026_Q1", report_profile="insurance"),
            )

            completed = _run_generator(
                "--clean-report-artifact",
                str(clean_artifact),
                "--output-dir",
                str(output_dir),
                "--scenario-family",
                "revenue_receivables",
                "--run-id",
                "grounded-insurance-native-accounts",
            )

            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            records = _read_jsonl(output_dir / "staged_synthetic_packets.jsonl")
            self.assertEqual(len(records), 32)
            packet = records[0]["input"]["data"]
            rows_by_account = {
                row["standard_account"]: row
                for row in packet["relevant_table_rows"]
            }
            self.assertEqual(set(rows_by_account), {"gross_written_premium", "premium_receivables"})
            self.assertEqual(packet["metadata"]["report_profile"], "insurance")
            self.assertNotIn("trade_receivables", json.dumps(packet))
            validate_detector_packet(packet)

    def test_public_command_generates_securities_packets_from_profile_native_accounts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clean_artifact = root / "securities_clean_structured_report.json"
            output_dir = root / "generated"
            _write_json(
                clean_artifact,
                _clean_structured_report("MBS", "2026_Q1", report_profile="securities"),
            )

            completed = _run_generator(
                "--clean-report-artifact",
                str(clean_artifact),
                "--output-dir",
                str(output_dir),
                "--scenario-family",
                "revenue_receivables",
                "--run-id",
                "grounded-securities-native-accounts",
            )

            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            records = _read_jsonl(output_dir / "staged_synthetic_packets.jsonl")
            self.assertEqual(len(records), 32)
            packet = records[0]["input"]["data"]
            rows_by_account = {
                row["standard_account"]: row
                for row in packet["relevant_table_rows"]
            }
            self.assertEqual(set(rows_by_account), {"profit_after_tax", "margin_lending"})
            self.assertEqual(packet["metadata"]["report_profile"], "securities")
            self.assertNotIn("trade_receivables", json.dumps(packet))
            validate_detector_packet(packet)

    def test_public_command_generates_credit_institution_packets_from_profile_native_accounts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clean_artifact = root / "credit_clean_structured_report.json"
            output_dir = root / "generated"
            _write_json(
                clean_artifact,
                _clean_structured_report("ACB", "2026_Q1", report_profile="credit_institution"),
            )

            completed = _run_generator(
                "--clean-report-artifact",
                str(clean_artifact),
                "--output-dir",
                str(output_dir),
                "--scenario-family",
                "revenue_receivables",
                "--run-id",
                "grounded-credit-native-accounts",
            )

            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            records = _read_jsonl(output_dir / "staged_synthetic_packets.jsonl")
            self.assertEqual(len(records), 32)
            packet = records[0]["input"]["data"]
            rows_by_account = {
                row["standard_account"]: row
                for row in packet["relevant_table_rows"]
            }
            self.assertEqual(set(rows_by_account), {"profit_after_tax", "loans_to_customers"})
            self.assertEqual(packet["metadata"]["report_profile"], "credit_institution")
            self.assertNotIn("trade_receivables", json.dumps(packet))
            validate_detector_packet(packet)

    def test_public_command_rejects_mixed_validity_batch_before_writing_any_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            valid_artifact = root / "valid_clean_structured_report.json"
            invalid_artifact = root / "invalid_clean_structured_report.json"
            output_dir = root / "generated"
            invalid = _clean_structured_report("VNM", "2024_Q4")
            invalid["structured_evidence"]["rows"] = [
                row for row in invalid["structured_evidence"]["rows"] if row["standard_account"] != "trade_receivables"
            ]
            _write_json(valid_artifact, _clean_structured_report())
            _write_json(invalid_artifact, invalid)

            completed = _run_generator(
                "--clean-report-artifact",
                str(valid_artifact),
                "--clean-report-artifact",
                str(invalid_artifact),
                "--output-dir",
                str(output_dir),
                "--run-id",
                "grounded-mixed-invalid",
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("invalid_clean_structured_report.json", completed.stdout)
            self.assertIn("missing required structured evidence account: trade_receivables", completed.stdout)
            self.assertFalse((output_dir / "staged_synthetic_packets.jsonl").exists())
            self.assertFalse((output_dir / "manifest.json").exists())
            self.assertFalse((output_dir / "metrics.json").exists())

    def test_public_command_generates_grounded_staged_candidates_for_all_support_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clean_artifact = root / "clean_structured_report.json"
            output_dir = root / "generated"
            _write_json(clean_artifact, _clean_structured_report())

            completed = _run_generator(
                "--clean-report-artifact",
                str(clean_artifact),
                "--output-dir",
                str(output_dir),
                "--run-id",
                "grounded-smoke",
            )

            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            staged_path = output_dir / "staged_synthetic_packets.jsonl"
            manifest_path = output_dir / "manifest.json"
            metrics_path = output_dir / "metrics.json"
            self.assertTrue(staged_path.exists())
            self.assertTrue(manifest_path.exists())
            self.assertTrue(metrics_path.exists())

            records = _read_jsonl(staged_path)
            self.assertEqual(len(records), 32)
            records_by_target = {
                record["metadata"]["generation_metadata"]["target_support_level"]: record
                for record in records
                if record["metadata"]["generation_metadata"]["variant_slot_id"] == "V1_easy_quantitative_clear"
            }
            self.assertEqual(set(records_by_target), ALL_SUPPORT_TARGETS)

            for target_support_level, record in records_by_target.items():
                self.assertEqual(record["source_type"], "synthetic_injected_raw")
                self.assertEqual(record["input"]["type"], "DetectorPacket")
                self.assertNotIn("output", record)

                packet = record["input"]["data"]
                self.assertEqual(packet["task"]["risk_category"], "revenue_income_recognition_risk")
                for finding in packet["tool_findings"]:
                    self.assertEqual(finding["risk_category"], "revenue_income_recognition_risk")
                self.assertNotIn("target_support_level", json.dumps(packet, sort_keys=True))
                self.assertNotIn("source_file_sha256", json.dumps(packet, sort_keys=True))

                generation = record["metadata"]["generation_metadata"]
                self.assertEqual(generation["generation_method"], "grounded_structured_evidence_injection")
                self.assertEqual(generation["base_report_id"], "FPT_2024_Q3_CLEAN")
                self.assertEqual(generation["injection_scenario_id"], "revenue_receivables_divergence_v1")
                self.assertEqual(generation["target_support_level"], target_support_level)
                self.assertEqual(generation["variant_slot_id"], "V1_easy_quantitative_clear")

                split = record["metadata"]["split_metadata"]
                self.assertEqual(split["derived_from_group_key"], "FPT_2024_Q3_CLEAN")
                self.assertEqual(split["source_file_sha256"], "sha256-source-FPT_2024_Q3")
                self.assertEqual(split["normalized_text_hash"], "sha256-normalized-FPT_2024_Q3")
                self.assertEqual(split["table_content_hash"], "sha256-table-FPT_2024_Q3")
                for field in [
                    "source_file_sha256",
                    "normalized_text_hash",
                    "table_content_hash",
                    "derived_from_report_artifact_id",
                    "derived_from_source_document_id",
                ]:
                    self.assertTrue(split[field])

            supported_generation = records_by_target["supported"]["metadata"]["generation_metadata"]
            self.assertEqual(supported_generation["synthetic_report_id"], "FPT_2024_Q3_SYN_REV_REC_001_V01")
            self.assertEqual(records_by_target["supported"]["input"]["data"]["report_id"], "FPT_2024_Q3_SYN_REV_REC_001_V01")
            weak_generation = records_by_target["weakly_supported"]["metadata"]["generation_metadata"]
            self.assertEqual(weak_generation["synthetic_report_id"], "FPT_2024_Q3_SYN_REV_REC_002_V01")
            self.assertEqual(records_by_target["weakly_supported"]["input"]["data"]["report_id"], "FPT_2024_Q3_SYN_REV_REC_002_V01")
            not_supported_generation = records_by_target["not_supported"]["metadata"]["generation_metadata"]
            self.assertEqual(not_supported_generation["synthetic_report_id"], "FPT_2024_Q3_SYN_REV_REC_003_V01")
            self.assertEqual(records_by_target["not_supported"]["input"]["data"]["report_id"], "FPT_2024_Q3_SYN_REV_REC_003_V01")
            insufficient_generation = records_by_target["insufficient_evidence"]["metadata"]["generation_metadata"]
            self.assertEqual(insufficient_generation["synthetic_report_id"], "FPT_2024_Q3_SYN_REV_REC_004_V01")
            self.assertEqual(records_by_target["insufficient_evidence"]["input"]["data"]["report_id"], "FPT_2024_Q3_SYN_REV_REC_004_V01")

            manifest = _read_json(manifest_path)
            self.assertEqual(manifest["run_id"], "grounded-smoke")
            self.assertEqual(manifest["scenario"]["scenario_id"], "revenue_receivables_divergence_v1")
            self.assertEqual(manifest["scenario"]["version"], "1.0.0")
            self.assertEqual(manifest["records_written"], 32)
            self.assertEqual(manifest["base_group"], "FPT_2024_Q3_CLEAN")
            self.assertEqual(set(manifest["synthetic_groups"]), set(_synthetic_group_counts("SYN_REV_REC", ["FPT_2024_Q3"])))
            self.assertEqual(
                manifest["synthetic_groups"][:4],
                [
                    "FPT_2024_Q3_SYN_REV_REC_001_V01",
                    "FPT_2024_Q3_SYN_REV_REC_002_V01",
                    "FPT_2024_Q3_SYN_REV_REC_003_V01",
                    "FPT_2024_Q3_SYN_REV_REC_004_V01",
                ],
            )

            metrics = _read_json(metrics_path)
            self.assertEqual(metrics["run_id"], "grounded-smoke")
            self.assertEqual(metrics["generated_records"], 32)
            self.assertEqual(metrics["risk_categories"], {"revenue_income_recognition_risk": 32})
            self.assertEqual(metrics["target_support_levels"], _support_counts(8))
            self.assertEqual(metrics["report_profiles"], {"standard_corporate": 32})
            self.assertEqual(metrics["base_groups"], {"FPT_2024_Q3_CLEAN": 32})
            self.assertEqual(metrics["synthetic_groups"], _synthetic_group_counts("SYN_REV_REC", ["FPT_2024_Q3"]))

    def test_generated_candidate_passes_packet_validation_raw_staging_and_gate_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clean_artifact = root / "clean_structured_report.json"
            second_artifact = root / "second_clean_structured_report.json"
            generated_dir = root / "generated"
            raw_dir = root / "raw"
            gate_root = root / "gate"
            _write_json(clean_artifact, _clean_structured_report(source_group_key="vietstock:FPT:2024_Q3"))
            _write_json(second_artifact, _clean_structured_report("VNM", "2024_Q4", source_group_key="vietstock:VNM:2024_Q4"))

            generated = _run_generator(
                "--clean-report-artifact",
                str(clean_artifact),
                "--clean-report-artifact",
                str(second_artifact),
                "--output-dir",
                str(generated_dir),
                "--run-id",
                "grounded-chain",
            )
            self.assertEqual(generated.returncode, 0, generated.stdout + generated.stderr)
            staged_path = generated_dir / "staged_synthetic_packets.jsonl"
            staged_records = _read_jsonl(staged_path)
            self.assertEqual(len(staged_records), 64)
            for record in staged_records:
                validate_detector_packet(record["input"]["data"])

            staged = _run_module(
                "research.synthetic_raw_stager",
                "--input-jsonl",
                str(staged_path),
                "--output-dir",
                str(raw_dir),
                "--run-id",
                "raw-chain",
            )
            self.assertEqual(staged.returncode, 0, staged.stdout + staged.stderr)
            raw_path = raw_dir / "synthetic_injected_raw.jsonl"
            self.assertEqual(len(_read_jsonl(raw_path)), 64)
            raw_metrics = _read_json(raw_dir / "metrics.json")
            self.assertEqual(raw_metrics["records_written"], 64)
            self.assertEqual(raw_metrics["risk_categories"], {"revenue_income_recognition_risk": 64})
            self.assertEqual(raw_metrics["target_support_levels"], _support_counts(16))
            self.assertEqual(raw_metrics["report_profiles"], {"standard_corporate": 64})
            self.assertEqual(raw_metrics["base_groups"], {"FPT_2024_Q3_CLEAN": 32, "VNM_2024_Q4_CLEAN": 32})
            self.assertEqual(raw_metrics["synthetic_groups"], _synthetic_group_counts("SYN_REV_REC", ["FPT_2024_Q3", "VNM_2024_Q4"]))
            self.assertEqual(raw_metrics["source_groups"], {"vietstock:FPT:2024_Q3": 32, "vietstock:VNM:2024_Q4": 32})
            self.assertEqual(raw_metrics["scenario_ids"], {"revenue_receivables_divergence_v1": 64})

            gate = _run_module(
                "research.synthetic_detector_assessment_gate",
                "--input-jsonl",
                str(raw_path),
                "--output-root",
                str(gate_root),
                "--mode",
                "dry_run",
                "--run-id",
                "gate-chain",
            )
            self.assertEqual(gate.returncode, 0, gate.stdout + gate.stderr)
            gate_metrics = _read_json(gate_root / "gate-chain" / "metrics.json")
            self.assertEqual(gate_metrics["input_records"], 64)
            self.assertEqual(gate_metrics["input_target_support_levels"], _support_counts(16))
            self.assertEqual(gate_metrics["input_risk_categories"], {"revenue_income_recognition_risk": 64})
            self.assertEqual(gate_metrics["input_report_profiles"], {"standard_corporate": 64})
            self.assertEqual(gate_metrics["input_base_groups"], {"FPT_2024_Q3_CLEAN": 32, "VNM_2024_Q4_CLEAN": 32})
            self.assertEqual(gate_metrics["input_synthetic_groups"], _synthetic_group_counts("SYN_REV_REC", ["FPT_2024_Q3", "VNM_2024_Q4"]))
            self.assertEqual(
                gate_metrics["input_source_groups"],
                {"vietstock:FPT:2024_Q3": 32, "vietstock:VNM:2024_Q4": 32},
            )
            self.assertEqual(gate_metrics["input_scenario_ids"], {"revenue_receivables_divergence_v1": 64})

    def test_public_command_keeps_hidden_generation_metadata_out_of_both_detector_packets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clean_artifact = root / "clean_structured_report.json"
            output_dir = root / "generated"
            _write_json(clean_artifact, _clean_structured_report())

            completed = _run_generator(
                "--clean-report-artifact",
                str(clean_artifact),
                "--output-dir",
                str(output_dir),
                "--run-id",
                "grounded-leakage-boundary",
            )

            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            records = _read_jsonl(output_dir / "staged_synthetic_packets.jsonl")
            self.assertEqual(len(records), 32)
            for record in records:
                self.assertNotIn("output", record)
                self.assertEqual(record["input"]["type"], "DetectorPacket")
                packet = record["input"]["data"]
                validate_detector_packet(packet)
                detector_visible_text = json.dumps(packet, ensure_ascii=False, sort_keys=True).lower()
                for prohibited_text in [
                    "target_support_level",
                    "hidden_injection_details",
                    "hidden injection",
                    "source_file_sha256",
                    "normalized_text_hash",
                    "table_content_hash",
                    "derived_from_report_artifact_id",
                    "derived_from_source_document_id",
                    "raw_ocr",
                    "raw_pdf",
                    "cache",
                    "omitted_evidence",
                    "final_report",
                ]:
                    self.assertNotIn(prohibited_text, detector_visible_text)

    def test_weakly_supported_family_is_milder_and_tool_finding_matches_visible_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clean_artifact = root / "clean_structured_report.json"
            output_dir = root / "generated"
            _write_json(clean_artifact, _clean_structured_report())

            completed = _run_generator(
                "--clean-report-artifact",
                str(clean_artifact),
                "--output-dir",
                str(output_dir),
                "--run-id",
                "grounded-milder-family",
            )

            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            records_by_target = _records_by_target_first_slot(output_dir / "staged_synthetic_packets.jsonl")
            supported_packet = records_by_target["supported"]["input"]["data"]
            weak_packet = records_by_target["weakly_supported"]["input"]["data"]

            supported_growths = _packet_growths(supported_packet)
            weak_growths = _packet_growths(weak_packet)
            self.assertAlmostEqual(supported_growths["revenue"], 0.25)
            self.assertAlmostEqual(supported_growths["trade_receivables"], 1.25)
            self.assertAlmostEqual(weak_growths["revenue"], 0.25)
            self.assertAlmostEqual(weak_growths["trade_receivables"], 0.355)
            self.assertGreater(
                supported_growths["trade_receivables"] - supported_growths["revenue"],
                weak_growths["trade_receivables"] - weak_growths["revenue"],
            )
            self.assertGreater(weak_growths["trade_receivables"] - weak_growths["revenue"], 0.10)
            self.assertLess(weak_growths["trade_receivables"] - weak_growths["revenue"], 0.11)

            self.assertIn("Trade receivables increased 35.5%", weak_packet["candidate_summary"]["reason_for_candidate"])
            self.assertIn("revenue increased 25.0%", weak_packet["candidate_summary"]["reason_for_candidate"])
            self.assertIn("only slightly above the threshold", weak_packet["candidate_summary"]["reason_for_candidate"])
            self.assertIn("partially mitigates", weak_packet["candidate_summary"]["reason_for_candidate"])
            self.assertEqual(weak_packet["candidate_summary"]["priority"], "medium")
            self.assertEqual(weak_packet["candidate_summary"]["supporting_signal_ids"], ["receivables_growth_outpaces_revenue"])
            self.assertIn("Subsequent collection after period end", weak_packet["relevant_notes"][0]["text"])

            finding = weak_packet["tool_findings"][0]
            self.assertEqual(finding["tool_name"], "receivables_vs_revenue_growth_tool")
            self.assertEqual(finding["signal_id"], "receivables_growth_outpaces_revenue")
            self.assertIs(finding["flag"], True)
            self.assertEqual(finding["metric"], "receivables_growth_minus_revenue_growth_pct_points")
            self.assertEqual(finding["threshold"], "flag if receivables growth exceeds revenue growth by more than 10 percentage points")
            self.assertIn("modestly exceeded", finding["finding_summary"])
            self.assertIn("10.5 percentage points", finding["finding_summary"])
            self.assertEqual(
                [ref["ref_id"] for ref in finding["evidence_refs"]],
                [
                    "FPT_2024_Q3_SYN_REV_REC_002_V01:ROW_REVENUE",
                    "FPT_2024_Q3_SYN_REV_REC_002_V01:ROW_TRADE_RECEIVABLES",
                ],
            )

    def test_generic_weakly_supported_families_use_near_threshold_visible_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clean_artifact = root / "clean_structured_report.json"
            output_dir = root / "generated"
            _write_json(clean_artifact, _clean_structured_report())

            completed = _run_generator(
                "--clean-report-artifact",
                str(clean_artifact),
                "--output-dir",
                str(output_dir),
                "--scenario-family",
                "receivables_credit_quality",
                "--scenario-family",
                "inventory_cost_asset_flow",
                "--scenario-family",
                "expense_liability_understatement",
                "--scenario-family",
                "asset_quality_valuation",
                "--scenario-family",
                "related_party_disclosure",
                "--scenario-family",
                "disclosure_inconsistency",
            )

            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            metrics_by_target = {}
            for record in _read_jsonl(output_dir / "staged_synthetic_packets.jsonl"):
                generation = record["metadata"]["generation_metadata"]
                if (
                    generation["target_support_level"] == "weakly_supported"
                    and generation["variant_slot_id"] == "V1_easy_quantitative_clear"
                ):
                    finding = record["input"]["data"]["tool_findings"][0]
                    metrics_by_target[generation["canonical_target_id"]] = (
                        finding["metric_value"],
                        finding["finding_summary"],
                    )

            self.assertEqual(
                set(metrics_by_target),
                {
                    "receivables_credit_quality__weakly_supported",
                    "inventory_cost_asset_flow__weakly_supported",
                    "expense_liability_understatement__weakly_supported",
                    "asset_quality_valuation__weakly_supported",
                    "related_party_disclosure__weakly_supported",
                    "disclosure_inconsistency__weakly_supported",
                },
            )
            self.assertEqual(metrics_by_target["receivables_credit_quality__weakly_supported"][0], 2.7)
            self.assertLess(metrics_by_target["receivables_credit_quality__weakly_supported"][0], 3.0)
            self.assertLessEqual(metrics_by_target["inventory_cost_asset_flow__weakly_supported"][0], 18.0)
            self.assertLessEqual(metrics_by_target["expense_liability_understatement__weakly_supported"][0], 22.0)
            self.assertLessEqual(metrics_by_target["asset_quality_valuation__weakly_supported"][0], 36.5)
            self.assertLessEqual(metrics_by_target["related_party_disclosure__weakly_supported"][0], 5.3)
            self.assertLessEqual(metrics_by_target["disclosure_inconsistency__weakly_supported"][0], 7.5)
            for _, summary in metrics_by_target.values():
                self.assertRegex(summary, "modestly|just below|only slightly|partly")

            weak_records_by_target = {}
            for record in _read_jsonl(output_dir / "staged_synthetic_packets.jsonl"):
                generation = record["metadata"]["generation_metadata"]
                if (
                    generation["target_support_level"] == "weakly_supported"
                    and generation["variant_slot_id"] == "V1_easy_quantitative_clear"
                ):
                    weak_records_by_target[generation["canonical_target_id"]] = record

            asset_packet = weak_records_by_target["asset_quality_valuation__weakly_supported"]["input"]["data"]
            related_packet = weak_records_by_target["related_party_disclosure__weakly_supported"]["input"]["data"]
            self.assertIn("only slightly above the threshold", asset_packet["candidate_summary"]["reason_for_candidate"])
            self.assertIn("partly to software and project costs", asset_packet["relevant_notes"][0]["text"])
            self.assertIn("just above materiality", related_packet["candidate_summary"]["reason_for_candidate"])
            self.assertIn("short-term settlement intent", related_packet["relevant_notes"][0]["text"])

    def test_public_command_uses_documented_fixture_evidence_profile_and_tool_finding_quality_fields(self) -> None:
        fixture = _clean_structured_report()
        self.assertEqual(fixture["report_period_type"], "quarterly")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clean_artifact = root / "clean_structured_report.json"
            output_dir = root / "generated"
            _write_json(clean_artifact, fixture)

            completed = _run_generator(
                "--clean-report-artifact",
                str(clean_artifact),
                "--output-dir",
                str(output_dir),
                "--run-id",
                "grounded-quality-fields",
            )

            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            for record in _read_jsonl(output_dir / "staged_synthetic_packets.jsonl"):
                generation = record["metadata"]["generation_metadata"]
                target_support_level = generation["target_support_level"]
                evidence_profile = record["metadata"]["evidence_profile"]
                if target_support_level == "insufficient_evidence":
                    self.assertEqual(evidence_profile["evidence_types"], ["table_row"])
                    self.assertEqual(record["input"]["data"]["tool_findings"], [])
                    self.assertEqual(record["input"]["data"]["relevant_notes"], [])
                elif generation["variant_slot_id"] == "V4_hard_missing_required_context":
                    self.assertEqual(evidence_profile["evidence_types"], ["table_row", "note_span"])
                    self.assertTrue(evidence_profile["has_missing_required_evidence"])
                    self.assertFalse(evidence_profile["has_tool_findings"])
                    self.assertEqual(record["input"]["data"]["tool_findings"], [])
                else:
                    self.assertIn("note_span", evidence_profile["evidence_types"])
                    self.assertTrue(evidence_profile["has_tool_findings"])
                    finding = record["input"]["data"]["tool_findings"][0]
                    self.assertEqual(finding["tool_name"], "receivables_vs_revenue_growth_tool")
                    for field in ["metric", "metric_value", "threshold", "finding_summary", "calculation_basis"]:
                        self.assertIn(field, finding)

    def test_public_command_refuses_missing_required_structured_scenario_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clean_artifact = root / "clean_structured_report.json"
            output_dir = root / "generated"
            artifact = _clean_structured_report()
            artifact["structured_evidence"]["rows"] = [
                row for row in artifact["structured_evidence"]["rows"] if row["standard_account"] != "trade_receivables"
            ]
            _write_json(clean_artifact, artifact)

            completed = _run_generator(
                "--clean-report-artifact",
                str(clean_artifact),
                "--output-dir",
                str(output_dir),
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("missing required structured evidence account: trade_receivables", completed.stdout)
            self.assertFalse(output_dir.exists())

    def test_public_command_requires_rich_clean_report_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clean_artifact = root / "partial_clean_structured_report.json"
            output_dir = root / "generated"
            artifact = _clean_structured_report()
            artifact["structured_evidence"]["rows"] = [
                row
                for row in artifact["structured_evidence"]["rows"]
                if row["standard_account"] in {"revenue", "trade_receivables"}
            ]
            _write_json(clean_artifact, artifact)

            completed = _run_generator(
                "--clean-report-artifact",
                str(clean_artifact),
                "--output-dir",
                str(output_dir),
                "--scenario-family",
                "revenue_receivables",
                "--require-rich-clean-report",
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("does not support all final-scale canonical targets", completed.stdout)
            self.assertFalse(output_dir.exists())

    def test_public_command_refuses_missing_split_traceability_before_writing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clean_artifact = root / "clean_structured_report.json"
            output_dir = root / "generated"
            artifact = _clean_structured_report()
            artifact["company_key"] = ""
            _write_json(clean_artifact, artifact)

            completed = _run_generator(
                "--clean-report-artifact",
                str(clean_artifact),
                "--output-dir",
                str(output_dir),
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("company_key", completed.stdout)
            self.assertFalse(output_dir.exists())

    def test_public_command_refuses_detector_visible_omitted_evidence_hint_before_writing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            clean_artifact = root / "clean_structured_report.json"
            output_dir = root / "generated"
            artifact = _clean_structured_report()
            artifact["structured_evidence"]["notes"][0]["text"] = "Omitted evidence: the source note was withheld."
            _write_json(clean_artifact, artifact)

            completed = _run_generator(
                "--clean-report-artifact",
                str(clean_artifact),
                "--output-dir",
                str(output_dir),
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("prohibited", completed.stdout)
            self.assertFalse(output_dir.exists())


def _run_generator(*args: str) -> subprocess.CompletedProcess:
    return _run_module("research.grounded_synthetic_packet_generator", *args)


def _run_module(module: str, *args: str) -> subprocess.CompletedProcess:
    repo_root = Path(__file__).resolve().parents[3]
    pythonpath = os.pathsep.join([str(repo_root / "src"), str(repo_root)])
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": pythonpath},
        check=False,
    )


def _clean_structured_report(
    company_key: str = "FPT",
    period_key: str = "2024_Q3",
    *,
    report_profile: str = "standard_corporate",
    source_group_key: str | None = None,
) -> dict:
    report = {
        "artifact_id": "RPT_FPT_2024_Q3_CLEAN",
        "source_document_id": "DOC_FPT_2024_Q3_FS",
        "report_id": "FPT_2024_Q3_CLEAN",
        "company_key": "FPT",
        "company_name": "FPT Corporation",
        "ticker": "FPT",
        "period_key": "2024_Q3",
        "period": "2024-Q3",
        "report_period_type": "quarterly",
        "report_profile": "standard_corporate",
        "currency": "VND",
        "unit": "million_vnd",
        "language": "vi",
        "traceability": {
            "source_file_sha256": "sha256-source-FPT_2024_Q3",
            "normalized_text_hash": "sha256-normalized-FPT_2024_Q3",
            "table_content_hash": "sha256-table-FPT_2024_Q3",
        },
        "structured_evidence": {
            "rows": [
                {"row_id": "ROW_REVENUE", "standard_account": "revenue", "current_value": 1500, "prior_value": 1200},
                {"row_id": "ROW_TRADE_RECEIVABLES", "standard_account": "trade_receivables", "current_value": 1180, "prior_value": 1000},
                {"row_id": "ROW_NET_PROFIT", "standard_account": "net_profit", "current_value": 420, "prior_value": 300},
                {"row_id": "ROW_OPERATING_CASH_FLOW", "standard_account": "operating_cash_flow", "current_value": 390, "prior_value": 320},
            ],
            "notes": [
                {"note_id": "NOTE_REVENUE", "note_type": "revenue", "text": "Revenue is presented for the current quarter."}
            ],
        },
    }
    if company_key == "FPT" and period_key == "2024_Q3":
        if report_profile != "standard_corporate":
            report = deepcopy(report)
            report["report_profile"] = report_profile
            if report_profile == "insurance":
                report["structured_evidence"]["rows"] = _insurance_structured_rows()
            if report_profile == "securities":
                report["structured_evidence"]["rows"] = _securities_structured_rows()
            if report_profile == "credit_institution":
                report["structured_evidence"]["rows"] = _credit_institution_structured_rows()
        if source_group_key:
            report = deepcopy(report)
            report["traceability"]["source_group_key"] = source_group_key
        return report

    report = deepcopy(report)
    period = period_key.replace("_", "-")
    report.update(
        {
            "artifact_id": f"RPT_{company_key}_{period_key}_CLEAN",
            "source_document_id": f"DOC_{company_key}_{period_key}_FS",
            "report_id": f"{company_key}_{period_key}_CLEAN",
            "company_key": company_key,
            "company_name": f"{company_key} Corporation",
            "ticker": company_key,
            "period_key": period_key,
            "period": period,
            "report_profile": report_profile,
        }
    )
    report["traceability"] = {
        "source_file_sha256": f"sha256-source-{company_key}_{period_key}",
        "normalized_text_hash": f"sha256-normalized-{company_key}_{period_key}",
        "table_content_hash": f"sha256-table-{company_key}_{period_key}",
    }
    if source_group_key:
        report["traceability"]["source_group_key"] = source_group_key
    if report_profile == "insurance":
        report["structured_evidence"]["rows"] = _insurance_structured_rows()
    if report_profile == "securities":
        report["structured_evidence"]["rows"] = _securities_structured_rows()
    if report_profile == "credit_institution":
        report["structured_evidence"]["rows"] = _credit_institution_structured_rows()
    return report


def _insurance_structured_rows() -> list[dict]:
    return [
        {
            "row_id": "ROW_GROSS_WRITTEN_PREMIUM",
            "standard_account": "gross_written_premium",
            "current_value": 1500,
            "prior_value": 1200,
        },
        {
            "row_id": "ROW_PREMIUM_RECEIVABLES",
            "standard_account": "premium_receivables",
            "current_value": 1180,
            "prior_value": 1000,
        },
        {
            "row_id": "ROW_OPERATING_CASH_FLOW",
            "standard_account": "operating_cash_flow",
            "current_value": 390,
            "prior_value": 320,
        },
    ]


def _securities_structured_rows() -> list[dict]:
    return [
        {
            "row_id": "ROW_PROFIT_AFTER_TAX",
            "standard_account": "profit_after_tax",
            "current_value": 1500,
            "prior_value": 1200,
        },
        {
            "row_id": "ROW_MARGIN_LENDING",
            "standard_account": "margin_lending",
            "current_value": 1180,
            "prior_value": 1000,
        },
        {
            "row_id": "ROW_MARGIN_IMPAIRMENT",
            "standard_account": "margin_impairment",
            "current_value": -40,
            "prior_value": -35,
        },
        {
            "row_id": "ROW_FVTPL_ASSETS",
            "standard_account": "fvtpl_assets",
            "current_value": 900,
            "prior_value": 760,
        },
        {
            "row_id": "ROW_FVTPL_UNREALIZED_GAIN",
            "standard_account": "fvtpl_unrealized_gain",
            "current_value": 180,
            "prior_value": 120,
        },
        {
            "row_id": "ROW_OPERATING_CASH_FLOW",
            "standard_account": "operating_cash_flow",
            "current_value": -390,
            "prior_value": -320,
        },
    ]


def _credit_institution_structured_rows() -> list[dict]:
    return [
        {"row_id": "ROW_PROFIT_AFTER_TAX", "standard_account": "profit_after_tax", "current_value": 1500, "prior_value": 1200},
        {"row_id": "ROW_LOANS_TO_CUSTOMERS", "standard_account": "loans_to_customers", "current_value": 1180, "prior_value": 1000},
        {"row_id": "ROW_LOAN_GROUP_1", "standard_account": "loan_group_1", "current_value": 950, "prior_value": 880},
        {"row_id": "ROW_LOAN_GROUP_2", "standard_account": "loan_group_2", "current_value": 80, "prior_value": 60},
        {"row_id": "ROW_LOAN_GROUP_3", "standard_account": "loan_group_3", "current_value": 50, "prior_value": 35},
        {"row_id": "ROW_LOAN_GROUP_4", "standard_account": "loan_group_4", "current_value": 35, "prior_value": 20},
        {"row_id": "ROW_LOAN_GROUP_5", "standard_account": "loan_group_5", "current_value": 30, "prior_value": 18},
        {"row_id": "ROW_GENERAL_PROVISION", "standard_account": "general_provision", "current_value": 22, "prior_value": 20},
        {"row_id": "ROW_SPECIFIC_PROVISION", "standard_account": "specific_provision", "current_value": 52, "prior_value": 48},
        {"row_id": "ROW_OPERATING_CASH_FLOW", "standard_account": "operating_cash_flow", "current_value": 390, "prior_value": 320},
    ]


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records), encoding="utf-8")


def _write_real_manual_release(release_dir: Path, *, company_key: str, period_key: str) -> None:
    report_id = f"{company_key}_{period_key}"
    release_dir.mkdir(parents=True, exist_ok=True)
    _write_json(release_dir / "manifest.json", {"num_examples_by_split": {"validation": 1}})
    _write_jsonl(
        release_dir / "validation.jsonl",
        [
            {
                "source_type": "human_gold_real_report",
                "input": {"data": {"report_id": f"{report_id}_parent"}},
                "metadata": {
                    "split_metadata": {
                        "group_key": report_id,
                        "source_file_sha256": f"sha256-source-{report_id}",
                        "normalized_text_hash": f"sha256-normalized-{report_id}",
                        "table_content_hash": f"sha256-table-{report_id}",
                        "derived_from_report_artifact_id": f"RPT_{report_id}_CLEAN",
                        "derived_from_source_document_id": f"DOC_{report_id}_FS",
                    },
                    "audit_metadata": {
                        "dataset_artifact_traceability": {
                            "derived_from_report_artifact_id": f"RPT_{report_id}_CLEAN",
                            "derived_from_source_document_id": f"DOC_{report_id}_FS",
                        }
                    },
                },
            }
        ],
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _records_by_target_first_slot(path: Path) -> dict[str, dict]:
    return {
        record["metadata"]["generation_metadata"]["target_support_level"]: record
        for record in _read_jsonl(path)
        if record["metadata"]["generation_metadata"]["variant_slot_id"] == "V1_easy_quantitative_clear"
    }


def _packet_growths(packet: dict) -> dict[str, float]:
    rows_by_account = {row["standard_account"]: row for row in packet["relevant_table_rows"]}
    return {
        account: (
            (row["values"]["current"]["value"] - row["values"]["prior"]["value"])
            / abs(row["values"]["prior"]["value"])
        )
        for account, row in rows_by_account.items()
    }


def _packet_profile(packet: dict) -> dict:
    return {
        "candidate_reason": packet["candidate_summary"]["reason_for_candidate"],
        "has_tool_findings": bool(packet["tool_findings"]),
        "tool_flag": packet["tool_findings"][0]["flag"] if packet["tool_findings"] else None,
        "has_variance_explanation": bool(packet["relevant_variance_explanations"]),
        "note_text_length": sum(len(note["text"]) for note in packet["relevant_notes"]),
        "row_ids": [row["row_id"] for row in packet["relevant_table_rows"]],
    }


def _support_counts(count_per_target: int) -> dict[str, int]:
    return {target: count_per_target for target in sorted(ALL_SUPPORT_TARGETS)}


def _variant_slots() -> set[str]:
    return {
        "V1_easy_quantitative_clear",
        "V2_easy_quantitative_contradiction",
        "V3_medium_partial_secondary_evidence",
        "V4_hard_missing_required_context",
        "V5_note_or_disclosure_quality",
        "V6_tool_finding_contradiction",
        "V7_profile_specific_mapping",
        "V8_language_or_account_label_style",
    }


def _synthetic_group_counts(tag: str, base_ids: list[str]) -> dict[str, int]:
    return {
        f"{base_id}_{tag}_{suffix:03d}_V{variant_index:02d}": 1
        for base_id in base_ids
        for suffix in range(1, 5)
        for variant_index in range(1, 9)
    }


if __name__ == "__main__":
    unittest.main()
