import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from research.testing import FakeOcrExtractionAdapter
from research.real_source_extraction_adapter import (
    run_real_source_extraction_adapter,
)


class Wave4RealSourceExtractionAdapterTest(unittest.TestCase):
    def test_live_ocr_persists_raw_table_artifacts_even_without_candidates(self):
        target_path = Path("artifacts/source_documents/BIC_2026_Q1_parent_545811.pdf")
        prior_path = Path("artifacts/source_documents/BIC_2025_Q1_parent_503892.pdf")

        ocr_adapter = FakeOcrExtractionAdapter(
            raw_html="<table><tr><td>raw table only</td></tr></table>",
            raw_tables=[{"raw_table_id": "RAW_001", "cells": [["raw table only"]]}],
            provider_metadata={"provider": "nanonets_ocr_3_docstrange", "status": "completed"},
        )

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache"
            result = run_real_source_extraction_adapter(
                output_dir=Path(tmp) / "out",
                inventory_records=[
                    _inventory_record(
                        ticker="BIC",
                        year=2026,
                        quarter="Q1",
                        report_basis="parent",
                        report_profile_guess="insurance",
                        target_path=str(target_path),
                        prior_year_path=str(prior_path),
                    )
                ],
                raw_extraction_cache_dir=cache_dir,
                live_ocr_enabled=True,
                ocr_adapter=ocr_adapter,
                write_artifacts=False,
            )
            cached_payloads = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in sorted(cache_dir.glob("*__raw_extraction_artifact.json"))
            ]

        self.assertEqual(result.records[0].status, "needs_follow_up")
        self.assertEqual(len(cached_payloads), 2)
        self.assertTrue(all(payload["raw_tables"] for payload in cached_payloads))
        self.assertTrue(
            all("extraction_candidates" not in payload for payload in cached_payloads)
        )
        self.assertIn(
            "real_pdf_table_extraction_unavailable",
            {
                item["reason_code"]
                for item in result.records[0].dataset_eligibility_result.records
            },
        )

    def test_cache_miss_with_live_ocr_disabled_returns_role_specific_follow_up(self):
        target_path = Path("artifacts/source_documents/BIC_2026_Q1_parent_545811.pdf")
        prior_path = Path("artifacts/source_documents/BIC_2025_Q1_parent_503892.pdf")

        with tempfile.TemporaryDirectory() as tmp:
            result = run_real_source_extraction_adapter(
                output_dir=Path(tmp) / "out",
                inventory_records=[
                    _inventory_record(
                        ticker="BIC",
                        year=2026,
                        quarter="Q1",
                        report_basis="parent",
                        report_profile_guess="insurance",
                        target_path=str(target_path),
                        prior_year_path=str(prior_path),
                    )
                ],
                raw_extraction_cache_dir=Path(tmp) / "empty_cache",
                live_ocr_enabled=False,
                write_artifacts=False,
            )

        record = result.records[0]
        self.assertEqual(record.status, "needs_follow_up")
        self.assertEqual(record.detector_packets, [])
        records_by_role = {}
        for follow_up in record.dataset_eligibility_result.records:
            records_by_role.setdefault(follow_up.get("role"), set()).add(follow_up["reason_code"])
        for role in ("target", "prior_year"):
            self.assertGreaterEqual(
                records_by_role.get(role, set()),
                {"cached_raw_extraction_artifact_missing", "live_ocr_not_configured"},
            )

    def test_cached_artifact_source_fingerprint_mismatch_returns_follow_up(self):
        target_path = Path("artifacts/source_documents/BIC_2026_Q1_parent_545811.pdf")
        prior_path = Path("artifacts/source_documents/BIC_2025_Q1_parent_503892.pdf")
        record = _inventory_record(
            ticker="BIC",
            year=2026,
            quarter="Q1",
            report_basis="parent",
            report_profile_guess="insurance",
            target_path=str(target_path),
            prior_year_path=str(prior_path),
        )

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache"
            cache_dir.mkdir()
            _write_cached_artifact(
                cache_dir,
                case_id="real_source_BIC_2026_Q1_parent",
                role="target",
                source_document_id="DOC_BIC_2026_Q1_FS",
                source_hash=_sha256(target_path),
                package_id="PKG_BIC_2026_Q1_PARENT",
                artifact_fingerprint={"hash_algorithm": "sha256", "hash_value": "not-the-selected-pdf"},
            )

            result = run_real_source_extraction_adapter(
                output_dir=Path(tmp) / "out",
                inventory_records=[record],
                raw_extraction_cache_dir=cache_dir,
                live_ocr_enabled=False,
                write_artifacts=False,
            )

        follow_ups = result.records[0].dataset_eligibility_result.records
        self.assertIn(
            ("target", "raw_extraction_artifact_source_mismatch"),
            {(item.get("role"), item["reason_code"]) for item in follow_ups},
        )

    def test_selected_zip_member_identity_uses_member_hash_not_container_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp)
            target_zip = fixture_dir / "vnr_2026_q1.zip"
            prior_path = fixture_dir / "vnr_2025_q1.pdf"
            target_member = "selected/vnr_2026_q1.pdf"
            target_member_bytes = b"%PDF selected member bytes"
            _write_zip_member_bytes(target_zip, target_member, target_member_bytes)
            container_hash = _sha256(target_zip)
            prior_path.write_bytes(b"%PDF prior bytes")

            result = run_real_source_extraction_adapter(
                output_dir=fixture_dir / "out",
                inventory_records=[
                    _inventory_record(
                        ticker="VNR",
                        company_name="VNR Company",
                        year=2026,
                        quarter="Q1",
                        report_basis="parent",
                        report_profile_guess="insurance",
                        target_container_path=str(target_zip),
                        target_selected_member_path=target_member,
                        prior_year_path=str(prior_path),
                    )
                ],
                raw_extraction_cache_dir=fixture_dir / "empty_cache",
                live_ocr_enabled=False,
                write_artifacts=False,
            )

        record = result.records[0]
        self.assertEqual(record.status, "needs_follow_up")
        self.assertEqual(record.audit_traceability["target_selected_member_path"], target_member)
        self.assertEqual(record.audit_traceability["target_selected_member_sha256"], _sha256_bytes(target_member_bytes))
        self.assertNotEqual(container_hash, record.audit_traceability["target_selected_member_sha256"])

    def test_missing_selected_zip_member_returns_role_specific_follow_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp)
            target_zip = fixture_dir / "vnr_2026_q1.zip"
            prior_path = fixture_dir / "vnr_2025_q1.pdf"
            _write_zip_member(target_zip, "other.pdf", "not selected")
            prior_path.write_bytes(b"%PDF prior bytes")

            result = run_real_source_extraction_adapter(
                output_dir=fixture_dir / "out",
                inventory_records=[
                    _inventory_record(
                        ticker="VNR",
                        company_name="VNR Company",
                        year=2026,
                        quarter="Q1",
                        report_basis="parent",
                        report_profile_guess="insurance",
                        target_container_path=str(target_zip),
                        target_selected_member_path="selected/vnr_2026_q1.pdf",
                        prior_year_path=str(prior_path),
                    )
                ],
                raw_extraction_cache_dir=fixture_dir / "empty_cache",
                live_ocr_enabled=False,
                write_artifacts=False,
            )

        follow_ups = result.records[0].dataset_eligibility_result.records
        self.assertIn(
            ("target", "selected_zip_member_unavailable"),
            {(item.get("role"), item["reason_code"]) for item in follow_ups},
        )

    def test_bic_parent_direct_pdf_loads_seeded_cached_raw_extraction_artifacts(self):
        inventory = json.loads(Path("artifacts/real_source_curation_inventory.json").read_text(encoding="utf-8"))
        bic_parent = next(record for record in inventory["records"] if record["case_id"] == "BIC_2026_Q1_parent")

        with tempfile.TemporaryDirectory() as tmp:
            result = run_real_source_extraction_adapter(
                output_dir=Path(tmp) / "out",
                inventory_records=[bic_parent],
                write_artifacts=False,
            )

        record = result.records[0]
        self.assertIn(record.status, {"packet_ready_unadjudicated", "extracted_no_candidates"})
        self.assertEqual(record.detector_assessments if hasattr(record, "detector_assessments") else [], [])
        self.assertEqual(record.quality_gate_result.status, "passed")
        enabled_tools = {item["tool_name"] for item in record.tool_gating_records if item["status"] == "enabled"}
        self.assertIn("insurance_premium_receivable_coherence_tool", enabled_tools)
        if record.detector_packets:
            self.assertEqual(record.status, "packet_ready_unadjudicated")
            self.assertEqual(record.detector_packets[0]["metadata"]["report_profile"], "insurance")
            self.assertEqual(record.detector_packets[0]["metadata"]["insurance_subprofile"], "non_life")

    def test_curation_inventory_is_accepted_without_unknown_case_ids(self):
        inventory = json.loads(Path("artifacts/real_source_curation_inventory.json").read_text(encoding="utf-8"))

        with tempfile.TemporaryDirectory() as tmp:
            result = run_real_source_extraction_adapter(
                output_dir=Path(tmp) / "out",
                inventory_records=inventory["records"],
                write_artifacts=False,
            )

        case_ids = {record.case_id for record in result.records}
        self.assertNotIn("real_source_unknown_unknown_unknown", case_ids)
        self.assertIn("real_source_BIC_2026_Q1_parent", case_ids)
        self.assertIn("real_source_BIC_2026_Q1_consolidated", case_ids)
        self.assertEqual(len(case_ids), len(result.records))

    def test_curation_source_ready_records_behave_honestly(self):
        inventory = json.loads(Path("artifacts/real_source_curation_inventory.json").read_text(encoding="utf-8"))

        with tempfile.TemporaryDirectory() as tmp:
            result = run_real_source_extraction_adapter(
                output_dir=Path(tmp) / "out",
                inventory_records=inventory["records"],
                write_artifacts=False,
            )

        by_case = {record.case_id: record for record in result.records}
        bic_parent = by_case["real_source_BIC_2026_Q1_parent"]
        bic_consolidated = by_case["real_source_BIC_2026_Q1_consolidated"]
        pre_parent = by_case["real_source_PRE_2026_Q1_parent"]
        bmi_follow_up = by_case["real_source_BMI_2025_Q1_parent"]

        self.assertEqual(bic_parent.status, "packet_ready_unadjudicated")
        self.assertGreater(len(bic_parent.detector_packets), 0)
        self.assertEqual(bic_parent.audit_traceability["source_file_sha256"], "0a414091e7f17bbb4a60cab55d86cf90f4851438d77043152de583b8a1db945b")

        self.assertEqual(bic_consolidated.status, "needs_follow_up")
        self.assertIn("cached_raw_extraction_artifact_missing", {item["reason_code"] for item in bic_consolidated.dataset_eligibility_result.records})

        self.assertEqual(pre_parent.status, "needs_follow_up")
        self.assertIn("cached_raw_extraction_artifact_missing", {item["reason_code"] for item in pre_parent.dataset_eligibility_result.records})
        self.assertIn("PRE_2026_Q1_parent_543910", pre_parent.audit_traceability["selected_member_path"])
        self.assertNotEqual(pre_parent.audit_traceability["source_file_sha256"], pre_parent.audit_traceability["selected_member_sha256"])

        self.assertEqual(bmi_follow_up.status, "blocked")
        self.assertIn("corrected_value_unresolved", {item["reason_code"] for item in bmi_follow_up.dataset_eligibility_result.records})

    def test_actual_local_direct_pdf_returns_precise_extraction_follow_up(self):
        target_path = Path("artifacts/source_documents/BIC_2026_Q1_parent_545811.pdf")
        prior_path = Path("artifacts/source_documents/BIC_2025_Q1_parent_503892.pdf")
        self.assertTrue(target_path.read_bytes().startswith(b"%PDF"))
        self.assertTrue(prior_path.read_bytes().startswith(b"%PDF"))

        with tempfile.TemporaryDirectory() as tmp:
            result = run_real_source_extraction_adapter(
                output_dir=Path(tmp) / "out",
                inventory_records=[
                    _inventory_record(
                        ticker="BIC",
                        year=2026,
                        quarter="Q1",
                        report_basis="parent",
                        report_profile_guess="insurance",
                        target_path=str(target_path),
                        prior_year_path=str(prior_path),
                    )
                ],
                write_artifacts=False,
            )

        record = result.records[0]
        self.assertEqual(record.status, "needs_follow_up")
        self.assertEqual(record.detector_packets, [])
        reason_codes = {item["reason_code"] for item in record.dataset_eligibility_result.records}
        self.assertGreaterEqual(reason_codes, {"cached_raw_extraction_artifact_missing", "live_ocr_not_configured"})

    def test_actual_local_zip_member_pdf_preserves_member_identity_and_hash(self):
        target_container = Path("artifacts/source_documents/PRE_2026_Q1_parent_543910.zip")
        target_member = Path(
            "artifacts/source_documents/extracted/PRE_2026_Q1_parent_543910/"
            "1_pre_2026_4_28_8354618_vn_baocaotaichinh_q1_2026.pdf"
        )
        prior_container = Path("artifacts/source_documents/PRE_2025_Q1_parent_ordinary_502357.zip")
        prior_member = Path(
            "artifacts/source_documents/extracted/PRE_2025_Q1_parent_ordinary_502357/"
            "2_pre_2025_4_22_57cbb0c_vn_baocaotaichinh_q1_2025.pdf"
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_real_source_extraction_adapter(
                output_dir=Path(tmp) / "out",
                inventory_records=[
                    _inventory_record(
                        ticker="PRE",
                        year=2026,
                        quarter="Q1",
                        report_basis="parent",
                        report_profile_guess="insurance",
                        target_container_path=str(target_container),
                        target_selected_member_path=str(target_member),
                        target_selected_member_sha256=_sha256(target_member),
                        prior_year_container_path=str(prior_container),
                        prior_year_selected_member_path=str(prior_member),
                        prior_year_selected_member_sha256=_sha256(prior_member),
                        non_detector_visible_provenance={
                            "company_period_group_key": "PRE_2026_Q1_parent",
                            "source_file_sha256": _sha256(target_container),
                        },
                    )
                ],
                write_artifacts=False,
            )

        record = result.records[0]
        self.assertEqual(record.status, "needs_follow_up")
        reason_codes = {item["reason_code"] for item in record.dataset_eligibility_result.records}
        self.assertGreaterEqual(reason_codes, {"cached_raw_extraction_artifact_missing", "live_ocr_not_configured"})
        self.assertEqual(record.audit_traceability["target_selected_member_path"], str(target_member))
        self.assertEqual(record.audit_traceability["target_selected_member_sha256"], _sha256(target_member))
        self.assertNotEqual(record.audit_traceability["source_file_sha256"], record.audit_traceability["target_selected_member_sha256"])

    def test_bic_cached_packet_excludes_detector_labels_and_prohibited_surfaces(self):
        inventory = json.loads(Path("artifacts/real_source_curation_inventory.json").read_text(encoding="utf-8"))
        bic_parent = next(record for record in inventory["records"] if record["case_id"] == "BIC_2026_Q1_parent")

        with tempfile.TemporaryDirectory() as tmp:
            result = run_real_source_extraction_adapter(
                output_dir=Path(tmp) / "out",
                inventory_records=[bic_parent],
                write_artifacts=False,
            )

        self.assertEqual(result.status, "completed")
        self.assertEqual(len(result.records), 1)
        record = result.records[0]
        self.assertEqual(record.status, "packet_ready_unadjudicated")
        self.assertGreater(len(record.detector_packets), 0)
        self.assertEqual(record.dataset_eligibility_result.status, "passed")
        self.assertEqual(result.detector_assessments, [])
        self.assertNotIn("support_level", str(record.detector_packets))

        self.assertIn("normalized_text_hash", record.audit_traceability)
        self.assertIn("table_content_hash", record.audit_traceability)
        packet_surface = str(record.detector_packets)
        for prohibited in [
            "artifacts/source_documents",
            "0a414091e7f17bbb4a60cab55d86cf90f4851438d77043152de583b8a1db945b",
            "67309532ac7f157c62385aec31761edff3c33c0b1a4f089952a71bb549b20354",
            "raw_ocr_text",
            "raw_coordinates",
            "cache_record_id",
            "provider_metadata",
            "operative_corrected_source_document_id",
            "source_package_id",
            "superseded_source_document_ids",
            "proof_elements",
            "correction_enriched_evaluation_pool",
            "sampling_pool",
            "selected_member_sha256",
            "manual_extraction",
        ]:
            self.assertNotIn(prohibited, packet_surface)

    def test_vnr_zip_record_extracts_selected_member_not_container(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp)
            target_zip = fixture_dir / "vnr_2026_q1.zip"
            prior_zip = fixture_dir / "vnr_2025_q1.zip"
            target_member = "selected/vnr_2026_q1.pdf"
            prior_member = "selected/vnr_2025_q1.pdf"
            _write_zip_member(target_zip, target_member, _standard_corporate_source_json("VNR", "2026-Q1", target=True))
            _write_zip_member(prior_zip, prior_member, _standard_corporate_source_json("VNR", "2025-Q1", target=False))

            result = run_real_source_extraction_adapter(
                output_dir=fixture_dir / "out",
                inventory_records=[
                    _inventory_record(
                        inventory_record_type="ordinary_control_candidate",
                        ticker="VNR",
                        company_name="VNR Company",
                        year=2026,
                        quarter="Q1",
                        report_profile_guess="standard_corporate",
                        sampling_pool="ordinary_filing_evaluation_pool",
                        curation_state="ordinary_control_candidate",
                        target_container_path=str(target_zip),
                        target_selected_member_path=target_member,
                        prior_year_container_path=str(prior_zip),
                        prior_year_selected_member_path=prior_member,
                    )
                ],
                write_artifacts=False,
            )

        record = result.records[0]
        self.assertEqual(record.status, "needs_follow_up")
        self.assertEqual(record.audit_traceability["target_selected_member_path"], target_member)
        self.assertIn("target_selected_member_sha256", record.audit_traceability)
        self.assertIn("cached_raw_extraction_artifact_missing", {item["reason_code"] for item in record.dataset_eligibility_result.records})
        packet_surface = str(record.detector_packets)
        self.assertNotIn(str(target_zip), packet_surface)
        self.assertNotIn(target_member, packet_surface)
        self.assertNotIn("ordinary_filing_evaluation_pool", packet_surface)

    def test_unresolved_correction_follow_up_is_blocked_before_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_real_source_extraction_adapter(
                output_dir=Path(tmp) / "out",
                inventory_records=[
                    _inventory_record(
                        ticker="BMI",
                        year=2025,
                        quarter="Q1",
                        curation_state="needs_follow_up",
                        non_detector_visible_provenance={
                            "reason_codes": ["corrected_value_resolution_required_before_dataset_eligibility"]
                        },
                    )
                ],
                write_artifacts=False,
            )

        record = result.records[0]
        self.assertEqual(record.status, "blocked")
        self.assertEqual(record.detector_packets, [])
        self.assertEqual(record.tool_findings, [])
        self.assertIn(
            "corrected_value_resolution_required_before_dataset_eligibility",
            {item["reason_code"] for item in record.dataset_eligibility_result.records},
        )

    def test_missing_selected_zip_member_returns_follow_up_without_packets(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture_dir = Path(tmp)
            target_zip = fixture_dir / "vnr_2026_q1.zip"
            prior_zip = fixture_dir / "vnr_2025_q1.zip"
            _write_zip_member(target_zip, "other.pdf", _standard_corporate_source_json("VNR", "2026-Q1", target=True))
            _write_zip_member(prior_zip, "selected/vnr_2025_q1.pdf", _standard_corporate_source_json("VNR", "2025-Q1", target=False))

            result = run_real_source_extraction_adapter(
                output_dir=fixture_dir / "out",
                inventory_records=[
                    _inventory_record(
                        inventory_record_type="ordinary_control_candidate",
                        ticker="VNR",
                        company_name="VNR Company",
                        year=2026,
                        quarter="Q1",
                        report_profile_guess="standard_corporate",
                        sampling_pool="ordinary_filing_evaluation_pool",
                        curation_state="ordinary_control_candidate",
                        target_container_path=str(target_zip),
                        target_selected_member_path="selected/vnr_2026_q1.pdf",
                        prior_year_container_path=str(prior_zip),
                        prior_year_selected_member_path="selected/vnr_2025_q1.pdf",
                    )
                ],
                write_artifacts=False,
            )

        record = result.records[0]
        self.assertEqual(record.status, "needs_follow_up")
        self.assertEqual(record.detector_packets, [])
        self.assertIn("selected_zip_member_unavailable", {item["reason_code"] for item in record.dataset_eligibility_result.records})


def _inventory_record(**overrides):
    record = {
        "inventory_record_type": "correction_enriched_quarterly_candidate",
        "ticker": "BIC",
        "company_name": "BIC Company",
        "year": 2026,
        "quarter": "Q1",
        "report_period_type": "quarterly",
        "report_basis": "parent",
        "report_profile_guess": "credit_institution",
        "source_entry_identity": {"file_id": "bic-2026-q1", "title": "BIC Q1 2026"},
        "source_document_identity": {"source_document_id": "DOC_BIC_2026_Q1_FS"},
        "curation_state": "correction_enriched_candidate",
        "sampling_pool": "correction_enriched_evaluation_pool",
        "label_policy": {
            "assigns_detector_assessment_label": False,
            "provenance_is_positive_label": False,
            "detector_visible": False,
        },
        "non_detector_visible_provenance": {
            "company_period_group_key": "BIC_2026_Q1_parent",
            "source_file_sha256": "sha256-bic-2026-q1",
            "derived_from_report_artifact_id": "RPT_BIC_2026_Q1",
            "derived_from_source_document_id": "DOC_BIC_2026_Q1_FS",
        },
    }
    record.update(overrides)
    return record


def _credit_institution_source_json(ticker, period, *, target):
    values = {
        "loans_to_customers": 1500 if target else 1000,
        "general_provision": 80 if target else 70,
        "specific_provision": 90 if target else 80,
        "loan_group_1": 1200 if target else 850,
        "loan_group_2": 120 if target else 100,
        "loan_group_3": 90 if target else 30,
        "loan_group_4": 55 if target else 15,
        "loan_group_5": 35 if target else 5,
    }
    payload = {
        "metadata": {
            "company_name": f"{ticker} Company",
            "ticker": ticker,
            "period": period,
            "report_profile": "credit_institution",
            "report_basis": "parent",
            "report_assurance_type": "unaudited",
            "currency": "VND",
            "unit": "million_vnd",
            "language": "vi",
        },
        "statement_tables": [
            {
                "table_type": "balance_sheet",
                "period_basis": "balance_sheet_date",
                "title": "Credit institution balance sheet",
                "rows": [
                    {"standard_account": account, "account_group": "receivables_credit" if account.startswith("loan") or account == "loans_to_customers" else "expense_liability", "value": value}
                    for account, value in values.items()
                ],
            },
            {
                "table_type": "income_statement",
                "period_basis": "year_to_date",
                "title": "Credit institution income statement",
                "rows": [
                    {"standard_account": "profit_after_tax", "account_group": "revenue_income", "value": 180 if target else 120}
                ],
            },
            {
                "table_type": "cash_flow_statement",
                "period_basis": "year_to_date",
                "title": "Credit institution cash flow statement",
                "rows": [
                    {"standard_account": "operating_cash_flow", "account_group": "cashflow", "value": -50 if target else 80}
                ],
            },
        ],
        "notes": [],
        "variance_explanations": [],
    }
    return "VINACOUNT_REAL_SOURCE_JSON\n" + json.dumps(payload, sort_keys=True)


def _standard_corporate_source_json(ticker, period, *, target):
    values = {
        "revenue": 1800 if target else 1000,
        "trade_receivables": 950 if target else 400,
        "profit_after_tax": 220 if target else 130,
        "operating_cash_flow": -90 if target else 120,
        "total_assets": 5000 if target else 4200,
    }
    payload = {
        "metadata": {
            "company_name": f"{ticker} Company",
            "ticker": ticker,
            "period": period,
            "report_profile": "standard_corporate",
            "report_basis": "parent",
            "report_assurance_type": "unaudited",
            "currency": "VND",
            "unit": "million_vnd",
            "language": "vi",
            "business_context_tags": ["manufacturing_inventory"],
        },
        "statement_tables": [
            {
                "table_type": "balance_sheet",
                "period_basis": "quarter",
                "title": "Balance sheet",
                "rows": [
                    {"standard_account": "trade_receivables", "account_group": "receivables_credit", "value": values["trade_receivables"]},
                    {"standard_account": "total_assets", "account_group": "asset_quality", "value": values["total_assets"]},
                ],
            },
            {
                "table_type": "income_statement",
                "period_basis": "quarter",
                "title": "Income statement",
                "rows": [{"standard_account": "revenue", "account_group": "revenue_income", "value": values["revenue"]}],
            },
            {
                "table_type": "income_statement",
                "period_basis": "year_to_date",
                "title": "Income statement YTD",
                "rows": [{"standard_account": "profit_after_tax", "account_group": "revenue_income", "value": values["profit_after_tax"]}],
            },
            {
                "table_type": "cash_flow_statement",
                "period_basis": "year_to_date",
                "title": "Cash flow statement",
                "rows": [{"standard_account": "operating_cash_flow", "account_group": "cashflow", "value": values["operating_cash_flow"]}],
            },
        ],
        "notes": [],
        "variance_explanations": [],
    }
    return "VINACOUNT_REAL_SOURCE_JSON\n" + json.dumps(payload, sort_keys=True)


def _write_zip_member(path, member, text):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, text)


def _write_zip_member_bytes(path, member, payload):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, payload)


def _write_cached_artifact(
    cache_dir,
    *,
    case_id,
    role,
    source_document_id,
    source_hash,
    package_id,
    artifact_fingerprint,
    extraction_candidates=None,
):
    path = cache_dir / f"{case_id}__{role}__{source_document_id}__{source_hash[:12]}__raw_extraction_artifact.json"
    payload = {
        "artifact_id": f"RAW_EXT_{source_document_id}",
        "filing_package_id": package_id,
        "source_document_id": source_document_id,
        "source_document_fingerprint": artifact_fingerprint,
        "extraction_method": "test_cached_raw_extraction",
        "extraction_version": "v1",
        "raw_html": "<p>raw OCR must not become detector evidence</p>",
        "raw_tables": [],
        "diagnostics": [],
        "parser_warnings": [],
        "provider_metadata": {"cache_record_id": "hidden-cache-id"},
        "extraction_candidates": extraction_candidates
        or {
            "metadata": {
                "company_name": "BIC Company",
                "ticker": "BIC",
                "period": "2026-Q1",
                "report_profile": "insurance",
                "insurance_subprofile": "non_life",
                "report_basis": "parent",
                "business_context_tags": ["insurance_non_life"],
                "report_assurance_type": "unaudited",
                "currency": "VND",
                "unit": "vnd",
                "language": "vi",
                "report_period_type": "quarterly",
            },
            "sections": [],
            "statement_tables": [],
            "notes": [],
            "variance_explanations": [],
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    unittest.main()
