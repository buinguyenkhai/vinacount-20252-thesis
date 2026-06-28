import unittest

from vinacount.report_model import CompanyReportSet, ReportMemory
from research.dataset_eligibility import validate_real_extracted_packet_dataset_eligibility


class Wave4DatasetEligibilityTest(unittest.TestCase):
    def test_positive_vietnamese_canonical_target_prior_pair_is_dataset_eligible(self):
        report_set = _report_set()
        packet = _detector_packet()

        result = validate_real_extracted_packet_dataset_eligibility(report_set, [packet])

        self.assertEqual(result.status, "passed")
        self.assertEqual(len(result.records), 1)
        self.assertEqual(result.records[0]["record_type"], "dataset_eligible")
        self.assertEqual(result.records[0]["packet_id"], "PACKET_VCF_2025_Q3_001")

    def test_unresolved_amendment_context_blocks_dataset_eligibility_until_corrected_values_pass(self):
        target = _report_memory("GAS_2025_Q1", "2025-Q1", ticker="GAS")
        target["metadata"]["amendment_context_attachments"] = [
            {
                "source_document_id": "GAS_Q1_2025_ADJUSTED_NOTES",
                "role": "adjusted_notes",
                "affects_operative_values": True,
                "corrected_value_resolution_status": "unresolved",
            }
        ]
        report_set = _report_set(target=target, prior=_report_memory("GAS_2024_Q1", "2024-Q1", ticker="GAS"))

        result = validate_real_extracted_packet_dataset_eligibility(report_set, [_detector_packet("GAS_2025_Q1")])

        self.assertEqual(result.status, "failed")
        self.assertIn("amendment_context_corrected_value_resolution_required", _reason_codes(result))

    def test_unresolved_correction_lineage_blocks_dataset_eligibility(self):
        target = _report_memory("SSI_2025_Q3", "2025-Q3", ticker="SSI")
        target["metadata"]["amendment_context_attachments"] = [
            {
                "source_document_id": "SSI_Q3_2025_CORRECTION_PAGES",
                "role": "correction_pages",
                "affects_operative_values": True,
                "corrected_value_resolution_status": "conflicting_lineage",
            }
        ]
        report_set = _report_set(target=target, prior=_report_memory("SSI_2024_Q3", "2024-Q3", ticker="SSI"))

        result = validate_real_extracted_packet_dataset_eligibility(report_set, [_detector_packet("SSI_2025_Q3")])

        self.assertEqual(result.status, "failed")
        self.assertIn("amendment_context_corrected_value_resolution_required", _reason_codes(result))

    def test_english_duplicate_is_excluded_when_vietnamese_candidate_exists(self):
        target = _report_memory("VHM_2024_Q4", "2024-Q4", ticker="VHM")
        target["metadata"]["language"] = "en"
        target["metadata"]["source_quality_details"] = {
            "vietnamese_candidate_available": True,
            "source_role": "english_duplicate",
        }
        report_set = _report_set(target=target, prior=_report_memory("VHM_2023_Q4", "2023-Q4", ticker="VHM"))

        result = validate_real_extracted_packet_dataset_eligibility(report_set, [_detector_packet("VHM_2024_Q4")])

        self.assertEqual(result.status, "failed")
        self.assertIn("english_duplicate_excluded", _reason_codes(result))

    def test_english_searchable_candidate_is_excluded_when_vietnamese_searchable_candidate_passes(self):
        target = _report_memory("VPB_2026_Q1", "2026-Q1", ticker="VPB")
        target["metadata"]["language"] = "en"
        target["metadata"]["source_quality_details"] = {
            "source_role": "english_duplicate",
            "candidate_variant": "searchable_filing_version",
            "vietnamese_candidate_available": True,
            "vietnamese_candidate_identity_check_status": "confirmed",
        }
        report_set = _report_set(target=target, prior=_report_memory("VPB_2025_Q1", "2025-Q1", ticker="VPB"))

        result = validate_real_extracted_packet_dataset_eligibility(report_set, [_detector_packet("VPB_2026_Q1")])

        self.assertEqual(result.status, "failed")
        self.assertIn("english_duplicate_excluded", _reason_codes(result))

    def test_searchable_source_requires_confirmed_identity(self):
        target = _report_memory("ACB_2026_Q1", "2026-Q1", ticker="ACB")
        target["metadata"]["source_quality_details"] = {
            "source_role": "searchable_filing_version",
            "identity_check_status": "unresolved",
        }
        report_set = _report_set(target=target, prior=_report_memory("ACB_2025_Q1", "2025-Q1", ticker="ACB"))

        unresolved = validate_real_extracted_packet_dataset_eligibility(report_set, [_detector_packet("ACB_2026_Q1")])
        target["metadata"]["source_quality_details"]["identity_check_status"] = "confirmed"
        confirmed = validate_real_extracted_packet_dataset_eligibility(report_set, [_detector_packet("ACB_2026_Q1")])

        self.assertIn("unresolved_tra_cuu_identity", _reason_codes(unresolved))
        self.assertEqual(confirmed.status, "passed")

    def test_reviewed_supersession_must_be_resolved_before_dataset_eligibility(self):
        target = _report_memory("BVH_2025_Q1", "2025-Q1", ticker="BVH")
        target["metadata"]["source_quality_details"] = {"reviewed_supersession_status": "unresolved"}
        report_set = _report_set(target=target, prior=_report_memory("BVH_2024_Q1", "2024-Q1", ticker="BVH"))

        result = validate_real_extracted_packet_dataset_eligibility(report_set, [_detector_packet("BVH_2025_Q1")])

        self.assertEqual(result.status, "failed")
        self.assertIn("unresolved_reviewed_supersession", _reason_codes(result))

    def test_invalid_ids_raw_leakage_and_source_quality_states_are_rejected_with_explicit_reasons(self):
        report_set = _report_set()
        packet = _detector_packet()
        packet["relevant_table_rows"][0]["values"]["2025-Q3"]["cell_id"] = "MISSING_CELL"
        packet["relevant_notes"].append({"note_id": "MISSING_NOTE", "note_type": "receivables", "title": "n", "text": "short"})
        packet["raw_coordinates"] = [{"page": 1, "x": 10}]
        report_set.target.raw["metadata"]["source_confirmation_status"] = "hitl_needed"
        report_set.prior_year.raw["metadata"]["source_quality_status"] = "recoverable_source_quality_error"

        result = validate_real_extracted_packet_dataset_eligibility(report_set, [packet])

        self.assertEqual(result.status, "failed")
        self.assertIn("invalid_evidence_ids", _reason_codes(result))
        self.assertIn("prohibited_metadata_leakage", _reason_codes(result))
        self.assertIn("source_confirmation_required", _reason_codes(result))
        self.assertIn("recoverable_source_quality_error", _reason_codes(result))

    def test_detector_visible_sampling_correction_and_zip_audit_metadata_are_rejected_as_leakage(self):
        report_set = _report_set()
        packet = _detector_packet()
        packet["metadata"]["sampling_pool"] = "ordinary_filing_evaluation_pool"
        packet["metadata"]["correction_amendment_provenance_type"] = "directed_correction_package"
        packet["metadata"]["ordinary_provenance"] = "ordinary control sampling metadata"
        packet["metadata"]["zip_container_sha256"] = "sha256-container"
        packet["metadata"]["selected_member_sha256"] = "sha256-member"

        result = validate_real_extracted_packet_dataset_eligibility(report_set, [packet])

        self.assertEqual(result.status, "failed")
        self.assertIn("prohibited_metadata_leakage", _reason_codes(result))


def _reason_codes(result):
    return {record["reason_code"] for record in result.records}


def _report_set(target=None, prior=None):
    return CompanyReportSet(
        case_id="dataset_case",
        target=ReportMemory(raw=target or _report_memory("VCF_2025_Q3", "2025-Q3")),
        prior_year=ReportMemory(raw=prior or _report_memory("VCF_2024_Q3", "2024-Q3")),
    )


def _report_memory(report_id, period, ticker="VCF"):
    suffix = period.replace("-", "_")
    source_document_id = f"DOC_{ticker}_{suffix}_FS"
    cell_id = f"CELL_REVENUE_{suffix}"
    return {
        "report_id": report_id,
        "metadata": {
            "company_name": f"{ticker} Company",
            "ticker": ticker,
            "period": period,
            "report_period_type": "quarterly",
            "report_profile": "standard_corporate",
            "report_basis": "consolidated",
            "business_context_tags": ["manufacturing_inventory"],
            "report_assurance_type": "unaudited",
            "currency": "VND",
            "unit": "million_vnd",
            "filing_status": "original",
            "canonical_source_document_id": source_document_id,
            "source_file": f"fixtures/{report_id}.pdf",
            "language": "vi",
            "source_confirmation_status": "confirmed",
            "canonical_source_certainty": "resolved",
        },
        "sections": [],
        "tables": [
            {
                "table_id": f"TBL_BS_{suffix}",
                "table_type": "balance_sheet",
                "rows": [
                    {
                        "row_id": f"ROW_ASSETS_{suffix}",
                        "standard_account": "total_assets",
                        "cells": [{"cell_id": f"CELL_ASSETS_{suffix}", "period": period, "value": 500, "unit": "million_vnd"}],
                    }
                ],
            },
            {
                "table_id": f"TBL_REV_{suffix}",
                "table_type": "income_statement",
                "rows": [
                    {
                        "row_id": f"ROW_REVENUE_{suffix}",
                        "standard_account": "revenue",
                        "cells": [{"cell_id": cell_id, "period": period, "value": 100, "unit": "million_vnd"}],
                    }
                ],
            },
            {
                "table_id": f"TBL_CF_{suffix}",
                "table_type": "cash_flow_statement",
                "rows": [
                    {
                        "row_id": f"ROW_CFO_{suffix}",
                        "standard_account": "operating_cash_flow",
                        "cells": [{"cell_id": f"CELL_CFO_{suffix}", "period": period, "value": 80, "unit": "million_vnd"}],
                    }
                ],
            },
        ],
        "notes": [{"note_id": f"NOTE_REV_{suffix}", "text": "Doanh thu tang trong ky."}],
        "variance_explanations": [{"span_id": f"VAR_REV_{suffix}", "text": "Doanh thu tang do san luong."}],
        "cell_index": {
            f"CELL_ASSETS_{suffix}": {"table_id": f"TBL_BS_{suffix}", "row_id": f"ROW_ASSETS_{suffix}", "value": 500},
            cell_id: {"table_id": f"TBL_REV_{suffix}", "row_id": f"ROW_REVENUE_{suffix}", "value": 100},
            f"CELL_CFO_{suffix}": {"table_id": f"TBL_CF_{suffix}", "row_id": f"ROW_CFO_{suffix}", "value": 80},
        },
    }


def _detector_packet(report_id="VCF_2025_Q3"):
    suffix = "_".join(report_id.split("_")[-2:])
    period = suffix.replace("_", "-")
    return {
        "packet_id": f"PACKET_{report_id}_001",
        "candidate_id": f"CAND_{report_id}_001",
        "report_id": report_id,
        "task": {
            "risk_category": "revenue_income_recognition_risk",
            "question": "Does the evidence support the risk signal?",
            "expected_output": "Return a structured DetectorAssessment.",
        },
        "metadata": {"report_profile": "standard_corporate", "period": period},
        "candidate_summary": {
            "reason_for_candidate": "Revenue growth requires review.",
            "supporting_signal_ids": ["revenue_growth_high"],
            "priority": "high",
            "review_mode": "required",
        },
        "relevant_table_rows": [
            {
                "table_id": f"TBL_REV_{suffix}",
                "row_id": f"ROW_REVENUE_{suffix}",
                "values": {period: {"cell_id": f"CELL_REVENUE_{suffix}", "value": 100}},
            }
        ],
        "relevant_notes": [{"note_id": f"NOTE_REV_{suffix}", "note_type": "revenue", "title": "Revenue", "text": "Doanh thu tang trong ky."}],
        "relevant_variance_explanations": [{"span_id": f"VAR_REV_{suffix}", "title": "Revenue", "text": "Doanh thu tang do san luong."}],
        "tool_findings": [{"tool_result_id": "TOOL_REV", "summary": "Revenue increased.", "evidence_cell_ids": [f"CELL_REVENUE_{suffix}"], "evidence_note_ids": []}],
        "rules": [{"rule_id": "RULE_REV", "risk_category": "revenue_income_recognition_risk", "description": "Review revenue growth."}],
        "constraints": {"allowed_decisions": ["supported", "weakly_supported", "not_supported", "insufficient_evidence"], "evidence_must_reference_provided_ids": True, "do_not_claim_fraud": True, "max_rationale_sentences": 3},
    }


if __name__ == "__main__":
    unittest.main()
