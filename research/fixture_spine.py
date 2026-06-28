from __future__ import annotations

import argparse
import json
import os
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from typing import Any

from vinacount.detector_contract import (
    DETECTOR_PACKET_CAPS,
    sentence_count,
    validate_detector_assessment as _contract_validate_detector_assessment,
    validate_detector_packet as _contract_validate_detector_packet,
    visible_packet_evidence_ids as _contract_visible_packet_evidence_ids,
)
from vinacount.final_report import (
    build_final_report,
    render_final_report_markdown,
    validate_final_report,
    write_final_report,
)
from vinacount.report_model import (
    CompanyReportSet,
    RealExtractionQualityGateResult,
    ReportMemory,
    validate_company_report_set,
    validate_real_extraction_quality_gate,
    validate_report_memory,
)


API_DETECTOR_PROVIDER = "openrouter"
API_DETECTOR_MODEL = "deepseek/deepseek-v4-flash"
API_DETECTOR_ENABLE_ENV = "VINACOUNT_API_DETECTOR_ENABLED"
API_DETECTOR_KEY_ENV = "OPENROUTER_API_KEY"
API_DETECTOR_TIMEOUT_ENV = "VINACOUNT_API_DETECTOR_TIMEOUT_SECONDS"
API_DETECTOR_MAX_RETRIES_ENV = "VINACOUNT_API_DETECTOR_MAX_RETRIES"
API_DETECTOR_DEBUG_RAW_ENV = "VINACOUNT_API_DETECTOR_DEBUG_RAW"
API_DETECTOR_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass(frozen=True)
class ApiDetectorConfig:
    provider: str
    model: str
    enabled: bool
    api_key_present: bool
    timeout_seconds: float
    max_retries: int
    debug_raw_logging: bool
    status: str
    reason_code: str

    def audit_metadata(self, network_call_attempted: bool = False) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "status": self.status,
            "reason_code": self.reason_code,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "debug_raw_logging": self.debug_raw_logging,
            "network_call_attempted": network_call_attempted,
            "credentials_logged": False,
            "config_provisional_wave": "wave1_revisit_before_detector",
        }


@dataclass(frozen=True)
class FixtureSpineRunResult:
    status: str
    loaded_report_sets: list[str]
    report_sets: list[CompanyReportSet]
    audit_log_path: Path
    tool_gating_records: list[dict[str, Any]]
    tool_findings: list[dict[str, Any]]
    candidate_risks: list[dict[str, Any]]
    detector_packets: list[dict[str, Any]]
    detector_packet_audit_records: list[dict[str, Any]]
    detector_packet_log_path: Path
    detector_assessments: list[dict[str, Any]]
    detector_assessment_audit_records: list[dict[str, Any]]
    detector_assessment_log_path: Path
    final_reports: list[dict[str, Any]]
    final_report_paths: list[dict[str, Path]]
    final_report_audit_records: list[dict[str, Any]]
    final_report_log_path: Path
    exit_demo_manifest_path: Path
    thesis_artifact_paths: dict[str, Path]


def run_fixture_spine(
    output_dir: Path | str,
    fixture_cases: list[dict[str, Any]] | None = None,
    *,
    interface: str = "research.fixture_spine.run_fixture_spine",
) -> FixtureSpineRunResult:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    audit_log = AppendOnlyAuditLog(output_path / "fixture_spine_audit.jsonl")
    detector_packet_log = AppendOnlyAuditLog(output_path / "detector_packets.jsonl")
    detector_assessment_log = AppendOnlyAuditLog(output_path / "detector_assessments.jsonl")
    final_report_log = AppendOnlyAuditLog(output_path / "final_reports.jsonl")

    audit_log.write("run_started", {"interface": interface})

    try:
        fixture_cases = fixture_cases if fixture_cases is not None else _load_fixture_cases()
        audit_log.write(
            "fixtures_loaded",
            {
                "case_ids": [fixture_case["case_id"] for fixture_case in fixture_cases],
                "report_set_count": len(fixture_cases),
            },
        )

        report_sets: list[CompanyReportSet] = []
        tool_gating_records: list[dict[str, Any]] = []
        tool_findings: list[dict[str, Any]] = []
        candidate_risks: list[dict[str, Any]] = []
        detector_packets: list[dict[str, Any]] = []
        detector_packet_audit_records: list[dict[str, Any]] = []
        detector_assessments: list[dict[str, Any]] = []
        detector_assessment_audit_records: list[dict[str, Any]] = []
        final_reports: list[dict[str, Any]] = []
        final_report_paths: list[dict[str, Path]] = []
        final_report_audit_records: list[dict[str, Any]] = []
        for fixture_case in fixture_cases:
            target = validate_report_memory(fixture_case["target_report_memory"])
            audit_log.write(
                "report_memory_validated",
                {"case_id": fixture_case["case_id"], "role": "target", "report_id": target.report_id},
            )

            prior_year = validate_report_memory(fixture_case["prior_year_report_memory"])
            audit_log.write(
                "report_memory_validated",
                {"case_id": fixture_case["case_id"], "role": "prior_year", "report_id": prior_year.report_id},
            )

            report_set = validate_company_report_set(fixture_case["case_id"], target, prior_year)
            report_sets.append(report_set)
            audit_log.write(
                "company_report_set_constructed",
                {
                    "case_id": report_set.case_id,
                    "target_report_id": target.report_id,
                    "prior_year_report_id": prior_year.report_id,
                },
            )

            quality_gate_result = validate_real_extraction_quality_gate(report_set)
            audit_log.write(
                "real_extraction_quality_gate_completed",
                {
                    "case_id": report_set.case_id,
                    "status": quality_gate_result.status,
                    "allows_tool_availability_gating": quality_gate_result.allows_tool_availability_gating,
                    "records": quality_gate_result.records,
                },
            )
            if not quality_gate_result.allows_tool_availability_gating:
                continue

            case_gating_records = gate_wave1_tools(report_set)
            tool_gating_records.extend(case_gating_records)
            audit_log.write(
                "tool_availability_gating_completed",
                {
                    "case_id": report_set.case_id,
                    "enabled_tools": [
                        record["tool_name"] for record in case_gating_records if record["status"] == "enabled"
                    ],
                    "disabled_tools": [
                        record["tool_name"] for record in case_gating_records if record["status"] != "enabled"
                    ],
                    "records": case_gating_records,
                },
            )

            case_tool_findings = run_wave1_tools(report_set, case_gating_records)
            tool_findings.extend(case_tool_findings)
            audit_log.write(
                "tool_execution_completed",
                {
                    "case_id": report_set.case_id,
                    "tool_result_ids": [finding["tool_result_id"] for finding in case_tool_findings],
                },
            )

            case_candidates = generate_wave1_candidates(report_set, case_tool_findings)
            candidate_risks.extend(case_candidates)
            audit_log.write(
                "candidate_generation_completed",
                {
                    "case_id": report_set.case_id,
                    "candidate_ids": [candidate["candidate_id"] for candidate in case_candidates],
                },
            )

            case_detector_packets = build_detector_packets(
                report_set,
                case_candidates,
                case_tool_findings,
                detector_packet_audit_records,
            )
            detector_packets.extend(case_detector_packets)
            for packet in case_detector_packets:
                detector_packet_log.write(
                    "detector_packet",
                    {
                        "packet_id": packet["packet_id"],
                        "candidate_id": packet["candidate_id"],
                        "report_id": packet["report_id"],
                        "risk_category": packet["task"]["risk_category"],
                        "artifact": packet,
                    },
                )
            audit_log.write(
                "detector_packet_build_completed",
                {
                    "case_id": report_set.case_id,
                    "packets": [
                        {
                            "packet_id": packet["packet_id"],
                            "candidate_id": packet["candidate_id"],
                            "risk_category": packet["task"]["risk_category"],
                        }
                        for packet in case_detector_packets
                    ],
                    "omitted_evidence_records": [
                        record
                        for record in detector_packet_audit_records
                        if record.get("packet_id") in {packet["packet_id"] for packet in case_detector_packets}
                    ],
                    "discarded_candidate_records": [
                        record
                        for record in detector_packet_audit_records
                        if record.get("reason_code") == "too_broad_for_detector_packet"
                        and record["report_id"] == report_set.target.report_id
                    ],
                },
            )

            case_detector_assessments = assess_detector_packets(
                case_detector_packets,
                detector_assessment_audit_records,
            )
            detector_assessments.extend(case_detector_assessments)
            for assessment in case_detector_assessments:
                detector_assessment_log.write(
                    "detector_assessment",
                    {
                        "assessment_id": assessment["assessment_id"],
                        "packet_id": assessment["packet_id"],
                        "candidate_id": assessment["candidate_id"],
                        "support_level": assessment["support_level"],
                        "artifact": assessment,
                    },
                )
            audit_log.write(
                "detector_assessment_completed",
                {
                    "case_id": report_set.case_id,
                    "assessments": [
                        {
                            "assessment_id": assessment["assessment_id"],
                            "packet_id": assessment["packet_id"],
                            "support_level": assessment["support_level"],
                        }
                        for assessment in case_detector_assessments
                    ],
                    "call_metadata": [
                        record
                        for record in detector_assessment_audit_records
                        if record.get("packet_id")
                        in {packet["packet_id"] for packet in case_detector_packets}
                    ],
                },
            )

            final_report = build_final_report(
                report_set,
                case_candidates,
                case_tool_findings,
                case_detector_assessments,
                case_gating_records,
                final_report_audit_records,
            )
            report_paths = write_final_report(output_path, final_report)
            final_reports.append(final_report)
            final_report_paths.append(report_paths)
            final_report_log.write(
                "final_report",
                {
                    "report_id": final_report["report_id"],
                    "target_report_id": report_set.target.report_id,
                    "json_path": str(report_paths["json"]),
                    "markdown_path": str(report_paths["markdown"]),
                    "artifact": final_report,
                },
            )
            audit_log.write(
                "final_report_generation_completed",
                {
                    "case_id": report_set.case_id,
                    "report_id": final_report["report_id"],
                    "grouped_finding_ids": [
                        finding["finding_id"] for finding in final_report["grouped_findings"]
                    ],
                    "weak_signal_assessment_ids": [
                        item["assessment_id"] for item in final_report["weak_or_limited_signals"]
                    ],
                    "reviewed_candidate_count": len(final_report["reviewed_candidate_audit"]),
                },
            )

        persisted_artifact_paths = write_wave1_exit_demo_artifacts(
            output_path=output_path,
            report_sets=report_sets,
            tool_gating_records=tool_gating_records,
            tool_findings=tool_findings,
            candidate_risks=candidate_risks,
            detector_packets=detector_packets,
            detector_assessments=detector_assessments,
            final_reports=final_reports,
        )
        thesis_artifact_paths = write_wave1_thesis_artifacts(output_path)
        exit_demo_manifest_path = write_wave1_exit_demo_manifest(
            output_path=output_path,
            report_sets=report_sets,
            persisted_artifact_paths=persisted_artifact_paths,
            final_report_paths=final_report_paths,
            log_paths={
                "fixture_spine_audit": audit_log.path,
                "detector_packets": detector_packet_log.path,
                "detector_assessments": detector_assessment_log.path,
                "final_reports": final_report_log.path,
            },
            thesis_artifact_paths=thesis_artifact_paths,
        )
        audit_log.write(
            "wave1_exit_demo_artifacts_written",
            {
                "manifest_path": str(exit_demo_manifest_path),
                "artifact_names": sorted(
                    set(persisted_artifact_paths) | set(thesis_artifact_paths)
                ),
            },
        )
        audit_log.write("run_completed", {"status": "completed", "report_set_count": len(report_sets)})
        return FixtureSpineRunResult(
            status="completed",
            loaded_report_sets=[report_set.case_id for report_set in report_sets],
            report_sets=report_sets,
            audit_log_path=audit_log.path,
            tool_gating_records=tool_gating_records,
            tool_findings=tool_findings,
            candidate_risks=candidate_risks,
            detector_packets=detector_packets,
            detector_packet_audit_records=detector_packet_audit_records,
            detector_packet_log_path=detector_packet_log.path,
            detector_assessments=detector_assessments,
            detector_assessment_audit_records=detector_assessment_audit_records,
            detector_assessment_log_path=detector_assessment_log.path,
            final_reports=final_reports,
            final_report_paths=final_report_paths,
            final_report_audit_records=final_report_audit_records,
            final_report_log_path=final_report_log.path,
            exit_demo_manifest_path=exit_demo_manifest_path,
            thesis_artifact_paths=thesis_artifact_paths,
        )
    except Exception as error:
        audit_log.write("run_failed", {"status": "recoverable_failure", "error": str(error)})
        raise


def gate_wave1_tools(report_set: CompanyReportSet) -> list[dict[str, Any]]:
    tool_requirements = {
        "standard_corporate_revenue_growth_tool": {
            "required_accounts": ["revenue"],
            "period_basis": "quarter",
        },
        "standard_corporate_receivables_vs_revenue_growth_tool": {
            "required_accounts": ["revenue", "trade_receivables"],
            "period_basis": "quarter",
        },
        "standard_corporate_earnings_cashflow_mismatch_tool": {
            "required_accounts": ["profit_after_tax", "operating_cash_flow"],
            "period_basis": "year_to_date",
        },
        "standard_corporate_disclosure_consistency_tool": {
            "required_accounts": [],
            "required_disclosures": ["notes", "variance_explanations"],
            "period_basis": "quarter",
        },
        "standard_corporate_variance_explanation_quality_tool": {
            "required_accounts": ["profit_after_tax"],
            "required_disclosures": ["variance_explanations"],
            "period_basis": "year_to_date",
        },
        "standard_corporate_related_party_exposure_tool": {
            "required_accounts": [],
            "required_disclosures": ["related_party_notes"],
            "period_basis": "not_applicable",
        },
        "standard_corporate_accounting_policy_change_tool": {
            "required_accounts": [],
            "required_disclosures": ["accounting_policy_change_notes"],
            "period_basis": "not_applicable",
        },
        "credit_institution_loan_quality_tool": {
            "required_accounts": [
                "loan_group_1",
                "loan_group_2",
                "loan_group_3",
                "loan_group_4",
                "loan_group_5",
            ],
            "period_basis": "balance_sheet_date",
        },
        "credit_institution_provision_movement_tool": {
            "required_accounts": [
                "loans_to_customers",
                "general_provision",
                "specific_provision",
            ],
            "period_basis": "balance_sheet_date",
        },
        "ctck_margin_book_quality_tool": {
            "required_accounts": ["margin_lending", "margin_impairment"],
            "period_basis": "balance_sheet_date",
        },
        "ctck_trading_book_valuation_bridge_tool": {
            "required_accounts": ["fvtpl_assets", "afs_assets", "htm_assets"],
            "period_basis": "balance_sheet_date",
        },
        "ctck_earnings_cash_bridge_tool": {
            "required_accounts": ["profit_after_tax", "operating_cash_flow", "fvtpl_unrealized_gain"],
            "period_basis": "year_to_date",
        },
        "ctck_disclosure_consistency_tool": {
            "required_accounts": ["fvtpl_unrealized_gain"],
            "required_disclosures": ["notes"],
            "period_basis": "year_to_date",
        },
        "insurance_premium_receivable_coherence_tool": {
            "required_accounts": ["gross_written_premium", "premium_receivables"],
            "period_basis": "year_to_date",
        },
        "insurance_reserve_movement_tool": {
            "required_accounts": ["gross_written_premium", "claims_expense", "insurance_reserves"],
            "period_basis": "year_to_date",
        },
        "insurance_reinsurance_balance_tool": {
            "required_accounts": ["reinsurance_recoverables", "reinsurance_payables", "operating_cash_flow"],
            "period_basis": "year_to_date",
        },
    }
    records = []
    for tool_name, requirements in tool_requirements.items():
        required_accounts = requirements["required_accounts"]
        period_basis = requirements["period_basis"]
        missing_accounts = [
            account
            for account in required_accounts
            if _find_tool_account_cell(report_set.target, tool_name, account, period_basis) is None
            or _find_tool_account_cell(report_set.prior_year, tool_name, account, period_basis) is None
        ]
        disclosure_status = _disclosure_gating_status(report_set, tool_name)
        tool_profile = _tool_report_profile(tool_name)
        if report_set.target.metadata["report_profile"] != tool_profile:
            status = "disabled_not_applicable"
            reason_code = "disabled_not_applicable"
            reason = f"Tool applies only to {tool_profile} report_profile."
        elif missing_accounts:
            status = "disabled_missing_context"
            reason_code = "missing_required_account_evidence"
            reason = f"Missing required account evidence: {', '.join(missing_accounts)}."
        elif disclosure_status is not None:
            status = disclosure_status["status"]
            reason_code = disclosure_status["reason_code"]
            reason = disclosure_status["reason"]
        else:
            status = "enabled"
            reason_code = "required_evidence_available"
            reason = "Required standard_corporate account evidence is available."

        records.append(
            {
                "case_id": report_set.case_id,
                "tool_name": tool_name,
                "status": status,
                "reason_code": reason_code,
                "reason": reason,
                "analysis_scope": "company_report_set",
                "period_basis": period_basis,
                "required_accounts": required_accounts,
                "required_disclosures": requirements.get("required_disclosures", []),
                "metadata": {
                    "report_profile": report_set.target.metadata["report_profile"],
                    "insurance_subprofile": report_set.target.metadata.get("insurance_subprofile"),
                },
            }
        )
    return records


def _tool_report_profile(tool_name: str) -> str:
    if tool_name.startswith("credit_institution_"):
        return "credit_institution"
    if tool_name.startswith("ctck_"):
        return "securities"
    if tool_name.startswith("insurance_"):
        return "insurance"
    return "standard_corporate"


def run_wave1_tools(
    report_set: CompanyReportSet,
    gating_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    enabled_tools = {record["tool_name"] for record in gating_records if record["status"] == "enabled"}
    findings = []
    if "standard_corporate_revenue_growth_tool" in enabled_tools:
        findings.append(_build_growth_finding(report_set, "revenue"))
    if "standard_corporate_receivables_vs_revenue_growth_tool" in enabled_tools:
        findings.append(_build_receivables_vs_revenue_finding(report_set))
    if "standard_corporate_earnings_cashflow_mismatch_tool" in enabled_tools:
        findings.append(_build_earnings_cashflow_mismatch_finding(report_set))
    if "standard_corporate_disclosure_consistency_tool" in enabled_tools:
        findings.append(_build_disclosure_consistency_finding(report_set))
    if "standard_corporate_variance_explanation_quality_tool" in enabled_tools:
        findings.append(_build_variance_explanation_quality_finding(report_set))
    if "credit_institution_loan_quality_tool" in enabled_tools:
        findings.append(_build_credit_institution_loan_quality_finding(report_set))
    if "credit_institution_provision_movement_tool" in enabled_tools:
        findings.append(_build_credit_institution_provision_movement_finding(report_set))
    if "ctck_margin_book_quality_tool" in enabled_tools:
        findings.append(_build_ctck_margin_book_quality_finding(report_set))
    if "ctck_trading_book_valuation_bridge_tool" in enabled_tools:
        findings.append(_build_ctck_trading_book_valuation_bridge_finding(report_set))
    if "ctck_earnings_cash_bridge_tool" in enabled_tools:
        findings.append(_build_ctck_earnings_cash_bridge_finding(report_set))
    if "ctck_disclosure_consistency_tool" in enabled_tools:
        findings.append(_build_ctck_disclosure_consistency_finding(report_set))
    if "insurance_premium_receivable_coherence_tool" in enabled_tools:
        findings.append(_build_insurance_premium_receivable_coherence_finding(report_set))
    if "insurance_reserve_movement_tool" in enabled_tools:
        findings.append(_build_insurance_reserve_movement_finding(report_set))
    if "insurance_reinsurance_balance_tool" in enabled_tools:
        findings.append(_build_insurance_reinsurance_balance_finding(report_set))
    return findings


def generate_wave1_candidates(
    report_set: CompanyReportSet,
    tool_findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    flagged = {finding["signal_id"]: finding for finding in tool_findings if finding["flag"]}
    candidates = []
    if not {"revenue_growth_high", "receivables_growth_outpaces_revenue"} <= flagged.keys():
        pass
    else:
        linked_findings = [flagged["revenue_growth_high"], flagged["receivables_growth_outpaces_revenue"]]
        required_evidence_refs = _candidate_evidence_refs(linked_findings)

        candidates.append(
            {
                "candidate_id": f"CAND_{report_set.target.report_id}_REV_REC_001",
                "report_id": report_set.target.report_id,
                "risk_category": "revenue_income_recognition_risk",
                "candidate_status": "pending_detector_review",
                "generation_source": "rule_generated",
                "reason_for_candidate": (
                    "Revenue increased significantly while receivables grew faster than revenue, "
                    "requiring review of revenue quality risk."
                ),
                "priority": "high",
                "review_mode": "required",
                "supporting_signal_ids": ["revenue_growth_high", "receivables_growth_outpaces_revenue"],
                "linked_tool_result_ids": [finding["tool_result_id"] for finding in linked_findings],
                "required_evidence_refs": required_evidence_refs,
                "required_context_queries": [
                    {
                        "query_type": "table_rows",
                        "account_group": "revenue_income",
                        "standard_account": "revenue",
                        "periods": [report_set.target.metadata["period"], report_set.prior_year.metadata["period"]],
                        "max_items": 4,
                    },
                    {
                        "query_type": "table_rows",
                        "account_group": "receivables_credit",
                        "standard_account": "trade_receivables",
                        "periods": [report_set.target.metadata["period"], report_set.prior_year.metadata["period"]],
                        "max_items": 4,
                    },
                ],
                "applicability": {
                    "report_profile": "standard_corporate",
                    "is_applicable": True,
                    "reason": "Revenue and receivables checks are applicable to standard corporate companies.",
                },
            }
        )

    if "positive_profit_negative_cfo" in flagged:
        mismatch = flagged["positive_profit_negative_cfo"]
        candidates.append(
            {
                "candidate_id": f"CAND_{report_set.target.report_id}_EARN_CASH_001",
                "report_id": report_set.target.report_id,
                "risk_category": "earnings_cashflow_mismatch",
                "candidate_status": "pending_detector_review",
                "generation_source": "rule_generated",
                "reason_for_candidate": (
                    "The company reported positive profit while operating cash flow was negative, "
                    "requiring review of earnings quality and cash realization."
                ),
                "priority": "high",
                "review_mode": "required",
                "supporting_signal_ids": ["positive_profit_negative_cfo"],
                "linked_tool_result_ids": [mismatch["tool_result_id"]],
                "required_evidence_refs": _candidate_evidence_refs([mismatch]),
                "required_context_queries": [
                    {
                        "query_type": "table_rows",
                        "account_group": "cashflow",
                        "standard_account": "operating_cash_flow",
                        "periods": [report_set.target.metadata["period"], report_set.prior_year.metadata["period"]],
                        "max_items": 8,
                    },
                    {
                        "query_type": "table_rows",
                        "account_group": "revenue_income",
                        "standard_account": "profit_after_tax",
                        "periods": [report_set.target.metadata["period"], report_set.prior_year.metadata["period"]],
                        "max_items": 4,
                    },
                ],
                "applicability": {
                    "report_profile": "standard_corporate",
                    "is_applicable": True,
                    "reason": "Earnings-cash flow mismatch checks are applicable to standard corporate reports.",
                },
            }
        )

    if {"disclosure_narrative_tension", "variance_explanation_weak_connection"} <= flagged.keys():
        linked_findings = [
            flagged["disclosure_narrative_tension"],
            flagged["variance_explanation_weak_connection"],
        ]
        candidates.append(
            {
                "candidate_id": f"CAND_{report_set.target.report_id}_DISC_VAR_001",
                "report_id": report_set.target.report_id,
                "risk_category": "disclosure_inconsistency_or_obfuscation",
                "candidate_status": "pending_detector_review",
                "generation_source": "rule_generated",
                "reason_for_candidate": (
                    "Disclosure evidence shows narrative tension around receivables and cash collection, "
                    "while the variance explanation does not clearly connect the stated driver to the changed metric."
                ),
                "priority": "medium",
                "review_mode": "required",
                "supporting_signal_ids": [
                    "disclosure_narrative_tension",
                    "variance_explanation_weak_connection",
                ],
                "linked_tool_result_ids": [finding["tool_result_id"] for finding in linked_findings],
                "required_evidence_refs": _candidate_evidence_refs(linked_findings),
                "required_context_queries": [
                    {
                        "query_type": "notes",
                        "evidence_types": ["note_span"],
                        "periods": [report_set.target.metadata["period"]],
                        "max_items": 3,
                    },
                    {
                        "query_type": "variance_explanations",
                        "evidence_types": ["variance_explanation_span"],
                        "periods": [report_set.target.metadata["period"]],
                        "max_items": 2,
                    },
                ],
                "applicability": {
                    "report_profile": "standard_corporate",
                    "is_applicable": True,
                    "reason": "Disclosure checks are applicable when extracted notes and variance explanations are available.",
                },
            }
        )

    if "loan_quality_deterioration" in flagged:
        loan_quality = flagged["loan_quality_deterioration"]
        candidates.append(
            {
                "candidate_id": f"CAND_{report_set.target.report_id}_LOAN_QUALITY_001",
                "report_id": report_set.target.report_id,
                "risk_category": "receivables_credit_quality_risk",
                "candidate_status": "pending_detector_review",
                "generation_source": "rule_generated",
                "reason_for_candidate": (
                    "Loan-quality indicators deteriorated, requiring review of credit-quality risk."
                ),
                "priority": "high",
                "review_mode": "required",
                "supporting_signal_ids": ["loan_quality_deterioration"],
                "linked_tool_result_ids": [loan_quality["tool_result_id"]],
                "required_evidence_refs": _candidate_evidence_refs([loan_quality]),
                "required_context_queries": [
                    {
                        "query_type": "table_rows",
                        "account_group": "receivables_credit",
                        "standard_account": "loan_group",
                        "periods": [report_set.target.metadata["period"], report_set.prior_year.metadata["period"]],
                        "max_items": 10,
                    }
                ],
                "applicability": {
                    "report_profile": "credit_institution",
                    "is_applicable": True,
                    "reason": "Loan-group checks are applicable to credit institution reports with TCTD-style evidence.",
                },
            }
        )

    if "provision_growth_lags_risk_assets" in flagged:
        provision = flagged["provision_growth_lags_risk_assets"]
        candidates.append(
            {
                "candidate_id": f"CAND_{report_set.target.report_id}_PROVISION_001",
                "report_id": report_set.target.report_id,
                "risk_category": "expense_liability_understatement_risk",
                "candidate_status": "pending_detector_review",
                "generation_source": "rule_generated",
                "reason_for_candidate": (
                    "Risk assets increased faster than related provision balances, requiring review of provision movement risk."
                ),
                "priority": "high",
                "review_mode": "required",
                "supporting_signal_ids": ["provision_growth_lags_risk_assets"],
                "linked_tool_result_ids": [provision["tool_result_id"]],
                "required_evidence_refs": _candidate_evidence_refs([provision]),
                "required_context_queries": [
                    {
                        "query_type": "table_rows",
                        "account_group": "expense_liability",
                        "standard_account": "provisions",
                        "periods": [report_set.target.metadata["period"], report_set.prior_year.metadata["period"]],
                        "max_items": 6,
                    }
                ],
                "applicability": {
                    "report_profile": "credit_institution",
                    "is_applicable": True,
                    "reason": "Provision movement checks are applicable to credit institution reports with provision evidence.",
                },
            }
        )

    if "ctck_margin_book_provision_gap" in flagged:
        margin = flagged["ctck_margin_book_provision_gap"]
        candidates.append(
            {
                "candidate_id": f"CAND_{report_set.target.report_id}_MARGIN_BOOK_001",
                "report_id": report_set.target.report_id,
                "risk_category": "receivables_credit_quality_risk",
                "candidate_status": "pending_detector_review",
                "generation_source": "rule_generated",
                "reason_for_candidate": (
                    "Margin lending increased while impairment coverage declined, requiring review of CTCK credit-quality risk."
                ),
                "priority": "high",
                "review_mode": "required",
                "supporting_signal_ids": ["ctck_margin_book_provision_gap"],
                "linked_tool_result_ids": [margin["tool_result_id"]],
                "required_evidence_refs": _candidate_evidence_refs([margin]),
                "required_context_queries": [
                    {
                        "query_type": "table_rows",
                        "account_group": "receivables_credit",
                        "standard_account": "margin_lending",
                        "periods": [report_set.target.metadata["period"], report_set.prior_year.metadata["period"]],
                        "max_items": 6,
                    }
                ],
                "applicability": {
                    "report_profile": "securities",
                    "is_applicable": True,
                    "reason": "Margin-book checks are applicable to CTCK-style securities reports with margin lending evidence.",
                },
            }
        )

    if "ctck_trading_book_valuation_concentration_with_weak_disclosure" in flagged:
        valuation = flagged["ctck_trading_book_valuation_concentration_with_weak_disclosure"]
        candidates.append(
            {
                "candidate_id": f"CAND_{report_set.target.report_id}_TRADING_VAL_001",
                "report_id": report_set.target.report_id,
                "risk_category": "asset_quality_valuation_risk",
                "candidate_status": "pending_detector_review",
                "generation_source": "rule_generated",
                "reason_for_candidate": (
                    "FVTPL asset concentration and weak valuation disclosure require review of asset valuation risk."
                ),
                "priority": "high",
                "review_mode": "required",
                "supporting_signal_ids": ["ctck_trading_book_valuation_concentration_with_weak_disclosure"],
                "linked_tool_result_ids": [valuation["tool_result_id"]],
                "required_evidence_refs": _candidate_evidence_refs([valuation]),
                "required_context_queries": [
                    {
                        "query_type": "table_rows",
                        "account_group": "asset_quality",
                        "standard_account": "fvtpl_assets",
                        "periods": [report_set.target.metadata["period"], report_set.prior_year.metadata["period"]],
                        "max_items": 6,
                    },
                    {
                        "query_type": "notes",
                        "evidence_types": ["note_span"],
                        "periods": [report_set.target.metadata["period"]],
                        "max_items": 3,
                    },
                ],
                "applicability": {
                    "report_profile": "securities",
                    "is_applicable": True,
                    "reason": "Trading-book valuation checks are applicable to CTCK-style securities reports with FVTPL evidence.",
                },
            }
        )

    if "ctck_profit_supported_by_noncash_fvtpl_with_weak_cash_support" in flagged:
        cash_bridge = flagged["ctck_profit_supported_by_noncash_fvtpl_with_weak_cash_support"]
        candidates.append(
            {
                "candidate_id": f"CAND_{report_set.target.report_id}_EARN_CASH_001",
                "report_id": report_set.target.report_id,
                "risk_category": "earnings_cashflow_mismatch",
                "candidate_status": "pending_detector_review",
                "generation_source": "rule_generated",
                "reason_for_candidate": (
                    "Profit is supported by non-cash FVTPL gains while operating cash flow is weak, requiring review of cash support."
                ),
                "priority": "high",
                "review_mode": "required",
                "supporting_signal_ids": ["ctck_profit_supported_by_noncash_fvtpl_with_weak_cash_support"],
                "linked_tool_result_ids": [cash_bridge["tool_result_id"]],
                "required_evidence_refs": _candidate_evidence_refs([cash_bridge]),
                "required_context_queries": [
                    {
                        "query_type": "table_rows",
                        "account_group": "cashflow",
                        "standard_account": "operating_cash_flow",
                        "periods": [report_set.target.metadata["period"], report_set.prior_year.metadata["period"]],
                        "max_items": 6,
                    }
                ],
                "applicability": {
                    "report_profile": "securities",
                    "is_applicable": True,
                    "reason": "CTCK cash-bridge checks are applicable when profit, cash flow, and FVTPL evidence are visible.",
                },
            }
        )

    if "premium_receivables_growth_outpaces_premium_growth" in flagged:
        premium = flagged["premium_receivables_growth_outpaces_premium_growth"]
        candidates.append(
            {
                "candidate_id": f"CAND_{report_set.target.report_id}_PREM_REC_001",
                "report_id": report_set.target.report_id,
                "risk_category": "receivables_credit_quality_risk",
                "candidate_status": "pending_detector_review",
                "generation_source": "rule_generated",
                "reason_for_candidate": (
                    "Premium receivables increased faster than written premium, requiring review of receivable quality risk."
                ),
                "priority": "high",
                "review_mode": "required",
                "supporting_signal_ids": ["premium_receivables_growth_outpaces_premium_growth"],
                "linked_tool_result_ids": [premium["tool_result_id"]],
                "required_evidence_refs": _candidate_evidence_refs([premium]),
                "required_context_queries": [
                    {
                        "query_type": "table_rows",
                        "account_group": "receivables_credit",
                        "standard_account": "premium_receivables",
                        "periods": [report_set.target.metadata["period"], report_set.prior_year.metadata["period"]],
                        "max_items": 6,
                    }
                ],
                "applicability": {
                    "report_profile": "insurance",
                    "is_applicable": True,
                    "reason": "Premium receivable checks are applicable to insurance reports with premium evidence.",
                },
            }
        )

    if "claims_reserve_growth_lags_claims_and_premium_exposure" in flagged:
        reserve = flagged["claims_reserve_growth_lags_claims_and_premium_exposure"]
        candidates.append(
            {
                "candidate_id": f"CAND_{report_set.target.report_id}_RESERVE_MOVE_001",
                "report_id": report_set.target.report_id,
                "risk_category": "expense_liability_understatement_risk",
                "candidate_status": "pending_detector_review",
                "generation_source": "rule_generated",
                "reason_for_candidate": (
                    "Claims and premium exposure increased faster than the report-visible reserve movement, "
                    "requiring review of reserve-flow coherence."
                ),
                "priority": "high",
                "review_mode": "required",
                "supporting_signal_ids": ["claims_reserve_growth_lags_claims_and_premium_exposure"],
                "linked_tool_result_ids": [reserve["tool_result_id"]],
                "required_evidence_refs": _candidate_evidence_refs([reserve]),
                "required_context_queries": [
                    {
                        "query_type": "table_rows",
                        "account_group": "expense_liability",
                        "standard_account": "insurance_reserves",
                        "periods": [report_set.target.metadata["period"], report_set.prior_year.metadata["period"]],
                        "max_items": 6,
                    },
                    {
                        "query_type": "notes",
                        "evidence_types": ["note_span"],
                        "periods": [report_set.target.metadata["period"]],
                        "max_items": 2,
                    },
                ],
                "applicability": {
                    "report_profile": "insurance",
                    "is_applicable": True,
                    "reason": "Reserve movement checks are applicable to insurance reports with observable reserve evidence.",
                },
            }
        )

    if "reinsurance_recoverables_expand_with_weak_cash_support" in flagged:
        reinsurance = flagged["reinsurance_recoverables_expand_with_weak_cash_support"]
        candidates.append(
            {
                "candidate_id": f"CAND_{report_set.target.report_id}_REINS_CASH_001",
                "report_id": report_set.target.report_id,
                "risk_category": "earnings_cashflow_mismatch",
                "candidate_status": "pending_detector_review",
                "generation_source": "rule_generated",
                "reason_for_candidate": (
                    "Reinsurance recoverables expanded while operating cash flow was weak, requiring review of cash support."
                ),
                "priority": "high",
                "review_mode": "required",
                "supporting_signal_ids": ["reinsurance_recoverables_expand_with_weak_cash_support"],
                "linked_tool_result_ids": [reinsurance["tool_result_id"]],
                "required_evidence_refs": _candidate_evidence_refs([reinsurance]),
                "required_context_queries": [
                    {
                        "query_type": "table_rows",
                        "account_group": "cashflow",
                        "standard_account": "operating_cash_flow",
                        "periods": [report_set.target.metadata["period"], report_set.prior_year.metadata["period"]],
                        "max_items": 6,
                    }
                ],
                "applicability": {
                    "report_profile": "insurance",
                    "is_applicable": True,
                    "reason": "Reinsurance balance checks are applicable to insurance reports with reinsurance evidence.",
                },
            }
        )

    return candidates


def build_detector_packets(
    report_set: CompanyReportSet,
    candidates: list[dict[str, Any]],
    tool_findings: list[dict[str, Any]],
    audit_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if audit_records is None:
        audit_records = []
    tool_findings_by_id = {finding["tool_result_id"]: finding for finding in tool_findings}
    packets = []
    for index, candidate in enumerate(candidates, start=1):
        required_fit_error = _required_evidence_fit_error(candidate)
        if required_fit_error is not None:
            audit_records.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "report_id": candidate["report_id"],
                    "reason_code": "too_broad_for_detector_packet",
                    "reason": required_fit_error,
                    "candidate_status": "discarded_by_agent",
                }
            )
            continue
        linked_tool_findings = [
            tool_findings_by_id[tool_result_id]
            for tool_result_id in candidate["linked_tool_result_ids"]
        ]
        packet = {
            "packet_id": f"PACKET_{candidate['report_id']}_{index:03d}",
            "candidate_id": candidate["candidate_id"],
            "report_id": candidate["report_id"],
            "report_set_id": report_set.case_id,
            "task": {
                "risk_category": candidate["risk_category"],
                "question": _detector_task_question(candidate["risk_category"]),
                "expected_output": "Return a structured DetectorAssessment using risk-signal language only.",
            },
            "metadata": _detector_packet_metadata(report_set.target),
            "candidate_summary": {
                "reason_for_candidate": candidate["reason_for_candidate"],
                "supporting_signal_ids": candidate["supporting_signal_ids"],
                "priority": candidate["priority"],
                "review_mode": candidate["review_mode"],
            },
            "relevant_table_rows": _packet_table_rows(report_set, candidate, linked_tool_findings),
            "relevant_notes": _packet_notes(report_set, candidate),
            "relevant_variance_explanations": _packet_variance_explanations(report_set, candidate),
            "tool_findings": [_packet_tool_finding(finding) for finding in linked_tool_findings],
            "rules": _packet_rules(candidate),
            "constraints": {
                "allowed_decisions": [
                    "supported",
                    "weakly_supported",
                    "not_supported",
                    "insufficient_evidence",
                ],
                "evidence_must_reference_provided_ids": True,
                "avoid_prohibited_legal_claims": True,
                "max_rationale_sentences": 3,
            },
        }
        packet = _trim_detector_packet(packet, candidate, audit_records)
        validate_detector_packet(packet)
        packets.append(packet)
    return packets


def _required_evidence_fit_error(candidate: dict[str, Any]) -> str | None:
    field_counts = {
        "relevant_table_rows": sum(
            1 for ref in candidate["required_evidence_refs"] if ref["evidence_ref_type"] == "table_cell"
        ),
        "relevant_notes": sum(
            1 for ref in candidate["required_evidence_refs"] if ref["evidence_ref_type"] == "note_span"
        ),
        "relevant_variance_explanations": sum(
            1
            for ref in candidate["required_evidence_refs"]
            if ref["evidence_ref_type"] == "variance_explanation_span"
        ),
        "tool_findings": len(candidate["linked_tool_result_ids"]),
    }
    for field_name, count in field_counts.items():
        if count > DETECTOR_PACKET_CAPS[field_name]:
            return f"Required {field_name} count {count} exceeds hard cap {DETECTOR_PACKET_CAPS[field_name]}."
    return None


def _build_growth_finding(report_set: CompanyReportSet, account: str) -> dict[str, Any]:
    current = _require_account_cell(report_set.target, account)
    comparison = _require_account_cell(report_set.prior_year, account)
    growth_pct = _percentage_growth(current["value"], comparison["value"])
    threshold_value = 20.0
    flag = growth_pct > threshold_value
    metric_name = "revenue_growth_pct"
    signal_id = "revenue_growth_high" if flag else "revenue_growth_not_high"

    return {
        "tool_result_id": f"TOOL_{report_set.target.report_id}_REV_GROWTH_001",
        "report_id": report_set.target.report_id,
        "tool_name": "standard_corporate_revenue_growth_tool",
        "tool_version": "v1.0",
        "tool_category": "quantitative",
        "risk_category": "revenue_income_recognition_risk",
        "analysis_scope": "company_report_set",
        "period_basis": "quarter",
        "signal_id": signal_id,
        "metric": {
            "metric_name": metric_name,
            "value": growth_pct,
            "unit": "percent",
            "period_current": report_set.target.metadata["period"],
            "period_comparison": report_set.prior_year.metadata["period"],
            "direction": "increase" if growth_pct > 0 else "decrease" if growth_pct < 0 else "unchanged",
        },
        "threshold": {
            "threshold_type": "greater_than",
            "value": threshold_value,
            "unit": "percent",
            "basis": "configured_default_v1",
            "config_version": "standard_corporate_v1",
            "description": "Flag revenue growth above 20% year over year.",
        },
        "flag": flag,
        "strength": "moderate" if flag else "not_applicable",
        "finding_summary": (
            f"Revenue increased by {growth_pct:.2f}% year over year, "
            f"{'exceeding' if flag else 'not exceeding'} the configured {threshold_value:.1f}% threshold."
        ),
        "evidence_refs": [
            _cell_evidence_ref(report_set.target.report_id, current["cell_id"]),
            _cell_evidence_ref(report_set.prior_year.report_id, comparison["cell_id"]),
        ],
        "calculation": {
            "formula": "(current_value - comparison_value) / abs(comparison_value) * 100",
            "inputs": [
                {"name": "current_value", "value": current["value"], "cell_id": current["cell_id"]},
                {"name": "comparison_value", "value": comparison["value"], "cell_id": comparison["cell_id"]},
            ],
            "computed_value": growth_pct,
            "rounding": "rounded_to_2_decimal_places",
        },
        "limitations": [
            "The tool compares only the provided current and comparison periods.",
            "External context is excluded in v1.",
        ],
        "status": "completed",
    }


def validate_detector_packet(packet: dict[str, Any]) -> None:
    required_fields = {
        "packet_id",
        "candidate_id",
        "report_id",
        "task",
        "metadata",
        "candidate_summary",
        "relevant_table_rows",
        "relevant_notes",
        "relevant_variance_explanations",
        "tool_findings",
        "rules",
        "constraints",
    }
    if not isinstance(packet, dict):
        raise ValueError("DetectorPacket is invalid")
    missing_fields = required_fields - packet.keys()
    if missing_fields:
        raise ValueError(f"DetectorPacket is missing fields: {sorted(missing_fields)}")
    try:
        _contract_validate_detector_packet(packet)
    except ValueError as error:
        message = str(error)
        if "raw extraction payload" in message:
            raise ValueError("DetectorPacket must not include raw extraction payload fields") from error
        if "relevant_table_rows exceeds" in message:
            raise ValueError("DetectorPacket relevant_table_rows exceeds the hard v1 cap") from error
        if "relevant_notes exceeds" in message:
            raise ValueError("DetectorPacket relevant_notes exceeds the hard v1 cap") from error
        if "relevant_variance_explanations exceeds" in message:
            raise ValueError("DetectorPacket relevant_variance_explanations exceeds the hard v1 cap") from error
        if "tool_findings exceeds" in message:
            raise ValueError("DetectorPacket tool_findings exceeds the hard v1 cap") from error
        if "rules exceeds" in message:
            raise ValueError("DetectorPacket rules exceeds the hard v1 cap") from error
        if message == "invalid_detector_packet":
            raise ValueError("DetectorPacket is invalid") from error
        raise


def assess_detector_packets(
    detector_packets: list[dict[str, Any]],
    audit_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if audit_records is None:
        audit_records = []
    api_config = _load_api_detector_config()
    assessments = []
    for packet in detector_packets:
        network_call_attempted = False
        if api_config.enabled:
            network_call_attempted = True
            assessment = _assess_openrouter_detector_packet(packet, api_config)
            detector_name = "openrouter_api_detector"
            detector_version = "wave1_provisional"
        else:
            assessment = _assess_detector_packet(packet)
            detector_name = "deterministic_wave1_detector"
            detector_version = "v1.0"
        validate_detector_assessment(assessment, packet)
        assessments.append(assessment)
        audit_records.append(
            {
                "assessment_id": assessment["assessment_id"],
                "packet_id": packet["packet_id"],
                "candidate_id": packet["candidate_id"],
                "detector_name": detector_name,
                "detector_version": detector_version,
                "api_detector": api_config.audit_metadata(network_call_attempted=network_call_attempted),
                "stored_artifact_fields": [
                    "support_level",
                    "confidence",
                    "severity",
                    "validated_signals",
                    "cited_evidence_refs",
                    "rationale_short",
                ],
                "raw_prompt_stored": False,
                "raw_response_stored": bool(api_config.enabled and api_config.debug_raw_logging),
                "long_reasoning_stored": False,
            }
        )
    return assessments


def write_wave1_exit_demo_artifacts(
    output_path: Path,
    report_sets: list[CompanyReportSet],
    tool_gating_records: list[dict[str, Any]],
    tool_findings: list[dict[str, Any]],
    candidate_risks: list[dict[str, Any]],
    detector_packets: list[dict[str, Any]],
    detector_assessments: list[dict[str, Any]],
    final_reports: list[dict[str, Any]],
) -> dict[str, Path]:
    report_memories = [
        report_memory.raw
        for report_set in report_sets
        for report_memory in [report_set.target, report_set.prior_year]
    ]
    company_report_sets = [
        {
            "case_id": report_set.case_id,
            "target_report_id": report_set.target.report_id,
            "prior_year_report_id": report_set.prior_year.report_id,
            "company_name": report_set.target.metadata["company_name"],
            "period": report_set.target.metadata["period"],
            "prior_year_period": report_set.prior_year.metadata["period"],
            "report_profile": report_set.target.metadata["report_profile"],
            "report_basis": report_set.target.metadata["report_basis"],
        }
        for report_set in report_sets
    ]
    aggregation_output = {
        "aggregation_version": "wave1_fixture_aggregation_v1",
        "final_reports": [
            {
                "report_id": report["report_id"],
                "target_report_id": report["target_report_id"],
                "overall_assessment": report["overall_assessment"],
                "grouped_findings": report["grouped_findings"],
                "weak_or_limited_signals": report["weak_or_limited_signals"],
                "reviewed_candidate_audit": report["reviewed_candidate_audit"],
                "insufficient_evidence_or_data_gaps": report["insufficient_evidence_or_data_gaps"],
            }
            for report in final_reports
        ],
    }
    artifacts = {
        "report_memories": output_path / "report_memories.json",
        "company_report_sets": output_path / "company_report_sets.json",
        "tool_gating_records": output_path / "tool_gating_records.json",
        "tool_findings": output_path / "tool_findings.json",
        "candidate_risks": output_path / "candidate_risks.json",
        "detector_packets": output_path / "detector_packets.json",
        "detector_assessments": output_path / "detector_assessments.json",
        "aggregation_output": output_path / "aggregation_output.json",
    }
    payloads = {
        "report_memories": report_memories,
        "company_report_sets": company_report_sets,
        "tool_gating_records": tool_gating_records,
        "tool_findings": tool_findings,
        "candidate_risks": candidate_risks,
        "detector_packets": detector_packets,
        "detector_assessments": detector_assessments,
        "aggregation_output": aggregation_output,
    }
    for artifact_name, artifact_path in artifacts.items():
        artifact_path.write_text(
            json.dumps(payloads[artifact_name], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return artifacts


def write_wave1_thesis_artifacts(output_path: Path) -> dict[str, Path]:
    artifacts = {
        "thesis_architecture_notes": output_path / "thesis_architecture_notes.md",
        "thesis_pipeline_flow": output_path / "thesis_pipeline_flow.md",
        "thesis_contract_rationale": output_path / "thesis_contract_rationale.md",
        "language_compliance_examples": output_path / "language_compliance_examples.md",
    }
    artifacts["thesis_architecture_notes"].write_text(
        "\n".join(
            [
                "# Wave 1 Architecture Notes",
                "",
                "Wave 1 proves a fixture-only architecture spine for standard_corporate quarterly filings.",
                "The public command validates ReportMemory fixtures, constructs CompanyReportSet pairs, runs gating, tools, candidate generation, detector packets, deterministic assessments, aggregation, final reports, and append-only logs.",
                "Later waves extend the same contracts by adding profile breadth, real extraction, detector dataset work, and thesis evaluation around these artifacts.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    artifacts["thesis_pipeline_flow"].write_text(
        "\n".join(
            [
                "# Wave 1 Pipeline Flow",
                "",
                "Fixture ReportMemory",
                "-> CompanyReportSet",
                "-> tool availability gating",
                "-> deterministic ToolFinding records",
                "-> CandidateRisk records",
                "-> DetectorPacket records",
                "-> deterministic DetectorAssessment records",
                "-> aggregation output",
                "-> final JSON and Markdown reports",
                "-> append-only audit logs.",
                "",
                "System boundary: Wave 1 uses hand-authored fixtures only. It excludes real extraction, source retrieval, external context, non-standard profiles, model training data, and full thesis experiments.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    artifacts["thesis_contract_rationale"].write_text(
        "\n".join(
            [
                "# Wave 1 Contract Rationale",
                "",
                "- ReportMemory keeps tool-facing filing evidence structured and excludes current-filing comparison fallback values.",
                "- CompanyReportSet binds target and same-quarter prior-year filings with matching company, profile, basis, and period relationship.",
                "- Gating records make enabled, disabled, and skipped checks audit-visible without creating findings by themselves.",
                "- ToolFinding records carry deterministic calculations, thresholds, limitations, and evidence references.",
                "- CandidateRisk records remain one-category hypotheses for detector review.",
                "- DetectorPacket records preserve one candidate and one detector task with capped visible evidence.",
                "- DetectorAssessment records provide compact support level, confidence, severity, cited evidence, and short rationale.",
                "- Aggregation uses detector-reviewed assessments and does not upgrade severity beyond reviewed evidence.",
                "- Final reports summarize supported and weakly supported signals while retaining reviewed-candidate audit visibility.",
                "- Append-only logs preserve traceability for each stage.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    artifacts["language_compliance_examples"].write_text(
        "\n".join(
            [
                "# Wave 1 Language Compliance Examples",
                "",
                "Accepted wording:",
                "- The provided evidence supports a revenue quality risk signal that warrants human review.",
                "- The packet evidence weakly supports a possible cash-flow mismatch signal.",
                "- The provided evidence does not support this candidate risk signal.",
                "- The visible evidence is insufficient for assessment.",
                "",
                "Rejected wording categories:",
                "- Proven misconduct claims.",
                "- Management state-of-mind claims.",
                "- Hidden-action claims.",
                "- Market-abuse claims.",
                "- Legal conclusion claims.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return artifacts


def write_wave1_exit_demo_manifest(
    output_path: Path,
    report_sets: list[CompanyReportSet],
    persisted_artifact_paths: dict[str, Path],
    final_report_paths: list[dict[str, Path]],
    log_paths: dict[str, Path],
    thesis_artifact_paths: dict[str, Path],
) -> Path:
    manifest_path = output_path / "wave1_exit_demo_manifest.json"
    manifest = {
        "demo_id": "wave1_exit_demo",
        "public_interface": {
            "api": "research.fixture_spine.run_fixture_spine(output_dir)",
            "command": "python -m research.fixture_spine --output-dir <path>",
        },
        "fixture_cases": [report_set.case_id for report_set in report_sets],
        "scope": {
            "input_mode": "fixture_only",
            "report_profiles": sorted({report_set.target.metadata["report_profile"] for report_set in report_sets}),
            "requires_optional_api_detector": False,
            "excluded": [
                "real_pdf_or_ocr_extraction",
                "source_fetching",
                "external_data",
                "detector_sft_or_training_data",
                "full_thesis_evaluation_experiments",
                "current_filing_prior_comparison_fallback",
            ],
        },
        "contracts_extended_by_later_waves": [
            "ReportMemory",
            "CompanyReportSet",
            "ToolFinding",
            "CandidateRisk",
            "DetectorPacket",
            "DetectorAssessment",
            "FinalReport",
            "AppendOnlyAuditLog",
        ],
        "artifacts": {
            **{
                name: _relative_artifact_path(output_path, path)
                for name, path in persisted_artifact_paths.items()
            },
            "final_json_reports": [
                _relative_artifact_path(output_path, paths["json"]) for paths in final_report_paths
            ],
            "final_markdown_reports": [
                _relative_artifact_path(output_path, paths["markdown"]) for paths in final_report_paths
            ],
            "append_only_logs": {
                name: _relative_artifact_path(output_path, path) for name, path in log_paths.items()
            },
            **{
                name: _relative_artifact_path(output_path, path)
                for name, path in thesis_artifact_paths.items()
            },
        },
        "verification": {
            "regenerate": "python -m research.fixture_spine --output-dir artifacts/wave1_smoke",
            "test": "python -m pytest research/tests/test_fixture_spine.py",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def _relative_artifact_path(output_path: Path, artifact_path: Path) -> str:
    return artifact_path.relative_to(output_path).as_posix()


def validate_detector_assessment(assessment: dict[str, Any], packet: dict[str, Any]) -> None:
    try:
        _contract_validate_detector_assessment(assessment, packet)
    except ValueError as error:
        message = str(error)
        if message == "schema_mismatch":
            raise ValueError("DetectorAssessment is missing fields or has an invalid schema") from error
        if message == "identity_mismatch":
            if assessment.get("packet_id") != packet.get("packet_id"):
                raise ValueError("DetectorAssessment packet_id must match the DetectorPacket") from error
            if assessment.get("candidate_id") != packet.get("candidate_id"):
                raise ValueError("DetectorAssessment candidate_id must match the DetectorPacket") from error
            if assessment.get("report_id") != packet.get("report_id"):
                raise ValueError("DetectorAssessment report_id must match the DetectorPacket") from error
            raise ValueError("DetectorAssessment identity fields must match the DetectorPacket") from error
        if message == "risk_category_mismatch":
            raise ValueError("DetectorAssessment risk_category must match the DetectorPacket task") from error
        if message == "invalid_support_level":
            raise ValueError("DetectorAssessment support_level is not allowed") from error
        if message == "invalid_severity":
            raise ValueError("DetectorAssessment severity is not allowed") from error
        if message == "invalid_confidence":
            raise ValueError("DetectorAssessment confidence must be between 0 and 1") from error
        if message == "invalid_evidence_ids":
            raise ValueError("DetectorAssessment cited evidence must be visible in the DetectorPacket") from error
        if message == "rationale_too_long":
            raise ValueError("DetectorAssessment rationale_short must be 1-3 concise sentences") from error
        if message == "prohibited_risk_language":
            raise ValueError("DetectorAssessment contains prohibited legal or misconduct wording") from error
        raise


def _load_api_detector_config() -> ApiDetectorConfig:
    _load_dotenv_if_available()
    enabled = _env_flag(API_DETECTOR_ENABLE_ENV)
    api_key_present = bool(os.environ.get(API_DETECTOR_KEY_ENV))
    timeout_seconds = _env_float(API_DETECTOR_TIMEOUT_ENV, 30.0)
    max_retries = _env_int(API_DETECTOR_MAX_RETRIES_ENV, 1)
    debug_raw_logging = _env_flag(API_DETECTOR_DEBUG_RAW_ENV)
    if not enabled:
        status = "disabled"
        reason_code = "api_detector_not_explicitly_enabled"
    elif not api_key_present:
        status = "disabled"
        reason_code = "missing_openrouter_api_key"
        enabled = False
    else:
        status = "enabled"
        reason_code = "explicit_config_and_credentials_present"
    return ApiDetectorConfig(
        provider=API_DETECTOR_PROVIDER,
        model=os.environ.get("VINACOUNT_API_DETECTOR_MODEL", API_DETECTOR_MODEL),
        enabled=enabled,
        api_key_present=api_key_present,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        debug_raw_logging=debug_raw_logging,
        status=status,
        reason_code=reason_code,
    )


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a number") from error


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error


def _assess_openrouter_detector_packet(packet: dict[str, Any], config: ApiDetectorConfig) -> dict[str, Any]:
    payload = {
        "model": config.model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Return JSON only. Return exactly one DetectorAssessment. "
                    "Use only evidence IDs present in the DetectorPacket. "
                    "Use conservative risk-signal language. Do not return long reasoning."
                ),
            },
            {"role": "user", "content": json.dumps(packet, sort_keys=True)},
        ],
        "temperature": 0.2,
    }
    raw_response = _post_openrouter_json(payload, config)
    content = raw_response["choices"][0]["message"]["content"]
    try:
        assessment = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("OpenRouter detector returned malformed assessment output") from error
    validate_detector_assessment(assessment, packet)
    return assessment


def _post_openrouter_json(payload: dict[str, Any], config: ApiDetectorConfig) -> dict[str, Any]:
    api_key = os.environ[API_DETECTOR_KEY_ENV]
    last_error: Exception | None = None
    for _attempt in range(config.max_retries + 1):
        request = urllib.request.Request(
            API_DETECTOR_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/buinguyenkhai/vinacount",
                "X-Title": "vinacount-wave1-api-detector",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as error:
            last_error = error
    raise RuntimeError("OpenRouter detector call failed") from last_error


def _assess_detector_packet(packet: dict[str, Any]) -> dict[str, Any]:
    flagged_tool_findings = [finding for finding in packet["tool_findings"] if finding["flag"]]
    cited_evidence_refs = _assessment_cited_evidence_refs(packet, flagged_tool_findings)
    support_level = "supported" if len(flagged_tool_findings) >= 2 else "weakly_supported"
    if not flagged_tool_findings:
        support_level = "not_supported" if _has_substantive_packet_evidence(packet) else "insufficient_evidence"
        cited_evidence_refs = _assessment_context_evidence_refs(packet)

    return {
        "assessment_id": f"ASSESS_{packet['packet_id']}",
        "packet_id": packet["packet_id"],
        "candidate_id": packet["candidate_id"],
        "report_id": packet["report_id"],
        "risk_category": packet["task"]["risk_category"],
        "support_level": support_level,
        "confidence": _assessment_confidence(support_level),
        "severity": _assessment_severity(packet, support_level),
        "validated_signals": _assessment_validated_signals(
            packet,
            support_level,
            flagged_tool_findings,
            cited_evidence_refs,
        ),
        "cited_evidence_refs": cited_evidence_refs,
        "rationale_short": _assessment_rationale(packet, support_level, flagged_tool_findings),
    }


def _assessment_confidence(support_level: str) -> float:
    return {
        "supported": 0.82,
        "weakly_supported": 0.58,
        "not_supported": 0.42,
        "insufficient_evidence": 0.25,
    }[support_level]


def _has_substantive_packet_evidence(packet: dict[str, Any]) -> bool:
    return bool(
        packet.get("tool_findings")
        or packet.get("relevant_table_rows")
        or packet.get("relevant_notes")
        or packet.get("relevant_variance_explanations")
    )


def _assessment_severity(packet: dict[str, Any], support_level: str) -> str:
    if support_level not in {"supported", "weakly_supported"}:
        return "unknown"
    if packet["candidate_summary"]["priority"] == "high" and support_level == "supported":
        return "high"
    return "medium"


def _assessment_rationale(
    packet: dict[str, Any],
    support_level: str,
    flagged_tool_findings: list[dict[str, Any]],
) -> str:
    if support_level == "supported":
        signal_count = len(flagged_tool_findings)
        return (
            f"The provided packet includes {signal_count} aligned tool signals for the candidate risk category. "
            "The cited evidence supports a risk signal that warrants human review."
        )
    if support_level == "weakly_supported":
        return (
            "The provided packet includes one supporting tool signal for the candidate risk category. "
            "The evidence is limited, so the assessment remains weakly supported."
        )
    if support_level == "not_supported":
        return "The provided packet evidence does not support the candidate risk signal."
    return "The packet lacks enough visible evidence to assess the candidate risk signal."


def _assessment_cited_evidence_refs(
    packet: dict[str, Any],
    flagged_tool_findings: list[dict[str, Any]],
) -> list[dict[str, str]]:
    visible_evidence_ids = _detector_visible_evidence_ids(packet)
    cited = []
    for finding in flagged_tool_findings:
        cited.append(
            {
                "evidence_ref_type": "tool_result",
                "ref_id": finding["tool_result_id"],
                "role": "supporting",
            }
        )
        cited.extend(
            _assessment_evidence_ref(ref, role="supporting")
            for ref in finding["evidence_refs"]
            if ref["ref_id"] in visible_evidence_ids
        )
    return _dedupe_evidence_refs(cited)


def _assessment_context_evidence_refs(packet: dict[str, Any]) -> list[dict[str, str]]:
    cited = [
        {"evidence_ref_type": "rule", "ref_id": rule["rule_id"], "role": "context"}
        for rule in packet.get("rules", [])
        if rule.get("rule_id")
    ]
    if cited:
        return cited
    for row in packet.get("relevant_table_rows", []):
        report_id = row.get("report_id")
        for cell in row.get("values", {}).values():
            if isinstance(cell, dict) and cell.get("cell_id") and report_id:
                cited.append(
                    {
                        "evidence_ref_type": "table_cell",
                        "ref_id": f"{report_id}:{cell['cell_id']}",
                        "role": "context",
                    }
                )
    return _dedupe_evidence_refs(cited)


def _assessment_validated_signals(
    packet: dict[str, Any],
    support_level: str,
    flagged_tool_findings: list[dict[str, Any]],
    cited_evidence_refs: list[dict[str, str]],
) -> list[dict[str, Any]]:
    if flagged_tool_findings:
        signal_status = "validated" if support_level == "supported" else "partially_validated"
        return [
            {
                "signal_id": finding["signal_id"],
                "tool_result_id": finding["tool_result_id"],
                "status": signal_status,
                "support_level": support_level,
                "cited_evidence_refs": [
                    _assessment_evidence_ref(ref, role="supporting")
                    for ref in finding["evidence_refs"]
                    if ref["ref_id"] in {evidence_ref["ref_id"] for evidence_ref in cited_evidence_refs}
                ]
                or [{"evidence_ref_type": "tool_result", "ref_id": finding["tool_result_id"], "role": "supporting"}],
            }
            for finding in flagged_tool_findings
        ]
    if not cited_evidence_refs:
        return []
    return [
        {
            "signal_id": _assessment_fallback_signal_id(packet),
            "status": "rejected" if support_level == "not_supported" else "not_assessable",
            "support_level": support_level,
            "cited_evidence_refs": cited_evidence_refs[:1],
        }
    ]


def _assessment_fallback_signal_id(packet: dict[str, Any]) -> str:
    signal_ids = packet.get("candidate_summary", {}).get("supporting_signal_ids", [])
    if signal_ids:
        return signal_ids[0]
    for rule in packet.get("rules", []):
        related = rule.get("related_signal_ids", [])
        if related:
            return related[0]
    return "candidate_signal_not_assessable"


def _assessment_evidence_ref(ref: dict[str, str], *, role: str) -> dict[str, str]:
    return {
        "evidence_ref_type": ref["evidence_ref_type"],
        "ref_id": ref["ref_id"],
        "role": role,
    }


def _detector_visible_evidence_ids(packet: dict[str, Any]) -> set[str]:
    return _contract_visible_packet_evidence_ids(packet)


def _trim_detector_packet(
    packet: dict[str, Any],
    candidate: dict[str, Any],
    audit_records: list[dict[str, Any]],
) -> dict[str, Any]:
    for field_name, id_field in [
        ("relevant_table_rows", "row_id"),
        ("relevant_notes", "note_id"),
        ("relevant_variance_explanations", "span_id"),
        ("tool_findings", "tool_result_id"),
        ("rules", "rule_id"),
    ]:
        cap = DETECTOR_PACKET_CAPS[field_name]
        items = packet[field_name]
        if len(items) <= cap:
            continue
        kept = items[:cap]
        omitted = items[cap:]
        packet[field_name] = kept
        audit_records.extend(
            {
                "packet_id": packet["packet_id"],
                "candidate_id": candidate["candidate_id"],
                "field": field_name,
                "omitted_evidence_id": item[id_field],
                "reason_code": "structural_cap_trimmed_optional_evidence",
            }
            for item in omitted
        )
    return packet


def _detector_packet_metadata(report_memory: ReportMemory) -> dict[str, Any]:
    metadata = report_memory.metadata
    return {
        "company_name": metadata["company_name"],
        "ticker": metadata.get("ticker"),
        "period": metadata["period"],
        "report_period_type": metadata["report_period_type"],
        "report_profile": metadata["report_profile"],
        "insurance_subprofile": metadata.get("insurance_subprofile"),
        "industry": metadata.get("industry"),
        "report_assurance_type": metadata["report_assurance_type"],
        "currency": metadata["currency"],
        "unit": metadata["unit"],
        "report_basis": metadata["report_basis"],
        "business_context_tags": metadata["business_context_tags"],
    }


def _detector_task_question(risk_category: str) -> str:
    labels = {
        "revenue_income_recognition_risk": "revenue or income recognition risk signal",
        "receivables_credit_quality_risk": "credit quality risk signal",
        "expense_liability_understatement_risk": "expense or liability risk signal",
        "earnings_cashflow_mismatch": "earnings and cash-flow mismatch risk signal",
        "disclosure_inconsistency_or_obfuscation": "disclosure inconsistency risk signal",
    }
    return f"Does the provided evidence support a {labels.get(risk_category, 'candidate risk signal')}?"


def _packet_table_rows(
    report_set: CompanyReportSet,
    candidate: dict[str, Any],
    linked_tool_findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    refs = _required_table_cell_refs(candidate) + [
        ref
        for finding in linked_tool_findings
        for ref in finding["evidence_refs"]
        if ref["evidence_ref_type"] == "table_cell"
    ]
    rows = []
    seen = set()
    report_memories = {
        report_set.target.report_id: report_set.target,
        report_set.prior_year.report_id: report_set.prior_year,
    }
    for ref in _dedupe_evidence_refs(refs):
        report_memory = report_memories[ref["report_id"]]
        row = _find_row_for_cell(report_memory, ref["local_evidence_id"])
        key = (ref["report_id"], row["row_id"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "report_id": ref["report_id"],
                "local_evidence_id": ref["local_evidence_id"],
                "table_id": row["table_id"],
                "table_type": row["section_type"],
                "row_id": row["row_id"],
                "label": row["label"],
                "original_label": row.get("original_label"),
                "standard_account": _standard_account_for_row(row),
                "account_group": _account_group_for_row(row),
                "period_basis": row["period_basis"],
                "source_document_id": row.get("source_document_id"),
                "values": {
                    cell["period"]: {
                        "cell_id": cell["cell_id"],
                        "value": cell["value"],
                        "unit": cell.get("unit"),
                    }
                    for cell in row["cells"]
                },
            }
        )
    return rows


def _required_table_cell_refs(candidate: dict[str, Any]) -> list[dict[str, str]]:
    return [
        ref
        for ref in candidate["required_evidence_refs"]
        if ref["evidence_ref_type"] == "table_cell"
    ]


def _packet_notes(report_set: CompanyReportSet, candidate: dict[str, Any]) -> list[dict[str, Any]]:
    wanted_ids = {
        ref["local_evidence_id"]
        for ref in candidate["required_evidence_refs"]
        if ref["evidence_ref_type"] == "note_span" and ref.get("report_id") == report_set.target.report_id
    }
    return [
        {
            "report_id": report_set.target.report_id,
            "note_id": note["note_id"],
            "note_type": note.get("note_type", "selected_note"),
            "title": note.get("title", "Selected note"),
            "text": note["text"],
            "linked_cell_ids": note.get("linked_cell_ids", []),
            "linked_row_ids": note.get("linked_row_ids", []),
            "source_document_id": note.get("source_document_id"),
        }
        for note in report_set.target.raw["notes"]
        if note["note_id"] in wanted_ids
    ]


def _packet_variance_explanations(report_set: CompanyReportSet, candidate: dict[str, Any]) -> list[dict[str, Any]]:
    wanted_ids = {
        ref["local_evidence_id"]
        for ref in candidate["required_evidence_refs"]
        if ref["evidence_ref_type"] == "variance_explanation_span"
        and ref.get("report_id") == report_set.target.report_id
    }
    return [
        {
            "report_id": report_set.target.report_id,
            "span_id": explanation["span_id"],
            "title": explanation.get("title", "Selected variance explanation"),
            "text": explanation["text"],
            "related_metric": explanation.get("related_metric"),
            "related_row_ids": explanation.get("related_row_ids", []),
            "source_document_id": explanation.get("source_document_id"),
        }
        for explanation in report_set.target.raw["variance_explanations"]
        if explanation["span_id"] in wanted_ids
    ]


def _packet_tool_finding(finding: dict[str, Any]) -> dict[str, Any]:
    return {
        "tool_result_id": finding["tool_result_id"],
        "tool_name": finding["tool_name"],
        "tool_category": finding["tool_category"],
        "risk_category": finding["risk_category"],
        "signal_id": finding["signal_id"],
        "metric_name": finding["metric"].get("metric_name"),
        "value": finding["metric"].get("value"),
        "threshold": finding["threshold"],
        "flag": finding["flag"],
        "strength": finding["strength"],
        "summary": finding["finding_summary"],
        "evidence_refs": finding["evidence_refs"],
    }


def _packet_rules(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "rule_id": f"RULE_{candidate['risk_category'].upper()}_001",
            "rule_name": "Assess only the candidate risk category using provided evidence",
            "description": (
                "Use the visible evidence and linked tool findings to assess whether the candidate "
                "risk signal is supported; do not add secondary detector-side risk categories."
            ),
            "risk_category": candidate["risk_category"],
            "related_signal_ids": candidate["supporting_signal_ids"],
        }
    ]


def _find_row_for_cell(report_memory: ReportMemory, cell_id: str) -> dict[str, Any]:
    index_entry = report_memory.raw["cell_index"][cell_id]
    for table in report_memory.raw["tables"]:
        if table["table_id"] != index_entry["table_id"]:
            continue
        section_type = _section_type(report_memory, table["section_id"])
        for row in table["rows"]:
            if row["row_id"] == index_entry["row_id"]:
                return {
                    **row,
                    "table_id": table["table_id"],
                    "section_type": section_type,
                    "period_basis": table["period_basis"],
                    "source_document_id": table.get("source_document_id"),
                }
    raise ValueError(f"Missing table row for evidence cell {cell_id}")


def _section_type(report_memory: ReportMemory, section_id: str) -> str:
    for section in report_memory.raw["sections"]:
        if section["section_id"] == section_id:
            return section["section_type"]
    return "unknown"


def _standard_account_for_row(row: dict[str, Any]) -> str:
    for account in [
        "revenue",
        "trade_receivables",
        "profit_after_tax",
        "operating_cash_flow",
        "loans_to_customers",
        "loan_group_1",
        "loan_group_2",
        "loan_group_3",
        "loan_group_4",
        "loan_group_5",
        "general_provision",
        "specific_provision",
        "margin_lending",
        "customer_advances",
        "margin_impairment",
        "fvtpl_assets",
        "afs_assets",
        "htm_assets",
        "fvtpl_unrealized_gain",
        "gross_written_premium",
        "premium_receivables",
        "claims_expense",
        "insurance_reserves",
        "reinsurance_recoverables",
        "reinsurance_payables",
    ]:
        if _row_matches_account(row, account):
            return account
    return "unknown"


def _account_group_for_row(row: dict[str, Any]) -> str:
    account = _standard_account_for_row(row)
    if account == "revenue":
        return "revenue_income"
    if account == "trade_receivables":
        return "receivables_credit"
    if account in {"loans_to_customers", "loan_group_1", "loan_group_2", "loan_group_3", "loan_group_4", "loan_group_5"}:
        return "receivables_credit"
    if account in {"general_provision", "specific_provision"}:
        return "expense_liability"
    if account in {"margin_lending", "customer_advances", "margin_impairment"}:
        return "receivables_credit"
    if account in {"fvtpl_assets", "afs_assets", "htm_assets", "fvtpl_unrealized_gain"}:
        return "asset_quality"
    if account in {"gross_written_premium", "claims_expense"}:
        return "revenue_income"
    if account == "premium_receivables":
        return "receivables_credit"
    if account in {"insurance_reserves", "reinsurance_payables"}:
        return "expense_liability"
    if account == "reinsurance_recoverables":
        return "asset_quality"
    if account == "profit_after_tax":
        return "revenue_income"
    if account == "operating_cash_flow":
        return "cashflow"
    return "unknown"


def _build_receivables_vs_revenue_finding(report_set: CompanyReportSet) -> dict[str, Any]:
    current_receivables = _require_account_cell(report_set.target, "trade_receivables")
    comparison_receivables = _require_account_cell(report_set.prior_year, "trade_receivables")
    current_revenue = _require_account_cell(report_set.target, "revenue")
    comparison_revenue = _require_account_cell(report_set.prior_year, "revenue")

    receivables_growth = _percentage_growth(current_receivables["value"], comparison_receivables["value"])
    revenue_growth = _percentage_growth(current_revenue["value"], comparison_revenue["value"])
    growth_gap = round(receivables_growth - revenue_growth, 2)
    threshold_value = 10.0
    flag = growth_gap > threshold_value

    return {
        "tool_result_id": f"TOOL_{report_set.target.report_id}_REC_VS_REV_001",
        "report_id": report_set.target.report_id,
        "tool_name": "standard_corporate_receivables_vs_revenue_growth_tool",
        "tool_version": "v1.0",
        "tool_category": "quantitative",
        "risk_category": "revenue_income_recognition_risk",
        "analysis_scope": "company_report_set",
        "period_basis": "quarter",
        "signal_id": "receivables_growth_outpaces_revenue" if flag else "receivables_growth_not_outpacing_revenue",
        "metric": {
            "metric_name": "receivables_growth_minus_revenue_growth_pct_points",
            "value": growth_gap,
            "unit": "percent",
            "period_current": report_set.target.metadata["period"],
            "period_comparison": report_set.prior_year.metadata["period"],
            "direction": "increase" if growth_gap > 0 else "decrease" if growth_gap < 0 else "unchanged",
        },
        "threshold": {
            "threshold_type": "greater_than",
            "value": threshold_value,
            "unit": "percent",
            "basis": "configured_default_v1",
            "config_version": "standard_corporate_v1",
            "description": "Flag when receivables growth exceeds revenue growth by more than 10 percentage points.",
        },
        "flag": flag,
        "strength": "moderate" if flag else "not_applicable",
        "finding_summary": (
            f"Receivables growth exceeded revenue growth by {growth_gap:.2f} percentage points, "
            f"{'exceeding' if flag else 'not exceeding'} the configured {threshold_value:.1f} point threshold."
        ),
        "evidence_refs": [
            _cell_evidence_ref(report_set.target.report_id, current_receivables["cell_id"]),
            _cell_evidence_ref(report_set.prior_year.report_id, comparison_receivables["cell_id"]),
            _cell_evidence_ref(report_set.target.report_id, current_revenue["cell_id"]),
            _cell_evidence_ref(report_set.prior_year.report_id, comparison_revenue["cell_id"]),
        ],
        "calculation": {
            "formula": "receivables_growth_pct - revenue_growth_pct",
            "inputs": [
                {"name": "current_receivables", "value": current_receivables["value"], "cell_id": current_receivables["cell_id"]},
                {
                    "name": "comparison_receivables",
                    "value": comparison_receivables["value"],
                    "cell_id": comparison_receivables["cell_id"],
                },
                {"name": "current_revenue", "value": current_revenue["value"], "cell_id": current_revenue["cell_id"]},
                {"name": "comparison_revenue", "value": comparison_revenue["value"], "cell_id": comparison_revenue["cell_id"]},
            ],
            "computed_value": growth_gap,
            "rounding": "rounded_to_2_decimal_places",
        },
        "limitations": [
            "The tool compares only the provided current and comparison periods.",
            "External context is excluded in v1.",
        ],
        "status": "completed",
    }


def _build_earnings_cashflow_mismatch_finding(report_set: CompanyReportSet) -> dict[str, Any]:
    current_profit = _require_account_cell(report_set.target, "profit_after_tax", "year_to_date")
    current_cfo = _require_account_cell(report_set.target, "operating_cash_flow", "year_to_date")
    comparison_profit = _require_account_cell(report_set.prior_year, "profit_after_tax", "year_to_date")
    comparison_cfo = _require_account_cell(report_set.prior_year, "operating_cash_flow", "year_to_date")

    cfo_to_net_income_ratio = round(current_cfo["value"] / abs(current_profit["value"]), 2)
    cfo_net_income_gap = current_profit["value"] - current_cfo["value"]
    threshold_value = 0.70
    positive_profit_negative_cfo = current_profit["value"] > 0 and current_cfo["value"] < 0
    flag = cfo_to_net_income_ratio < threshold_value or positive_profit_negative_cfo

    if positive_profit_negative_cfo:
        signal_id = "positive_profit_negative_cfo"
    elif flag:
        signal_id = "cfo_net_income_gap_high"
    else:
        signal_id = "cashflow_support_adequate"

    return {
        "tool_result_id": f"TOOL_{report_set.target.report_id}_EARN_CASH_001",
        "report_id": report_set.target.report_id,
        "tool_name": "standard_corporate_earnings_cashflow_mismatch_tool",
        "tool_version": "v1.0",
        "tool_category": "quantitative",
        "risk_category": "earnings_cashflow_mismatch",
        "analysis_scope": "company_report_set",
        "period_basis": "year_to_date",
        "signal_id": signal_id,
        "metric": {
            "metric_name": "cfo_to_net_income_ratio",
            "value": cfo_to_net_income_ratio,
            "unit": "ratio",
            "period_current": report_set.target.metadata["period"],
            "period_comparison": report_set.prior_year.metadata["period"],
            "direction": "decrease" if cfo_to_net_income_ratio < threshold_value else "not_applicable",
        },
        "threshold": {
            "threshold_type": "ratio_less_than",
            "value": threshold_value,
            "unit": "ratio",
            "basis": "configured_default_v1",
            "config_version": "standard_corporate_v1",
            "description": "Flag when operating cash flow is less than 70% of net income.",
        },
        "flag": flag,
        "strength": "strong" if positive_profit_negative_cfo else "moderate" if flag else "not_applicable",
        "finding_summary": (
            f"Year-to-date operating cash flow was {cfo_to_net_income_ratio:.2f} times net income, "
            f"{'with negative operating cash flow despite positive profit' if positive_profit_negative_cfo else 'below' if flag else 'not below'} "
            f"the configured {threshold_value:.2f} threshold."
        ),
        "evidence_refs": [
            _cell_evidence_ref(report_set.target.report_id, current_profit["cell_id"]),
            _cell_evidence_ref(report_set.target.report_id, current_cfo["cell_id"]),
            _cell_evidence_ref(report_set.prior_year.report_id, comparison_profit["cell_id"]),
            _cell_evidence_ref(report_set.prior_year.report_id, comparison_cfo["cell_id"]),
        ],
        "calculation": {
            "formula": "operating_cash_flow / abs(profit_after_tax)",
            "inputs": [
                {
                    "name": "current_profit_after_tax",
                    "value": current_profit["value"],
                    "cell_id": current_profit["cell_id"],
                    "report_id": report_set.target.report_id,
                    "local_evidence_id": current_profit["cell_id"],
                    "period_basis": "year_to_date",
                },
                {
                    "name": "current_operating_cash_flow",
                    "value": current_cfo["value"],
                    "cell_id": current_cfo["cell_id"],
                    "report_id": report_set.target.report_id,
                    "local_evidence_id": current_cfo["cell_id"],
                    "period_basis": "year_to_date",
                },
                {
                    "name": "comparison_profit_after_tax",
                    "value": comparison_profit["value"],
                    "cell_id": comparison_profit["cell_id"],
                    "report_id": report_set.prior_year.report_id,
                    "local_evidence_id": comparison_profit["cell_id"],
                    "period_basis": "year_to_date",
                },
                {
                    "name": "comparison_operating_cash_flow",
                    "value": comparison_cfo["value"],
                    "cell_id": comparison_cfo["cell_id"],
                    "report_id": report_set.prior_year.report_id,
                    "local_evidence_id": comparison_cfo["cell_id"],
                    "period_basis": "year_to_date",
                },
            ],
            "computed_value": cfo_to_net_income_ratio,
            "secondary_values": {"cfo_net_income_gap": cfo_net_income_gap},
            "rounding": "rounded_to_2_decimal_places",
        },
        "limitations": [
            "The tool uses year-to-date evidence only and does not fall back to quarter-only cash-flow evidence.",
            "External context is excluded in v1.",
        ],
        "status": "completed",
    }


def _build_disclosure_consistency_finding(report_set: CompanyReportSet) -> dict[str, Any]:
    note = report_set.target.raw["notes"][0]
    variance = report_set.target.raw["variance_explanations"][0]
    note_text = note["text"].lower()
    variance_text = variance["text"].lower()
    narrative_tension = (
        "receivable" in note_text
        and "cash collection" in note_text
        and "revenue" in variance_text
        and "receivable" not in variance_text
        and "cash" not in variance_text
    )

    return {
        "tool_result_id": f"TOOL_{report_set.target.report_id}_DISC_CONSISTENCY_001",
        "report_id": report_set.target.report_id,
        "tool_name": "standard_corporate_disclosure_consistency_tool",
        "tool_version": "v1.0",
        "tool_category": "consistency",
        "risk_category": "disclosure_inconsistency_or_obfuscation",
        "analysis_scope": "company_report_set",
        "period_basis": "quarter",
        "signal_id": "disclosure_narrative_tension" if narrative_tension else "disclosure_context_aligned",
        "metric": {
            "metric_name": "disclosure_narrative_tension",
            "value": narrative_tension,
            "unit": "boolean",
            "period_current": report_set.target.metadata["period"],
            "checks": {
                "note_mentions_receivables": "receivable" in note_text,
                "note_mentions_cash_collection": "cash collection" in note_text,
                "variance_mentions_revenue": "revenue" in variance_text,
                "variance_addresses_note_context": "receivable" in variance_text or "cash" in variance_text,
            },
        },
        "threshold": {
            "threshold_type": "rule",
            "value": "narrative_tension_when_note_context_is_not_addressed",
            "unit": "boolean",
            "basis": "configured_default_v1",
            "config_version": "standard_corporate_v1",
            "description": "Flag disclosure tension when extracted note context is not addressed by the variance explanation.",
        },
        "flag": narrative_tension,
        "strength": "moderate" if narrative_tension else "not_applicable",
        "finding_summary": (
            "Disclosure notes mention receivables and weak cash collection, while the variance explanation "
            "focuses on revenue growth without addressing that note context."
            if narrative_tension
            else "Extracted disclosure context is aligned with the variance explanation."
        ),
        "evidence_refs": [
            _span_evidence_ref(report_set.target.report_id, "note_span", note["note_id"]),
            _span_evidence_ref(report_set.target.report_id, "variance_explanation_span", variance["span_id"]),
        ],
        "calculation": {
            "method": "deterministic_keyword_consistency_check_v1",
            "inputs": [
                {"name": "note_text", "report_id": report_set.target.report_id, "local_evidence_id": note["note_id"]},
                {
                    "name": "variance_explanation_text",
                    "report_id": report_set.target.report_id,
                    "local_evidence_id": variance["span_id"],
                },
            ],
            "computed_value": narrative_tension,
        },
        "limitations": [
            "The check uses extracted fixture notes and variance explanations only.",
            "External context is excluded in v1.",
        ],
        "status": "completed",
    }


def _build_variance_explanation_quality_finding(report_set: CompanyReportSet) -> dict[str, Any]:
    current_profit = _require_account_cell(report_set.target, "profit_after_tax", "year_to_date")
    comparison_profit = _require_account_cell(report_set.prior_year, "profit_after_tax", "year_to_date")
    profit_growth = _percentage_growth(current_profit["value"], comparison_profit["value"])
    threshold_value = 20.0
    explanation = report_set.target.raw["variance_explanations"][0]
    text = explanation["text"].lower()
    material_movement = abs(profit_growth) > threshold_value
    checks = {
        "concrete_driver": any(term in text for term in ["sales volume", "price", "cost", "margin", "collection"]),
        "connected_to_changed_metric": "profit" in text or "net income" in text or "profit after tax" in text,
        "directionally_consistent": ("growth" in text or "expanded" in text or "increase" in text)
        if profit_growth > 0
        else ("decrease" in text or "decline" in text),
        "non_boilerplate": len(text.split()) >= 6 and "business condition" not in text,
        "missing_explanation_for_material_change": material_movement and not text.strip(),
    }
    weak_connection = material_movement and not checks["connected_to_changed_metric"]
    flag = weak_connection or checks["missing_explanation_for_material_change"]

    return {
        "tool_result_id": f"TOOL_{report_set.target.report_id}_VAR_EXPLAIN_001",
        "report_id": report_set.target.report_id,
        "tool_name": "standard_corporate_variance_explanation_quality_tool",
        "tool_version": "v1.0",
        "tool_category": "disclosure",
        "risk_category": "disclosure_inconsistency_or_obfuscation",
        "analysis_scope": "company_report_set",
        "period_basis": "year_to_date",
        "signal_id": "variance_explanation_weak_connection" if flag else "variance_explanation_quality_supported",
        "metric": {
            "metric_name": "profit_variance_explanation_quality",
            "value": profit_growth,
            "unit": "percent",
            "period_current": report_set.target.metadata["period"],
            "period_comparison": report_set.prior_year.metadata["period"],
            "direction": "increase" if profit_growth > 0 else "decrease" if profit_growth < 0 else "unchanged",
            "checks": checks,
        },
        "threshold": {
            "threshold_type": "absolute_change_greater_than",
            "value": threshold_value,
            "unit": "percent",
            "basis": "configured_default_v1",
            "config_version": "standard_corporate_v1",
            "description": "Review variance explanation quality when profit movement is material.",
        },
        "flag": flag,
        "strength": "moderate" if flag else "not_applicable",
        "finding_summary": (
            f"Profit changed by {profit_growth:.2f}% year over year; the variance explanation includes a concrete "
            "driver but does not clearly connect it to the changed profit metric."
            if flag
            else f"Profit changed by {profit_growth:.2f}% year over year and the variance explanation passed the simple quality checks."
        ),
        "evidence_refs": [
            _span_evidence_ref(report_set.target.report_id, "variance_explanation_span", explanation["span_id"]),
            _cell_evidence_ref(report_set.target.report_id, current_profit["cell_id"]),
            _cell_evidence_ref(report_set.prior_year.report_id, comparison_profit["cell_id"]),
        ],
        "calculation": {
            "method": "deterministic_variance_explanation_quality_checks_v1",
            "inputs": [
                {
                    "name": "variance_explanation_text",
                    "report_id": report_set.target.report_id,
                    "local_evidence_id": explanation["span_id"],
                },
                {
                    "name": "current_profit_after_tax",
                    "value": current_profit["value"],
                    "report_id": report_set.target.report_id,
                    "local_evidence_id": current_profit["cell_id"],
                    "period_basis": "year_to_date",
                },
                {
                    "name": "comparison_profit_after_tax",
                    "value": comparison_profit["value"],
                    "report_id": report_set.prior_year.report_id,
                    "local_evidence_id": comparison_profit["cell_id"],
                    "period_basis": "year_to_date",
                },
            ],
            "computed_value": profit_growth,
            "checks": checks,
            "rounding": "rounded_to_2_decimal_places",
        },
        "limitations": [
            "The check is a simple auditable fixture rule for the extracted explanation text.",
            "External context is excluded in v1.",
        ],
        "status": "completed",
    }


def _build_credit_institution_loan_quality_finding(report_set: CompanyReportSet) -> dict[str, Any]:
    current_groups = {
        group: _require_account_cell(report_set.target, f"loan_group_{group}", "balance_sheet_date")
        for group in range(1, 6)
    }
    comparison_groups = {
        group: _require_account_cell(report_set.prior_year, f"loan_group_{group}", "balance_sheet_date")
        for group in range(1, 6)
    }
    current_total = sum(cell["value"] for cell in current_groups.values())
    comparison_total = sum(cell["value"] for cell in comparison_groups.values())
    current_npl = sum(current_groups[group]["value"] for group in [3, 4, 5])
    comparison_npl = sum(comparison_groups[group]["value"] for group in [3, 4, 5])
    current_npl_ratio = round(current_npl / current_total * 100, 2)
    comparison_npl_ratio = round(comparison_npl / comparison_total * 100, 2)
    ratio_change = round(current_npl_ratio - comparison_npl_ratio, 2)
    threshold_value = 0.75
    flag = ratio_change > threshold_value

    return {
        "tool_result_id": f"TOOL_{report_set.target.report_id}_LOAN_QUALITY_001",
        "report_id": report_set.target.report_id,
        "tool_name": "credit_institution_loan_quality_tool",
        "tool_version": "v1.0",
        "tool_category": "quantitative",
        "risk_category": "receivables_credit_quality_risk",
        "analysis_scope": "company_report_set",
        "period_basis": "balance_sheet_date",
        "signal_id": "loan_quality_deterioration" if flag else "loan_quality_stable",
        "metric": {
            "metric_name": "npl_ratio_change_pct_points",
            "value": ratio_change,
            "unit": "percentage_points",
            "period_current": report_set.target.metadata["period"],
            "period_comparison": report_set.prior_year.metadata["period"],
            "direction": "increase" if ratio_change > 0 else "decrease" if ratio_change < 0 else "unchanged",
            "secondary_values": {
                "npl_ratio_current": current_npl_ratio,
                "npl_ratio_comparison": comparison_npl_ratio,
                "current_npl_amount": current_npl,
                "comparison_npl_amount": comparison_npl,
                "current_total_loans_by_group": current_total,
                "comparison_total_loans_by_group": comparison_total,
            },
        },
        "threshold": {
            "threshold_type": "increase_greater_than",
            "value": threshold_value,
            "unit": "percentage_points",
            "basis": "configured_default_v1",
            "config_version": "credit_institution_v1",
            "description": "Review when report-internal NPL ratio increases materially year over year.",
        },
        "flag": flag,
        "strength": "strong" if flag else "not_applicable",
        "finding_summary": (
            f"NPL ratio increased by {ratio_change:.2f} percentage points based on loan groups 3-5 over total loan groups."
            if flag
            else f"NPL ratio changed by {ratio_change:.2f} percentage points based on loan groups 3-5 over total loan groups."
        ),
        "evidence_refs": [
            _cell_evidence_ref(report_set.target.report_id, cell["cell_id"])
            for cell in current_groups.values()
        ]
        + [
            _cell_evidence_ref(report_set.prior_year.report_id, cell["cell_id"])
            for cell in comparison_groups.values()
        ],
        "calculation": {
            "formula": "(groups_3_to_5 / groups_1_to_5) current minus comparison",
            "inputs": [
                {
                    "name": f"current_loan_group_{group}",
                    "value": cell["value"],
                    "cell_id": cell["cell_id"],
                    "report_id": report_set.target.report_id,
                    "local_evidence_id": cell["cell_id"],
                    "period_basis": "balance_sheet_date",
                }
                for group, cell in current_groups.items()
            ]
            + [
                {
                    "name": f"comparison_loan_group_{group}",
                    "value": cell["value"],
                    "cell_id": cell["cell_id"],
                    "report_id": report_set.prior_year.report_id,
                    "local_evidence_id": cell["cell_id"],
                    "period_basis": "balance_sheet_date",
                }
                for group, cell in comparison_groups.items()
            ],
            "computed_value": ratio_change,
            "secondary_values": {
                "npl_ratio_current": current_npl_ratio,
                "npl_ratio_comparison": comparison_npl_ratio,
            },
            "rounding": "rounded_to_2_decimal_places",
        },
        "limitations": [
            "The tool uses fixture-visible loan group evidence only.",
            "Restructuring disclosures are context only and are not standalone candidate triggers.",
            "External context is excluded in v1.",
        ],
        "status": "completed",
    }


def _build_credit_institution_provision_movement_finding(report_set: CompanyReportSet) -> dict[str, Any]:
    current_loans = _require_account_cell(report_set.target, "loans_to_customers", "balance_sheet_date")
    comparison_loans = _require_account_cell(report_set.prior_year, "loans_to_customers", "balance_sheet_date")
    current_general = _require_account_cell(report_set.target, "general_provision", "balance_sheet_date")
    comparison_general = _require_account_cell(report_set.prior_year, "general_provision", "balance_sheet_date")
    current_specific = _require_account_cell(report_set.target, "specific_provision", "balance_sheet_date")
    comparison_specific = _require_account_cell(report_set.prior_year, "specific_provision", "balance_sheet_date")
    loan_growth = _percentage_growth(current_loans["value"], comparison_loans["value"])
    general_growth = _percentage_growth(current_general["value"], comparison_general["value"])
    specific_growth = _percentage_growth(current_specific["value"], comparison_specific["value"])
    combined_current = current_general["value"] + current_specific["value"]
    combined_comparison = comparison_general["value"] + comparison_specific["value"]
    combined_growth = _percentage_growth(combined_current, combined_comparison)
    growth_gap = round(loan_growth - combined_growth, 2)
    threshold_value = 10.0
    flag = growth_gap > threshold_value

    return {
        "tool_result_id": f"TOOL_{report_set.target.report_id}_PROVISION_MOVE_001",
        "report_id": report_set.target.report_id,
        "tool_name": "credit_institution_provision_movement_tool",
        "tool_version": "v1.0",
        "tool_category": "quantitative",
        "risk_category": "expense_liability_understatement_risk",
        "analysis_scope": "company_report_set",
        "period_basis": "balance_sheet_date",
        "signal_id": "provision_growth_lags_risk_assets" if flag else "provision_growth_aligned_with_risk_assets",
        "metric": {
            "metric_name": "loan_growth_minus_provision_growth_pct_points",
            "value": growth_gap,
            "unit": "percentage_points",
            "period_current": report_set.target.metadata["period"],
            "period_comparison": report_set.prior_year.metadata["period"],
            "direction": "increase" if growth_gap > 0 else "decrease" if growth_gap < 0 else "unchanged",
            "secondary_values": {
                "loan_growth_pct": loan_growth,
                "combined_provision_growth_pct": combined_growth,
                "general_provision_growth_pct": general_growth,
                "specific_provision_growth_pct": specific_growth,
            },
        },
        "threshold": {
            "threshold_type": "gap_greater_than",
            "value": threshold_value,
            "unit": "percentage_points",
            "basis": "configured_default_v1",
            "config_version": "credit_institution_v1",
            "description": "Review when loan growth materially exceeds report-visible provision movement.",
        },
        "flag": flag,
        "strength": "strong" if flag else "not_applicable",
        "finding_summary": (
            f"Loans grew {loan_growth:.2f}% while combined provisions grew {combined_growth:.2f}%, "
            f"a {growth_gap:.2f} percentage-point movement gap."
        ),
        "evidence_refs": [
            _cell_evidence_ref(report_set.target.report_id, current_loans["cell_id"]),
            _cell_evidence_ref(report_set.prior_year.report_id, comparison_loans["cell_id"]),
            _cell_evidence_ref(report_set.target.report_id, current_general["cell_id"]),
            _cell_evidence_ref(report_set.prior_year.report_id, comparison_general["cell_id"]),
            _cell_evidence_ref(report_set.target.report_id, current_specific["cell_id"]),
            _cell_evidence_ref(report_set.prior_year.report_id, comparison_specific["cell_id"]),
        ],
        "calculation": {
            "formula": "loan_growth_pct - combined_provision_growth_pct",
            "inputs": [
                {
                    "name": "current_loans_to_customers",
                    "value": current_loans["value"],
                    "cell_id": current_loans["cell_id"],
                    "report_id": report_set.target.report_id,
                    "local_evidence_id": current_loans["cell_id"],
                    "period_basis": "balance_sheet_date",
                },
                {
                    "name": "comparison_loans_to_customers",
                    "value": comparison_loans["value"],
                    "cell_id": comparison_loans["cell_id"],
                    "report_id": report_set.prior_year.report_id,
                    "local_evidence_id": comparison_loans["cell_id"],
                    "period_basis": "balance_sheet_date",
                },
                {
                    "name": "current_general_provision",
                    "value": current_general["value"],
                    "cell_id": current_general["cell_id"],
                    "report_id": report_set.target.report_id,
                    "local_evidence_id": current_general["cell_id"],
                    "period_basis": "balance_sheet_date",
                },
                {
                    "name": "comparison_general_provision",
                    "value": comparison_general["value"],
                    "cell_id": comparison_general["cell_id"],
                    "report_id": report_set.prior_year.report_id,
                    "local_evidence_id": comparison_general["cell_id"],
                    "period_basis": "balance_sheet_date",
                },
                {
                    "name": "current_specific_provision",
                    "value": current_specific["value"],
                    "cell_id": current_specific["cell_id"],
                    "report_id": report_set.target.report_id,
                    "local_evidence_id": current_specific["cell_id"],
                    "period_basis": "balance_sheet_date",
                },
                {
                    "name": "comparison_specific_provision",
                    "value": comparison_specific["value"],
                    "cell_id": comparison_specific["cell_id"],
                    "report_id": report_set.prior_year.report_id,
                    "local_evidence_id": comparison_specific["cell_id"],
                    "period_basis": "balance_sheet_date",
                },
            ],
            "computed_value": growth_gap,
            "secondary_values": {
                "loan_growth_pct": loan_growth,
                "combined_provision_growth_pct": combined_growth,
                "general_provision_growth_pct": general_growth,
                "specific_provision_growth_pct": specific_growth,
            },
            "rounding": "rounded_to_2_decimal_places",
        },
        "limitations": [
            "The tool compares report-internal movement signals only.",
            "The tool does not enforce provision-rate formulas.",
            "External context is excluded in v1.",
        ],
        "status": "completed",
    }


def _build_ctck_margin_book_quality_finding(report_set: CompanyReportSet) -> dict[str, Any]:
    current_margin = _require_account_cell(report_set.target, "margin_lending", "balance_sheet_date")
    comparison_margin = _require_account_cell(report_set.prior_year, "margin_lending", "balance_sheet_date")
    current_impairment = _require_account_cell(report_set.target, "margin_impairment", "balance_sheet_date")
    comparison_impairment = _require_account_cell(report_set.prior_year, "margin_impairment", "balance_sheet_date")
    margin_growth = _percentage_growth_or_none(current_margin["value"], comparison_margin["value"])
    impairment_growth = _percentage_growth_or_none(current_impairment["value"], comparison_impairment["value"])
    coverage_current = _percentage_ratio_or_none(current_impairment["value"], current_margin["value"])
    coverage_comparison = _percentage_ratio_or_none(comparison_impairment["value"], comparison_margin["value"])
    coverage_change = (
        round(coverage_current - coverage_comparison, 2)
        if coverage_current is not None and coverage_comparison is not None
        else None
    )
    threshold_value = 15.0
    metric_value = (
        round(margin_growth - impairment_growth, 2)
        if margin_growth is not None and impairment_growth is not None
        else None
    )
    flag = (
        metric_value is not None
        and coverage_change is not None
        and metric_value > threshold_value
        and coverage_change < 0
    )
    signal_id = "ctck_margin_book_provision_gap" if flag else "ctck_margin_book_coverage_aligned"
    if metric_value is None or coverage_change is None:
        signal_id = "ctck_margin_book_growth_not_computable"
    return {
        "tool_result_id": f"TOOL_{report_set.target.report_id}_CTCK_MARGIN_001",
        "report_id": report_set.target.report_id,
        "tool_name": "ctck_margin_book_quality_tool",
        "tool_version": "v1.0",
        "tool_category": "quantitative",
        "risk_category": "receivables_credit_quality_risk",
        "analysis_scope": "company_report_set",
        "period_basis": "balance_sheet_date",
        "signal_id": signal_id,
        "metric": {
            "metric_name": "margin_growth_minus_impairment_growth_pct_points",
            "value": metric_value,
            "unit": "percentage_points",
            "period_current": report_set.target.metadata["period"],
            "period_comparison": report_set.prior_year.metadata["period"],
            "direction": "increase",
            "secondary_values": {
                "margin_lending_growth_pct": margin_growth,
                "margin_impairment_growth_pct": impairment_growth,
                "margin_impairment_coverage_current_pct": coverage_current,
                "margin_impairment_coverage_comparison_pct": coverage_comparison,
                "coverage_change_pct_points": coverage_change,
            },
        },
        "threshold": {
            "threshold_type": "gap_greater_than",
            "value": threshold_value,
            "unit": "percentage_points",
            "basis": "configured_default_v1",
            "config_version": "securities_ctck_v1",
            "description": "Review when CTCK margin lending grows materially faster than impairment and coverage declines.",
        },
        "flag": flag,
        "strength": "strong" if flag else "not_applicable",
        "finding_summary": (
            _ctck_margin_book_summary(
                margin_growth=margin_growth,
                impairment_growth=impairment_growth,
                coverage_change=coverage_change,
            )
        ),
        "evidence_refs": [
            _cell_evidence_ref(report_set.target.report_id, current_margin["cell_id"]),
            _cell_evidence_ref(report_set.prior_year.report_id, comparison_margin["cell_id"]),
            _cell_evidence_ref(report_set.target.report_id, current_impairment["cell_id"]),
            _cell_evidence_ref(report_set.prior_year.report_id, comparison_impairment["cell_id"]),
        ],
        "calculation": {
            "formula": "margin_lending_growth_pct - margin_impairment_growth_pct with coverage movement",
            "inputs": [
                {"name": "current_margin_lending", "value": current_margin["value"], "cell_id": current_margin["cell_id"], "report_id": report_set.target.report_id, "local_evidence_id": current_margin["cell_id"], "period_basis": "balance_sheet_date"},
                {"name": "comparison_margin_lending", "value": comparison_margin["value"], "cell_id": comparison_margin["cell_id"], "report_id": report_set.prior_year.report_id, "local_evidence_id": comparison_margin["cell_id"], "period_basis": "balance_sheet_date"},
                {"name": "current_margin_impairment", "value": current_impairment["value"], "cell_id": current_impairment["cell_id"], "report_id": report_set.target.report_id, "local_evidence_id": current_impairment["cell_id"], "period_basis": "balance_sheet_date"},
                {"name": "comparison_margin_impairment", "value": comparison_impairment["value"], "cell_id": comparison_impairment["cell_id"], "report_id": report_set.prior_year.report_id, "local_evidence_id": comparison_impairment["cell_id"], "period_basis": "balance_sheet_date"},
            ],
            "computed_value": metric_value,
            "secondary_values": {"coverage_change_pct_points": coverage_change},
            "rounding": "rounded_to_2_decimal_places",
        },
        "limitations": [
            "The tool uses fixture-visible margin-book evidence only.",
            "Growth is not computed when a comparison-period denominator is zero.",
            "External context is excluded in v1.",
        ],
        "status": "completed",
    }


def _build_ctck_trading_book_valuation_bridge_finding(report_set: CompanyReportSet) -> dict[str, Any]:
    current_fvtpl = _require_account_cell(report_set.target, "fvtpl_assets", "balance_sheet_date")
    comparison_fvtpl = _require_account_cell(report_set.prior_year, "fvtpl_assets", "balance_sheet_date")
    unrealized = _require_account_cell(report_set.target, "fvtpl_unrealized_gain", "year_to_date")
    fvtpl_growth = _percentage_growth(current_fvtpl["value"], comparison_fvtpl["value"])
    concentration_ratio = round(current_fvtpl["value"] / sum(_require_account_cell(report_set.target, account, "balance_sheet_date")["value"] for account in ["fvtpl_assets", "afs_assets", "htm_assets"]) * 100, 2)
    weak_disclosure = any("concentrated" in note["text"].lower() and "no issuer-level" in note["text"].lower() for note in report_set.target.raw["notes"])
    flag = fvtpl_growth > 20.0 and concentration_ratio > 60.0 and weak_disclosure
    note_refs = [
        _span_evidence_ref(report_set.target.report_id, "note_span", note["note_id"])
        for note in report_set.target.raw["notes"]
        if "concentrated" in note["text"].lower() or "issuer-level" in note["text"].lower()
    ]
    return {
        "tool_result_id": f"TOOL_{report_set.target.report_id}_CTCK_TRADING_VAL_001",
        "report_id": report_set.target.report_id,
        "tool_name": "ctck_trading_book_valuation_bridge_tool",
        "tool_version": "v1.0",
        "tool_category": "consistency",
        "risk_category": "asset_quality_valuation_risk",
        "analysis_scope": "company_report_set",
        "period_basis": "balance_sheet_date",
        "signal_id": "ctck_trading_book_valuation_concentration_with_weak_disclosure" if flag else "ctck_trading_book_valuation_context_only",
        "metric": {
            "metric_name": "fvtpl_growth_and_concentration",
            "value": fvtpl_growth,
            "unit": "percent",
            "period_current": report_set.target.metadata["period"],
            "period_comparison": report_set.prior_year.metadata["period"],
            "direction": "increase",
            "secondary_values": {"fvtpl_concentration_pct": concentration_ratio, "unrealized_fvtpl_gain": unrealized["value"], "weak_disclosure": weak_disclosure},
        },
        "threshold": {
            "threshold_type": "boolean_rule",
            "value": True,
            "unit": "boolean",
            "basis": "configured_default_v1",
            "config_version": "securities_ctck_v1",
            "description": "Review FVTPL growth only with report-internal valuation concentration or disclosure concern.",
        },
        "flag": flag,
        "strength": "strong" if flag else "not_applicable",
        "finding_summary": (
            f"FVTPL assets grew {fvtpl_growth:.2f}% and represented {concentration_ratio:.2f}% of visible investment assets, with weak issuer-level valuation disclosure."
            if flag
            else f"FVTPL assets grew {fvtpl_growth:.2f}%, but no corroborating valuation disclosure concern was identified."
        ),
        "evidence_refs": [
            _cell_evidence_ref(report_set.target.report_id, current_fvtpl["cell_id"]),
            _cell_evidence_ref(report_set.prior_year.report_id, comparison_fvtpl["cell_id"]),
            _cell_evidence_ref(report_set.target.report_id, unrealized["cell_id"]),
        ] + note_refs,
        "calculation": {
            "formula": "fvtpl_growth_pct plus visible FVTPL concentration and disclosure check",
            "inputs": [
                {"name": "current_fvtpl_assets", "value": current_fvtpl["value"], "cell_id": current_fvtpl["cell_id"], "report_id": report_set.target.report_id, "local_evidence_id": current_fvtpl["cell_id"], "period_basis": "balance_sheet_date"},
                {"name": "comparison_fvtpl_assets", "value": comparison_fvtpl["value"], "cell_id": comparison_fvtpl["cell_id"], "report_id": report_set.prior_year.report_id, "local_evidence_id": comparison_fvtpl["cell_id"], "period_basis": "balance_sheet_date"},
            ],
            "computed_value": fvtpl_growth,
            "secondary_values": {"fvtpl_concentration_pct": concentration_ratio},
            "rounding": "rounded_to_2_decimal_places",
        },
        "limitations": ["FVTPL volatility alone is treated as context only.", "External market context is excluded in v1."],
        "status": "completed",
    }


def _build_ctck_earnings_cash_bridge_finding(report_set: CompanyReportSet) -> dict[str, Any]:
    profit = _require_account_cell(report_set.target, "profit_after_tax", "year_to_date")
    cfo = _require_account_cell(report_set.target, "operating_cash_flow", "year_to_date")
    unrealized = _require_account_cell(report_set.target, "fvtpl_unrealized_gain", "year_to_date")
    fvtpl_share = round(unrealized["value"] / abs(profit["value"]) * 100, 2)
    flag = profit["value"] > 0 and cfo["value"] < 0 and fvtpl_share > 30.0
    return {
        "tool_result_id": f"TOOL_{report_set.target.report_id}_CTCK_EARN_CASH_001",
        "report_id": report_set.target.report_id,
        "tool_name": "ctck_earnings_cash_bridge_tool",
        "tool_version": "v1.0",
        "tool_category": "quantitative",
        "risk_category": "earnings_cashflow_mismatch",
        "analysis_scope": "company_report_set",
        "period_basis": "year_to_date",
        "signal_id": "ctck_profit_supported_by_noncash_fvtpl_with_weak_cash_support" if flag else "ctck_cash_support_context_only",
        "metric": {
            "metric_name": "fvtpl_unrealized_gain_to_profit_pct",
            "value": fvtpl_share,
            "unit": "percent",
            "period_current": report_set.target.metadata["period"],
            "period_comparison": report_set.prior_year.metadata["period"],
            "direction": "increase",
            "secondary_values": {"profit_after_tax": profit["value"], "operating_cash_flow": cfo["value"], "fvtpl_unrealized_gain": unrealized["value"]},
        },
        "threshold": {
            "threshold_type": "greater_than",
            "value": 30.0,
            "unit": "percent",
            "basis": "configured_default_v1",
            "config_version": "securities_ctck_v1",
            "description": "Review when non-cash FVTPL gains are material to profit and operating cash flow is weak.",
        },
        "flag": flag,
        "strength": "strong" if flag else "not_applicable",
        "finding_summary": (
            f"Unrealized FVTPL gains were {fvtpl_share:.2f}% of profit while operating cash flow was negative."
        ),
        "evidence_refs": [
            _cell_evidence_ref(report_set.target.report_id, profit["cell_id"]),
            _cell_evidence_ref(report_set.target.report_id, cfo["cell_id"]),
            _cell_evidence_ref(report_set.target.report_id, unrealized["cell_id"]),
        ],
        "calculation": {
            "formula": "fvtpl_unrealized_gain / abs(profit_after_tax)",
            "inputs": [
                {"name": "profit_after_tax", "value": profit["value"], "cell_id": profit["cell_id"], "report_id": report_set.target.report_id, "local_evidence_id": profit["cell_id"], "period_basis": "year_to_date"},
                {"name": "operating_cash_flow", "value": cfo["value"], "cell_id": cfo["cell_id"], "report_id": report_set.target.report_id, "local_evidence_id": cfo["cell_id"], "period_basis": "year_to_date"},
                {"name": "fvtpl_unrealized_gain", "value": unrealized["value"], "cell_id": unrealized["cell_id"], "report_id": report_set.target.report_id, "local_evidence_id": unrealized["cell_id"], "period_basis": "year_to_date"},
            ],
            "computed_value": fvtpl_share,
            "rounding": "rounded_to_2_decimal_places",
        },
        "limitations": ["The tool requires report-internal cash support evidence.", "External context is excluded in v1."],
        "status": "completed",
    }


def _build_ctck_disclosure_consistency_finding(report_set: CompanyReportSet) -> dict[str, Any]:
    unrealized = _require_account_cell(report_set.target, "fvtpl_unrealized_gain", "year_to_date")
    return {
        "tool_result_id": f"TOOL_{report_set.target.report_id}_CTCK_DISC_001",
        "report_id": report_set.target.report_id,
        "tool_name": "ctck_disclosure_consistency_tool",
        "tool_version": "v1.0",
        "tool_category": "disclosure",
        "risk_category": "asset_quality_valuation_risk",
        "analysis_scope": "company_report_set",
        "period_basis": "year_to_date",
        "signal_id": "ctck_fvtpl_volatility_context_only",
        "metric": {
            "metric_name": "fvtpl_volatility_standalone_guardrail",
            "value": unrealized["value"],
            "unit": "amount",
            "period_current": report_set.target.metadata["period"],
            "period_comparison": report_set.prior_year.metadata["period"],
            "direction": "increase",
        },
        "threshold": {
            "threshold_type": "not_applicable",
            "value": None,
            "unit": "not_applicable",
            "basis": "configured_default_v1",
            "config_version": "securities_ctck_v1",
            "description": "FVTPL gains or losses are not standalone candidate triggers.",
        },
        "flag": False,
        "strength": "not_applicable",
        "finding_summary": "FVTPL gains and volatility are retained as CTCK context and are not used as a standalone risk candidate.",
        "evidence_refs": [_cell_evidence_ref(report_set.target.report_id, unrealized["cell_id"])],
        "calculation": {"method": "ctck_fvtpl_guardrail_v1", "inputs": [], "computed_value": "context_only"},
        "limitations": ["Candidates require report-internal corroboration beyond FVTPL volatility alone."],
        "status": "completed",
    }


def _build_insurance_premium_receivable_coherence_finding(report_set: CompanyReportSet) -> dict[str, Any]:
    current_premium = _require_tool_account_cell(
        report_set.target,
        "insurance_premium_receivable_coherence_tool",
        "gross_written_premium",
        "year_to_date",
    )
    comparison_premium = _require_tool_account_cell(
        report_set.prior_year,
        "insurance_premium_receivable_coherence_tool",
        "gross_written_premium",
        "year_to_date",
    )
    current_receivables = _require_tool_account_cell(
        report_set.target,
        "insurance_premium_receivable_coherence_tool",
        "premium_receivables",
        "year_to_date",
    )
    comparison_receivables = _require_tool_account_cell(
        report_set.prior_year,
        "insurance_premium_receivable_coherence_tool",
        "premium_receivables",
        "year_to_date",
    )
    premium_growth = _percentage_growth(current_premium["value"], comparison_premium["value"])
    receivables_growth = _percentage_growth(current_receivables["value"], comparison_receivables["value"])
    growth_gap = round(receivables_growth - premium_growth, 2)
    threshold_value = 20.0
    flag = growth_gap > threshold_value
    return {
        "tool_result_id": f"TOOL_{report_set.target.report_id}_INS_PREM_REC_001",
        "report_id": report_set.target.report_id,
        "tool_name": "insurance_premium_receivable_coherence_tool",
        "tool_version": "v1.0",
        "tool_category": "quantitative",
        "risk_category": "receivables_credit_quality_risk",
        "analysis_scope": "company_report_set",
        "period_basis": "year_to_date",
        "signal_id": "premium_receivables_growth_outpaces_premium_growth" if flag else "premium_receivables_growth_coherent",
        "metric": {
            "metric_name": "premium_receivables_growth_minus_premium_growth_pct_points",
            "value": growth_gap,
            "unit": "percent",
            "period_current": report_set.target.metadata["period"],
            "period_comparison": report_set.prior_year.metadata["period"],
            "direction": "increase" if growth_gap > 0 else "decrease" if growth_gap < 0 else "unchanged",
            "secondary_values": {
                "premium_growth_pct": premium_growth,
                "premium_receivables_growth_pct": receivables_growth,
            },
        },
        "threshold": {
            "threshold_type": "greater_than",
            "value": threshold_value,
            "unit": "percentage_points",
            "basis": "configured_default_v1",
            "config_version": "insurance_v1",
            "description": "Review when premium receivables growth materially exceeds written premium growth.",
        },
        "flag": flag,
        "strength": "strong" if flag else "not_applicable",
        "finding_summary": (
            f"Premium receivables growth exceeded written premium growth by {growth_gap:.2f} percentage points."
        ),
        "evidence_refs": [
            _cell_evidence_ref(report_set.target.report_id, current_premium["cell_id"]),
            _cell_evidence_ref(report_set.prior_year.report_id, comparison_premium["cell_id"]),
            _cell_evidence_ref(report_set.target.report_id, current_receivables["cell_id"]),
            _cell_evidence_ref(report_set.prior_year.report_id, comparison_receivables["cell_id"]),
        ],
        "calculation": {
            "formula": "premium_receivables_growth_pct - gross_written_premium_growth_pct",
            "inputs": [
                {"name": "gross_written_premium", "value": current_premium["value"], "cell_id": current_premium["cell_id"]},
                {"name": "comparison_gross_written_premium", "value": comparison_premium["value"], "cell_id": comparison_premium["cell_id"]},
                {"name": "premium_receivables", "value": current_receivables["value"], "cell_id": current_receivables["cell_id"]},
                {"name": "comparison_premium_receivables", "value": comparison_receivables["value"], "cell_id": comparison_receivables["cell_id"]},
            ],
            "computed_value": growth_gap,
            "rounding": "rounded_to_2_decimal_places",
        },
        "limitations": ["The tool uses report-internal premium and receivable movements only.", "External collection context is excluded in v1."],
        "status": "completed",
    }


def _build_insurance_reserve_movement_finding(report_set: CompanyReportSet) -> dict[str, Any]:
    current_premium = _require_account_cell(report_set.target, "gross_written_premium", "year_to_date")
    comparison_premium = _require_account_cell(report_set.prior_year, "gross_written_premium", "year_to_date")
    current_claims = _require_account_cell(report_set.target, "claims_expense", "year_to_date")
    comparison_claims = _require_account_cell(report_set.prior_year, "claims_expense", "year_to_date")
    current_reserves = _require_account_cell(report_set.target, "insurance_reserves", "year_to_date")
    comparison_reserves = _require_account_cell(report_set.prior_year, "insurance_reserves", "year_to_date")
    premium_growth = _percentage_growth(current_premium["value"], comparison_premium["value"])
    claims_growth = _percentage_growth(current_claims["value"], comparison_claims["value"])
    reserve_growth = _percentage_growth(current_reserves["value"], comparison_reserves["value"])
    exposure_growth = round((premium_growth + claims_growth) / 2, 2)
    lag = round(exposure_growth - reserve_growth, 2)
    reserve_movement_ratio = round(reserve_growth / exposure_growth, 2) if exposure_growth else 0.0
    threshold_value = 20.0
    flag = lag > threshold_value
    return {
        "tool_result_id": f"TOOL_{report_set.target.report_id}_INS_RESERVE_001",
        "report_id": report_set.target.report_id,
        "tool_name": "insurance_reserve_movement_tool",
        "tool_version": "v1.0",
        "tool_category": "quantitative",
        "risk_category": "expense_liability_understatement_risk",
        "analysis_scope": "company_report_set",
        "period_basis": "year_to_date",
        "signal_id": "claims_reserve_growth_lags_claims_and_premium_exposure" if flag else "reserve_movement_coherent",
        "metric": {
            "metric_name": "exposure_growth_minus_reserve_growth_pct_points",
            "value": lag,
            "unit": "percent",
            "period_current": report_set.target.metadata["period"],
            "period_comparison": report_set.prior_year.metadata["period"],
            "direction": "increase" if lag > 0 else "decrease" if lag < 0 else "unchanged",
            "secondary_values": {
                "premium_growth_pct": premium_growth,
                "claims_growth_pct": claims_growth,
                "reserve_growth_pct": reserve_growth,
                "reserve_movement_ratio": reserve_movement_ratio,
            },
        },
        "threshold": {
            "threshold_type": "greater_than",
            "value": threshold_value,
            "unit": "percentage_points",
            "basis": "configured_default_v1",
            "config_version": "insurance_v1",
            "description": "Review observable reserve movement only when it lags report-visible premium and claims movement.",
        },
        "flag": flag,
        "strength": "strong" if flag else "not_applicable",
        "finding_summary": (
            f"Report-visible claims and premium exposure growth exceeded reserve growth by {lag:.2f} percentage points."
        ),
        "evidence_refs": [
            _cell_evidence_ref(report_set.target.report_id, current_premium["cell_id"]),
            _cell_evidence_ref(report_set.prior_year.report_id, comparison_premium["cell_id"]),
            _cell_evidence_ref(report_set.target.report_id, current_claims["cell_id"]),
            _cell_evidence_ref(report_set.prior_year.report_id, comparison_claims["cell_id"]),
            _cell_evidence_ref(report_set.target.report_id, current_reserves["cell_id"]),
            _cell_evidence_ref(report_set.prior_year.report_id, comparison_reserves["cell_id"]),
        ]
        + [
            _span_evidence_ref(report_set.target.report_id, "note_span", "NOTE_INS_RESERVE_2025_Q3")
        ],
        "calculation": {
            "formula": "average(premium_growth_pct, claims_growth_pct) - reserve_growth_pct",
            "inputs": [
                {"name": "gross_written_premium", "value": current_premium["value"], "cell_id": current_premium["cell_id"]},
                {"name": "claims_expense", "value": current_claims["value"], "cell_id": current_claims["cell_id"]},
                {"name": "insurance_reserves", "value": current_reserves["value"], "cell_id": current_reserves["cell_id"]},
            ],
            "computed_value": lag,
            "rounding": "rounded_to_2_decimal_places",
        },
        "limitations": ["This is an observable movement and coherence check only.", "The tool does not assess actuarial assumptions or model sufficiency."],
        "status": "completed",
    }


def _build_insurance_reinsurance_balance_finding(report_set: CompanyReportSet) -> dict[str, Any]:
    current_recoverables = _require_account_cell(report_set.target, "reinsurance_recoverables", "year_to_date")
    comparison_recoverables = _require_account_cell(report_set.prior_year, "reinsurance_recoverables", "year_to_date")
    current_payables = _require_account_cell(report_set.target, "reinsurance_payables", "year_to_date")
    comparison_payables = _require_account_cell(report_set.prior_year, "reinsurance_payables", "year_to_date")
    current_cfo = _require_account_cell(report_set.target, "operating_cash_flow", "year_to_date")
    recoverables_growth = _percentage_growth(current_recoverables["value"], comparison_recoverables["value"])
    payables_growth = _percentage_growth(current_payables["value"], comparison_payables["value"])
    net_gap = round(recoverables_growth - payables_growth, 2)
    flag = recoverables_growth > 30.0 and current_cfo["value"] < 0
    return {
        "tool_result_id": f"TOOL_{report_set.target.report_id}_INS_REINS_001",
        "report_id": report_set.target.report_id,
        "tool_name": "insurance_reinsurance_balance_tool",
        "tool_version": "v1.0",
        "tool_category": "quantitative",
        "risk_category": "earnings_cashflow_mismatch",
        "analysis_scope": "company_report_set",
        "period_basis": "year_to_date",
        "signal_id": "reinsurance_recoverables_expand_with_weak_cash_support" if flag else "reinsurance_balance_cash_support_not_flagged",
        "metric": {
            "metric_name": "reinsurance_recoverables_growth_with_cash_support",
            "value": recoverables_growth,
            "unit": "percent",
            "period_current": report_set.target.metadata["period"],
            "period_comparison": report_set.prior_year.metadata["period"],
            "direction": "increase" if recoverables_growth > 0 else "decrease" if recoverables_growth < 0 else "unchanged",
            "secondary_values": {
                "reinsurance_payables_growth_pct": payables_growth,
                "recoverables_minus_payables_growth_pct_points": net_gap,
                "operating_cash_flow": current_cfo["value"],
            },
        },
        "threshold": {
            "threshold_type": "compound",
            "value": 30.0,
            "unit": "percent",
            "basis": "configured_default_v1",
            "config_version": "insurance_v1",
            "description": "Review when reinsurance recoverables expand materially and operating cash flow is negative.",
        },
        "flag": flag,
        "strength": "strong" if flag else "not_applicable",
        "finding_summary": (
            f"Reinsurance recoverables grew {recoverables_growth:.2f}% while operating cash flow was negative."
        ),
        "evidence_refs": [
            _cell_evidence_ref(report_set.target.report_id, current_recoverables["cell_id"]),
            _cell_evidence_ref(report_set.prior_year.report_id, comparison_recoverables["cell_id"]),
            _cell_evidence_ref(report_set.target.report_id, current_payables["cell_id"]),
            _cell_evidence_ref(report_set.prior_year.report_id, comparison_payables["cell_id"]),
            _cell_evidence_ref(report_set.target.report_id, current_cfo["cell_id"]),
            _span_evidence_ref(report_set.target.report_id, "note_span", "NOTE_INS_REINSURANCE_2025_Q3"),
        ],
        "calculation": {
            "formula": "reinsurance_recoverables_growth_pct with operating_cash_flow support check",
            "inputs": [
                {"name": "reinsurance_recoverables", "value": current_recoverables["value"], "cell_id": current_recoverables["cell_id"]},
                {"name": "reinsurance_payables", "value": current_payables["value"], "cell_id": current_payables["cell_id"]},
                {"name": "operating_cash_flow", "value": current_cfo["value"], "cell_id": current_cfo["cell_id"]},
            ],
            "computed_value": recoverables_growth,
            "rounding": "rounded_to_2_decimal_places",
        },
        "limitations": ["The tool uses only report-internal reinsurance and cash-flow evidence.", "External counterparty context is excluded in v1."],
        "status": "completed",
    }


def _find_account_cell(
    report_memory: ReportMemory,
    account: str,
    period_basis: str = "quarter",
) -> dict[str, Any] | None:
    for table in report_memory.raw["tables"]:
        if table.get("period_basis") != period_basis:
            continue
        for row in table["rows"]:
            if _row_matches_account(row, account):
                cell = row["cells"][0]
                return {"cell_id": cell["cell_id"], "value": cell["value"]}
    return None


def _find_tool_account_cell(
    report_memory: ReportMemory,
    tool_name: str,
    account: str,
    default_period_basis: str,
) -> dict[str, Any] | None:
    for period_basis in _tool_account_period_bases(tool_name, account, default_period_basis):
        cell = _find_account_cell(report_memory, account, period_basis)
        if cell is not None:
            return {**cell, "period_basis": period_basis}
    return None


def _require_tool_account_cell(
    report_memory: ReportMemory,
    tool_name: str,
    account: str,
    default_period_basis: str,
) -> dict[str, Any]:
    cell = _find_tool_account_cell(report_memory, tool_name, account, default_period_basis)
    if cell is None:
        period_bases = ", ".join(_tool_account_period_bases(tool_name, account, default_period_basis))
        raise ValueError(f"Missing required account cell for {account} with period_basis {period_bases}")
    return cell


def _tool_account_period_bases(tool_name: str, account: str, default_period_basis: str) -> list[str]:
    if tool_name == "insurance_premium_receivable_coherence_tool" and account == "premium_receivables":
        return ["balance_sheet_date", "year_to_date"]
    return [default_period_basis]


def _disclosure_gating_status(report_set: CompanyReportSet, tool_name: str) -> dict[str, str] | None:
    if tool_name == "standard_corporate_disclosure_consistency_tool":
        if not report_set.target.raw["notes"] or not report_set.target.raw["variance_explanations"]:
            return {
                "status": "skipped_missing_disclosure_context",
                "reason_code": "missing_notes_or_variance_explanation",
                "reason": "Disclosure consistency requires extracted notes and variance explanation context.",
            }
    elif tool_name == "standard_corporate_variance_explanation_quality_tool":
        material_profit_movement = _has_material_profit_movement(report_set)
        if not material_profit_movement:
            return {
                "status": "skipped_no_material_movement",
                "reason_code": "no_material_profit_movement",
                "reason": "Variance explanation quality is skipped because no material profit movement was detected.",
            }
        if not report_set.target.raw["variance_explanations"]:
            return {
                "status": "skipped_missing_disclosure_context",
                "reason_code": "missing_variance_explanation",
                "reason": "Variance explanation quality requires confidently extracted variance explanation context.",
            }
    elif tool_name == "standard_corporate_related_party_exposure_tool":
        return {
            "status": "skipped_missing_disclosure_context",
            "reason_code": "related_party_extraction_not_available",
            "reason": "Related-party exposure requires structured related-party extraction, which is unavailable in this fixture.",
        }
    elif tool_name == "standard_corporate_accounting_policy_change_tool":
        return {
            "status": "skipped_missing_disclosure_context",
            "reason_code": "accounting_policy_change_extraction_not_available",
            "reason": "Accounting-policy change analysis requires structured policy-change extraction, which is unavailable in this fixture.",
        }
    return None


def _has_material_profit_movement(report_set: CompanyReportSet) -> bool:
    current_profit = _find_account_cell(report_set.target, "profit_after_tax", "year_to_date")
    comparison_profit = _find_account_cell(report_set.prior_year, "profit_after_tax", "year_to_date")
    if current_profit is None or comparison_profit is None:
        return False
    return abs(_percentage_growth(current_profit["value"], comparison_profit["value"])) > 20.0


def _require_account_cell(
    report_memory: ReportMemory,
    account: str,
    period_basis: str = "quarter",
) -> dict[str, Any]:
    cell = _find_account_cell(report_memory, account, period_basis)
    if cell is None:
        raise ValueError(f"Missing required account evidence for {account}")
    return cell


def _row_matches_account(row: dict[str, Any], account: str) -> bool:
    label = row["label"].lower()
    if account == "revenue":
        return row.get("account_code") == "01" or label == "revenue"
    if account == "trade_receivables":
        return row.get("account_code") == "131" or "trade receivable" in label
    if account == "profit_after_tax":
        return row.get("account_code") == "60" or label == "profit after tax"
    if account == "operating_cash_flow":
        return row.get("account_code") == "20" or "operating cash flow" in label
    if account == "loans_to_customers":
        return row.get("standard_account") == account or row.get("account_code") == "LOANS_CUSTOMERS"
    if account.startswith("loan_group_"):
        group_number = account.rsplit("_", maxsplit=1)[1]
        return row.get("standard_account") == account or row.get("account_code") == f"LOAN_GROUP_{group_number}"
    if account == "general_provision":
        return row.get("standard_account") == account or row.get("account_code") == "GENERAL_PROVISION"
    if account == "specific_provision":
        return row.get("standard_account") == account or row.get("account_code") == "SPECIFIC_PROVISION"
    if account == "margin_lending":
        return row.get("standard_account") == account or row.get("account_code") == "MARGIN_LENDING"
    if account == "customer_advances":
        return row.get("standard_account") == account or row.get("account_code") == "CUSTOMER_ADVANCES"
    if account == "margin_impairment":
        return row.get("standard_account") == account or row.get("account_code") == "MARGIN_IMPAIRMENT"
    if account == "fvtpl_assets":
        return row.get("standard_account") == account or row.get("account_code") == "FVTPL_ASSETS"
    if account == "afs_assets":
        return row.get("standard_account") == account or row.get("account_code") == "AFS_ASSETS"
    if account == "htm_assets":
        return row.get("standard_account") == account or row.get("account_code") == "HTM_ASSETS"
    if account == "fvtpl_unrealized_gain":
        return row.get("standard_account") == account or row.get("account_code") == "FVTPL_UNREALIZED_GAIN"
    if account == "gross_written_premium":
        return row.get("standard_account") == account or row.get("account_code") == "GROSS_WRITTEN_PREMIUM"
    if account == "premium_receivables":
        return row.get("standard_account") == account or row.get("account_code") == "PREMIUM_RECEIVABLES"
    if account == "claims_expense":
        return row.get("standard_account") == account or row.get("account_code") == "CLAIMS_EXPENSE"
    if account == "insurance_reserves":
        return row.get("standard_account") == account or row.get("account_code") == "INSURANCE_RESERVES"
    if account == "reinsurance_recoverables":
        return row.get("standard_account") == account or row.get("account_code") == "REINSURANCE_RECOVERABLES"
    if account == "reinsurance_payables":
        return row.get("standard_account") == account or row.get("account_code") == "REINSURANCE_PAYABLES"
    return False


def _percentage_growth(current_value: float, comparison_value: float) -> float:
    if comparison_value == 0:
        raise ValueError("Cannot compute growth against a zero comparison value")
    return round((current_value - comparison_value) / abs(comparison_value) * 100, 2)


def _percentage_growth_or_none(current_value: float, comparison_value: float) -> float | None:
    if comparison_value == 0:
        return None
    return _percentage_growth(current_value, comparison_value)


def _percentage_ratio_or_none(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator * 100, 2)


def _ctck_margin_book_summary(
    *,
    margin_growth: float | None,
    impairment_growth: float | None,
    coverage_change: float | None,
) -> str:
    if margin_growth is None or impairment_growth is None or coverage_change is None:
        return (
            "Margin-book growth or coverage movement was not computed because "
            "one comparison-period denominator is zero."
        )
    return (
        f"Margin lending grew {margin_growth:.2f}% while related impairment grew {impairment_growth:.2f}%, "
        f"and impairment coverage changed by {coverage_change:.2f} percentage points."
    )


def _cell_evidence_ref(report_id: str, cell_id: str) -> dict[str, str]:
    return {
        "evidence_ref_type": "table_cell",
        "ref_id": f"{report_id}:{cell_id}",
        "report_id": report_id,
        "local_evidence_id": cell_id,
        "role": "input",
    }


def _span_evidence_ref(report_id: str, evidence_ref_type: str, local_evidence_id: str) -> dict[str, str]:
    return {
        "evidence_ref_type": evidence_ref_type,
        "ref_id": f"{report_id}:{local_evidence_id}",
        "report_id": report_id,
        "local_evidence_id": local_evidence_id,
        "role": "input",
    }


def _candidate_evidence_refs(linked_findings: list[dict[str, Any]]) -> list[dict[str, str]]:
    return _dedupe_evidence_refs(
        [
            {"evidence_ref_type": "tool_result", "ref_id": finding["tool_result_id"], "role": "required_for_review"}
            for finding in linked_findings
        ]
        + [
            {**evidence_ref, "role": "supporting"}
            for finding in linked_findings
            for evidence_ref in finding["evidence_refs"]
        ]
    )


def _dedupe_evidence_refs(evidence_refs: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = set()
    deduped = []
    for evidence_ref in evidence_refs:
        key = (evidence_ref["evidence_ref_type"], evidence_ref["ref_id"], evidence_ref["role"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(evidence_ref)
    return deduped


class AppendOnlyAuditLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, stage: str, details: dict[str, Any]) -> None:
        event = {
            "sequence": self._next_sequence(),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "details": details,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=True, sort_keys=True) + "\n")

    def _next_sequence(self) -> int:
        if not self.path.exists():
            return 1
        with self.path.open(encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip()) + 1


def _load_fixture_cases() -> list[dict[str, Any]]:
    fixture_path = files("research.fixtures.wave1").joinpath("fixture_spine_cases.json")
    with fixture_path.open(encoding="utf-8") as handle:
        loaded = json.load(handle)
    return [*loaded["report_sets"], _securities_fixture_case(), _insurance_fixture_case()]


def _securities_fixture_case() -> dict[str, Any]:
    return {
        "case_id": "signal_present_securities",
        "case_type": "signal_present",
        "target_report_memory": _securities_report_memory("2025-Q3"),
        "prior_year_report_memory": _securities_report_memory("2024-Q3"),
    }


def _insurance_fixture_case() -> dict[str, Any]:
    return {
        "case_id": "signal_present_insurance",
        "case_type": "signal_present",
        "target_report_memory": _insurance_report_memory("2025-Q3"),
        "prior_year_report_memory": _insurance_report_memory("2024-Q3"),
    }


def _insurance_report_memory(period: str) -> dict[str, Any]:
    year = period[:4]
    is_target = year == "2025"
    report_id = f"VINAINS_SIGNAL_{year}_Q3"
    doc_id = f"DOC_VNI_{year}_Q3_FS_001"
    values = {
        "gross_written_premium": 1_920_000 if is_target else 1_360_000,
        "premium_receivables": 680_000 if is_target else 310_000,
        "claims_expense": 1_140_000 if is_target else 720_000,
        "insurance_reserves": 1_360_000 if is_target else 1_140_000,
        "reinsurance_recoverables": 540_000 if is_target else 260_000,
        "reinsurance_payables": 220_000 if is_target else 180_000,
        "investment_assets": 2_850_000 if is_target else 2_360_000,
        "commission_acquisition_costs": 250_000 if is_target else 190_000,
        "operating_cash_flow": -185_000 if is_target else 126_000,
    }
    rows = [
        _fixture_row("GROSS_WRITTEN_PREMIUM", "gross_written_premium", "revenue_income", "Gross written premium", "Doanh thu phí bảo hiểm gốc", values["gross_written_premium"], period),
        _fixture_row("PREMIUM_RECEIVABLES", "premium_receivables", "receivables_credit", "Premium receivables", "Phải thu phí bảo hiểm", values["premium_receivables"], period),
        _fixture_row("CLAIMS_EXPENSE", "claims_expense", "revenue_income", "Claims expense", "Chi bồi thường bảo hiểm", values["claims_expense"], period),
        _fixture_row("INSURANCE_RESERVES", "insurance_reserves", "expense_liability", "Insurance technical reserves", "Dự phòng nghiệp vụ bảo hiểm", values["insurance_reserves"], period),
        _fixture_row("REINSURANCE_RECOVERABLES", "reinsurance_recoverables", "asset_quality", "Reinsurance recoverables", "Phải thu tái bảo hiểm", values["reinsurance_recoverables"], period),
        _fixture_row("REINSURANCE_PAYABLES", "reinsurance_payables", "expense_liability", "Reinsurance payables", "Phải trả tái bảo hiểm", values["reinsurance_payables"], period),
        _fixture_row("INVESTMENT_ASSETS", "investment_assets", "asset_quality", "Investment assets", "Tài sản đầu tư", values["investment_assets"], period),
        _fixture_row("COMMISSION_ACQUISITION_COSTS", "commission_acquisition_costs", "expense_liability", "Commission and acquisition costs", "Chi phí hoa hồng và khai thác", values["commission_acquisition_costs"], period),
        _fixture_row("20", "operating_cash_flow", "cashflow", "Operating cash flow", "Lưu chuyển tiền thuần từ hoạt động kinh doanh", values["operating_cash_flow"], period),
    ]
    tables = [
        {
            "table_id": f"TBL_INS_YTD_{year}_Q3",
            "section_id": f"SEC_INS_{year}_Q3",
            "title": "Insurance premium, reserve, reinsurance, and cash-flow evidence",
            "period_basis": "year_to_date",
            "source_document_id": doc_id,
            "rows": rows,
        }
    ]
    notes = []
    if is_target:
        notes = [
            {
                "note_id": "NOTE_INS_RESERVE_2025_Q3",
                "section_id": "SEC_INS_NOTES_2025_Q3",
                "note_type": "reserve_movement",
                "title": "Technical reserve movement",
                "source_document_id": doc_id,
                "text": "Technical reserves increased during the period, with reserve movement presented against claims and premium activity.",
                "linked_row_ids": ["ROW_INSURANCE_RESERVES_2025_Q3", "ROW_CLAIMS_EXPENSE_2025_Q3"],
                "linked_cell_ids": ["CELL_INSURANCE_RESERVES_2025_Q3", "CELL_CLAIMS_EXPENSE_2025_Q3"],
                "periods": [period],
                "status": "included",
            },
            {
                "note_id": "NOTE_INS_REINSURANCE_2025_Q3",
                "section_id": "SEC_INS_NOTES_2025_Q3",
                "note_type": "reinsurance_balances",
                "title": "Reinsurance balances",
                "source_document_id": doc_id,
                "text": "Reinsurance recoverables increased with outstanding balances from ceded claims and premiums.",
                "linked_row_ids": ["ROW_REINSURANCE_RECOVERABLES_2025_Q3", "ROW_REINSURANCE_PAYABLES_2025_Q3"],
                "linked_cell_ids": ["CELL_REINSURANCE_RECOVERABLES_2025_Q3", "CELL_REINSURANCE_PAYABLES_2025_Q3"],
                "periods": [period],
                "status": "included",
            },
        ]
    return {
        "report_id": report_id,
        "metadata": {
            "company_name": "Vinains Fixture Insurance Corporation",
            "ticker": "VNI",
            "period": period,
            "report_period_type": "quarterly",
            "report_profile": "insurance",
            "report_basis": "consolidated",
            "business_context_tags": ["insurance_non_life"],
            "insurance_subprofile": "non_life",
            "industry": "insurance",
            "report_assurance_type": "unaudited",
            "currency": "VND",
            "unit": "million_vnd",
            "filing_status": "original",
            "canonical_source_document_id": doc_id,
            "source_file": f"fixtures/VNI_{year}_Q3_fixture.pdf",
            "language": "vi",
            "extraction_method": "hand_authored_reference_v1",
        },
        "sections": [
            {
                "section_id": f"SEC_INS_{year}_Q3",
                "section_type": "income_statement",
                "title": "Insurance statement and balance evidence",
                "status": "included",
                "page_range": None,
                "ignore_reason": None,
            },
            {
                "section_id": f"SEC_INS_NOTES_{year}_Q3",
                "section_type": "notes",
                "title": "Insurance notes",
                "status": "included",
                "page_range": None,
                "ignore_reason": None,
            },
        ],
        "tables": tables,
        "notes": notes,
        "variance_explanations": [],
        "cell_index": {
            cell["cell_id"]: {"table_id": table["table_id"], "row_id": row["row_id"]}
            for table in tables
            for row in table["rows"]
            for cell in row["cells"]
        },
    }


def _securities_report_memory(period: str) -> dict[str, Any]:
    year = period[:4]
    is_target = year == "2025"
    report_id = f"VINASEC_SIGNAL_{year}_Q3"
    doc_id = f"DOC_VNS_{year}_Q3_FS_001"
    values = {
        "margin_lending": 2_250_000 if is_target else 1_420_000,
        "customer_advances": 540_000 if is_target else 460_000,
        "margin_impairment": 42_000 if is_target else 38_000,
        "fvtpl_assets": 1_850_000 if is_target else 1_120_000,
        "afs_assets": 620_000 if is_target else 580_000,
        "htm_assets": 310_000 if is_target else 300_000,
        "fvtpl_unrealized_gain": 210_000 if is_target else 84_000,
        "profit_after_tax": 330_000 if is_target else 240_000,
        "operating_cash_flow": -125_000 if is_target else 98_000,
    }
    rows = [
        _fixture_row("MARGIN_LENDING", "margin_lending", "receivables_credit", "Margin lending", "Cho vay giao dịch ký quỹ", values["margin_lending"], period),
        _fixture_row("CUSTOMER_ADVANCES", "customer_advances", "receivables_credit", "Customer advances and receivables", "Ứng trước và phải thu khách hàng", values["customer_advances"], period),
        _fixture_row("MARGIN_IMPAIRMENT", "margin_impairment", "receivables_credit", "Impairment allowance for margin lending", "Dự phòng suy giảm cho vay ký quỹ", values["margin_impairment"], period),
        _fixture_row("FVTPL_ASSETS", "fvtpl_assets", "asset_quality", "FVTPL financial assets", "Tài sản tài chính ghi nhận thông qua lãi/lỗ", values["fvtpl_assets"], period),
        _fixture_row("AFS_ASSETS", "afs_assets", "asset_quality", "AFS financial assets", "Tài sản tài chính sẵn sàng để bán", values["afs_assets"], period),
        _fixture_row("HTM_ASSETS", "htm_assets", "asset_quality", "HTM investments", "Đầu tư nắm giữ đến ngày đáo hạn", values["htm_assets"], period),
    ]
    ytd_rows = [
        _fixture_row("60", "profit_after_tax", "revenue_income", "Profit after tax", "Lợi nhuận sau thuế", values["profit_after_tax"], period),
        _fixture_row("20", "operating_cash_flow", "cashflow", "Operating cash flow", "Lưu chuyển tiền thuần từ hoạt động kinh doanh", values["operating_cash_flow"], period),
        _fixture_row("FVTPL_UNREALIZED_GAIN", "fvtpl_unrealized_gain", "asset_quality", "Unrealized FVTPL gain", "Lãi chưa thực hiện từ tài sản FVTPL", values["fvtpl_unrealized_gain"], period),
    ]
    tables = [
        {
            "table_id": f"TBL_CTCK_BAL_{year}_Q3",
            "section_id": f"SEC_CTCK_{year}_Q3",
            "title": "CTCK balance sheet and trading book",
            "period_basis": "balance_sheet_date",
            "source_document_id": doc_id,
            "rows": rows,
        },
        {
            "table_id": f"TBL_CTCK_YTD_{year}_Q3",
            "section_id": f"SEC_CTCK_{year}_Q3",
            "title": "CTCK income and cash bridge",
            "period_basis": "year_to_date",
            "source_document_id": doc_id,
            "rows": ytd_rows,
        },
    ]
    notes = []
    if is_target:
        notes = [
            {
                "note_id": "NOTE_CTCK_MARGIN_2025_Q3",
                "section_id": "SEC_CTCK_NOTES_2025_Q3",
                "note_type": "receivables_breakdown",
                "title": "Margin lending and customer receivables",
                "source_document_id": doc_id,
                "text": "Margin lending expanded with customer receivables, while the impairment note gives limited aging detail.",
                "linked_row_ids": ["ROW_MARGIN_LENDING_2025_Q3", "ROW_MARGIN_IMPAIRMENT_2025_Q3"],
                "linked_cell_ids": ["CELL_MARGIN_LENDING_2025_Q3", "CELL_MARGIN_IMPAIRMENT_2025_Q3"],
                "periods": [period],
                "status": "included",
            },
            {
                "note_id": "NOTE_CTCK_FVTPL_2025_Q3",
                "section_id": "SEC_CTCK_NOTES_2025_Q3",
                "note_type": "asset_quality",
                "title": "FVTPL valuation disclosure",
                "source_document_id": doc_id,
                "text": "FVTPL portfolio value is concentrated in several listed shares, with no issuer-level valuation table in this fixture.",
                "linked_row_ids": ["ROW_FVTPL_ASSETS_2025_Q3", "ROW_FVTPL_UNREALIZED_GAIN_2025_Q3"],
                "linked_cell_ids": ["CELL_FVTPL_ASSETS_2025_Q3", "CELL_FVTPL_UNREALIZED_GAIN_2025_Q3"],
                "periods": [period],
                "status": "included",
            },
        ]
    return {
        "report_id": report_id,
        "metadata": {
            "company_name": "Vinasec Fixture Securities JSC",
            "ticker": "VNS",
            "period": period,
            "report_period_type": "quarterly",
            "report_profile": "securities",
            "report_basis": "consolidated",
            "business_context_tags": ["holding_company"],
            "insurance_subprofile": None,
            "industry": "securities",
            "report_assurance_type": "unaudited",
            "currency": "VND",
            "unit": "million_vnd",
            "filing_status": "original",
            "canonical_source_document_id": doc_id,
            "source_file": f"fixtures/VNS_{year}_Q3_fixture.pdf",
            "language": "vi",
            "extraction_method": "hand_authored_reference_v1",
        },
        "sections": [
            {
                "section_id": f"SEC_CTCK_{year}_Q3",
                "section_type": "balance_sheet",
                "title": "B01a-CTCK style consolidated report",
                "status": "included",
                "page_range": None,
                "ignore_reason": None,
            },
            {
                "section_id": f"SEC_CTCK_NOTES_{year}_Q3",
                "section_type": "notes",
                "title": "CTCK notes",
                "status": "included",
                "page_range": None,
                "ignore_reason": None,
            },
        ],
        "tables": tables,
        "notes": notes,
        "variance_explanations": [],
        "cell_index": {
            cell["cell_id"]: {"table_id": table["table_id"], "row_id": row["row_id"]}
            for table in tables
            for row in table["rows"]
            for cell in row["cells"]
        },
    }


def _fixture_row(
    account_code: str,
    standard_account: str,
    account_group: str,
    label: str,
    original_label: str,
    value: int,
    period: str,
) -> dict[str, Any]:
    cell_id = f"CELL_{standard_account.upper()}_{period.replace('-', '_')}"
    return {
        "row_id": f"ROW_{standard_account.upper()}_{period.replace('-', '_')}",
        "account_code": account_code,
        "standard_account": standard_account,
        "account_group": account_group,
        "label": label,
        "original_label": original_label,
        "cells": [{"cell_id": cell_id, "period": period, "value": value, "unit": "million_vnd"}],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Wave 1 fixture spine harness.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/wave1_smoke"))
    args = parser.parse_args()

    result = run_fixture_spine(args.output_dir)
    print(
        json.dumps(
            {
                "status": result.status,
                "audit_log_path": str(result.audit_log_path),
                "exit_demo_manifest_path": str(result.exit_demo_manifest_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
