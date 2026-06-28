from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from research.detector_contract_validation import (
    enrich_detector_packet_evidence_roles,
    validate_detector_packet,
)
from research.synthetic_detector_assessment_gate import (
    CORROBORATION_BALANCED_JUDGE_PROMPT_VERSION,
    CORROBORATION_BALANCED_TEACHER_PROMPT_VERSION,
    CORROBORATION_STRICT_JUDGE_PROMPT_VERSION,
    CORROBORATION_STRICT_TEACHER_PROMPT_VERSION,
)
from research.synthetic_raw_candidate_validation import validate_synthetic_raw_candidate
from research.grounded_synthetic_packet_generator import (
    _reserved_real_manual_identities,
    _rows_by_profile_role,
    _synthetic_report_id,
    _validate_clean_report_not_reserved,
    _validate_generated_packet,
)
from research.sft_exporter import _record_to_chat, detector_sft_system_prompt


ARTIFACT_CONTRACT_VERSION = "detector_cashflow_evidence_structure_dataset_v1"
RISK_CATEGORY = "earnings_cashflow_mismatch"
DEFAULT_CASE_SET = "v2"
SCENARIO_IDS = {
    "v1": "earnings_cashflow_evidence_structure_v1",
    "v2": "earnings_cashflow_evidence_structure_v2",
}
SYSTEM_PROMPT_VERSION = "v2_evidence_bundle"


@dataclass(frozen=True)
class EvidenceStructureCase:
    case_id: str
    suffix: str
    magnitude: float
    support_level: str
    audit_status: str
    audit_independent_corroboration: bool
    corroboration_kind: str
    disagreement_case: bool
    primary_trigger_present: bool = True
    cashflow_input_missing: bool = False


V1_CASE_SPECS = (
    EvidenceStructureCase(
        case_id="isolated_trigger_weak",
        suffix="ISO_W",
        magnitude=2.50,
        support_level="weakly_supported",
        audit_status="isolated",
        audit_independent_corroboration=False,
        corroboration_kind="none",
        disagreement_case=False,
    ),
    EvidenceStructureCase(
        case_id="collection_delay_supported",
        suffix="DISC_S",
        magnitude=2.50,
        support_level="supported",
        audit_status="isolated",
        audit_independent_corroboration=False,
        corroboration_kind="collection_delay_disclosure",
        disagreement_case=True,
    ),
    EvidenceStructureCase(
        case_id="balance_growth_supported",
        suffix="BAL_S",
        magnitude=2.50,
        support_level="supported",
        audit_status="corroborated",
        audit_independent_corroboration=True,
        corroboration_kind="strong_balance_growth",
        disagreement_case=False,
    ),
    EvidenceStructureCase(
        case_id="corroborated_mitigated_weak",
        suffix="MIT_W",
        magnitude=1.50,
        support_level="weakly_supported",
        audit_status="corroborated_but_mitigated",
        audit_independent_corroboration=True,
        corroboration_kind="mitigated_balance_growth",
        disagreement_case=True,
    ),
)
V2_CASE_SPECS = (
    EvidenceStructureCase(
        case_id="isolated_trigger_weak",
        suffix="ISO_W",
        magnitude=2.50,
        support_level="weakly_supported",
        audit_status="isolated",
        audit_independent_corroboration=False,
        corroboration_kind="none",
        disagreement_case=False,
    ),
    EvidenceStructureCase(
        case_id="balance_growth_supported",
        suffix="BAL_S",
        magnitude=2.50,
        support_level="supported",
        audit_status="corroborated",
        audit_independent_corroboration=True,
        corroboration_kind="strong_balance_growth",
        disagreement_case=False,
    ),
    EvidenceStructureCase(
        case_id="adequate_cashflow_not_supported",
        suffix="ADEQ_N",
        magnitude=0.90,
        support_level="not_supported",
        audit_status="isolated",
        audit_independent_corroboration=False,
        corroboration_kind="none",
        disagreement_case=True,
        primary_trigger_present=False,
    ),
    EvidenceStructureCase(
        case_id="missing_cashflow_insufficient",
        suffix="MISS_I",
        magnitude=0.0,
        support_level="insufficient_evidence",
        audit_status="not_assessable",
        audit_independent_corroboration=False,
        corroboration_kind="missing_cashflow_input",
        disagreement_case=False,
        primary_trigger_present=False,
        cashflow_input_missing=True,
    ),
)
CASE_SETS = {
    "v1": V1_CASE_SPECS,
    "v2": V2_CASE_SPECS,
}
ACCEPTED_GATE_TEACHER_PROMPTS = {
    CORROBORATION_STRICT_TEACHER_PROMPT_VERSION,
    CORROBORATION_BALANCED_TEACHER_PROMPT_VERSION,
}
ACCEPTED_GATE_JUDGE_PROMPTS = {
    CORROBORATION_STRICT_JUDGE_PROMPT_VERSION,
    CORROBORATION_BALANCED_JUDGE_PROMPT_VERSION,
}


@dataclass(frozen=True)
class DetectorCashflowEvidenceStructureRawBuilderResult:
    status: str
    output_dir: Path
    raw_jsonl_path: Path
    manifest_path: Path
    metrics_path: Path
    records_written: int
    errors: list[str]


@dataclass(frozen=True)
class DetectorCashflowEvidenceStructureSftComposerResult:
    status: str
    output_dir: Path
    sft_jsonl_path: Path
    manifest_path: Path
    metrics_path: Path
    records_written: int
    errors: list[str]


def run_detector_cashflow_evidence_structure_raw_builder(
    *,
    training_clean_reports: list[Path | str],
    development_clean_reports: list[Path | str],
    output_dir: Path | str,
    reserved_real_manual_release_dirs: list[Path | str] | None = None,
    case_set: str = DEFAULT_CASE_SET,
) -> DetectorCashflowEvidenceStructureRawBuilderResult:
    case_specs = _case_specs(case_set)
    output_path = Path(output_dir)
    raw_jsonl_path = output_path / "synthetic_injected_raw.jsonl"
    manifest_path = output_path / "manifest.json"
    metrics_path = output_path / "metrics.json"
    reserved_identities = _reserved_real_manual_identities(reserved_real_manual_release_dirs)
    role_paths = {
        "train": [Path(path) for path in training_clean_reports],
        "development": [Path(path) for path in development_clean_reports],
    }
    if not role_paths["train"] or not role_paths["development"]:
        raise ValueError("training_clean_reports and development_clean_reports must both be non-empty")

    try:
        records: list[dict[str, Any]] = []
        seen_source_identities: set[tuple[str, str]] = set()
        input_artifacts: list[dict[str, Any]] = []
        for role, paths in role_paths.items():
            for path in paths:
                clean_report = json.loads(path.read_text(encoding="utf-8"))
                _validate_clean_report_not_reserved(clean_report, reserved_identities=reserved_identities)
                identity = (str(clean_report.get("company_key")), str(clean_report.get("period_key")))
                if identity in seen_source_identities:
                    raise ValueError(f"clean report source identity appears more than once: {identity}")
                seen_source_identities.add(identity)
                input_artifacts.append(
                    {
                        "calibration_role": role,
                        "path": str(path),
                        "sha256": _file_sha256(path),
                        "company_key": clean_report.get("company_key"),
                        "period_key": clean_report.get("period_key"),
                        "report_profile": clean_report.get("report_profile"),
                    }
                )
                records.extend(_matched_records(clean_report, calibration_role=role, case_set=case_set, case_specs=case_specs))
        for record in records:
            validate_synthetic_raw_candidate(record)
    except (OSError, json.JSONDecodeError, ValueError, KeyError) as error:
        return DetectorCashflowEvidenceStructureRawBuilderResult(
            "failed",
            output_path,
            raw_jsonl_path,
            manifest_path,
            metrics_path,
            0,
            [str(error)],
        )

    output_path.mkdir(parents=True, exist_ok=True)
    _write_jsonl(raw_jsonl_path, records)
    counts = _raw_counts(records)
    manifest = {
        "status": "passed",
        "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
        "artifact_kind": "cashflow_evidence_structure_raw_pool",
        "case_set": case_set,
        "scenario_id": SCENARIO_IDS[case_set],
        "records_written": len(records),
        "raw_jsonl": raw_jsonl_path.name,
        "raw_jsonl_sha256": _file_sha256(raw_jsonl_path),
        "input_artifacts": input_artifacts,
        "counts": counts,
        "experiment_boundary": {
            "real_manual_records_used_as_training_data": False,
            "reserved_real_manual_source_identities_rejected": bool(reserved_real_manual_release_dirs),
            "development_sources_disjoint_from_training_sources": True,
            "synthetic_test_consumed": False,
        },
        "packet_label_leakage_policy": {
            "target_support_level_visible_to_detector": False,
            "support_ceiling_visible_to_detector": False,
            "audit_status_visible_to_detector": True,
            "audit_status_deterministically_maps_to_label": False,
            "real_manual_packet_surface_used": True,
        },
        "created_at": _created_at(),
    }
    _write_json(manifest_path, manifest)
    _write_json(metrics_path, {"status": "passed", "records_written": len(records), **counts})
    return DetectorCashflowEvidenceStructureRawBuilderResult(
        "passed",
        output_path,
        raw_jsonl_path,
        manifest_path,
        metrics_path,
        len(records),
        [],
    )


def run_detector_cashflow_evidence_structure_sft_composer(
    *,
    base_sft_jsonl: Path | str,
    gate_run_dirs: list[Path | str],
    output_dir: Path | str,
    case_set: str | None = None,
) -> DetectorCashflowEvidenceStructureSftComposerResult:
    output_path = Path(output_dir)
    sft_jsonl_path = output_path / "detector_sft_chat.jsonl"
    manifest_path = output_path / "manifest.json"
    metrics_path = output_path / "metrics.json"
    base_path = Path(base_sft_jsonl)
    if not gate_run_dirs:
        raise ValueError("at least one gate_run_dir is required")

    try:
        base_rows = _read_jsonl(base_path)
        promoted_records: list[dict[str, Any]] = []
        gate_artifacts: list[dict[str, Any]] = []
        for gate_run_dir in [Path(path) for path in gate_run_dirs]:
            manifest = _read_json(gate_run_dir / "manifest.json")
            _validate_gate_manifest(manifest, gate_run_dir=gate_run_dir)
            filtered_path = gate_run_dir / "filtered.jsonl"
            metrics_path_source = gate_run_dir / "metrics.json"
            promoted_records.extend(_read_jsonl(filtered_path))
            gate_artifacts.append(
                {
                    "path": str(gate_run_dir),
                    "manifest_sha256": _file_sha256(gate_run_dir / "manifest.json"),
                    "metrics_sha256": _file_sha256(metrics_path_source),
                    "filtered_jsonl_sha256": _file_sha256(filtered_path),
                }
            )

        resolved_case_set = case_set or _case_set_from_records(promoted_records)
        case_specs = _case_specs(resolved_case_set)
        complete_records, group_filter = _complete_group_records(promoted_records, case_specs=case_specs)
        composed_rows = [_normalize_base_chat_row(row) for row in base_rows]
        composed_rows.extend(_evidence_structure_chat_rows(complete_records))
        composed_rows.sort(key=_chat_sort_key)
    except (OSError, json.JSONDecodeError, ValueError, KeyError) as error:
        return DetectorCashflowEvidenceStructureSftComposerResult(
            "failed",
            output_path,
            sft_jsonl_path,
            manifest_path,
            metrics_path,
            0,
            [str(error)],
        )

    output_path.mkdir(parents=True, exist_ok=True)
    _write_jsonl(sft_jsonl_path, composed_rows)
    base_test_rows = sum(row.get("metadata", {}).get("split") == "test" for row in base_rows)
    composed_test_rows = sum(row.get("metadata", {}).get("split") == "test" for row in composed_rows)
    base_test_ids = {
        row.get("metadata", {}).get("example_id")
        for row in base_rows
        if row.get("metadata", {}).get("split") == "test"
    }
    composed_test_ids = {
        row.get("metadata", {}).get("example_id")
        for row in composed_rows
        if row.get("metadata", {}).get("split") == "test"
    }
    if composed_test_rows != base_test_rows or composed_test_ids != base_test_ids:
        raise ValueError("reserved test example identities were not preserved")
    counts = _chat_counts(composed_rows)
    manifest = {
        "status": "passed",
        "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
        "artifact_kind": "cashflow_evidence_structure_sft_corpus",
        "case_set": resolved_case_set,
        "scenario_id": SCENARIO_IDS[resolved_case_set],
        "system_prompt_version": SYSTEM_PROMPT_VERSION,
        "base_sft_jsonl": str(base_path),
        "base_sft_jsonl_sha256": _file_sha256(base_path),
        "gate_artifacts": gate_artifacts,
        "sft_jsonl": sft_jsonl_path.name,
        "sft_jsonl_sha256": _file_sha256(sft_jsonl_path),
        "records_written": len(composed_rows),
        "reserved_test_rows_preserved": composed_test_rows,
        "group_filter": {
            "complete_groups_included": len(complete_records) // len(case_specs),
            **group_filter,
        },
        "counts": counts,
        "experiment_boundary": {
            "base_training_rows_preserved": True,
            "base_validation_rows_preserved": True,
            "base_test_rows_preserved": True,
            "evidence_structure_development_rows_added_to_validation": True,
            "real_manual_records_used_as_training_data": False,
            "synthetic_test_consumed": False,
            "base_packet_evidence_role_enrichment_applied": True,
            "audit_status_deterministically_maps_to_label": False,
        },
        "created_at": _created_at(),
    }
    _write_json(manifest_path, manifest)
    _write_json(
        metrics_path,
        {
            "status": "passed",
            "records_written": len(composed_rows),
            "complete_groups_included": len(complete_records) // len(case_specs),
            **group_filter,
            **counts,
        },
    )
    return DetectorCashflowEvidenceStructureSftComposerResult(
        "passed",
        output_path,
        sft_jsonl_path,
        manifest_path,
        metrics_path,
        len(composed_rows),
        [],
    )


def _matched_records(
    clean_report: dict[str, Any],
    *,
    calibration_role: str,
    case_set: str,
    case_specs: tuple[EvidenceStructureCase, ...],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    group_id = f"{clean_report['report_id']}__cashflow_evidence_structure_{case_set}"
    for case in case_specs:
        records.append(
            _record(
                clean_report,
                calibration_role=calibration_role,
                group_id=group_id,
                case=case,
                case_set=case_set,
            )
        )
    return records


def _record(
    clean_report: dict[str, Any],
    *,
    calibration_role: str,
    group_id: str,
    case: EvidenceStructureCase,
    case_set: str,
) -> dict[str, Any]:
    base_report_id = clean_report["report_id"]
    synthetic_report_id = _synthetic_report_id(base_report_id, "SYN_EARN_STRUCT", case.suffix)
    rows_by_account = {
        row["standard_account"]: row
        for row in clean_report["structured_evidence"]["rows"]
    }
    role_rows = _rows_by_profile_role(clean_report, rows_by_account)
    packet = _evidence_structure_packet(
        clean_report=clean_report,
        synthetic_report_id=synthetic_report_id,
        role_rows=role_rows,
        case=case,
    )
    traceability = clean_report["traceability"]
    split_metadata = {
        "company_key": clean_report["company_key"],
        "period_key": clean_report["period_key"],
        "group_key": synthetic_report_id,
        "derived_from_group_key": base_report_id,
        "source_file_sha256": traceability["source_file_sha256"],
        "normalized_text_hash": traceability["normalized_text_hash"],
        "table_content_hash": traceability["table_content_hash"],
        "derived_from_report_artifact_id": clean_report["artifact_id"],
        "derived_from_source_document_id": clean_report["source_document_id"],
    }
    if traceability.get("source_group_key"):
        split_metadata["source_group_key"] = traceability["source_group_key"]
    return {
        "example_id": f"SYN_RAW_{synthetic_report_id}",
        "dataset_version": "tdf_v1.0.0",
        "source_type": "synthetic_injected_raw",
        "input": {"type": "DetectorPacket", "data": packet},
        "metadata": {
            "risk_category": RISK_CATEGORY,
            "report_profile": clean_report["report_profile"],
            "report_period_type": clean_report["report_period_type"],
            "language": clean_report["language"],
            "evidence_profile": {
                "has_table_evidence": True,
                "has_note_evidence": case.corroboration_kind == "collection_delay_disclosure",
                "has_variance_explanation": case.corroboration_kind == "mitigated_balance_growth",
                "has_tool_findings": True,
                "has_contradicting_evidence": False,
                "has_missing_required_evidence": case.audit_status == "isolated" or case.cashflow_input_missing,
                "evidence_types": _evidence_types_for_case(case),
            },
            "generation_metadata": {
                "generation_method": "grounded_structured_evidence_cashflow_evidence_structure",
                "case_set": case_set,
                "base_report_id": base_report_id,
                "synthetic_report_id": synthetic_report_id,
                "injection_scenario_id": SCENARIO_IDS[case_set],
                "target_risk_category": RISK_CATEGORY,
                "target_support_level": case.support_level,
                "canonical_target_id": f"earnings_cashflow_evidence_structure__{case.case_id}",
                "variant_slot_id": "evidence_structure_surface_v1",
                "variant_pattern_id": case.case_id,
                "calibration_role": calibration_role,
                "matched_group_id": group_id,
                "trigger_magnitude": case.magnitude,
                "audit_status": case.audit_status,
                "audit_independent_corroboration": case.audit_independent_corroboration,
                "audit_disagreement_case": case.disagreement_case,
                "corroboration_kind": case.corroboration_kind,
            },
            "split_metadata": split_metadata,
            "audit_metadata": {
                "generator_version": ARTIFACT_CONTRACT_VERSION,
                "dataset_artifact_traceability": {
                    "derived_from_report_artifact_id": clean_report["artifact_id"],
                    "derived_from_source_document_id": clean_report["source_document_id"],
                },
                "hidden_injection_details": {
                    "modified_standard_account": role_rows["cashflow"]["standard_account"],
                    "synthetic_value": _synthetic_cashflow_value(role_rows["earnings"], case),
                },
            },
        },
    }


def _evidence_structure_packet(
    *,
    clean_report: dict[str, Any],
    synthetic_report_id: str,
    role_rows: dict[str, dict[str, Any]],
    case: EvidenceStructureCase,
) -> dict[str, Any]:
    current_period = _period_display(clean_report["period_key"])
    prior_period = _prior_period_display(clean_report["period_key"])
    current_report_id = synthetic_report_id
    prior_report_id = f"{synthetic_report_id}_PRIOR"
    profit_account = role_rows["earnings"]["standard_account"]
    cashflow_account = role_rows["cashflow"]["standard_account"]
    profit_value = _positive_profit_value(role_rows["earnings"])
    current_cashflow_value = (
        None
        if case.cashflow_input_missing
        else (
            -round(profit_value * case.magnitude)
            if case.primary_trigger_present
            else round(profit_value * case.magnitude)
        )
    )
    prior_profit_value = _prior_profit_value(role_rows["earnings"], profit_value)
    prior_cashflow_value = _prior_cashflow_value(profit_value, case.magnitude or 1.0)

    current_profit_row = _period_table_row(
        report_id=current_report_id,
        period=current_period,
        standard_account=profit_account,
        account_tag="PROFIT",
        value=profit_value,
    )
    prior_profit_row = _period_table_row(
        report_id=prior_report_id,
        period=prior_period,
        standard_account=profit_account,
        account_tag="PROFIT",
        value=prior_profit_value,
    )
    prior_cashflow_row = _period_table_row(
        report_id=prior_report_id,
        period=prior_period,
        standard_account=cashflow_account,
        account_tag="OPERATING_CASH_FLOW",
        value=prior_cashflow_value,
    )
    rows = [
        current_profit_row,
        prior_profit_row,
        prior_cashflow_row,
    ]
    if current_cashflow_value is not None:
        rows.insert(
            1,
            _period_table_row(
                report_id=current_report_id,
                period=current_period,
                standard_account=cashflow_account,
                account_tag="OPERATING_CASH_FLOW",
                value=current_cashflow_value,
            ),
        )
    primary_signal_id = "positive_profit_negative_cfo"
    primary_tool_result_id = f"TOOL_{synthetic_report_id}_EARN_CASH_001"
    primary_evidence_refs = [_cell_ref(row, role="input") for row in rows]
    primary_finding = {
        "tool_result_id": primary_tool_result_id,
        "tool_category": "quantitative",
        "tool_name": "standard_corporate_earnings_cashflow_mismatch_tool",
        "risk_category": RISK_CATEGORY,
        "signal_id": primary_signal_id,
        "flag": case.primary_trigger_present,
        "metric_name": "cfo_to_net_income_ratio",
        "value": None if case.cashflow_input_missing else (-case.magnitude if case.primary_trigger_present else case.magnitude),
        "strength": "strong" if case.primary_trigger_present else "not_applicable",
        "trigger_strength": "strong" if case.primary_trigger_present else "not_applicable",
        "evidence_role": "primary_trigger",
        "independent_corroboration_present": case.audit_independent_corroboration,
        "corroboration_evidence_refs": [],
        "corroborates_signal_ids": [],
        "summary": _primary_finding_summary(case),
        "threshold": {
            "basis": "configured_default_v1",
            "config_version": "standard_corporate_v1",
            "description": "Flag when operating cash flow is less than 70% of net income.",
            "threshold_type": "ratio_less_than",
            "unit": "ratio",
            "value": 0.7,
        },
        "evidence_refs": primary_evidence_refs,
    }
    tool_findings = [primary_finding]
    relevant_notes: list[dict[str, Any]] = []
    relevant_variance_explanations: list[dict[str, Any]] = []
    supporting_signal_ids = [primary_signal_id]
    candidate_reason = _candidate_reason_for_case(case)
    if case.corroboration_kind in {"strong_balance_growth", "mitigated_balance_growth"}:
        balance_rows, balance_finding = _balance_corroboration(
            synthetic_report_id=synthetic_report_id,
            current_report_id=current_report_id,
            prior_report_id=prior_report_id,
            current_period=current_period,
            prior_period=prior_period,
            source_row=role_rows["receivables"],
            primary_tool_result_id=primary_tool_result_id,
            mitigated=case.corroboration_kind == "mitigated_balance_growth",
        )
        rows.extend(balance_rows)
        tool_findings.append(balance_finding)
        supporting_signal_ids.append(balance_finding["signal_id"])
        primary_finding["corroboration_evidence_refs"] = [
            {"evidence_ref_type": "tool_result", "ref_id": balance_finding["tool_result_id"]},
            *balance_finding["evidence_refs"],
        ]
        candidate_reason = (
            f"{candidate_reason} A linked operating-balance signal also increased materially in the same packet."
        )
        if case.corroboration_kind == "mitigated_balance_growth":
            relevant_variance_explanations.append(_mitigating_variance(synthetic_report_id))
            candidate_reason = (
                f"{candidate_reason} Management also provides timing context that limits the strength "
                "of the corroboration."
            )
    elif case.corroboration_kind == "collection_delay_disclosure":
        note = _collection_delay_note(synthetic_report_id)
        relevant_notes.append(note)
        supporting_signal_ids.append("post_period_collection_delay_disclosure")
        candidate_reason = (
            f"{candidate_reason} A disclosure describes delayed customer collections after the reporting period."
        )
    tool_findings.append(
        _cashflow_audit_finding(
            synthetic_report_id=synthetic_report_id,
            primary_tool_result_id=primary_tool_result_id,
            primary_signal_id=primary_signal_id,
            case=case,
            prior_rows=[prior_profit_row, prior_cashflow_row],
        )
    )

    packet = {
        "packet_id": f"PACKET_{synthetic_report_id}",
        "candidate_id": f"CAND_{synthetic_report_id}",
        "report_id": synthetic_report_id,
        "task": {
            "risk_category": RISK_CATEGORY,
            "question": "Does the provided evidence support the candidate risk signal?",
            "expected_output": "Return a structured DetectorAssessment.",
        },
        "metadata": {
            "company_name": clean_report["company_name"],
            "ticker": clean_report["ticker"],
            "period": clean_report["period"],
            "report_period_type": clean_report["report_period_type"],
            "report_profile": clean_report["report_profile"],
            "currency": clean_report["currency"],
            "unit": clean_report["unit"],
            "language": clean_report["language"],
        },
        "candidate_summary": {
            "priority": "high",
            "reason_for_candidate": candidate_reason,
            "review_mode": "required",
            "supporting_signal_ids": supporting_signal_ids,
        },
        "relevant_table_rows": rows,
        "relevant_notes": relevant_notes,
        "relevant_variance_explanations": relevant_variance_explanations,
        "tool_findings": tool_findings,
        "rules": [
            {
                "description": (
                    "Use the visible evidence and linked tool findings to assess whether the candidate risk "
                    "signal is supported. Treat the cash-flow corroboration audit as evidence context, "
                    "not as a deterministic support label."
                ),
                "related_signal_ids": supporting_signal_ids,
                "risk_category": RISK_CATEGORY,
                "rule_id": "RULE_EARNINGS_CASHFLOW_MISMATCH_001",
                "rule_name": "Assess only the candidate risk category using provided evidence",
            }
        ],
        "constraints": {
            "allowed_decisions": ["supported", "weakly_supported", "not_supported", "insufficient_evidence"],
            "avoid_prohibited_legal_claims": True,
            "evidence_must_reference_provided_ids": True,
            "evidence_bundle_semantics": {
                "tool_finding_strength_scope": "trigger_magnitude_only",
                "trigger_inputs_are_not_independent_corroboration": True,
                "independent_corroboration_requires_distinct_signal_or_disclosure": True,
                "cashflow_corroboration_audit_is_context_only": True,
            },
            "max_rationale_sentences": 3,
        },
    }
    _validate_generated_packet(packet)
    validate_detector_packet(packet)
    return packet


def _balance_corroboration(
    *,
    synthetic_report_id: str,
    current_report_id: str,
    prior_report_id: str,
    current_period: str,
    prior_period: str,
    source_row: dict[str, Any],
    primary_tool_result_id: str,
    mitigated: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    account = source_row["standard_account"]
    prior_value = max(abs(float(source_row["prior_value"])), 100.0)
    current_value = round(prior_value * (1.6 if not mitigated else 1.35))
    rows = [
        _period_table_row(
            report_id=current_report_id,
            period=current_period,
            standard_account=account,
            account_tag="LINKED_BALANCE",
            value=current_value,
        ),
        _period_table_row(
            report_id=prior_report_id,
            period=prior_period,
            standard_account=account,
            account_tag="LINKED_BALANCE",
            value=prior_value,
        ),
    ]
    signal_id = "linked_balance_growth_corroborates_cash_conversion_pressure"
    return rows, {
        "tool_result_id": f"TOOL_{synthetic_report_id}_LINKED_BALANCE_001",
        "tool_category": "quantitative",
        "tool_name": "standard_corporate_linked_balance_growth_tool",
        "risk_category": RISK_CATEGORY,
        "signal_id": signal_id,
        "flag": True,
        "metric_name": "linked_operating_balance_growth",
        "value": 0.60 if not mitigated else 0.35,
        "strength": "moderate",
        "trigger_strength": "moderate",
        "evidence_role": "independent_corroboration",
        "independent_corroboration_present": True,
        "summary": (
            "The linked operating balance increased 60.0%, providing a second packet-visible signal "
            "consistent with cash-conversion pressure."
            if not mitigated
            else (
                "The linked operating balance increased 35.0%, but the packet also includes timing "
                "context that limits how strongly it corroborates the cash-flow trigger."
            )
        ),
        "threshold": {
            "basis": "configured_default_v1",
            "config_version": "standard_corporate_v1",
            "description": "Corroborate when a linked operating balance increases by more than 30%.",
            "threshold_type": "growth_greater_than",
            "unit": "ratio",
            "value": 0.3,
        },
        "corroborates_signal_ids": ["positive_profit_negative_cfo"],
        "corroboration_evidence_refs": [
            {"evidence_ref_type": "tool_result", "ref_id": primary_tool_result_id},
        ],
        "evidence_refs": [_cell_ref(row, role="input") for row in rows],
    }


def _cashflow_audit_finding(
    *,
    synthetic_report_id: str,
    primary_tool_result_id: str,
    primary_signal_id: str,
    case: EvidenceStructureCase,
    prior_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if case.audit_status == "isolated":
        summary = (
            "Audit checked receivables, inventory, payables or advances, allowances, reserves, "
            "variance explanations, and note inconsistency; no independent corroborating signal was found."
        )
    elif case.audit_status == "corroborated":
        summary = "Audit found a distinct operating-balance signal that corroborates the cash-flow trigger."
    elif case.audit_status == "corroborated_but_mitigated":
        summary = (
            "Audit found a distinct operating-balance signal, but packet-visible timing context limits "
            "how strongly it corroborates the cash-flow trigger."
        )
    else:
        summary = (
            "Audit could not assess independent corroboration because the current-period operating "
            "cash-flow input needed for the primary comparison is missing."
        )
    return {
        "tool_result_id": f"TOOL_{synthetic_report_id}_CASHFLOW_AUDIT_001",
        "tool_category": "evidence_audit",
        "tool_name": "cashflow_corroboration_audit_tool",
        "risk_category": RISK_CATEGORY,
        "signal_id": "cashflow_corroboration_audit",
        "flag": False,
        "metric_name": "cashflow_corroboration_audit_status",
        "value": case.audit_status,
        "strength": "context",
        "trigger_strength": "not_applicable",
        "evidence_role": "context",
        "independent_corroboration_present": case.audit_independent_corroboration,
        "corroborates_signal_ids": [primary_signal_id] if case.audit_independent_corroboration else [],
        "corroboration_evidence_refs": (
            [{"evidence_ref_type": "tool_result", "ref_id": primary_tool_result_id}]
            if case.audit_independent_corroboration
            else []
        ),
        "checked_corroboration_families": [
            "receivables",
            "inventory",
            "payables_or_advances",
            "allowance_or_reserve_pressure",
            "unexplained_variance",
            "note_inconsistency",
        ],
        "non_corroborating_context": [
            f"{row['report_id']}:{next(iter(row['values'].values()))['cell_id']}"
            for row in prior_rows
        ],
        "finding_summary": summary,
        "evidence_refs": [{"evidence_ref_type": "tool_result", "ref_id": primary_tool_result_id}],
    }


def _collection_delay_note(report_id: str) -> dict[str, str]:
    return {
        "report_id": report_id,
        "note_id": f"NOTE_{report_id}_COLLECTION_DELAY",
        "note_type": "cash_flow",
        "title": "Post-period collection timing",
        "text": (
            "The report notes that a material portion of current-period revenue was collected after "
            "the reporting date, creating a timing gap between recognized profit and operating cash flow."
        ),
    }


def _mitigating_variance(report_id: str) -> dict[str, str]:
    return {
        "report_id": report_id,
        "span_id": f"SPAN_{report_id}_TIMING_CONTEXT",
        "title": "Operating cash-flow timing context",
        "text": (
            "Management attributes part of the operating-cash-flow decline to planned settlement timing "
            "and short-term working-capital movements expected to reverse after the reporting period."
        ),
    }


def _evidence_types_for_case(case: EvidenceStructureCase) -> list[str]:
    evidence_types = ["table_cell", "tool_result"]
    if case.corroboration_kind == "collection_delay_disclosure":
        evidence_types.append("note")
    if case.corroboration_kind == "mitigated_balance_growth":
        evidence_types.append("variance_explanation_span")
    return evidence_types


def _synthetic_cashflow_value(profit_row: dict[str, Any], case: EvidenceStructureCase) -> int | float | None:
    if case.cashflow_input_missing:
        return None
    profit_value = _positive_profit_value(profit_row)
    if case.primary_trigger_present:
        return -round(profit_value * case.magnitude)
    return round(profit_value * case.magnitude)


def _primary_finding_summary(case: EvidenceStructureCase) -> str:
    if case.cashflow_input_missing:
        return "Current-period operating cash flow is missing, so the cash-flow mismatch ratio cannot be assessed."
    if case.primary_trigger_present:
        return (
            f"Operating cash flow was {-case.magnitude:.2f} times net income, below the configured "
            "0.70 threshold despite positive reported profit."
        )
    return (
        f"Operating cash flow was {case.magnitude:.2f} times net income, above the configured "
        "0.70 threshold, so the primary mismatch trigger is not present."
    )


def _candidate_reason_for_case(case: EvidenceStructureCase) -> str:
    if case.cashflow_input_missing:
        return (
            "The candidate requires review because the packet lacks current-period operating cash-flow "
            "evidence needed to assess cash realization."
        )
    if case.primary_trigger_present:
        return (
            "Reported profit is positive while operating cash flow is materially negative, creating "
            "a candidate earnings-cash-flow mismatch."
        )
    return (
        "Reported profit is positive and operating cash flow remains above the threshold, so the "
        "candidate requires rejection unless other packet evidence supports escalation."
    )


def _period_table_row(
    *,
    report_id: str,
    period: str,
    standard_account: str,
    account_tag: str,
    value: int | float,
) -> dict[str, Any]:
    compact_period = period.replace("-", "_")
    row_id = f"ROW_{account_tag}_{compact_period}"
    cell_id = f"CELL_{account_tag}_{compact_period}"
    return {
        "report_id": report_id,
        "row_id": row_id,
        "standard_account": standard_account,
        "values": {
            period: {
                "cell_id": cell_id,
                "unit": "vnd",
                "value": value,
            }
        },
    }


def _cell_ref(row: dict[str, Any], *, role: str) -> dict[str, str]:
    period_value = next(iter(row["values"].values()))
    return {
        "evidence_ref_type": "table_cell",
        "local_evidence_id": period_value["cell_id"],
        "ref_id": f"{row['report_id']}:{period_value['cell_id']}",
        "report_id": row["report_id"],
        "role": role,
    }


def _positive_profit_value(profit_row: dict[str, Any]) -> float:
    return max(abs(float(profit_row["current_value"])), 100.0)


def _prior_profit_value(profit_row: dict[str, Any], fallback_profit: float) -> float:
    source_value = float(profit_row.get("prior_value") or 0)
    if source_value != 0:
        return round(source_value)
    return round(fallback_profit * 0.8)


def _prior_cashflow_value(profit_value: float, magnitude: float) -> float:
    if magnitude >= 2.0:
        return -round(profit_value * 0.75)
    return round(profit_value * 0.90)


def _period_display(period_key: str) -> str:
    return period_key.replace("_", "-")


def _prior_period_display(period_key: str) -> str:
    year_text, quarter = period_key.split("_", 1)
    return f"{int(year_text) - 1}-{quarter}"


def _validate_gate_manifest(manifest: dict[str, Any], *, gate_run_dir: Path) -> None:
    if manifest.get("mode") != "live":
        raise ValueError(f"evidence-structure gate must be live: {gate_run_dir}")
    if manifest.get("run_purpose") != "approved_corroboration_calibration":
        raise ValueError(f"unexpected evidence-structure gate run purpose: {gate_run_dir}")
    if manifest.get("trainable_labels_approved") is not True:
        raise ValueError(f"evidence-structure gate is not approved for training: {gate_run_dir}")
    if manifest.get("teacher", {}).get("prompt_version") not in ACCEPTED_GATE_TEACHER_PROMPTS:
        raise ValueError(f"evidence-structure gate used the wrong teacher prompt: {gate_run_dir}")
    if manifest.get("judge", {}).get("prompt_version") not in ACCEPTED_GATE_JUDGE_PROMPTS:
        raise ValueError(f"evidence-structure gate used the wrong judge prompt: {gate_run_dir}")


def _case_specs(case_set: str) -> tuple[EvidenceStructureCase, ...]:
    try:
        return CASE_SETS[case_set]
    except KeyError as error:
        raise ValueError(f"unknown cash-flow evidence-structure case set: {case_set}") from error


def _case_set_from_records(records: list[dict[str, Any]]) -> str:
    case_sets = {
        record.get("metadata", {}).get("generation_metadata", {}).get("case_set")
        for record in records
    }
    case_sets.discard(None)
    if len(case_sets) != 1:
        raise ValueError("promoted evidence-structure records must come from exactly one case set")
    case_set = next(iter(case_sets))
    if not isinstance(case_set, str):
        raise ValueError("promoted evidence-structure records are missing case-set metadata")
    _case_specs(case_set)
    return case_set


def _complete_group_records(
    records: list[dict[str, Any]],
    *,
    case_specs: tuple[EvidenceStructureCase, ...],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    label_mismatch_group_ids: set[str] = set()
    expected_case_ids = {case.case_id for case in case_specs}
    for record in records:
        generation = record.get("metadata", {}).get("generation_metadata", {})
        group_id = generation.get("matched_group_id")
        role = generation.get("calibration_role")
        target = generation.get("target_support_level")
        case_id = generation.get("variant_pattern_id")
        if not group_id or role not in {"train", "development"} or case_id not in expected_case_ids:
            raise ValueError("promoted evidence-structure record is missing matched-group role metadata")
        if record.get("metadata", {}).get("support_level") != target:
            label_mismatch_group_ids.add(group_id)
        groups.setdefault(group_id, []).append(record)

    complete: list[dict[str, Any]] = []
    incomplete = 0
    for group_id, group_records in sorted(groups.items()):
        if group_id in label_mismatch_group_ids:
            continue
        roles = {
            record["metadata"]["generation_metadata"]["calibration_role"]
            for record in group_records
        }
        case_ids = {
            record["metadata"]["generation_metadata"]["variant_pattern_id"]
            for record in group_records
        }
        if len(group_records) == len(case_specs) and case_ids == expected_case_ids and len(roles) == 1:
            complete.extend(group_records)
        else:
            incomplete += 1
    if not complete:
        raise ValueError("no complete cash-flow evidence-structure matched groups were promoted")
    return complete, {
        "incomplete_groups_excluded": incomplete,
        "label_mismatch_groups_excluded": len(label_mismatch_group_ids),
    }


def _evidence_structure_chat_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for original in records:
        record = deepcopy(original)
        generation = record["metadata"]["generation_metadata"]
        split = "train" if generation["calibration_role"] == "train" else "validation"
        split_metadata = record["metadata"].setdefault("split_metadata", {})
        split_metadata.update(
            {
                "split": split,
                "usable_for_training": True,
                "exclusion_reason": None,
                "decontamination_group_id": _group_id(split_metadata.get("derived_from_group_key", "")),
            }
        )
        chat = _record_to_chat(
            record,
            split=split,
            allowed_source_types={"synthetic_injected_filtered"},
            excluded_source_types=set(),
            system_prompt_version=SYSTEM_PROMPT_VERSION,
        )
        chat["metadata"]["calibration_matched_group_id"] = generation["matched_group_id"]
        chat["metadata"]["calibration_role"] = generation["calibration_role"]
        chat["metadata"]["audit_status"] = generation["audit_status"]
        chat["metadata"]["audit_independent_corroboration"] = generation["audit_independent_corroboration"]
        chat["metadata"]["audit_disagreement_case"] = generation["audit_disagreement_case"]
        chat["metadata"]["corroboration_kind"] = generation["corroboration_kind"]
        chat["metadata"]["cashflow_evidence_structure"] = True
        rows.append(chat)
    return rows


def _normalize_base_chat_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(row)
    normalized["messages"][0]["content"] = detector_sft_system_prompt(SYSTEM_PROMPT_VERSION)
    for message in normalized.get("messages", [])[1:]:
        try:
            payload = json.loads(message["content"])
        except (json.JSONDecodeError, TypeError, KeyError):
            continue
        payload = _replace_risk_category(payload)
        if message.get("role") == "user":
            payload = enrich_detector_packet_evidence_roles(payload)
        message["content"] = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    metadata = normalized.get("metadata", {})
    if metadata.get("risk_category") == "earnings_cashflow_quality_risk":
        metadata["risk_category"] = RISK_CATEGORY
    return normalized


def _replace_risk_category(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _replace_risk_category(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_risk_category(item) for item in value]
    if value == "earnings_cashflow_quality_risk":
        return RISK_CATEGORY
    return value


def _chat_sort_key(row: dict[str, Any]) -> tuple[int, str]:
    split_order = {"train": 0, "validation": 1, "test": 2}
    metadata = row.get("metadata", {})
    return split_order.get(metadata.get("split"), 99), str(metadata.get("example_id", ""))


def _raw_counts(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        "calibration_roles": dict(
            sorted(Counter(record["metadata"]["generation_metadata"]["calibration_role"] for record in records).items())
        ),
        "target_support_levels": dict(
            sorted(
                Counter(
                    record["metadata"]["generation_metadata"]["target_support_level"]
                    for record in records
                ).items()
            )
        ),
        "report_profiles": dict(sorted(Counter(record["metadata"]["report_profile"] for record in records).items())),
        "matched_groups": {
            "total": len(
                {
                    record["metadata"]["generation_metadata"]["matched_group_id"]
                    for record in records
                }
            )
        },
        "audit_statuses": dict(
            sorted(Counter(record["metadata"]["generation_metadata"]["audit_status"] for record in records).items())
        ),
        "audit_disagreement_cases": dict(
            sorted(
                Counter(
                    str(record["metadata"]["generation_metadata"]["audit_disagreement_case"])
                    for record in records
                ).items()
            )
        ),
    }


def _chat_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        "splits": _metadata_counter(rows, "split"),
        "support_levels": _metadata_counter(rows, "support_level"),
        "report_profiles": _metadata_counter(rows, "report_profile"),
        "risk_categories": _metadata_counter(rows, "risk_category"),
        "audit_statuses": _metadata_counter(rows, "audit_status"),
        "audit_disagreement_cases": _metadata_counter(rows, "audit_disagreement_case"),
    }


def _metadata_counter(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    values = [
        str(row.get("metadata", {}).get(field) or "unknown")
        for row in rows
    ]
    return dict(sorted(Counter(values).items()))


def _group_id(derived_from_group_key: str) -> str:
    digest = hashlib.sha256(derived_from_group_key.encode("utf-8")).hexdigest()[:16].upper()
    return f"CASHFLOWSTRUCT_DG_{digest}"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _created_at() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cash-flow evidence-structure detector artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    raw_parser = subparsers.add_parser("build-raw")
    raw_parser.add_argument("--training-clean-report", action="append", required=True)
    raw_parser.add_argument("--development-clean-report", action="append", required=True)
    raw_parser.add_argument("--reserved-real-manual-release-dir", action="append", default=[])
    raw_parser.add_argument("--case-set", choices=sorted(CASE_SETS), default=DEFAULT_CASE_SET)
    raw_parser.add_argument("--output-dir", required=True)

    compose_parser = subparsers.add_parser("compose-sft")
    compose_parser.add_argument("--base-sft-jsonl", required=True)
    compose_parser.add_argument("--gate-run-dir", action="append", required=True)
    compose_parser.add_argument("--case-set", choices=sorted(CASE_SETS), default=None)
    compose_parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()
    if args.command == "build-raw":
        result = run_detector_cashflow_evidence_structure_raw_builder(
            training_clean_reports=args.training_clean_report,
            development_clean_reports=args.development_clean_report,
            reserved_real_manual_release_dirs=args.reserved_real_manual_release_dir,
            output_dir=args.output_dir,
            case_set=args.case_set,
        )
        payload = {
            "status": result.status,
            "raw_jsonl_path": str(result.raw_jsonl_path),
            "records_written": result.records_written,
            "errors": result.errors,
        }
    else:
        result = run_detector_cashflow_evidence_structure_sft_composer(
            base_sft_jsonl=args.base_sft_jsonl,
            gate_run_dirs=args.gate_run_dir,
            output_dir=args.output_dir,
            case_set=args.case_set,
        )
        payload = {
            "status": result.status,
            "sft_jsonl_path": str(result.sft_jsonl_path),
            "records_written": result.records_written,
            "errors": result.errors,
        }
    print(json.dumps(payload, sort_keys=True))
    raise SystemExit(0 if result.status == "passed" else 1)


if __name__ == "__main__":
    main()
