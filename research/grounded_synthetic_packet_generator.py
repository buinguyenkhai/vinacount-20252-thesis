from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from research.detector_contract_validation import (
    contains_prohibited_detector_visible_payload,
    validate_detector_packet,
    visible_packet_evidence_ids,
)
from research.synthetic_raw_candidate_validation import validate_synthetic_raw_candidate


ARTIFACT_CONTRACT_VERSION = "grounded_synthetic_packet_generator_v1"
GENERATOR_VERSION = "grounded_synthetic_packet_generator_v1"
SCENARIO_ID = "revenue_receivables_divergence_v1"
SCENARIO_VERSION = "1.0.0"
RISK_CATEGORY = "revenue_income_recognition_risk"
STAGED_JSONL_FILENAME = "staged_synthetic_packets.jsonl"
DEFAULT_SCENARIO_FAMILY = "revenue_receivables"
EARNINGS_CASHFLOW_SCENARIO_ID = "earnings_cashflow_divergence_v1"
EARNINGS_CASHFLOW_RISK_CATEGORY = "earnings_cashflow_quality_risk"
RICH_CORE_ACCOUNTS = ("revenue", "trade_receivables", "net_profit", "operating_cash_flow")
PROFILE_ACCOUNT_ROLES = {
    "standard_corporate": {
        "revenue": "revenue",
        "receivables": "trade_receivables",
        "earnings": "net_profit",
        "cashflow": "operating_cash_flow",
        "rich_core": ("revenue", "trade_receivables", "net_profit", "operating_cash_flow"),
    },
    "insurance": {
        "revenue": "gross_written_premium",
        "receivables": "premium_receivables",
        "earnings": "gross_written_premium",
        "cashflow": "operating_cash_flow",
        "rich_core": ("gross_written_premium", "premium_receivables", "operating_cash_flow"),
    },
    "securities": {
        "revenue": "profit_after_tax",
        "receivables": "margin_lending",
        "earnings": "profit_after_tax",
        "cashflow": "operating_cash_flow",
        "rich_core": (
            "profit_after_tax",
            "margin_lending",
            "margin_impairment",
            "fvtpl_assets",
            "fvtpl_unrealized_gain",
            "operating_cash_flow",
        ),
    },
    "credit_institution": {
        "revenue": "profit_after_tax",
        "receivables": "loans_to_customers",
        "earnings": "profit_after_tax",
        "cashflow": "operating_cash_flow",
        "rich_core": (
            "profit_after_tax",
            "loans_to_customers",
            "loan_group_1",
            "loan_group_2",
            "loan_group_3",
            "loan_group_4",
            "loan_group_5",
            "general_provision",
            "specific_provision",
            "operating_cash_flow",
        ),
    },
}
THESIS_FAMILY_ORDER = (
    "revenue_receivables",
    "receivables_credit_quality",
    "inventory_cost_asset_flow",
    "expense_liability_understatement",
    "earnings_cashflow",
    "asset_quality_valuation",
    "related_party_disclosure",
    "disclosure_inconsistency",
)
SUPPORT_TARGET_ORDER = ("supported", "weakly_supported", "not_supported", "insufficient_evidence")
CANONICAL_VARIANT_SLOTS = (
    "V1_easy_quantitative_clear",
    "V2_easy_quantitative_contradiction",
    "V3_medium_partial_secondary_evidence",
    "V4_hard_missing_required_context",
    "V5_note_or_disclosure_quality",
    "V6_tool_finding_contradiction",
    "V7_profile_specific_mapping",
    "V8_language_or_account_label_style",
)


@dataclass(frozen=True)
class ScenarioDefinition:
    family: str
    scenario_id: str
    risk_category: str
    synthetic_report_tag: str
    synthetic_suffix: str
    target_support_level: str
    synthetic_multiplier: float


@dataclass(frozen=True)
class VariantSlotDefinition:
    slot_id: str
    pattern_id: str
    synthetic_suffix: str


def _canonical_variant_slots() -> tuple[VariantSlotDefinition, ...]:
    return tuple(
        VariantSlotDefinition(
            slot_id=slot_id,
            pattern_id=f"{slot_id}_standard_corporate_v1",
            synthetic_suffix=f"V{index:02d}",
        )
        for index, slot_id in enumerate(CANONICAL_VARIANT_SLOTS, start=1)
    )


def _additional_thesis_scenarios() -> tuple[ScenarioDefinition, ...]:
    families = (
        (
            "receivables_credit_quality",
            "receivables_credit_quality_injection_v1",
            "receivables_credit_quality_risk",
            "SYN_AR_QUALITY",
        ),
        (
            "inventory_cost_asset_flow",
            "inventory_cost_asset_flow_injection_v1",
            "inventory_cost_asset_flow_risk",
            "SYN_INV_COST",
        ),
        (
            "expense_liability_understatement",
            "expense_liability_understatement_injection_v1",
            "expense_liability_understatement_risk",
            "SYN_EXP_LIAB",
        ),
        (
            "asset_quality_valuation",
            "asset_quality_valuation_injection_v1",
            "asset_quality_valuation_risk",
            "SYN_ASSET_VAL",
        ),
        (
            "related_party_disclosure",
            "related_party_disclosure_injection_v1",
            "related_party_disclosure_risk",
            "SYN_RP_DISC",
        ),
        (
            "disclosure_inconsistency",
            "disclosure_inconsistency_injection_v1",
            "disclosure_inconsistency_or_obfuscation",
            "SYN_DISC_OBF",
        ),
    )
    scenarios: list[ScenarioDefinition] = []
    for family, scenario_id, risk_category, tag in families:
        scenarios.extend(
            (
                ScenarioDefinition(family, scenario_id, risk_category, tag, "001", "supported", 1.50),
                ScenarioDefinition(family, scenario_id, risk_category, tag, "002", "weakly_supported", 1.05),
                ScenarioDefinition(family, scenario_id, risk_category, tag, "003", "not_supported", 0.92),
                ScenarioDefinition(family, scenario_id, risk_category, tag, "004", "insufficient_evidence", 1.15),
            )
        )
    return tuple(scenarios)


SCENARIOS = (
    ScenarioDefinition(DEFAULT_SCENARIO_FAMILY, SCENARIO_ID, RISK_CATEGORY, "SYN_REV_REC", "001", "supported", 1.5),
    ScenarioDefinition(DEFAULT_SCENARIO_FAMILY, SCENARIO_ID, RISK_CATEGORY, "SYN_REV_REC", "002", "weakly_supported", 1.08),
    ScenarioDefinition(DEFAULT_SCENARIO_FAMILY, SCENARIO_ID, RISK_CATEGORY, "SYN_REV_REC", "003", "not_supported", 0.88),
    ScenarioDefinition(
        DEFAULT_SCENARIO_FAMILY,
        SCENARIO_ID,
        RISK_CATEGORY,
        "SYN_REV_REC",
        "004",
        "insufficient_evidence",
        1.08,
    ),
    ScenarioDefinition(
        "earnings_cashflow",
        EARNINGS_CASHFLOW_SCENARIO_ID,
        EARNINGS_CASHFLOW_RISK_CATEGORY,
        "SYN_EARN_CASH",
        "001",
        "supported",
        0.35,
    ),
    ScenarioDefinition(
        "earnings_cashflow",
        EARNINGS_CASHFLOW_SCENARIO_ID,
        EARNINGS_CASHFLOW_RISK_CATEGORY,
        "SYN_EARN_CASH",
        "002",
        "weakly_supported",
        0.78,
    ),
    ScenarioDefinition(
        "earnings_cashflow",
        EARNINGS_CASHFLOW_SCENARIO_ID,
        EARNINGS_CASHFLOW_RISK_CATEGORY,
        "SYN_EARN_CASH",
        "003",
        "not_supported",
        0.90,
    ),
    ScenarioDefinition(
        "earnings_cashflow",
        EARNINGS_CASHFLOW_SCENARIO_ID,
        EARNINGS_CASHFLOW_RISK_CATEGORY,
        "SYN_EARN_CASH",
        "004",
        "insufficient_evidence",
        0.72,
    ),
) + _additional_thesis_scenarios()
SUPPORTED_SCENARIO_FAMILIES = sorted({scenario.family for scenario in SCENARIOS})


@dataclass(frozen=True)
class GroundedSyntheticPacketGenerationResult:
    status: str
    output_dir: Path
    staged_jsonl_path: Path
    manifest_path: Path
    metrics_path: Path
    records_written: int
    errors: list[str]


def run_grounded_synthetic_packet_generator(
    *,
    clean_report_artifact: Path | str | list[Path | str] | tuple[Path | str, ...],
    output_dir: Path | str,
    run_id: str = "grounded_synthetic_packet_generator",
    scenario_family: str | list[str] | tuple[str, ...] | None = None,
    require_rich_clean_report: bool = False,
    reserved_real_manual_release_dir: Path | str | list[Path | str] | tuple[Path | str, ...] | None = None,
) -> GroundedSyntheticPacketGenerationResult:
    input_paths = _clean_report_artifact_paths(clean_report_artifact)
    selected_families = _scenario_families(scenario_family)
    reserved_identities = _reserved_real_manual_identities(reserved_real_manual_release_dir)
    output_dir_path = Path(output_dir)
    staged_jsonl_path = output_dir_path / STAGED_JSONL_FILENAME
    manifest_path = output_dir_path / "manifest.json"
    metrics_path = output_dir_path / "metrics.json"

    try:
        record_batches = []
        for input_path in input_paths:
            try:
                clean_report = json.loads(input_path.read_text(encoding="utf-8"))
                _validate_clean_report_not_reserved(clean_report, reserved_identities=reserved_identities)
                record_batches.append(
                    _generate_records(
                        clean_report,
                        scenario_families=selected_families,
                        require_rich_clean_report=require_rich_clean_report,
                    )
                )
            except (OSError, json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"{input_path}: {error}") from error
        records = _interleave_record_batches(record_batches)
        for record in records:
            validate_synthetic_raw_candidate(record)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        return GroundedSyntheticPacketGenerationResult(
            "failed",
            output_dir_path,
            staged_jsonl_path,
            manifest_path,
            metrics_path,
            0,
            [str(error)],
        )

    output_dir_path.mkdir(parents=True, exist_ok=True)
    _write_jsonl(staged_jsonl_path, records)
    _write_json(manifest_path, _manifest(run_id=run_id, input_paths=input_paths, records=records))
    _write_json(metrics_path, _metrics(run_id=run_id, records=records))
    return GroundedSyntheticPacketGenerationResult(
        "passed",
        output_dir_path,
        staged_jsonl_path,
        manifest_path,
        metrics_path,
        len(records),
        [],
    )


def _clean_report_artifact_paths(clean_report_artifact: Path | str | list[Path | str] | tuple[Path | str, ...]) -> list[Path]:
    if isinstance(clean_report_artifact, (str, Path)):
        paths = [Path(clean_report_artifact)]
    else:
        paths = [Path(path) for path in clean_report_artifact]
    if not paths:
        raise ValueError("at least one clean report artifact is required")
    return paths


def _reserved_real_manual_identities(
    release_dirs: Path | str | list[Path | str] | tuple[Path | str, ...] | None,
) -> set[str]:
    if release_dirs is None:
        return set()
    paths = [Path(release_dirs)] if isinstance(release_dirs, (str, Path)) else [Path(path) for path in release_dirs]
    identities: set[str] = set()
    for release_dir in paths:
        manifest = json.loads((release_dir / "manifest.json").read_text(encoding="utf-8"))
        for split in manifest.get("num_examples_by_split", {}):
            split_path = release_dir / f"{split}.jsonl"
            if not split_path.exists():
                continue
            for line in split_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("source_type") != "human_gold_real_report":
                    raise ValueError("reserved real/manual release dirs must contain only human_gold_real_report records")
                identities.update(_record_source_identity_values(record))
    return identities


def _validate_clean_report_not_reserved(clean_report: dict[str, Any], *, reserved_identities: set[str]) -> None:
    if not reserved_identities:
        return
    overlap = sorted(_clean_report_source_identity_values(clean_report) & reserved_identities)
    if overlap:
        report_id = clean_report.get("report_id", "<unknown>")
        preview = ", ".join(overlap[:5])
        raise ValueError(
            f"clean report artifact {report_id} collides with reserved real/manual source identities: {preview}. "
            "Remove reserved anchors from the synthetic source pool before generation."
        )


def _clean_report_source_identity_values(clean_report: dict[str, Any]) -> set[str]:
    traceability = clean_report.get("traceability", {})
    company_key = clean_report.get("company_key")
    period_key = clean_report.get("period_key")
    values = {
        clean_report.get("report_id"),
        clean_report.get("artifact_id"),
        clean_report.get("source_document_id"),
        traceability.get("source_file_sha256"),
        traceability.get("normalized_text_hash"),
        traceability.get("table_content_hash"),
        traceability.get("source_group_key"),
    }
    if company_key and period_key:
        values.add(f"{company_key}_{period_key}")
    return {str(value) for value in values if value}


def _record_source_identity_values(record: dict[str, Any]) -> set[str]:
    split_metadata = record.get("metadata", {}).get("split_metadata", {})
    audit_traceability = record.get("metadata", {}).get("audit_metadata", {}).get("dataset_artifact_traceability", {})
    packet = record.get("input", {}).get("data", {})
    values = {
        split_metadata.get("group_key"),
        split_metadata.get("derived_from_group_key"),
        split_metadata.get("base_report_id"),
        split_metadata.get("synthetic_report_id"),
        split_metadata.get("source_file_sha256"),
        split_metadata.get("normalized_text_hash"),
        split_metadata.get("table_content_hash"),
        split_metadata.get("derived_from_report_artifact_id"),
        split_metadata.get("derived_from_source_document_id"),
        audit_traceability.get("derived_from_report_artifact_id"),
        audit_traceability.get("derived_from_source_document_id"),
        packet.get("report_id"),
    }
    return {str(value) for value in values if value}


def _scenario_families(scenario_family: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if scenario_family is None:
        families = [DEFAULT_SCENARIO_FAMILY]
    elif isinstance(scenario_family, str):
        families = [scenario_family]
    else:
        families = list(scenario_family)
    if not families:
        raise ValueError("at least one scenario family is required")
    unsupported = sorted(set(families) - set(SUPPORTED_SCENARIO_FAMILIES))
    if unsupported:
        raise ValueError(f"unsupported scenario family: {unsupported}")
    return families


def _generate_records(
    clean_report: dict[str, Any],
    *,
    scenario_families: list[str],
    require_rich_clean_report: bool = False,
) -> list[dict[str, Any]]:
    selected = [scenario for scenario in SCENARIOS if scenario.family in scenario_families]
    if require_rich_clean_report:
        _validate_rich_clean_report(clean_report)
    supported = [
        scenario
        for scenario in selected
        if _clean_report_supports_scenario(clean_report, scenario=scenario)
    ]
    if not supported:
        _validate_clean_report(clean_report, scenario=selected[0])
        raise ValueError("clean report artifact does not support any selected scenario family")
    return [
        _generate_record(clean_report, scenario=scenario, variant_slot=variant_slot)
        for variant_slot in _canonical_variant_slots()
        for scenario in sorted(supported, key=_scenario_sort_key)
    ]


def _validate_rich_clean_report(clean_report: dict[str, Any]) -> None:
    unsupported_targets = [
        _canonical_target_id(scenario)
        for scenario in sorted(SCENARIOS, key=_scenario_sort_key)
        if not _clean_report_supports_scenario(clean_report, scenario=scenario)
    ]
    if unsupported_targets:
        report_id = clean_report.get("report_id", "<unknown>")
        raise ValueError(
            f"clean report {report_id} does not support all final-scale canonical targets: {unsupported_targets}"
        )


def _scenario_sort_key(scenario: ScenarioDefinition) -> tuple[int, int]:
    family_index = THESIS_FAMILY_ORDER.index(scenario.family) if scenario.family in THESIS_FAMILY_ORDER else 99
    support_index = (
        SUPPORT_TARGET_ORDER.index(scenario.target_support_level)
        if scenario.target_support_level in SUPPORT_TARGET_ORDER
        else 99
    )
    return ((support_index - family_index) % len(SUPPORT_TARGET_ORDER), family_index)


def _interleave_record_batches(record_batches: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    max_batch_length = max((len(batch) for batch in record_batches), default=0)
    for index in range(max_batch_length):
        for batch in record_batches:
            if index < len(batch):
                records.append(batch[index])
    return records


def _clean_report_supports_scenario(clean_report: dict[str, Any], *, scenario: ScenarioDefinition) -> bool:
    evidence = clean_report.get("structured_evidence")
    if not isinstance(evidence, dict) or not isinstance(evidence.get("rows"), list):
        return False
    rows_by_account = {
        row.get("standard_account"): row
        for row in evidence["rows"]
        if isinstance(row, dict)
    }
    required_accounts = _required_accounts_for_scenario(clean_report, scenario)
    return all(_valid_clean_report_row(rows_by_account.get(account)) for account in required_accounts)


def _required_accounts_for_scenario(clean_report: dict[str, Any], scenario: ScenarioDefinition) -> tuple[str, ...]:
    roles = _profile_account_roles(clean_report)
    if scenario.family == "earnings_cashflow":
        return (roles["earnings"], roles["cashflow"])
    if scenario.family == DEFAULT_SCENARIO_FAMILY:
        return (roles["revenue"], roles["receivables"])
    return tuple(roles["rich_core"])


def _generate_record(
    clean_report: dict[str, Any],
    *,
    scenario: ScenarioDefinition,
    variant_slot: VariantSlotDefinition,
) -> dict[str, Any]:
    _validate_clean_report(clean_report, scenario=scenario)
    base_report_id = clean_report["report_id"]
    synthetic_report_id = _synthetic_report_id(
        base_report_id,
        scenario.synthetic_report_tag,
        f"{scenario.synthetic_suffix}_{variant_slot.synthetic_suffix}",
    )
    rows_by_account = {
        row["standard_account"]: row
        for row in clean_report["structured_evidence"]["rows"]
    }
    role_rows = _rows_by_profile_role(clean_report, rows_by_account)
    if scenario.family == "earnings_cashflow":
        profit = role_rows["earnings"]
        operating_cash_flow = role_rows["cashflow"]
        synthetic_cash_flow_value = round(profit["current_value"] * scenario.synthetic_multiplier)
        packet = _earnings_cashflow_packet(
            clean_report=clean_report,
            synthetic_report_id=synthetic_report_id,
            profit=profit,
            operating_cash_flow=operating_cash_flow,
            synthetic_cash_flow_value=synthetic_cash_flow_value,
        )
        hidden_injection_details = {
            "modified_standard_account": "operating_cash_flow",
            "original_value": operating_cash_flow["current_value"],
            "synthetic_value": synthetic_cash_flow_value,
        }
    elif scenario.family == DEFAULT_SCENARIO_FAMILY:
        revenue = role_rows["revenue"]
        receivables = role_rows["receivables"]
        synthetic_receivables_value = max(
            receivables["current_value"] + 1,
            round(revenue["current_value"] * scenario.synthetic_multiplier),
        )
        if scenario.target_support_level == "weakly_supported":
            revenue_growth = _growth_ratio(revenue["current_value"], revenue["prior_value"])
            weak_receivables_growth = revenue_growth + 0.105
            synthetic_receivables_value = max(
                receivables["current_value"] + 1,
                round(receivables["prior_value"] * (1 + weak_receivables_growth)),
            )
        packet = _revenue_receivables_packet(
            clean_report=clean_report,
            synthetic_report_id=synthetic_report_id,
            revenue=revenue,
            receivables=receivables,
            synthetic_receivables_value=synthetic_receivables_value,
            target_support_level=scenario.target_support_level,
        )
        hidden_injection_details = {
            "modified_standard_account": _profile_account_roles(clean_report)["receivables"],
            "original_value": receivables["current_value"],
            "synthetic_value": synthetic_receivables_value,
        }
    else:
        packet, hidden_injection_details = _generic_thesis_family_packet(
            clean_report=clean_report,
            synthetic_report_id=synthetic_report_id,
            scenario=scenario,
            rows_by_account={
                "revenue": role_rows["revenue"],
                "trade_receivables": role_rows["receivables"],
            },
        )
    packet = _apply_support_target_packet_variant(packet, target_support_level=scenario.target_support_level)
    packet = _apply_variant_slot_packet_profile(packet, variant_slot=variant_slot)
    _validate_generated_packet(packet)

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
            "risk_category": scenario.risk_category,
            "report_profile": clean_report["report_profile"],
            "report_period_type": clean_report["report_period_type"],
            "language": clean_report["language"],
            "evidence_profile": _evidence_profile_for_support_target(
                scenario.target_support_level,
                variant_slot=variant_slot,
            ),
            "generation_metadata": {
                "generation_method": "grounded_structured_evidence_injection",
                "base_report_id": base_report_id,
                "synthetic_report_id": synthetic_report_id,
                "injection_scenario_id": scenario.scenario_id,
                "target_risk_category": scenario.risk_category,
                "target_support_level": scenario.target_support_level,
                "canonical_target_id": _canonical_target_id(scenario),
                "variant_slot_id": variant_slot.slot_id,
                "variant_pattern_id": _variant_pattern_id(clean_report, variant_slot),
            },
            "split_metadata": split_metadata,
            "audit_metadata": {
                "generator_version": GENERATOR_VERSION,
                "dataset_artifact_traceability": {
                    "derived_from_report_artifact_id": clean_report["artifact_id"],
                    "derived_from_source_document_id": clean_report["source_document_id"],
                },
                "hidden_injection_details": hidden_injection_details,
            },
        },
    }


def _apply_support_target_packet_variant(packet: dict[str, Any], *, target_support_level: str) -> dict[str, Any]:
    if target_support_level != "insufficient_evidence":
        return packet
    packet = json.loads(json.dumps(packet))
    packet["tool_findings"] = []
    packet["relevant_notes"] = []
    packet["candidate_summary"] = {
        **packet["candidate_summary"],
        "reason_for_candidate": (
            "The candidate requires review, but the packet only provides table rows without a tool finding "
            "or disclosure context for the risk signal."
        ),
        "priority": "medium",
        "supporting_signal_ids": [],
    }
    return packet


def _apply_variant_slot_packet_profile(packet: dict[str, Any], *, variant_slot: VariantSlotDefinition) -> dict[str, Any]:
    packet = json.loads(json.dumps(packet))
    report_id = packet["report_id"]
    slot_id = variant_slot.slot_id
    if slot_id == "V1_easy_quantitative_clear":
        return packet
    if slot_id == "V2_easy_quantitative_contradiction":
        packet["relevant_variance_explanations"] = [
            {
                "span_id": f"SPAN_{report_id}_MGMT_CONTEXT",
                "report_id": report_id,
                "text": "Management attributes the movement mainly to quarter-end billing timing.",
            }
        ]
        packet["candidate_summary"]["reason_for_candidate"] += " Management context gives a timing explanation."
        return packet
    if slot_id == "V3_medium_partial_secondary_evidence":
        packet["relevant_notes"] = packet["relevant_notes"][:1]
        packet["candidate_summary"]["priority"] = "medium"
        packet["candidate_summary"]["reason_for_candidate"] += " Only one secondary note is available."
        return packet
    if slot_id == "V4_hard_missing_required_context":
        packet["tool_findings"] = []
        packet["candidate_summary"] = {
            **packet["candidate_summary"],
            "priority": "medium",
            "supporting_signal_ids": [],
            "reason_for_candidate": "The table rows suggest review, but no tool finding is available in this packet.",
        }
        return packet
    if slot_id == "V5_note_or_disclosure_quality":
        for note in packet["relevant_notes"]:
            note["text"] = (
                f"{note['text']} The disclosure describes the account movement but gives limited customer, "
                "collection, or aging detail."
            )
        packet["candidate_summary"]["reason_for_candidate"] += " Disclosure detail is limited."
        return packet
    if slot_id == "V6_tool_finding_contradiction":
        for finding in packet["tool_findings"]:
            finding["flag"] = False
            finding["finding_summary"] = "A secondary threshold check does not flag the movement."
        packet["candidate_summary"]["reason_for_candidate"] += " The tool result is conservative."
        return packet
    if slot_id == "V7_profile_specific_mapping":
        row_id_mapping = {row["row_id"]: f"{row['row_id']}_ALT" for row in packet["relevant_table_rows"]}
        for row in packet["relevant_table_rows"]:
            row["row_id"] = row_id_mapping[row["row_id"]]
        for finding in packet["tool_findings"]:
            for evidence_ref in finding.get("evidence_refs", []):
                for original_row_id, mapped_row_id in row_id_mapping.items():
                    original_ref_id = f"{report_id}:{original_row_id}"
                    if evidence_ref.get("ref_id") == original_ref_id:
                        evidence_ref["ref_id"] = f"{report_id}:{mapped_row_id}"
        packet["candidate_summary"]["reason_for_candidate"] += " Account labels use an alternate reporting map."
        return packet
    if slot_id == "V8_language_or_account_label_style":
        packet["candidate_summary"]["reason_for_candidate"] = (
            "Doanh thu and the linked balance sheet account move in different directions: "
            f"{packet['candidate_summary']['reason_for_candidate']}"
        )
        return packet
    raise ValueError(f"unsupported canonical variant slot: {slot_id}")


def _canonical_target_id(scenario: ScenarioDefinition) -> str:
    return f"{scenario.family}__{scenario.target_support_level}"


def _variant_pattern_id(clean_report: dict[str, Any], variant_slot: VariantSlotDefinition) -> str:
    report_profile = clean_report.get("report_profile") or "standard_corporate"
    return f"{variant_slot.slot_id}_{report_profile}_v1"


def _evidence_profile_for_support_target(
    target_support_level: str,
    *,
    variant_slot: VariantSlotDefinition,
) -> dict[str, Any]:
    if target_support_level == "insufficient_evidence":
        return {
            "has_table_evidence": True,
            "has_note_evidence": False,
            "has_variance_explanation": False,
            "has_tool_findings": False,
            "has_contradicting_evidence": False,
            "has_missing_required_evidence": True,
            "evidence_types": ["table_row"],
        }
    if variant_slot.slot_id == "V4_hard_missing_required_context":
        return {
            "has_table_evidence": True,
            "has_note_evidence": True,
            "has_variance_explanation": False,
            "has_tool_findings": False,
            "has_contradicting_evidence": False,
            "has_missing_required_evidence": True,
            "evidence_types": ["table_row", "note_span"],
        }
    return {
        "has_table_evidence": True,
        "has_note_evidence": True,
        "has_variance_explanation": variant_slot.slot_id == "V2_easy_quantitative_contradiction",
        "has_tool_findings": True,
        "has_contradicting_evidence": (
            target_support_level == "not_supported"
            or variant_slot.slot_id == "V6_tool_finding_contradiction"
        ),
        "has_missing_required_evidence": False,
        "evidence_types": (
            ["table_row", "note_span", "variance_explanation_span"]
            if variant_slot.slot_id == "V2_easy_quantitative_contradiction"
            else ["table_row", "note_span"]
        ),
    }


def _revenue_receivables_packet(
    *,
    clean_report: dict[str, Any],
    synthetic_report_id: str,
    revenue: dict[str, Any],
    receivables: dict[str, Any],
    synthetic_receivables_value: int | float,
    target_support_level: str,
) -> dict[str, Any]:
    revenue_row = _visible_row(synthetic_report_id, revenue)
    receivables_row = _visible_row(
        synthetic_report_id,
        {**receivables, "current_value": synthetic_receivables_value},
    )
    revenue_growth = _growth_ratio(revenue["current_value"], revenue["prior_value"])
    receivables_growth = _growth_ratio(synthetic_receivables_value, receivables["prior_value"])
    growth_delta_pct_points = (receivables_growth - revenue_growth) * 100
    signal_id = "receivables_growth_outpaces_revenue"
    threshold_exceeded = receivables_growth - revenue_growth > 0.10
    weak_target = target_support_level == "weakly_supported"
    candidate_reason = (
        f"Trade receivables increased {receivables_growth:.1%} while revenue increased "
        f"{revenue_growth:.1%}."
    )
    if weak_target:
        candidate_reason += (
            " The divergence is only slightly above the threshold and collection context partially mitigates it."
        )
    relevant_notes = [
        {
            "note_id": note["note_id"],
            "report_id": synthetic_report_id,
            "note_type": note["note_type"],
            "text": note["text"],
        }
        for note in clean_report["structured_evidence"]["notes"]
    ]
    if weak_target and relevant_notes:
        relevant_notes[0]["text"] = (
            f"{relevant_notes[0]['text']} Subsequent collection after period end partly reduced the balance, "
            "but the remaining aging detail is limited."
        )
    return {
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
            "reason_for_candidate": candidate_reason,
            "priority": "medium" if weak_target else "high",
            "supporting_signal_ids": [signal_id],
        },
        "relevant_table_rows": [revenue_row, receivables_row],
        "relevant_notes": relevant_notes,
        "relevant_variance_explanations": [],
        "tool_findings": [
            {
                "tool_result_id": "TOOL_REV_REC_DIVERGENCE_001",
                "tool_name": "receivables_vs_revenue_growth_tool",
                "risk_category": RISK_CATEGORY,
                "signal_id": signal_id,
                "flag": threshold_exceeded,
                "metric": "receivables_growth_minus_revenue_growth_pct_points",
                "metric_value": round(growth_delta_pct_points, 1),
                "threshold": "flag if receivables growth exceeds revenue growth by more than 10 percentage points",
                "finding_summary": _threshold_finding_summary(
                    subject="Trade receivables growth",
                    comparator="revenue growth",
                    metric_value=growth_delta_pct_points,
                    exceeded=threshold_exceeded,
                ),
                "calculation_basis": {
                    "revenue_growth_pct": round(revenue_growth * 100, 1),
                    "receivables_growth_pct": round(receivables_growth * 100, 1),
                    "period_basis": clean_report["period"],
                },
                "evidence_refs": [
                    {
                        "evidence_ref_type": "table_row",
                        "ref_id": f"{synthetic_report_id}:{revenue_row['row_id']}",
                    },
                    {
                        "evidence_ref_type": "table_row",
                        "ref_id": f"{synthetic_report_id}:{receivables_row['row_id']}",
                    },
                ],
            }
        ],
        "rules": [
            {
                "rule_id": "RULE_REV_REC_DIVERGENCE",
                "related_signal_ids": [signal_id],
                "risk_category": RISK_CATEGORY,
                "description": "Receivables growth above revenue growth may support a revenue quality risk signal.",
            }
        ],
        "constraints": {
            "allowed_decisions": ["supported", "weakly_supported", "not_supported", "insufficient_evidence"],
            "evidence_must_reference_provided_ids": True,
            "do_not_claim_fraud": True,
            "max_rationale_sentences": 3,
        },
    }


def _visible_row(report_id: str, source_row: dict[str, Any]) -> dict[str, Any]:
    row_id = source_row["row_id"]
    return {
        "row_id": row_id,
        "report_id": report_id,
        "standard_account": source_row["standard_account"],
        "values": {
            "current": {"cell_id": f"{row_id}_CURRENT", "value": source_row["current_value"]},
            "prior": {"cell_id": f"{row_id}_PRIOR", "value": source_row["prior_value"]},
        },
    }


def _validate_clean_report(clean_report: Any, *, scenario: ScenarioDefinition) -> None:
    if not isinstance(clean_report, dict):
        raise ValueError("clean report artifact must be a JSON object")
    required_fields = {
        "artifact_id",
        "source_document_id",
        "report_id",
        "company_key",
        "company_name",
        "ticker",
        "period_key",
        "period",
        "report_period_type",
        "report_profile",
        "currency",
        "unit",
        "language",
        "traceability",
        "structured_evidence",
    }
    missing_fields = sorted(required_fields - clean_report.keys())
    if missing_fields:
        raise ValueError(f"clean report artifact is missing required fields: {missing_fields}")
    evidence = clean_report["structured_evidence"]
    if not isinstance(evidence, dict) or not isinstance(evidence.get("rows"), list):
        raise ValueError("clean report artifact is missing required structured evidence rows")
    if not isinstance(evidence.get("notes"), list):
        raise ValueError("clean report artifact is missing required structured evidence notes")
    traceability = clean_report["traceability"]
    required_traceability_fields = {"source_file_sha256", "normalized_text_hash", "table_content_hash"}
    if not isinstance(traceability, dict) or any(not traceability.get(field) for field in required_traceability_fields):
        raise ValueError("clean report artifact is missing required source traceability hashes")
    rows_by_account = {
        row.get("standard_account"): row
        for row in evidence["rows"]
        if isinstance(row, dict)
    }
    required_accounts = _required_accounts_for_scenario(clean_report, scenario)
    for account in required_accounts:
        row = rows_by_account.get(account)
        if not row:
            raise ValueError(f"clean report artifact is missing required structured evidence account: {account}")
        if not _valid_clean_report_row(row):
            raise ValueError(f"clean report artifact has invalid structured evidence account: {account}")


def _valid_clean_report_row(row: dict[str, Any] | None) -> bool:
    return bool(
        row
        and row.get("row_id")
        and _is_number(row.get("current_value"))
        and _is_number(row.get("prior_value"))
        and row["prior_value"] != 0
    )


def _profile_account_roles(clean_report: dict[str, Any]) -> dict[str, Any]:
    return PROFILE_ACCOUNT_ROLES.get(clean_report.get("report_profile"), PROFILE_ACCOUNT_ROLES["standard_corporate"])


def _rows_by_profile_role(
    clean_report: dict[str, Any],
    rows_by_account: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    roles = _profile_account_roles(clean_report)
    return {
        "revenue": rows_by_account[roles["revenue"]],
        "receivables": rows_by_account[roles["receivables"]],
        "earnings": rows_by_account[roles["earnings"]],
        "cashflow": rows_by_account[roles["cashflow"]],
    }


def _validate_generated_packet(packet: dict[str, Any]) -> None:
    if contains_prohibited_detector_visible_payload(packet):
        raise ValueError("generated DetectorPacket contains prohibited hidden or raw metadata")
    validate_detector_packet(packet)
    risk_category = packet["task"]["risk_category"]
    if any(finding.get("risk_category") != risk_category for finding in packet["tool_findings"]):
        raise ValueError("generated DetectorPacket tool finding risk category mismatch")
    if any(rule.get("risk_category") != risk_category for rule in packet["rules"]):
        raise ValueError("generated DetectorPacket rule risk category mismatch")
    visible_ids = visible_packet_evidence_ids(packet)
    for finding in packet["tool_findings"]:
        for evidence_ref in finding.get("evidence_refs", []):
            if evidence_ref.get("ref_id") not in visible_ids:
                raise ValueError("generated DetectorPacket tool finding references missing visible evidence")


def _generic_thesis_family_packet(
    *,
    clean_report: dict[str, Any],
    synthetic_report_id: str,
    scenario: ScenarioDefinition,
    rows_by_account: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    specs = _generic_family_specs(rows_by_account=rows_by_account, scenario=scenario)
    spec = specs[scenario.family]
    visible_rows = [_visible_row(synthetic_report_id, row) for row in spec["rows"]]
    threshold_exceeded = scenario.target_support_level != "not_supported"
    packet = {
        "packet_id": f"PACKET_{synthetic_report_id}",
        "candidate_id": f"CAND_{synthetic_report_id}",
        "report_id": synthetic_report_id,
        "task": {
            "risk_category": scenario.risk_category,
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
            "reason_for_candidate": spec["candidate_reason"],
            "priority": "high" if scenario.target_support_level == "supported" else "medium",
            "supporting_signal_ids": [spec["signal_id"]],
        },
        "relevant_table_rows": visible_rows,
        "relevant_notes": [
            {
                "note_id": f"NOTE_{synthetic_report_id}_{spec['tag']}",
                "report_id": synthetic_report_id,
                "note_type": spec["note_type"],
                "text": spec["note_text"],
            }
        ],
        "relevant_variance_explanations": [],
        "tool_findings": [
            {
                "tool_result_id": f"TOOL_{synthetic_report_id}_{spec['tag']}",
                "tool_name": spec["tool_name"],
                "risk_category": scenario.risk_category,
                "signal_id": spec["signal_id"],
                "flag": threshold_exceeded,
                "metric": spec["metric"],
                "metric_value": spec["metric_value"],
                "threshold": spec["threshold"],
                "finding_summary": (
                    spec["supported_summary"] if threshold_exceeded else spec["not_supported_summary"]
                ),
                "calculation_basis": {"period_basis": clean_report["period"], **spec["calculation_basis"]},
                "evidence_refs": [
                    {"evidence_ref_type": "table_row", "ref_id": f"{synthetic_report_id}:{row['row_id']}"}
                    for row in visible_rows
                ],
            }
        ],
        "rules": [
            {
                "rule_id": f"RULE_{spec['tag']}",
                "related_signal_ids": [spec["signal_id"]],
                "risk_category": scenario.risk_category,
                "description": spec["rule_description"],
            }
        ],
        "constraints": {
            "allowed_decisions": ["supported", "weakly_supported", "not_supported", "insufficient_evidence"],
            "evidence_must_reference_provided_ids": True,
            "do_not_claim_fraud": True,
            "max_rationale_sentences": 3,
        },
    }
    return packet, {
        "modified_standard_account": spec["modified_standard_account"],
        "original_value": spec["original_value"],
        "synthetic_value": spec["synthetic_value"],
    }


def _generic_family_specs(*, rows_by_account: dict[str, dict[str, Any]], scenario: ScenarioDefinition) -> dict[str, dict[str, Any]]:
    revenue = rows_by_account["revenue"]
    receivables = rows_by_account["trade_receivables"]
    revenue_current = revenue["current_value"]
    revenue_prior = revenue["prior_value"]
    revenue_growth = _growth_ratio(revenue_current, revenue_prior)
    receivables_current = max(receivables["current_value"], round(revenue_current * scenario.synthetic_multiplier))
    receivables_prior = receivables["prior_value"]
    threshold_exceeded = scenario.target_support_level != "not_supported"
    weak_target = scenario.target_support_level == "weakly_supported"
    inventory_prior = max(round(revenue_prior * 0.45), 1)
    inventory_current = round(revenue_current * (scenario.synthetic_multiplier if threshold_exceeded else 0.42))
    expense_prior = max(round(revenue_prior * 0.60), 1)
    expense_current = round(revenue_current * (0.18 if threshold_exceeded else 0.62))
    soft_assets_prior = max(round(revenue_prior * 0.55), 1)
    soft_assets_current = round(revenue_current * (scenario.synthetic_multiplier if threshold_exceeded else 0.58))
    total_assets_current = max(round(revenue_current * 2.0), 1)
    related_party_current = round(total_assets_current * (0.09 if threshold_exceeded else 0.02))
    related_party_prior = max(round(total_assets_current * 0.01), 1)
    note_value = round(receivables_current * (1.35 if threshold_exceeded else 1.01))
    note_table_mismatch_pct = abs(note_value - receivables_current) / max(abs(receivables_current), 1) * 100
    allowance_current = max(round(receivables_current * (0.01 if threshold_exceeded else 0.08)), 1)
    allowance_prior = max(round(receivables_prior * 0.08), 1)
    if weak_target:
        receivables_current = max(receivables_current, round(receivables_prior * (1 + max(revenue_growth + 0.08, 0.12))))
        allowance_current = max(round(receivables_current * 0.027), 1)
        inventory_current = round(inventory_prior * (1 + revenue_growth + 0.17))
        expense_current = max(round(expense_prior * max(1 + revenue_growth - 0.22, 0.05)), 1)
        soft_assets_current = round(soft_assets_prior * 1.365)
        related_party_current = round(total_assets_current * 0.053)
        note_value = round(receivables_current * 1.07)
        note_table_mismatch_pct = abs(note_value - receivables_current) / max(abs(receivables_current), 1) * 100

    return {
        "receivables_credit_quality": {
            "tag": "AR_QUALITY",
            "signal_id": "allowance_ratio_declines_while_receivables_grow",
            "tool_name": "receivables_credit_quality_injection_tool",
            "metric": "allowance_to_receivables_ratio_pct",
            "metric_value": round(allowance_current / max(receivables_current, 1) * 100, 1),
            "threshold": "flag if allowance ratio is below 3 percent while receivables grow materially",
            "candidate_reason": "Receivables increased materially while the allowance ratio stayed low.",
            "supported_summary": (
                "Allowance coverage stayed just below the configured threshold while receivables grew materially."
                if weak_target
                else "Allowance coverage stayed below the configured threshold while receivables grew materially."
            ),
            "not_supported_summary": "Allowance coverage did not fall below the configured low-coverage threshold.",
            "rule_description": "Low allowance coverage with material receivables growth may support a receivables quality risk signal.",
            "note_type": "receivables_note",
            "note_text": "The receivables note provides limited aging detail and does not explain the low allowance coverage.",
            "rows": [
                _synthetic_clean_row("ROW_TRADE_RECEIVABLES", "trade_receivables", receivables_current, receivables_prior),
                _synthetic_clean_row("ROW_RECEIVABLES_ALLOWANCE", "receivables_allowance", allowance_current, allowance_prior),
            ],
            "calculation_basis": {"receivables_current": receivables_current},
            "modified_standard_account": "receivables_allowance",
            "original_value": allowance_prior,
            "synthetic_value": allowance_current,
        },
        "inventory_cost_asset_flow": {
            "tag": "INV_COST",
            "signal_id": "inventory_growth_outpaces_revenue",
            "tool_name": "inventory_cost_asset_flow_injection_tool",
            "metric": "inventory_growth_minus_revenue_growth_pct_points",
            "metric_value": round((_growth_ratio(inventory_current, inventory_prior) - _growth_ratio(revenue_current, revenue_prior)) * 100, 1),
            "threshold": "flag if inventory growth exceeds revenue growth by more than 15 percentage points",
            "candidate_reason": "Inventory increased faster than revenue while provision detail stayed limited.",
            "supported_summary": (
                "Inventory growth modestly exceeded revenue growth by more than the configured threshold."
                if weak_target
                else "Inventory growth exceeded revenue growth by more than the configured threshold."
            ),
            "not_supported_summary": "Inventory growth did not exceed revenue growth by the configured threshold.",
            "rule_description": "Inventory growth above revenue growth may support an inventory or cost-flow risk signal.",
            "note_type": "inventory_note",
            "note_text": "The inventory note does not provide a specific write-down or provision explanation for the increase.",
            "rows": [
                _synthetic_clean_row("ROW_REVENUE", "revenue", revenue_current, revenue_prior),
                _synthetic_clean_row("ROW_INVENTORY", "inventory", inventory_current, inventory_prior),
            ],
            "calculation_basis": {"revenue_current": revenue_current},
            "modified_standard_account": "inventory",
            "original_value": inventory_prior,
            "synthetic_value": inventory_current,
        },
        "expense_liability_understatement": {
            "tag": "EXP_LIAB",
            "signal_id": "expense_growth_lags_revenue_growth",
            "tool_name": "expense_liability_understatement_injection_tool",
            "metric": "revenue_growth_minus_expense_growth_pct_points",
            "metric_value": round((_growth_ratio(revenue_current, revenue_prior) - _growth_ratio(expense_current, expense_prior)) * 100, 1),
            "threshold": "flag if revenue growth exceeds expense growth by more than 20 percentage points without explanation",
            "candidate_reason": "Revenue growth materially outpaced major operating expense growth.",
            "supported_summary": (
                "Revenue growth modestly exceeded expense growth by more than the configured threshold."
                if weak_target
                else "Revenue growth exceeded expense growth by more than the configured threshold."
            ),
            "not_supported_summary": "Expense growth remained broadly consistent with revenue growth.",
            "rule_description": "Major expense growth lagging revenue growth may support an expense or liability understatement risk signal.",
            "note_type": "expense_note",
            "note_text": "The expense note does not provide a specific explanation for the divergence from revenue growth.",
            "rows": [
                _synthetic_clean_row("ROW_REVENUE", "revenue", revenue_current, revenue_prior),
                _synthetic_clean_row("ROW_OPERATING_EXPENSES", "operating_expenses", expense_current, expense_prior),
            ],
            "calculation_basis": {"expense_current": expense_current},
            "modified_standard_account": "operating_expenses",
            "original_value": expense_prior,
            "synthetic_value": expense_current,
        },
        "asset_quality_valuation": {
            "tag": "ASSET_VAL",
            "signal_id": "soft_assets_growth_high_without_impairment_explanation",
            "tool_name": "asset_quality_valuation_injection_tool",
            "metric": "soft_assets_growth_pct",
            "metric_value": round(_growth_ratio(soft_assets_current, soft_assets_prior) * 100, 1),
            "threshold": "flag if soft assets grow more than 35 percent without impairment or valuation explanation",
            "candidate_reason": (
                "Soft assets increased only slightly above the threshold while valuation explanation remained limited."
                if weak_target
                else "Soft assets increased sharply while valuation explanation remained limited."
            ),
            "supported_summary": (
                "Soft-asset growth only slightly exceeded the configured threshold while disclosure gives partial context."
                if weak_target
                else "Soft-asset growth exceeded the configured threshold without a specific valuation explanation."
            ),
            "not_supported_summary": "Soft-asset growth did not exceed the configured high-growth threshold.",
            "rule_description": "Sharp growth in soft assets without valuation explanation may support an asset quality risk signal.",
            "note_type": "asset_note",
            "note_text": (
                "The asset note says additions relate partly to software and project costs, but does not provide "
                "a specific impairment or valuation explanation."
                if weak_target
                else "The asset note does not provide a specific impairment or valuation explanation for the increase."
            ),
            "rows": [
                _synthetic_clean_row("ROW_SOFT_ASSETS", "soft_assets", soft_assets_current, soft_assets_prior),
                _synthetic_clean_row("ROW_TOTAL_ASSETS", "total_assets", total_assets_current, round(total_assets_current * 0.85)),
            ],
            "calculation_basis": {"soft_assets_current": soft_assets_current},
            "modified_standard_account": "soft_assets",
            "original_value": soft_assets_prior,
            "synthetic_value": soft_assets_current,
        },
        "related_party_disclosure": {
            "tag": "RP_DISC",
            "signal_id": "related_party_balance_material_terms_vague",
            "tool_name": "related_party_disclosure_injection_tool",
            "metric": "related_party_balance_to_total_assets_pct",
            "metric_value": round(related_party_current / total_assets_current * 100, 1),
            "threshold": "flag if related-party balance exceeds 5 percent of total assets and terms are vague",
            "candidate_reason": (
                "Related-party receivables were just above materiality while transaction terms were only partly described."
                if weak_target
                else "Related-party receivables were material while transaction terms were vague."
            ),
            "supported_summary": (
                "Related-party balance only slightly exceeded the configured materiality threshold and terms were partly vague."
                if weak_target
                else "Related-party balance exceeded the configured materiality threshold and terms were vague."
            ),
            "not_supported_summary": "Related-party balance did not exceed the configured materiality threshold.",
            "rule_description": "Material related-party balances with vague terms may support a related-party disclosure risk signal.",
            "note_type": "related_party_note",
            "note_text": (
                "The related-party note lists balances and short-term settlement intent, but gives limited pricing "
                "and transaction-term detail."
                if weak_target
                else "The related-party note lists balances but gives limited transaction terms and settlement detail."
            ),
            "rows": [
                _synthetic_clean_row("ROW_TOTAL_ASSETS", "total_assets", total_assets_current, round(total_assets_current * 0.85)),
                _synthetic_clean_row("ROW_RELATED_PARTY_RECEIVABLES", "related_party_receivables", related_party_current, related_party_prior),
            ],
            "calculation_basis": {"total_assets_current": total_assets_current},
            "modified_standard_account": "related_party_receivables",
            "original_value": related_party_prior,
            "synthetic_value": related_party_current,
        },
        "disclosure_inconsistency": {
            "tag": "DISC_OBF",
            "signal_id": "note_table_value_mismatch",
            "tool_name": "disclosure_consistency_injection_tool",
            "metric": "note_table_mismatch_pct",
            "metric_value": round(note_table_mismatch_pct, 1),
            "threshold": "flag if note and table values differ by more than 5 percent after unit tolerance",
            "candidate_reason": "The note value materially diverged from the linked statement table value.",
            "supported_summary": (
                "The note value modestly differed from the table value by more than the configured tolerance."
                if weak_target
                else "The note value differed from the table value by more than the configured tolerance."
            ),
            "not_supported_summary": "The note and table values stayed within the configured tolerance.",
            "rule_description": "Material note-table mismatch may support a disclosure inconsistency risk signal.",
            "note_type": "disclosure_consistency_note",
            "note_text": f"The receivables note states a balance of {note_value}.",
            "rows": [
                _synthetic_clean_row("ROW_TRADE_RECEIVABLES", "trade_receivables", receivables_current, receivables_prior),
            ],
            "calculation_basis": {"note_value": note_value, "table_value": receivables_current},
            "modified_standard_account": "trade_receivables_note_value",
            "original_value": receivables_current,
            "synthetic_value": note_value,
        },
    }


def _synthetic_clean_row(row_id: str, standard_account: str, current_value: int | float, prior_value: int | float) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "standard_account": standard_account,
        "current_value": current_value,
        "prior_value": prior_value if prior_value != 0 else 1,
    }


def _earnings_cashflow_packet(
    *,
    clean_report: dict[str, Any],
    synthetic_report_id: str,
    profit: dict[str, Any],
    operating_cash_flow: dict[str, Any],
    synthetic_cash_flow_value: int | float,
) -> dict[str, Any]:
    profit_row = _visible_row(synthetic_report_id, profit)
    cash_flow_row = _visible_row(
        synthetic_report_id,
        {**operating_cash_flow, "current_value": synthetic_cash_flow_value},
    )
    profit_growth = _growth_ratio(profit["current_value"], profit["prior_value"])
    cash_flow_growth = _growth_ratio(synthetic_cash_flow_value, operating_cash_flow["prior_value"])
    conversion_ratio = synthetic_cash_flow_value / profit["current_value"]
    signal_id = "profit_growth_outpaces_operating_cash_flow"
    threshold_exceeded = conversion_ratio < 0.80 and profit_growth > 0
    return {
        "packet_id": f"PACKET_{synthetic_report_id}",
        "candidate_id": f"CAND_{synthetic_report_id}",
        "report_id": synthetic_report_id,
        "task": {
            "risk_category": EARNINGS_CASHFLOW_RISK_CATEGORY,
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
            "reason_for_candidate": (
                f"Profit increased {profit_growth:.1%} while operating cash flow changed "
                f"{cash_flow_growth:.1%}."
            ),
            "priority": "high",
            "supporting_signal_ids": [signal_id],
        },
        "relevant_table_rows": [profit_row, cash_flow_row],
        "relevant_notes": [
            {
                "note_id": note["note_id"],
                "report_id": synthetic_report_id,
                "note_type": note["note_type"],
                "text": note["text"],
            }
            for note in clean_report["structured_evidence"]["notes"]
        ],
        "relevant_variance_explanations": [],
        "tool_findings": [
            {
                "tool_result_id": "TOOL_EARN_CASH_DIVERGENCE_001",
                "tool_name": "earnings_vs_operating_cash_flow_tool",
                "risk_category": EARNINGS_CASHFLOW_RISK_CATEGORY,
                "signal_id": signal_id,
                "flag": threshold_exceeded,
                "metric": "operating_cash_flow_to_profit_ratio",
                "metric_value": round(conversion_ratio, 2),
                "threshold": "flag if operating cash flow is below 80 percent of profit while profit grows",
                "finding_summary": _cash_flow_finding_summary(
                    conversion_ratio=conversion_ratio,
                    profit_growth=profit_growth,
                    threshold_exceeded=threshold_exceeded,
                ),
                "calculation_basis": {
                    "profit_growth_pct": round(profit_growth * 100, 1),
                    "operating_cash_flow_growth_pct": round(cash_flow_growth * 100, 1),
                    "period_basis": clean_report["period"],
                },
                "evidence_refs": [
                    {
                        "evidence_ref_type": "table_row",
                        "ref_id": f"{synthetic_report_id}:{profit_row['row_id']}",
                    },
                    {
                        "evidence_ref_type": "table_row",
                        "ref_id": f"{synthetic_report_id}:{cash_flow_row['row_id']}",
                    },
                ],
            }
        ],
        "rules": [
            {
                "rule_id": "RULE_EARN_CASH_DIVERGENCE",
                "related_signal_ids": [signal_id],
                "risk_category": EARNINGS_CASHFLOW_RISK_CATEGORY,
                "description": "Profit growth with weak operating cash conversion may support an earnings quality risk signal.",
            }
        ],
        "constraints": {
            "allowed_decisions": ["supported", "weakly_supported", "not_supported", "insufficient_evidence"],
            "evidence_must_reference_provided_ids": True,
            "do_not_claim_fraud": True,
            "max_rationale_sentences": 3,
        },
    }


def _threshold_finding_summary(*, subject: str, comparator: str, metric_value: float, exceeded: bool) -> str:
    if exceeded:
        if metric_value <= 15:
            return f"{subject} modestly exceeded {comparator} by {metric_value:.1f} percentage points."
        return f"{subject} exceeded {comparator} by {metric_value:.1f} percentage points."
    return f"{subject} did not exceed {comparator} by more than the configured threshold."


def _cash_flow_finding_summary(*, conversion_ratio: float, profit_growth: float, threshold_exceeded: bool) -> str:
    if threshold_exceeded:
        return (
            f"Operating cash flow was {conversion_ratio:.1%} of current profit while profit growth was "
            f"{profit_growth:.1%}."
        )
    return "Operating cash flow did not fall below the configured low-conversion threshold while profit grew."


def _synthetic_report_id(base_report_id: str, tag: str, suffix: str) -> str:
    if not isinstance(base_report_id, str) or not base_report_id.endswith("_CLEAN"):
        raise ValueError("clean report artifact report_id must end with _CLEAN")
    return f"{base_report_id[:-len('_CLEAN')]}_{tag}_{suffix}"


def _growth_ratio(current: int | float, prior: int | float) -> float:
    return (current - prior) / abs(prior)


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _manifest(*, run_id: str, input_paths: list[Path], records: list[dict[str, Any]]) -> dict[str, Any]:
    first_record = records[0]
    generation = first_record["metadata"]["generation_metadata"]
    split = first_record["metadata"]["split_metadata"]
    coverage_counts = _coverage_counts(records)
    input_artifacts = [
        {"path": str(input_path), "sha256": _file_sha256(input_path)}
        for input_path in input_paths
    ]
    return {
        "run_id": run_id,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "created_by": "research.grounded_synthetic_packet_generator",
        "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
        "generator_version": GENERATOR_VERSION,
        "clean_report_artifact_path": str(input_paths[0]),
        "clean_report_artifact_sha256": input_artifacts[0]["sha256"],
        "clean_report_artifacts": input_artifacts,
        "scenario": {"scenario_id": generation["injection_scenario_id"], "version": SCENARIO_VERSION},
        "scenarios": _scenario_summaries(records),
        "records_written": len(records),
        "risk_category": generation["target_risk_category"],
        "risk_categories": sorted(coverage_counts["risk_categories"]),
        "target_support_levels": sorted(
            record["metadata"]["generation_metadata"]["target_support_level"] for record in records
        ),
        "report_profile": first_record["metadata"]["report_profile"],
        "report_profiles": sorted(coverage_counts["report_profiles"]),
        "base_group": split["derived_from_group_key"],
        "base_groups": sorted(coverage_counts["base_groups"]),
        "synthetic_groups": [record["metadata"]["split_metadata"]["group_key"] for record in records],
        "coverage_counts": coverage_counts,
        "artifact_policy": {
            "detector_assessments_stored": False,
            "teacher_outputs_stored": False,
            "judge_outputs_stored": False,
            "final_report_text_stored": False,
            "sft_messages_stored": False,
            "labels_stored": False,
        },
    }


def _metrics(*, run_id: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    first_generation = records[0]["metadata"]["generation_metadata"]
    return {
        "run_id": run_id,
        "scenario_id": first_generation["injection_scenario_id"],
        "scenarios": _scenario_summaries(records),
        "scenario_version": SCENARIO_VERSION,
        "generated_records": len(records),
        **_coverage_counts(records),
    }


def _scenario_summaries(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    scenario_ids = sorted(
        {
            record["metadata"]["generation_metadata"]["injection_scenario_id"]
            for record in records
        }
    )
    return [{"scenario_id": scenario_id, "version": SCENARIO_VERSION} for scenario_id in scenario_ids]


def _coverage_counts(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "risk_categories": dict(Counter(record["metadata"]["generation_metadata"]["target_risk_category"] for record in records)),
        "target_support_levels": dict(Counter(record["metadata"]["generation_metadata"]["target_support_level"] for record in records)),
        "report_profiles": dict(Counter(record["metadata"]["report_profile"] for record in records)),
        "source_groups": dict(
            Counter(
                record["metadata"]["split_metadata"].get("source_group_key")
                for record in records
                if record["metadata"]["split_metadata"].get("source_group_key")
            )
        ),
        "base_groups": dict(Counter(record["metadata"]["split_metadata"]["derived_from_group_key"] for record in records)),
        "synthetic_groups": dict(Counter(record["metadata"]["split_metadata"]["group_key"] for record in records)),
        "scenario_ids": dict(Counter(record["metadata"]["generation_metadata"]["injection_scenario_id"] for record in records)),
        "variant_slots": dict(Counter(record["metadata"]["generation_metadata"]["variant_slot_id"] for record in records)),
        "variant_patterns": dict(Counter(record["metadata"]["generation_metadata"]["variant_pattern_id"] for record in records)),
        "canonical_targets": dict(Counter(record["metadata"]["generation_metadata"]["canonical_target_id"] for record in records)),
        "rejected_invalid_reasons": {},
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate one grounded synthetic DetectorPacket staged candidate.")
    parser.add_argument("--clean-report-artifact", required=True, type=Path, action="append")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", default="grounded_synthetic_packet_generator")
    parser.add_argument("--scenario-family", choices=SUPPORTED_SCENARIO_FAMILIES, action="append")
    parser.add_argument("--require-rich-clean-report", action="store_true")
    parser.add_argument("--reserved-real-manual-release-dir", action="append", default=[], type=Path)
    args = parser.parse_args(argv)
    result = run_grounded_synthetic_packet_generator(
        clean_report_artifact=args.clean_report_artifact,
        output_dir=args.output_dir,
        run_id=args.run_id,
        scenario_family=args.scenario_family,
        require_rich_clean_report=args.require_rich_clean_report,
        reserved_real_manual_release_dir=args.reserved_real_manual_release_dir,
    )
    print(
        json.dumps(
            {
                "status": result.status,
                "output_dir": str(result.output_dir),
                "staged_jsonl": str(result.staged_jsonl_path),
                "records_written": result.records_written,
                "errors": result.errors,
            },
            sort_keys=True,
        )
    )
    return 0 if result.status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
