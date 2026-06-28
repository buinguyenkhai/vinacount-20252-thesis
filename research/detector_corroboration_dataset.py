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
    ScenarioDefinition,
    VariantSlotDefinition,
    _generate_record,
    _reserved_real_manual_identities,
    _rows_by_profile_role,
    _validate_clean_report_not_reserved,
    _validate_generated_packet,
    _visible_row,
)
from research.sft_exporter import (
    _record_to_chat,
    detector_sft_system_prompt,
)


ARTIFACT_CONTRACT_VERSION = "detector_corroboration_dataset_v2"
RISK_CATEGORY = "earnings_cashflow_mismatch"
SCENARIO_ID = "earnings_cashflow_corroboration_v2"
SYSTEM_PROMPT_VERSION = "v2_evidence_bundle"
PAIR_SPECS = (
    ("strong_negative_cfo_150", 1.50),
    ("strong_negative_cfo_250", 2.50),
)
ACCEPTED_CALIBRATION_TEACHER_PROMPTS = {
    CORROBORATION_STRICT_TEACHER_PROMPT_VERSION,
    CORROBORATION_BALANCED_TEACHER_PROMPT_VERSION,
}
ACCEPTED_CALIBRATION_JUDGE_PROMPTS = {
    CORROBORATION_STRICT_JUDGE_PROMPT_VERSION,
    CORROBORATION_BALANCED_JUDGE_PROMPT_VERSION,
}


@dataclass(frozen=True)
class DetectorCorroborationRawBuilderResult:
    status: str
    output_dir: Path
    raw_jsonl_path: Path
    manifest_path: Path
    metrics_path: Path
    records_written: int
    errors: list[str]


@dataclass(frozen=True)
class DetectorCorroborationSftComposerResult:
    status: str
    output_dir: Path
    sft_jsonl_path: Path
    manifest_path: Path
    metrics_path: Path
    records_written: int
    errors: list[str]


def run_detector_corroboration_raw_builder(
    *,
    training_clean_reports: list[Path | str],
    development_clean_reports: list[Path | str],
    output_dir: Path | str,
    reserved_real_manual_release_dirs: list[Path | str] | None = None,
) -> DetectorCorroborationRawBuilderResult:
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
                records.extend(_matched_records(clean_report, calibration_role=role))
        for record in records:
            validate_synthetic_raw_candidate(record)
    except (OSError, json.JSONDecodeError, ValueError, KeyError) as error:
        return DetectorCorroborationRawBuilderResult(
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
        "artifact_kind": "corroboration_calibration_raw_pool",
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
            "corroboration_boundary_encoded_as_evidence_roles": True,
        },
        "created_at": _created_at(),
    }
    _write_json(manifest_path, manifest)
    _write_json(
        metrics_path,
        {
            "status": "passed",
            "records_written": len(records),
            **counts,
        },
    )
    return DetectorCorroborationRawBuilderResult(
        "passed",
        output_path,
        raw_jsonl_path,
        manifest_path,
        metrics_path,
        len(records),
        [],
    )


def run_detector_corroboration_sft_composer(
    *,
    base_sft_jsonl: Path | str,
    gate_run_dirs: list[Path | str],
    output_dir: Path | str,
) -> DetectorCorroborationSftComposerResult:
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

        complete_records, pair_filter = _complete_pair_records(promoted_records)
        composed_rows = [_normalize_base_chat_row(row) for row in base_rows]
        composed_rows.extend(_calibration_chat_rows(complete_records))
        composed_rows.sort(key=_chat_sort_key)
    except (OSError, json.JSONDecodeError, ValueError, KeyError) as error:
        return DetectorCorroborationSftComposerResult(
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
        "artifact_kind": "corroboration_calibration_sft_corpus",
        "system_prompt_version": SYSTEM_PROMPT_VERSION,
        "base_sft_jsonl": str(base_path),
        "base_sft_jsonl_sha256": _file_sha256(base_path),
        "gate_artifacts": gate_artifacts,
        "sft_jsonl": sft_jsonl_path.name,
        "sft_jsonl_sha256": _file_sha256(sft_jsonl_path),
        "records_written": len(composed_rows),
        "reserved_test_rows_preserved": composed_test_rows,
        "pair_filter": {
            "complete_pairs_included": len(complete_records) // 2,
            **pair_filter,
        },
        "counts": counts,
        "experiment_boundary": {
            "base_training_rows_preserved": True,
            "base_validation_rows_preserved": True,
            "base_test_rows_preserved": True,
            "calibration_development_rows_added_to_validation": True,
            "real_manual_records_used_as_training_data": False,
            "synthetic_test_consumed": False,
        },
        "created_at": _created_at(),
    }
    _write_json(manifest_path, manifest)
    _write_json(
        metrics_path,
        {
            "status": "passed",
            "records_written": len(composed_rows),
            "complete_pairs_included": len(complete_records) // 2,
            **pair_filter,
            **counts,
        },
    )
    return DetectorCorroborationSftComposerResult(
        "passed",
        output_path,
        sft_jsonl_path,
        manifest_path,
        metrics_path,
        len(composed_rows),
        [],
    )


def _matched_records(clean_report: dict[str, Any], *, calibration_role: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for pair_name, magnitude in PAIR_SPECS:
        pair_id = f"{clean_report['report_id']}__{pair_name}"
        for target_support_level, suffix in [
            ("weakly_supported", f"{pair_name}_W"),
            ("supported", f"{pair_name}_S"),
        ]:
            scenario = ScenarioDefinition(
                family="earnings_cashflow",
                scenario_id=SCENARIO_ID,
                risk_category=RISK_CATEGORY,
                synthetic_report_tag="SYN_EARN_CORR",
                synthetic_suffix=suffix,
                target_support_level=target_support_level,
                synthetic_multiplier=-magnitude,
            )
            record = _generate_record(
                clean_report,
                scenario=scenario,
                variant_slot=VariantSlotDefinition(
                    slot_id="V1_easy_quantitative_clear",
                    pattern_id="corroboration_matched_pair_v2",
                    synthetic_suffix="V01",
                ),
            )
            records.append(
                _shape_matched_record(
                    record,
                    clean_report=clean_report,
                    calibration_role=calibration_role,
                    pair_id=pair_id,
                    magnitude=magnitude,
                    target_support_level=target_support_level,
                )
            )
    return records


def _shape_matched_record(
    record: dict[str, Any],
    *,
    clean_report: dict[str, Any],
    calibration_role: str,
    pair_id: str,
    magnitude: float,
    target_support_level: str,
) -> dict[str, Any]:
    shaped = deepcopy(record)
    packet = shaped["input"]["data"]
    report_id = packet["report_id"]
    packet["task"] = {
        "risk_category": RISK_CATEGORY,
        "question": "Does the provided evidence support an earnings and cash-flow mismatch risk signal?",
        "expected_output": "Return a structured DetectorAssessment using risk-signal language only.",
    }
    packet["relevant_notes"] = []
    packet["relevant_variance_explanations"] = []

    rows_by_account = {
        row["standard_account"]: row
        for row in clean_report["structured_evidence"]["rows"]
    }
    role_rows = _rows_by_profile_role(clean_report, rows_by_account)
    profit_account = role_rows["earnings"]["standard_account"]
    cashflow_account = role_rows["cashflow"]["standard_account"]
    profit_row = next(row for row in packet["relevant_table_rows"] if row["standard_account"] == profit_account)
    cashflow_row = next(row for row in packet["relevant_table_rows"] if row["standard_account"] == cashflow_account)
    profit_value = max(abs(float(role_rows["earnings"]["current_value"])), 100.0)
    prior_profit = max(round(profit_value * 0.72), 1)
    cashflow_value = -round(profit_value * magnitude)
    profit_row["values"]["current"]["value"] = profit_value
    profit_row["values"]["prior"]["value"] = prior_profit
    cashflow_row["values"]["current"]["value"] = cashflow_value

    primary_signal_id = "positive_profit_negative_operating_cash_flow"
    primary_finding = packet["tool_findings"][0]
    primary_finding.update(
        {
            "tool_result_id": f"TOOL_{report_id}_EARN_CASH",
            "tool_name": "earnings_cashflow_mismatch_tool",
            "risk_category": RISK_CATEGORY,
            "signal_id": primary_signal_id,
            "flag": True,
            "metric": "operating_cash_flow_to_profit_ratio",
            "metric_name": "operating_cash_flow_to_profit_ratio",
            "metric_value": -magnitude,
            "value": -magnitude,
            "strength": "strong",
            "trigger_strength": "strong",
            "evidence_role": "primary_trigger",
            "independent_corroboration_present": False,
            "corroboration_evidence_refs": [],
            "corroborates_signal_ids": [],
            "threshold": "flag when operating cash flow is below 70 percent of positive profit",
            "finding_summary": (
                f"Operating cash flow was {-magnitude:.2f} times positive profit, indicating a strong "
                "cash-conversion mismatch trigger."
            ),
            "evidence_refs": [
                {
                    "evidence_ref_type": "table_row",
                    "ref_id": f"{report_id}:{profit_row['row_id']}",
                },
                {
                    "evidence_ref_type": "table_row",
                    "ref_id": f"{report_id}:{cashflow_row['row_id']}",
                },
            ],
        }
    )
    isolated_trigger_summary = (
        "The packet shows positive profit and strongly negative operating cash flow. "
        "Support must be calibrated from the complete evidence bundle; the profit and cash-flow "
        "rows are inputs to the same trigger, not independent corroboration."
    )
    packet["candidate_summary"] = {
        "reason_for_candidate": isolated_trigger_summary,
        "priority": "high",
        "supporting_signal_ids": [primary_signal_id],
    }
    packet["rules"] = [
        {
            "rule_id": f"RULE_{report_id}_EVIDENCE_BUNDLE",
            "rule_name": "Assess complete evidence-bundle sufficiency",
            "risk_category": RISK_CATEGORY,
            "related_signal_ids": [primary_signal_id],
            "description": (
                "Assess whether the complete visible evidence bundle contains an isolated primary trigger "
                "or a distinct corroborating signal. Trigger magnitude describes the primary computation; "
                "the table rows used as inputs to one tool finding are not independent corroboration."
            ),
        }
    ]
    packet["constraints"]["evidence_bundle_semantics"] = {
        "tool_finding_strength_scope": "trigger_magnitude_only",
        "trigger_inputs_are_not_independent_corroboration": True,
        "independent_corroboration_requires_distinct_signal_or_disclosure": True,
    }

    if target_support_level == "supported":
        corroborating_row_source = deepcopy(role_rows["receivables"])
        corroborating_prior = max(abs(float(corroborating_row_source["prior_value"])), 100.0)
        corroborating_row_source["prior_value"] = corroborating_prior
        corroborating_row_source["current_value"] = round(corroborating_prior * 1.60)
        corroborating_row = _visible_row(report_id, corroborating_row_source)
        packet["relevant_table_rows"].append(corroborating_row)
        secondary_signal_id = "balance_growth_corroborates_cash_conversion_pressure"
        secondary_tool_result_id = f"TOOL_{report_id}_BALANCE_GROWTH"
        primary_finding["independent_corroboration_present"] = True
        primary_finding["corroboration_evidence_refs"] = [
            {
                "evidence_ref_type": "tool_result",
                "ref_id": secondary_tool_result_id,
            },
            {
                "evidence_ref_type": "table_row",
                "ref_id": f"{report_id}:{corroborating_row['row_id']}",
            },
        ]
        packet["tool_findings"].append(
            {
                "tool_result_id": secondary_tool_result_id,
                "tool_name": "cash_conversion_corroboration_tool",
                "risk_category": RISK_CATEGORY,
                "signal_id": secondary_signal_id,
                "flag": True,
                "metric": "linked_balance_growth",
                "metric_name": "linked_balance_growth",
                "metric_value": 0.60,
                "value": 0.60,
                "strength": "moderate",
                "trigger_strength": "moderate",
                "evidence_role": "independent_corroboration",
                "independent_corroboration_present": True,
                "corroboration_evidence_refs": [
                    {
                        "evidence_ref_type": "tool_result",
                        "ref_id": primary_finding["tool_result_id"],
                    }
                ],
                "corroborates_signal_ids": [primary_signal_id],
                "threshold": "corroborate when the linked operating balance grows by more than 30 percent",
                "finding_summary": (
                    "The linked operating balance increased 60 percent, independently corroborating "
                    "cash-conversion pressure."
                ),
                "evidence_refs": [
                    {
                        "evidence_ref_type": "table_row",
                        "ref_id": f"{report_id}:{corroborating_row['row_id']}",
                    },
                ],
            }
        )
        packet["candidate_summary"]["supporting_signal_ids"].append(secondary_signal_id)
        packet["candidate_summary"]["reason_for_candidate"] = (
            f"{isolated_trigger_summary} The packet also contains a separate corroboration "
            "tool finding with its own signal_id and a distinct linked-balance row; this second "
            "visible signal independently corroborates cash-conversion pressure."
        )
        packet["rules"][0]["related_signal_ids"].append(secondary_signal_id)
        packet["rules"][0]["description"] = (
            "Assess the complete visible evidence bundle. Trigger magnitude alone does not establish "
            "full support, and the profit and cash-flow rows used by the primary trigger are not "
            "independent corroboration. When a separate corroboration tool finding cites a distinct "
            "linked operating-balance row and aligns with the primary cash-conversion trigger, the "
            "two visible signals support the earnings and cash-flow mismatch risk signal."
        )

    shaped["metadata"]["risk_category"] = RISK_CATEGORY
    shaped["metadata"]["evidence_profile"] = {
        "has_table_evidence": True,
        "has_note_evidence": False,
        "has_variance_explanation": False,
        "has_tool_findings": True,
        "has_contradicting_evidence": False,
        "has_missing_required_evidence": target_support_level == "weakly_supported",
        "evidence_types": ["table_row", "tool_result"],
    }
    generation = shaped["metadata"]["generation_metadata"]
    generation.update(
        {
            "injection_scenario_id": SCENARIO_ID,
            "target_risk_category": RISK_CATEGORY,
            "target_support_level": target_support_level,
            "canonical_target_id": f"earnings_cashflow_corroboration__{target_support_level}",
            "variant_pattern_id": "corroboration_matched_pair_v2",
            "calibration_role": calibration_role,
            "matched_pair_id": pair_id,
            "trigger_magnitude": magnitude,
            "corroboration_present": target_support_level == "supported",
        }
    )
    _validate_generated_packet(packet)
    validate_detector_packet(packet)
    return shaped


def _validate_gate_manifest(manifest: dict[str, Any], *, gate_run_dir: Path) -> None:
    if manifest.get("mode") != "live":
        raise ValueError(f"calibration gate must be live: {gate_run_dir}")
    if manifest.get("run_purpose") != "approved_corroboration_calibration":
        raise ValueError(f"unexpected calibration gate run purpose: {gate_run_dir}")
    if manifest.get("trainable_labels_approved") is not True:
        raise ValueError(f"calibration gate is not approved for training: {gate_run_dir}")
    if manifest.get("teacher", {}).get("prompt_version") not in ACCEPTED_CALIBRATION_TEACHER_PROMPTS:
        raise ValueError(f"calibration gate used the wrong teacher prompt: {gate_run_dir}")
    if manifest.get("judge", {}).get("prompt_version") not in ACCEPTED_CALIBRATION_JUDGE_PROMPTS:
        raise ValueError(f"calibration gate used the wrong judge prompt: {gate_run_dir}")


def _complete_pair_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    pairs: dict[str, list[dict[str, Any]]] = {}
    label_mismatch_pair_ids: set[str] = set()
    for record in records:
        generation = record.get("metadata", {}).get("generation_metadata", {})
        pair_id = generation.get("matched_pair_id")
        role = generation.get("calibration_role")
        target = generation.get("target_support_level")
        if not pair_id or role not in {"train", "development"}:
            raise ValueError("promoted calibration record is missing matched-pair role metadata")
        if record.get("metadata", {}).get("support_level") != target:
            label_mismatch_pair_ids.add(pair_id)
        pairs.setdefault(pair_id, []).append(record)

    complete: list[dict[str, Any]] = []
    incomplete = 0
    for pair_id, pair_records in sorted(pairs.items()):
        if pair_id in label_mismatch_pair_ids:
            continue
        targets = {
            record["metadata"]["generation_metadata"]["target_support_level"]
            for record in pair_records
        }
        roles = {
            record["metadata"]["generation_metadata"]["calibration_role"]
            for record in pair_records
        }
        if len(pair_records) == 2 and targets == {"supported", "weakly_supported"} and len(roles) == 1:
            complete.extend(pair_records)
        else:
            incomplete += 1
    if not complete:
        raise ValueError("no complete corroboration matched pairs were promoted")
    return complete, {
        "incomplete_pairs_excluded": incomplete,
        "label_mismatch_pairs_excluded": len(label_mismatch_pair_ids),
    }


def _calibration_chat_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
        chat["metadata"]["calibration_matched_pair_id"] = generation["matched_pair_id"]
        chat["metadata"]["calibration_role"] = generation["calibration_role"]
        chat["metadata"]["corroboration_present"] = generation["corroboration_present"]
        rows.append(chat)
    return rows


def _normalize_base_chat_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(row)
    normalized["messages"][0]["content"] = detector_sft_system_prompt(SYSTEM_PROMPT_VERSION)
    for index, message in enumerate(normalized.get("messages", [])[1:], start=1):
        try:
            payload = json.loads(message["content"])
        except (json.JSONDecodeError, TypeError, KeyError):
            continue
        payload = _replace_risk_category(payload)
        if index == 1:
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
        "report_profiles": dict(
            sorted(Counter(record["metadata"]["report_profile"] for record in records).items())
        ),
        "matched_pairs": {
            "total": len(
                {
                    record["metadata"]["generation_metadata"]["matched_pair_id"]
                    for record in records
                }
            )
        },
    }


def _chat_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        "splits": _metadata_counter(rows, "split"),
        "support_levels": dict(
            _metadata_counter(rows, "support_level")
        ),
        "report_profiles": _metadata_counter(rows, "report_profile"),
        "risk_categories": _metadata_counter(rows, "risk_category"),
    }


def _metadata_counter(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    values = [
        str(row.get("metadata", {}).get(field) or "unknown")
        for row in rows
    ]
    return dict(sorted(Counter(values).items()))


def _group_id(derived_from_group_key: str) -> str:
    digest = hashlib.sha256(derived_from_group_key.encode("utf-8")).hexdigest()[:16].upper()
    return f"CAL_DG_{digest}"


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
    parser = argparse.ArgumentParser(description="Build corroboration-aware detector calibration artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    raw_parser = subparsers.add_parser("build-raw")
    raw_parser.add_argument("--training-clean-report", action="append", required=True)
    raw_parser.add_argument("--development-clean-report", action="append", required=True)
    raw_parser.add_argument("--reserved-real-manual-release-dir", action="append", default=[])
    raw_parser.add_argument("--output-dir", required=True)

    compose_parser = subparsers.add_parser("compose-sft")
    compose_parser.add_argument("--base-sft-jsonl", required=True)
    compose_parser.add_argument("--gate-run-dir", action="append", required=True)
    compose_parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()
    if args.command == "build-raw":
        result = run_detector_corroboration_raw_builder(
            training_clean_reports=args.training_clean_report,
            development_clean_reports=args.development_clean_report,
            reserved_real_manual_release_dirs=args.reserved_real_manual_release_dir,
            output_dir=args.output_dir,
        )
        payload = {
            "status": result.status,
            "raw_jsonl_path": str(result.raw_jsonl_path),
            "records_written": result.records_written,
            "errors": result.errors,
        }
    else:
        result = run_detector_corroboration_sft_composer(
            base_sft_jsonl=args.base_sft_jsonl,
            gate_run_dirs=args.gate_run_dir,
            output_dir=args.output_dir,
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
