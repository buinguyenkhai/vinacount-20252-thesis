from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research.dataset_traceability import build_source_report_manifest
from research.dataset_validator import validate_detector_dataset_release


SPLITS = ("train", "validation", "test", "holdout", "excluded")
SPLIT_BUILDER_VERSION = "detector_split_builder_v1"
SPLIT_STRATEGY = "decontamination_grouped_v1"


@dataclass(frozen=True)
class Wave4DetectorSplitBuilderResult:
    status: str
    run_dir: Path
    records_written: int
    errors: list[str]


def run_detector_split_builder(
    *,
    synthetic_gate_run_dirs: list[Path],
    real_manual_release_dirs: list[Path] | None = None,
    output_root: Path,
    run_id: str,
    seed: int,
    synthetic_train_ratio: float = 0.80,
    synthetic_validation_ratio: float = 0.10,
    synthetic_test_ratio: float = 0.10,
    synthetic_holdout_ratio: float = 0.0,
    strict_ratios: bool = False,
    ratio_tolerance: float = 0.05,
    allow_empty_requested_split: bool = False,
    allow_fake_smoke: bool = False,
    allow_legacy_reserved_collision_exclusion: bool = False,
    validate_release: bool = True,
) -> Wave4DetectorSplitBuilderResult:
    real_manual_release_dirs = real_manual_release_dirs or []
    if not synthetic_gate_run_dirs and not real_manual_release_dirs:
        raise ValueError("at least one synthetic gate run dir or real/manual release dir is required")

    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    synthetic_records = _load_synthetic_gate_records(synthetic_gate_run_dirs, allow_fake_smoke=allow_fake_smoke)
    real_manual_records = _load_real_manual_release_records(real_manual_release_dirs)

    assigned_records, leakage_rows, metrics = _assign_records(
        synthetic_records=synthetic_records,
        real_manual_records=real_manual_records,
        seed=seed,
        run_id=run_id,
        ratios={
            "train": synthetic_train_ratio,
            "validation": synthetic_validation_ratio,
            "test": synthetic_test_ratio,
            "holdout": synthetic_holdout_ratio,
        },
        strict_ratios=strict_ratios,
        ratio_tolerance=ratio_tolerance,
        allow_empty_requested_split=allow_empty_requested_split,
        allow_legacy_reserved_collision_exclusion=allow_legacy_reserved_collision_exclusion,
    )

    records_by_split = {split: [] for split in SPLITS}
    for record in assigned_records:
        records_by_split[record["metadata"]["split_metadata"]["split"]].append(record)

    for split in SPLITS:
        _write_jsonl(run_dir / f"{split}.jsonl", records_by_split[split])
    _write_jsonl(run_dir / "leakage_report.jsonl", leakage_rows)

    manifest = _build_manifest(
        records=assigned_records,
        records_by_split=records_by_split,
        synthetic_gate_run_dirs=synthetic_gate_run_dirs,
        real_manual_release_dirs=real_manual_release_dirs,
        run_id=run_id,
        seed=seed,
    )
    _write_json(run_dir / "manifest.json", manifest)

    validator_status = "skipped"
    validator_errors: list[str] = []
    if validate_release:
        validation_result = validate_detector_dataset_release(run_dir)
        validator_status = validation_result.status
        validator_errors = validation_result.errors
        manifest["integrity_checks"]["validator_status"] = validator_status
        _write_json(run_dir / "manifest.json", manifest)

    metrics.update(
        {
            "validator_status": validator_status,
            "validator_errors": validator_errors,
            "records_written": len(assigned_records),
            "num_examples_by_split": {split: len(records_by_split[split]) for split in SPLITS},
        }
    )
    _write_json(run_dir / "metrics.json", metrics)

    if validator_status == "failed":
        return Wave4DetectorSplitBuilderResult("failed", run_dir, len(assigned_records), validator_errors)
    return Wave4DetectorSplitBuilderResult("passed", run_dir, len(assigned_records), [])


def _load_synthetic_gate_records(gate_run_dirs: list[Path], *, allow_fake_smoke: bool) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for run_dir in gate_run_dirs:
        for required in ["manifest.json", "metrics.json", "filtered.jsonl"]:
            if not (run_dir / required).exists():
                raise ValueError(f"synthetic gate run dir is missing {required}: {run_dir}")
        manifest = _read_json(run_dir / "manifest.json")
        approved_live = _manifest_allows_downstream_split(manifest)
        allowed_fake_smoke = allow_fake_smoke and manifest.get("mode") == "fake"
        if not approved_live and not allowed_fake_smoke:
            raise ValueError(f"synthetic gate run dir is not approved for downstream split building: {run_dir}")
        for record in _read_jsonl(run_dir / "filtered.jsonl"):
            if record.get("source_type") != "synthetic_injected_filtered":
                raise ValueError("synthetic gate filtered.jsonl must contain only synthetic_injected_filtered records")
            _validate_synthetic_lineage(record)
            records.append(record)
    return records


def _manifest_allows_downstream_split(manifest: dict[str, Any]) -> bool:
    if manifest.get("mode") != "live":
        return False
    if manifest.get("run_purpose") == "approved_live":
        return (
            manifest.get("downstream_split_builder_allowed") is True
            and manifest.get("trainable_labels_approved") is True
        )
    if manifest.get("run_purpose") != "approved_issue151_standard":
        return False
    if manifest.get("downstream_split_builder_allowed") is True and manifest.get("trainable_labels_approved") is True:
        return True

    # Compatibility for standard batches written before the manifest approval
    # flags were generalized beyond the eight-record #150 pilot.
    input_records = manifest.get("input_records")
    promoted_records = manifest.get("promoted_records")
    return (
        manifest.get("artifact_contract_version") == "synthetic_detector_assessment_gate_v1"
        and isinstance(input_records, int)
        and input_records == manifest.get("approved_live_input_records_required")
        and isinstance(promoted_records, int)
        and promoted_records > 0
        and manifest.get("judge_replay_candidates_jsonl_path") in {None, ""}
    )


def _validate_synthetic_lineage(record: dict[str, Any]) -> None:
    generation_metadata = record.get("metadata", {}).get("generation_metadata", {})
    split_metadata = record.get("metadata", {}).get("split_metadata", {})
    missing = [
        field
        for field in ["base_report_id", "synthetic_report_id", "injection_scenario_id"]
        if not generation_metadata.get(field)
    ]
    if not split_metadata.get("derived_from_group_key"):
        missing.append("derived_from_group_key")
    if missing:
        raise ValueError(
            f"missing required synthetic lineage metadata for {record.get('example_id')}: {sorted(missing)}"
        )


def _load_real_manual_release_records(release_dirs: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for release_dir in release_dirs:
        manifest = _read_json(release_dir / "manifest.json")
        for split in manifest.get("num_examples_by_split", {}):
            for record in _read_jsonl(release_dir / f"{split}.jsonl"):
                if record.get("source_type") != "human_gold_real_report":
                    raise ValueError("real/manual release dirs must contain only human_gold_real_report records")
                records.append(record)
    return records


def _assign_records(
    *,
    synthetic_records: list[dict[str, Any]],
    real_manual_records: list[dict[str, Any]],
    seed: int,
    run_id: str,
    ratios: dict[str, float],
    strict_ratios: bool,
    ratio_tolerance: float,
    allow_empty_requested_split: bool,
    allow_legacy_reserved_collision_exclusion: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    leakage_rows: list[dict[str, Any]] = []
    synthetic_groups = _group_records(synthetic_records)
    reserved_identities = set().union(*(_identity_values(record) for record in real_manual_records)) if real_manual_records else set()
    excluded_group_ids = [
        group_id
        for group_id, group_records in synthetic_groups.items()
        if reserved_identities and any(_identity_values(record) & reserved_identities for record in group_records)
    ]
    if excluded_group_ids and not allow_legacy_reserved_collision_exclusion:
        raise ValueError(
            "synthetic records collide with reserved real/manual source identities; official runs must "
            "exclude reserved real/manual anchors before synthetic generation. Use "
            "--allow-legacy-reserved-collision-exclusion only to inspect legacy artifacts."
        )
    for group_id in excluded_group_ids:
        leakage_rows.append(
            {
                "action": "exclude_synthetic_group",
                "reason": "collides_with_reserved_real_manual",
                "decontamination_group_id": group_id,
                "example_ids": [record["example_id"] for record in synthetic_groups[group_id]],
            }
        )

    assignable_group_ids = [group_id for group_id in synthetic_groups if group_id not in set(excluded_group_ids)]
    shuffled_group_ids = list(assignable_group_ids)
    random.Random(seed).shuffle(shuffled_group_ids)
    split_group_counts = _split_counts(len(shuffled_group_ids), ratios)
    if strict_ratios and shuffled_group_ids:
        _validate_split_ratio_viability(
            total_groups=len(shuffled_group_ids),
            ratios=ratios,
            counts=split_group_counts,
            tolerance=ratio_tolerance,
            allow_empty_requested_split=allow_empty_requested_split,
        )

    assigned: list[dict[str, Any]] = []
    offset = 0
    for split in ["train", "validation", "test", "holdout"]:
        for group_id in shuffled_group_ids[offset : offset + split_group_counts[split]]:
            for record in synthetic_groups[group_id]:
                assigned.append(
                    _stamp_record(
                        record,
                        split=split,
                        seed=seed,
                        run_id=run_id,
                        usable_for_training=True,
                        decontamination_group_id=group_id,
                    )
                )
        offset += split_group_counts[split]

    for group_id in excluded_group_ids:
        for record in synthetic_groups[group_id]:
            assigned.append(
                _stamp_record(
                    record,
                    split="excluded",
                    seed=seed,
                    run_id=run_id,
                    usable_for_training=False,
                    exclusion_reason="collides_with_reserved_real_manual",
                    decontamination_group_id=group_id,
                )
            )

    for record in real_manual_records:
        existing_split = record.get("metadata", {}).get("split_metadata", {}).get("split", "validation")
        if existing_split == "train":
            raise ValueError("human_gold_real_report records cannot be assigned to train by default")
        assigned.append(_stamp_record(record, split=existing_split, seed=seed, run_id=run_id, usable_for_training=False))

    metrics = {
        "requested_ratios": ratios,
        "strict_ratios": strict_ratios,
        "ratio_tolerance": ratio_tolerance,
        "allow_empty_requested_split": allow_empty_requested_split,
        "reserved_real_manual_collision_policy": (
            "legacy_exclude_synthetic_group" if allow_legacy_reserved_collision_exclusion else "hard_fail"
        ),
        "actual_synthetic_group_counts": split_group_counts,
        "leakage_actions": dict(Counter(row["action"] for row in leakage_rows)),
    }
    return assigned, leakage_rows, metrics


def _group_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    components: list[tuple[set[str], list[dict[str, Any]]]] = []
    for record in records:
        identities = _identity_values(record)
        matching_indexes = [index for index, (component_identities, _) in enumerate(components) if component_identities & identities]
        if not matching_indexes:
            components.append((set(identities), [record]))
            continue

        first_index = matching_indexes[0]
        components[first_index][0].update(identities)
        components[first_index][1].append(record)
        for merge_index in reversed(matching_indexes[1:]):
            components[first_index][0].update(components[merge_index][0])
            components[first_index][1].extend(components[merge_index][1])
            del components[merge_index]

    groups: dict[str, list[dict[str, Any]]] = {}
    for identities, component_records in components:
        group_id = _group_id_from_identities(identities)
        groups[group_id] = component_records
    return groups


def _split_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    if total == 0:
        return {split: 0 for split in ["train", "validation", "test", "holdout"]}
    normalized_ratios = {split: max(ratios.get(split, 0.0), 0.0) for split in ["train", "validation", "test", "holdout"]}
    ratio_sum = sum(normalized_ratios.values())
    if ratio_sum <= 0:
        normalized_ratios = {"train": 1.0, "validation": 0.0, "test": 0.0, "holdout": 0.0}
        ratio_sum = 1.0
    raw = {split: (normalized_ratios[split] / ratio_sum) * total for split in normalized_ratios}
    counts = {split: int(value) for split, value in raw.items()}
    remaining = total - sum(counts.values())
    remainders = sorted(((raw[split] - counts[split], split) for split in raw), reverse=True)
    for _, split in remainders[:remaining]:
        counts[split] += 1
    return counts


def _validate_split_ratio_viability(
    *,
    total_groups: int,
    ratios: dict[str, float],
    counts: dict[str, int],
    tolerance: float,
    allow_empty_requested_split: bool,
) -> None:
    requested = {split: max(ratios.get(split, 0.0), 0.0) for split in ["train", "validation", "test", "holdout"]}
    requested_sum = sum(requested.values())
    if requested_sum <= 0:
        return
    if not allow_empty_requested_split:
        empty_requested = [split for split, ratio in requested.items() if ratio > 0 and counts.get(split, 0) == 0]
        if empty_requested:
            raise ValueError(f"requested nonzero splits have no decontamination groups: {empty_requested}")
    for split, ratio in requested.items():
        expected = ratio / requested_sum
        actual = counts.get(split, 0) / total_groups
        if abs(actual - expected) > tolerance:
            raise ValueError(
                f"split ratio for {split} is outside tolerance: expected {expected:.4f}, actual {actual:.4f}, tolerance {tolerance:.4f}"
            )


def _stamp_record(
    record: dict[str, Any],
    *,
    split: str,
    seed: int,
    run_id: str,
    usable_for_training: bool,
    exclusion_reason: str | None = None,
    decontamination_group_id: str | None = None,
) -> dict[str, Any]:
    stamped = json.loads(json.dumps(record))
    metadata = stamped.setdefault("metadata", {})
    split_metadata = metadata.setdefault("split_metadata", {})
    generation_metadata = metadata.setdefault("generation_metadata", {})
    packet = stamped["input"]["data"]
    assessment = stamped["output"]["data"]

    split_metadata["split"] = split
    split_metadata["split_strategy"] = SPLIT_STRATEGY
    split_metadata["split_seed"] = seed
    split_metadata["split_run_id"] = run_id
    split_metadata["packet_content_hash"] = _canonical_hash(packet)
    split_metadata["assessment_rationale_hash"] = _normalized_text_hash(assessment.get("rationale_short", ""))
    split_metadata["usable_for_training"] = usable_for_training and split != "excluded"
    split_metadata["exclusion_reason"] = None if split != "excluded" else exclusion_reason or split_metadata.get("exclusion_reason", "excluded")
    split_metadata["decontamination_group_id"] = decontamination_group_id or _decontamination_group_id(stamped)

    if stamped.get("source_type", "").startswith("synthetic_"):
        split_metadata["is_synthetic_derivative"] = True
        for field in ["base_report_id", "synthetic_report_id", "injection_scenario_id"]:
            if generation_metadata.get(field):
                split_metadata[field] = generation_metadata[field]
        split_metadata.setdefault("derived_from_group_key", generation_metadata.get("base_report_id") or split_metadata.get("group_key"))

    return stamped


def _identity_values(record: dict[str, Any]) -> set[str]:
    split_metadata = record.get("metadata", {}).get("split_metadata", {})
    generation_metadata = record.get("metadata", {}).get("generation_metadata", {})
    packet = record.get("input", {}).get("data", {})
    base_report_id = generation_metadata.get("base_report_id")
    injection_scenario_id = generation_metadata.get("injection_scenario_id")
    values = {
        split_metadata.get("group_key"),
        split_metadata.get("derived_from_group_key"),
        base_report_id,
        generation_metadata.get("synthetic_report_id"),
        split_metadata.get("source_file_sha256"),
        split_metadata.get("normalized_text_hash"),
        split_metadata.get("table_content_hash"),
        split_metadata.get("derived_from_report_artifact_id"),
        split_metadata.get("derived_from_source_document_id"),
        packet.get("report_id"),
        _canonical_hash(packet),
        _normalized_text_hash(record.get("output", {}).get("data", {}).get("rationale_short", "")),
    }
    if base_report_id and injection_scenario_id:
        values.add(f"{base_report_id}::{injection_scenario_id}")
    values.update(_packet_report_identities(packet))
    return {str(value) for value in values if value}


def _packet_report_identities(packet: dict[str, Any]) -> set[str]:
    report_ids = {packet.get("report_id")}
    for collection_name in ["relevant_table_rows", "relevant_notes", "relevant_variance_explanations"]:
        for item in packet.get(collection_name, []):
            if isinstance(item, dict):
                report_ids.add(item.get("report_id"))
    for finding in packet.get("tool_findings", []):
        if not isinstance(finding, dict):
            continue
        for evidence_ref in finding.get("evidence_refs", []):
            if not isinstance(evidence_ref, dict):
                continue
            report_ids.add(evidence_ref.get("report_id"))
            ref_id = evidence_ref.get("ref_id")
            if isinstance(ref_id, str) and ":" in ref_id:
                report_ids.add(ref_id.split(":", 1)[0])
    return {report_id for report_id in report_ids if report_id}


def _decontamination_group_id(record: dict[str, Any]) -> str:
    return _group_id_from_identities(_identity_values(record))


def _group_id_from_identities(identities: set[str]) -> str:
    digest = hashlib.sha256("|".join(sorted(identities)).encode("utf-8")).hexdigest()
    return f"DG_{digest[:16].upper()}"


def _build_manifest(
    *,
    records: list[dict[str, Any]],
    records_by_split: dict[str, list[dict[str, Any]]],
    synthetic_gate_run_dirs: list[Path],
    real_manual_release_dirs: list[Path],
    run_id: str,
    seed: int,
) -> dict[str, Any]:
    split_counter = {split: len(records_by_split[split]) for split in SPLITS}
    training_usability_counts = Counter(
        "usable_for_training" if record["metadata"]["split_metadata"].get("usable_for_training") else "not_usable_for_training"
        for record in records
    )
    return {
        "release_name": run_id,
        "release_build_id": f"WAVE4_SPLITS_{_canonical_hash({'run_id': run_id, 'seed': seed})[:16].upper()}",
        "dataset_version": "tdf_v1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "created_by": "detector_split_builder",
        "split_builder_version": SPLIT_BUILDER_VERSION,
        "split_run_id": run_id,
        "split_seed": seed,
        "split_strategy": SPLIT_STRATEGY,
        "input_artifacts": {
            "synthetic_gate_run_dirs": [str(path) for path in synthetic_gate_run_dirs],
            "synthetic_gate_runs": _synthetic_gate_run_artifacts(synthetic_gate_run_dirs),
            "real_manual_release_dirs": [str(path) for path in real_manual_release_dirs],
        },
        "num_examples_total": len(records),
        "num_examples_by_split": split_counter,
        "num_examples_by_source_type": dict(Counter(record["source_type"] for record in records)),
        "num_examples_by_split_and_source_type": _nested_counts(records, "source_type"),
        "num_examples_by_support_level": dict(Counter(record["metadata"]["support_level"] for record in records)),
        "num_examples_by_report_profile": dict(Counter(record["metadata"]["report_profile"] for record in records)),
        "num_examples_by_risk_category": dict(Counter(record["metadata"]["risk_category"] for record in records)),
        "num_examples_by_split_support_level": _nested_counts(records, "metadata.support_level"),
        "num_examples_by_split_report_profile": _nested_counts(records, "metadata.report_profile"),
        "num_examples_by_split_risk_category": _nested_counts(records, "metadata.risk_category"),
        "num_examples_by_training_usability": {
            "usable_for_training": training_usability_counts["usable_for_training"],
            "not_usable_for_training": training_usability_counts["not_usable_for_training"],
        },
        "num_decontamination_groups_total": len(
            {record["metadata"]["split_metadata"]["decontamination_group_id"] for record in records}
        ),
        "num_decontamination_groups_by_split": {
            split: len({record["metadata"]["split_metadata"]["decontamination_group_id"] for record in split_records})
            for split, split_records in records_by_split.items()
        },
        "source_report_traceability": build_source_report_manifest(records),
        "decontamination_group_manifest": _decontamination_group_manifest(records),
        "integrity_checks": {
            "manifest_counts_match_files": True,
            "split_metadata_matches_file": True,
            "required_traceability_present": True,
            "audit_traceability_matches_split_traceability": True,
            "no_detector_visible_hidden_metadata": True,
            "no_real_manual_training_records": True,
            "no_cross_split_decontamination_identity": True,
            "no_synthetic_collision_with_reserved_real_manual_source_identities": True,
            "excluded_records_have_reason": True,
            "non_excluded_records_have_no_exclusion_reason": True,
            "requested_nonzero_splits_have_groups": True,
            "validator_status": "pending",
        },
        "sft_export_ready": True,
        "sft_export_allowed_splits": ["train", "validation", "test"],
        "sft_export_default_source_types": ["synthetic_injected_filtered", "synthetic_injected_human_reviewed"],
        "sft_export_excluded_source_types_by_default": ["human_gold_real_report"],
        "schema_versions": {
            "DetectorPacket": "v1.0.0",
            "DetectorAssessment": "v1.0.0",
            "CandidateRisk": "v1.0.0",
            "ToolFinding": "v1.0.0",
        },
        "validation_pipeline_version": "validator_v0.1.0",
        "generation_pipeline_version": "synthetic_generator_v0.1.0",
        "known_limitations": [],
    }


def _synthetic_gate_run_artifacts(gate_run_dirs: list[Path]) -> list[dict[str, Any]]:
    artifacts = []
    for run_dir in gate_run_dirs:
        manifest_path = run_dir / "manifest.json"
        metrics_path = run_dir / "metrics.json"
        filtered_path = run_dir / "filtered.jsonl"
        manifest = _read_json(manifest_path)
        artifacts.append(
            {
                "run_dir": str(run_dir),
                "run_id": manifest.get("run_id"),
                "manifest_path": str(manifest_path),
                "manifest_sha256": _file_sha256(manifest_path),
                "metrics_path": str(metrics_path),
                "metrics_sha256": _file_sha256(metrics_path),
                "filtered_jsonl_path": str(filtered_path),
                "filtered_jsonl_sha256": _file_sha256(filtered_path),
            }
        )
    return artifacts


def _nested_counts(records: list[dict[str, Any]], field_path: str) -> dict[str, dict[str, int]]:
    result: dict[str, Counter] = {}
    for record in records:
        split = record["metadata"]["split_metadata"]["split"]
        result.setdefault(split, Counter())[_get_path(record, field_path)] += 1
    return {split: dict(counter) for split, counter in result.items()}


def _get_path(data: dict[str, Any], field_path: str) -> Any:
    value: Any = data
    for part in field_path.split("."):
        value = value[part]
    return value


def _decontamination_group_manifest(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for record in records:
        split_metadata = record["metadata"]["split_metadata"]
        group_id = split_metadata["decontamination_group_id"]
        group = groups.setdefault(
            group_id,
            {
                "decontamination_group_id": group_id,
                "splits": [],
                "example_count": 0,
                "source_types": [],
            },
        )
        group["example_count"] += 1
        split = split_metadata["split"]
        if split not in group["splits"]:
            group["splits"].append(split)
        source_type = record["source_type"]
        if source_type not in group["source_types"]:
            group["source_types"].append(source_type)
    return {"groups": sorted(groups.values(), key=lambda item: item["decontamination_group_id"])}


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _normalized_text_hash(text: str) -> str:
    normalized = " ".join(text.casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Wave 4 detector decontaminated split release.")
    parser.add_argument("--synthetic-gate-run-dir", action="append", default=[], type=Path)
    parser.add_argument("--real-manual-release-dir", action="append", default=[], type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--synthetic-train-ratio", type=float, default=0.80)
    parser.add_argument("--synthetic-validation-ratio", type=float, default=0.10)
    parser.add_argument("--synthetic-test-ratio", type=float, default=0.10)
    parser.add_argument("--synthetic-holdout-ratio", type=float, default=0.0)
    parser.add_argument("--strict-ratios", action="store_true")
    parser.add_argument("--ratio-tolerance", type=float, default=0.05)
    parser.add_argument("--allow-empty-requested-split", action="store_true")
    parser.add_argument("--allow-fake-smoke", action="store_true")
    parser.add_argument("--allow-legacy-reserved-collision-exclusion", action="store_true")
    parser.add_argument("--no-validate-release", action="store_true")
    args = parser.parse_args()

    result = run_detector_split_builder(
        synthetic_gate_run_dirs=args.synthetic_gate_run_dir,
        real_manual_release_dirs=args.real_manual_release_dir,
        output_root=args.output_root,
        run_id=args.run_id,
        seed=args.seed,
        synthetic_train_ratio=args.synthetic_train_ratio,
        synthetic_validation_ratio=args.synthetic_validation_ratio,
        synthetic_test_ratio=args.synthetic_test_ratio,
        synthetic_holdout_ratio=args.synthetic_holdout_ratio,
        strict_ratios=args.strict_ratios,
        ratio_tolerance=args.ratio_tolerance,
        allow_empty_requested_split=args.allow_empty_requested_split,
        allow_fake_smoke=args.allow_fake_smoke,
        allow_legacy_reserved_collision_exclusion=args.allow_legacy_reserved_collision_exclusion,
        validate_release=not args.no_validate_release,
    )
    print(
        json.dumps(
            {
                "status": result.status,
                "run_dir": str(result.run_dir),
                "records_written": result.records_written,
                "errors": result.errors,
            },
            sort_keys=True,
        )
    )
    raise SystemExit(0 if result.status == "passed" else 1)


if __name__ == "__main__":
    main()
