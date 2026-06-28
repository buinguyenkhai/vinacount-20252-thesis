from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Callable

from vinacount.runtime_contract import (
    DEVELOPER_AUDIT_BUNDLE_SCHEMA_VERSION,
    STAGE_IDS,
    validate_developer_audit_bundle,
)
from vinacount.runtime_run_registry import FilesystemArtifactBodyStore, RuntimeRunRegistry


TimestampFactory = Callable[[], str]


_EXPECTED_ARTIFACT_COUNTS_BY_STAGE = {
    "extraction": {
        "report_memory_json": 2,
        "company_report_set_reference_json": 1,
    },
    "tool_analysis": {
        "tool_findings_json": 1,
        "candidate_risks_json": 1,
        "detector_packets_json": 1,
    },
    "detector_assessment": {
        "detector_assessments_json": 1,
    },
    "aggregation": {
        "aggregation_output_json": 1,
    },
    "report_generation": {
        "final_report_json": 1,
        "final_report_markdown": 1,
        "canonical_final_report_endpoint_json": 1,
    },
}
_COMPLETED_EXPECTED_ARTIFACT_COUNTS = {
    kind: count
    for stage_id in [
        "extraction",
        "tool_analysis",
        "detector_assessment",
        "aggregation",
        "report_generation",
    ]
    for kind, count in _EXPECTED_ARTIFACT_COUNTS_BY_STAGE[stage_id].items()
}


class DeveloperAuditBundleBuilder:
    def __init__(
        self,
        *,
        registry: RuntimeRunRegistry,
        artifact_store: FilesystemArtifactBodyStore,
        timestamp_factory: TimestampFactory,
    ) -> None:
        self._registry = registry
        self._artifact_store = artifact_store
        self._timestamp = timestamp_factory

    def build(self, run_id: str) -> dict[str, Any]:
        view = self._registry.get_run_view(run_id)
        if not _is_exportable(view):
            raise ValueError(
                "Developer Audit Bundle is only available for completed or recoverably failed Runtime Analysis Runs."
            )

        artifact_manifest_refs = [
            self._manifest_ref(ref)
            for ref in self._registry.get_artifact_refs(run_id)
        ]
        artifact_manifest = {
            "refs": artifact_manifest_refs,
            "missing_artifacts": _missing_artifacts(view, artifact_manifest_refs),
        }
        detector_mode = self._registry.get_run_audit_metadata(run_id).get("runtime_detector_mode", {})
        bundle = {
            "schema_version": DEVELOPER_AUDIT_BUNDLE_SCHEMA_VERSION,
            "run_id": run_id,
            "exported_at": self._timestamp(),
            "run_status": view["status"],
            "recoverable": view["recoverable"],
            "can_resume": view["can_resume"],
            "filing_intent": copy.deepcopy(view["filing_intent"]),
            "runtime_config": copy.deepcopy(view["runtime_config"]),
            "source_confirmation": copy.deepcopy(view["source_confirmation"]),
            "stage_records": copy.deepcopy(view["stages"]),
            "failed_stage": _failed_stage(view),
            "resume_eligibility": _resume_eligibility(view),
            "runtime_detector_mode": copy.deepcopy(detector_mode),
            "cache_trace": _cache_trace(
                view,
                artifact_manifest_refs,
                self._registry.get_run_audit_metadata(run_id),
            ),
            "artifact_manifest": artifact_manifest,
            "detector_audit": {
                "mode": copy.deepcopy(detector_mode),
                "packet_artifact_refs": _refs_by_kind(artifact_manifest_refs, "detector_packets_json"),
                "assessment_artifact_refs": _refs_by_kind(artifact_manifest_refs, "detector_assessments_json"),
            },
            "aggregation": {
                "artifact_refs": _refs_by_kind(artifact_manifest_refs, "aggregation_output_json"),
            },
            "final_report": {
                "metadata": copy.deepcopy(view["final_report"]),
                "artifact_refs": _refs_by_kinds(
                    artifact_manifest_refs,
                    [
                        "final_report_json",
                        "final_report_markdown",
                        "canonical_final_report_endpoint_json",
                        "report_synthesis_request_json",
                        "report_synthesis_response_json",
                    ],
                ),
            },
            "warnings": copy.deepcopy(view["warnings"]),
            "errors": [copy.deepcopy(view["error"])] if view["error"] is not None else [],
        }
        return validate_developer_audit_bundle(bundle)

    def _manifest_ref(self, ref: dict[str, Any]) -> dict[str, Any]:
        storage = _storage_ref(ref)
        body_status = "present"
        hash_status = "unavailable"
        observed_sha256 = None
        try:
            payload = self._artifact_store.read_bytes(ref)
        except KeyError:
            body_status = "missing_reference"
        except FileNotFoundError:
            body_status = "missing"
        else:
            observed_sha256 = hashlib.sha256(payload).hexdigest()
            hash_status = "verified" if observed_sha256 == ref.get("sha256") else "mismatch"

        return {
            "artifact_id": ref.get("artifact_id"),
            "kind": ref.get("kind"),
            "run_id": ref.get("run_id"),
            "sha256": ref.get("sha256"),
            "observed_sha256": observed_sha256,
            "size_bytes": ref.get("size_bytes"),
            "version": ref.get("version"),
            "schema_version": ref.get("schema_version"),
            "metadata": copy.deepcopy(ref.get("metadata") or {}),
            "storage": storage,
            "body_status": body_status,
            "hash_status": hash_status,
        }


def _is_exportable(view: dict[str, Any]) -> bool:
    if view["status"] == "completed":
        return True
    return view["status"] == "failed" and view["recoverable"]


def _storage_ref(ref: dict[str, Any]) -> dict[str, Any] | None:
    path = ref.get("path")
    if not path:
        return None
    return {
        "type": "filesystem",
        "path": str(Path(path)),
    }


def _failed_stage(view: dict[str, Any]) -> str | None:
    for stage in view["stages"]:
        if stage["status"] == "failed":
            return stage["stage_id"]
    return None


def _resume_eligibility(view: dict[str, Any]) -> dict[str, Any]:
    latest_completed_stage = None
    for stage in view["stages"]:
        if stage["status"] in {"completed", "skipped"}:
            latest_completed_stage = stage["stage_id"]
    next_stage = None
    if latest_completed_stage is not None:
        latest_index = STAGE_IDS.index(latest_completed_stage)
        for stage_id in STAGE_IDS[latest_index + 1 :]:
            stage = next(stage for stage in view["stages"] if stage["stage_id"] == stage_id)
            if stage["status"] not in {"completed", "skipped"}:
                next_stage = stage_id
                break
    return {
        "eligible": bool(view["can_resume"]),
        "latest_completed_stage": latest_completed_stage,
        "next_stage": next_stage,
        "reason": "recoverable_failed_stage_boundary" if view["can_resume"] else None,
    }


def _cache_trace(
    view: dict[str, Any],
    refs: list[dict[str, Any]],
    audit_metadata: dict[str, Any],
) -> dict[str, Any]:
    cache_stage = next(stage for stage in view["stages"] if stage["stage_id"] == "cache_lookup")
    counts = cache_stage.get("counts") or {}
    if counts.get("reusable_report_memory_artifacts", 0):
        outcome = "report_memory_reusable"
    elif counts:
        outcome = "miss_or_rebuild_required"
    else:
        outcome = "not_recorded"
    cache_warnings = [
        warning
        for warning in view["warnings"] + cache_stage.get("warnings", [])
        if warning.get("stage_id") == "cache_lookup"
    ]
    return {
        "outcome": outcome,
        "stage_record": copy.deepcopy(cache_stage),
        "warnings": copy.deepcopy(cache_warnings),
        "invalidity_traces": copy.deepcopy(
            audit_metadata.get("cache_lookup_invalidity_traces") or []
        ),
        "artifact_refs": _refs_by_kind(refs, "report_memory_json"),
    }


def _missing_artifacts(view: dict[str, Any], refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expected_counts = _expected_artifact_counts(view)
    actual_counts: dict[str, int] = {}
    for ref in refs:
        actual_counts[ref["kind"]] = actual_counts.get(ref["kind"], 0) + 1

    missing = []
    for kind, expected_count in expected_counts.items():
        actual_count = actual_counts.get(kind, 0)
        if actual_count < expected_count:
            missing.append(
                {
                    "kind": kind,
                    "expected_count": expected_count,
                    "actual_count": actual_count,
                    "reason": "missing_reference",
                }
            )
    for ref in refs:
        if ref["body_status"] in {"missing", "missing_reference"}:
            missing.append(
                {
                    "kind": ref["kind"],
                    "expected_count": 1,
                    "actual_count": 0,
                    "reason": ref["body_status"],
                }
            )
    return missing


def _expected_artifact_counts(view: dict[str, Any]) -> dict[str, int]:
    if view["status"] == "completed":
        return dict(_COMPLETED_EXPECTED_ARTIFACT_COUNTS)

    expected: dict[str, int] = {}
    for stage in view["stages"]:
        if stage["status"] not in {"completed", "skipped"}:
            continue
        for kind, count in _EXPECTED_ARTIFACT_COUNTS_BY_STAGE.get(stage["stage_id"], {}).items():
            expected[kind] = max(expected.get(kind, 0), count)
    return expected


def _refs_by_kind(refs: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [copy.deepcopy(ref) for ref in refs if ref["kind"] == kind]


def _refs_by_kinds(refs: list[dict[str, Any]], kinds: list[str]) -> list[dict[str, Any]]:
    selected = []
    for kind in kinds:
        selected.extend(_refs_by_kind(refs, kind))
    return selected
