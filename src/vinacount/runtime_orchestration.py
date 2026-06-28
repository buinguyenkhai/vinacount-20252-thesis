from __future__ import annotations

import copy
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

from vinacount.analysis_chain_tracer import (
    aggregate_detector_assessments,
    build_candidate_risks,
    build_detector_packets,
    run_deterministic_tool_checks,
)
from vinacount.detector_contract import (
    CandidateRisk,
    DetectorAdapter,
    DetectorAssessment,
    DetectorPacket,
    ToolFinding,
    validate_detector_assessment,
)
from vinacount.final_report import build_final_report, render_final_report_markdown
from vinacount.filing_package import select_canonical_source, validate_filing_package
from vinacount.report_model import CompanyReportSet, ReportMemory, validate_company_report_set, validate_report_memory
from vinacount.report_artifact_cache import (
    ReportMemoryArtifactCache,
    ReportMemoryCacheCompatibility,
    ReportMemoryCacheIdentity,
)
from vinacount.report_synthesis import (
    ReportSynthesisAdapter,
    build_report_synthesis_request,
    merge_validated_report_narrative,
    report_narrative_draft_json_schema,
    report_synthesis_output_hash,
)
from vinacount.runtime_contract import (
    FINAL_REPORT_ENDPOINT_SCHEMA_VERSION,
    HITL_BOUNDARY,
    stable_json_dumps,
    validate_final_report_endpoint,
)
from vinacount.runtime_detector_modes import (
    DEFAULT_SFT_CONTRACT_GUARD_VERSION,
    RuntimeDetectorGuardError,
    RuntimeDetectorInvalidJsonError,
    RuntimeDetectorModeRegistry,
    RuntimeDetectorProviderResponseError,
    RuntimeDetectorTimeoutError,
    RuntimeDetectorTransportError,
    injected_runtime_detector_mode_registry,
)
from vinacount.runtime_run_registry import FilesystemArtifactBodyStore, RuntimeRunRegistry


TimestampFactory = Callable[[], str]


class CacheLookupAdapter(Protocol):
    def lookup(self, run_view: dict[str, Any]) -> dict[str, Any]:
        ...


class DemoCacheLookupAdapter:
    def lookup(self, run_view: dict[str, Any]) -> dict[str, Any]:
        return {
            "outcome": "miss",
            "reusable_report_memory_refs": [],
            "warnings": [],
        }


class FilingCacheLookupAdapter:
    def __init__(
        self,
        *,
        report_memory_cache: ReportMemoryArtifactCache,
        artifact_store: FilesystemArtifactBodyStore,
        compatibility: ReportMemoryCacheCompatibility | None = None,
        raw_ocr_cache: Any | None = None,
        raw_ocr_identity_factory: Callable[[dict[str, Any]], Any] | None = None,
        source_refresh: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        latest_quarters: list[tuple[int, int]] | None = None,
    ) -> None:
        self._report_memory_cache = report_memory_cache
        self._artifact_store = artifact_store
        self._compatibility = compatibility or ReportMemoryCacheCompatibility()
        self._raw_ocr_cache = raw_ocr_cache
        self._raw_ocr_identity_factory = raw_ocr_identity_factory
        self._source_refresh = source_refresh
        self._latest_quarters = set(latest_quarters or _default_latest_two_quarters())

    def lookup(self, run_view: dict[str, Any]) -> dict[str, Any]:
        source_confirmation = run_view.get("source_confirmation", {})
        if source_confirmation.get("status") != "confirmed":
            return {"outcome": "miss", "reusable_report_memory_refs": [], "warnings": []}
        slots = {slot["role"]: slot for slot in source_confirmation.get("slots", [])}
        if set(slots) != {"target", "prior_year_same_quarter"}:
            return self._miss("filing_cache_lookup_incomplete_source_pair")

        hits = {}
        stale_roles = []
        raw_only_roles = []
        for role in ("target", "prior_year_same_quarter"):
            candidate = slots[role].get("candidate")
            if not isinstance(candidate, dict):
                return self._miss("filing_cache_lookup_incomplete_source_pair")
            source_safety = self._source_safety(candidate, role=role, run_view=run_view)
            if source_safety["outcome"] != "safe":
                return source_safety["result"]
            identity = self._identity(candidate, role=role)
            hit = self._report_memory_cache.lookup(identity)
            if hit is not None:
                hits[role] = hit
                continue
            source_record = self._report_memory_cache.lookup_any_for_source(identity)
            if source_record is not None and (
                source_record.validity_status != "valid"
                or source_record.quality_status != "validated"
            ):
                return self._blocked(
                    "invalid_blocked",
                    "filing_cache_lookup_invalid_blocked",
                    [role],
                    invalidity_traces=[_report_memory_invalidity_trace(role, source_record)],
                )
            if source_record is not None:
                stale_roles.append(role)
                continue
            if self._has_raw_ocr(candidate):
                raw_only_roles.append(role)

        if set(hits) == {"target", "prior_year_same_quarter"}:
            refs = [
                self._report_memory_ref(run_view["run_id"], "target", hits["target"]),
                self._report_memory_ref(
                    run_view["run_id"],
                    "prior_year_same_quarter",
                    hits["prior_year_same_quarter"],
                ),
            ]
            return {
                "outcome": "report_memory_reusable",
                "reusable_report_memory_refs": refs,
                "warnings": [
                    {
                        "code": "filing_cache_lookup_report_memory_reusable",
                        "severity": "info",
                        "message": "ReportMemory cache contains a compatible complete filing pair.",
                        "stage_id": "cache_lookup",
                        "artifact_refs": copy.deepcopy(refs),
                    }
                ],
            }
        if hits:
            return self._miss("filing_cache_lookup_incomplete_report_memory_pair")
        if stale_roles:
            return self._rebuild("stale_rebuild_required", stale_roles)
        if raw_only_roles:
            return self._rebuild("source_only", raw_only_roles)
        return self._miss("filing_cache_lookup_miss")

    def _identity(self, candidate: dict[str, Any], *, role: str) -> ReportMemoryCacheIdentity:
        report_profile = str(
            candidate.get("report_profile")
            or candidate.get("audit_references", {}).get("report_profile")
            or "standard_corporate"
        )
        return ReportMemoryCacheIdentity.from_candidate(
            candidate=candidate,
            report_role=role,
            report_profile=report_profile,
            compatibility=self._compatibility,
        )

    def _has_raw_ocr(self, candidate: dict[str, Any]) -> bool:
        if self._raw_ocr_cache is None or self._raw_ocr_identity_factory is None:
            return False
        try:
            identity = self._raw_ocr_identity_factory(candidate)
        except Exception:
            return False
        return self._raw_ocr_cache.lookup(identity) is not None

    def _source_safety(
        self,
        candidate: dict[str, Any],
        *,
        role: str,
        run_view: dict[str, Any],
    ) -> dict[str, Any]:
        if not self._requires_source_refresh(candidate):
            return {"outcome": "safe"}
        if self._source_refresh is None:
            return {
                "outcome": "unsafe",
                "result": self._blocked(
                    "invalid_blocked",
                    "filing_cache_lookup_recent_source_refresh_unavailable",
                    [role],
                ),
            }
        try:
            refreshed = self._source_refresh(candidate, role=role, run_view=run_view)  # type: ignore[misc]
        except TypeError:
            refreshed = self._source_refresh(candidate)  # type: ignore[misc]
        if not isinstance(refreshed, dict):
            return {
                "outcome": "unsafe",
                "result": self._blocked(
                    "invalid_blocked",
                    "filing_cache_lookup_recent_source_refresh_invalid",
                    [role],
                ),
            }
        if refreshed.get("status") in {"hitl_needed", "ambiguous"}:
            return {
                "outcome": "unsafe",
                "result": self._blocked(
                    "ambiguous_source_requires_resolution",
                    "filing_cache_lookup_ambiguous_source_requires_resolution",
                    [role],
                ),
            }
        if refreshed.get("status") != "selected":
            return {
                "outcome": "unsafe",
                "result": self._blocked(
                    "invalid_blocked",
                    "filing_cache_lookup_refreshed_source_metadata_missing",
                    [role],
                ),
            }
        candidate_fingerprint = (candidate.get("audit_references") or {}).get(
            "source_document_fingerprint_sha256"
        )
        refreshed_fingerprint = refreshed.get("source_document_fingerprint_sha256")
        if not candidate_fingerprint or not refreshed_fingerprint:
            return {
                "outcome": "unsafe",
                "result": self._blocked(
                    "invalid_blocked",
                    "filing_cache_lookup_refreshed_source_metadata_missing",
                    [role],
                ),
            }
        if refreshed_fingerprint != candidate_fingerprint:
            return {
                "outcome": "unsafe",
                "result": self._blocked(
                    "stale_rebuild_required",
                    "filing_cache_lookup_refreshed_source_fingerprint_changed",
                    [role],
                    require_source_reconfirmation=True,
                ),
            }
        if refreshed.get("canonical_source_document_id") and refreshed.get(
            "canonical_source_document_id"
        ) != candidate.get("source_document_id"):
            return {
                "outcome": "unsafe",
                "result": self._blocked(
                    "stale_rebuild_required",
                    "filing_cache_lookup_refreshed_source_identity_changed",
                    [role],
                    require_source_reconfirmation=True,
                ),
            }
        if refreshed.get("filing_status") and refreshed.get("filing_status") != candidate.get(
            "filing_status"
        ):
            return {
                "outcome": "unsafe",
                "result": self._blocked(
                    "stale_rebuild_required",
                    "filing_cache_lookup_refreshed_source_status_changed",
                    [role],
                    require_source_reconfirmation=True,
                ),
            }
        superseded_ids = set(refreshed.get("superseded_source_document_ids") or [])
        if candidate.get("source_document_id") in superseded_ids:
            return {
                "outcome": "unsafe",
                "result": self._blocked(
                    "stale_rebuild_required",
                    "filing_cache_lookup_refreshed_source_superseded",
                    [role],
                    require_source_reconfirmation=True,
                ),
            }
        if _refreshed_source_has_correction_marker(refreshed):
            return {
                "outcome": "unsafe",
                "result": self._blocked(
                    "stale_rebuild_required",
                    "filing_cache_lookup_refreshed_source_corrected",
                    [role],
                    require_source_reconfirmation=True,
                ),
            }
        return {"outcome": "safe"}

    def _requires_source_refresh(self, candidate: dict[str, Any]) -> bool:
        try:
            filing_period = (int(candidate["fiscal_year"]), int(candidate["quarter"]))
        except (KeyError, TypeError, ValueError):
            return True
        return filing_period in self._latest_quarters

    def _report_memory_ref(self, run_id: str, role: str, hit: Any) -> dict[str, Any]:
        record = hit.record
        return self._artifact_store.reference_existing(
            run_id=run_id,
            artifact_id=f"artifact_cached_report_memory_{role}",
            kind="report_memory_json",
            path=Path(record.body_path),
            version=record.identity.builder_version,
            schema_version=record.identity.schema_version,
            metadata={
                "report_role": role,
                "cache_outcome": "report_memory_reusable",
                "cache_validity_status": record.validity_status,
                "cache_quality_status": record.quality_status,
                "source_document_fingerprint_sha256": record.identity.canonical_source_sha256,
                "report_profile": record.identity.report_profile,
                "report_basis": record.identity.report_basis,
                "schema_version": record.identity.schema_version,
                "builder_version": record.identity.builder_version,
                "mapper_version": record.identity.mapper_version,
                "normalization_version": record.identity.normalization_version,
                "extraction_version": record.identity.extraction_version,
                "quality_version": record.identity.quality_version,
            },
        )

    def _miss(self, code: str) -> dict[str, Any]:
        return {
            "outcome": "miss",
            "reusable_report_memory_refs": [],
            "warnings": [
                {
                    "code": code,
                    "severity": "info",
                    "message": "ReportMemory cache did not contain a complete compatible filing pair.",
                    "stage_id": "cache_lookup",
                    "artifact_refs": [],
                }
            ],
        }

    def _rebuild(self, outcome: str, roles: list[str]) -> dict[str, Any]:
        return {
            "outcome": outcome,
            "reusable_report_memory_refs": [],
            "warnings": [
                {
                    "code": f"filing_cache_lookup_{outcome}",
                    "severity": "info",
                    "message": "ReportMemory must be rebuilt from source-bound cached extraction artifacts.",
                    "stage_id": "cache_lookup",
                    "artifact_refs": [],
                }
            ],
        }

    def _blocked(
        self,
        outcome: str,
        code: str,
        roles: list[str],
        *,
        require_source_reconfirmation: bool = False,
        invalidity_traces: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        result = {
            "outcome": outcome,
            "reusable_report_memory_refs": [],
            "warnings": [
                {
                    "code": code,
                    "severity": "warning",
                    "message": "ReportMemory cache reuse is blocked until source safety is established.",
                    "stage_id": "cache_lookup",
                    "artifact_refs": [],
                    **({"source_slot_role": roles[0]} if len(roles) == 1 else {}),
                }
            ],
        }
        if require_source_reconfirmation:
            result["source_reconfirmation_required_roles"] = list(roles)
        if invalidity_traces:
            result["invalidity_traces"] = copy.deepcopy(invalidity_traces)
        return result


def _default_latest_two_quarters() -> list[tuple[int, int]]:
    now = datetime.now(timezone.utc)
    current_quarter = ((now.month - 1) // 3) + 1
    previous_year = now.year
    previous_quarter = current_quarter - 1
    if previous_quarter == 0:
        previous_year -= 1
        previous_quarter = 4
    return [(now.year, current_quarter), (previous_year, previous_quarter)]


def _refreshed_source_has_correction_marker(refreshed: dict[str, Any]) -> bool:
    correction_fields = (
        "corrected_source_document_id",
        "operative_corrected_source_document_id",
        "correction_status",
        "corrected_value_resolution_status",
    )
    if any(refreshed.get(field) for field in correction_fields):
        return True
    return bool(refreshed.get("is_corrected") or refreshed.get("has_corrected_values"))


def _report_memory_invalidity_trace(role: str, record: Any) -> dict[str, Any]:
    return {
        "source_slot_role": role,
        "artifact_id": record.artifact_id,
        "validity_status": record.validity_status,
        "quality_status": record.quality_status,
        "invalidity_reason": record.invalidity_reason or "unknown",
    }


class RuntimeStageExecutionError(RuntimeError):
    def __init__(
        self,
        *,
        stage_id: str,
        code: str,
        message: str,
        detail: str | None = None,
        recoverable: bool = True,
        can_resume: bool = True,
        artifact_refs: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.stage_id = stage_id
        self.code = code
        self.message = message
        self.detail = detail
        self.recoverable = recoverable
        self.can_resume = can_resume
        self.artifact_refs = copy.deepcopy(artifact_refs or [])

    def to_error_view(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "detail": self.detail,
            "stage_id": self.stage_id,
            "recoverable": self.recoverable,
            "can_resume": self.can_resume,
            "artifact_refs": copy.deepcopy(self.artifact_refs),
        }


class RuntimeStageSpine:
    def __init__(
        self,
        *,
        registry: RuntimeRunRegistry,
        artifact_store: FilesystemArtifactBodyStore,
        timestamp_factory: TimestampFactory,
        cache_lookup: CacheLookupAdapter | None = None,
        detector_adapter: DetectorAdapter | None = None,
        detector_mode: str | None = None,
        detector_mode_registry: RuntimeDetectorModeRegistry | None = None,
        report_synthesis_adapter: ReportSynthesisAdapter | None = None,
        report_generation_mode: str = "model_synthesis",
        extraction_adapter: Any | None = None,
        report_memory_cache: ReportMemoryArtifactCache | None = None,
        report_memory_cache_compatibility: ReportMemoryCacheCompatibility | None = None,
    ) -> None:
        self._registry = registry
        self._artifact_store = artifact_store
        self._timestamp = timestamp_factory
        self._cache_lookup = cache_lookup or DemoCacheLookupAdapter()
        if detector_adapter is not None:
            detector_mode_registry = injected_runtime_detector_mode_registry(
                detector_adapter,
                mode=detector_mode,
            )
        self._detector_mode_selection = (detector_mode_registry or RuntimeDetectorModeRegistry()).resolve(
            detector_mode
        )
        if self._detector_mode_selection.adapter is None:
            raise ValueError("Runtime detector mode did not resolve an adapter.")
        self._detector_adapter = self._detector_mode_selection.adapter
        self._report_synthesis_adapter = report_synthesis_adapter
        if report_generation_mode not in {"model_synthesis", "deterministic_template"}:
            raise ValueError("Unknown report generation mode")
        if report_generation_mode == "model_synthesis" and report_synthesis_adapter is None:
            raise ValueError("Model report generation requires a synthesis adapter")
        self._report_generation_mode = report_generation_mode
        self._extraction_adapter = extraction_adapter
        self._report_memory_cache = report_memory_cache
        self._report_memory_cache_compatibility = (
            report_memory_cache_compatibility or ReportMemoryCacheCompatibility()
        )

    @property
    def detector_mode_metadata(self) -> dict[str, Any]:
        return self._detector_mode_selection.audit_metadata()

    @property
    def artifact_store(self) -> FilesystemArtifactBodyStore:
        return self._artifact_store

    def advance_run(self, run_id: str) -> dict[str, Any]:
        view = self._registry.get_run_view(run_id)
        if view["status"] != "analyzing" or view["current_stage"] is None:
            raise ValueError("Runtime run is not analyzing.")
        stage_id = view["current_stage"]
        try:
            if stage_id == "cache_lookup":
                return self._advance_cache_lookup(run_id, view)
            if stage_id == "extraction":
                return self._advance_extraction(run_id, view)
            if stage_id == "tool_analysis":
                return self._advance_tool_analysis(run_id)
            if stage_id == "detector_assessment":
                return self._advance_detector_assessment(run_id)
            if stage_id == "aggregation":
                return self._advance_aggregation(run_id)
            if stage_id == "report_generation":
                return self._advance_report_generation(run_id)
        except RuntimeStageExecutionError as error:
            return self._mark_failed(run_id, error)
        except Exception as error:
            return self._mark_failed(run_id, _stage_error(stage_id, error))
        raise ValueError(f"Runtime stage cannot be advanced: {stage_id}")

    def get_final_report_endpoint(self, run_id: str) -> dict[str, Any]:
        view = self._registry.get_run_view(run_id)
        if view["status"] != "completed" or not view["final_report"]["available"]:
            raise ValueError("Final report is not available for this Runtime Analysis Run.")
        ref = self._artifact_ref(run_id, "canonical_final_report_endpoint_json")
        payload = json.loads(self._artifact_store.read_bytes(ref).decode("utf-8"))
        return validate_final_report_endpoint(payload)

    def resume_run(self, run_id: str) -> dict[str, Any]:
        return self._registry.resume_from_latest_completed_stage(
            run_id,
            updated_at=self._timestamp(),
        )

    def _advance_cache_lookup(self, run_id: str, view: dict[str, Any]) -> dict[str, Any]:
        result = self._cache_lookup.lookup(view)
        if result.get("source_confirmation_candidate_overrides"):
            self._apply_source_confirmation_candidate_overrides(
                run_id,
                result["source_confirmation_candidate_overrides"],
            )
            view = self._registry.get_run_view(run_id)
        for ref in result.get("artifact_refs", []):
            self._add_artifact_ref(run_id, ref)
        reusable_refs = [
            ref
            for ref in result.get("reusable_report_memory_refs", [])
            if _is_explicit_reusable_report_memory_ref(ref)
        ]
        for ref in reusable_refs:
            self._add_artifact_ref(run_id, ref)
        warnings = copy.deepcopy(result.get("warnings", []))
        counts = {
            "source_slots_checked": 2,
            "reusable_report_memory_artifacts": len(reusable_refs),
            "cache_misses": max(0, 2 - len(reusable_refs)),
        }
        outcome = result.get("outcome")
        invalidity_traces = copy.deepcopy(result.get("invalidity_traces") or [])
        if invalidity_traces:
            self._registry.update_run_audit_metadata(
                run_id,
                {"cache_lookup_invalidity_traces": invalidity_traces},
            )
        reconfirm_roles = result.get("source_reconfirmation_required_roles") or []
        if outcome == "stale_rebuild_required" and reconfirm_roles:
            for warning in warnings:
                self._registry.add_warning(run_id, warning)
            source_confirmation = _source_confirmation_requiring_reconfirmation(
                view["source_confirmation"],
                roles={str(role) for role in reconfirm_roles},
                warnings=warnings,
            )
            return self._registry.require_source_reconfirmation(
                run_id,
                updated_at=self._timestamp(),
                source_confirmation=source_confirmation,
                cache_lookup_warnings=warnings,
            )
        if outcome in {"ambiguous_source_requires_resolution", "invalid_blocked"}:
            for warning in warnings:
                self._registry.add_warning(run_id, warning)
            return self._mark_failed(
                run_id,
                RuntimeStageExecutionError(
                    stage_id="cache_lookup",
                    code="cache_lookup_failed",
                    message="Runtime analysis stopped during cache lookup.",
                    detail=f"Locked Filing Cache Lookup outcome: {outcome}.",
                    recoverable=True,
                    can_resume=True,
                ),
            )
        if _has_complete_reusable_report_memory_pair(reusable_refs):
            cache_warning = {
                "code": "cache_reused_report_memory_artifacts",
                "severity": "info",
                "message": "Hiện vật ReportMemory được tái sử dụng sau bước kiểm tra độ sẵn sàng của nguồn.",
                "stage_id": "cache_lookup",
                "artifact_refs": copy.deepcopy(reusable_refs),
            }
            warnings.append(cache_warning)
            self._registry.add_warning(run_id, cache_warning)
            self._persist_confirmed_source_artifact_refs(run_id)
            report_set = self._load_report_set_from_report_memory_refs(reusable_refs)
            self._persist_report_set(run_id, report_set)
            return self._registry.complete_stage_and_activate(
                run_id,
                completed_stage_id="cache_lookup",
                next_stage_id="tool_analysis",
                updated_at=self._timestamp(),
                completed_counts=counts,
                completed_warnings=warnings,
                skipped_stage_updates=[
                    {
                        "stage_id": "extraction",
                        "counts": {
                            "report_memory_artifacts_reused": 2,
                            "live_extraction_runs": 0,
                        },
                        "warnings": [cache_warning],
                    }
                ],
            )
        return self._registry.complete_stage_and_activate(
            run_id,
            completed_stage_id="cache_lookup",
            next_stage_id="extraction",
            updated_at=self._timestamp(),
            completed_counts=counts,
            completed_warnings=warnings,
        )

    def _apply_source_confirmation_candidate_overrides(
        self,
        run_id: str,
        overrides: dict[str, Any],
    ) -> None:
        view = self._registry.get_run_view(run_id)
        source_confirmation = copy.deepcopy(view["source_confirmation"])
        changed = False
        for slot in source_confirmation.get("slots", []):
            if not isinstance(slot, dict):
                continue
            role = slot.get("role")
            override = overrides.get(role)
            candidate = slot.get("candidate")
            if not isinstance(override, dict) or not isinstance(candidate, dict):
                continue
            for key, value in override.items():
                if key == "audit_references" and isinstance(value, dict):
                    audit = candidate.setdefault("audit_references", {})
                    if isinstance(audit, dict):
                        audit.update(copy.deepcopy(value))
                else:
                    candidate[key] = copy.deepcopy(value)
            changed = True
        if changed:
            self._registry.update_source_confirmation(
                run_id,
                updated_at=self._timestamp(),
                source_confirmation=source_confirmation,
            )

    def _persist_confirmed_source_artifact_refs(self, run_id: str) -> None:
        packages_by_role = self._registry.get_run_audit_metadata(run_id).get(
            "live_extraction_source_packages"
        )
        if not isinstance(packages_by_role, dict):
            return
        existing_source_ref_keys = {
            (
                ref.get("kind"),
                (ref.get("metadata") or {}).get("report_role"),
                (ref.get("metadata") or {}).get("source_document_id"),
            )
            for ref in self._registry.get_artifact_refs(run_id)
            if ref.get("kind") in {"vietstock_source_pdf", "vietstock_source_package"}
        }
        for role, raw_package in packages_by_role.items():
            if not isinstance(raw_package, dict):
                continue
            package = validate_filing_package(copy.deepcopy(raw_package))
            document = _canonical_source_document(package)
            source_ref_key = ("vietstock_source_pdf", role, document.document_id)
            if source_ref_key in existing_source_ref_keys:
                continue
            local_path = Path(document.raw["local_artifact"]["path"])
            metadata = {
                "report_role": role,
                "source_document_id": document.document_id,
                "source_document_fingerprint_sha256": document.raw["fingerprint"]["hash_value"],
                "source_url": document.provenance.get("source_url"),
                "source_origin": document.provenance.get("source_name"),
                "source_container_type": (document.raw.get("vietstock_container") or {}).get("file_type"),
                "source_container_fingerprint_sha256": (document.raw.get("vietstock_container") or {}).get("hash_value"),
                "selected_package_member_filename": (document.raw.get("vietstock_member") or {}).get("filename"),
                "cache_outcome": "report_memory_reusable",
            }
            try:
                source_ref = self._artifact_store.reference_existing(
                    run_id=run_id,
                    artifact_id=f"artifact_cached_source_{role}",
                    kind="vietstock_source_pdf",
                    path=local_path,
                    version="live_vietstock_source_v1",
                    schema_version="vietstock_filing_source.v1",
                    metadata=metadata,
                )
            except OSError as error:
                raise RuntimeStageExecutionError(
                    stage_id="cache_lookup",
                    code="source_artifact_unreachable",
                    message="Cache lookup paused because a confirmed source artifact is unavailable.",
                    detail=f"Confirmed source artifact for role {role} could not be opened.",
                    recoverable=True,
                    can_resume=True,
                ) from error
            self._add_artifact_ref(run_id, source_ref)

    def ensure_confirmed_source_artifact_refs(self, run_id: str) -> None:
        self._persist_confirmed_source_artifact_refs(run_id)

    def _load_report_set_from_report_memory_refs(self, refs: list[dict[str, Any]]) -> CompanyReportSet:
        refs_by_role = {ref["metadata"]["report_role"]: ref for ref in refs}
        target = validate_report_memory(
            json.loads(self._artifact_store.read_bytes(refs_by_role["target"]).decode("utf-8"))
        )
        prior_year = validate_report_memory(
            json.loads(self._artifact_store.read_bytes(refs_by_role["prior_year_same_quarter"]).decode("utf-8"))
        )
        return validate_company_report_set(
            f"{target.report_id}_VS_{prior_year.report_id}",
            target,
            prior_year,
        )

    def _advance_extraction(self, run_id: str, view: dict[str, Any]) -> dict[str, Any]:
        if self._extraction_adapter is None:
            report_set = _demo_report_set_from_confirmed_sources(view)
            completed_counts = {
                "report_memories_built": 2,
                "company_report_sets": 1,
            }
        else:
            try:
                report_set = self._extraction_adapter.extract(
                    view,
                    self._registry.get_run_audit_metadata(run_id),
                )
            except Exception:
                self._persist_live_extraction_inputs(run_id, self._extraction_adapter)
                raise
            self._persist_live_extraction_inputs(run_id, self._extraction_adapter)
            cache_outcomes = getattr(
                self._extraction_adapter,
                "last_raw_cache_outcomes_by_role",
                {},
            ) or {}
            cache_hits = sum(outcome == "hit" for outcome in cache_outcomes.values())
            cache_misses = sum(outcome == "miss" for outcome in cache_outcomes.values())
            completed_counts = {
                "report_memories_built": 2,
                "company_report_sets": 1,
                "live_extraction_runs": cache_misses if cache_outcomes else 2,
            }
            if cache_outcomes:
                completed_counts.update(
                    {
                        "raw_ocr_cache_hits": cache_hits,
                        "raw_ocr_cache_misses": cache_misses,
                    }
                )
        extraction_warnings = []
        if self._extraction_adapter is not None and cache_outcomes:
            cache_activity_warning = {
                "code": "raw_ocr_cache_activity",
                "severity": "info",
                "message": (
                    f"OCR thô: tái sử dụng {cache_hits} hiện vật, tạo mới {cache_misses} hiện vật."
                ),
                "stage_id": "extraction",
                "artifact_refs": [],
            }
            extraction_warnings.append(cache_activity_warning)
            self._registry.add_warning(run_id, cache_activity_warning)
        self._persist_report_memories(run_id, report_set)
        self._store_report_memories_in_cache(view, report_set)
        self._persist_report_set(run_id, report_set)
        return self._registry.complete_stage_and_activate(
            run_id,
            completed_stage_id="extraction",
            next_stage_id="tool_analysis",
            updated_at=self._timestamp(),
            completed_counts=completed_counts,
            completed_warnings=extraction_warnings,
        )

    def _persist_live_extraction_inputs(self, run_id: str, extraction_adapter: Any) -> None:
        packages_by_role = getattr(extraction_adapter, "last_source_packages_by_role", {}) or {}
        artifacts_by_role = getattr(extraction_adapter, "last_raw_extraction_artifacts_by_role", {}) or {}
        cache_records_by_role = getattr(extraction_adapter, "last_raw_cache_records_by_role", {}) or {}
        cache_outcomes_by_role = getattr(extraction_adapter, "last_raw_cache_outcomes_by_role", {}) or {}
        normalization_metadata_by_role = getattr(
            extraction_adapter,
            "last_normalization_metadata_by_role",
            {},
        ) or {}
        for role, package in packages_by_role.items():
            document = _canonical_source_document(package)
            local_path = Path(document.raw["local_artifact"]["path"])
            metadata = {
                "report_role": role,
                "source_document_id": document.document_id,
                "source_document_fingerprint_sha256": document.raw["fingerprint"]["hash_value"],
                "source_url": document.provenance.get("source_url"),
                "source_origin": document.provenance.get("source_name"),
                "source_container_type": (document.raw.get("vietstock_container") or {}).get("file_type"),
                "source_container_fingerprint_sha256": (document.raw.get("vietstock_container") or {}).get("hash_value"),
                "selected_package_member_filename": (document.raw.get("vietstock_member") or {}).get("filename"),
            }
            try:
                source_ref = self._artifact_store.reference_existing(
                    run_id=run_id,
                    artifact_id=f"artifact_live_source_{role}",
                    kind="vietstock_source_pdf",
                    path=local_path,
                    version="live_vietstock_source_v1",
                    schema_version="vietstock_filing_source.v1",
                    metadata=metadata,
                )
            except OSError:
                continue
            self._add_artifact_ref(run_id, source_ref)
        for role, artifact in artifacts_by_role.items():
            cache_record = cache_records_by_role.get(role)
            metadata = {
                "report_role": role,
                "source_document_id": artifact.source_document_id,
                "ocr_artifact_id": artifact.artifact_id,
                "ocr_provider": artifact.raw.get("provider_metadata", {}).get("provider"),
                "ocr_status": artifact.raw.get("provider_metadata", {}).get("status"),
                "ocr_extraction_method": artifact.raw.get("extraction_method"),
                "ocr_extraction_version": artifact.raw.get("extraction_version"),
                "raw_ocr_cache_outcome": cache_outcomes_by_role.get(role),
            }
            normalization_metadata = normalization_metadata_by_role.get(role)
            if isinstance(normalization_metadata, dict):
                metadata["normalization"] = copy.deepcopy(normalization_metadata)
            if cache_record is not None:
                metadata.update(
                    {
                        "report_artifact_cache_id": cache_record.artifact_id,
                        "cache_validity_status": cache_record.validity_status,
                        "source_document_fingerprint_sha256": (
                            cache_record.identity.canonical_source_sha256
                        ),
                        "ocr_model": cache_record.identity.model,
                        "ocr_extraction_schema_version": (
                            cache_record.identity.extraction_schema_version
                        ),
                        "ocr_configuration_identity": (
                            cache_record.identity.configuration_identity
                        ),
                    }
                )
                self._add_artifact_ref(
                    run_id,
                    self._artifact_store.reference_existing(
                        run_id=run_id,
                        artifact_id=f"artifact_live_raw_extraction_{role}",
                        kind="ocr_artifact_json",
                        path=cache_record.body_path,
                        version=cache_record.identity.extraction_version,
                        schema_version=cache_record.identity.extraction_schema_version,
                        metadata=metadata,
                    ),
                )
                continue
            self._put_json(
                run_id=run_id,
                artifact_id=f"artifact_live_raw_extraction_{role}",
                kind="ocr_artifact_json",
                payload=artifact.raw,
                schema_version="raw_extraction_artifact.v1",
                metadata=metadata,
            )

    def _advance_tool_analysis(self, run_id: str) -> dict[str, Any]:
        report_set = self._load_report_set(run_id)
        tool_findings = run_deterministic_tool_checks(report_set)
        candidate_risks = build_candidate_risks(report_set, tool_findings)
        detector_packets = build_detector_packets(report_set, candidate_risks, tool_findings)
        self._put_json(
            run_id=run_id,
            artifact_id="artifact_tool_findings",
            kind="tool_findings_json",
            payload=[asdict(finding) for finding in tool_findings],
            schema_version="tool_findings.runtime_demo.v1",
        )
        self._put_json(
            run_id=run_id,
            artifact_id="artifact_candidate_risks",
            kind="candidate_risks_json",
            payload=[asdict(candidate) for candidate in candidate_risks],
            schema_version="candidate_risks.runtime_demo.v1",
        )
        self._put_json(
            run_id=run_id,
            artifact_id="artifact_detector_packets",
            kind="detector_packets_json",
            payload=[asdict(packet) for packet in detector_packets],
            schema_version="detector_packets.runtime_demo.v1",
            metadata={"runtime_detector_mode": self.detector_mode_metadata},
        )
        total_packets = len(detector_packets)
        return self._registry.complete_stage_and_activate(
            run_id,
            completed_stage_id="tool_analysis",
            next_stage_id="detector_assessment",
            updated_at=self._timestamp(),
            completed_counts={
                "tool_findings_total": len(tool_findings),
                "candidate_risks_total": len(candidate_risks),
                "detector_packets_total": total_packets,
            },
            next_progress={"processed": 0, "total": total_packets},
        )

    def _advance_detector_assessment(self, run_id: str) -> dict[str, Any]:
        packet_ref = self._artifact_ref(run_id, "detector_packets_json")
        packets = self._load_detector_packets(run_id)
        assessments = []
        guard_records = []
        try:
            for packet in packets:
                assessment = self._detector_adapter(packet)
                validate_detector_assessment(assessment, packet)
                assessments.append(assessment)
                guard_actions = _detector_contract_guard_actions(self._detector_adapter)
                if guard_actions is not None:
                    guard_records.append(
                        {
                            "packet_id": packet.packet_id,
                            "actions": guard_actions,
                        }
                    )
        except RuntimeStageExecutionError as error:
            raise _with_artifact_refs(error, [packet_ref]) from error
        except RuntimeDetectorTimeoutError as error:
            raise RuntimeStageExecutionError(
                stage_id="detector_assessment",
                code="detector_timeout",
                message="Analysis paused while assessing evidence packets.",
                detail=str(error),
                recoverable=True,
                can_resume=True,
                artifact_refs=[packet_ref],
            ) from error
        except RuntimeDetectorTransportError as error:
            raise RuntimeStageExecutionError(
                stage_id="detector_assessment",
                code="detector_transport_failure",
                message="Analysis paused because the configured detector service was unavailable.",
                detail=str(error),
                recoverable=True,
                can_resume=True,
                artifact_refs=[packet_ref],
            ) from error
        except RuntimeDetectorProviderResponseError as error:
            raise RuntimeStageExecutionError(
                stage_id="detector_assessment",
                code="detector_provider_response_invalid",
                message="Analysis paused because the detector service returned an invalid response.",
                detail=str(error),
                recoverable=True,
                can_resume=True,
                artifact_refs=[packet_ref],
            ) from error
        except RuntimeDetectorInvalidJsonError as error:
            raise RuntimeStageExecutionError(
                stage_id="detector_assessment",
                code="detector_output_invalid_json",
                message="Analysis paused because detector output was not valid JSON.",
                detail=str(error),
                recoverable=True,
                can_resume=True,
                artifact_refs=[packet_ref],
            ) from error
        except RuntimeDetectorGuardError as error:
            raise RuntimeStageExecutionError(
                stage_id="detector_assessment",
                code="detector_guard_unrecoverable",
                message="Analysis stopped because detector output could not be repaired without changing its meaning.",
                detail=str(error),
                recoverable=True,
                can_resume=True,
                artifact_refs=[packet_ref],
            ) from error
        except (TimeoutError, OSError) as error:
            raise RuntimeStageExecutionError(
                stage_id="detector_assessment",
                code="detector_timeout",
                message="Analysis paused while assessing evidence packets.",
                detail=str(error),
                recoverable=True,
                can_resume=True,
                artifact_refs=[packet_ref],
            ) from error
        except Exception as error:
            raise RuntimeStageExecutionError(
                stage_id="detector_assessment",
                code="detector_contract_invalid",
                message="Analysis stopped because detector output did not match the runtime contract.",
                detail=str(error),
                recoverable=True,
                can_resume=True,
                artifact_refs=[packet_ref],
            ) from error
        detector_metadata = {
            "runtime_detector_mode": self.detector_mode_metadata,
            "detector_contract_validation": {
                "status": "passed",
                "packet_count": len(packets),
                "assessment_count": len(assessments),
            },
        }
        guard_metadata = _detector_contract_guard_metadata(guard_records)
        if guard_metadata is not None:
            detector_metadata["detector_contract_guard"] = guard_metadata
        self._put_json(
            run_id=run_id,
            artifact_id="artifact_detector_assessments",
            kind="detector_assessments_json",
            payload=[asdict(assessment) for assessment in assessments],
            schema_version="detector_assessments.runtime_demo.v1",
            metadata=detector_metadata,
        )
        return self._registry.complete_stage_and_activate(
            run_id,
            completed_stage_id="detector_assessment",
            next_stage_id="aggregation",
            updated_at=self._timestamp(),
            completed_progress={"processed": len(assessments), "total": len(packets)},
            completed_counts={"detector_assessments_total": len(assessments)},
        )

    def _advance_aggregation(self, run_id: str) -> dict[str, Any]:
        report_set = self._load_report_set(run_id)
        candidates = self._load_candidate_risks(run_id)
        findings = self._load_tool_findings(run_id)
        assessments = self._load_detector_assessments(run_id)
        aggregation = aggregate_detector_assessments(report_set, candidates, findings, assessments)
        self._put_json(
            run_id=run_id,
            artifact_id="artifact_aggregation_output",
            kind="aggregation_output_json",
            payload=aggregation,
            schema_version="aggregation.runtime_demo.v1",
        )
        summary = aggregation["summary"]
        return self._registry.complete_stage_and_activate(
            run_id,
            completed_stage_id="aggregation",
            next_stage_id="report_generation",
            updated_at=self._timestamp(),
            completed_counts={
                "assessed_candidate_count": summary["assessed_candidate_count"],
                "supported_count": summary["supported_count"],
            },
        )

    def _advance_report_generation(self, run_id: str) -> dict[str, Any]:
        view = self._registry.get_run_view(run_id)
        report_set = self._load_report_set(run_id)
        candidates = [asdict(candidate) for candidate in self._load_candidate_risks(run_id)]
        findings = [asdict(finding) for finding in self._load_tool_findings(run_id)]
        assessments = [asdict(assessment) for assessment in self._load_detector_assessments(run_id)]
        report_generation_context = _report_generation_context_from_run_view(
            view,
            report_generation_mode=self._report_generation_mode,
        )
        final_report = build_final_report(
            report_set,
            candidates,
            findings,
            assessments,
            gating_records=[],
            report_generation_context=report_generation_context,
        )
        report_generation_mode = self._report_generation_mode
        synthesis_metadata = final_report["method_and_scope"].get("report_synthesis_model")
        if self._report_generation_mode == "model_synthesis":
            assert self._report_synthesis_adapter is not None
            model = view["runtime_config"]["report_synthesis_model"]
            report_language = report_generation_context.get("report_language", "vi")
            request = build_report_synthesis_request(
                final_report,
                findings,
                report_language=report_language,
            )
            invocation_number = 1 + sum(
                ref.get("kind") == "report_synthesis_request_json"
                for ref in self._registry.get_artifact_refs(run_id)
            )
            invocation_id = f"report_synthesis_{invocation_number:03d}"
            request_ref = self._put_json(
                run_id=run_id,
                artifact_id=f"artifact_{invocation_id}_request",
                kind="report_synthesis_request_json",
                payload=request,
                schema_version=request["schema_version"],
                metadata={
                    "invocation_id": invocation_id,
                    "provider": model["provider"],
                    "model_id": model["id"],
                    "invocation_status": "requested",
                    "prompt_version": request["prompt_version"],
                    "schema_version": request["schema_version"],
                    "decoding_version": request["decoding_version"],
                },
            )
            try:
                result = self._report_synthesis_adapter.synthesize(
                    model_id=model["id"],
                    request=copy.deepcopy(request),
                    response_schema=report_narrative_draft_json_schema(report_language=report_language),
                )
            except Exception as error:
                failure_ref = self._put_json(
                    run_id=run_id,
                    artifact_id=f"artifact_{invocation_id}_response",
                    kind="report_synthesis_response_json",
                    payload={
                        "status": "failed",
                        "error_type": type(error).__name__,
                    },
                    schema_version="report_synthesis_response.v1",
                    metadata={
                        "invocation_id": invocation_id,
                        "provider": model["provider"],
                        "model_id": model["id"],
                        "invocation_status": "failed",
                        "prompt_version": request["prompt_version"],
                        "schema_version": request["schema_version"],
                        "decoding_version": request["decoding_version"],
                        "request_artifact_id": request_ref["artifact_id"],
                    },
                )
                raise RuntimeStageExecutionError(
                    stage_id="report_generation",
                    code="final_report_invalid",
                    message="Report generation paused because the synthesis provider was unavailable.",
                    detail="The report synthesis invocation failed before a draft could be accepted.",
                    recoverable=True,
                    can_resume=True,
                    artifact_refs=[
                        _public_artifact_ref(request_ref),
                        _public_artifact_ref(failure_ref),
                    ],
                ) from error
            try:
                if not isinstance(result, dict) or not isinstance(result.get("draft"), dict):
                    raise ValueError("Report synthesis provider returned a malformed response")
                output_hash = report_synthesis_output_hash(result["draft"])
                final_report = merge_validated_report_narrative(
                    final_report,
                    request,
                    result["draft"],
                    model_id=model["id"],
                    provider=result.get("provider") or model["provider"],
                )
            except Exception as error:
                validation_ref = self._put_json(
                    run_id=run_id,
                    artifact_id=f"artifact_{invocation_id}_response",
                    kind="report_synthesis_response_json",
                    payload={
                        "status": "validation_failed",
                        "draft": result.get("draft") if isinstance(result, dict) else None,
                        "provider_response": result.get("response_body") if isinstance(result, dict) else None,
                    },
                    schema_version="report_synthesis_response.v1",
                    metadata={
                        "invocation_id": invocation_id,
                        "provider": result.get("provider") or model["provider"] if isinstance(result, dict) else model["provider"],
                        "model_id": result.get("model_id") or model["id"] if isinstance(result, dict) else model["id"],
                        "invocation_status": "validation_failed",
                        "prompt_version": request["prompt_version"],
                        "schema_version": request["schema_version"],
                        "decoding_version": request["decoding_version"],
                        "latency_ms": result.get("latency_ms") if isinstance(result, dict) else None,
                        "usage": copy.deepcopy(result.get("usage")) if isinstance(result, dict) else None,
                        "output_hash": (
                            report_synthesis_output_hash(result["draft"])
                            if isinstance(result, dict) and isinstance(result.get("draft"), dict)
                            else None
                        ),
                        "request_artifact_id": request_ref["artifact_id"],
                    },
                )
                raise RuntimeStageExecutionError(
                    stage_id="report_generation",
                    code="final_report_invalid",
                    message="Report generation paused because the synthesis draft was invalid.",
                    detail="The complete narrative draft was rejected before canonical report publication.",
                    recoverable=True,
                    can_resume=True,
                    artifact_refs=[
                        _public_artifact_ref(request_ref),
                        _public_artifact_ref(validation_ref),
                    ],
                ) from error
            response_ref = self._put_json(
                run_id=run_id,
                artifact_id=f"artifact_{invocation_id}_response",
                kind="report_synthesis_response_json",
                payload={
                    "draft": result["draft"],
                    "provider_response": result.get("response_body"),
                },
                schema_version="report_synthesis_response.v1",
                metadata={
                    "invocation_id": invocation_id,
                    "provider": result.get("provider") or model["provider"],
                    "model_id": result.get("model_id") or model["id"],
                    "invocation_status": "succeeded",
                    "prompt_version": request["prompt_version"],
                    "schema_version": request["schema_version"],
                    "decoding_version": request["decoding_version"],
                    "latency_ms": result.get("latency_ms"),
                    "usage": copy.deepcopy(result.get("usage")),
                    "output_hash": output_hash,
                    "request_artifact_id": request_ref["artifact_id"],
                },
            )
            report_generation_mode = "model_synthesis"
            synthesis_metadata = {
                **final_report["method_and_scope"]["report_synthesis_model"],
                "output_hash": output_hash,
                "request_artifact_id": request_ref["artifact_id"],
                "response_artifact_id": response_ref["artifact_id"],
            }
        report_markdown = render_final_report_markdown(final_report)
        generated_at = self._timestamp()
        final_report_metadata = {
            "report_generation_basis": final_report["report_basis"],
            "report_generation_mode": report_generation_mode,
            "report_synthesis_model": synthesis_metadata,
        }
        json_ref = self._put_json(
            run_id=run_id,
            artifact_id="artifact_final_report_json",
            kind="final_report_json",
            payload=final_report,
            schema_version="final_report.v1",
            metadata=final_report_metadata,
        )
        markdown_ref = self._put_bytes(
            run_id=run_id,
            artifact_id="artifact_final_report_markdown",
            kind="final_report_markdown",
            body=report_markdown.encode("utf-8"),
            schema_version="canonical_final_report.v1",
            metadata=final_report_metadata,
        )
        endpoint = validate_final_report_endpoint(
            {
                "schema_version": FINAL_REPORT_ENDPOINT_SCHEMA_VERSION,
                "run_id": run_id,
                "report_id": final_report["report_id"],
                "generated_at": generated_at,
                "report_json": final_report,
                "report_markdown": report_markdown,
                "artifact_refs": [_public_artifact_ref(json_ref), _public_artifact_ref(markdown_ref)],
            }
        )
        self._put_json(
            run_id=run_id,
            artifact_id="artifact_final_report_endpoint",
            kind="canonical_final_report_endpoint_json",
            payload=endpoint,
            schema_version=FINAL_REPORT_ENDPOINT_SCHEMA_VERSION,
            metadata=final_report_metadata,
        )
        return self._registry.mark_completed(
            run_id,
            updated_at=generated_at,
            report_id=final_report["report_id"],
            generated_at=generated_at,
            report_generation_counts={"final_reports_generated": 1},
        )

    def _persist_report_memories(self, run_id: str, report_set: CompanyReportSet) -> None:
        self._put_json(
            run_id=run_id,
            artifact_id="artifact_report_memory_target",
            kind="report_memory_json",
            payload=report_set.target.raw,
            schema_version="report_memory.v1",
            metadata={"report_role": "target"},
        )
        self._put_json(
            run_id=run_id,
            artifact_id="artifact_report_memory_prior_year",
            kind="report_memory_json",
            payload=report_set.prior_year.raw,
            schema_version="report_memory.v1",
            metadata={"report_role": "prior_year_same_quarter"},
        )

    def _persist_report_set(self, run_id: str, report_set: CompanyReportSet) -> None:
        self._put_json(
            run_id=run_id,
            artifact_id="artifact_company_report_set",
            kind="company_report_set_reference_json",
            payload={
                "case_id": report_set.case_id,
                "target_report_memory_artifact_id": "artifact_report_memory_target",
                "prior_year_report_memory_artifact_id": "artifact_report_memory_prior_year",
                "target": report_set.target.raw,
                "prior_year": report_set.prior_year.raw,
            },
            schema_version="company_report_set_reference.runtime_demo.v1",
        )

    def _store_report_memories_in_cache(self, view: dict[str, Any], report_set: CompanyReportSet) -> None:
        if self._report_memory_cache is None:
            return
        slots = {
            slot["role"]: slot
            for slot in view.get("source_confirmation", {}).get("slots", [])
            if isinstance(slot.get("candidate"), dict)
        }
        pairs = (
            ("target", report_set.target),
            ("prior_year_same_quarter", report_set.prior_year),
        )
        for role, report_memory in pairs:
            slot = slots.get(role)
            if slot is None:
                continue
            identity = ReportMemoryCacheIdentity.from_candidate(
                candidate=slot["candidate"],
                report_role=role,
                report_profile=report_memory.metadata["report_profile"],
                compatibility=self._report_memory_cache_compatibility_for_role(role),
            )
            self._report_memory_cache.store(identity=identity, report_memory=report_memory.raw)

    def _report_memory_cache_compatibility_for_role(self, role: str) -> ReportMemoryCacheCompatibility:
        compatibilities_by_role = getattr(
            self._extraction_adapter,
            "last_report_memory_cache_compatibility_by_role",
            {},
        ) or {}
        compatibility = compatibilities_by_role.get(role)
        if isinstance(compatibility, ReportMemoryCacheCompatibility):
            return compatibility
        return self._report_memory_cache_compatibility

    def _load_report_set(self, run_id: str) -> CompanyReportSet:
        ref = self._artifact_ref(run_id, "company_report_set_reference_json")
        payload = json.loads(self._artifact_store.read_bytes(ref).decode("utf-8"))
        target = validate_report_memory(payload["target"])
        prior_year = validate_report_memory(payload["prior_year"])
        return validate_company_report_set(payload["case_id"], target, prior_year)

    def _load_tool_findings(self, run_id: str) -> list[ToolFinding]:
        ref = self._artifact_ref(run_id, "tool_findings_json")
        payload = json.loads(self._artifact_store.read_bytes(ref).decode("utf-8"))
        return [ToolFinding(**item) for item in payload]

    def _load_candidate_risks(self, run_id: str) -> list[CandidateRisk]:
        ref = self._artifact_ref(run_id, "candidate_risks_json")
        payload = json.loads(self._artifact_store.read_bytes(ref).decode("utf-8"))
        return [CandidateRisk(**item) for item in payload]

    def _load_detector_packets(self, run_id: str) -> list[DetectorPacket]:
        ref = self._artifact_ref(run_id, "detector_packets_json")
        payload = json.loads(self._artifact_store.read_bytes(ref).decode("utf-8"))
        return [DetectorPacket(**item) for item in payload]

    def _load_detector_assessments(self, run_id: str) -> list[DetectorAssessment]:
        ref = self._artifact_ref(run_id, "detector_assessments_json")
        payload = json.loads(self._artifact_store.read_bytes(ref).decode("utf-8"))
        return [DetectorAssessment(**item) for item in payload]

    def _put_json(
        self,
        *,
        run_id: str,
        artifact_id: str,
        kind: str,
        payload: Any,
        schema_version: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._put_bytes(
            run_id=run_id,
            artifact_id=artifact_id,
            kind=kind,
            body=stable_json_dumps(payload).encode("utf-8"),
            schema_version=schema_version,
            metadata=metadata,
        )

    def _put_bytes(
        self,
        *,
        run_id: str,
        artifact_id: str,
        kind: str,
        body: bytes,
        schema_version: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ref = self._artifact_store.put_bytes(
            run_id=run_id,
            artifact_id=artifact_id,
            kind=kind,
            body=body,
            schema_version=schema_version,
            version="runtime_demo_v1",
            metadata=metadata,
        )
        return self._add_artifact_ref(run_id, ref)

    def _add_artifact_ref(self, run_id: str, ref: dict[str, Any]) -> dict[str, Any]:
        try:
            return self._registry.add_artifact_ref(run_id, ref)
        except ValueError as error:
            if "already exists" not in str(error):
                raise
            for existing in self._registry.get_artifact_refs(run_id):
                if existing["artifact_id"] == ref["artifact_id"]:
                    return existing
            raise

    def _artifact_ref(self, run_id: str, kind: str) -> dict[str, Any]:
        for ref in reversed(self._registry.get_artifact_refs(run_id)):
            if ref.get("kind") == kind:
                return ref
        raise ValueError(f"Missing artifact kind: {kind}")

    def _mark_failed(self, run_id: str, error: RuntimeStageExecutionError) -> dict[str, Any]:
        return self._registry.mark_failed(
            run_id,
            stage_id=error.stage_id,
            updated_at=self._timestamp(),
            error=error.to_error_view(),
        )


def _stage_error(stage_id: str, error: Exception) -> RuntimeStageExecutionError:
    code_by_stage = {
        "cache_lookup": "cache_lookup_failed",
        "extraction": "extraction_failed",
        "tool_analysis": "tool_analysis_failed",
        "detector_assessment": "detector_contract_invalid",
        "aggregation": "aggregation_failed",
        "report_generation": "final_report_invalid",
    }
    return RuntimeStageExecutionError(
        stage_id=stage_id,
        code=code_by_stage.get(stage_id, "internal_error"),
        message=f"Runtime analysis stopped during {stage_id.replace('_', ' ')}.",
        detail=str(error),
        recoverable=True,
        can_resume=True,
    )


def _with_artifact_refs(
    error: RuntimeStageExecutionError,
    artifact_refs: list[dict[str, Any]],
) -> RuntimeStageExecutionError:
    return RuntimeStageExecutionError(
        stage_id=error.stage_id,
        code=error.code,
        message=error.message,
        detail=error.detail,
        recoverable=error.recoverable,
        can_resume=error.can_resume,
        artifact_refs=error.artifact_refs or artifact_refs,
    )


def _detector_contract_guard_actions(adapter: DetectorAdapter) -> list[str] | None:
    actions = getattr(adapter, "last_contract_guard_actions", None)
    if actions is None:
        return None
    if isinstance(actions, (list, tuple)):
        return [str(action) for action in actions]
    return []


def _detector_contract_guard_metadata(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None
    action_counts: dict[str, int] = {}
    for record in records:
        for action in record["actions"]:
            action_counts[action] = action_counts.get(action, 0) + 1
    return {
        "enabled": True,
        "version": DEFAULT_SFT_CONTRACT_GUARD_VERSION,
        "packet_count": len(records),
        "action_counts": dict(sorted(action_counts.items())),
        "records": copy.deepcopy(records),
        "support_level_or_severity_overrides": False,
        "rationale_overrides": False,
        "strict_validation_after_guard": True,
    }


def _demo_report_set_from_confirmed_sources(view: dict[str, Any]) -> CompanyReportSet:
    slots = {slot["role"]: slot for slot in view["source_confirmation"]["slots"]}
    target = _demo_report_memory(
        slots["target"]["candidate"],
        revenue=150_000,
        receivables=90_000,
        profit_after_tax=120_000,
        operating_cash_flow=-30_000,
    )
    prior_year = _demo_report_memory(
        slots["prior_year_same_quarter"]["candidate"],
        revenue=100_000,
        receivables=45_000,
        profit_after_tax=100_000,
        operating_cash_flow=70_000,
    )
    return validate_company_report_set(
        f"{target.report_id}_VS_{prior_year.report_id}",
        target,
        prior_year,
    )


def _source_confirmation_requiring_reconfirmation(
    source_confirmation: dict[str, Any],
    *,
    roles: set[str],
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = copy.deepcopy(source_confirmation)
    payload["status"] = "partially_rejected"
    payload["confirmable"] = False
    payload["hitl_boundary"] = payload.get("hitl_boundary", HITL_BOUNDARY)
    role_warnings = {
        role: [
            copy.deepcopy(warning)
            for warning in warnings
            if warning.get("source_slot_role") == role
        ]
        for role in roles
    }
    for slot in payload.get("slots", []):
        role = slot.get("role")
        if role in roles:
            slot["status"] = "unavailable"
            slot["candidate"] = None
            slot["rejection"] = None
            slot["warnings"] = role_warnings.get(role) or [
                {
                    "code": "filing_cache_lookup_stale_rebuild_required",
                    "severity": "warning",
                    "message": "ReportMemory cache reuse is blocked until source safety is established.",
                    "stage_id": "cache_lookup",
                    "artifact_refs": [],
                    "source_slot_role": role,
                }
            ]
        elif slot.get("status") in {"confirmed", "locked"}:
            slot["status"] = "ready_for_review"
            slot["rejection"] = None
    payload["package_warnings"] = copy.deepcopy(warnings)
    return payload


def _canonical_source_document(package: Any) -> Any:
    canonical_id = select_canonical_source(package).canonical_source_document_id
    for document in package.source_documents:
        if document.document_id == canonical_id:
            return document
    raise ValueError("Canonical source document is unavailable.")


def _demo_report_memory(
    candidate: dict[str, Any],
    *,
    revenue: int,
    receivables: int,
    profit_after_tax: int,
    operating_cash_flow: int,
) -> ReportMemory:
    report_id = _report_id(candidate)
    rows = [
        _row(report_id, "ROW_REVENUE", "revenue", "01", "Revenue", revenue),
        _row(report_id, "ROW_RECEIVABLES", "trade_receivables", "131", "Trade receivables", receivables),
        _row(report_id, "ROW_PROFIT", "profit_after_tax", "60", "Profit after tax", profit_after_tax),
        _row(report_id, "ROW_OCF", "operating_cash_flow", "20", "Operating cash flow", operating_cash_flow),
    ]
    cells = [cell for row in rows for cell in row["cells"]]
    return validate_report_memory(
        {
            "report_id": report_id,
            "metadata": {
                "company_name": candidate["company_name_vi"],
                "period": f"{candidate['fiscal_year']}-Q{candidate['quarter']}",
                "report_period_type": "quarterly",
                "report_profile": "standard_corporate",
                "report_basis": candidate["report_basis"],
                "business_context_tags": ["runtime_demo"],
                "report_assurance_type": candidate["filing_status"],
                "currency": "VND",
                "unit": "million_vnd",
                "filing_status": "original",
                "canonical_source_document_id": candidate["source_document_id"],
                "source_document_fingerprint_sha256": candidate["audit_references"][
                    "source_document_fingerprint_sha256"
                ],
                "source_file": candidate["source_url"],
                "extraction_method": "runtime_demo_report_memory_v1",
            },
            "sections": [],
            "tables": [
                {
                    "table_id": f"TBL_{report_id}_IS",
                    "table_type": "income_statement",
                    "rows": rows,
                }
            ],
            "notes": [],
            "variance_explanations": [],
            "cell_index": {
                cell["cell_id"]: {
                    "table_id": f"TBL_{report_id}_IS",
                    "row_id": cell["row_id"],
                }
                for cell in cells
            },
        }
    )


def _row(report_id: str, row_id: str, standard_account: str, account_code: str, label: str, value: int) -> dict[str, Any]:
    cell_id = f"CELL_{row_id}_{report_id}"
    return {
        "row_id": row_id,
        "standard_account": standard_account,
        "account_code": account_code,
        "label": label,
        "cells": [
            {
                "cell_id": cell_id,
                "row_id": row_id,
                "period": _period_from_report_id(report_id),
                "value": value,
            }
        ],
    }


def _report_id(candidate: dict[str, Any]) -> str:
    return (
        f"{candidate['ticker'].upper()}_"
        f"{candidate['fiscal_year']}_"
        f"Q{candidate['quarter']}_"
        f"{candidate['report_basis'].upper()}"
    )


def _period_from_report_id(report_id: str) -> str:
    parts = report_id.split("_")
    return f"{parts[1]}-{parts[2]}"


def _is_explicit_reusable_report_memory_ref(ref: dict[str, Any]) -> bool:
    metadata = ref.get("metadata") if isinstance(ref, dict) else None
    return (
        isinstance(ref, dict)
        and ref.get("kind") == "report_memory_json"
        and isinstance(metadata, dict)
        and metadata.get("report_role") in {"target", "prior_year_same_quarter"}
        and ref.get("artifact_id")
        and ref.get("sha256")
    )


def _has_complete_reusable_report_memory_pair(refs: list[dict[str, Any]]) -> bool:
    return len(refs) == 2 and {ref["metadata"]["report_role"] for ref in refs} == {
        "target",
        "prior_year_same_quarter",
    }


def _public_artifact_ref(ref: dict[str, Any]) -> dict[str, Any]:
    return {
        key: ref[key]
        for key in ["artifact_id", "kind", "sha256"]
        if key in ref
    }


def _report_generation_context_from_run_view(
    view: dict[str, Any],
    *,
    report_generation_mode: str,
) -> dict[str, Any]:
    runtime_config = view.get("runtime_config") if isinstance(view, dict) else None
    if not isinstance(runtime_config, dict):
        return {}
    report_synthesis_model = runtime_config.get("report_synthesis_model")
    if not isinstance(report_synthesis_model, dict):
        return {}
    filing_intent = view.get("filing_intent") if isinstance(view, dict) else None
    report_language = "vi"
    if isinstance(filing_intent, dict) and filing_intent.get("report_language") in {"vi", "en"}:
        report_language = filing_intent["report_language"]
    return {
        "report_synthesis_model": copy.deepcopy(report_synthesis_model),
        "report_generation_mode": report_generation_mode,
        "report_language": report_language,
    }
