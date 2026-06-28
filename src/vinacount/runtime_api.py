from __future__ import annotations

import copy
import hashlib
import os
import zipfile
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from vinacount.detector_contract import DetectorAdapter
from vinacount.env_loader import load_dotenv_if_available
from vinacount.runtime_contract import (
    REPORT_SYNTHESIS_MODEL_IDS,
    REJECTION_REASON_CODES,
    REPORT_BASIS_PREFERENCES,
    SOURCE_SLOT_ROLES,
    validate_filing_intent_error_response,
    validate_runtime_run_create_request,
)
from vinacount.runtime_audit_bundle import DeveloperAuditBundleBuilder
from vinacount.runtime_detector_modes import (
    RuntimeDetectorModeRegistry,
    runtime_detector_mode_from_environment,
)
from vinacount.runtime_orchestration import (
    CacheLookupAdapter,
    DemoCacheLookupAdapter,
    FilingCacheLookupAdapter,
    RuntimeStageSpine,
)
from vinacount.runtime_live_extraction import LiveNanonetsReportMemoryExtractionAdapter
from vinacount.raw_ocr_llm_normalizer import make_deepseek_raw_ocr_normalizer_from_env
from vinacount.report_synthesis import (
    ReportSynthesisAdapter,
    make_deepseek_report_synthesis_adapter_from_env,
)
from vinacount.runtime_run_registry import (
    FilesystemArtifactBodyStore,
    RuntimeRunRegistry,
    SqliteRuntimeRunStore,
)
from vinacount.runtime_vietstock_source_discovery import VietstockRuntimeSourceDiscoveryAdapter
from vinacount.nanonets_provider_client import make_nanonets_ocr3_docstrange_adapter_from_env
from vinacount.report_artifact_cache import (
    RAW_EXTRACTION_ARTIFACT_SCHEMA_VERSION,
    PostgresReportArtifactCacheRepository,
    RawOcrArtifactCache,
    RawOcrCacheIdentity,
    ReportArtifactCacheRepository,
    ReportMemoryArtifactCache,
    SqliteReportArtifactCacheRepository,
)


RunIdFactory = Callable[[], str]
Clock = Callable[[], datetime]


def create_runtime_app(
    *,
    registry: RuntimeRunRegistry | None = None,
    source_discovery: SourceDiscoveryAdapter | None = None,
    artifact_store: FilesystemArtifactBodyStore | None = None,
    cache_lookup: CacheLookupAdapter | None = None,
    detector_adapter: DetectorAdapter | None = None,
    detector_mode: str | None = None,
    detector_mode_registry: RuntimeDetectorModeRegistry | None = None,
    report_synthesis_adapter: ReportSynthesisAdapter | None = None,
    report_generation_mode: str | None = None,
    extraction_adapter: Any | None = None,
    raw_ocr_cache: RawOcrArtifactCache | None = None,
    report_memory_cache: ReportMemoryArtifactCache | None = None,
    stage_spine: RuntimeStageSpine | None = None,
    run_id_factory: RunIdFactory | None = None,
    clock: Clock | None = None,
    auto_advance: bool | None = None,
) -> FastAPI:
    load_dotenv_if_available()
    clock = clock or _utcnow
    artifact_store = artifact_store or FilesystemArtifactBodyStore(Path("/tmp") / "vinacount_runtime_artifacts")
    registry = registry or _runtime_run_registry_from_environment(artifact_store)
    cached_first_live_confirmation = _runtime_cached_first_live_confirmation_enabled()
    report_artifact_cache_repository = _report_artifact_cache_repository(artifact_store)
    raw_ocr_cache = raw_ocr_cache or RawOcrArtifactCache(
        repository=report_artifact_cache_repository,
        body_store=artifact_store,
    )
    report_memory_cache = report_memory_cache or ReportMemoryArtifactCache(
        repository=report_artifact_cache_repository,
        body_store=artifact_store,
    )
    source_discovery_was_supplied = source_discovery is not None
    if cached_first_live_confirmation:
        cached_first_cache_lookup_adapter, _, _ = _thesis_demo_components()
        source_discovery = source_discovery or VietstockRuntimeSourceDiscoveryAdapter(clock=clock)
        cache_lookup = cache_lookup or cached_first_cache_lookup_adapter(
            artifact_store=artifact_store,
        )
    elif _runtime_locked_thesis_demo_enabled():
        _, locked_cache_lookup_adapter, locked_source_discovery_adapter = _thesis_demo_components()
        source_discovery = source_discovery or locked_source_discovery_adapter()
        cache_lookup = cache_lookup or locked_cache_lookup_adapter(artifact_store=artifact_store)
    else:
        source_discovery = source_discovery or VietstockRuntimeSourceDiscoveryAdapter(clock=clock)
        cache_lookup = cache_lookup or FilingCacheLookupAdapter(
            report_memory_cache=report_memory_cache,
            artifact_store=artifact_store,
            raw_ocr_cache=raw_ocr_cache,
            raw_ocr_identity_factory=_raw_ocr_identity_for_candidate,
        )
    if report_generation_mode is not None:
        selected_report_generation_mode = report_generation_mode
    else:
        selected_report_generation_mode = os.environ.get(
            "VINACOUNT_REPORT_GENERATION_MODE",
            "model_synthesis",
        )
    if selected_report_generation_mode not in {"model_synthesis", "deterministic_template"}:
        raise ValueError("Unknown report generation mode")
    if selected_report_generation_mode == "deterministic_template" and report_synthesis_adapter is not None:
        raise ValueError("Deterministic report generation mode cannot use a synthesis adapter")
    if selected_report_generation_mode == "model_synthesis" and report_synthesis_adapter is None:
        report_synthesis_adapter = make_deepseek_report_synthesis_adapter_from_env()
    if (
        extraction_adapter is None
        and not source_discovery_was_supplied
        and not _runtime_locked_thesis_demo_enabled()
        and not cached_first_live_confirmation
    ):
        extraction_adapter = LiveNanonetsReportMemoryExtractionAdapter(
            ocr_adapter=make_nanonets_ocr3_docstrange_adapter_from_env(),
            raw_ocr_cache=raw_ocr_cache,
            raw_ocr_normalizer=make_deepseek_raw_ocr_normalizer_from_env(),
        )
    stage_spine = stage_spine or RuntimeStageSpine(
        registry=registry,
        artifact_store=artifact_store,
        timestamp_factory=lambda: _runtime_timestamp(clock()),
        cache_lookup=cache_lookup,
        detector_adapter=detector_adapter,
        detector_mode=detector_mode
        or ("deterministic_local" if cached_first_live_confirmation else runtime_detector_mode_from_environment()),
        detector_mode_registry=detector_mode_registry,
        report_synthesis_adapter=report_synthesis_adapter,
        report_generation_mode=selected_report_generation_mode,
        extraction_adapter=extraction_adapter,
        report_memory_cache=report_memory_cache,
    )
    service = RuntimeApiService(
        registry=registry,
        source_discovery=source_discovery or DemoSourceDiscoveryAdapter(),
        run_id_factory=run_id_factory or _default_run_id,
        clock=clock,
        stage_spine=stage_spine,
        audit_bundle_builder=DeveloperAuditBundleBuilder(
            registry=registry,
            artifact_store=stage_spine.artifact_store,
            timestamp_factory=lambda: _runtime_timestamp(clock()),
        ),
        auto_advance=(
            (cached_first_live_confirmation or _runtime_auto_advance_enabled())
            if auto_advance is None
            else auto_advance
        ),
    )
    app = FastAPI(title="Vinacount Runtime API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_runtime_cors_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )
    app.state.runtime_service = service

    @app.post("/runtime-runs")
    async def create_runtime_run(request: Request) -> JSONResponse:
        payload = await request.json()
        try:
            view = service.create_runtime_run(payload)
        except ValueError as exc:
            return JSONResponse(
                status_code=422,
                content=_filing_intent_error_response(payload, str(exc)),
            )
        return JSONResponse(status_code=201, content=view)

    @app.get("/runtime-runs/{run_id}")
    async def get_runtime_run(run_id: str) -> JSONResponse:
        try:
            view = service.get_runtime_run(run_id)
        except KeyError:
            return JSONResponse(
                status_code=404,
                content={"error": {"code": "runtime_run_not_found", "message": "Runtime run was not found."}},
            )
        return JSONResponse(status_code=200, content=view)

    @app.post("/runtime-runs/{run_id}/source-confirmation")
    async def update_source_confirmation(
        run_id: str,
        request: Request,
        background_tasks: BackgroundTasks,
    ) -> JSONResponse:
        payload = await request.json()
        try:
            view = service.update_source_confirmation(run_id, payload)
        except KeyError:
            return JSONResponse(
                status_code=404,
                content={"error": {"code": "runtime_run_not_found", "message": "Runtime run was not found."}},
            )
        except ValueError as exc:
            return JSONResponse(
                status_code=400,
                content={"error": {"code": "source_confirmation_invalid", "message": str(exc)}},
            )
        service.schedule_auto_advance(run_id, view, background_tasks)
        return JSONResponse(status_code=200, content=view)

    @app.post("/runtime-runs/{run_id}/actions/stop")
    async def stop_runtime_run(run_id: str) -> JSONResponse:
        try:
            view = service.stop_runtime_run(run_id)
        except KeyError:
            return JSONResponse(
                status_code=404,
                content={"error": {"code": "runtime_run_not_found", "message": "Runtime run was not found."}},
            )
        except ValueError as exc:
            return JSONResponse(
                status_code=422,
                content={"error": {"code": "runtime_run_action_invalid", "message": str(exc)}},
            )
        return JSONResponse(status_code=200, content=view)

    @app.post("/runtime-runs/{run_id}/actions/resume")
    async def resume_runtime_run(run_id: str, background_tasks: BackgroundTasks) -> JSONResponse:
        try:
            view = service.resume_runtime_run(run_id)
        except KeyError:
            return JSONResponse(
                status_code=404,
                content={"error": {"code": "runtime_run_not_found", "message": "Runtime run was not found."}},
            )
        except ValueError as exc:
            return JSONResponse(
                status_code=422,
                content={"error": {"code": "runtime_run_action_invalid", "message": str(exc)}},
            )
        service.schedule_auto_advance(run_id, view, background_tasks)
        return JSONResponse(status_code=200, content=view)

    @app.get("/runtime-runs/{run_id}/report")
    async def get_final_report(run_id: str) -> JSONResponse:
        try:
            report = service.get_final_report(run_id)
        except KeyError:
            return JSONResponse(
                status_code=404,
                content={"error": {"code": "runtime_run_not_found", "message": "Runtime run was not found."}},
            )
        except ValueError as exc:
            return JSONResponse(
                status_code=409,
                content={"error": {"code": "final_report_unavailable", "message": str(exc)}},
            )
        return JSONResponse(status_code=200, content=report)

    @app.get("/runtime-runs/{run_id}/source-documents/{source_document_id}/pdf")
    async def get_source_document_pdf(run_id: str, source_document_id: str) -> Response:
        try:
            document = service.get_source_document_pdf(run_id, source_document_id)
        except KeyError:
            return JSONResponse(
                status_code=404,
                content={"error": {"code": "runtime_run_not_found", "message": "Runtime run was not found."}},
            )
        except SourceDocumentUnavailable as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={"error": {"code": exc.code, "message": exc.message}},
            )
        return Response(
            content=document["body"],
            media_type="application/pdf",
            headers=document["headers"],
        )

    @app.get("/runtime-runs/{run_id}/developer-audit-bundle")
    async def get_developer_audit_bundle(run_id: str) -> JSONResponse:
        try:
            bundle = service.get_developer_audit_bundle(run_id)
        except KeyError:
            return JSONResponse(
                status_code=404,
                content={"error": {"code": "runtime_run_not_found", "message": "Runtime run was not found."}},
            )
        except ValueError as exc:
            return JSONResponse(
                status_code=409,
                content={"error": {"code": "developer_audit_bundle_unavailable", "message": str(exc)}},
            )
        return JSONResponse(status_code=200, content=bundle)

    return app


def create_public_demo_runtime_app(
    *,
    registry: RuntimeRunRegistry | None = None,
    artifact_store: FilesystemArtifactBodyStore | None = None,
    run_id_factory: RunIdFactory | None = None,
    auto_advance: bool = True,
) -> FastAPI:
    """Create the no-key reviewer demo API used by the public submission package."""
    return create_runtime_app(
        registry=registry,
        artifact_store=artifact_store,
        source_discovery=DemoSourceDiscoveryAdapter(),
        cache_lookup=DemoCacheLookupAdapter(),
        run_id_factory=run_id_factory,
        report_generation_mode="deterministic_template",
        detector_mode="deterministic_local",
        auto_advance=auto_advance,
    )


class RuntimeApiService:
    def __init__(
        self,
        *,
        registry: RuntimeRunRegistry,
        source_discovery: SourceDiscoveryAdapter,
        run_id_factory: RunIdFactory,
        clock: Clock,
        stage_spine: RuntimeStageSpine,
        audit_bundle_builder: DeveloperAuditBundleBuilder,
        auto_advance: bool = False,
    ) -> None:
        self._registry = registry
        self._source_discovery = source_discovery
        self._run_id_factory = run_id_factory
        self._clock = clock
        self._stage_spine = stage_spine
        self._audit_bundle_builder = audit_bundle_builder
        self._auto_advance = auto_advance

    def create_runtime_run(self, payload: Any) -> dict[str, Any]:
        request = validate_runtime_run_create_request(payload)
        run_id = self._run_id_factory()
        created_at = _runtime_timestamp(self._clock())
        self._registry.create_run(
            run_id=run_id,
            created_at=created_at,
            filing_intent=request,
            report_synthesis_model_id=request.get("report_synthesis_model_id"),
            runtime_detector_mode=self._stage_spine.detector_mode_metadata,
        )
        self._registry.update_stage(
            run_id,
            "source_discovery",
            status="active",
            started_at=created_at,
            updated_at=created_at,
        )
        source_confirmation = self._source_discovery.discover(request)
        self._capture_internal_live_extraction_source_packages(run_id, source_confirmation)
        self._registry.update_stage(
            run_id,
            "source_confirmation",
            status="active",
            started_at=created_at,
            updated_at=created_at,
            completed_stage_ids=["source_discovery"],
        )
        return self._registry.update_source_confirmation(
            run_id,
            updated_at=created_at,
            source_confirmation=source_confirmation,
        )

    def get_runtime_run(self, run_id: str) -> dict[str, Any]:
        return self._registry.get_run_view(run_id)

    def update_source_confirmation(self, run_id: str, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("Source confirmation action body must be an object.")
        action = payload.get("action")
        if action == "confirm_sources":
            return self._confirm_sources(run_id)
        if action == "reject_source":
            return self._reject_source(run_id, payload)
        if action == "retry_source_discovery":
            return self._retry_source_discovery(run_id, payload)
        if action == "select_source_candidate":
            return self._select_source_candidate(run_id, payload)
        raise ValueError("Unknown source confirmation action.")

    def _confirm_sources(self, run_id: str) -> dict[str, Any]:
        view = self._registry.get_run_view(run_id)
        source_confirmation = copy.deepcopy(view["source_confirmation"])
        if not source_confirmation["confirmable"]:
            raise ValueError("Source package is not confirmable.")
        for slot in source_confirmation["slots"]:
            if slot["status"] != "ready_for_review":
                raise ValueError("Every source slot must be ready before confirmation.")
            slot["status"] = "locked"
            slot["rejection"] = None
        source_confirmation["status"] = "confirmed"
        source_confirmation["confirmable"] = False
        updated_at = _runtime_timestamp(self._clock())
        self._registry.update_stage(
            run_id,
            "cache_lookup",
            status="active",
            started_at=updated_at,
            updated_at=updated_at,
            completed_stage_ids=["source_confirmation"],
        )
        return self._registry.update_source_confirmation(
            run_id,
            updated_at=updated_at,
            source_confirmation=source_confirmation,
        )

    def _reject_source(self, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        slot_role = _source_slot_role(payload)
        reason_code = _rejection_reason_code(payload)
        comment = payload.get("comment")
        if comment is not None and (not isinstance(comment, str) or not comment.strip()):
            raise ValueError("comment must be a non-empty string when provided.")

        view = self._registry.get_run_view(run_id)
        if view["status"] != "awaiting_source_confirmation":
            raise ValueError("Source confirmation actions are only valid while awaiting source confirmation.")
        source_confirmation = copy.deepcopy(view["source_confirmation"])
        slot = _slot_by_role(source_confirmation, slot_role)
        if slot["status"] != "ready_for_review":
            raise ValueError("Only ready source slots can be rejected.")

        slot["status"] = "rejected"
        slot["rejection"] = {
            "reason_code": reason_code,
            "message": _rejection_message(reason_code),
            "comment": comment,
        }
        source_confirmation["status"] = "partially_rejected"
        source_confirmation["confirmable"] = False
        return self._registry.update_source_confirmation(
            run_id,
            updated_at=_runtime_timestamp(self._clock()),
            source_confirmation=source_confirmation,
        )

    def _retry_source_discovery(self, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        slot_role = _source_slot_role(payload)
        view = self._registry.get_run_view(run_id)
        if view["status"] != "awaiting_source_confirmation":
            raise ValueError("Source confirmation actions are only valid while awaiting source confirmation.")

        source_confirmation = copy.deepcopy(view["source_confirmation"])
        slot = _slot_by_role(source_confirmation, slot_role)
        if slot["status"] not in {"rejected", "unavailable"}:
            raise ValueError("Only rejected or unavailable source slots can be retried.")

        rediscovered = self._source_discovery.discover(view["filing_intent"])
        self._capture_internal_live_extraction_source_packages(run_id, rediscovered)
        rediscovered_slot = _slot_by_role(rediscovered, slot_role)
        slot["status"] = rediscovered_slot["status"]
        slot["candidate"] = copy.deepcopy(rediscovered_slot["candidate"])
        slot["candidate_documents"] = copy.deepcopy(rediscovered_slot.get("candidate_documents", []))
        slot["rejection"] = copy.deepcopy(rediscovered_slot["rejection"])
        slot["warnings"] = copy.deepcopy(rediscovered_slot["warnings"])
        source_confirmation["confirmable"] = all(
            item["status"] == "ready_for_review" for item in source_confirmation["slots"]
        )
        source_confirmation["status"] = (
            "ready_for_review" if source_confirmation["confirmable"] else "partially_rejected"
        )
        return self._registry.update_source_confirmation(
            run_id,
            updated_at=_runtime_timestamp(self._clock()),
            source_confirmation=source_confirmation,
        )

    def _select_source_candidate(self, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        slot_role = _source_slot_role(payload)
        source_document_id = payload.get("source_document_id")
        if not isinstance(source_document_id, str) or not source_document_id.strip():
            raise ValueError("source_document_id must be a non-empty string.")
        view = self._registry.get_run_view(run_id)
        if view["status"] != "awaiting_source_confirmation":
            raise ValueError("Source confirmation actions are only valid while awaiting source confirmation.")

        source_confirmation = copy.deepcopy(view["source_confirmation"])
        slot = _slot_by_role(source_confirmation, slot_role)
        candidate_documents = slot.get("candidate_documents") or []
        candidates_by_id = {
            candidate["source_document_id"]: candidate
            for candidate in candidate_documents
            if isinstance(candidate, dict)
        }
        if not candidates_by_id:
            raise ValueError("Source slot does not require source candidate selection.")
        if source_document_id not in candidates_by_id:
            raise ValueError("source_document_id is not a selectable source candidate for this slot.")

        slot["status"] = "ready_for_review"
        slot["candidate"] = copy.deepcopy(candidates_by_id[source_document_id])
        slot["candidate_documents"] = []
        slot["rejection"] = None
        slot["warnings"] = [
            warning
            for warning in slot.get("warnings", [])
            if warning.get("code")
            not in {
                "ambiguous_financial_statement_candidates",
                "ambiguous_reviewed_supersession_order",
            }
        ]

        self._bind_selected_source_package_snapshot(
            run_id,
            slot_role=slot_role,
            source_document_id=source_document_id,
        )
        source_confirmation["confirmable"] = all(
            item["status"] == "ready_for_review" for item in source_confirmation["slots"]
        )
        source_confirmation["status"] = (
            "ready_for_review" if source_confirmation["confirmable"] else "partially_rejected"
        )
        return self._registry.update_source_confirmation(
            run_id,
            updated_at=_runtime_timestamp(self._clock()),
            source_confirmation=source_confirmation,
        )

    def _bind_selected_source_package_snapshot(
        self,
        run_id: str,
        *,
        slot_role: str,
        source_document_id: str,
    ) -> None:
        audit_metadata = self._registry.get_run_audit_metadata(run_id)
        packages = copy.deepcopy(audit_metadata.get("live_extraction_source_packages") or {})
        package = packages.get(slot_role)
        if not isinstance(package, dict):
            return
        packages[slot_role] = _source_package_snapshot_for_selected_document(
            package,
            source_document_id,
        )
        self._registry.update_run_audit_metadata(
            run_id,
            {"live_extraction_source_packages": packages},
        )

    def stop_runtime_run(self, run_id: str) -> dict[str, Any]:
        view = self._registry.get_run_view(run_id)
        if view["status"] in {"failed", "completed", "cancelled"}:
            raise ValueError("Runtime run is already terminal.")
        stage_id = view["current_stage"]
        if stage_id is None:
            stage_id = "source_discovery"
        return self._registry.mark_cancelled(
            run_id,
            stage_id=stage_id,
            updated_at=_runtime_timestamp(self._clock()),
        )

    def advance_runtime_run(self, run_id: str) -> dict[str, Any]:
        return self._stage_spine.advance_run(run_id)

    def resume_runtime_run(self, run_id: str) -> dict[str, Any]:
        return self._stage_spine.resume_run(run_id)

    def schedule_auto_advance(
        self,
        run_id: str,
        view: dict[str, Any],
        background_tasks: BackgroundTasks,
    ) -> None:
        if self._auto_advance and view["status"] == "analyzing":
            background_tasks.add_task(self.advance_runtime_run_to_terminal, run_id)

    def advance_runtime_run_to_terminal(self, run_id: str, *, max_steps: int = 8) -> dict[str, Any]:
        view = self._registry.get_run_view(run_id)
        for _ in range(max_steps):
            if view["status"] != "analyzing" or view["current_stage"] is None:
                return view
            view = self.advance_runtime_run(run_id)
        return view

    def get_final_report(self, run_id: str) -> dict[str, Any]:
        return self._stage_spine.get_final_report_endpoint(run_id)

    def get_source_document_pdf(self, run_id: str, source_document_id: str) -> dict[str, Any]:
        view = self._registry.get_run_view(run_id)
        try:
            slot = _locked_source_slot_for_document(view["source_confirmation"], source_document_id)
            candidate = slot["candidate"]
            source_context = "confirmed_source_pdf"
        except SourceDocumentUnavailable:
            return self._get_source_confirmation_candidate_pdf(
                run_id,
                view,
                source_document_id,
            )
        self._stage_spine.ensure_confirmed_source_artifact_refs(run_id)
        source_ref = _source_artifact_ref_for_document(
            self._registry.get_artifact_refs(run_id),
            source_document_id=source_document_id,
            source_slot_role=slot["role"],
            allowed_kinds={"vietstock_source_pdf", "vietstock_source_package"},
        )
        expected_sha256 = candidate.get("audit_references", {}).get("source_document_fingerprint_sha256")
        if not isinstance(expected_sha256, str) or not expected_sha256:
            raise SourceDocumentUnavailable()
        if source_ref["kind"] == "vietstock_source_pdf":
            if source_ref.get("sha256") != expected_sha256:
                raise SourceDocumentUnavailable()
            try:
                body = self._stage_spine.artifact_store.read_bytes(source_ref)
            except OSError as exc:
                raise SourceDocumentUnavailable() from exc
            if hashlib.sha256(body).hexdigest() != expected_sha256 or not body.startswith(b"%PDF"):
                raise SourceDocumentUnavailable()
            headers = {
                "Content-Disposition": f'inline; filename="{source_document_id}.pdf"',
                "X-Vinacount-Source-Document-Id": source_document_id,
                "X-Vinacount-Source-Slot-Role": slot["role"],
                "X-Vinacount-Source-Artifact-Kind": source_ref["kind"],
                "X-Vinacount-Source-Document-Context": source_context,
            }
        else:
            try:
                package_body = self._stage_spine.artifact_store.read_bytes(source_ref)
            except OSError as exc:
                raise SourceDocumentUnavailable() from exc
            package_sha256 = hashlib.sha256(package_body).hexdigest()
            if package_sha256 != source_ref.get("sha256"):
                raise SourceDocumentUnavailable()
            selected_path = _selected_pdf_path_for_package_ref(source_ref)
            try:
                body = selected_path.read_bytes()
            except OSError as exc:
                raise SourceDocumentUnavailable() from exc
            if hashlib.sha256(body).hexdigest() != expected_sha256 or not body.startswith(b"%PDF"):
                raise SourceDocumentUnavailable()
            member_name = _package_member_name_for_sha256(source_ref, expected_sha256)
            headers = {
                "Content-Disposition": f'inline; filename="{source_document_id}.pdf"',
                "X-Vinacount-Source-Document-Id": source_document_id,
                "X-Vinacount-Source-Slot-Role": slot["role"],
                "X-Vinacount-Source-Artifact-Kind": source_ref["kind"],
                "X-Vinacount-Source-Document-Context": source_context,
                "X-Vinacount-Source-Package-Sha256": package_sha256,
                "X-Vinacount-Display-Document-Sha256": expected_sha256,
                "X-Vinacount-Source-Package-Member": member_name,
            }
        return {
            "body": body,
            "headers": headers,
        }

    def _get_source_confirmation_candidate_pdf(
        self,
        run_id: str,
        view: dict[str, Any],
        source_document_id: str,
    ) -> dict[str, Any]:
        slot, candidate = _source_confirmation_candidate_slot_for_document(
            view["source_confirmation"],
            source_document_id,
        )
        expected_sha256 = candidate.get("audit_references", {}).get("source_document_fingerprint_sha256")
        if not isinstance(expected_sha256, str) or not expected_sha256:
            raise SourceDocumentUnavailable()
        audit_metadata = self._registry.get_run_audit_metadata(run_id)
        packages = audit_metadata.get("live_extraction_source_packages") or {}
        package = packages.get(slot["role"])
        if not isinstance(package, dict):
            raise SourceDocumentUnavailable()
        source_document = _source_document_snapshot_by_id(package, source_document_id)
        try:
            artifact_path = Path(source_document["local_artifact"]["path"])
            body = artifact_path.read_bytes()
        except (KeyError, TypeError, OSError) as exc:
            raise SourceDocumentUnavailable() from exc
        if hashlib.sha256(body).hexdigest() != expected_sha256 or not body.startswith(b"%PDF"):
            raise SourceDocumentUnavailable()
        return {
            "body": body,
            "headers": {
                "Content-Disposition": f'inline; filename="{source_document_id}.pdf"',
                "X-Vinacount-Source-Document-Id": source_document_id,
                "X-Vinacount-Source-Slot-Role": slot["role"],
                "X-Vinacount-Source-Artifact-Kind": "source_confirmation_candidate_pdf",
                "X-Vinacount-Source-Document-Context": "source_confirmation_candidate_preview",
            },
        }

    def get_developer_audit_bundle(self, run_id: str) -> dict[str, Any]:
        return self._audit_bundle_builder.build(run_id)

    def _capture_internal_live_extraction_source_packages(
        self,
        run_id: str,
        source_confirmation: dict[str, Any],
    ) -> None:
        internal_source_packages = source_confirmation.pop("_internal_live_extraction_source_packages", None)
        if isinstance(internal_source_packages, dict):
            self._registry.update_run_audit_metadata(
                run_id,
                {"live_extraction_source_packages": internal_source_packages},
            )


class SourceDocumentUnavailable(Exception):
    def __init__(
        self,
        *,
        code: str = "source_document_unavailable",
        message: str = "Source document is unavailable for display.",
        status_code: int = 404,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class SourceDiscoveryAdapter(Protocol):
    def discover(self, filing_intent: dict[str, Any]) -> dict[str, Any]:
        ...


class DemoSourceDiscoveryAdapter:
    def discover(self, filing_intent: dict[str, Any]) -> dict[str, Any]:
        target_year = filing_intent["target_fiscal_year"]
        quarter = filing_intent["target_quarter"]
        basis = filing_intent["report_basis_preference"]
        company_identifier = filing_intent["company_identifier"].upper()
        company_name = _company_name_vi(company_identifier)
        return {
            "status": "ready_for_review",
            "confirmable": True,
            "slots": [
                {
                    "role": "target",
                    "status": "ready_for_review",
                    "candidate": _source_candidate(
                        company_identifier=company_identifier,
                        company_name_vi=company_name,
                        fiscal_year=target_year,
                        quarter=quarter,
                        report_basis=basis,
                    ),
                    "rejection": None,
                    "warnings": [],
                },
                {
                    "role": "prior_year_same_quarter",
                    "status": "ready_for_review",
                    "candidate": _source_candidate(
                        company_identifier=company_identifier,
                        company_name_vi=company_name,
                        fiscal_year=target_year - 1,
                        quarter=quarter,
                        report_basis=basis,
                    ),
                    "rejection": None,
                    "warnings": [],
                },
            ],
            "package_warnings": [],
        }


def _default_run_id() -> str:
    return f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _report_artifact_cache_db_path(artifact_store: FilesystemArtifactBodyStore) -> Path:
    return Path(
        os.environ.get(
            "VINACOUNT_REPORT_ARTIFACT_CACHE_DB_PATH",
            str(artifact_store.root / "report_artifact_cache.sqlite3"),
        )
    )


def _runtime_run_registry_from_environment(artifact_store: FilesystemArtifactBodyStore) -> RuntimeRunRegistry:
    database_path = Path(
        os.environ.get(
            "VINACOUNT_RUNTIME_RUN_REGISTRY_DB_PATH",
            str(artifact_store.root / "runtime_run_registry.sqlite3"),
        )
    )
    return RuntimeRunRegistry(store=SqliteRuntimeRunStore(database_path))


def _report_artifact_cache_repository(
    artifact_store: FilesystemArtifactBodyStore,
) -> ReportArtifactCacheRepository:
    postgres_dsn = os.environ.get("VINACOUNT_REPORT_ARTIFACT_CACHE_POSTGRES_DSN")
    if postgres_dsn:
        return PostgresReportArtifactCacheRepository(postgres_dsn)
    return SqliteReportArtifactCacheRepository(_report_artifact_cache_db_path(artifact_store))


def _runtime_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _runtime_auto_advance_enabled() -> bool:
    return os.environ.get("VINACOUNT_RUNTIME_AUTO_ADVANCE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _runtime_locked_thesis_demo_enabled() -> bool:
    scenario_id = os.environ.get("VINACOUNT_RUNTIME_THESIS_DEMO_SCENARIO")
    cache_mode = os.environ.get("VINACOUNT_RUNTIME_REAL_CACHE_MODE")
    if not scenario_id and not cache_mode:
        return False
    try:
        from vinacount.runtime_thesis_demo import locked_thesis_demo_enabled
    except ModuleNotFoundError as error:
        if error.name == "vinacount.runtime_thesis_demo":
            return False
        raise
    return locked_thesis_demo_enabled(scenario_id=scenario_id, cache_mode=cache_mode)


def _thesis_demo_components() -> tuple[type[Any], type[Any], type[Any]]:
    try:
        from vinacount.runtime_thesis_demo import (
            CachedFirstLiveConfirmationCacheLookupAdapter,
            LockedThesisDemoCacheLookupAdapter,
            LockedThesisDemoSourceDiscoveryAdapter,
        )
    except ModuleNotFoundError as error:
        if error.name == "vinacount.runtime_thesis_demo":
            raise RuntimeError("Thesis demo adapters are not included in this package.") from error
        raise
    return (
        CachedFirstLiveConfirmationCacheLookupAdapter,
        LockedThesisDemoCacheLookupAdapter,
        LockedThesisDemoSourceDiscoveryAdapter,
    )


def _runtime_cached_first_live_confirmation_enabled() -> bool:
    return (
        os.environ.get("VINACOUNT_RUNTIME_PRESENTATION_MODE", "").strip().lower()
        == "cached_first_live_confirmation"
    )


def _raw_ocr_identity_for_candidate(candidate: dict[str, Any]) -> RawOcrCacheIdentity:
    return RawOcrCacheIdentity.from_non_secret_config(
        canonical_source_sha256=candidate["audit_references"]["source_document_fingerprint_sha256"],
        provider="nanonets_ocr_3_docstrange",
        model=os.environ.get("NANONETS_OCR_MODEL", "ocr-3-docstrange"),
        extraction_schema_version=RAW_EXTRACTION_ARTIFACT_SCHEMA_VERSION,
        extraction_version="v1",
        non_secret_config={
            "extraction_method": "nanonets_ocr_3_docstrange_html",
            "output_format": "html",
            "include_metadata": ["bounding_boxes", "confidence_score"],
        },
    )


def _runtime_cors_origins() -> list[str]:
    configured = os.environ.get("VINACOUNT_RUNTIME_CORS_ORIGINS")
    if configured:
        return [origin.strip() for origin in configured.split(",") if origin.strip()]
    return [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3024",
        "http://127.0.0.1:3024",
    ]


def _source_candidate(
    *,
    company_identifier: str,
    company_name_vi: str,
    fiscal_year: int,
    quarter: int,
    report_basis: str,
) -> dict[str, Any]:
    basis_clue = "hợp nhất" if report_basis == "consolidated" else "riêng lẻ"
    source_document_id = (
        f"src_{company_identifier.lower()}_{fiscal_year}_q{quarter}_{report_basis}_vi"
    )
    source_document_fingerprint = hashlib.sha256(
        f"{source_document_id}:demo-pdf-placeholder".encode("utf-8")
    ).hexdigest()
    return {
        "source_document_id": source_document_id,
        "company_name_vi": company_name_vi,
        "ticker": company_identifier,
        "period_label": f"Q{quarter} {fiscal_year}",
        "quarter": quarter,
        "fiscal_year": fiscal_year,
        "report_basis": report_basis,
        "filing_status": "reviewed",
        "document_type": "quarterly_bctc",
        "language": "vi",
        "source_origin": "LocalDemoSourceDiscovery",
        "source_name": "Đăng ký nguồn demo Vinacount",
        "source_url": f"https://example.invalid/vinacount/{company_identifier}/{fiscal_year}/q{quarter}/{report_basis}.pdf",
        "is_searchable_version": True,
        "file_size_bytes": 4219320,
        "page_count": 84,
        "visible_filing_label": f"Báo cáo tài chính {basis_clue} quý {quarter} năm {fiscal_year}",
        "first_page_identity": {
            "visible_company_name": company_name_vi,
            "visible_period": f"Quý {quarter} năm {fiscal_year}",
            "visible_basis_clue": basis_clue,
        },
        "classification_evidence": [
            f"Trang đầu hiển thị báo cáo tài chính {basis_clue} theo quý.",
            f"Tên tệp chứa BCTC Q{quarter} {fiscal_year}.",
        ],
        "audit_references": {
            "package_id": f"pkg_{company_identifier.lower()}_{fiscal_year}_q{quarter}_{report_basis}",
            "event_id": f"demo_event_{company_identifier.lower()}_{fiscal_year}_q{quarter}",
            "canonical_source_document_id": source_document_id,
            "source_document_fingerprint_sha256": source_document_fingerprint,
        },
    }


def _company_name_vi(company_identifier: str) -> str:
    known_names = {
        "VIC": "Tap doan Vingroup - CTCP",
        "VCF": "Cong ty Co phan San xuat Vinacount",
    }
    return known_names.get(company_identifier, f"Cong ty {company_identifier}")


def _source_slot_role(payload: dict[str, Any]) -> str:
    role = payload.get("slot_role")
    if role not in SOURCE_SLOT_ROLES:
        raise ValueError("slot_role must be target or prior_year_same_quarter.")
    return role


def _rejection_reason_code(payload: dict[str, Any]) -> str:
    reason_code = payload.get("reason_code")
    if reason_code not in REJECTION_REASON_CODES:
        raise ValueError("reason_code is not a valid source rejection reason.")
    return reason_code


def _slot_by_role(source_confirmation: dict[str, Any], role: str) -> dict[str, Any]:
    for slot in source_confirmation["slots"]:
        if slot["role"] == role:
            return slot
    raise ValueError(f"Source confirmation slot does not exist: {role}")


def _source_package_snapshot_for_selected_document(
    package: dict[str, Any],
    source_document_id: str,
) -> dict[str, Any]:
    selected_documents = []
    found = False
    for document in package.get("source_documents", []):
        if not isinstance(document, dict):
            continue
        if document.get("document_id") == source_document_id:
            selected_documents.append(copy.deepcopy(document))
            found = True
        elif document.get("document_type") not in {
            "main_financial_statement",
            "amended_or_replacement_financial_statement",
        }:
            selected_documents.append(copy.deepcopy(document))
    if not found:
        raise ValueError("source_document_id is not present in the source package snapshot.")
    narrowed = copy.deepcopy(package)
    narrowed["source_documents"] = selected_documents
    return narrowed


def _locked_source_slot_for_document(
    source_confirmation: dict[str, Any],
    source_document_id: str,
) -> dict[str, Any]:
    if source_confirmation.get("status") != "confirmed":
        raise SourceDocumentUnavailable()
    for slot in source_confirmation.get("slots", []):
        candidate = slot.get("candidate")
        if (
            slot.get("status") == "locked"
            and isinstance(candidate, dict)
            and candidate.get("source_document_id") == source_document_id
        ):
            return slot
    raise SourceDocumentUnavailable()


def _source_confirmation_candidate_slot_for_document(
    source_confirmation: dict[str, Any],
    source_document_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if source_confirmation.get("status") not in {"ready_for_review", "partially_rejected", "retrying"}:
        raise SourceDocumentUnavailable()
    for slot in source_confirmation.get("slots", []):
        for candidate in slot.get("candidate_documents", []) or []:
            if (
                isinstance(candidate, dict)
                and candidate.get("source_document_id") == source_document_id
            ):
                return slot, candidate
    raise SourceDocumentUnavailable()


def _source_document_snapshot_by_id(
    package: dict[str, Any],
    source_document_id: str,
) -> dict[str, Any]:
    for document in package.get("source_documents", []):
        if isinstance(document, dict) and document.get("document_id") == source_document_id:
            return document
    raise SourceDocumentUnavailable()


def _source_artifact_ref_for_document(
    artifact_refs: list[dict[str, Any]],
    *,
    source_document_id: str,
    source_slot_role: str,
    allowed_kinds: set[str],
) -> dict[str, Any]:
    for ref in artifact_refs:
        metadata = ref.get("metadata", {})
        if (
            ref.get("kind") in allowed_kinds
            and metadata.get("source_document_id") == source_document_id
            and metadata.get("report_role") == source_slot_role
        ):
            return ref
    raise SourceDocumentUnavailable()


def _selected_pdf_path_for_package_ref(source_ref: dict[str, Any]) -> Path:
    package_path = Path(source_ref["path"])
    if package_path.suffix.lower() != ".zip":
        raise SourceDocumentUnavailable()
    return package_path.with_name(f"{package_path.stem}__selected.pdf")


def _package_member_name_for_sha256(source_ref: dict[str, Any], expected_sha256: str) -> str:
    try:
        with zipfile.ZipFile(source_ref["path"]) as package:
            for member in package.infolist():
                if member.is_dir() or not member.filename.lower().endswith(".pdf"):
                    continue
                with package.open(member) as handle:
                    if hashlib.sha256(handle.read()).hexdigest() == expected_sha256:
                        return Path(member.filename).name
    except (OSError, zipfile.BadZipFile) as exc:
        raise SourceDocumentUnavailable() from exc
    raise SourceDocumentUnavailable()


def _rejection_message(reason_code: str) -> str:
    messages = {
        "wrong_company": "The visible source company does not match the requested issuer.",
        "wrong_period": "The visible source period does not match the requested filing period.",
        "wrong_basis": "The visible source basis does not match the requested report basis.",
        "wrong_filing_status": "The visible filing status does not match the expected source package.",
        "wrong_language": "The source language is not valid for the normal demo path.",
        "not_full_financial_statement": "The source is not a full quarterly financial statement.",
        "source_unreadable": "The source is not readable enough for confirmation.",
        "other": "The selected source was rejected by the user.",
    }
    return messages[reason_code]


def _filing_intent_error_response(payload: Any, message: str) -> dict[str, Any]:
    response = {
        "error": {
            "code": "filing_intent_invalid",
            "message": _validation_summary(message),
            "field_errors": _field_errors(payload, message),
        }
    }
    return validate_filing_intent_error_response(response)


def _validation_summary(message: str) -> str:
    if "target_quarter" in message:
        return "Quarter must be between 1 and 4."
    if "report_synthesis_model_id" in message:
        return "Unknown report synthesis model."
    if "report_language" in message:
        return "Unknown report language."
    return "Filing Intent is invalid."


def _field_errors(payload: Any, message: str) -> dict[str, str]:
    if not isinstance(payload, dict):
        return {"request": "Request body must be an object."}

    field_errors: dict[str, str] = {}
    allowed_fields = {
        "company_identifier",
        "target_fiscal_year",
        "target_quarter",
        "report_basis_preference",
        "report_synthesis_model_id",
        "report_language",
    }
    required_fields = {
        "company_identifier",
        "target_fiscal_year",
        "target_quarter",
        "report_basis_preference",
    }
    for field in sorted(set(payload) - allowed_fields):
        field_errors[field] = "Unknown field."
    for field in sorted(required_fields - set(payload)):
        field_errors[field] = "Required."

    if "company_identifier" in message:
        field_errors["company_identifier"] = "Must be a non-empty string."
    if "target_fiscal_year" in message:
        field_errors["target_fiscal_year"] = "Must be an integer."
    if "target_quarter" in message:
        field_errors["target_quarter"] = "Must be 1, 2, 3, or 4."
    if "report_basis_preference" in message:
        basis = " or ".join(sorted(REPORT_BASIS_PREFERENCES))
        field_errors["report_basis_preference"] = f"Must be {basis}."
    if "report_synthesis_model_id" in message:
        models = ", ".join(sorted(REPORT_SYNTHESIS_MODEL_IDS))
        field_errors["report_synthesis_model_id"] = f"Must be one of: {models}."
    if "report_language" in message:
        field_errors["report_language"] = "Must be en or vi."

    if not field_errors:
        field_errors["request"] = message or "Invalid request."
    return field_errors
