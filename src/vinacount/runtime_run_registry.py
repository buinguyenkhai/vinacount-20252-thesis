from __future__ import annotations

import copy
import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from vinacount.runtime_contract import (
    HITL_BOUNDARY,
    RUNTIME_RUN_VIEW_SCHEMA_VERSION,
    STAGE_IDS,
    runtime_config_for_report_synthesis_model,
    validate_runtime_run_view,
)
from vinacount.runtime_detector_modes import default_runtime_detector_mode_selection


STAGE_SUMMARIES = {
    "source_discovery": "Discovering target and prior-year source candidates",
    "source_confirmation": "Waiting for source identity confirmation",
    "cache_lookup": "Checking reusable ReportMemory artifacts",
    "extraction": "Building ReportMemory from confirmed sources",
    "tool_analysis": "Running deterministic accounting checks",
    "detector_assessment": "Assessing evidence packets",
    "aggregation": "Aggregating supported risk signals",
    "report_generation": "Generating the final report endpoint payload",
}

_ARTIFACT_ID_SEGMENT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
SQLITE_LOCAL_CONNECTION_TIMEOUT_SECONDS = 5.0


@dataclass
class RuntimeRunRecord:
    run_id: str
    created_at: str
    updated_at: str
    status: str
    filing_intent: dict[str, Any]
    runtime_config: dict[str, Any]
    source_confirmation: dict[str, Any]
    stages: list[dict[str, Any]]
    warnings: list[dict[str, Any]] = field(default_factory=list)
    error: dict[str, Any] | None = None
    final_report: dict[str, Any] = field(default_factory=dict)
    allowed_action_state: dict[str, Any] = field(default_factory=dict)
    artifact_refs: list[dict[str, Any]] = field(default_factory=list)
    audit_metadata: dict[str, Any] = field(default_factory=dict)


class RuntimeRunStore(Protocol):
    def insert_run(self, record: RuntimeRunRecord) -> None:
        ...

    def get_run(self, run_id: str) -> RuntimeRunRecord:
        ...

    def update_run(self, record: RuntimeRunRecord) -> None:
        ...


class InMemoryRuntimeRunStore:
    """Postgres-shaped test adapter: records are keyed by stable run id."""

    def __init__(self) -> None:
        self._records: dict[str, RuntimeRunRecord] = {}

    def insert_run(self, record: RuntimeRunRecord) -> None:
        if record.run_id in self._records:
            raise ValueError(f"RuntimeRun already exists: {record.run_id}")
        self._records[record.run_id] = copy.deepcopy(record)

    def get_run(self, run_id: str) -> RuntimeRunRecord:
        try:
            return copy.deepcopy(self._records[run_id])
        except KeyError as exc:
            raise KeyError(f"RuntimeRun does not exist: {run_id}") from exc

    def update_run(self, record: RuntimeRunRecord) -> None:
        if record.run_id not in self._records:
            raise KeyError(f"RuntimeRun does not exist: {record.run_id}")
        self._records[record.run_id] = copy.deepcopy(record)


class SqliteRuntimeRunStore:
    """Durable local/dev adapter for RuntimeRun metadata."""

    def __init__(self, database_path: Path | str) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def insert_run(self, record: RuntimeRunRecord) -> None:
        payload = _record_to_row(record)
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO runtime_runs (
                        run_id, created_at, updated_at, status, filing_intent,
                        runtime_config, source_confirmation, stages, warnings,
                        error, final_report, allowed_action_state, artifact_refs,
                        audit_metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    payload,
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"RuntimeRun already exists: {record.run_id}") from exc

    def get_run(self, run_id: str) -> RuntimeRunRecord:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT run_id, created_at, updated_at, status, filing_intent,
                       runtime_config, source_confirmation, stages, warnings,
                       error, final_report, allowed_action_state, artifact_refs,
                       audit_metadata
                FROM runtime_runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"RuntimeRun does not exist: {run_id}")
        return _record_from_row(row)

    def update_run(self, record: RuntimeRunRecord) -> None:
        payload = _record_to_row(record)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE runtime_runs
                SET created_at = ?, updated_at = ?, status = ?,
                    filing_intent = ?, runtime_config = ?,
                    source_confirmation = ?, stages = ?, warnings = ?,
                    error = ?, final_report = ?, allowed_action_state = ?,
                    artifact_refs = ?, audit_metadata = ?
                WHERE run_id = ?
                """,
                (
                    payload[1],
                    payload[2],
                    payload[3],
                    payload[4],
                    payload[5],
                    payload[6],
                    payload[7],
                    payload[8],
                    payload[9],
                    payload[10],
                    payload[11],
                    payload[12],
                    payload[13],
                    payload[0],
                ),
            )
        if cursor.rowcount == 0:
            raise KeyError(f"RuntimeRun does not exist: {record.run_id}")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=SQLITE_LOCAL_CONNECTION_TIMEOUT_SECONDS,
        )
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS runtime_runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    filing_intent TEXT NOT NULL,
                    runtime_config TEXT NOT NULL,
                    source_confirmation TEXT NOT NULL,
                    stages TEXT NOT NULL,
                    warnings TEXT NOT NULL,
                    error TEXT,
                    final_report TEXT NOT NULL,
                    allowed_action_state TEXT NOT NULL,
                    artifact_refs TEXT NOT NULL,
                    audit_metadata TEXT NOT NULL
                )
                """
            )


class RuntimeRunRegistry:
    def __init__(self, store: RuntimeRunStore | None = None) -> None:
        self._store = store or InMemoryRuntimeRunStore()

    def create_run(
        self,
        *,
        run_id: str,
        created_at: str,
        filing_intent: dict[str, Any],
        report_synthesis_model_id: str | None = None,
        runtime_detector_mode: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        detector_mode = runtime_detector_mode or default_runtime_detector_mode_selection().audit_metadata()
        record = RuntimeRunRecord(
            run_id=run_id,
            created_at=created_at,
            updated_at=created_at,
            status="created",
            filing_intent=_filing_intent_view(filing_intent),
            runtime_config=runtime_config_for_report_synthesis_model(report_synthesis_model_id),
            source_confirmation=_default_source_confirmation(),
            stages=_default_stage_registry(),
            final_report=_empty_final_report(),
            allowed_action_state={"can_stop": True},
            audit_metadata={
                "runtime_detector_mode": copy.deepcopy(detector_mode),
            },
        )
        self._store.insert_run(record)
        return self.get_run_view(run_id)

    def get_run_view(self, run_id: str) -> dict[str, Any]:
        return _runtime_run_view(self._store.get_run(run_id))

    def get_run_audit_metadata(self, run_id: str) -> dict[str, Any]:
        return copy.deepcopy(self._store.get_run(run_id).audit_metadata)

    def update_run_audit_metadata(self, run_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
        record = self._store.get_run(run_id)
        record.audit_metadata.update(copy.deepcopy(metadata))
        self._store.update_run(record)
        return self.get_run_audit_metadata(run_id)

    def add_artifact_ref(self, run_id: str, artifact_ref: dict[str, Any]) -> dict[str, Any]:
        record = self._store.get_run(run_id)
        if artifact_ref.get("run_id") != run_id:
            raise ValueError("Artifact reference run_id must match RuntimeRun")
        ref = copy.deepcopy(artifact_ref)
        if any(existing["artifact_id"] == ref["artifact_id"] for existing in record.artifact_refs):
            raise ValueError(f"Artifact reference already exists: {ref['artifact_id']}")
        record.artifact_refs.append(ref)
        self._store.update_run(record)
        return copy.deepcopy(ref)

    def get_artifact_refs(self, run_id: str) -> list[dict[str, Any]]:
        return copy.deepcopy(self._store.get_run(run_id).artifact_refs)

    def add_warning(self, run_id: str, warning: dict[str, Any]) -> dict[str, Any]:
        record = self._store.get_run(run_id)
        record.warnings.append(copy.deepcopy(warning))
        self._store.update_run(record)
        return self.get_run_view(run_id)

    def update_source_confirmation(
        self,
        run_id: str,
        *,
        updated_at: str,
        source_confirmation: dict[str, Any],
    ) -> dict[str, Any]:
        record = self._store.get_run(run_id)
        payload = copy.deepcopy(source_confirmation)
        payload["hitl_boundary"] = payload.get("hitl_boundary", HITL_BOUNDARY)
        record.source_confirmation = payload
        record.updated_at = updated_at
        if payload["status"] in {"ready_for_review", "partially_rejected", "retrying"}:
            record.status = "awaiting_source_confirmation"
        self._store.update_run(record)
        return self.get_run_view(run_id)

    def require_source_reconfirmation(
        self,
        run_id: str,
        *,
        updated_at: str,
        source_confirmation: dict[str, Any],
        cache_lookup_warnings: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        record = self._store.get_run(run_id)
        payload = copy.deepcopy(source_confirmation)
        payload["hitl_boundary"] = payload.get("hitl_boundary", HITL_BOUNDARY)
        record.source_confirmation = payload
        record.status = "awaiting_source_confirmation"
        record.updated_at = updated_at
        record.error = None
        record.allowed_action_state["can_stop"] = True
        record.allowed_action_state["can_resume"] = False

        source_discovery = _stage_by_id(record, "source_discovery")
        source_discovery["status"] = "completed"
        if source_discovery["started_at"] is None:
            source_discovery["started_at"] = record.created_at
        if source_discovery["completed_at"] is None:
            source_discovery["completed_at"] = updated_at

        source_confirmation_stage = _stage_by_id(record, "source_confirmation")
        source_confirmation_stage["status"] = "active"
        if source_confirmation_stage["started_at"] is None:
            source_confirmation_stage["started_at"] = updated_at
        source_confirmation_stage["completed_at"] = None

        cache_lookup = _stage_by_id(record, "cache_lookup")
        cache_lookup["status"] = "pending"
        cache_lookup["started_at"] = None
        cache_lookup["completed_at"] = None
        cache_lookup["warnings"] = copy.deepcopy(cache_lookup_warnings or [])
        cache_lookup["counts"] = {}

        for stage in record.stages:
            if stage["stage_id"] in {
                "extraction",
                "tool_analysis",
                "detector_assessment",
                "aggregation",
                "report_generation",
            }:
                stage["status"] = "pending"
                stage["started_at"] = None
                stage["completed_at"] = None
                stage["warnings"] = []
                stage["counts"] = {}

        self._store.update_run(record)
        return self.get_run_view(run_id)

    def mark_failed(
        self,
        run_id: str,
        *,
        stage_id: str,
        updated_at: str,
        error: dict[str, Any],
        completed_stage_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        if stage_id not in STAGE_IDS:
            raise ValueError(f"Unknown runtime stage: {stage_id}")
        if error.get("stage_id") != stage_id:
            raise ValueError("Runtime error stage_id must match failed stage")
        record = self._store.get_run(run_id)
        record.status = "failed"
        record.updated_at = updated_at
        record.error = copy.deepcopy(error)

        for stage in record.stages:
            if stage["status"] == "active":
                stage["status"] = "completed"
                stage["completed_at"] = updated_at
        for completed_stage_id in completed_stage_ids or []:
            stage = _stage_by_id(record, completed_stage_id)
            stage["status"] = "completed"
            if stage["started_at"] is None:
                stage["started_at"] = record.created_at
            stage["completed_at"] = updated_at

        failed_stage = _stage_by_id(record, stage_id)
        failed_stage["status"] = "failed"
        if failed_stage["started_at"] is None:
            failed_stage["started_at"] = updated_at
        failed_stage["completed_at"] = updated_at
        record.allowed_action_state["can_stop"] = False
        record.allowed_action_state["can_resume"] = bool(error.get("can_resume"))

        self._store.update_run(record)
        return self.get_run_view(run_id)

    def resume_from_latest_completed_stage(
        self,
        run_id: str,
        *,
        updated_at: str,
    ) -> dict[str, Any]:
        record = self._store.get_run(run_id)
        if record.status != "failed" or record.error is None:
            raise ValueError("Only failed Runtime Analysis Runs can be resumed.")
        if not record.error.get("can_resume"):
            raise ValueError("Runtime Analysis Run is not resumable.")

        latest_boundary_index = -1
        for index, stage in enumerate(record.stages):
            if stage["status"] in {"completed", "skipped"}:
                latest_boundary_index = index

        next_stage = None
        for stage in record.stages[latest_boundary_index + 1 :]:
            if stage["status"] not in {"completed", "skipped"}:
                next_stage = stage
                break
        if next_stage is None:
            raise ValueError("No resumable stage is available.")

        record.status = _status_for_active_stage(next_stage["stage_id"])
        record.updated_at = updated_at
        record.error = None
        record.allowed_action_state["can_resume"] = False
        record.allowed_action_state["can_stop"] = True
        for stage in record.stages:
            if stage["status"] == "active":
                stage["status"] = "completed"
                stage["completed_at"] = updated_at
        next_stage["status"] = "active"
        if next_stage["started_at"] is None:
            next_stage["started_at"] = updated_at
        next_stage["completed_at"] = None

        self._store.update_run(record)
        return self.get_run_view(run_id)

    def mark_completed(
        self,
        run_id: str,
        *,
        updated_at: str,
        report_id: str,
        generated_at: str,
        report_format: str = "json+markdown",
        report_generation_counts: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        record = self._store.get_run(run_id)
        record.status = "completed"
        record.updated_at = updated_at
        record.error = None
        for stage in record.stages:
            if stage["status"] not in {"skipped", "completed"}:
                stage["status"] = "completed"
            if stage["started_at"] is None:
                stage["started_at"] = record.created_at
            if stage["completed_at"] is None:
                stage["completed_at"] = updated_at
        report_generation = _stage_by_id(record, "report_generation")
        report_generation["status"] = "completed"
        report_generation["completed_at"] = updated_at
        if report_generation_counts is not None:
            report_generation["counts"] = dict(report_generation_counts)
        record.final_report = {
            "available": True,
            "report_id": report_id,
            "generated_at": generated_at,
            "format": report_format,
            "href": f"/runtime-runs/{record.run_id}/report",
        }
        record.allowed_action_state["can_stop"] = False
        self._store.update_run(record)
        return self.get_run_view(run_id)

    def mark_cancelled(
        self,
        run_id: str,
        *,
        stage_id: str,
        updated_at: str,
    ) -> dict[str, Any]:
        if stage_id not in STAGE_IDS:
            raise ValueError(f"Unknown runtime stage: {stage_id}")
        record = self._store.get_run(run_id)
        record.status = "cancelled"
        record.updated_at = updated_at
        record.error = None
        record.final_report = _empty_final_report()
        record.source_confirmation["status"] = "stopped"
        record.source_confirmation["confirmable"] = False

        for stage in record.stages:
            if stage["status"] == "active":
                stage["status"] = "completed"
                stage["completed_at"] = updated_at
        cancelled_stage = _stage_by_id(record, stage_id)
        cancelled_stage["status"] = "cancelled"
        if cancelled_stage["started_at"] is None:
            cancelled_stage["started_at"] = updated_at
        cancelled_stage["completed_at"] = updated_at
        record.allowed_action_state["can_stop"] = False
        self._store.update_run(record)
        return self.get_run_view(run_id)

    def update_stage(
        self,
        run_id: str,
        stage_id: str,
        *,
        status: str,
        updated_at: str,
        started_at: str | None = None,
        completed_at: str | None = None,
        summary: str | None = None,
        progress: dict[str, int] | None = None,
        counts: dict[str, int] | None = None,
        warnings: list[dict[str, Any]] | None = None,
        completed_stage_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        if stage_id not in STAGE_IDS:
            raise ValueError(f"Unknown runtime stage: {stage_id}")
        record = self._store.get_run(run_id)
        record.updated_at = updated_at

        for completed_stage_id in completed_stage_ids or []:
            stage = _stage_by_id(record, completed_stage_id)
            stage["status"] = "completed"
            if stage["started_at"] is None:
                stage["started_at"] = started_at or updated_at
            stage["completed_at"] = completed_at or updated_at

        target = _stage_by_id(record, stage_id)
        if status == "active":
            for stage in record.stages:
                if stage["status"] == "active":
                    stage["status"] = "completed"
                    stage["completed_at"] = completed_at or updated_at
            target["completed_at"] = None
            record.status = _status_for_active_stage(stage_id)
            if stage_id == "source_confirmation" and record.source_confirmation["status"] == "not_started":
                record.source_confirmation["status"] = "retrying"
        target["status"] = status
        if started_at is not None:
            target["started_at"] = started_at
        if completed_at is not None:
            target["completed_at"] = completed_at
        if summary is not None:
            target["summary"] = summary
        if progress is not None:
            target["progress"] = dict(progress)
        if counts is not None:
            target["counts"] = dict(counts)
        if warnings is not None:
            target["warnings"] = copy.deepcopy(warnings)

        self._store.update_run(record)
        return self.get_run_view(run_id)

    def complete_stage_and_activate(
        self,
        run_id: str,
        *,
        completed_stage_id: str,
        next_stage_id: str,
        updated_at: str,
        completed_at: str | None = None,
        completed_progress: dict[str, int] | None = None,
        completed_counts: dict[str, int] | None = None,
        completed_warnings: list[dict[str, Any]] | None = None,
        next_started_at: str | None = None,
        next_summary: str | None = None,
        next_progress: dict[str, int] | None = None,
        next_counts: dict[str, int] | None = None,
        next_warnings: list[dict[str, Any]] | None = None,
        skipped_stage_updates: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if completed_stage_id not in STAGE_IDS:
            raise ValueError(f"Unknown runtime stage: {completed_stage_id}")
        if next_stage_id not in STAGE_IDS:
            raise ValueError(f"Unknown runtime stage: {next_stage_id}")
        record = self._store.get_run(run_id)
        record.updated_at = updated_at

        completed_stage = _stage_by_id(record, completed_stage_id)
        completed_stage["status"] = "completed"
        if completed_stage["started_at"] is None:
            completed_stage["started_at"] = updated_at
        completed_stage["completed_at"] = completed_at or updated_at
        if completed_progress is not None:
            completed_stage["progress"] = dict(completed_progress)
        if completed_counts is not None:
            completed_stage["counts"] = dict(completed_counts)
        if completed_warnings is not None:
            completed_stage["warnings"] = copy.deepcopy(completed_warnings)

        for stage in record.stages:
            if stage["status"] == "active" and stage["stage_id"] != completed_stage_id:
                stage["status"] = "completed"
                stage["completed_at"] = completed_at or updated_at

        for skipped_update in skipped_stage_updates or []:
            skipped_stage_id = skipped_update["stage_id"]
            if skipped_stage_id not in STAGE_IDS:
                raise ValueError(f"Unknown runtime stage: {skipped_stage_id}")
            skipped_stage = _stage_by_id(record, skipped_stage_id)
            skipped_stage["status"] = "skipped"
            skipped_stage["started_at"] = skipped_update.get("started_at") or updated_at
            skipped_stage["completed_at"] = skipped_update.get("completed_at") or updated_at
            if "summary" in skipped_update:
                skipped_stage["summary"] = skipped_update["summary"]
            if "progress" in skipped_update:
                skipped_stage["progress"] = dict(skipped_update["progress"])
            if "counts" in skipped_update:
                skipped_stage["counts"] = dict(skipped_update["counts"])
            if "warnings" in skipped_update:
                skipped_stage["warnings"] = copy.deepcopy(skipped_update["warnings"])

        next_stage = _stage_by_id(record, next_stage_id)
        next_stage["status"] = "active"
        next_stage["started_at"] = next_started_at or updated_at
        next_stage["completed_at"] = None
        if next_summary is not None:
            next_stage["summary"] = next_summary
        if next_progress is not None:
            next_stage["progress"] = dict(next_progress)
        if next_counts is not None:
            next_stage["counts"] = dict(next_counts)
        if next_warnings is not None:
            next_stage["warnings"] = copy.deepcopy(next_warnings)

        record.status = _status_for_active_stage(next_stage_id)
        self._store.update_run(record)
        return self.get_run_view(run_id)


def _filing_intent_view(filing_intent: dict[str, Any]) -> dict[str, Any]:
    return {
        "company_identifier": filing_intent["company_identifier"],
        "company_name_vi": filing_intent.get("company_name_vi"),
        "target_fiscal_year": filing_intent["target_fiscal_year"],
        "target_quarter": filing_intent["target_quarter"],
        "report_basis_preference": filing_intent["report_basis_preference"],
        "report_language": filing_intent.get("report_language", "vi"),
    }


def _default_source_confirmation() -> dict[str, Any]:
    return {
        "status": "not_started",
        "confirmable": False,
        "hitl_boundary": HITL_BOUNDARY,
        "slots": [
            _source_confirmation_slot("target"),
            _source_confirmation_slot("prior_year_same_quarter"),
        ],
        "package_warnings": [],
    }


def _source_confirmation_slot(role: str) -> dict[str, Any]:
    return {
        "role": role,
        "status": "pending_discovery",
        "candidate": None,
        "rejection": None,
        "warnings": [],
    }


def _default_stage_registry() -> list[dict[str, Any]]:
    return [
        {
            "stage_id": stage_id,
            "status": "pending",
            "started_at": None,
            "completed_at": None,
            "summary": STAGE_SUMMARIES[stage_id],
            "warnings": [],
        }
        for stage_id in STAGE_IDS
    ]


def _empty_final_report() -> dict[str, Any]:
    return {
        "available": False,
        "report_id": None,
        "generated_at": None,
        "format": None,
        "href": None,
    }


def _runtime_run_view(record: RuntimeRunRecord) -> dict[str, Any]:
    view = {
        "schema_version": RUNTIME_RUN_VIEW_SCHEMA_VERSION,
        "run_id": record.run_id,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "status": record.status,
        "recoverable": bool(record.error["recoverable"]) if record.error else False,
        "can_resume": bool(record.error["can_resume"]) if record.error else False,
        "elapsed_seconds": _elapsed_seconds(record.created_at, record.updated_at),
        "filing_intent": copy.deepcopy(record.filing_intent),
        "runtime_config": copy.deepcopy(record.runtime_config),
        "source_confirmation": _public_source_confirmation(record.source_confirmation),
        "stages": [_public_stage(stage) for stage in record.stages],
        "current_stage": _current_stage(record),
        "warnings": [_public_warning(warning) for warning in record.warnings],
        "allowed_actions": _allowed_actions(record),
        "final_report": copy.deepcopy(record.final_report),
        "error": _public_error(record.error) if record.error is not None else None,
    }
    return validate_runtime_run_view(view)


def _public_source_confirmation(source_confirmation: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(source_confirmation)
    payload["package_warnings"] = [
        _public_warning(warning) for warning in payload.get("package_warnings", [])
    ]
    for slot in payload.get("slots", []):
        slot["warnings"] = [_public_warning(warning) for warning in slot.get("warnings", [])]
    return payload


def _public_stage(stage: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(stage)
    payload["warnings"] = [_public_warning(warning) for warning in payload.get("warnings", [])]
    return payload


def _public_warning(warning: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(warning)
    payload["artifact_refs"] = _public_artifact_refs(payload.get("artifact_refs", []))
    return payload


def _public_error(error: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(error)
    payload["artifact_refs"] = _public_artifact_refs(payload.get("artifact_refs", []))
    return payload


def _public_artifact_refs(refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    public_refs = []
    for ref in refs:
        public_ref = {
            key: ref[key]
            for key in ["artifact_id", "kind", "sha256"]
            if key in ref
        }
        public_refs.append(public_ref)
    return public_refs


def _stage_by_id(record: RuntimeRunRecord, stage_id: str) -> dict[str, Any]:
    for stage in record.stages:
        if stage["stage_id"] == stage_id:
            return stage
    raise ValueError(f"Unknown runtime stage: {stage_id}")


def _status_for_active_stage(stage_id: str) -> str:
    if stage_id == "source_discovery":
        return "discovering_sources"
    if stage_id == "source_confirmation":
        return "awaiting_source_confirmation"
    return "analyzing"


def _current_stage(record: RuntimeRunRecord) -> str | None:
    active = [stage["stage_id"] for stage in record.stages if stage["status"] == "active"]
    if record.status in {"created", "failed", "completed", "cancelled"}:
        return None
    return active[0] if active else None


def _allowed_actions(record: RuntimeRunRecord) -> list[dict[str, Any]]:
    actions = []
    if record.status == "failed" and record.allowed_action_state.get("can_resume"):
        actions.append(
            {
                "action": "resume_run",
                "method": "POST",
                "href": f"/runtime-runs/{record.run_id}/actions/resume",
            }
        )
    if record.status == "completed" and record.final_report.get("available"):
        actions.append(
            {
                "action": "open_final_report",
                "method": "GET",
                "href": f"/runtime-runs/{record.run_id}/report",
            }
        )
    if record.status == "awaiting_source_confirmation":
        ready_slots = [
            slot["role"]
            for slot in record.source_confirmation["slots"]
            if slot["status"] == "ready_for_review"
        ]
        retryable_slots = [
            slot["role"]
            for slot in record.source_confirmation["slots"]
            if slot["status"] in {"rejected", "unavailable"}
        ]
        selectable_slots = [
            slot
            for slot in record.source_confirmation["slots"]
            if slot.get("candidate_documents")
        ]
        if record.source_confirmation["confirmable"]:
            actions.append(
                {
                    "action": "confirm_sources",
                    "method": "POST",
                    "href": f"/runtime-runs/{record.run_id}/source-confirmation",
                }
            )
        if ready_slots:
            actions.append(
                {
                    "action": "reject_source",
                    "method": "POST",
                    "href": f"/runtime-runs/{record.run_id}/source-confirmation",
                    "scope": {"source_slots": ready_slots},
                }
            )
        if retryable_slots:
            actions.append(
                {
                    "action": "retry_source_discovery",
                    "method": "POST",
                    "href": f"/runtime-runs/{record.run_id}/source-confirmation",
                    "scope": {"source_slots": retryable_slots},
                }
            )
        if selectable_slots:
            actions.append(
                {
                    "action": "select_source_candidate",
                    "method": "POST",
                    "href": f"/runtime-runs/{record.run_id}/source-confirmation",
                    "scope": {
                        "source_slots": [slot["role"] for slot in selectable_slots],
                        "candidate_document_ids": {
                            slot["role"]: [
                                candidate["source_document_id"]
                                for candidate in slot.get("candidate_documents", [])
                            ]
                            for slot in selectable_slots
                        },
                    },
                }
            )
    if record.allowed_action_state.get("can_stop") and record.status not in {"failed", "completed", "cancelled"}:
        actions.append(
            {
                "action": "stop_run",
                "method": "POST",
                "href": f"/runtime-runs/{record.run_id}/actions/stop",
            }
        )
    return actions


def _elapsed_seconds(created_at: str, updated_at: str) -> int:
    created = _parse_runtime_timestamp(created_at)
    updated = _parse_runtime_timestamp(updated_at)
    return max(0, int((updated - created).total_seconds()))


def _parse_runtime_timestamp(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _json_dumps(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _json_loads(value: str | None, default: Any = None) -> Any:
    if value is None:
        return copy.deepcopy(default)
    return json.loads(value)


def _record_to_row(record: RuntimeRunRecord) -> tuple[Any, ...]:
    return (
        record.run_id,
        record.created_at,
        record.updated_at,
        record.status,
        _json_dumps(record.filing_intent),
        _json_dumps(record.runtime_config),
        _json_dumps(record.source_confirmation),
        _json_dumps(record.stages),
        _json_dumps(record.warnings),
        _json_dumps(record.error) if record.error is not None else None,
        _json_dumps(record.final_report),
        _json_dumps(record.allowed_action_state),
        _json_dumps(record.artifact_refs),
        _json_dumps(record.audit_metadata),
    )


def _record_from_row(row: tuple[Any, ...]) -> RuntimeRunRecord:
    return RuntimeRunRecord(
        run_id=row[0],
        created_at=row[1],
        updated_at=row[2],
        status=row[3],
        filing_intent=_json_loads(row[4], {}),
        runtime_config=_json_loads(row[5], {}),
        source_confirmation=_json_loads(row[6], {}),
        stages=_json_loads(row[7], []),
        warnings=_json_loads(row[8], []),
        error=_json_loads(row[9], None),
        final_report=_json_loads(row[10], {}),
        allowed_action_state=_json_loads(row[11], {}),
        artifact_refs=_json_loads(row[12], []),
        audit_metadata=_json_loads(row[13], {}),
    )


class FilesystemArtifactBodyStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def put_bytes(
        self,
        *,
        run_id: str,
        artifact_id: str,
        kind: str,
        body: bytes,
        relative_path: str | None = None,
        version: str | None = None,
        schema_version: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        path = self._artifact_path(run_id, artifact_id, relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        return self.reference_existing(
            run_id=run_id,
            artifact_id=artifact_id,
            kind=kind,
            path=path,
            version=version,
            schema_version=schema_version,
            metadata=metadata,
        )

    def reference_existing(
        self,
        *,
        run_id: str,
        artifact_id: str,
        kind: str,
        path: Path | str,
        version: str | None = None,
        schema_version: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body_path = Path(path)
        body = body_path.read_bytes()
        return {
            "artifact_id": artifact_id,
            "kind": kind,
            "run_id": run_id,
            "path": str(body_path),
            "size_bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
            "version": version,
            "schema_version": schema_version,
            "metadata": copy.deepcopy(metadata or {}),
        }

    def read_bytes(self, artifact_ref: dict[str, Any]) -> bytes:
        return Path(artifact_ref["path"]).read_bytes()

    def _artifact_path(self, run_id: str, artifact_id: str, relative_path: str | None) -> Path:
        _require_safe_artifact_segment(run_id, "run_id")
        _require_safe_artifact_segment(artifact_id, "artifact_id")
        if relative_path is not None:
            path = self.root / _require_safe_artifact_relative_path(relative_path)
        else:
            path = self.root / run_id / artifact_id
        resolved = path.resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError("Artifact path must stay under artifact root")
        return resolved


def _require_safe_artifact_segment(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not _ARTIFACT_ID_SEGMENT_PATTERN.fullmatch(value):
        raise ValueError(f"Artifact {field_name} must be a non-empty safe path segment")
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"Artifact {field_name} must be a single safe path segment")


def _require_safe_artifact_relative_path(relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        raise ValueError("Artifact relative_path must be relative to artifact root")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("Artifact relative_path must not contain traversal segments")
    return path
