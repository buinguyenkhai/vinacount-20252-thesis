from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict, dataclass, is_dataclass
from numbers import Integral
from typing import Any, Literal

from vinacount.final_report import validate_final_report


RUNTIME_RUN_VIEW_SCHEMA_VERSION = "runtime_run_view.v1"
FINAL_REPORT_ENDPOINT_SCHEMA_VERSION = "canonical_final_report.v1"
DEVELOPER_AUDIT_BUNDLE_SCHEMA_VERSION = "developer_audit_bundle.v1"
HITL_BOUNDARY = "source_identity_confirmation_before_extraction_or_cached_reuse"

RUNTIME_STATUSES = {
    "created",
    "discovering_sources",
    "awaiting_source_confirmation",
    "analyzing",
    "failed",
    "completed",
    "cancelled",
}
STAGE_IDS = [
    "source_discovery",
    "source_confirmation",
    "cache_lookup",
    "extraction",
    "tool_analysis",
    "detector_assessment",
    "aggregation",
    "report_generation",
]
STAGE_STATUSES = {"pending", "active", "completed", "failed", "skipped", "cancelled"}
ACTION_ENUM = {
    "confirm_sources",
    "reject_source",
    "retry_source_discovery",
    "select_source_candidate",
    "stop_run",
    "resume_run",
    "open_final_report",
    "download_developer_audit_bundle",
}
SOURCE_CONFIRMATION_STATUSES = {
    "not_started",
    "ready_for_review",
    "partially_rejected",
    "retrying",
    "confirmed",
    "stopped",
}
SOURCE_SLOT_ROLES = {"target", "prior_year_same_quarter"}
SOURCE_SLOT_STATUSES = {
    "pending_discovery",
    "ready_for_review",
    "rejected",
    "retrying_discovery",
    "locked",
    "unavailable",
}
REPORT_BASIS_PREFERENCES = {"consolidated", "separate"}
WARNING_SEVERITIES = {"info", "warning", "limitation"}
DEFAULT_REPORT_SYNTHESIS_MODEL_ID = "deepseek-v4-flash"
REPORT_SYNTHESIS_MODEL_REGISTRY = {
    "deepseek-v4-flash": {
        "label": "DeepSeek V4 Flash",
        "provider": "deepseek",
    },
    "deepseek-v4-pro": {
        "label": "DeepSeek V4 Pro",
        "provider": "deepseek",
    },
}
REPORT_SYNTHESIS_MODEL_IDS = set(REPORT_SYNTHESIS_MODEL_REGISTRY)
REPORT_SYNTHESIS_MODEL_SELECTIONS = {"default", "user_selected"}
ERROR_CODES = {
    "filing_intent_invalid",
    "source_discovery_unavailable",
    "source_package_unavailable",
    "source_identity_mismatch_after_confirmation",
    "cache_lookup_failed",
    "extraction_failed",
    "source_artifact_unreachable",
    "ocr_config_missing",
    "ocr_provider_failed",
    "raw_extraction_invalid",
    "report_memory_build_failed",
    "tool_analysis_failed",
    "detector_timeout",
    "detector_transport_failure",
    "detector_provider_response_invalid",
    "detector_output_invalid_json",
    "detector_guard_unrecoverable",
    "detector_contract_invalid",
    "aggregation_failed",
    "final_report_invalid",
    "audit_bundle_failed",
    "internal_error",
}
REJECTION_REASON_CODES = {
    "wrong_company",
    "wrong_period",
    "wrong_basis",
    "wrong_filing_status",
    "wrong_language",
    "not_full_financial_statement",
    "source_unreadable",
    "other",
}
FINAL_REPORT_FORMATS = {"json+markdown"}
ARTIFACT_BODY_STATUSES = {"present", "missing", "missing_reference"}
ARTIFACT_HASH_STATUSES = {"verified", "mismatch", "unavailable"}

_DEVELOPER_AUDIT_BUNDLE_REQUIRED_FIELDS = {
    "schema_version",
    "run_id",
    "exported_at",
    "run_status",
    "recoverable",
    "can_resume",
    "filing_intent",
    "runtime_config",
    "source_confirmation",
    "stage_records",
    "failed_stage",
    "resume_eligibility",
    "runtime_detector_mode",
    "cache_trace",
    "artifact_manifest",
    "detector_audit",
    "aggregation",
    "final_report",
    "warnings",
    "errors",
}
_PROHIBITED_AUDIT_KEYS = {
    "raw_prompt",
    "prompt",
    "prompt_text",
    "system_prompt",
    "developer_prompt",
    "hidden_reasoning",
    "chain_of_thought",
    "cot",
    "provider_debug",
    "llm_request",
    "llm_response",
    "body",
    "artifact_body",
}

_RUNTIME_REQUIRED_FIELDS = {
    "schema_version",
    "run_id",
    "created_at",
    "updated_at",
    "status",
    "recoverable",
    "can_resume",
    "elapsed_seconds",
    "filing_intent",
    "runtime_config",
    "source_confirmation",
    "stages",
    "current_stage",
    "warnings",
    "allowed_actions",
    "final_report",
    "error",
}
_PROHIBITED_PUBLIC_KEYS = {
    "raw_prompt",
    "prompt",
    "prompt_text",
    "system_prompt",
    "developer_prompt",
    "hidden_reasoning",
    "chain_of_thought",
    "cot",
    "detector_packet",
    "detector_packets",
    "raw_detector_packet",
    "detector_assessment_payload",
    "raw_assessment",
    "developer_audit_bundle",
    "audit_bundle",
    "stage_log",
    "sidechain",
    "raw_extraction_artifact",
    "raw_extraction_payload",
    "provider_debug",
    "llm_request",
    "llm_response",
}
_PROHIBITED_PUBLIC_TEXT = [
    re.compile(pattern)
    for pattern in [
        r"raw\s+prompt",
        r"hidden\s+reasoning",
        r"chain\s+of\s+thought",
        r"system\s+prompt",
        r"developer\s+audit\s+bundle",
        r"detector\s+packet",
        r"provider\s+debug",
        r"raw\s+extraction\s+artifact",
    ]
]


RuntimeStatus = Literal[
    "created",
    "discovering_sources",
    "awaiting_source_confirmation",
    "analyzing",
    "failed",
    "completed",
    "cancelled",
]
StageId = Literal[
    "source_discovery",
    "source_confirmation",
    "cache_lookup",
    "extraction",
    "tool_analysis",
    "detector_assessment",
    "aggregation",
    "report_generation",
]
StageStatus = Literal["pending", "active", "completed", "failed", "skipped", "cancelled"]
ActionName = Literal[
    "confirm_sources",
    "reject_source",
    "retry_source_discovery",
    "stop_run",
    "resume_run",
    "open_final_report",
    "download_developer_audit_bundle",
]


@dataclass(frozen=True)
class RuntimeFilingIntentView:
    company_identifier: str
    company_name_vi: str | None
    target_fiscal_year: int
    target_quarter: int
    report_basis_preference: Literal["consolidated", "separate"]
    report_language: Literal["vi", "en"] = "vi"


@dataclass(frozen=True)
class RuntimeRunCreateRequest:
    company_identifier: str
    target_fiscal_year: int
    target_quarter: int
    report_basis_preference: Literal["consolidated", "separate"]
    report_synthesis_model_id: str | None = None
    report_language: Literal["vi", "en"] = "vi"


@dataclass(frozen=True)
class ReportSynthesisModelConfig:
    id: str
    label: str
    provider: str
    selection: Literal["default", "user_selected"]


@dataclass(frozen=True)
class RuntimeConfig:
    report_synthesis_model: ReportSynthesisModelConfig


@dataclass(frozen=True)
class RuntimeProgress:
    processed: int
    total: int


@dataclass(frozen=True)
class RuntimeWarning:
    code: str
    severity: Literal["info", "warning", "limitation"]
    message: str
    stage_id: StageId | None = None
    source_slot_role: Literal["target", "prior_year_same_quarter"] | None = None
    artifact_refs: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class RuntimeErrorView:
    code: str
    message: str
    detail: str | None
    stage_id: StageId | None
    recoverable: bool
    can_resume: bool
    artifact_refs: list[dict[str, Any]]


@dataclass(frozen=True)
class RuntimeStage:
    stage_id: StageId
    status: StageStatus
    started_at: str | None
    completed_at: str | None
    summary: str
    progress: RuntimeProgress | None = None
    counts: dict[str, int] | None = None
    warnings: list[RuntimeWarning] | None = None


@dataclass(frozen=True)
class AllowedAction:
    action: ActionName
    method: Literal["GET", "POST"]
    href: str
    scope: dict[str, Any] | None = None


@dataclass(frozen=True)
class SourceCandidate:
    source_document_id: str
    company_name_vi: str
    ticker: str
    period_label: str
    quarter: int
    fiscal_year: int
    report_basis: Literal["consolidated", "separate"]
    filing_status: str
    document_type: str
    language: str
    source_origin: str
    source_name: str
    source_url: str
    is_searchable_version: bool
    file_size_bytes: int
    page_count: int
    visible_filing_label: str
    first_page_identity: dict[str, str | None]
    classification_evidence: list[str]
    audit_references: dict[str, Any]


@dataclass(frozen=True)
class SourceConfirmationSlot:
    role: Literal["target", "prior_year_same_quarter"]
    status: Literal[
        "pending_discovery",
        "ready_for_review",
        "rejected",
        "retrying_discovery",
        "locked",
        "unavailable",
    ]
    candidate: SourceCandidate | dict[str, Any] | None
    rejection: dict[str, Any] | None
    warnings: list[RuntimeWarning] | None = None


@dataclass(frozen=True)
class SourceConfirmationView:
    status: Literal[
        "not_started",
        "ready_for_review",
        "partially_rejected",
        "retrying",
        "confirmed",
        "stopped",
    ]
    confirmable: bool
    hitl_boundary: str
    slots: list[SourceConfirmationSlot]
    package_warnings: list[RuntimeWarning] | None = None


@dataclass(frozen=True)
class FinalReportMetadata:
    available: bool
    report_id: str | None
    generated_at: str | None
    format: Literal["json+markdown"] | None
    href: str | None


@dataclass(frozen=True)
class RuntimeRunView:
    schema_version: str
    run_id: str
    created_at: str
    updated_at: str
    status: RuntimeStatus
    recoverable: bool
    can_resume: bool
    elapsed_seconds: int
    filing_intent: RuntimeFilingIntentView
    runtime_config: RuntimeConfig
    source_confirmation: SourceConfirmationView
    stages: list[RuntimeStage]
    current_stage: StageId | None
    warnings: list[RuntimeWarning]
    allowed_actions: list[AllowedAction]
    final_report: FinalReportMetadata
    error: RuntimeErrorView | None


def runtime_contract_json_schemas() -> dict[str, dict[str, Any]]:
    return {
        "DeveloperAuditBundle": developer_audit_bundle_json_schema(),
        "RuntimeRunCreateRequest": runtime_run_create_request_json_schema(),
        "RuntimeRunView": runtime_run_view_json_schema(),
        "SourceCandidate": source_candidate_json_schema(),
        "FinalReportEndpoint": final_report_endpoint_json_schema(),
    }


def runtime_run_create_request_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "company_identifier": {"type": "string"},
            "target_fiscal_year": {"type": "integer"},
            "target_quarter": {"type": "integer", "minimum": 1, "maximum": 4},
            "report_basis_preference": {"type": "string", "enum": sorted(REPORT_BASIS_PREFERENCES)},
            "report_synthesis_model_id": {"type": "string", "enum": sorted(REPORT_SYNTHESIS_MODEL_IDS)},
            "report_language": {"type": "string", "enum": ["en", "vi"]},
        },
        "required": [
            "company_identifier",
            "target_fiscal_year",
            "target_quarter",
            "report_basis_preference",
        ],
        "additionalProperties": False,
    }


def runtime_run_view_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "_fixture_meta": {"type": "object"},
            "schema_version": {"const": RUNTIME_RUN_VIEW_SCHEMA_VERSION},
            "run_id": {"type": "string"},
            "created_at": {"type": "string"},
            "updated_at": {"type": "string"},
            "status": {"type": "string", "enum": sorted(RUNTIME_STATUSES)},
            "recoverable": {"type": "boolean"},
            "can_resume": {"type": "boolean"},
            "elapsed_seconds": {"type": "integer", "minimum": 0},
            "filing_intent": runtime_filing_intent_json_schema(),
            "runtime_config": runtime_config_json_schema(),
            "source_confirmation": source_confirmation_json_schema(),
            "stages": {
                "type": "array",
                "items": runtime_stage_json_schema(),
                "minItems": len(STAGE_IDS),
                "maxItems": len(STAGE_IDS),
            },
            "current_stage": {"type": ["string", "null"], "enum": STAGE_IDS + [None]},
            "warnings": {"type": "array", "items": runtime_warning_json_schema()},
            "allowed_actions": {"type": "array", "items": allowed_action_json_schema()},
            "final_report": final_report_metadata_json_schema(),
            "error": {"anyOf": [runtime_error_json_schema(), {"type": "null"}]},
        },
        "required": sorted(_RUNTIME_REQUIRED_FIELDS),
        "additionalProperties": False,
    }


def runtime_config_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "report_synthesis_model": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "enum": sorted(REPORT_SYNTHESIS_MODEL_IDS)},
                    "label": {"type": "string"},
                    "provider": {"type": "string"},
                    "selection": {"type": "string", "enum": sorted(REPORT_SYNTHESIS_MODEL_SELECTIONS)},
                },
                "required": ["id", "label", "provider", "selection"],
                "additionalProperties": False,
            },
        },
        "required": ["report_synthesis_model"],
        "additionalProperties": False,
    }


def runtime_filing_intent_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "company_identifier": {"type": "string"},
            "company_name_vi": {"type": ["string", "null"]},
            "target_fiscal_year": {"type": "integer"},
            "target_quarter": {"type": "integer", "minimum": 1, "maximum": 4},
            "report_basis_preference": {"type": "string", "enum": sorted(REPORT_BASIS_PREFERENCES)},
            "report_language": {"type": "string", "enum": ["en", "vi"]},
        },
        "required": [
            "company_identifier",
            "company_name_vi",
            "target_fiscal_year",
            "target_quarter",
            "report_basis_preference",
            "report_language",
        ],
        "additionalProperties": False,
    }


def runtime_stage_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "stage_id": {"type": "string", "enum": STAGE_IDS},
            "status": {"type": "string", "enum": sorted(STAGE_STATUSES)},
            "started_at": {"type": ["string", "null"]},
            "completed_at": {"type": ["string", "null"]},
            "summary": {"type": "string"},
            "progress": {
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {
                            "processed": {"type": "integer", "minimum": 0},
                            "total": {"type": "integer", "minimum": 0},
                        },
                        "required": ["processed", "total"],
                        "additionalProperties": False,
                    },
                    {"type": "null"},
                ]
            },
            "counts": {"type": ["object", "null"]},
            "warnings": {"type": "array", "items": runtime_warning_json_schema()},
        },
        "required": ["stage_id", "status", "started_at", "completed_at", "summary", "warnings"],
        "additionalProperties": False,
    }


def runtime_warning_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "code": {"type": "string"},
            "severity": {"type": "string", "enum": sorted(WARNING_SEVERITIES)},
            "message": {"type": "string"},
            "stage_id": {"type": ["string", "null"], "enum": STAGE_IDS + [None]},
            "source_slot_role": {"type": ["string", "null"], "enum": sorted(SOURCE_SLOT_ROLES) + [None]},
            "artifact_refs": {"type": "array"},
        },
        "required": ["code", "severity", "message", "artifact_refs"],
        "additionalProperties": False,
    }


def runtime_error_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "code": {"type": "string", "enum": sorted(ERROR_CODES)},
            "message": {"type": "string"},
            "detail": {"type": ["string", "null"]},
            "stage_id": {"type": ["string", "null"], "enum": STAGE_IDS + [None]},
            "recoverable": {"type": "boolean"},
            "can_resume": {"type": "boolean"},
            "artifact_refs": {"type": "array"},
        },
        "required": ["code", "message", "detail", "stage_id", "recoverable", "can_resume", "artifact_refs"],
        "additionalProperties": False,
    }


def allowed_action_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": sorted(ACTION_ENUM)},
            "method": {"type": "string", "enum": ["GET", "POST"]},
            "href": {"type": "string"},
            "scope": {"type": ["object", "null"]},
        },
        "required": ["action", "method", "href"],
        "additionalProperties": False,
    }


def source_confirmation_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": sorted(SOURCE_CONFIRMATION_STATUSES)},
            "confirmable": {"type": "boolean"},
            "hitl_boundary": {"const": HITL_BOUNDARY},
            "slots": {"type": "array", "items": source_confirmation_slot_json_schema()},
            "package_warnings": {"type": "array", "items": runtime_warning_json_schema()},
        },
        "required": ["status", "confirmable", "hitl_boundary", "slots", "package_warnings"],
        "additionalProperties": False,
    }


def source_confirmation_slot_json_schema() -> dict[str, Any]:
    rejection_schema = {
        "type": "object",
        "properties": {
            "reason_code": {"type": "string", "enum": sorted(REJECTION_REASON_CODES)},
            "message": {"type": "string"},
            "comment": {"type": ["string", "null"]},
        },
        "required": ["reason_code", "message", "comment"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "role": {"type": "string", "enum": sorted(SOURCE_SLOT_ROLES)},
            "status": {"type": "string", "enum": sorted(SOURCE_SLOT_STATUSES)},
            "candidate": {"anyOf": [source_candidate_json_schema(), {"type": "null"}]},
            "candidate_documents": {
                "type": "array",
                "items": source_candidate_json_schema(),
            },
            "rejection": {"anyOf": [rejection_schema, {"type": "null"}]},
            "warnings": {"type": "array", "items": runtime_warning_json_schema()},
        },
        "required": ["role", "status", "candidate", "rejection", "warnings"],
        "additionalProperties": False,
    }


def source_candidate_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "source_document_id": {"type": "string"},
            "company_name_vi": {"type": "string"},
            "ticker": {"type": "string"},
            "period_label": {"type": "string"},
            "quarter": {"type": "integer", "minimum": 1, "maximum": 4},
            "fiscal_year": {"type": "integer"},
            "report_basis": {"type": "string", "enum": sorted(REPORT_BASIS_PREFERENCES)},
            "filing_status": {"type": "string"},
            "document_type": {"type": "string"},
            "language": {"type": "string"},
            "source_origin": {"type": "string"},
            "source_name": {"type": "string"},
            "source_url": {"type": "string"},
            "is_searchable_version": {"type": "boolean"},
            "file_size_bytes": {"type": "integer", "minimum": 0},
            "page_count": {"type": "integer", "minimum": 0},
            "visible_filing_label": {"type": "string"},
            "first_page_identity": {"type": "object"},
            "classification_evidence": {"type": "array", "items": {"type": "string"}},
            "audit_references": {"type": "object"},
        },
        "required": [
            "source_document_id",
            "company_name_vi",
            "ticker",
            "period_label",
            "quarter",
            "fiscal_year",
            "report_basis",
            "filing_status",
            "document_type",
            "language",
            "source_origin",
            "source_name",
            "source_url",
            "is_searchable_version",
            "file_size_bytes",
            "page_count",
            "visible_filing_label",
            "first_page_identity",
            "classification_evidence",
            "audit_references",
        ],
        "additionalProperties": False,
    }


def final_report_metadata_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "available": {"type": "boolean"},
            "report_id": {"type": ["string", "null"]},
            "generated_at": {"type": ["string", "null"]},
            "format": {"type": ["string", "null"], "enum": sorted(FINAL_REPORT_FORMATS) + [None]},
            "href": {"type": ["string", "null"]},
        },
        "required": ["available", "report_id", "generated_at", "format", "href"],
        "additionalProperties": False,
    }


def final_report_endpoint_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "_fixture_meta": {"type": "object"},
            "schema_version": {"const": FINAL_REPORT_ENDPOINT_SCHEMA_VERSION},
            "run_id": {"type": "string"},
            "report_id": {"type": "string"},
            "generated_at": {"type": "string"},
            "report_json": {"type": "object"},
            "report_markdown": {"type": "string"},
            "artifact_refs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "artifact_id": {"type": "string"},
                        "kind": {"type": "string"},
                        "sha256": {"type": "string"},
                    },
                    "required": ["artifact_id", "kind", "sha256"],
                    "additionalProperties": False,
                },
            },
        },
        "required": [
            "schema_version",
            "run_id",
            "report_id",
            "generated_at",
            "report_json",
            "report_markdown",
            "artifact_refs",
        ],
        "additionalProperties": False,
    }


def developer_audit_bundle_json_schema() -> dict[str, Any]:
    artifact_ref_schema = {
        "type": "object",
        "properties": {
            "artifact_id": {"type": "string"},
            "kind": {"type": "string"},
            "run_id": {"type": "string"},
            "sha256": {"type": ["string", "null"]},
            "observed_sha256": {"type": ["string", "null"]},
            "size_bytes": {"type": ["integer", "null"]},
            "version": {"type": ["string", "null"]},
            "schema_version": {"type": ["string", "null"]},
            "metadata": {"type": "object"},
            "storage": {
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {
                            "type": {"const": "filesystem"},
                            "path": {"type": "string"},
                        },
                        "required": ["type", "path"],
                        "additionalProperties": False,
                    },
                    {"type": "null"},
                ]
            },
            "body_status": {"type": "string", "enum": sorted(ARTIFACT_BODY_STATUSES)},
            "hash_status": {"type": "string", "enum": sorted(ARTIFACT_HASH_STATUSES)},
        },
        "required": [
            "artifact_id",
            "kind",
            "run_id",
            "sha256",
            "observed_sha256",
            "size_bytes",
            "version",
            "schema_version",
            "metadata",
            "storage",
            "body_status",
            "hash_status",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "schema_version": {"const": DEVELOPER_AUDIT_BUNDLE_SCHEMA_VERSION},
            "run_id": {"type": "string"},
            "exported_at": {"type": "string"},
            "run_status": {"type": "string", "enum": sorted(RUNTIME_STATUSES)},
            "recoverable": {"type": "boolean"},
            "can_resume": {"type": "boolean"},
            "filing_intent": runtime_filing_intent_json_schema(),
            "runtime_config": runtime_config_json_schema(),
            "source_confirmation": source_confirmation_json_schema(),
            "stage_records": {"type": "array", "items": runtime_stage_json_schema()},
            "failed_stage": {"type": ["string", "null"], "enum": STAGE_IDS + [None]},
            "resume_eligibility": {"type": "object"},
            "runtime_detector_mode": {"type": "object"},
            "cache_trace": {"type": "object"},
            "artifact_manifest": {
                "type": "object",
                "properties": {
                    "refs": {"type": "array", "items": artifact_ref_schema},
                    "missing_artifacts": {"type": "array", "items": {"type": "object"}},
                },
                "required": ["refs", "missing_artifacts"],
                "additionalProperties": False,
            },
            "detector_audit": {"type": "object"},
            "aggregation": {"type": "object"},
            "final_report": {"type": "object"},
            "warnings": {"type": "array", "items": runtime_warning_json_schema()},
            "errors": {"type": "array", "items": runtime_error_json_schema()},
        },
        "required": sorted(_DEVELOPER_AUDIT_BUNDLE_REQUIRED_FIELDS),
        "additionalProperties": False,
    }


def validate_runtime_run_view(payload: Any) -> dict[str, Any]:
    data = _contract_dict(payload)
    if not isinstance(data, dict):
        raise ValueError("RuntimeRunView must be an object")
    extra = set(data) - _RUNTIME_REQUIRED_FIELDS - {"_fixture_meta"}
    if extra:
        raise ValueError(f"RuntimeRunView has unknown fields: {sorted(extra)}")
    missing = _RUNTIME_REQUIRED_FIELDS - set(data)
    if missing:
        raise ValueError(f"RuntimeRunView is missing fields: {sorted(missing)}")
    _validate_fixture_meta(data.get("_fixture_meta"))
    _require_equal(data["schema_version"], RUNTIME_RUN_VIEW_SCHEMA_VERSION, "schema_version")
    _require_non_empty_string(data["run_id"], "run_id")
    _require_non_empty_string(data["created_at"], "created_at")
    _require_non_empty_string(data["updated_at"], "updated_at")
    _require_member(data["status"], RUNTIME_STATUSES, "status")
    _require_bool(data["recoverable"], "recoverable")
    _require_bool(data["can_resume"], "can_resume")
    _require_non_negative_int(data["elapsed_seconds"], "elapsed_seconds")

    _validate_filing_intent(data["filing_intent"])
    _validate_runtime_config(data["runtime_config"])
    _validate_source_confirmation(data["source_confirmation"])
    _validate_stage_registry(data["stages"])
    _validate_current_stage(data["current_stage"])
    _validate_warning_list(data["warnings"], "warnings")
    for action in data["allowed_actions"]:
        _validate_allowed_action(action, data["run_id"])
    _validate_final_report_metadata(data["final_report"], data["run_id"])
    if data["error"] is not None:
        _validate_runtime_error(data["error"])

    _validate_status_invariants(data)
    assert_public_runtime_payload_safe(data)
    return data


def validate_developer_audit_bundle(payload: Any) -> dict[str, Any]:
    data = _contract_dict(payload)
    if not isinstance(data, dict):
        raise ValueError("DeveloperAuditBundle must be an object")
    extra = set(data) - _DEVELOPER_AUDIT_BUNDLE_REQUIRED_FIELDS
    if extra:
        raise ValueError(f"DeveloperAuditBundle has unknown fields: {sorted(extra)}")
    missing = _DEVELOPER_AUDIT_BUNDLE_REQUIRED_FIELDS - set(data)
    if missing:
        raise ValueError(f"DeveloperAuditBundle is missing fields: {sorted(missing)}")

    _require_equal(data["schema_version"], DEVELOPER_AUDIT_BUNDLE_SCHEMA_VERSION, "schema_version")
    _require_non_empty_string(data["run_id"], "run_id")
    _require_non_empty_string(data["exported_at"], "exported_at")
    _require_member(data["run_status"], RUNTIME_STATUSES, "run_status")
    _require_bool(data["recoverable"], "recoverable")
    _require_bool(data["can_resume"], "can_resume")
    if data["run_status"] not in {"completed", "failed"}:
        raise ValueError("DeveloperAuditBundle can only be exported for completed or failed runs")
    if data["run_status"] == "failed" and not data["recoverable"]:
        raise ValueError("DeveloperAuditBundle failed exports require recoverable failures in V1")
    if data["can_resume"] and not data["recoverable"]:
        raise ValueError("DeveloperAuditBundle can_resume requires recoverable")

    _validate_filing_intent(data["filing_intent"])
    _validate_runtime_config(data["runtime_config"])
    _validate_source_confirmation(data["source_confirmation"])
    _validate_stage_registry(data["stage_records"])
    if data["failed_stage"] is not None:
        _require_member(data["failed_stage"], set(STAGE_IDS), "failed_stage")
    if data["run_status"] == "failed" and data["failed_stage"] is None:
        raise ValueError("failed DeveloperAuditBundle requires failed_stage")
    if data["run_status"] == "completed" and data["failed_stage"] is not None:
        raise ValueError("completed DeveloperAuditBundle requires failed_stage null")

    _validate_resume_eligibility(data["resume_eligibility"])
    if not isinstance(data["runtime_detector_mode"], dict) or not data["runtime_detector_mode"]:
        raise ValueError("runtime_detector_mode must be a non-empty object")
    _validate_cache_trace(data["cache_trace"])
    _validate_artifact_manifest(data["artifact_manifest"], data["run_id"])
    _validate_detector_audit(data["detector_audit"], data["run_id"])
    _validate_audit_artifact_group(data["aggregation"], "aggregation", data["run_id"])
    _validate_audit_final_report(data["final_report"], data["run_id"])
    _validate_warning_list(data["warnings"], "warnings")
    for error in data["errors"]:
        _validate_runtime_error(error)
    if data["run_status"] == "failed" and not data["errors"]:
        raise ValueError("failed DeveloperAuditBundle requires errors")
    assert_developer_audit_bundle_safe(data)
    return data


def validate_runtime_run_create_request(payload: Any) -> dict[str, Any]:
    data = _contract_dict(payload)
    if not isinstance(data, dict):
        raise ValueError("RuntimeRunCreateRequest must be an object")
    schema = runtime_run_create_request_json_schema()
    required = set(schema["required"])
    allowed = set(schema["properties"])
    extra = set(data) - allowed
    if extra:
        raise ValueError(f"RuntimeRunCreateRequest has unknown fields: {sorted(extra)}")
    missing = required - set(data)
    if missing:
        raise ValueError(f"RuntimeRunCreateRequest is missing fields: {sorted(missing)}")
    _require_non_empty_string(data["company_identifier"], "company_identifier")
    _require_int(data["target_fiscal_year"], "target_fiscal_year")
    if data["target_quarter"] not in {1, 2, 3, 4}:
        raise ValueError("target_quarter must be between 1 and 4")
    _require_member(data["report_basis_preference"], REPORT_BASIS_PREFERENCES, "report_basis_preference")
    if "report_synthesis_model_id" in data:
        _require_member(data["report_synthesis_model_id"], REPORT_SYNTHESIS_MODEL_IDS, "report_synthesis_model_id")
    if "report_language" in data:
        _require_member(data["report_language"], {"vi", "en"}, "report_language")
    else:
        data["report_language"] = "vi"
    assert_public_runtime_payload_safe(data)
    return data


def validate_final_report_endpoint(payload: Any) -> dict[str, Any]:
    data = _contract_dict(payload)
    if not isinstance(data, dict):
        raise ValueError("FinalReportEndpoint must be an object")
    required = set(final_report_endpoint_json_schema()["required"])
    extra = set(data) - required - {"_fixture_meta"}
    if extra:
        raise ValueError(f"FinalReportEndpoint has unknown fields: {sorted(extra)}")
    missing = required - set(data)
    if missing:
        raise ValueError(f"FinalReportEndpoint is missing fields: {sorted(missing)}")
    _validate_fixture_meta(data.get("_fixture_meta"))
    _require_equal(data["schema_version"], FINAL_REPORT_ENDPOINT_SCHEMA_VERSION, "schema_version")
    _require_non_empty_string(data["run_id"], "run_id")
    _require_non_empty_string(data["report_id"], "report_id")
    _require_non_empty_string(data["generated_at"], "generated_at")
    if not isinstance(data["report_json"], dict):
        raise ValueError("FinalReportEndpoint.report_json must be an object")
    validate_final_report(data["report_json"])
    if data["report_id"] != data["report_json"]["report_id"]:
        raise ValueError("FinalReportEndpoint.report_id must match report_json.report_id")
    _require_non_empty_string(data["report_markdown"], "report_markdown")
    _validate_artifact_refs(data["artifact_refs"], "artifact_refs")
    assert_public_runtime_payload_safe(data)
    return data


def validate_filing_intent_error_response(payload: Any) -> dict[str, Any]:
    data = _contract_dict(payload)
    if not isinstance(data, dict) or set(data) - {"_fixture_meta", "error"}:
        raise ValueError("Filing intent error response must only contain error and optional fixture metadata")
    _validate_fixture_meta(data.get("_fixture_meta"))
    error = data.get("error")
    if not isinstance(error, dict):
        raise ValueError("Filing intent error response requires error object")
    allowed_fields = {"code", "message", "field_errors"}
    if set(error) - allowed_fields:
        raise ValueError("Filing intent error has unknown fields")
    if error.get("code") != "filing_intent_invalid":
        raise ValueError("Filing intent error code must be filing_intent_invalid")
    _require_non_empty_string(error.get("message"), "error.message")
    field_errors = error.get("field_errors")
    if not isinstance(field_errors, dict) or not field_errors:
        raise ValueError("Filing intent error requires non-empty field_errors")
    for field, message in field_errors.items():
        _require_non_empty_string(field, "field_errors key")
        _require_non_empty_string(message, f"field_errors.{field}")
    assert_public_runtime_payload_safe(data)
    return data


def assert_public_runtime_payload_safe(payload: Any) -> None:
    _scan_public_payload(_contract_dict(payload), path="$")


def assert_developer_audit_bundle_safe(payload: Any) -> None:
    _scan_developer_audit_payload(_contract_dict(payload), path="$")


def _validate_filing_intent(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("filing_intent must be an object")
    required = set(runtime_filing_intent_json_schema()["required"])
    if set(value) != required:
        raise ValueError("filing_intent fields do not match contract")
    _require_non_empty_string(value["company_identifier"], "filing_intent.company_identifier")
    if value["company_name_vi"] is not None:
        _require_non_empty_string(value["company_name_vi"], "filing_intent.company_name_vi")
    _require_int(value["target_fiscal_year"], "filing_intent.target_fiscal_year")
    if value["target_quarter"] not in {1, 2, 3, 4}:
        raise ValueError("filing_intent.target_quarter must be between 1 and 4")
    _require_member(value["report_basis_preference"], REPORT_BASIS_PREFERENCES, "filing_intent.report_basis_preference")
    _require_member(value["report_language"], {"vi", "en"}, "filing_intent.report_language")


def runtime_config_for_report_synthesis_model(report_synthesis_model_id: str | None = None) -> dict[str, Any]:
    model_id = report_synthesis_model_id or DEFAULT_REPORT_SYNTHESIS_MODEL_ID
    _require_member(model_id, REPORT_SYNTHESIS_MODEL_IDS, "report_synthesis_model_id")
    registry_entry = REPORT_SYNTHESIS_MODEL_REGISTRY[model_id]
    return {
        "report_synthesis_model": {
            "id": model_id,
            "label": registry_entry["label"],
            "provider": registry_entry["provider"],
            "selection": "user_selected" if report_synthesis_model_id else "default",
        }
    }


def _validate_runtime_config(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("runtime_config must be an object")
    if set(value) != {"report_synthesis_model"}:
        raise ValueError("runtime_config fields do not match contract")
    model = value["report_synthesis_model"]
    if not isinstance(model, dict):
        raise ValueError("runtime_config.report_synthesis_model must be an object")
    if set(model) != {"id", "label", "provider", "selection"}:
        raise ValueError("runtime_config.report_synthesis_model fields do not match contract")
    _require_member(model["id"], REPORT_SYNTHESIS_MODEL_IDS, "runtime_config.report_synthesis_model.id")
    _require_non_empty_string(model["label"], "runtime_config.report_synthesis_model.label")
    _require_non_empty_string(model["provider"], "runtime_config.report_synthesis_model.provider")
    _require_member(model["selection"], REPORT_SYNTHESIS_MODEL_SELECTIONS, "runtime_config.report_synthesis_model.selection")
    expected = REPORT_SYNTHESIS_MODEL_REGISTRY[model["id"]]
    if model["label"] != expected["label"]:
        raise ValueError("runtime_config.report_synthesis_model.label must match the backend model registry")
    if model["provider"] != expected["provider"]:
        raise ValueError("runtime_config.report_synthesis_model.provider must match the backend model registry")


def _validate_source_confirmation(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("source_confirmation must be an object")
    required = set(source_confirmation_json_schema()["required"])
    if set(value) != required:
        raise ValueError("source_confirmation fields do not match contract")
    _require_member(value["status"], SOURCE_CONFIRMATION_STATUSES, "source_confirmation.status")
    _require_bool(value["confirmable"], "source_confirmation.confirmable")
    _require_equal(value["hitl_boundary"], HITL_BOUNDARY, "source_confirmation.hitl_boundary")
    if not isinstance(value["slots"], list):
        raise ValueError("source_confirmation.slots must be a list")
    roles = []
    for slot in value["slots"]:
        _validate_source_confirmation_slot(slot)
        roles.append(slot["role"])
    if set(roles) != SOURCE_SLOT_ROLES:
        raise ValueError("source_confirmation.slots must contain target and prior_year_same_quarter")
    if len(roles) != len(set(roles)):
        raise ValueError("source_confirmation.slots contains duplicate roles")
    all_slots_ready = all(slot["status"] == "ready_for_review" for slot in value["slots"])
    if value["confirmable"] and not all_slots_ready:
        raise ValueError("source_confirmation.confirmable requires every slot to be ready_for_review")
    if value["status"] == "ready_for_review" and all_slots_ready and not value["confirmable"]:
        raise ValueError("ready_for_review source_confirmation should be confirmable")
    _validate_warning_list(value["package_warnings"], "source_confirmation.package_warnings")


def _validate_source_confirmation_slot(slot: Any) -> None:
    if not isinstance(slot, dict):
        raise ValueError("source_confirmation slot must be an object")
    required = set(source_confirmation_slot_json_schema()["required"])
    allowed = required | {"candidate_documents"}
    if not required <= set(slot) <= allowed:
        raise ValueError("source_confirmation slot fields do not match contract")
    _require_member(slot["role"], SOURCE_SLOT_ROLES, "source_confirmation.slot.role")
    _require_member(slot["status"], SOURCE_SLOT_STATUSES, "source_confirmation.slot.status")
    if slot["candidate"] is not None:
        _validate_source_candidate(slot["candidate"])
    candidate_documents = slot.get("candidate_documents", [])
    if not isinstance(candidate_documents, list):
        raise ValueError("source_confirmation slot candidate_documents must be a list")
    for candidate_document in candidate_documents:
        _validate_source_candidate(candidate_document)
    rejection = slot["rejection"]
    if rejection is not None:
        if not isinstance(rejection, dict):
            raise ValueError("source_confirmation slot rejection must be an object or null")
        if set(rejection) != {"reason_code", "message", "comment"}:
            raise ValueError("source_confirmation slot rejection fields do not match contract")
        if rejection.get("reason_code") not in REJECTION_REASON_CODES:
            raise ValueError("source_confirmation slot rejection has invalid reason_code")
        _require_non_empty_string(rejection.get("message"), "source_confirmation.slot.rejection.message")
        if rejection.get("comment") is not None:
            _require_non_empty_string(rejection.get("comment"), "source_confirmation.slot.rejection.comment")
    if slot["status"] == "rejected" and rejection is None:
        raise ValueError("rejected source slot requires rejection details")
    if slot["status"] in {"ready_for_review", "locked", "rejected"} and slot["candidate"] is None:
        raise ValueError("reviewable source slot requires candidate")
    _validate_warning_list(slot["warnings"], "source_confirmation.slot.warnings")


def _validate_source_candidate(candidate: Any) -> None:
    if not isinstance(candidate, dict):
        raise ValueError("SourceCandidate must be an object")
    required = set(source_candidate_json_schema()["required"])
    if set(candidate) != required:
        raise ValueError("SourceCandidate fields do not match contract")
    for field in [
        "source_document_id",
        "company_name_vi",
        "ticker",
        "period_label",
        "filing_status",
        "document_type",
        "language",
        "source_origin",
        "source_name",
        "source_url",
        "visible_filing_label",
    ]:
        _require_non_empty_string(candidate[field], f"SourceCandidate.{field}")
    if candidate["quarter"] not in {1, 2, 3, 4}:
        raise ValueError("SourceCandidate.quarter must be between 1 and 4")
    _require_int(candidate["fiscal_year"], "SourceCandidate.fiscal_year")
    _require_member(candidate["report_basis"], REPORT_BASIS_PREFERENCES, "SourceCandidate.report_basis")
    _require_bool(candidate["is_searchable_version"], "SourceCandidate.is_searchable_version")
    _require_non_negative_int(candidate["file_size_bytes"], "SourceCandidate.file_size_bytes")
    _require_non_negative_int(candidate["page_count"], "SourceCandidate.page_count")
    if not isinstance(candidate["first_page_identity"], dict):
        raise ValueError("SourceCandidate.first_page_identity must be an object")
    if not isinstance(candidate["classification_evidence"], list) or not candidate["classification_evidence"]:
        raise ValueError("SourceCandidate.classification_evidence must be a non-empty list")
    for evidence in candidate["classification_evidence"]:
        _require_non_empty_string(evidence, "SourceCandidate.classification_evidence item")
    if not isinstance(candidate["audit_references"], dict):
        raise ValueError("SourceCandidate.audit_references must be an object")


def _validate_stage_registry(stages: Any) -> None:
    if not isinstance(stages, list):
        raise ValueError("stages must be a list")
    if [stage.get("stage_id") if isinstance(stage, dict) else None for stage in stages] != STAGE_IDS:
        raise ValueError("stages must match the fixed runtime stage registry order")
    for stage in stages:
        _validate_stage(stage)


def _validate_stage(stage: Any) -> None:
    if not isinstance(stage, dict):
        raise ValueError("stage must be an object")
    allowed = {"stage_id", "status", "started_at", "completed_at", "summary", "progress", "counts", "warnings"}
    required = {"stage_id", "status", "started_at", "completed_at", "summary", "warnings"}
    if set(stage) - allowed:
        raise ValueError("stage has unknown fields")
    if required - set(stage):
        raise ValueError("stage is missing fields")
    _require_member(stage["stage_id"], set(STAGE_IDS), "stage.stage_id")
    _require_member(stage["status"], STAGE_STATUSES, "stage.status")
    if stage["started_at"] is not None:
        _require_non_empty_string(stage["started_at"], "stage.started_at")
    if stage["completed_at"] is not None:
        _require_non_empty_string(stage["completed_at"], "stage.completed_at")
    _require_non_empty_string(stage["summary"], "stage.summary")
    progress = stage.get("progress")
    if progress is not None:
        if not isinstance(progress, dict) or set(progress) != {"processed", "total"}:
            raise ValueError("stage.progress must contain processed and total")
        _require_non_negative_int(progress["processed"], "stage.progress.processed")
        _require_non_negative_int(progress["total"], "stage.progress.total")
        if progress["processed"] > progress["total"]:
            raise ValueError("stage.progress.processed cannot exceed total")
    counts = stage.get("counts")
    if counts is not None:
        if not isinstance(counts, dict):
            raise ValueError("stage.counts must be an object or null")
        for key, value in counts.items():
            _require_non_empty_string(key, "stage.counts key")
            _require_non_negative_int(value, f"stage.counts.{key}")
    _validate_warning_list(stage["warnings"], "stage.warnings")


def _validate_current_stage(value: Any) -> None:
    if value is not None:
        _require_member(value, set(STAGE_IDS), "current_stage")


def _validate_allowed_action(action: Any, run_id: str) -> None:
    if not isinstance(action, dict):
        raise ValueError("allowed_actions must contain typed action objects")
    allowed = {"action", "method", "href", "scope"}
    required = {"action", "method", "href"}
    if set(action) - allowed or required - set(action):
        raise ValueError("allowed action fields do not match contract")
    _require_member(action["action"], ACTION_ENUM, "allowed_action.action")
    _require_member(action["method"], {"GET", "POST"}, "allowed_action.method")
    _require_non_empty_string(action["href"], "allowed_action.href")
    if not action["href"].startswith(f"/runtime-runs/{run_id}"):
        raise ValueError("allowed_action.href must be scoped to this run")
    if action["action"] == "open_final_report" and action["method"] != "GET":
        raise ValueError("open_final_report must use GET")
    if action["action"] != "open_final_report" and action["method"] != "POST":
        raise ValueError("mutating runtime actions must use POST")
    if "scope" in action and action["scope"] is not None and not isinstance(action["scope"], dict):
        raise ValueError("allowed_action.scope must be an object or null")


def _validate_warning_list(warnings: Any, field_name: str) -> None:
    if not isinstance(warnings, list):
        raise ValueError(f"{field_name} must be a list")
    for warning in warnings:
        _validate_warning(warning)


def _validate_warning(warning: Any) -> None:
    if not isinstance(warning, dict):
        raise ValueError("Runtime warning must be an object")
    required = {"code", "severity", "message", "artifact_refs"}
    optional = {"stage_id", "source_slot_role"}
    if set(warning) - required - optional or required - set(warning):
        raise ValueError("Runtime warning fields do not match contract")
    _require_non_empty_string(warning["code"], "warning.code")
    _require_member(warning["severity"], WARNING_SEVERITIES, "warning.severity")
    _require_non_empty_string(warning["message"], "warning.message")
    if warning.get("stage_id") is not None:
        _require_member(warning["stage_id"], set(STAGE_IDS), "warning.stage_id")
    if warning.get("source_slot_role") is not None:
        _require_member(warning["source_slot_role"], SOURCE_SLOT_ROLES, "warning.source_slot_role")
    _validate_artifact_refs(warning["artifact_refs"], "warning.artifact_refs")


def _validate_runtime_error(error: Any) -> None:
    if not isinstance(error, dict):
        raise ValueError("Runtime error must be an object")
    required = set(runtime_error_json_schema()["required"])
    if set(error) != required:
        raise ValueError("Runtime error fields do not match contract")
    _require_member(error["code"], ERROR_CODES, "error.code")
    _require_non_empty_string(error["message"], "error.message")
    if error["detail"] is not None:
        _require_non_empty_string(error["detail"], "error.detail")
    if error["stage_id"] is not None:
        _require_member(error["stage_id"], set(STAGE_IDS), "error.stage_id")
    _require_bool(error["recoverable"], "error.recoverable")
    _require_bool(error["can_resume"], "error.can_resume")
    if error["can_resume"] and not error["recoverable"]:
        raise ValueError("error.can_resume requires error.recoverable")
    _validate_artifact_refs(error["artifact_refs"], "error.artifact_refs")


def _validate_final_report_metadata(value: Any, run_id: str) -> None:
    if not isinstance(value, dict):
        raise ValueError("final_report must be an object")
    required = set(final_report_metadata_json_schema()["required"])
    if set(value) != required:
        raise ValueError("final_report fields do not match contract")
    _require_bool(value["available"], "final_report.available")
    if value["available"]:
        _require_non_empty_string(value["report_id"], "final_report.report_id")
        _require_non_empty_string(value["generated_at"], "final_report.generated_at")
        _require_member(value["format"], FINAL_REPORT_FORMATS, "final_report.format")
        _require_equal(value["href"], f"/runtime-runs/{run_id}/report", "final_report.href")
    else:
        if any(value[field] is not None for field in ["report_id", "generated_at", "format", "href"]):
            raise ValueError("unavailable final_report must use null metadata")


def _validate_status_invariants(data: dict[str, Any]) -> None:
    status = data["status"]
    stages_by_id = {stage["stage_id"]: stage for stage in data["stages"]}
    active_stage_ids = [stage["stage_id"] for stage in data["stages"] if stage["status"] == "active"]

    if status == "created":
        _require_no_current_or_active_stage(data, active_stage_ids)
    elif status == "discovering_sources":
        _require_single_active_stage(data, active_stage_ids, "source_discovery")
    elif status == "awaiting_source_confirmation":
        _require_single_active_stage(data, active_stage_ids, "source_confirmation")
        _require_stage_status(stages_by_id, "source_discovery", "completed")
        if data["source_confirmation"]["status"] not in {"ready_for_review", "partially_rejected", "retrying"}:
            raise ValueError("awaiting_source_confirmation requires a reviewable source confirmation state")
    elif status == "analyzing":
        if data["current_stage"] not in STAGE_IDS[2:]:
            raise ValueError("analyzing current_stage must be cache_lookup through report_generation")
        _require_single_active_stage(data, active_stage_ids, data["current_stage"])
        _require_stage_status(stages_by_id, "source_confirmation", "completed")
    elif status == "failed":
        _require_no_current_or_active_stage(data, active_stage_ids)
        if data["error"] is None:
            raise ValueError("failed RuntimeRunView requires error")
        if not any(stage["status"] == "failed" for stage in data["stages"]):
            raise ValueError("failed RuntimeRunView requires a failed stage")
        if data["recoverable"] != data["error"]["recoverable"] or data["can_resume"] != data["error"]["can_resume"]:
            raise ValueError("failed RuntimeRunView recoverability must match error")
        actions = {action["action"] for action in data["allowed_actions"]}
        if data["can_resume"] and "resume_run" not in actions:
            raise ValueError("recoverable failed RuntimeRunView requires resume_run action")
        if not data["can_resume"] and "resume_run" in actions:
            raise ValueError("non-resumable failed RuntimeRunView cannot expose resume_run")
    elif status == "completed":
        _require_no_current_or_active_stage(data, active_stage_ids)
        _require_stage_status(stages_by_id, "report_generation", "completed")
        if not data["final_report"]["available"]:
            raise ValueError("completed RuntimeRunView requires final_report.available")
        if data["error"] is not None:
            raise ValueError("completed RuntimeRunView error must be null")
    elif status == "cancelled":
        _require_no_current_or_active_stage(data, active_stage_ids)
        if not any(stage["status"] == "cancelled" for stage in data["stages"]):
            raise ValueError("cancelled RuntimeRunView requires a cancelled stage")
    if status != "failed" and (data["recoverable"] or data["can_resume"]):
        raise ValueError("only failed RuntimeRunView may be recoverable or resumable")
    if data["can_resume"] and not data["recoverable"]:
        raise ValueError("can_resume requires recoverable")
    if status != "completed" and data["final_report"]["available"]:
        raise ValueError("final_report.available is only valid after completion")


def _require_no_current_or_active_stage(data: dict[str, Any], active_stage_ids: list[str]) -> None:
    if data["current_stage"] is not None:
        raise ValueError(f"{data['status']} RuntimeRunView requires current_stage null")
    if active_stage_ids:
        raise ValueError(f"{data['status']} RuntimeRunView cannot have active stages")


def _require_single_active_stage(data: dict[str, Any], active_stage_ids: list[str], expected_stage: str) -> None:
    if data["current_stage"] != expected_stage:
        raise ValueError(f"{data['status']} RuntimeRunView current_stage must be {expected_stage}")
    if active_stage_ids != [expected_stage]:
        raise ValueError(f"{data['status']} RuntimeRunView requires exactly one active stage: {expected_stage}")


def _require_stage_status(stages_by_id: dict[str, dict[str, Any]], stage_id: str, status: str) -> None:
    if stages_by_id[stage_id]["status"] != status:
        raise ValueError(f"{stage_id} must be {status}")


def _validate_artifact_refs(value: Any, field_name: str) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    for ref in value:
        if not isinstance(ref, dict):
            raise ValueError(f"{field_name} must contain objects")
        if "artifact_id" in ref:
            _require_non_empty_string(ref["artifact_id"], f"{field_name}.artifact_id")
        if "sha256" in ref and not re.fullmatch(r"[0-9a-f]{64}", str(ref["sha256"])):
            raise ValueError(f"{field_name}.sha256 must be a 64-character lowercase hex digest")


def _validate_resume_eligibility(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("resume_eligibility must be an object")
    required = {"eligible", "latest_completed_stage", "next_stage", "reason"}
    if set(value) != required:
        raise ValueError("resume_eligibility fields do not match contract")
    _require_bool(value["eligible"], "resume_eligibility.eligible")
    for field in ["latest_completed_stage", "next_stage"]:
        if value[field] is not None:
            _require_member(value[field], set(STAGE_IDS), f"resume_eligibility.{field}")
    if value["reason"] is not None:
        _require_non_empty_string(value["reason"], "resume_eligibility.reason")


def _validate_cache_trace(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("cache_trace must be an object")
    required = {"outcome", "stage_record", "warnings", "invalidity_traces", "artifact_refs"}
    if set(value) != required:
        raise ValueError("cache_trace fields do not match contract")
    _require_non_empty_string(value["outcome"], "cache_trace.outcome")
    _validate_stage(value["stage_record"])
    if value["stage_record"]["stage_id"] != "cache_lookup":
        raise ValueError("cache_trace.stage_record must be cache_lookup")
    _validate_warning_list(value["warnings"], "cache_trace.warnings")
    _validate_cache_invalidity_traces(value["invalidity_traces"])
    _validate_audit_artifact_ref_list(value["artifact_refs"], "cache_trace.artifact_refs", allow_empty=True)


def _validate_cache_invalidity_traces(value: Any) -> None:
    if not isinstance(value, list):
        raise ValueError("cache_trace.invalidity_traces must be a list")
    allowed_reasons = {
        "manual_invalidation",
        "missing_body",
        "hash_mismatch",
        "tampered_body",
        "identity_mismatch",
        "unknown",
    }
    for trace in value:
        if not isinstance(trace, dict):
            raise ValueError("cache_trace.invalidity_traces must contain objects")
        required = {
            "source_slot_role",
            "artifact_id",
            "validity_status",
            "quality_status",
            "invalidity_reason",
        }
        if set(trace) != required:
            raise ValueError("cache_trace.invalidity_trace fields do not match contract")
        _require_member(trace["source_slot_role"], SOURCE_SLOT_ROLES, "cache_trace.invalidity_trace.source_slot_role")
        _require_non_empty_string(trace["artifact_id"], "cache_trace.invalidity_trace.artifact_id")
        _require_non_empty_string(trace["validity_status"], "cache_trace.invalidity_trace.validity_status")
        _require_non_empty_string(trace["quality_status"], "cache_trace.invalidity_trace.quality_status")
        _require_member(trace["invalidity_reason"], allowed_reasons, "cache_trace.invalidity_trace.invalidity_reason")


def _validate_artifact_manifest(value: Any, run_id: str) -> None:
    if not isinstance(value, dict) or set(value) != {"refs", "missing_artifacts"}:
        raise ValueError("artifact_manifest fields do not match contract")
    _validate_audit_artifact_ref_list(value["refs"], "artifact_manifest.refs", allow_empty=True, run_id=run_id)
    if not isinstance(value["missing_artifacts"], list):
        raise ValueError("artifact_manifest.missing_artifacts must be a list")
    for missing in value["missing_artifacts"]:
        if not isinstance(missing, dict):
            raise ValueError("artifact_manifest.missing_artifacts must contain objects")
        for field in ["kind", "reason"]:
            _require_non_empty_string(missing.get(field), f"artifact_manifest.missing_artifacts.{field}")
        _require_non_negative_int(missing.get("expected_count"), "artifact_manifest.missing_artifacts.expected_count")
        _require_non_negative_int(missing.get("actual_count"), "artifact_manifest.missing_artifacts.actual_count")


def _validate_detector_audit(value: Any, run_id: str) -> None:
    if not isinstance(value, dict):
        raise ValueError("detector_audit must be an object")
    required = {"mode", "packet_artifact_refs", "assessment_artifact_refs"}
    if set(value) != required:
        raise ValueError("detector_audit fields do not match contract")
    if not isinstance(value["mode"], dict) or not value["mode"]:
        raise ValueError("detector_audit.mode must be a non-empty object")
    _validate_audit_artifact_ref_list(
        value["packet_artifact_refs"],
        "detector_audit.packet_artifact_refs",
        allow_empty=True,
        run_id=run_id,
    )
    _validate_audit_artifact_ref_list(
        value["assessment_artifact_refs"],
        "detector_audit.assessment_artifact_refs",
        allow_empty=True,
        run_id=run_id,
    )


def _validate_audit_artifact_group(value: Any, field_name: str, run_id: str) -> None:
    if not isinstance(value, dict) or set(value) != {"artifact_refs"}:
        raise ValueError(f"{field_name} fields do not match contract")
    _validate_audit_artifact_ref_list(value["artifact_refs"], f"{field_name}.artifact_refs", allow_empty=True, run_id=run_id)


def _validate_audit_final_report(value: Any, run_id: str) -> None:
    if not isinstance(value, dict):
        raise ValueError("final_report must be an object")
    required = {"metadata", "artifact_refs"}
    if set(value) != required:
        raise ValueError("final_report fields do not match contract")
    _validate_final_report_metadata(value["metadata"], run_id)
    _validate_audit_artifact_ref_list(value["artifact_refs"], "final_report.artifact_refs", allow_empty=True, run_id=run_id)


def _validate_audit_artifact_ref_list(
    value: Any,
    field_name: str,
    *,
    allow_empty: bool = False,
    run_id: str | None = None,
) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    if not value and not allow_empty:
        raise ValueError(f"{field_name} must not be empty")
    for ref in value:
        _validate_audit_artifact_ref(ref, field_name, run_id=run_id)


def _validate_audit_artifact_ref(ref: Any, field_name: str, *, run_id: str | None = None) -> None:
    if not isinstance(ref, dict):
        raise ValueError(f"{field_name} must contain objects")
    required = {
        "artifact_id",
        "kind",
        "run_id",
        "sha256",
        "observed_sha256",
        "size_bytes",
        "version",
        "schema_version",
        "metadata",
        "storage",
        "body_status",
        "hash_status",
    }
    if set(ref) != required:
        raise ValueError(f"{field_name} artifact ref fields do not match contract")
    _require_non_empty_string(ref["artifact_id"], f"{field_name}.artifact_id")
    _require_non_empty_string(ref["kind"], f"{field_name}.kind")
    _require_non_empty_string(ref["run_id"], f"{field_name}.run_id")
    if run_id is not None and ref["run_id"] != run_id:
        raise ValueError(f"{field_name}.run_id must match DeveloperAuditBundle.run_id")
    if ref["sha256"] is not None and not re.fullmatch(r"[0-9a-f]{64}", str(ref["sha256"])):
        raise ValueError(f"{field_name}.sha256 must be a 64-character lowercase hex digest")
    if ref["observed_sha256"] is not None and not re.fullmatch(r"[0-9a-f]{64}", str(ref["observed_sha256"])):
        raise ValueError(f"{field_name}.observed_sha256 must be a 64-character lowercase hex digest")
    if ref["size_bytes"] is not None:
        _require_non_negative_int(ref["size_bytes"], f"{field_name}.size_bytes")
    for optional_string in ["version", "schema_version"]:
        if ref[optional_string] is not None:
            _require_non_empty_string(ref[optional_string], f"{field_name}.{optional_string}")
    if not isinstance(ref["metadata"], dict):
        raise ValueError(f"{field_name}.metadata must be an object")
    if ref["storage"] is not None:
        if not isinstance(ref["storage"], dict) or set(ref["storage"]) != {"type", "path"}:
            raise ValueError(f"{field_name}.storage fields do not match contract")
        _require_equal(ref["storage"]["type"], "filesystem", f"{field_name}.storage.type")
        _require_non_empty_string(ref["storage"]["path"], f"{field_name}.storage.path")
    _require_member(ref["body_status"], ARTIFACT_BODY_STATUSES, f"{field_name}.body_status")
    _require_member(ref["hash_status"], ARTIFACT_HASH_STATUSES, f"{field_name}.hash_status")
    if ref["body_status"] == "present" and ref["hash_status"] == "unavailable":
        raise ValueError(f"{field_name}.hash_status must be verified or mismatch when body is present")


def _validate_fixture_meta(value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise ValueError("_fixture_meta must be an object")
    if "scenario" in value:
        _require_non_empty_string(value["scenario"], "_fixture_meta.scenario")
    if "step" in value:
        _require_non_negative_int(value["step"], "_fixture_meta.step")


def _scan_public_payload(value: Any, *, path: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            normalized_key = key_text.lower()
            if normalized_key in _PROHIBITED_PUBLIC_KEYS:
                raise ValueError(f"public runtime payload exposes prohibited key at {path}.{key_text}")
            _scan_public_payload(item, path=f"{path}.{key_text}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_public_payload(item, path=f"{path}[{index}]")
    elif isinstance(value, str):
        lowered = value.lower()
        for pattern in _PROHIBITED_PUBLIC_TEXT:
            if pattern.search(lowered):
                raise ValueError(f"public runtime payload exposes prohibited text at {path}")


def _scan_developer_audit_payload(value: Any, *, path: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            if key_text.lower() in _PROHIBITED_AUDIT_KEYS:
                raise ValueError(f"DeveloperAuditBundle exposes prohibited key at {path}.{key_text}")
            _scan_developer_audit_payload(item, path=f"{path}.{key_text}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_developer_audit_payload(item, path=f"{path}[{index}]")


def _contract_dict(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return copy.deepcopy(value)
    return value


def _require_equal(value: Any, expected: Any, field_name: str) -> None:
    if value != expected:
        raise ValueError(f"{field_name} must be {expected!r}")


def _require_bool(value: Any, field_name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")


def _require_int(value: Any, field_name: str) -> None:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")


def _require_non_negative_int(value: Any, field_name: str) -> None:
    _require_int(value, field_name)
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")


def _require_non_empty_string(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _require_member(value: Any, allowed: set[str], field_name: str) -> None:
    if value not in allowed:
        raise ValueError(f"{field_name} must be one of {sorted(allowed)}")


def stable_json_dumps(payload: Any) -> str:
    return json.dumps(_contract_dict(payload), indent=2, sort_keys=True) + "\n"
