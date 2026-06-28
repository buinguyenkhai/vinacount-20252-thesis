from __future__ import annotations

import hashlib
import json
import os
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from dataclasses import field
from numbers import Real
from typing import Any, Callable, Protocol

from vinacount.env_loader import load_dotenv_if_available
from vinacount.filing_package import FilingPackage
from vinacount.raw_extraction_artifact import RawExtractionArtifact
from vinacount.report_artifact_cache import ReportMemoryCacheCompatibility


LLM_NORMALIZER_OUTPUT_SCHEMA_VERSION = "raw_ocr_llm_normalization.v1"
LLM_NORMALIZER_VERSION = "llm_raw_ocr_normalizer.v1"
ALLOWED_UNITS = {"vnd", "thousand_vnd", "million_vnd", "billion_vnd"}
ALLOWED_REPORT_PROFILES = {"standard_corporate", "credit_institution", "securities", "insurance"}
DEEPSEEK_RAW_OCR_NORMALIZER_URL = "https://api.deepseek.com/chat/completions"
RAW_OCR_LLM_NORMALIZER_ENABLED_ENV = "VINACOUNT_RAW_OCR_LLM_NORMALIZER_ENABLED"
RAW_OCR_LLM_NORMALIZER_TIMEOUT_ENV = "VINACOUNT_RAW_OCR_LLM_NORMALIZER_TIMEOUT_SECONDS"
RAW_OCR_LLM_NORMALIZER_MAX_TOKENS_ENV = "VINACOUNT_RAW_OCR_LLM_NORMALIZER_MAX_TOKENS"
DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"
DEFAULT_DEEPSEEK_NORMALIZER_MODEL = "deepseek-v4-flash"
DEEPSEEK_NORMALIZER_PROMPT_VERSION = "raw_ocr_normalizer.vi.v1"

RawOcrNormalizerTransport = Callable[..., dict[str, Any]]


class RawOcrLlmNormalizationError(ValueError):
    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


@dataclass(frozen=True)
class LlmRawOcrNormalizerIdentity:
    provider: str
    model: str
    schema_version: str
    prompt_config_hash: str
    normalizer_version: str = LLM_NORMALIZER_VERSION

    def audit_metadata(
        self,
        *,
        extraction_method: str,
        extraction_version: str,
    ) -> dict[str, Any]:
        return {
            "strategy": "llm_assisted",
            "provider": self.provider,
            "model": self.model,
            "schema_version": self.schema_version,
            "prompt_config_hash": self.prompt_config_hash,
            "normalizer_version": self.normalizer_version,
            "extraction_method": extraction_method,
            "extraction_version": extraction_version,
        }


class RawOcrLlmNormalizer(Protocol):
    identity: LlmRawOcrNormalizerIdentity

    def normalize(
        self,
        *,
        raw_artifact: dict[str, Any],
        filing_package: dict[str, Any],
        source_document_id: str,
    ) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class DeepSeekRawOcrLlmNormalizer:
    api_key: str
    timeout_seconds: float = 180.0
    max_tokens: int = 8192
    transport: RawOcrNormalizerTransport = field(default=None)  # type: ignore[assignment]
    model: str = field(init=False, default=DEFAULT_DEEPSEEK_NORMALIZER_MODEL)
    identity: LlmRawOcrNormalizerIdentity = field(init=False)

    def __post_init__(self) -> None:
        if self.transport is None:
            object.__setattr__(self, "transport", _post_deepseek_raw_ocr_normalization)
        object.__setattr__(
            self,
            "identity",
            LlmRawOcrNormalizerIdentity(
                provider="deepseek",
                model=DEFAULT_DEEPSEEK_NORMALIZER_MODEL,
                schema_version=LLM_NORMALIZER_OUTPUT_SCHEMA_VERSION,
                prompt_config_hash=_normalizer_prompt_config_hash(DEFAULT_DEEPSEEK_NORMALIZER_MODEL),
            ),
        )

    def normalize(
        self,
        *,
        raw_artifact: dict[str, Any],
        filing_package: dict[str, Any],
        source_document_id: str,
    ) -> dict[str, Any]:
        if not self.api_key.strip():
            raise RawOcrLlmNormalizationError(
                "normalizer_unconfigured",
                f"{DEEPSEEK_API_KEY_ENV} is required for DeepSeek raw OCR normalization.",
            )
        provider_input = _provider_safe_normalizer_input(
            raw_artifact=raw_artifact,
            filing_package=filing_package,
            source_document_id=source_document_id,
            identity=self.identity,
        )
        payload = {
            "model": DEFAULT_DEEPSEEK_NORMALIZER_MODEL,
            "messages": [
                {"role": "system", "content": _normalizer_system_prompt()},
                {
                    "role": "user",
                    "content": json.dumps(provider_input, ensure_ascii=False, sort_keys=True),
                },
            ],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"},
        }
        try:
            response = self.transport(
                url=DEEPSEEK_RAW_OCR_NORMALIZER_URL,
                payload=payload,
                timeout_seconds=self.timeout_seconds,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            choice = response["choices"][0]
            if choice.get("finish_reason") == "length":
                raise RawOcrLlmNormalizationError(
                    "provider_output_truncated",
                    "DeepSeek raw OCR normalizer output exceeded the configured token budget.",
                )
            message = choice["message"]
            content = message["content"]
            output = content if isinstance(content, dict) else json.loads(content)
        except RawOcrLlmNormalizationError:
            raise
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise RawOcrLlmNormalizationError(
                "provider_response_invalid",
                "DeepSeek raw OCR normalizer returned an invalid response.",
            ) from error
        except ValueError as error:
            raise RawOcrLlmNormalizationError(
                "provider_response_invalid",
                "DeepSeek raw OCR normalizer returned an invalid response.",
            ) from error
        except (TimeoutError, OSError) as error:
            raise RawOcrLlmNormalizationError(
                "provider_unavailable",
                "DeepSeek raw OCR normalizer was unavailable.",
            ) from error
        if not isinstance(output, dict):
            raise RawOcrLlmNormalizationError(
                "provider_response_invalid",
                "DeepSeek raw OCR normalizer output must be a JSON object.",
            )
        return output


def make_deepseek_raw_ocr_normalizer_from_env() -> DeepSeekRawOcrLlmNormalizer | None:
    load_dotenv_if_available()
    if os.environ.get(RAW_OCR_LLM_NORMALIZER_ENABLED_ENV, "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return None
    api_key = os.environ.get(DEEPSEEK_API_KEY_ENV, "").strip()
    if not api_key:
        raise ValueError(
            f"{DEEPSEEK_API_KEY_ENV} is required when {RAW_OCR_LLM_NORMALIZER_ENABLED_ENV} is enabled"
        )
    return DeepSeekRawOcrLlmNormalizer(
        api_key=api_key,
        timeout_seconds=_positive_float_env(RAW_OCR_LLM_NORMALIZER_TIMEOUT_ENV, 180.0),
        max_tokens=_positive_int_env(RAW_OCR_LLM_NORMALIZER_MAX_TOKENS_ENV, 8192),
    )


def report_memory_cache_compatibility_for_llm_normalizer(
    identity: LlmRawOcrNormalizerIdentity,
    *,
    extraction_method: str,
    extraction_version: str,
) -> ReportMemoryCacheCompatibility:
    cache_identity = hashlib.sha256(
        _stable_json_bytes(
            {
                "provider": identity.provider,
                "model": identity.model,
                "schema_version": identity.schema_version,
                "prompt_config_hash": identity.prompt_config_hash,
                "normalizer_version": identity.normalizer_version,
                "extraction_method": extraction_method,
                "extraction_version": extraction_version,
            }
        )
    ).hexdigest()
    return ReportMemoryCacheCompatibility(
        mapper_version=identity.normalizer_version,
        normalization_version=(
            "llm_raw_ocr_normalization"
            f":provider={identity.provider}"
            f":model={identity.model}"
            f":schema={identity.schema_version}"
            f":prompt_config_hash={identity.prompt_config_hash}"
            f":identity={cache_identity}"
        ),
        extraction_version=f"{extraction_version}:method={extraction_method}",
    )


def validate_llm_normalizer_output(
    output: dict[str, Any],
    *,
    artifact: RawExtractionArtifact,
    filing_package: FilingPackage,
    identity: LlmRawOcrNormalizerIdentity,
) -> dict[str, Any]:
    if not isinstance(output, dict):
        raise RawOcrLlmNormalizationError("schema_invalid", "LLM normalizer output must be a JSON object.")
    if set(output) != {"schema_version", "normalizer_provenance", "extraction_candidates"}:
        raise RawOcrLlmNormalizationError(
            "schema_invalid",
            "LLM normalizer output must contain only schema_version, normalizer_provenance, and extraction_candidates.",
        )
    if output["schema_version"] != LLM_NORMALIZER_OUTPUT_SCHEMA_VERSION:
        raise RawOcrLlmNormalizationError("schema_invalid", "LLM normalizer schema_version is not supported.")
    provenance = output["normalizer_provenance"]
    if not isinstance(provenance, dict):
        raise RawOcrLlmNormalizationError("weak_provenance", "LLM normalizer provenance must be structured.")
    expected = identity.audit_metadata(
        extraction_method=str(artifact.raw.get("extraction_method") or ""),
        extraction_version=str(artifact.raw.get("extraction_version") or ""),
    )
    for key, expected_value in expected.items():
        if provenance.get(key) != expected_value:
            raise RawOcrLlmNormalizationError(
                "weak_provenance",
                f"LLM normalizer provenance field {key} did not match configured identity.",
            )
    candidates = output["extraction_candidates"]
    if not isinstance(candidates, dict):
        raise RawOcrLlmNormalizationError("schema_invalid", "extraction_candidates must be an object.")
    _validate_metadata(candidates.get("metadata"), filing_package=filing_package, identity=identity)
    _validate_statement_tables(
        candidates.get("statement_tables"),
        artifact=artifact,
        filing_package=filing_package,
        source_document_id=artifact.source_document_id,
        identity=identity,
    )
    if not isinstance(candidates.get("sections", []), list):
        raise RawOcrLlmNormalizationError("schema_invalid", "extraction_candidates.sections must be a list.")
    _validate_non_table_evidence(
        candidates.get("notes", []),
        evidence_kind="note",
        id_field="note_id",
        source_document_id=artifact.source_document_id,
        identity=identity,
    )
    _validate_non_table_evidence(
        candidates.get("variance_explanations", []),
        evidence_kind="variance explanation",
        id_field="span_id",
        source_document_id=artifact.source_document_id,
        identity=identity,
    )
    _validate_evidence_surface_status(
        candidates.get("evidence_surface_status", []),
        source_document_id=artifact.source_document_id,
        identity=identity,
    )
    if not isinstance(candidates.get("variance_explanations", []), list):
        raise RawOcrLlmNormalizationError(
            "schema_invalid",
            "extraction_candidates.variance_explanations must be a list.",
        )
    return candidates


def _validate_metadata(
    metadata: Any,
    *,
    filing_package: FilingPackage,
    identity: LlmRawOcrNormalizerIdentity,
) -> None:
    if not isinstance(metadata, dict):
        raise RawOcrLlmNormalizationError("schema_invalid", "extraction_candidates.metadata must be an object.")
    required = {
        "period",
        "report_profile",
        "report_basis",
        "business_context_tags",
        "report_assurance_type",
        "currency",
        "unit",
        "language",
        "report_period_type",
        "normalizer_version",
    }
    missing = required - metadata.keys()
    if missing:
        raise RawOcrLlmNormalizationError(
            "missing_required_fields",
            f"extraction_candidates.metadata missing required fields: {sorted(missing)}.",
        )
    if metadata["period"] != filing_package.period:
        raise RawOcrLlmNormalizationError("ambiguous_period", "LLM normalizer period did not match source package.")
    if metadata["report_basis"] != filing_package.report_basis:
        raise RawOcrLlmNormalizationError("basis_mismatch", "LLM normalizer report_basis did not match source package.")
    if metadata["report_profile"] not in ALLOWED_REPORT_PROFILES:
        raise RawOcrLlmNormalizationError("unsupported_report_profile", "LLM normalizer report_profile is not supported.")
    if metadata["currency"] != "VND" or metadata["unit"] not in ALLOWED_UNITS:
        raise RawOcrLlmNormalizationError("unit_ambiguity", "LLM normalizer must emit an accepted VND unit.")
    if metadata["report_period_type"] != "quarterly":
        raise RawOcrLlmNormalizationError("ambiguous_period", "LLM normalizer must emit quarterly period type.")
    if metadata["normalizer_version"] != identity.normalizer_version:
        raise RawOcrLlmNormalizationError("weak_provenance", "LLM normalizer metadata must identify the normalizer.")
    if not isinstance(metadata["business_context_tags"], list) or not metadata["business_context_tags"]:
        raise RawOcrLlmNormalizationError(
            "missing_required_fields",
            "LLM normalizer must emit business_context_tags.",
        )


def _validate_statement_tables(
    tables: Any,
    *,
    artifact: RawExtractionArtifact,
    filing_package: FilingPackage,
    source_document_id: str,
    identity: LlmRawOcrNormalizerIdentity,
) -> None:
    if not isinstance(tables, list) or not tables:
        raise RawOcrLlmNormalizationError(
            "missing_required_fields",
            "LLM normalizer must emit at least one statement table.",
        )
    raw_tables_by_id = {
        raw_table.get("raw_table_id"): raw_table
        for raw_table in artifact.raw.get("raw_tables", [])
        if isinstance(raw_table, dict) and raw_table.get("raw_table_id")
    }
    for table in tables:
        if not isinstance(table, dict):
            raise RawOcrLlmNormalizationError("schema_invalid", "statement table entries must be objects.")
        for field in ["table_id", "section_id", "table_type", "period_basis", "source_document_id", "rows"]:
            if field not in table:
                raise RawOcrLlmNormalizationError("missing_required_fields", f"statement table missing {field}.")
        if table["source_document_id"] != source_document_id:
            raise RawOcrLlmNormalizationError("weak_provenance", "statement table source_document_id mismatch.")
        if not isinstance(table["rows"], list) or not table["rows"]:
            raise RawOcrLlmNormalizationError("missing_required_fields", "statement table rows must not be empty.")
        for row in table["rows"]:
            _validate_row(
                row,
                filing_package=filing_package,
                source_document_id=source_document_id,
                identity=identity,
                raw_tables_by_id=raw_tables_by_id,
            )


def _validate_row(
    row: Any,
    *,
    filing_package: FilingPackage,
    source_document_id: str,
    identity: LlmRawOcrNormalizerIdentity,
    raw_tables_by_id: dict[str, dict[str, Any]],
) -> None:
    if not isinstance(row, dict):
        raise RawOcrLlmNormalizationError("schema_invalid", "statement rows must be objects.")
    for field in ["row_id", "standard_account", "account_group", "label", "original_label", "cells", "evidence_provenance"]:
        if field not in row:
            raise RawOcrLlmNormalizationError("missing_required_fields", f"statement row missing {field}.")
    provenance = row["evidence_provenance"]
    if not isinstance(provenance, dict):
        raise RawOcrLlmNormalizationError("weak_provenance", "statement row provenance must be structured.")
    if provenance.get("source_document_id") != source_document_id:
        raise RawOcrLlmNormalizationError("weak_provenance", "statement row provenance source mismatch.")
    raw_table_id = provenance.get("raw_table_id")
    if not raw_table_id:
        raise RawOcrLlmNormalizationError("weak_provenance", "statement row provenance must include raw_table_id.")
    raw_table = raw_tables_by_id.get(raw_table_id)
    if raw_table is None:
        raise RawOcrLlmNormalizationError("weak_provenance", "statement row provenance raw_table_id was not found.")
    raw_row_index = provenance.get("raw_row_index")
    raw_rows = raw_table.get("cells")
    if (
        not isinstance(raw_row_index, int)
        or isinstance(raw_row_index, bool)
        or not isinstance(raw_rows, list)
        or raw_row_index < 0
        or raw_row_index >= len(raw_rows)
    ):
        raise RawOcrLlmNormalizationError("weak_provenance", "statement row provenance raw_row_index was invalid.")
    if provenance.get("normalizer_version") != identity.normalizer_version:
        raise RawOcrLlmNormalizationError("weak_provenance", "statement row provenance normalizer_version mismatch.")
    confidence = provenance.get("confidence")
    if not isinstance(confidence, Real) or isinstance(confidence, bool) or confidence < 0.70:
        raise RawOcrLlmNormalizationError("weak_provenance", "statement row provenance confidence is too weak.")
    cells = row["cells"]
    if not isinstance(cells, list) or not cells:
        raise RawOcrLlmNormalizationError("missing_required_fields", "statement row cells must not be empty.")
    for cell in cells:
        if not isinstance(cell, dict):
            raise RawOcrLlmNormalizationError("schema_invalid", "statement row cells must be objects.")
        for field in ["cell_id", "period", "value", "source_document_id"]:
            if field not in cell:
                raise RawOcrLlmNormalizationError("missing_required_fields", f"statement cell missing {field}.")
        if cell["period"] != filing_package.period:
            raise RawOcrLlmNormalizationError("ambiguous_period", "statement cell period did not match source package.")
        if cell["source_document_id"] != source_document_id:
            raise RawOcrLlmNormalizationError("weak_provenance", "statement cell source_document_id mismatch.")
        if not isinstance(cell["value"], Real) or isinstance(cell["value"], bool):
            raise RawOcrLlmNormalizationError("missing_required_fields", "statement cell value must be numeric.")


def _validate_non_table_evidence(
    values: Any,
    *,
    evidence_kind: str,
    id_field: str,
    source_document_id: str,
    identity: LlmRawOcrNormalizerIdentity,
) -> None:
    if not isinstance(values, list):
        raise RawOcrLlmNormalizationError("schema_invalid", f"extraction_candidates.{evidence_kind}s must be a list.")
    for item in values:
        if not isinstance(item, dict):
            raise RawOcrLlmNormalizationError("schema_invalid", f"{evidence_kind} entries must be objects.")
        if not isinstance(item.get(id_field), str) or not item[id_field].strip():
            raise RawOcrLlmNormalizationError("missing_required_fields", f"{evidence_kind} must include {id_field}.")
        if item.get("source_document_id") != source_document_id:
            raise RawOcrLlmNormalizationError("weak_provenance", f"{evidence_kind} source_document_id mismatch.")
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            raise RawOcrLlmNormalizationError("missing_required_fields", f"{evidence_kind} must preserve source text.")
        provenance = item.get("evidence_provenance")
        if not isinstance(provenance, dict):
            raise RawOcrLlmNormalizationError("weak_provenance", f"{evidence_kind} provenance must be structured.")
        if provenance.get("source_document_id") != source_document_id:
            raise RawOcrLlmNormalizationError("weak_provenance", f"{evidence_kind} provenance source mismatch.")
        if provenance.get("normalizer_version") != identity.normalizer_version:
            raise RawOcrLlmNormalizationError("weak_provenance", f"{evidence_kind} provenance normalizer mismatch.")
        confidence = provenance.get("confidence")
        if not isinstance(confidence, Real) or isinstance(confidence, bool) or confidence < 0.70:
            raise RawOcrLlmNormalizationError("weak_provenance", f"{evidence_kind} provenance confidence is too weak.")


def _validate_evidence_surface_status(
    values: Any,
    *,
    source_document_id: str,
    identity: LlmRawOcrNormalizerIdentity,
) -> None:
    if not isinstance(values, list):
        raise RawOcrLlmNormalizationError("schema_invalid", "extraction_candidates.evidence_surface_status must be a list.")
    allowed_surfaces = {
        "notes",
        "variance_explanations",
        "related_party_evidence",
        "accounting_policy_evidence",
        "extraction_quality",
    }
    allowed_states = {
        "absent_in_source",
        "not_applicable",
        "unsupported_by_extraction_path",
        "not_extracted_yet",
        "ambiguous_failed_closed",
    }
    for status in values:
        if not isinstance(status, dict):
            raise RawOcrLlmNormalizationError("schema_invalid", "evidence surface status entries must be objects.")
        if status.get("surface") not in allowed_surfaces or status.get("state") not in allowed_states:
            raise RawOcrLlmNormalizationError("schema_invalid", "evidence surface status used an unsupported value.")
        if status.get("source_document_id") != source_document_id:
            raise RawOcrLlmNormalizationError("weak_provenance", "evidence surface status source mismatch.")
        if status.get("producer_version") != identity.normalizer_version:
            raise RawOcrLlmNormalizationError("weak_provenance", "evidence surface status producer mismatch.")
        if not isinstance(status.get("evidence_ref"), str) or not status["evidence_ref"].strip():
            raise RawOcrLlmNormalizationError("weak_provenance", "evidence surface status must include evidence_ref.")


def _stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _provider_safe_normalizer_input(
    *,
    raw_artifact: dict[str, Any],
    filing_package: dict[str, Any],
    source_document_id: str,
    identity: LlmRawOcrNormalizerIdentity,
) -> dict[str, Any]:
    raw_tables = raw_artifact.get("raw_tables")
    if not isinstance(raw_tables, list) or not raw_tables:
        raise RawOcrLlmNormalizationError(
            "raw_tables_missing",
            "DeepSeek raw OCR normalization requires provider-normalized raw tables.",
        )
    company = filing_package.get("company")
    if not isinstance(company, dict):
        company = {}
    provenance = identity.audit_metadata(
        extraction_method=str(raw_artifact.get("extraction_method") or ""),
        extraction_version=str(raw_artifact.get("extraction_version") or ""),
    )
    period = filing_package.get("period")
    report_basis = filing_package.get("report_basis")
    profile_hint = _report_profile_hint_from_raw_tables(raw_tables)
    return {
        "task": "normalize_vietnamese_quarterly_financial_statement_raw_ocr_to_json",
        "prompt_version": DEEPSEEK_NORMALIZER_PROMPT_VERSION,
        "source": {
            "source_document_id": source_document_id,
            "company_name": company.get("name"),
            "ticker": company.get("ticker"),
            "period": period,
            "report_basis": report_basis,
            "report_profile_hint": profile_hint,
            "extraction_method": raw_artifact.get("extraction_method"),
            "extraction_version": raw_artifact.get("extraction_version"),
        },
        "required_normalizer_provenance": provenance,
        "raw_tables": [
            {
                key: table[key]
                for key in ("raw_table_id", "title", "page_number", "cells")
                if key in table
            }
            for table in raw_tables
            if isinstance(table, dict)
        ],
        "output_contract": {
            "schema_version": LLM_NORMALIZER_OUTPUT_SCHEMA_VERSION,
            "top_level_fields": [
                "schema_version",
                "normalizer_provenance",
                "extraction_candidates",
            ],
            "required_metadata": [
                "period",
                "report_profile",
                "report_basis",
                "business_context_tags",
                "report_assurance_type",
                "currency",
                "unit",
                "language",
                "report_period_type",
                "normalizer_version",
            ],
            "allowed_units": sorted(ALLOWED_UNITS),
            "allowed_report_profiles": sorted(ALLOWED_REPORT_PROFILES),
            "report_profile_rules": {
                "insurance": [
                    "Use only when visible labels identify insurance operations, insurance premium revenue, claims, or technical reserves."
                ],
                "securities": [
                    "Use only when visible labels identify securities company operations, FVTPL/AFS/HTM securities, margin lending, brokerage, underwriting, or proprietary trading."
                ],
                "credit_institution": [
                    "Use only when visible labels identify a bank or credit institution, customer loans, deposits, interest income, credit loss provisions, or VAS bank forms."
                ],
                "standard_corporate": [
                    "Use when no insurance, securities, or credit-institution markers are visible."
                ],
            },
            "max_statement_rows": 12,
            "requested_standard_accounts": [
                "cash_and_cash_equivalents",
                "trade_receivables",
                "inventory",
                "fixed_assets",
                "insurance_premium_receivables",
                "technical_reserves",
                "customer_loans",
                "customer_deposits",
                "interest_income",
                "credit_loss_provision_expense",
                "fvtpl_financial_assets",
                "margin_lending",
                "brokerage_revenue",
                "insurance_premium_revenue",
                "insurance_claims_expense",
                "revenue",
                "profit_after_tax",
                "operating_cash_flow",
                "total_assets",
                "total_liabilities",
                "equity",
            ],
            "required_statement_table_fields": [
                "table_id",
                "section_id",
                "table_type",
                "title",
                "period_basis",
                "source_document_id",
                "rows",
            ],
            "required_row_fields": [
                "row_id",
                "standard_account",
                "account_group",
                "label",
                "original_label",
                "cells",
                "evidence_provenance",
            ],
            "required_evidence_provenance": [
                "source_document_id",
                "raw_table_id",
                "raw_row_index",
                "confidence",
                "normalizer_version",
            ],
            "evidence_provenance_field_types": {
                "source_document_id": "string copied from source.source_document_id",
                "raw_table_id": "string copied from one input raw_tables[].raw_table_id",
                "raw_row_index": "zero-based integer index into that raw table's cells array",
                "confidence": "JSON number from 0.70 through 1.0",
                "normalizer_version": f"exact string {identity.normalizer_version}",
            },
            "required_cell_fields": ["cell_id", "period", "value", "source_document_id"],
            "cell_field_types": {
                "cell_id": "non-empty string unique within the output",
                "period": f"exact string {period}; do not use a date or a Vietnamese period label",
                "value": "JSON number copied from the cited OCR row",
                "source_document_id": f"exact string {source_document_id}",
            },
            "non_table_evidence_contract": {
                "notes": "Optional source-bound snippets only; every entry requires note_id, text, source_document_id, and evidence_provenance.",
                "variance_explanations": "Optional source-bound snippets only; every entry requires span_id, text, source_document_id, and evidence_provenance.",
                "evidence_provenance": {
                    "source_document_id": f"exact string {source_document_id}",
                    "confidence": "JSON number from 0.70 through 1.0",
                    "normalizer_version": f"exact string {identity.normalizer_version}",
                },
                "related_party_evidence": "Represent as notes entries with note_type related_party_note when source-bound evidence exists; otherwise use evidence_surface_status.",
                "accounting_policy_evidence": "Represent as notes entries with note_type accounting_policy_change or generic_accounting_policy when source-bound evidence exists; otherwise use evidence_surface_status.",
            },
            "evidence_surface_status_contract": {
                "field": "evidence_surface_status",
                "surfaces": [
                    "notes",
                    "variance_explanations",
                    "related_party_evidence",
                    "accounting_policy_evidence",
                    "extraction_quality",
                ],
                "states": [
                    "absent_in_source",
                    "not_applicable",
                    "unsupported_by_extraction_path",
                    "not_extracted_yet",
                    "ambiguous_failed_closed",
                ],
                "required_fields": [
                    "surface",
                    "state",
                    "source_document_id",
                    "evidence_ref",
                    "producer",
                    "producer_version",
                    "reason_code",
                    "message",
                ],
                "producer_version": f"exact string {identity.normalizer_version}",
            },
            "json_output_example": {
                "schema_version": LLM_NORMALIZER_OUTPUT_SCHEMA_VERSION,
                "normalizer_provenance": provenance,
                "extraction_candidates": {
                    "metadata": {
                        "period": period,
                        "report_profile": profile_hint,
                        "report_basis": report_basis,
                        "business_context_tags": [profile_hint, "llm_raw_ocr_normalized"],
                        "report_assurance_type": "unaudited",
                        "currency": "VND",
                        "unit": "million_vnd",
                        "language": "vi",
                        "report_period_type": "quarterly",
                        "normalizer_version": identity.normalizer_version,
                        "extraction_limitations": ["llm_assisted_raw_ocr_normalization"],
                    },
                    "sections": [],
                    "statement_tables": [],
                    "notes": [],
                    "variance_explanations": [],
                    "evidence_surface_status": [
                        {
                            "surface": "related_party_evidence",
                            "state": "not_extracted_yet",
                            "source_document_id": source_document_id,
                            "evidence_ref": f"{company.get('ticker') or 'REPORT'}_{period}:LIMIT_RELATED_PARTY_NOT_EXTRACTED",
                            "producer": "llm_raw_ocr_normalizer",
                            "producer_version": identity.normalizer_version,
                            "reason_code": "not_requested_or_not_visible",
                            "message": "Related-party evidence was not extracted into source-bound ReportMemory objects.",
                        }
                    ],
                },
            },
        },
    }


def _normalizer_system_prompt() -> str:
    return (
        "You normalize Vietnamese quarterly financial-statement OCR tables. Return JSON only. "
        "Use output_contract.json_output_example for the top-level shape and metadata shape, but replace "
        "its empty statement_tables example with extracted statement tables whenever supported rows are visible. "
        "Copy required_normalizer_provenance exactly. "
        "Set report_profile to exactly one allowed_report_profiles value. Infer regulated profiles only "
        "from visible table/form labels. Use source.report_profile_hint as the expected profile unless the "
        "visible raw_tables clearly contradict it; if contradicted, return empty statement_tables so validation "
        "fails closed. If no insurance, securities, or credit-institution markers are visible, use "
        "standard_corporate. Vietnamese current-period headers such as Số cuối kỳ, Số cuối năm, "
        "Quý này, QUÝ I/II/III/IV, Năm nay, and Lũy kế từ đầu năm đến cuối kỳ này are not ambiguous "
        "when source.period is supplied; map the selected current-period values to source.period. "
        "Prefer the current-quarter column for income statement rows when both quarter and year-to-date "
        "columns are present, and prefer the current-year/YTD column for cash-flow rows when no separate "
        "quarter cash-flow column is visible. Extract only values visibly "
        "supported by raw_tables, only for source.period, and preserve each source row through raw_table_id "
        "and zero-based raw_row_index. Never invent a value, period, basis, unit, account, or provenance. "
        "raw_row_index is the Python-style array offset into the cited raw table's cells array: the first "
        "row, including a header row, is 0; the second row is 1. It is not an accounting code, page row "
        "number, display ordinal, or 1-based row number, and it must be less than the length of that "
        "raw table's cells array. "
        "If period columns, report basis, unit, row provenance, or non-table source provenance are ambiguous, "
        "return the contract with an empty statement_tables list so validation fails closed. Include sections, "
        "statement_tables, notes, variance_explanations, and evidence_surface_status in extraction_candidates. "
        "For non-table surfaces, either emit source-bound evidence with source_document_id and structured "
        "evidence_provenance, or emit a source-bound evidence_surface_status limitation; do not summarize "
        "or infer from unsupported raw OCR. Extract no more than max_statement_rows "
        "rows, prioritizing requested_standard_accounts and omitting all other rows. Numeric cell values must "
        "be JSON numbers. Every row's evidence_provenance must be a JSON object with exactly the requested "
        "typed provenance fields; never emit it as text, null, or a list. Keep labels and metadata concise."
        " Every cell period must exactly equal source.period, including its YYYY-QN format."
    )


def _report_profile_hint_from_raw_tables(raw_tables: Any) -> str:
    table_text = " ".join(
        str(cell)
        for table in raw_tables
        if isinstance(table, dict)
        for row in table.get("cells", [])
        if isinstance(row, list)
        for cell in row
    ).casefold()
    if any(
        marker in table_text
        for marker in (
            "dự phòng nghiệp vụ",
            "du phong nghiep vu",
            "phí bảo hiểm",
            "phi bao hiem",
            "doanh thu hoạt động bảo hiểm",
            "doanh thu hoat dong bao hiem",
            "bồi thường bảo hiểm",
            "boi thuong bao hiem",
        )
    ):
        return "insurance"
    if any(
        marker in table_text
        for marker in (
            "công ty chứng khoán",
            "cong ty chung khoan",
            "ủy ban chứng khoán",
            "uy ban chung khoan",
            "fvtpl",
            "môi giới chứng khoán",
            "moi gioi chung khoan",
            "cho vay margin",
            "tự doanh",
            "tu doanh",
            "lưu ký chứng khoán",
            "luu ky chung khoan",
        )
    ):
        return "securities"
    if any(
        marker in table_text
        for marker in (
            "ngân hàng",
            "ngan hang",
            "tổ chức tín dụng",
            "to chuc tin dung",
            "cho vay khách hàng",
            "cho vay khach hang",
            "tiền gửi khách hàng",
            "tien gui khach hang",
            "thu nhập lãi",
            "thu nhap lai",
        )
    ):
        return "credit_institution"
    return "standard_corporate"


def _normalizer_prompt_config_hash(model: str) -> str:
    return hashlib.sha256(
        _stable_json_bytes(
            {
                "prompt_version": DEEPSEEK_NORMALIZER_PROMPT_VERSION,
                "system_prompt": _normalizer_system_prompt(),
                "model": model,
                "schema_version": LLM_NORMALIZER_OUTPUT_SCHEMA_VERSION,
                "temperature": 0.0,
                "thinking": "disabled",
            }
        )
    ).hexdigest()


def _positive_float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive number") from error
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive number")
    return parsed


def _positive_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _post_deepseek_raw_ocr_normalization(
    *,
    url: str,
    payload: dict[str, Any],
    timeout_seconds: float,
    headers: dict[str, str],
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_text = response.read().decode("utf-8", errors="replace")
    except (TimeoutError, socket.timeout) as error:
        raise TimeoutError("raw_ocr_llm_normalizer_provider_timeout") from error
    except urllib.error.URLError as error:
        if isinstance(error.reason, (TimeoutError, socket.timeout)):
            raise TimeoutError("raw_ocr_llm_normalizer_provider_timeout") from error
        raise OSError("raw_ocr_llm_normalizer_provider_transport_error") from error
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise ValueError("raw_ocr_llm_normalizer_provider_response_invalid") from error
    if not isinstance(parsed, dict):
        raise ValueError("raw_ocr_llm_normalizer_provider_response_invalid")
    return parsed
