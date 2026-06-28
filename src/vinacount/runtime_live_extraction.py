from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any, Protocol

from vinacount.filing_package import FilingPackage, select_canonical_source, validate_filing_package
from vinacount.ocr_adapter import NanonetsOcr3DocstrangeAdapter
from vinacount.raw_extraction_artifact import RawExtractionArtifact, validate_raw_extraction_artifact
from vinacount.raw_ocr_candidate_mapper import (
    MAPPER_VERSION,
    RawOcrCandidateMappingError,
    map_raw_ocr_artifact_to_extraction_candidates,
)
from vinacount.raw_ocr_llm_normalizer import (
    RawOcrLlmNormalizationError,
    RawOcrLlmNormalizer,
    report_memory_cache_compatibility_for_llm_normalizer,
    validate_llm_normalizer_output,
)
from vinacount.report_artifact_cache import (
    RAW_EXTRACTION_ARTIFACT_SCHEMA_VERSION,
    ReportMemoryCacheCompatibility,
    RawOcrArtifactCache,
    RawOcrCacheIdentity,
    RawOcrCacheRecord,
)
from vinacount.report_memory_assembler import assemble_report_memory
from vinacount.report_model import CompanyReportSet, validate_company_report_set
from vinacount.runtime_orchestration import RuntimeStageExecutionError


class RuntimeExtractionAdapter(Protocol):
    def extract(self, run_view: dict[str, Any], audit_metadata: dict[str, Any]) -> CompanyReportSet:
        ...


class LiveNanonetsReportMemoryExtractionAdapter:
    def __init__(
        self,
        *,
        ocr_adapter: NanonetsOcr3DocstrangeAdapter,
        raw_ocr_cache: RawOcrArtifactCache | None = None,
        raw_ocr_normalizer: RawOcrLlmNormalizer | None = None,
    ) -> None:
        self._ocr_adapter = ocr_adapter
        self._raw_ocr_cache = raw_ocr_cache
        self._raw_ocr_normalizer = raw_ocr_normalizer
        self.last_source_packages_by_role: dict[str, FilingPackage] = {}
        self.last_raw_extraction_artifacts_by_role: dict[str, RawExtractionArtifact] = {}
        self.last_raw_cache_records_by_role: dict[str, RawOcrCacheRecord] = {}
        self.last_raw_cache_outcomes_by_role: dict[str, str] = {}
        self.last_normalization_metadata_by_role: dict[str, dict[str, Any]] = {}
        self.last_report_memory_cache_compatibility_by_role: dict[str, ReportMemoryCacheCompatibility] = {}

    def extract(self, run_view: dict[str, Any], audit_metadata: dict[str, Any]) -> CompanyReportSet:
        packages_by_role = audit_metadata.get("live_extraction_source_packages")
        if not isinstance(packages_by_role, dict):
            raise RuntimeStageExecutionError(
                stage_id="extraction",
                code="source_artifact_unreachable",
                message="Extraction paused because confirmed source artifacts are unavailable.",
                detail="Confirmed live source package snapshots were not available to the extraction stage.",
            )
        report_memories = {}
        self.last_source_packages_by_role = {}
        self.last_raw_extraction_artifacts_by_role = {}
        self.last_raw_cache_records_by_role = {}
        self.last_raw_cache_outcomes_by_role = {}
        self.last_normalization_metadata_by_role = {}
        self.last_report_memory_cache_compatibility_by_role = {}
        for role in ("target", "prior_year_same_quarter"):
            raw_package = packages_by_role.get(role)
            if not isinstance(raw_package, dict):
                raise RuntimeStageExecutionError(
                    stage_id="extraction",
                    code="source_artifact_unreachable",
                    message="Extraction paused because a confirmed source artifact is unavailable.",
                    detail=f"Missing confirmed source package for role {role}.",
                )
            package = validate_filing_package(copy.deepcopy(raw_package))
            self.last_source_packages_by_role[role] = package
            artifact = self._extract_raw_artifact(package, role=role)
            self.last_raw_extraction_artifacts_by_role[role] = artifact
            report_memories[role] = assemble_report_memory(
                filing_package=package,
                raw_extraction_artifacts=[artifact],
            )
        try:
            report_set = validate_company_report_set(
                f"{report_memories['target'].report_id}_VS_{report_memories['prior_year_same_quarter'].report_id}",
                report_memories["target"],
                report_memories["prior_year_same_quarter"],
            )
            return report_set
        except Exception as error:
            raise RuntimeStageExecutionError(
                stage_id="extraction",
                code="raw_extraction_invalid",
                message="Extraction paused because the extracted filing pair did not match the runtime contract.",
                detail=type(error).__name__,
            ) from error

    def _extract_raw_artifact(self, package: FilingPackage, *, role: str) -> RawExtractionArtifact:
        selection = select_canonical_source(package)
        source_document = next(
            document
            for document in package.source_documents
            if document.document_id == selection.canonical_source_document_id
        )
        _verify_source_artifact(source_document)
        cache_identity = self._cache_identity(source_document)
        if self._raw_ocr_cache is not None:
            cached = self._raw_ocr_cache.lookup(cache_identity)
            if cached is not None:
                self.last_raw_cache_records_by_role[role] = cached.record
                self.last_raw_cache_outcomes_by_role[role] = "hit"
                artifact = validate_raw_extraction_artifact(
                    _bind_cached_artifact(cached.raw_artifact, package, source_document),
                    package,
                )
                self.last_raw_extraction_artifacts_by_role[role] = artifact
                return self._require_mappable_artifact(artifact, package=package, role=role)
        try:
            artifact = self._ocr_adapter.extract_source_document(
                filing_package=package,
                source_document_id=selection.canonical_source_document_id,
            )
        except Exception as error:
            raise RuntimeStageExecutionError(
                stage_id="extraction",
                code="ocr_provider_failed",
                message="Extraction paused because the OCR provider could not process a confirmed source.",
                detail=type(error).__name__,
            ) from error
        provider_status = artifact.raw.get("provider_metadata", {}).get("status")
        if artifact.raw.get("extraction_errors"):
            code = "ocr_config_missing" if provider_status == "configuration_error" else "ocr_provider_failed"
            message = (
                "Extraction paused because OCR provider configuration is incomplete."
                if code == "ocr_config_missing"
                else "Extraction paused because the OCR provider could not process a confirmed source."
            )
            raise RuntimeStageExecutionError(
                stage_id="extraction",
                code=code,
                message=message,
                detail="The OCR stage returned a recoverable provider status.",
            )
        if self._raw_ocr_cache is not None:
            record = self._raw_ocr_cache.store(identity=cache_identity, raw_artifact=artifact.raw)
            self.last_raw_cache_records_by_role[role] = record
            self.last_raw_cache_outcomes_by_role[role] = "miss"
        self.last_raw_extraction_artifacts_by_role[role] = artifact
        return self._require_mappable_artifact(artifact, package=package, role=role)

    def _require_mappable_artifact(
        self,
        artifact: RawExtractionArtifact,
        *,
        package: FilingPackage,
        role: str,
    ) -> RawExtractionArtifact:
        if not isinstance(artifact.raw.get("extraction_candidates"), dict):
            try:
                artifact.raw["extraction_candidates"] = map_raw_ocr_artifact_to_extraction_candidates(
                    artifact,
                    filing_package=package,
                    report_profile=_infer_report_profile(artifact),
                )
                self._record_deterministic_normalization(role, artifact)
            except RawOcrCandidateMappingError as error:
                if self._raw_ocr_normalizer is None:
                    artifact.raw.setdefault("extraction_errors", []).append(
                        {
                            "error_id": f"ERR_{artifact.source_document_id}_RAW_OCR_MAPPING",
                            "message": str(error),
                            "recoverable": True,
                            "reason_code": error.reason_code,
                        }
                    )
                    raise RuntimeStageExecutionError(
                        stage_id="extraction",
                        code="raw_extraction_invalid",
                        message="Extraction paused because OCR output could not be assembled into ReportMemory.",
                        detail=error.reason_code,
                    ) from error
                self._normalize_with_llm_fallback(
                    artifact,
                    package=package,
                    role=role,
                    deterministic_reason_code=error.reason_code,
                )
        else:
            self._record_provided_structured_candidates(role, artifact)
        if not isinstance(artifact.raw.get("extraction_candidates"), dict):
            raise RuntimeStageExecutionError(
                stage_id="extraction",
                code="raw_extraction_invalid",
                message="Extraction paused because OCR output could not be assembled into ReportMemory.",
                detail="The provider output did not include the structured fields needed for ReportMemory assembly.",
            )
        return artifact

    def _normalize_with_llm_fallback(
        self,
        artifact: RawExtractionArtifact,
        *,
        package: FilingPackage,
        role: str,
        deterministic_reason_code: str,
    ) -> None:
        assert self._raw_ocr_normalizer is not None
        try:
            output = self._raw_ocr_normalizer.normalize(
                raw_artifact=copy.deepcopy(artifact.raw),
                filing_package=copy.deepcopy(package.raw),
                source_document_id=artifact.source_document_id,
            )
            artifact.raw["extraction_candidates"] = validate_llm_normalizer_output(
                output,
                artifact=artifact,
                filing_package=package,
                identity=self._raw_ocr_normalizer.identity,
            )
        except RawOcrLlmNormalizationError as error:
            artifact.raw.setdefault("extraction_errors", []).append(
                {
                    "error_id": f"ERR_{artifact.source_document_id}_RAW_OCR_LLM_NORMALIZATION",
                    "message": str(error),
                    "recoverable": True,
                    "reason_code": error.reason_code,
                }
            )
            raise RuntimeStageExecutionError(
                stage_id="extraction",
                code="raw_extraction_invalid",
                message="Extraction paused because OCR output could not be assembled into ReportMemory.",
                detail=error.reason_code,
            ) from error
        metadata = self._raw_ocr_normalizer.identity.audit_metadata(
            extraction_method=str(artifact.raw.get("extraction_method") or ""),
            extraction_version=str(artifact.raw.get("extraction_version") or ""),
        )
        metadata["deterministic_mapper_status"] = "failed_closed"
        metadata["deterministic_mapper_reason_code"] = deterministic_reason_code
        metadata["evidence_surface_production"] = _evidence_surface_production(
            artifact.raw.get("extraction_candidates"),
            strategy="llm_assisted",
        )
        artifact.raw["_normalization_metadata"] = copy.deepcopy(metadata)
        self.last_normalization_metadata_by_role[role] = copy.deepcopy(metadata)
        self.last_report_memory_cache_compatibility_by_role[role] = (
            report_memory_cache_compatibility_for_llm_normalizer(
                self._raw_ocr_normalizer.identity,
                extraction_method=str(artifact.raw.get("extraction_method") or ""),
                extraction_version=str(artifact.raw.get("extraction_version") or ""),
            )
        )

    def _record_deterministic_normalization(self, role: str, artifact: RawExtractionArtifact) -> None:
        metadata = {
            "strategy": "deterministic",
            "mapper_version": MAPPER_VERSION,
            "extraction_method": artifact.raw.get("extraction_method"),
            "extraction_version": artifact.raw.get("extraction_version"),
            "evidence_surface_production": _evidence_surface_production(
                artifact.raw.get("extraction_candidates"),
                strategy="deterministic",
            ),
        }
        artifact.raw["_normalization_metadata"] = copy.deepcopy(metadata)
        self.last_normalization_metadata_by_role[role] = metadata
        self.last_report_memory_cache_compatibility_by_role[role] = ReportMemoryCacheCompatibility()

    def _record_provided_structured_candidates(self, role: str, artifact: RawExtractionArtifact) -> None:
        metadata = {
            "strategy": "provided_structured_candidates",
            "extraction_method": artifact.raw.get("extraction_method"),
            "extraction_version": artifact.raw.get("extraction_version"),
            "evidence_surface_production": _evidence_surface_production(
                artifact.raw.get("extraction_candidates"),
                strategy="provided_structured_candidates",
            ),
        }
        artifact.raw["_normalization_metadata"] = copy.deepcopy(metadata)
        self.last_normalization_metadata_by_role[role] = metadata

    def _cache_identity(self, source_document: Any) -> RawOcrCacheIdentity:
        return RawOcrCacheIdentity.from_non_secret_config(
            canonical_source_sha256=source_document.raw["fingerprint"]["hash_value"],
            provider="nanonets_ocr_3_docstrange",
            model=self._ocr_adapter.config.model,
            extraction_schema_version=RAW_EXTRACTION_ARTIFACT_SCHEMA_VERSION,
            extraction_version=self._ocr_adapter.extraction_version,
            non_secret_config={
                "extraction_method": self._ocr_adapter.extraction_method,
                "output_format": "html",
                "include_metadata": ["bounding_boxes", "confidence_score"],
            },
        )


def _evidence_surface_production(candidates: Any, *, strategy: str) -> list[dict[str, Any]]:
    surfaces = {
        "notes": "not_extracted_yet",
        "variance_explanations": "not_extracted_yet",
        "related_party_evidence": "not_extracted_yet",
        "accounting_policy_evidence": "not_extracted_yet",
        "extraction_quality": "not_extracted_yet",
    }
    refs: dict[str, str] = {}
    source_document_ids: dict[str, str] = {}
    if not isinstance(candidates, dict):
        return [
            {"surface": surface, "state": state, "producer_strategy": strategy}
            for surface, state in surfaces.items()
        ]
    for note in candidates.get("notes", []):
        if not isinstance(note, dict):
            continue
        note_type = note.get("note_type")
        if note.get("note_id"):
            surfaces["notes"] = "present"
            refs.setdefault("notes", str(note["note_id"]))
            if note.get("source_document_id"):
                source_document_ids.setdefault("notes", str(note["source_document_id"]))
        if note_type == "related_party_note":
            surfaces["related_party_evidence"] = "present"
            refs.setdefault("related_party_evidence", str(note.get("note_id") or "related_party_note"))
            if note.get("source_document_id"):
                source_document_ids.setdefault("related_party_evidence", str(note["source_document_id"]))
        if note_type in {"accounting_policy_change", "generic_accounting_policy"}:
            surfaces["accounting_policy_evidence"] = "present"
            refs.setdefault("accounting_policy_evidence", str(note.get("note_id") or "accounting_policy_note"))
            if note.get("source_document_id"):
                source_document_ids.setdefault("accounting_policy_evidence", str(note["source_document_id"]))
    for explanation in candidates.get("variance_explanations", []):
        if not isinstance(explanation, dict):
            continue
        if explanation.get("span_id"):
            surfaces["variance_explanations"] = "present"
            refs.setdefault("variance_explanations", str(explanation["span_id"]))
            if explanation.get("source_document_id"):
                source_document_ids.setdefault("variance_explanations", str(explanation["source_document_id"]))
    for status in candidates.get("evidence_surface_status", []):
        if not isinstance(status, dict):
            continue
        surface = status.get("surface")
        if surface in surfaces and isinstance(status.get("state"), str):
            surfaces[surface] = status["state"]
            if isinstance(status.get("evidence_ref"), str):
                refs[surface] = status["evidence_ref"]
            if isinstance(status.get("source_document_id"), str):
                source_document_ids[surface] = status["source_document_id"]
    records = []
    for surface, state in surfaces.items():
        record = {"surface": surface, "state": state, "producer_strategy": strategy}
        if surface in refs:
            record["evidence_ref"] = refs[surface]
        if surface in source_document_ids:
            record["source_document_id"] = source_document_ids[surface]
        records.append(record)
    return records


def _verify_source_artifact(source_document: Any) -> None:
    path = Path(source_document.raw["local_artifact"]["path"])
    expected_sha256 = source_document.raw["fingerprint"]["hash_value"]
    try:
        body = path.read_bytes()
    except OSError as error:
        raise RuntimeStageExecutionError(
            stage_id="extraction",
            code="source_artifact_unreachable",
            message="Extraction paused because a confirmed source artifact is unavailable.",
            detail="A confirmed source artifact could not be opened for OCR.",
        ) from error
    if hashlib.sha256(body).hexdigest() != expected_sha256:
        raise RuntimeStageExecutionError(
            stage_id="extraction",
            code="source_artifact_unreachable",
            message="Extraction paused because a confirmed source artifact failed identity verification.",
            detail="A confirmed source artifact hash did not match the selected source identity.",
        )


def _bind_cached_artifact(
    raw_artifact: dict[str, Any],
    package: FilingPackage,
    source_document: Any,
) -> dict[str, Any]:
    rebound = copy.deepcopy(raw_artifact)
    rebound["filing_package_id"] = package.package_id
    rebound["source_document_id"] = source_document.document_id
    rebound["artifact_id"] = f"RAW_EXT_{source_document.document_id}"
    rebound["source_document_provenance"] = copy.deepcopy(source_document.provenance)
    rebound["source_local_artifact"] = copy.deepcopy(source_document.raw["local_artifact"])
    return rebound


def _infer_report_profile(artifact: RawExtractionArtifact) -> str:
    existing = artifact.raw.get("extraction_candidates")
    if isinstance(existing, dict):
        metadata = existing.get("metadata")
        if isinstance(metadata, dict) and isinstance(metadata.get("report_profile"), str):
            return metadata["report_profile"]
    table_text = " ".join(
        str(cell)
        for table in artifact.raw.get("raw_tables", [])
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
            "fvtpl",
            "lưu ký chứng khoán",
            "luu ky chung khoan",
            "môi giới chứng khoán",
            "moi gioi chung khoan",
            "tự doanh chứng khoán",
            "tu doanh chung khoan",
            "cho vay giao dịch ký quỹ",
            "cho vay giao dich ky quy",
            "phải trả hoạt động giao dịch chứng khoán",
            "phai tra hoat dong giao dich chung khoan",
        )
    ):
        return "securities"
    standard_markers = (
        "bảng cân đối kế toán",
        "bang can doi ke toan",
        "kết quả hoạt động kinh doanh",
        "ket qua hoat dong kinh doanh",
        "lưu chuyển tiền tệ",
        "luu chuyen tien te",
        "doanh thu thuần",
        "doanh thu thuan",
        "giá vốn hàng bán",
        "gia von hang ban",
        "lợi nhuận gộp",
        "loi nhuan gop",
        "hàng tồn kho",
        "hang ton kho",
        "phải thu khách hàng",
        "phai thu khach hang",
    )
    if sum(1 for marker in standard_markers if marker in table_text) >= 3:
        return "standard_corporate"
    return "unknown"
