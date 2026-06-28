from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from vinacount.filing_package import LocalSourceDocument, ingest_local_filing_package
from research.fixture_spine import (
    RealExtractionQualityGateResult,
    build_detector_packets,
    gate_wave1_tools,
    generate_wave1_candidates,
    run_wave1_tools,
    validate_company_report_set,
    validate_real_extraction_quality_gate,
)
from vinacount.ocr_adapter import NanonetsOcr3DocstrangeAdapter, NanonetsOcr3DocstrangeConfig
from vinacount.raw_extraction_artifact import RawExtractionArtifact, validate_raw_extraction_artifact
from vinacount.report_memory_assembler import assemble_report_memory
from research.dataset_eligibility import (
    Wave4DatasetEligibilityResult,
    validate_real_extracted_packet_dataset_eligibility,
)


DEFAULT_RAW_EXTRACTION_CACHE_DIR = Path("artifacts/raw_extraction_artifacts")


@dataclass(frozen=True)
class SelectedSourceDocument:
    role: str
    source_document_id: str
    local_artifact_path: str
    fingerprint: dict[str, str]
    bytes_payload: bytes


@dataclass(frozen=True)
class RawExtractionResolution:
    status: str
    artifact: RawExtractionArtifact | None
    reason_codes: list[str]
    records: list[dict[str, Any]]


@dataclass(frozen=True)
class Wave4RealSourceExtractionRecord:
    status: str
    case_id: str
    blockers: list[dict[str, Any]]
    quality_gate_result: RealExtractionQualityGateResult | None
    tool_gating_records: list[dict[str, Any]]
    tool_findings: list[dict[str, Any]]
    candidate_risks: list[dict[str, Any]]
    detector_packets: list[dict[str, Any]]
    detector_packet_audit_records: list[dict[str, Any]]
    dataset_eligibility_result: Wave4DatasetEligibilityResult
    audit_traceability: dict[str, Any]


@dataclass(frozen=True)
class Wave4RealSourceExtractionResult:
    status: str
    records: list[Wave4RealSourceExtractionRecord]
    detector_assessments: list[dict[str, Any]]


def run_real_source_extraction_adapter(
    output_dir: Path | str,
    *,
    inventory_records: list[dict[str, Any]],
    raw_extraction_cache_dir: Path | str = DEFAULT_RAW_EXTRACTION_CACHE_DIR,
    live_ocr_enabled: bool = False,
    ocr_adapter: Any | None = None,
    write_artifacts: bool = True,
) -> Wave4RealSourceExtractionResult:
    output_path = Path(output_dir)
    if write_artifacts:
        output_path.mkdir(parents=True, exist_ok=True)

    records = [
        _run_inventory_record(
            output_path,
            record,
            raw_extraction_cache_dir=Path(raw_extraction_cache_dir),
            live_ocr_enabled=live_ocr_enabled,
            ocr_adapter=ocr_adapter,
            write_artifacts=write_artifacts,
        )
        for record in inventory_records
    ]
    if write_artifacts:
        _write_json(output_path / "real_source_extraction_adapter.json", {"records": [asdict(record) for record in records]})
    return Wave4RealSourceExtractionResult(
        status="completed",
        records=records,
        detector_assessments=[],
    )


def _run_inventory_record(
    output_path: Path,
    record: dict[str, Any],
    *,
    raw_extraction_cache_dir: Path,
    live_ocr_enabled: bool,
    ocr_adapter: Any | None,
    write_artifacts: bool,
) -> Wave4RealSourceExtractionRecord:
    case_id = _case_id(record)
    audit_traceability = _audit_traceability(record)
    blockers = _pre_extraction_blockers(record)
    if blockers:
        terminal_status = _terminal_status_for_pre_extraction_blockers(blockers)
        return _terminal_record(terminal_status, case_id, blockers, audit_traceability)

    try:
        target_source = _selected_source_document(record, role="target")
        prior_source = _selected_source_document(record, role="prior_year")
        for key in (
            "target_selected_member_path",
            "target_selected_member_sha256",
            "prior_year_selected_member_path",
            "prior_year_selected_member_sha256",
        ):
            if key in record.get("non_detector_visible_provenance", {}):
                audit_traceability[key] = record["non_detector_visible_provenance"][key]

        target_year = _record_year(record)
        target_package = _filing_package(
            record,
            target_source.local_artifact_path,
            target_source.fingerprint["hash_value"],
            year=target_year,
            source_document_id=target_source.source_document_id,
        )
        prior_package = _filing_package(
            record,
            prior_source.local_artifact_path,
            prior_source.fingerprint["hash_value"],
            year=target_year - 1,
            source_document_id=prior_source.source_document_id,
        )
        target_resolution = _resolve_raw_extraction_artifact(
            case_id=case_id,
            role="target",
            filing_package=target_package,
            source_document_id=target_source.source_document_id,
            cache_dir=raw_extraction_cache_dir,
            live_ocr_enabled=live_ocr_enabled,
            ocr_adapter=ocr_adapter,
        )
        prior_resolution = _resolve_raw_extraction_artifact(
            case_id=case_id,
            role="prior_year",
            filing_package=prior_package,
            source_document_id=prior_source.source_document_id,
            cache_dir=raw_extraction_cache_dir,
            live_ocr_enabled=live_ocr_enabled,
            ocr_adapter=ocr_adapter,
        )
        follow_up_records = target_resolution.records + prior_resolution.records
        if follow_up_records:
            return _terminal_record("needs_follow_up", case_id, follow_up_records, audit_traceability)

        target_artifact = target_resolution.artifact
        prior_artifact = prior_resolution.artifact
        if target_artifact is None or prior_artifact is None:
            return _terminal_record(
                "needs_follow_up",
                case_id,
                [
                    {
                        "record_type": "blocker",
                        "reason_code": "raw_extraction_artifact_invalid",
                        "reason": "Raw extraction resolver returned no usable artifact.",
                    }
                ],
                audit_traceability,
            )
        normalized_text_hash = _hash_normalized_text(target_artifact.raw.get("normalized_text", target_artifact.raw.get("raw_html", "")))
        table_content_hash = _hash_table_content(target_artifact.raw.get("extraction_candidates", {}).get("statement_tables", []))
        audit_traceability["normalized_text_hash"] = normalized_text_hash
        audit_traceability["table_content_hash"] = table_content_hash

        target_report_memory = assemble_report_memory(filing_package=target_package, raw_extraction_artifacts=[target_artifact])
        prior_report_memory = assemble_report_memory(filing_package=prior_package, raw_extraction_artifacts=[prior_artifact])
        _apply_non_detector_visible_report_metadata(target_report_memory.raw, record, target_package.source_documents[0].document_id, audit_traceability)
        _apply_non_detector_visible_report_metadata(prior_report_memory.raw, record, prior_package.source_documents[0].document_id, audit_traceability)
        report_set = validate_company_report_set(case_id, target_report_memory, prior_report_memory)
    except Exception as error:
        blocker = {"record_type": "blocker", "reason_code": _reason_code_for_error(error), "reason": str(error)}
        role = _role_for_error(error)
        if role:
            blocker["role"] = role
        return _terminal_record(
            "needs_follow_up",
            case_id,
            [blocker],
            audit_traceability,
        )

    quality_gate_result = validate_real_extraction_quality_gate(report_set)
    if not quality_gate_result.allows_tool_availability_gating:
        return Wave4RealSourceExtractionRecord(
            status="needs_follow_up",
            case_id=case_id,
            blockers=quality_gate_result.records,
            quality_gate_result=quality_gate_result,
            tool_gating_records=[],
            tool_findings=[],
            candidate_risks=[],
            detector_packets=[],
            detector_packet_audit_records=[],
            dataset_eligibility_result=validate_real_extracted_packet_dataset_eligibility(report_set, []),
            audit_traceability=audit_traceability,
        )

    tool_gating_records = gate_wave1_tools(report_set)
    tool_findings = run_wave1_tools(report_set, tool_gating_records)
    candidate_risks = generate_wave1_candidates(report_set, tool_findings)
    if not candidate_risks and record.get("negative_control_candidate"):
        candidate_risks = _negative_control_candidates(
            report_set,
            tool_findings,
            record["negative_control_candidate"],
        )
    detector_packet_audit_records: list[dict[str, Any]] = []
    detector_packets = build_detector_packets(report_set, candidate_risks, tool_findings, detector_packet_audit_records)
    dataset_eligibility_result = validate_real_extracted_packet_dataset_eligibility(report_set, detector_packets)
    status = _packet_status(candidate_risks, detector_packets, dataset_eligibility_result)

    out = Wave4RealSourceExtractionRecord(
        status=status,
        case_id=case_id,
        blockers=[],
        quality_gate_result=quality_gate_result,
        tool_gating_records=tool_gating_records,
        tool_findings=tool_findings,
        candidate_risks=candidate_risks,
        detector_packets=detector_packets,
        detector_packet_audit_records=detector_packet_audit_records,
        dataset_eligibility_result=dataset_eligibility_result,
        audit_traceability=audit_traceability,
    )
    if write_artifacts:
        _write_json(output_path / f"{case_id}.json", asdict(out))
    return out


def _packet_status(
    candidate_risks: list[dict[str, Any]],
    detector_packets: list[dict[str, Any]],
    eligibility_result: Wave4DatasetEligibilityResult,
) -> str:
    if detector_packets and eligibility_result.status == "passed":
        return "packet_ready_unadjudicated"
    if detector_packets:
        return "dataset_ineligible"
    if not candidate_risks:
        return "extracted_no_candidates"
    return "needs_follow_up"


def _negative_control_candidates(
    report_set: Any,
    tool_findings: list[dict[str, Any]],
    spec: dict[str, Any],
) -> list[dict[str, Any]]:
    linked_findings = [
        finding
        for finding in tool_findings
        if finding.get("risk_category") == spec["risk_category"]
        and finding.get("signal_id") in set(spec["linked_signal_ids"])
    ]
    if not linked_findings:
        return []
    return [
        {
            "candidate_id": spec["candidate_id"],
            "report_id": report_set.target.report_id,
            "risk_category": spec["risk_category"],
            "candidate_status": "negative_control_pending_detector_review",
            "generation_source": "rule_generated_negative_control",
            "reason_for_candidate": spec["reason_for_candidate"],
            "priority": "low",
            "review_mode": "negative_control",
            "supporting_signal_ids": [finding["signal_id"] for finding in linked_findings],
            "linked_tool_result_ids": [finding["tool_result_id"] for finding in linked_findings],
            "required_evidence_refs": _candidate_evidence_refs_for_findings(linked_findings),
            "required_context_queries": [
                {
                    "query_type": "table_rows",
                    "account_group": "cashflow",
                    "standard_account": "operating_cash_flow",
                    "periods": [
                        report_set.target.metadata["period"],
                        report_set.prior_year.metadata["period"],
                    ],
                    "max_items": 6,
                }
            ],
            "applicability": {
                "report_profile": report_set.target.metadata["report_profile"],
                "is_applicable": True,
                "reason": "Negative-control review uses completed packet-visible tool evidence.",
            },
        }
    ]


def _candidate_evidence_refs_for_findings(
    findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    refs = []
    seen = set()
    for finding in findings:
        tool_ref = {
            "evidence_ref_type": "tool_result",
            "ref_id": finding["tool_result_id"],
            "role": "required_for_review",
        }
        refs.append(tool_ref)
        seen.add((tool_ref["evidence_ref_type"], tool_ref["ref_id"]))
        for evidence_ref in finding.get("evidence_refs", []):
            key = (evidence_ref["evidence_ref_type"], evidence_ref["ref_id"])
            if key in seen:
                continue
            refs.append({**evidence_ref, "role": "supporting"})
            seen.add(key)
    return refs


def _pre_extraction_blockers(record: dict[str, Any]) -> list[dict[str, Any]]:
    if _is_curation_record(record):
        return _curation_record_blockers(record)
    if record.get("curation_state") == "needs_follow_up":
        return [
            {"record_type": "blocker", "reason_code": code, "reason": "Curated source record requires follow-up before extraction."}
            for code in record.get("non_detector_visible_provenance", {}).get("reason_codes", ["source_or_profile_not_ready"])
        ]
    blockers = []
    for field, reason_code in (
        ("ticker", "source_or_profile_not_ready"),
        ("year", "source_or_profile_not_ready"),
        ("quarter", "source_or_profile_not_ready"),
        ("report_basis", "source_or_profile_not_ready"),
        ("report_profile_guess", "ambiguous_report_profile"),
    ):
        if not record.get(field):
            blockers.append({"record_type": "blocker", "reason_code": reason_code, "reason": f"Record is missing {field}."})
    if not (record.get("target_path") or (record.get("target_container_path") and record.get("target_selected_member_path"))):
        blockers.append({"record_type": "blocker", "reason_code": "source_or_profile_not_ready", "reason": "Record is missing target source path or selected ZIP member."})
    if not (record.get("prior_year_path") or (record.get("prior_year_container_path") and record.get("prior_year_selected_member_path"))):
        blockers.append({"record_type": "blocker", "reason_code": "source_or_profile_not_ready", "reason": "Record is missing prior-year source path or selected ZIP member."})
    return blockers


def _is_curation_record(record: dict[str, Any]) -> bool:
    return bool(record.get("case_id") and "source_readiness_status" in record and "source_path" in record)


def _curation_record_blockers(record: dict[str, Any]) -> list[dict[str, Any]]:
    if record.get("source_readiness_status") == "blocked" or record.get("status") == "blocked":
        return [
            {"record_type": "blocker", "reason_code": code, "reason": "Curated source record is blocked before extraction."}
            for code in record.get("reason_codes") or ["source_or_profile_not_ready"]
        ]
    if record.get("source_readiness_status") == "source_ready":
        return []
    return [
        {
            "record_type": "blocker",
            "terminal_status": "needs_follow_up",
            "reason_code": _curation_source_follow_up_reason_code(record),
            "reason": "Curated real source requires real PDF extraction before packets can be produced.",
        }
    ]


def _curation_source_follow_up_reason_code(record: dict[str, Any]) -> str:
    source_kind = record.get("source_kind")
    source_path = Path(record["source_path"]) if record.get("source_path") else None
    selected_member_path = Path(record["selected_member_path"]) if record.get("selected_member_path") else None
    candidate_path = selected_member_path if source_kind == "vietstock_zip_member_pdf" else source_path
    if candidate_path and candidate_path.exists():
        payload = candidate_path.read_bytes()[:16]
        if payload.lstrip().startswith(b"%PDF"):
            return "real_pdf_text_extraction_unavailable"
    return "cached_extraction_artifact_missing"


def _terminal_status_for_pre_extraction_blockers(blockers: list[dict[str, Any]]) -> str:
    terminal_statuses = {blocker.pop("terminal_status", None) for blocker in blockers}
    if "needs_follow_up" in terminal_statuses:
        return "needs_follow_up"
    return "blocked"


def _selected_source_document(record: dict[str, Any], *, role: str) -> SelectedSourceDocument:
    direct_path = record.get(f"{role}_path")
    if direct_path:
        path = Path(direct_path)
        if not path.exists():
            raise ValueError(f"{role}: selected source document unavailable")
        payload = path.read_bytes()
        return SelectedSourceDocument(
            role=role,
            source_document_id=_source_document_id(record, role=role),
            local_artifact_path=str(path),
            fingerprint={"hash_algorithm": "sha256", "hash_value": _sha256_bytes(payload)},
            bytes_payload=payload,
        )

    container_path_text = record.get(f"{role}_container_path")
    member_path = record.get(f"{role}_selected_member_path")
    if container_path_text and member_path:
        container_path = Path(container_path_text)
        try:
            member_bytes = _selected_zip_member_bytes(container_path, member_path)
        except ValueError as error:
            raise ValueError(f"{role}: selected ZIP member unavailable") from error
        member_hash = _sha256_bytes(member_bytes)
        expected_hash = record.get(f"{role}_selected_member_sha256")
        if expected_hash and expected_hash != member_hash:
            raise ValueError(f"{role}: selected ZIP member unavailable")
        record.setdefault("non_detector_visible_provenance", {})[f"{role}_selected_member_path"] = member_path
        record.setdefault("non_detector_visible_provenance", {})[f"{role}_selected_member_sha256"] = member_hash
        return SelectedSourceDocument(
            role=role,
            source_document_id=_source_document_id(record, role=role),
            local_artifact_path=member_path,
            fingerprint={"hash_algorithm": "sha256", "hash_value": member_hash},
            bytes_payload=member_bytes,
        )

    if role == "target" and _is_curation_record(record):
        return _curation_selected_source_document(record, role=role)
    if role == "prior_year" and _is_curation_record(record):
        return _curation_prior_year_source_document(record)

    raise ValueError(f"{role}: selected source document unavailable")


def _curation_selected_source_document(record: dict[str, Any], *, role: str) -> SelectedSourceDocument:
    if record.get("source_kind") == "vietstock_zip_member_pdf":
        member_path = record.get("selected_member_path")
        if not member_path:
            raise ValueError(f"{role}: selected ZIP member unavailable")
        member = Path(member_path)
        if not member.exists():
            raise ValueError(f"{role}: selected ZIP member unavailable")
        member_bytes = member.read_bytes()
        member_hash = _sha256_bytes(member_bytes)
        expected_hash = record.get("selected_member_sha256")
        if expected_hash and expected_hash != member_hash:
            raise ValueError(f"{role}: selected ZIP member unavailable")
        return SelectedSourceDocument(
            role=role,
            source_document_id=_source_document_id(record, role=role),
            local_artifact_path=str(member),
            fingerprint={"hash_algorithm": "sha256", "hash_value": member_hash},
            bytes_payload=member_bytes,
        )

    path = Path(record["source_path"])
    if not path.exists():
        raise ValueError(f"{role}: selected source document unavailable")
    payload = path.read_bytes()
    return SelectedSourceDocument(
        role=role,
        source_document_id=_source_document_id(record, role=role),
        local_artifact_path=str(path),
        fingerprint={"hash_algorithm": "sha256", "hash_value": _sha256_bytes(payload)},
        bytes_payload=payload,
    )


def _curation_prior_year_source_document(record: dict[str, Any]) -> SelectedSourceDocument:
    source_dir = Path(record["source_path"]).parent
    company = record["company_key"]
    year, quarter = _period_parts(record["period_key"])
    prior_year = year - 1
    basis = record["report_basis"]
    direct_matches = sorted(source_dir.glob(f"{company}_{prior_year}_{quarter}_{basis}_*.pdf"))
    if direct_matches:
        path = direct_matches[0]
        payload = path.read_bytes()
        return SelectedSourceDocument(
            role="prior_year",
            source_document_id=_source_document_id(record, role="prior_year"),
            local_artifact_path=str(path),
            fingerprint={"hash_algorithm": "sha256", "hash_value": _sha256_bytes(payload)},
            bytes_payload=payload,
        )

    zip_matches = sorted(source_dir.glob(f"{company}_{prior_year}_{quarter}_{basis}_*.zip"))
    if not zip_matches:
        raise ValueError("prior_year: selected source document unavailable")
    member = _selected_statement_member(source_dir / "extracted" / zip_matches[0].stem)
    if member is None:
        raise ValueError("prior_year: selected ZIP member unavailable")
    member_bytes = member.read_bytes()
    member_hash = _sha256_bytes(member_bytes)
    return SelectedSourceDocument(
        role="prior_year",
        source_document_id=_source_document_id(record, role="prior_year"),
        local_artifact_path=str(member),
        fingerprint={"hash_algorithm": "sha256", "hash_value": member_hash},
        bytes_payload=member_bytes,
    )


def _selected_statement_member(extracted_dir: Path) -> Path | None:
    if not extracted_dir.exists():
        return None
    members = sorted(extracted_dir.glob("*.pdf"))
    statement_members = [
        member
        for member in members
        if "giaitrinh" not in member.name.lower()
        and "explanation" not in member.name.lower()
        and "baocaotaichinh" in member.name.lower()
    ]
    vietnamese_members = [
        member
        for member in statement_members
        if "_vi_" in member.name.lower() or "_vn_" in member.name.lower() or member.name.startswith("VI_")
    ]
    if vietnamese_members:
        return vietnamese_members[0]
    if statement_members:
        return statement_members[0]
    return members[0] if members else None


def _selected_zip_member_bytes(container_path: Path, member_path: str) -> bytes:
    extracted_member = Path(member_path)
    if extracted_member.exists():
        return extracted_member.read_bytes()
    if not container_path.exists():
        raise ValueError("selected ZIP member unavailable")
    return _read_zip_member_bytes(container_path, member_path)


def _period_parts(period_key: str) -> tuple[int, str]:
    year_text, quarter = period_key.split("_", maxsplit=1)
    return int(year_text), quarter


def _record_year(record: dict[str, Any]) -> int:
    if "year" in record:
        return int(record["year"])
    return _period_parts(record["period_key"])[0]


def _record_quarter(record: dict[str, Any]) -> str:
    if "quarter" in record:
        return record["quarter"]
    return _period_parts(record["period_key"])[1]


def _source_document_id(record: dict[str, Any], *, role: str) -> str:
    if role == "target":
        operative_id = record.get("audit_traceability", {}).get("operative_corrected_source_document_id")
        if operative_id:
            return operative_id
        if record.get("source_document_identity", {}).get("source_document_id"):
            return record["source_document_identity"]["source_document_id"]
        if record.get("derived_from_source_document_id"):
            return record["derived_from_source_document_id"]
    ticker = record.get("ticker") or record.get("company_key")
    year = _record_year(record)
    if role == "prior_year":
        year -= 1
    quarter = _record_quarter(record)
    basis = record.get("report_basis", "parent")
    return f"DOC_{ticker}_{year}_{quarter}_{basis}"


def _resolve_raw_extraction_artifact(
    *,
    case_id: str,
    role: str,
    filing_package,
    source_document_id: str,
    cache_dir: Path,
    live_ocr_enabled: bool,
    ocr_adapter: Any | None,
) -> RawExtractionResolution:
    cache_path = _raw_extraction_cache_path(
        cache_dir,
        case_id=case_id,
        role=role,
        filing_package=filing_package,
        source_document_id=source_document_id,
    )
    if cache_path.exists():
        try:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            return _raw_extraction_follow_up(
                role,
                "raw_extraction_artifact_invalid",
                f"Cached Raw Extraction Artifact could not be loaded: {error}",
            )
        return _validate_cached_raw_extraction_artifact(role, raw, filing_package)

    if not live_ocr_enabled:
        return RawExtractionResolution(
            status="needs_follow_up",
            artifact=None,
            reason_codes=["cached_raw_extraction_artifact_missing", "live_ocr_not_configured"],
            records=[
                _resolution_record(
                    role,
                    "cached_raw_extraction_artifact_missing",
                    f"Cached Raw Extraction Artifact is missing at {cache_path}.",
                ),
                _resolution_record(
                    role,
                    "live_ocr_not_configured",
                    "Live OCR is disabled for this extraction run.",
                ),
            ],
        )

    adapter = ocr_adapter or NanonetsOcr3DocstrangeAdapter(config=NanonetsOcr3DocstrangeConfig(live_ocr_enabled=True))
    try:
        artifact = adapter.extract_source_document(
            filing_package=filing_package,
            source_document_id=source_document_id,
        )
    except Exception as error:
        return _raw_extraction_follow_up(role, "ocr_provider_error", f"OCR provider failed: {error}")

    if artifact.raw.get("extraction_errors") or artifact.raw.get("provider_metadata", {}).get("status") in {
        "configuration_error",
        "provider_error",
        "failed",
    }:
        return _raw_extraction_follow_up(role, "ocr_provider_error", "OCR provider returned a recoverable error artifact.")
    source_document = _find_package_source_document(filing_package, source_document_id)
    artifact.raw.setdefault("source_document_fingerprint", source_document.raw["fingerprint"])
    try:
        live_artifact = validate_raw_extraction_artifact(artifact.raw, filing_package)
    except (KeyError, TypeError, ValueError) as error:
        return _raw_extraction_follow_up(role, "raw_extraction_artifact_invalid", str(error))
    artifact_fingerprint = live_artifact.raw.get("source_document_fingerprint")
    if artifact_fingerprint != source_document.raw["fingerprint"]:
        return _raw_extraction_follow_up(
            role,
            "raw_extraction_artifact_source_mismatch",
            "Raw Extraction Artifact source fingerprint must match selected source document.",
        )
    _write_json(cache_path, live_artifact.raw)
    validation = _validate_cached_raw_extraction_artifact(role, artifact.raw, filing_package)
    return validation


def _raw_extraction_cache_path(
    cache_dir: Path,
    *,
    case_id: str,
    role: str,
    filing_package,
    source_document_id: str,
) -> Path:
    source_document = _find_package_source_document(filing_package, source_document_id)
    fingerprint = source_document.raw["fingerprint"]["hash_value"]
    filename = f"{case_id}__{role}__{source_document_id}__{fingerprint[:12]}__raw_extraction_artifact.json"
    return cache_dir / filename


def _validate_cached_raw_extraction_artifact(
    role: str,
    raw: dict[str, Any],
    filing_package,
) -> RawExtractionResolution:
    try:
        artifact = validate_raw_extraction_artifact(raw, filing_package)
    except (KeyError, TypeError, ValueError) as error:
        return _raw_extraction_follow_up(role, "raw_extraction_artifact_invalid", str(error))

    source_document = _find_package_source_document(filing_package, artifact.source_document_id)
    artifact_fingerprint = raw.get("source_document_fingerprint")
    if artifact_fingerprint != source_document.raw["fingerprint"]:
        return _raw_extraction_follow_up(
            role,
            "raw_extraction_artifact_source_mismatch",
            "Raw Extraction Artifact source fingerprint does not match selected source document.",
        )
    candidates = raw.get("extraction_candidates")
    if not isinstance(candidates, dict) or not candidates:
        if raw.get("raw_tables"):
            reason_code = "real_pdf_table_extraction_unavailable"
            reason = "Raw Extraction Artifact has raw tables but no reusable extraction_candidates."
        else:
            reason_code = "raw_extraction_artifact_invalid"
            reason = "Raw Extraction Artifact does not include reusable extraction_candidates."
        return _raw_extraction_follow_up(role, reason_code, reason)
    return RawExtractionResolution(
        status="resolved",
        artifact=artifact,
        reason_codes=[],
        records=[],
    )


def _find_package_source_document(filing_package, source_document_id: str):
    for source_document in filing_package.source_documents:
        if source_document.document_id == source_document_id:
            return source_document
    raise ValueError("selected source document unavailable")


def _raw_extraction_follow_up(role: str, reason_code: str, reason: str) -> RawExtractionResolution:
    return RawExtractionResolution(
        status="needs_follow_up",
        artifact=None,
        reason_codes=[reason_code],
        records=[_resolution_record(role, reason_code, reason)],
    )


def _resolution_record(role: str, reason_code: str, reason: str) -> dict[str, Any]:
    return {
        "record_type": "blocker",
        "role": role,
        "reason_code": reason_code,
        "reason": reason,
    }


def _read_zip_member_bytes(container_path: Path, member_path: str) -> bytes:
    try:
        with zipfile.ZipFile(container_path) as archive:
            try:
                return archive.read(member_path)
            except KeyError:
                extracted_member = Path(member_path)
                if extracted_member.exists():
                    return extracted_member.read_bytes()
                basename = Path(member_path).name
                if basename in archive.namelist():
                    return archive.read(basename)
                raise ValueError("selected ZIP member unavailable") from None
    except zipfile.BadZipFile as error:
        raise ValueError("selected ZIP member unavailable") from error


def _filing_package(
    record: dict[str, Any],
    source_path: str,
    source_hash: str,
    *,
    year: int,
    source_document_id: str | None = None,
):
    quarter = _record_quarter(record)
    period = f"{year}-{quarter}"
    ticker = record.get("ticker") or record.get("company_key")
    suffix = period.replace("-", "_")
    document_id = source_document_id or record.get("source_document_identity", {}).get("source_document_id")
    if document_id is None and year != _record_year(record):
        document_id = f"DOC_{ticker}_{suffix}_FS"
    return ingest_local_filing_package(
        package_id=f"PKG_{ticker}_{suffix}_{record['report_basis'].upper()}",
        company={"name": record.get("company_name") or f"{ticker} Company", "ticker": ticker},
        period=period,
        quarter=quarter,
        report_basis=record["report_basis"],
        filing_event={"event_id": f"EVT_{ticker}_{suffix}", "filing_status": "original"},
        source_documents=[
            LocalSourceDocument(
                document_id=document_id or f"DOC_{ticker}_{suffix}_FS",
                document_type="main_financial_statement",
                artifact_id=f"ART_{ticker}_{suffix}_FS",
                path=source_path,
                hash_value=source_hash,
                reported_period_clues=[period],
                basis_clues=[record["report_basis"]],
                source_name="real_source_extraction_adapter",
            )
        ],
    )


def _apply_non_detector_visible_report_metadata(
    report_memory: dict[str, Any],
    record: dict[str, Any],
    source_document_id: str,
    audit_traceability: dict[str, Any],
) -> None:
    metadata = report_memory["metadata"]
    metadata["source_confirmation_status"] = "confirmed"
    metadata["canonical_source_certainty"] = "resolved"
    metadata["source_quality_status"] = "passed"
    metadata["source_quality_details"] = {"identity_check_status": "confirmed"}
    metadata["cache_decontamination_metadata"] = {
        "company_key": record.get("ticker"),
        "period_key": metadata["period"],
        "group_key": audit_traceability.get("company_period_group_key"),
        "source_file_sha256": audit_traceability.get("source_file_sha256"),
        "normalized_text_hash": audit_traceability.get("normalized_text_hash"),
        "table_content_hash": audit_traceability.get("table_content_hash"),
        "derived_from_report_artifact_id": audit_traceability.get("derived_from_report_artifact_id"),
        "derived_from_source_document_id": source_document_id,
    }


def _terminal_record(
    status: str,
    case_id: str,
    blockers: list[dict[str, Any]],
    audit_traceability: dict[str, Any],
) -> Wave4RealSourceExtractionRecord:
    return Wave4RealSourceExtractionRecord(
        status=status,
        case_id=case_id,
        blockers=blockers,
        quality_gate_result=None,
        tool_gating_records=[],
        tool_findings=[],
        candidate_risks=[],
        detector_packets=[],
        detector_packet_audit_records=[],
        dataset_eligibility_result=Wave4DatasetEligibilityResult(
            status="failed",
            records=[{**blocker, "case_id": case_id, "record_type": "dataset_ineligible"} for blocker in blockers],
        ),
        audit_traceability=audit_traceability,
    )


def _reason_code_for_error(error: Exception) -> str:
    message = str(error)
    if "real PDF text extraction unavailable" in message:
        return "real_pdf_table_extraction_unavailable"
    if "text extraction failed" in message:
        return "real_pdf_table_extraction_unavailable"
    if "statement tables" in message:
        return "structured_statement_tables_not_found"
    if "profile" in message:
        return "report_profile_unresolved"
    if "currency" in message or "unit" in message:
        return "unit_or_currency_unresolved"
    if "cell_index" in message or "evidence ID" in message:
        return "evidence_registry_invalid"
    if "selected ZIP member unavailable" in message:
        return "selected_zip_member_unavailable"
    if "selected source document unavailable" in message:
        return "selected_source_document_unavailable"
    if "CompanyReportSet" in message:
        return "company_report_set_validation_failed"
    return "raw_extraction_artifact_invalid"


def _role_for_error(error: Exception) -> str | None:
    message = str(error)
    if message.startswith("target:"):
        return "target"
    if message.startswith("prior_year:"):
        return "prior_year"
    return None


def _audit_traceability(record: dict[str, Any]) -> dict[str, Any]:
    provenance = dict(record.get("non_detector_visible_provenance", {}))
    for key in (
        "company_key",
        "period_key",
        "group_key",
        "source_file_sha256",
        "normalized_text_hash",
        "table_content_hash",
        "derived_from_report_artifact_id",
        "derived_from_source_document_id",
        "source_kind",
        "source_path",
        "selected_member_path",
        "selected_member_sha256",
        "source_readiness_status",
        "resolution_strategy",
    ):
        if key in record:
            provenance[key] = record[key]
    for key in (
        "target_selected_member_path",
        "target_selected_member_sha256",
        "prior_year_selected_member_path",
        "prior_year_selected_member_sha256",
    ):
        if key in record:
            provenance[key] = record[key]
    for key in (
        "sampling_pool",
        "curation_state",
        "inventory_record_type",
        "correction_amendment_provenance_type",
        "ordinary_provenance_rationale",
    ):
        if key in record:
            provenance[key] = record[key]
    provenance["label_policy"] = record.get("label_policy", {})
    return provenance


def _case_id(record: dict[str, Any]) -> str:
    if record.get("case_id"):
        return f"real_source_{record['case_id']}"
    return (
        f"real_source_{record.get('ticker', 'unknown')}_{record.get('year', 'unknown')}_"
        f"{record.get('quarter', 'unknown')}_{record.get('report_basis', 'unknown')}"
    )


def _hash_normalized_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
    return _sha256_text(normalized)


def _hash_table_content(statement_tables: list[dict[str, Any]]) -> str:
    canonical = json.dumps(statement_tables, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_text(canonical)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path

