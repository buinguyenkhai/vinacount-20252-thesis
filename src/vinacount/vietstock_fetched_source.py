from __future__ import annotations

import hashlib
import re
import unicodedata
import zipfile
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from vinacount.filing_package import validate_filing_package
from vinacount.source_evidence import extract_source_evidence_for_documents


VIETSTOCK_BASE_URL = "https://finance.vietstock.vn"
NON_CONSOLIDATED_BASIS_FAMILIES = {"parent", "separate", "non_consolidated"}


class VietstockTokenError(ValueError):
    pass


class VietstockApiParseError(ValueError):
    pass


@dataclass(frozen=True)
class VietstockFetchRecord:
    severity: str
    code: str
    message: str


@dataclass(frozen=True)
class VietstockFetchedFilingPackageResult:
    status: str
    package: Any | None
    records: list[VietstockFetchRecord]


@dataclass(frozen=True)
class VietstockQuarterlyMetadataRow:
    ticker: str
    company_name_vi: str
    vietstock_profile_raw: str
    vinacount_profile_candidate: str
    year: int
    quarter: str
    source_entry_file_id: str
    source_entry_title: str
    source_entry_url: str
    last_update: str | None
    basis_clues: list[str]
    document_type_guess: str
    correction_signal_terms: list[str]
    assurance_signal_terms: list[str]
    coverage_status: str
    ambiguity_notes: list[str]


@dataclass(frozen=True)
class VietstockQuarterlyMetadataScanResult:
    status: str
    rows: list[VietstockQuarterlyMetadataRow]
    records: list[VietstockFetchRecord]


def scan_vietstock_quarterly_metadata(
    *,
    ticker: str,
    company_name_vi: str,
    vietstock_profile_raw: str,
    vinacount_profile_candidate: str,
    year: int,
    quarter: str,
    session: Any,
    base_url: str = VIETSTOCK_BASE_URL,
    max_pages: int = 30,
) -> VietstockQuarterlyMetadataScanResult:
    ticker = ticker.upper()
    page_url = f"{base_url}/{ticker}/tai-tai-lieu.htm?doctype=1"
    try:
        token = _get_request_verification_token(session, page_url)
        api_items = _fetch_document_metadata(
            session,
            base_url=base_url,
            ticker=ticker,
            year=year,
            token=token,
            page_url=page_url,
            max_pages=max_pages,
        )
    except Exception as exc:
        return VietstockQuarterlyMetadataScanResult(
            status="recoverable_data_error",
            rows=[],
            records=[VietstockFetchRecord(severity="recoverable_data_error", code="metadata_scan_failed", message=str(exc))],
        )

    rows = _metadata_rows_from_api_items(
        api_items,
        ticker=ticker,
        company_name_vi=company_name_vi,
        vietstock_profile_raw=vietstock_profile_raw,
        vinacount_profile_candidate=vinacount_profile_candidate,
        year=year,
        quarter=quarter,
    )
    if not any(row.coverage_status == "metadata_hit" for row in rows):
        return VietstockQuarterlyMetadataScanResult(
            status="completed_no_metadata_hits",
            rows=rows,
            records=[],
        )
    return VietstockQuarterlyMetadataScanResult(status="completed", rows=rows, records=[])


def scan_vietstock_quarterly_metadata_for_year(
    *,
    ticker: str,
    company_name_vi: str,
    vietstock_profile_raw: str,
    vinacount_profile_candidate: str,
    year: int,
    quarters: list[str],
    session: Any,
    base_url: str = VIETSTOCK_BASE_URL,
    max_pages: int = 30,
) -> VietstockQuarterlyMetadataScanResult:
    ticker = ticker.upper()
    page_url = f"{base_url}/{ticker}/tai-tai-lieu.htm?doctype=1"
    try:
        token = _get_request_verification_token(session, page_url)
        api_items = _fetch_document_metadata(
            session,
            base_url=base_url,
            ticker=ticker,
            year=year,
            token=token,
            page_url=page_url,
            max_pages=max_pages,
        )
    except Exception as exc:
        return VietstockQuarterlyMetadataScanResult(
            status="recoverable_data_error",
            rows=[],
            records=[VietstockFetchRecord(severity="recoverable_data_error", code="metadata_scan_failed", message=str(exc))],
        )

    rows = []
    for quarter in quarters:
        quarter_rows = _metadata_rows_from_api_items(
            api_items,
            ticker=ticker,
            company_name_vi=company_name_vi,
            vietstock_profile_raw=vietstock_profile_raw,
            vinacount_profile_candidate=vinacount_profile_candidate,
            year=year,
            quarter=quarter,
        )
        rows.extend(quarter_rows)
    status = "completed" if any(row.coverage_status == "metadata_hit" for row in rows) else "completed_no_metadata_hits"
    return VietstockQuarterlyMetadataScanResult(status=status, rows=rows, records=[])


def _metadata_rows_from_api_items(
    api_items: list[dict[str, Any]],
    *,
    ticker: str,
    company_name_vi: str,
    vietstock_profile_raw: str,
    vinacount_profile_candidate: str,
    year: int,
    quarter: str,
) -> list[VietstockQuarterlyMetadataRow]:
    period = f"{year}-{quarter.upper()}"
    rows = []
    for item in api_items:
        classified = _classify_document(item, period=period, year=year, quarter=quarter)
        if classified is None:
            continue
        rows.append(
            VietstockQuarterlyMetadataRow(
                ticker=ticker,
                company_name_vi=company_name_vi,
                vietstock_profile_raw=vietstock_profile_raw,
                vinacount_profile_candidate=vinacount_profile_candidate,
                year=year,
                quarter=quarter.upper(),
                source_entry_file_id=str(item.get("FileInfoID") or ""),
                source_entry_title=classified["title"],
                source_entry_url=classified["source_url"],
                last_update=item.get("LastUpdate"),
                basis_clues=classified["basis_clues"],
                document_type_guess=classified["document_type"],
                correction_signal_terms=classified["amendment_clues"],
                assurance_signal_terms=_assurance_signal_terms(classified["title"]),
                coverage_status="metadata_hit",
                ambiguity_notes=[],
            )
        )
    if rows:
        return rows
    return [
        VietstockQuarterlyMetadataRow(
            ticker=ticker,
            company_name_vi=company_name_vi,
            vietstock_profile_raw=vietstock_profile_raw,
            vinacount_profile_candidate=vinacount_profile_candidate,
            year=year,
            quarter=quarter.upper(),
            source_entry_file_id="",
            source_entry_title="",
            source_entry_url="",
            last_update=None,
            basis_clues=[],
            document_type_guess="unknown",
            correction_signal_terms=[],
            assurance_signal_terms=[],
            coverage_status="no_quarterly_metadata_hit",
            ambiguity_notes=["No matching Vietstock quarterly metadata entry found."],
        )
    ]


def fetch_vietstock_filing_package(
    *,
    ticker: str,
    company_name: str,
    year: int,
    quarter: str,
    report_basis: str,
    output_dir: Path | str,
    session: Any,
    fetched_at: str,
    base_url: str = VIETSTOCK_BASE_URL,
    max_pages: int = 30,
) -> VietstockFetchedFilingPackageResult:
    ticker = ticker.upper()
    period = f"{year}-{quarter.upper()}"
    page_url = f"{base_url}/{ticker}/tai-tai-lieu.htm?doctype=1"

    try:
        token = _get_request_verification_token(session, page_url)
    except Exception as exc:
        return _failed("recoverable_setup_error", "token_failure", str(exc))

    try:
        api_items = _fetch_document_metadata(
            session,
            base_url=base_url,
            ticker=ticker,
            year=year,
            token=token,
            page_url=page_url,
            max_pages=max_pages,
        )
    except Exception as exc:
        return _failed("recoverable_data_error", "api_parse_failure", str(exc))

    documents = []
    records: list[VietstockFetchRecord] = []
    same_period_classifications = []
    artifact_dir = Path(output_dir) / "vietstock_sources" / ticker / str(year) / quarter.upper()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for item in api_items:
        classified = _classify_document(item, period=period, year=year, quarter=quarter)
        if classified is None:
            continue
        classified = _align_non_consolidated_vietstock_basis(classified, report_basis=report_basis)
        same_period_classifications.append(classified)
        if not _vietstock_basis_matches_request(report_basis, classified["basis_clues"]):
            continue
        try:
            downloaded_documents = _download_source_documents(
                session,
                item=item,
                classified=classified,
                artifact_dir=artifact_dir,
                base_url=base_url,
                fetched_at=fetched_at,
                records=records,
            )
        except Exception as exc:
            return _failed("recoverable_data_error", "vietstock_download_failed", str(exc))
        documents.extend(downloaded_documents)

    if not documents:
        if same_period_classifications:
            return _failed("recoverable_data_error", "basis_mismatch", f"No exact Vietstock source found for {ticker} {period} {report_basis}; other basis candidates exist")
        return _failed("recoverable_data_error", "missing_exact_source", f"No exact Vietstock source found for {ticker} {period} {report_basis}")
    source_evidence, evidence_records = extract_source_evidence_for_documents(documents)
    records.extend(
        VietstockFetchRecord(
            severity=record.severity,
            code=record.code,
            message=record.message,
        )
        for record in evidence_records
    )
    raw_package = {
        "package_id": f"PKG_{ticker}_{year}_{quarter.upper()}_{report_basis.upper()}_FETCHED",
        "company": {"name": company_name, "ticker": ticker},
        "period": period,
        "quarter": quarter.upper(),
        "report_basis": report_basis,
        "filing_event": {
            "event_id": f"EVT_{ticker}_{year}_{quarter.upper()}_{report_basis.upper()}_FETCHED",
            "filing_status": _filing_status(documents),
        },
        "source_documents": documents,
        "source_evidence": source_evidence,
    }
    return VietstockFetchedFilingPackageResult(
        status="completed",
        package=validate_filing_package(raw_package),
        records=records,
    )


def _get_request_verification_token(session: Any, page_url: str) -> str:
    response = session.get(page_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    token_match = re.search(r"name=__RequestVerificationToken[^>]*value=([^\s>]+)", response.text)
    if not token_match:
        raise VietstockTokenError("missing Vietstock request verification token")
    return token_match.group(1).strip("\"'")


def _fetch_document_metadata(
    session: Any,
    *,
    base_url: str,
    ticker: str,
    year: int,
    token: str,
    page_url: str,
    max_pages: int,
) -> list[dict[str, Any]]:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": page_url,
    }
    items: list[dict[str, Any]] = []
    seen_ids = set()
    total_row = None
    page_size = None
    for page in range(1, max_pages + 1):
        payload = {
            "code": ticker,
            "page": page,
            "type": 1,
            "year": year,
            "__RequestVerificationToken": token,
        }
        response = session.post(
            f"{base_url}/data/getdocument",
            data=payload,
            headers=headers,
            timeout=60,
        )
        content_type = (response.headers.get("content-type") or "").lower()
        if "application/json" not in content_type:
            raise VietstockApiParseError(f"Vietstock API returned non-JSON content-type: {content_type}")
        page_items = response.json()
        if not isinstance(page_items, list):
            raise VietstockApiParseError("Vietstock API response must be a list")
        if not page_items:
            break
        if page_size is None:
            page_size = len(page_items)
        if total_row is None:
            total_row = _as_int(page_items[0].get("TotalRow"))
        for item in page_items:
            file_id = item.get("FileInfoID")
            if file_id in seen_ids:
                continue
            seen_ids.add(file_id)
            items.append(item)
        if total_row and len(items) >= total_row:
            break
        if page_size and len(page_items) < page_size:
            break
    return sorted(items, key=lambda item: str(item.get("LastUpdate") or item.get("FileInfoID") or ""), reverse=True)


def _classify_document(
    item: dict[str, Any],
    *,
    period: str,
    year: int,
    quarter: str,
) -> dict[str, Any] | None:
    title = str(item.get("Title") or "").strip()
    url = str(item.get("Url") or "").strip()
    if not title or not url:
        return None
    normalized = _normalize(title)
    quarter_number = quarter.upper().removeprefix("Q")
    if str(year) not in normalized or f"quy {quarter_number}" not in normalized:
        return None
    document_type = _document_type(normalized)
    if document_type is None:
        return None

    basis_clues = []
    if "hop nhat" in normalized:
        basis_clues.append("consolidated")
    if "cong ty me" in normalized:
        basis_clues.append("parent")
    if "rieng" in normalized:
        basis_clues.append("separate")
    if not basis_clues:
        basis_clues.append("parent")

    amendment_clues = _amendment_clues(normalized)
    if amendment_clues and document_type == "main_financial_statement":
        document_type = "amended_or_replacement_financial_statement"
    return {
        "title": title,
        "source_url": url,
        "document_type": document_type,
        "reported_period_clues": [period, f"Quy {quarter_number} nam {year}"],
        "basis_clues": basis_clues,
        "basis_clues_defaulted": basis_clues == ["parent"],
        "amendment_clues": amendment_clues,
    }


def _align_non_consolidated_vietstock_basis(classified: dict[str, Any], *, report_basis: str) -> dict[str, Any]:
    if report_basis == "separate" and classified.get("basis_clues_defaulted"):
        return {**classified, "basis_clues": ["separate"]}
    return classified


def _vietstock_basis_matches_request(report_basis: str, basis_clues: list[str]) -> bool:
    if report_basis in basis_clues:
        return True
    if report_basis in NON_CONSOLIDATED_BASIS_FAMILIES:
        return any(clue in NON_CONSOLIDATED_BASIS_FAMILIES for clue in basis_clues)
    return False


def _document_type(normalized_title: str) -> str | None:
    if "giai trinh" in normalized_title:
        return "variance_explanation_attachment"
    if "cong van" in normalized_title or "cong bo thong tin" in normalized_title:
        return "cover_letter"
    if "bao cao tai chinh" in normalized_title or "bctc" in normalized_title:
        return "main_financial_statement"
    return None


def _amendment_clues(normalized_title: str) -> list[str]:
    clues = []
    for marker in ["dieu chinh", "thay the", "bo sung", "sua doi"]:
        if marker in normalized_title:
            clues.append(marker)
    return clues


def _assurance_signal_terms(title: str) -> list[str]:
    normalized = _normalize(title)
    terms = []
    if "soat xet" in normalized:
        terms.append("soat xet")
    if "reviewed" in normalized:
        terms.append("reviewed")
    if "kiem toan" in normalized:
        terms.append("kiem toan")
    if "audited" in normalized:
        terms.append("audited")
    return terms


def _download_source_documents(
    session: Any,
    *,
    item: dict[str, Any],
    classified: dict[str, Any],
    artifact_dir: Path,
    base_url: str,
    fetched_at: str,
    records: list[VietstockFetchRecord],
) -> list[dict[str, Any]]:
    file_id = str(item["FileInfoID"])
    source_url = urljoin(base_url, classified["source_url"])
    response = session.get(source_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    if hasattr(response, "raise_for_status"):
        response.raise_for_status()
    content = response.content
    if not content:
        raise ValueError(f"empty Vietstock download for {file_id}")
    if _looks_like_zip(content, classified["source_url"]):
        return _unpack_vietstock_bundle(
            content,
            file_id=file_id,
            classified=classified,
            artifact_dir=artifact_dir,
            source_url=source_url,
            fetched_at=fetched_at,
            records=records,
        )
    return [
        _source_document(
            content=content,
            file_id=file_id,
            document_id=f"VIETSTOCK_{file_id}",
            artifact_id=f"ART_VIETSTOCK_{file_id}",
            artifact_path=artifact_dir / f"vietstock_{file_id}.pdf",
            classified=classified,
            source_url=source_url,
            fetched_at=fetched_at,
            file_type="pdf",
        )
    ]


def _unpack_vietstock_bundle(
    content: bytes,
    *,
    file_id: str,
    classified: dict[str, Any],
    artifact_dir: Path,
    source_url: str,
    fetched_at: str,
    records: list[VietstockFetchRecord],
) -> list[dict[str, Any]]:
    documents = []
    container_hash = hashlib.sha256(content).hexdigest()
    bundle_dir = artifact_dir / f"vietstock_{file_id}_members"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BytesIO(content)) as bundle:
        member_index = 0
        for member in bundle.infolist():
            if member.is_dir():
                continue
            member_bytes = bundle.read(member)
            if not _is_pdf_member(member.filename, member_bytes):
                records.append(
                    VietstockFetchRecord(
                        severity="recoverable_data_error",
                        code="source_quality_skipped_zip_member",
                        message=f"Skipped unreadable or non-PDF Vietstock ZIP member: {member.filename}",
                    )
                )
                continue
            member_index += 1
            safe_name = _safe_filename(member.filename)
            documents.append(
                _source_document(
                    content=member_bytes,
                    file_id=file_id,
                    document_id=f"VIETSTOCK_{file_id}_MEMBER_{member_index:03d}",
                    artifact_id=f"ART_VIETSTOCK_{file_id}_MEMBER_{member_index:03d}",
                    artifact_path=bundle_dir / safe_name,
                    classified=classified,
                    source_url=source_url,
                    fetched_at=fetched_at,
                    file_type="pdf",
                    container={
                        "file_type": "zip",
                        "file_id": file_id,
                        "source_url": source_url,
                        "hash_algorithm": "sha256",
                        "hash_value": container_hash,
                    },
                    member={
                        "filename": member.filename,
                        "file_size_bytes": member.file_size,
                        "compress_size_bytes": member.compress_size,
                    },
                )
            )
    if not documents:
        raise ValueError(f"Vietstock ZIP {file_id} contained no readable PDF members")
    return documents


def _source_document(
    *,
    content: bytes,
    file_id: str,
    document_id: str,
    artifact_id: str,
    artifact_path: Path,
    classified: dict[str, Any],
    source_url: str,
    fetched_at: str,
    file_type: str,
    container: dict[str, Any] | None = None,
    member: dict[str, Any] | None = None,
) -> dict[str, Any]:
    hash_value = hashlib.sha256(content).hexdigest()
    artifact_path.write_bytes(content)
    raw_metadata = {
        "vietstock_entry": {
            "file_id": file_id,
            "title": classified["title"],
            "url": classified["source_url"],
        },
        "file_type": file_type,
    }
    if container is not None:
        raw_metadata["vietstock_container"] = container
    if member is not None:
        raw_metadata["vietstock_member"] = member
    return {
        "document_id": document_id,
        "document_type": classified["document_type"],
        "provenance": {
            "source_kind": "fetched",
            "source_name": "vietstock",
            "source_url": source_url,
            "fetched_at": fetched_at,
        },
        "local_artifact": {
            "artifact_id": artifact_id,
            "path": str(artifact_path),
        },
        "fingerprint": {
            "hash_algorithm": "sha256",
            "hash_value": hash_value,
        },
        "reported_period_clues": classified["reported_period_clues"],
        "basis_clues": classified["basis_clues"],
        "amendment_clues": {
            "is_amended_or_replacement": bool(classified["amendment_clues"]),
            "clues": classified["amendment_clues"],
        },
        **raw_metadata,
    }


def _looks_like_zip(content: bytes, source_url: str) -> bool:
    return content.startswith(b"PK\x03\x04") or source_url.lower().endswith(".zip")


def _is_pdf_member(filename: str, content: bytes) -> bool:
    return filename.lower().endswith(".pdf") and content.lstrip().startswith(b"%PDF")


def _safe_filename(filename: str) -> str:
    name = Path(filename).name
    return re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip() or "member.pdf"


def _filing_status(documents: list[dict[str, Any]]) -> str:
    if any(document["amendment_clues"]["is_amended_or_replacement"] for document in documents):
        return "amended_or_replacement"
    return "original"


def _failed(severity: str, code: str, message: str) -> VietstockFetchedFilingPackageResult:
    return VietstockFetchedFilingPackageResult(
        status=severity,
        package=None,
        records=[VietstockFetchRecord(severity=severity, code=code, message=message)],
    )


def _normalize(value: str) -> str:
    value = value.replace("Đ", "D").replace("đ", "d")
    decomposed = unicodedata.normalize("NFKD", value)
    ascii_value = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", ascii_value).lower()


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
