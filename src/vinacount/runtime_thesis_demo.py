from __future__ import annotations

import copy
import hashlib
import html
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vinacount.report_model import validate_company_report_set, validate_report_memory
from vinacount.runtime_contract import HITL_BOUNDARY, stable_json_dumps
from vinacount.runtime_orchestration import RuntimeStageExecutionError
from vinacount.runtime_run_registry import FilesystemArtifactBodyStore


LOCKED_THESIS_DEMO_SCENARIO_ID = "nkg_2021_q3_consolidated"
LOCKED_THESIS_DEMO_CACHE_MODE = "locked_cached_report_memory"
CACHED_REAL_DEMO_CACHE_MODE = "cached_real_report_memory"
CACHED_FIRST_LIVE_CONFIRMATION_CACHE_MODE = "cached_first_live_confirmation"


@dataclass(frozen=True)
class _CachedSourceSpec:
    role: str
    runtime_role: str
    fiscal_year: int
    source_document_id: str
    url: str
    local_artifact_path: str
    source_sha256: str
    file_size_bytes: int
    page_count: int
    raw_ocr_artifact_id: str
    raw_ocr_path: str
    local_artifact_sha256: str | None = None
    source_artifact_kind: str = "vietstock_source_pdf"
    source_name: str = "Vietstock static filing PDF"


@dataclass(frozen=True)
class _CachedRealScenario:
    scenario_id: str
    candidate_id: str
    ticker: str
    company_name_vi: str
    target_fiscal_year: int
    target_quarter: int
    public_report_basis: str
    filing_visible_report_basis: str
    visible_basis_clue: str
    report_profile: str
    language: str
    clean_report_path: str
    extraction_probe_path: str
    target: _CachedSourceSpec
    prior_year: _CachedSourceSpec


CACHED_REAL_DEMO_SCENARIOS = (
    _CachedRealScenario(
        scenario_id=LOCKED_THESIS_DEMO_SCENARIO_ID,
        candidate_id="NBTP_NKG_2021_Q3_CONSOLIDATED",
        ticker="NKG",
        company_name_vi="CTCP Thep Nam Kim",
        target_fiscal_year=2021,
        target_quarter=3,
        public_report_basis="consolidated",
        filing_visible_report_basis="consolidated",
        visible_basis_clue="Hop nhat",
        report_profile="standard_corporate",
        language="vi",
        clean_report_path=(
            "artifacts/clean_structured_reports/"
            "clean_structured_report_nkg_2021_q3.json"
        ),
        extraction_probe_path=(
            "artifacts/clean_structured_reports/"
            "clean_structured_report_nkg_2021_q3.json"
        ),
        target=_CachedSourceSpec(
            role="target",
            runtime_role="target",
            fiscal_year=2021,
            source_document_id="DOC_NKG_2021_Q3_consolidated",
            url="https://static2.vietstock.vn/data/HOSE/2021/BCTC/VN/QUY%203/NKG_Baocaotaichinh_Q3_2021_Hopnhat.pdf",
            local_artifact_path="artifacts/source_documents/NBTP_NKG_2021_Q3_CONSOLIDATED__target.pdf",
            source_sha256="d87f379a420a35fbb07f5ead2ae45a974a1996e8587d6c81f10a777233eb8c22",
            file_size_bytes=9_623_301,
            page_count=43,
            raw_ocr_artifact_id="RAW_EXT_DOC_NKG_2021_Q3_consolidated",
            raw_ocr_path=(
                "artifacts/raw_extraction_artifacts/"
                "NBTP_NKG_2021_Q3_CONSOLIDATED__target__"
                "DOC_NKG_2021_Q3_consolidated__d87f379a420a__raw_extraction_artifact.json"
            ),
        ),
        prior_year=_CachedSourceSpec(
            role="prior_year",
            runtime_role="prior_year_same_quarter",
            fiscal_year=2020,
            source_document_id="DOC_NKG_2020_Q3_consolidated",
            url="https://static2.vietstock.vn/data/HOSE/2020/BCTC/VN/QUY%203/NKG_Baocaotaichinh_Q3_2020_Hopnhat.pdf",
            local_artifact_path="artifacts/source_documents/NBTP_NKG_2021_Q3_CONSOLIDATED__prior_year.pdf",
            source_sha256="d874b9ed079acc7664b68bdf54d540c4b9852e79f59a91aeae128974907211ba",
            file_size_bytes=9_498_545,
            page_count=42,
            raw_ocr_artifact_id="RAW_EXT_DOC_NKG_2020_Q3_consolidated",
            raw_ocr_path=(
                "artifacts/raw_extraction_artifacts/"
                "NBTP_NKG_2021_Q3_CONSOLIDATED__prior_year__"
                "DOC_NKG_2020_Q3_consolidated__d874b9ed079a__raw_extraction_artifact.json"
            ),
        ),
    ),
    _CachedRealScenario(
        scenario_id="hap_2024_q2_consolidated",
        candidate_id="NBTP_HAP_2024_Q2_CONSOLIDATED",
        ticker="HAP",
        company_name_vi="CTCP Tap doan Hapaco",
        target_fiscal_year=2024,
        target_quarter=2,
        public_report_basis="consolidated",
        filing_visible_report_basis="consolidated",
        visible_basis_clue="Hop nhat",
        report_profile="standard_corporate",
        language="vi",
        clean_report_path=(
            "artifacts/clean_structured_reports/"
            "clean_structured_report_hap_2024_q2.json"
        ),
        extraction_probe_path=(
            "artifacts/clean_structured_reports/"
            "clean_structured_report_hap_2024_q2.json"
        ),
        target=_CachedSourceSpec(
            role="target",
            runtime_role="target",
            fiscal_year=2024,
            source_document_id="DOC_HAP_2024_Q2_consolidated",
            url="https://static2.vietstock.vn/data/HOSE/2024/BCTC/VN/QUY%202/HAP_Baocaotaichinh_Q2_2024_Hopnhat.pdf",
            local_artifact_path="artifacts/source_documents/NBTP_HAP_2024_Q2_CONSOLIDATED__target.pdf",
            source_sha256="061d79b7490a31f2b0254cd10d1f951d5d1960840b9b2d880fab6b264592f916",
            file_size_bytes=12_403_753,
            page_count=43,
            raw_ocr_artifact_id="RAW_EXT_DOC_HAP_2024_Q2_consolidated",
            raw_ocr_path=(
                "artifacts/raw_extraction_artifacts/"
                "NBTP_HAP_2024_Q2_CONSOLIDATED__target__"
                "DOC_HAP_2024_Q2_consolidated__061d79b7490a__raw_extraction_artifact.json"
            ),
        ),
        prior_year=_CachedSourceSpec(
            role="prior_year",
            runtime_role="prior_year_same_quarter",
            fiscal_year=2023,
            source_document_id="DOC_HAP_2023_Q2_consolidated",
            url="https://static2.vietstock.vn/data/HOSE/2023/BCTC/VN/QUY%202/HAP_Baocaotaichinh_Q2_2023_Hopnhat.pdf",
            local_artifact_path="artifacts/source_documents/NBTP_HAP_2024_Q2_CONSOLIDATED__prior_year.pdf",
            source_sha256="e3ee50d093991667b1af9065f94ee015360c7062a69c1a29ed431903dc312a51",
            file_size_bytes=9_763_924,
            page_count=42,
            raw_ocr_artifact_id="RAW_EXT_DOC_HAP_2023_Q2_consolidated",
            raw_ocr_path=(
                "artifacts/raw_extraction_artifacts/"
                "NBTP_HAP_2024_Q2_CONSOLIDATED__prior_year__"
                "DOC_HAP_2023_Q2_consolidated__e3ee50d09399__raw_extraction_artifact.json"
            ),
        ),
    ),
    _CachedRealScenario(
        scenario_id="nhc_2024_q2_separate",
        candidate_id="NBTP_NHC_2024_Q2_PARENT",
        ticker="NHC",
        company_name_vi="CTCP Gach ngoi Nhi Hiep",
        target_fiscal_year=2024,
        target_quarter=2,
        public_report_basis="separate",
        filing_visible_report_basis="parent",
        visible_basis_clue="Cong ty me",
        report_profile="standard_corporate",
        language="vi",
        clean_report_path=(
            "artifacts/clean_structured_reports/"
            "clean_structured_report_nhc_2024_q2.json"
        ),
        extraction_probe_path=(
            "artifacts/clean_structured_reports/"
            "clean_structured_report_nhc_2024_q2.json"
        ),
        target=_CachedSourceSpec(
            role="target",
            runtime_role="target",
            fiscal_year=2024,
            source_document_id="DOC_NHC_2024_Q2_parent",
            url="https://static2.vietstock.vn/data/HNX/2024/BCTC/VN/QUY%202/NHC_Baocaotaichinh_Q2_2024_Congtyme.zip",
            local_artifact_path="artifacts/source_documents/NBTP_NHC_2024_Q2_PARENT__target.zip",
            source_sha256="50c768944594659e69e62211fa7708556ddd73d54e2c93f034d33266da92e7a4",
            file_size_bytes=12_182_204,
            page_count=29,
            raw_ocr_artifact_id="RAW_EXT_DOC_NHC_2024_Q2_parent",
            raw_ocr_path=(
                "artifacts/raw_extraction_artifacts/"
                "NBTP_NHC_2024_Q2_PARENT__target__"
                "DOC_NHC_2024_Q2_parent__50c768944594__raw_extraction_artifact.json"
            ),
            local_artifact_sha256="bb06c6e08d3a01ff1ee10c0b86f3df5b98f17fbef588787dabeb571eb61062ac",
            source_artifact_kind="vietstock_source_package",
            source_name="Vietstock static filing package",
        ),
        prior_year=_CachedSourceSpec(
            role="prior_year",
            runtime_role="prior_year_same_quarter",
            fiscal_year=2023,
            source_document_id="DOC_NHC_2023_Q2_parent",
            url="https://static2.vietstock.vn/data/HNX/2023/BCTC/VN/QUY%202/NHC_Baocaotaichinh_Q2_2023_Congtyme.zip",
            local_artifact_path="artifacts/source_documents/NBTP_NHC_2024_Q2_PARENT__prior_year.zip",
            source_sha256="4000527858a70ae02d9cb4b55921971913ce71d0ace447849834492f7764d226",
            file_size_bytes=8_599_862,
            page_count=28,
            raw_ocr_artifact_id="RAW_EXT_DOC_NHC_2023_Q2_parent",
            raw_ocr_path=(
                "artifacts/raw_extraction_artifacts/"
                "NBTP_NHC_2024_Q2_PARENT__prior_year__"
                "DOC_NHC_2023_Q2_parent__4000527858a7__raw_extraction_artifact.json"
            ),
            local_artifact_sha256="601b3c081df813c7f181bfacc7b46eda1b9e567782da475a13f1b9c1b701b12a",
            source_artifact_kind="vietstock_source_package",
            source_name="Vietstock static filing package",
        ),
    ),
)

LOCKED_THESIS_DEMO_SCENARIO = CACHED_REAL_DEMO_SCENARIOS[0]


def default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def locked_thesis_demo_enabled(
    *,
    scenario_id: str | None,
    cache_mode: str | None,
) -> bool:
    return (
        (scenario_id or "").strip() == LOCKED_THESIS_DEMO_SCENARIO_ID
        and (cache_mode or "").strip() == LOCKED_THESIS_DEMO_CACHE_MODE
    )


class CachedRealDemoSourceDiscoveryAdapter:
    def __init__(
        self,
        *,
        scenarios: tuple[_CachedRealScenario, ...] = CACHED_REAL_DEMO_SCENARIOS,
        project_root: Path | None = None,
        fallback: Any | None = None,
        unavailable_code: str = "cached_real_scenario_unavailable",
        unavailable_message: str = "No cached real source package matched this filing intent.",
    ) -> None:
        self._scenarios = scenarios
        self._project_root = project_root or default_project_root()
        self._fallback = fallback
        self._unavailable_code = unavailable_code
        self._unavailable_message = unavailable_message

    def discover(self, filing_intent: dict[str, Any]) -> dict[str, Any]:
        scenario = _resolve_scenario(filing_intent, self._scenarios)
        if scenario is None:
            if self._fallback is not None:
                return self._fallback.discover(filing_intent)
            return _unavailable_source_confirmation(self._unavailable_code, self._unavailable_message)
        try:
            target = self._candidate(scenario, scenario.target)
            prior_year = self._candidate(scenario, scenario.prior_year)
        except RuntimeStageExecutionError as error:
            return _unavailable_source_confirmation(error.code, error.message)
        return {
            "status": "ready_for_review",
            "confirmable": True,
            "hitl_boundary": HITL_BOUNDARY,
            "slots": [
                {
                    "role": "target",
                    "status": "ready_for_review",
                    "candidate": target,
                    "rejection": None,
                    "warnings": [],
                },
                {
                    "role": "prior_year_same_quarter",
                    "status": "ready_for_review",
                    "candidate": prior_year,
                    "rejection": None,
                    "warnings": [],
                },
            ],
            "package_warnings": [
                {
                    "code": "cached_real_report_memory_available",
                    "severity": "info",
                    "message": "Gói nguồn dùng tài liệu Vietstock đã ghim và hiện vật OCR thật đã lưu đệm.",
                    "stage_id": "source_discovery",
                    "artifact_refs": [],
                }
            ],
        }

    def _candidate(self, scenario: _CachedRealScenario, source: _CachedSourceSpec) -> dict[str, Any]:
        self._require_hash(source.local_artifact_path, _local_artifact_sha256(source), "source artifact")
        return {
            "source_document_id": source.source_document_id,
            "company_name_vi": scenario.company_name_vi,
            "ticker": scenario.ticker,
            "period_label": f"Q{scenario.target_quarter} {source.fiscal_year}",
            "quarter": scenario.target_quarter,
            "fiscal_year": source.fiscal_year,
            "report_basis": scenario.public_report_basis,
            "filing_status": "original",
            "document_type": "quarterly_bctc",
            "language": scenario.language,
            "source_origin": "Vietstock",
            "source_name": source.source_name,
            "source_url": source.url,
            "is_searchable_version": True,
            "file_size_bytes": source.file_size_bytes,
            "page_count": source.page_count,
            "visible_filing_label": (
                f"Báo cáo tài chính {_basis_clue_vi(scenario.visible_basis_clue)} "
                f"quý {scenario.target_quarter} năm {source.fiscal_year}"
            ),
            "first_page_identity": {
                "visible_company_name": scenario.company_name_vi,
                "visible_period": f"Quý {scenario.target_quarter} năm {source.fiscal_year}",
                "visible_basis_clue": _basis_clue_vi(scenario.visible_basis_clue),
            },
            "classification_evidence": [
                "Đăng ký nguồn đã ghim xác định đây là báo cáo tài chính quý từ Vietstock.",
                "Kiểm tra định danh nguồn khớp công ty, kỳ báo cáo và cơ sở lập báo cáo.",
                "Chẩn đoán OCR lưu đệm xác nhận các trang báo cáo đã được xử lý từ nguồn đã ghim.",
            ],
            "audit_references": {
                "cached_real_scenario_id": scenario.scenario_id,
                "locked_scenario_id": scenario.scenario_id,
                "source_candidate_id": scenario.candidate_id,
                "package_id": scenario.candidate_id,
                "event_id": f"cached_real_{scenario.scenario_id}",
                "canonical_source_document_id": source.source_document_id,
                "source_document_fingerprint_sha256": source.source_sha256,
                "source_identity_check_status": "ready",
                "public_report_basis": scenario.public_report_basis,
                "filing_visible_report_basis": scenario.filing_visible_report_basis,
                "ocr_artifact_id": source.raw_ocr_artifact_id,
                "cache_mode": self._cache_mode(),
                "demo_fixture_scope": self._demo_fixture_scope(),
            },
        }

    def _cache_mode(self) -> str:
        return CACHED_REAL_DEMO_CACHE_MODE

    def _demo_fixture_scope(self) -> str:
        return "cached_real_seed_fixture"

    def _require_hash(self, relative_path: str, expected_sha256: str, label: str) -> None:
        path = self._project_root / relative_path
        if not path.exists():
            raise RuntimeStageExecutionError(
                stage_id="source_discovery",
                code="source_package_unavailable",
                message=f"Cached real {label} is missing.",
                detail=relative_path,
            )
        actual = _sha256(path)
        if actual != expected_sha256:
            raise RuntimeStageExecutionError(
                stage_id="source_discovery",
                code="wrong_source_identity",
                message=f"Cached real {label} hash does not match the pinned artifact hash.",
                detail=f"{relative_path}: expected {expected_sha256}, got {actual}",
            )


class LockedThesisDemoSourceDiscoveryAdapter(CachedRealDemoSourceDiscoveryAdapter):
    def __init__(
        self,
        *,
        scenario: _CachedRealScenario = LOCKED_THESIS_DEMO_SCENARIO,
        project_root: Path | None = None,
    ) -> None:
        super().__init__(
            scenarios=(scenario,),
            project_root=project_root,
            fallback=None,
            unavailable_code="locked_thesis_demo_scenario_unavailable",
            unavailable_message="This runtime is locked to the NKG 2021 Q3 consolidated thesis-demo filing.",
        )

    def _cache_mode(self) -> str:
        return LOCKED_THESIS_DEMO_CACHE_MODE

    def _demo_fixture_scope(self) -> str:
        return "locked_thesis_demo_fixture_only"


class CachedRealDemoCacheLookupAdapter:
    def __init__(
        self,
        *,
        artifact_store: FilesystemArtifactBodyStore,
        scenarios: tuple[_CachedRealScenario, ...] = CACHED_REAL_DEMO_SCENARIOS,
        project_root: Path | None = None,
        fallback: Any | None = None,
    ) -> None:
        self._artifact_store = artifact_store
        self._scenarios = scenarios
        self._project_root = project_root or default_project_root()
        self._fallback = fallback

    def lookup(self, run_view: dict[str, Any]) -> dict[str, Any]:
        scenario = _resolve_scenario(run_view["filing_intent"], self._scenarios)
        if scenario is None:
            if self._fallback is not None:
                return self._fallback.lookup(run_view)
            return {
                "outcome": "miss",
                "reusable_report_memory_refs": [],
                "warnings": [
                    {
                        "code": "cached_real_report_memory_not_applicable",
                        "severity": "info",
                        "message": "Cached real ReportMemory lookup did not match this filing intent.",
                        "stage_id": "cache_lookup",
                        "artifact_refs": [],
                    }
                ],
            }
        try:
            clean_report = self._load_json(scenario.clean_report_path, "clean structured report")
            target_ocr = self._load_json(scenario.target.raw_ocr_path, "target OCR artifact")
            prior_ocr = self._load_json(scenario.prior_year.raw_ocr_path, "prior-year OCR artifact")
            self._validate_inputs(scenario, clean_report, target_ocr, prior_ocr)
            target_memory = self._report_memory(
                scenario,
                clean_report,
                scenario.target,
                value_key="current_value",
                ocr_artifact=target_ocr,
            )
            prior_year_memory = self._report_memory(
                scenario,
                clean_report,
                scenario.prior_year,
                value_key="prior_value",
                ocr_artifact=prior_ocr,
            )
            validate_company_report_set(
                f"{target_memory.report_id}_VS_{prior_year_memory.report_id}",
                target_memory,
                prior_year_memory,
            )
            artifact_refs = self._source_and_cache_artifact_refs(run_view["run_id"], scenario)
            reusable_refs = [
                self._put_report_memory(
                    run_id=run_view["run_id"],
                    role="target",
                    artifact_id=f"artifact_{scenario.scenario_id}_report_memory_target",
                    report_memory=target_memory.raw,
                    scenario=scenario,
                    source=scenario.target,
                    clean_report=clean_report,
                    ocr_artifact=target_ocr,
                ),
                self._put_report_memory(
                    run_id=run_view["run_id"],
                    role="prior_year_same_quarter",
                    artifact_id=f"artifact_{scenario.scenario_id}_report_memory_prior_year",
                    report_memory=prior_year_memory.raw,
                    scenario=scenario,
                    source=scenario.prior_year,
                    clean_report=clean_report,
                    ocr_artifact=prior_ocr,
                ),
            ]
        except RuntimeStageExecutionError:
            raise
        except Exception as error:
            raise RuntimeStageExecutionError(
                stage_id="cache_lookup",
                code="cache_lookup_failed",
                message="Cached real ReportMemory replay failed.",
                detail=str(error),
                recoverable=True,
                can_resume=True,
            ) from error
        return {
            "outcome": "report_memory_reusable",
            "artifact_refs": artifact_refs,
            "reusable_report_memory_refs": reusable_refs,
            "warnings": [
                {
                    "code": "cached_real_report_memory_replay",
                    "severity": "info",
                    "message": "Bằng chứng OCR thật đã lưu đệm được tái sử dụng thành hiện vật ReportMemory cho lượt phân tích.",
                    "stage_id": "cache_lookup",
                    "artifact_refs": copy.deepcopy(reusable_refs),
                }
            ],
        }

    def _load_json(self, relative_path: str, label: str) -> dict[str, Any]:
        path = self._project_root / relative_path
        if not path.exists():
            raise RuntimeStageExecutionError(
                stage_id="cache_lookup",
                code="cache_lookup_failed",
                message=f"Cached real {label} is unavailable.",
                detail=relative_path,
                recoverable=True,
                can_resume=True,
            )
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise RuntimeStageExecutionError(
                stage_id="cache_lookup",
                code="cache_lookup_failed",
                message=f"Cached real {label} is not valid JSON.",
                detail=str(error),
                recoverable=True,
                can_resume=True,
            ) from error

    def _validate_inputs(
        self,
        scenario: _CachedRealScenario,
        clean_report: dict[str, Any],
        target_ocr: dict[str, Any],
        prior_ocr: dict[str, Any],
    ) -> None:
        _require_equal(clean_report.get("ticker"), scenario.ticker, "clean_report.ticker")
        _require_equal(
            clean_report.get("period"),
            f"{scenario.target_fiscal_year}-Q{scenario.target_quarter}",
            "clean_report.period",
        )
        _require_equal(
            clean_report.get("traceability", {}).get("source_file_sha256"),
            scenario.target.source_sha256,
            "clean_report.traceability.source_file_sha256",
        )
        _require_equal(target_ocr.get("artifact_id"), scenario.target.raw_ocr_artifact_id, "target_ocr.artifact_id")
        _require_equal(prior_ocr.get("artifact_id"), scenario.prior_year.raw_ocr_artifact_id, "prior_ocr.artifact_id")
        self._require_hash(
            scenario.target.local_artifact_path,
            _local_artifact_sha256(scenario.target),
            "target source artifact",
        )
        self._require_hash(
            scenario.prior_year.local_artifact_path,
            _local_artifact_sha256(scenario.prior_year),
            "prior-year source artifact",
        )

    def _report_memory(
        self,
        scenario: _CachedRealScenario,
        clean_report: dict[str, Any],
        source: _CachedSourceSpec,
        *,
        value_key: str,
        ocr_artifact: dict[str, Any],
    ):
        report_id = f"{scenario.ticker}_{source.fiscal_year}_Q{scenario.target_quarter}_{scenario.public_report_basis.upper()}"
        rows = []
        page_segments = _ocr_page_segments(ocr_artifact)
        for item in clean_report["structured_evidence"]["rows"]:
            standard_account = item["standard_account"]
            row_id = f"ROW_{standard_account.upper()}_{source.fiscal_year}_Q{scenario.target_quarter}"
            navigation = _source_navigation_for_value(
                item[value_key],
                page_segments,
                standard_account=standard_account,
            )
            rows.append(
                {
                    "row_id": row_id,
                    "standard_account": standard_account,
                    "account_code": _account_code(standard_account),
                    "label": _account_label(standard_account),
                    "cells": [
                        {
                            "cell_id": f"CELL_{row_id}_{report_id}",
                            "row_id": row_id,
                            "period": f"{source.fiscal_year}-Q{scenario.target_quarter}",
                            "value": item[value_key],
                            "source_page_number": navigation["page_number"],
                            "source_excerpt": navigation["source_excerpt"],
                        }
                    ],
                }
            )
        cells = [cell for row in rows for cell in row["cells"]]
        raw = {
            "report_id": report_id,
            "metadata": {
                "company_name": scenario.company_name_vi,
                "period": f"{source.fiscal_year}-Q{scenario.target_quarter}",
                "report_period_type": "quarterly",
                "report_profile": scenario.report_profile,
                "report_basis": scenario.public_report_basis,
                "filing_visible_report_basis": scenario.filing_visible_report_basis,
                "business_context_tags": ["thesis_demo", "vietstock_cached_real_ocr"],
                "report_assurance_type": "unaudited",
                "currency": clean_report["currency"],
                "unit": clean_report["unit"],
                "filing_status": "original",
                "canonical_source_document_id": source.source_document_id,
                "source_file": source.url,
                "extraction_method": "cached_real_ocr_report_memory_replay_v1",
                "source_document_fingerprint_sha256": source.source_sha256,
                "ocr_artifact_id": source.raw_ocr_artifact_id,
                "cached_real_scenario_id": scenario.scenario_id,
            },
            "sections": [
                {"section_id": "SEC_BALANCE_SHEET", "section_type": "balance_sheet", "title": "Balance Sheet"},
                {"section_id": "SEC_INCOME_STATEMENT", "section_type": "income_statement", "title": "Income Statement"},
                {"section_id": "SEC_CASH_FLOW", "section_type": "cash_flow_statement", "title": "Cash Flow Statement"},
            ],
            "tables": [
                {
                    "table_id": f"TBL_{report_id}_CACHED_REAL_REPLAY",
                    "table_type": "income_statement",
                    "rows": rows,
                }
            ],
            "notes": copy.deepcopy(clean_report.get("structured_evidence", {}).get("notes", [])),
            "variance_explanations": [],
            "cell_index": {
                cell["cell_id"]: {
                    "table_id": f"TBL_{report_id}_CACHED_REAL_REPLAY",
                    "row_id": cell["row_id"],
                }
                for cell in cells
            },
        }
        return validate_report_memory(raw)

    def _source_and_cache_artifact_refs(
        self,
        run_id: str,
        scenario: _CachedRealScenario,
    ) -> list[dict[str, Any]]:
        return [
            self._reference_existing(
                run_id=run_id,
                artifact_id=f"artifact_{scenario.scenario_id}_source_target",
                kind=scenario.target.source_artifact_kind,
                relative_path=scenario.target.local_artifact_path,
                schema_version="vietstock_filing_source.v1",
                metadata=self._source_metadata(scenario, scenario.target),
            ),
            self._reference_existing(
                run_id=run_id,
                artifact_id=f"artifact_{scenario.scenario_id}_source_prior_year",
                kind=scenario.prior_year.source_artifact_kind,
                relative_path=scenario.prior_year.local_artifact_path,
                schema_version="vietstock_filing_source.v1",
                metadata=self._source_metadata(scenario, scenario.prior_year),
            ),
            self._reference_existing(
                run_id=run_id,
                artifact_id=f"artifact_{scenario.scenario_id}_ocr_target",
                kind="ocr_artifact_json",
                relative_path=scenario.target.raw_ocr_path,
                schema_version="raw_extraction_artifact.v1",
                metadata=self._ocr_metadata(scenario, scenario.target),
            ),
            self._reference_existing(
                run_id=run_id,
                artifact_id=f"artifact_{scenario.scenario_id}_ocr_prior_year",
                kind="ocr_artifact_json",
                relative_path=scenario.prior_year.raw_ocr_path,
                schema_version="raw_extraction_artifact.v1",
                metadata=self._ocr_metadata(scenario, scenario.prior_year),
            ),
            self._reference_existing(
                run_id=run_id,
                artifact_id=f"artifact_{scenario.scenario_id}_clean_structured_report",
                kind="clean_structured_report_json",
                relative_path=scenario.clean_report_path,
                schema_version="clean_structured_report.v1",
                metadata={
                    "cached_real_scenario_id": scenario.scenario_id,
                    "locked_scenario_id": scenario.scenario_id,
                    "source_candidate_id": scenario.candidate_id,
                    "target_source_document_id": scenario.target.source_document_id,
                    "target_source_document_fingerprint_sha256": scenario.target.source_sha256,
                    "public_report_basis": scenario.public_report_basis,
                    "filing_visible_report_basis": scenario.filing_visible_report_basis,
                    "cache_readiness_status": "valid",
                },
            ),
        ]

    def _put_report_memory(
        self,
        *,
        run_id: str,
        role: str,
        artifact_id: str,
        report_memory: dict[str, Any],
        scenario: _CachedRealScenario,
        source: _CachedSourceSpec,
        clean_report: dict[str, Any],
        ocr_artifact: dict[str, Any],
    ) -> dict[str, Any]:
        return self._artifact_store.put_bytes(
            run_id=run_id,
            artifact_id=artifact_id,
            kind="report_memory_json",
            body=stable_json_dumps(report_memory).encode("utf-8"),
            schema_version="report_memory.v1",
            version="cached_real_report_memory_replay_v1",
            metadata={
                "report_role": role,
                "cached_real_scenario_id": scenario.scenario_id,
                "locked_scenario_id": scenario.scenario_id,
                "cache_mode": self._cache_mode(),
                "demo_fixture_scope": self._demo_fixture_scope(),
                "cache_readiness_status": "valid",
                "public_report_basis": scenario.public_report_basis,
                "filing_visible_report_basis": scenario.filing_visible_report_basis,
                "source_document_id": source.source_document_id,
                "source_document_fingerprint_sha256": source.source_sha256,
                "ocr_artifact_id": source.raw_ocr_artifact_id,
                "ocr_artifact_sha256": _sha256(self._project_root / source.raw_ocr_path),
                "ocr_extraction_method": ocr_artifact.get("extraction_method"),
                "ocr_extraction_version": ocr_artifact.get("extraction_version"),
                "clean_structured_report_artifact_id": clean_report.get("artifact_id"),
                "clean_structured_report_sha256": _sha256(self._project_root / scenario.clean_report_path),
                "source_trace_normalized_text_hash": clean_report.get("traceability", {}).get("normalized_text_hash"),
                "source_trace_table_content_hash": clean_report.get("traceability", {}).get("table_content_hash"),
            },
        )

    def _reference_existing(
        self,
        *,
        run_id: str,
        artifact_id: str,
        kind: str,
        relative_path: str,
        schema_version: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        path = self._project_root / relative_path
        if not path.exists():
            raise RuntimeStageExecutionError(
                stage_id="cache_lookup",
                code="cache_lookup_failed",
                message="Cached real cache artifact is unavailable.",
                detail=relative_path,
                recoverable=True,
                can_resume=True,
            )
        return self._artifact_store.reference_existing(
            run_id=run_id,
            artifact_id=artifact_id,
            kind=kind,
            path=path,
            version="cached_real_demo_v1",
            schema_version=schema_version,
            metadata=metadata,
        )

    def _source_metadata(self, scenario: _CachedRealScenario, source: _CachedSourceSpec) -> dict[str, Any]:
        return {
            "cached_real_scenario_id": scenario.scenario_id,
            "locked_scenario_id": scenario.scenario_id,
            "source_candidate_id": scenario.candidate_id,
            "report_role": source.runtime_role,
            "source_document_id": source.source_document_id,
            "source_document_fingerprint_sha256": source.source_sha256,
            "source_url": source.url,
            "source_origin": "Vietstock",
            "source_identity_check_status": "ready",
            "public_report_basis": scenario.public_report_basis,
            "filing_visible_report_basis": scenario.filing_visible_report_basis,
            "cache_mode": self._cache_mode(),
            "demo_fixture_scope": self._demo_fixture_scope(),
        }

    def _ocr_metadata(self, scenario: _CachedRealScenario, source: _CachedSourceSpec) -> dict[str, Any]:
        return {
            "cached_real_scenario_id": scenario.scenario_id,
            "locked_scenario_id": scenario.scenario_id,
            "source_candidate_id": scenario.candidate_id,
            "report_role": source.runtime_role,
            "source_document_id": source.source_document_id,
            "source_document_fingerprint_sha256": source.source_sha256,
            "public_report_basis": scenario.public_report_basis,
            "filing_visible_report_basis": scenario.filing_visible_report_basis,
            "ocr_artifact_id": source.raw_ocr_artifact_id,
            "cache_readiness_status": "valid",
            "cache_mode": self._cache_mode(),
            "demo_fixture_scope": self._demo_fixture_scope(),
        }

    def _cache_mode(self) -> str:
        return CACHED_REAL_DEMO_CACHE_MODE

    def _demo_fixture_scope(self) -> str:
        return "cached_real_seed_fixture"

    def _require_hash(self, relative_path: str, expected_sha256: str, label: str) -> None:
        path = self._project_root / relative_path
        if not path.exists():
            raise RuntimeStageExecutionError(
                stage_id="cache_lookup",
                code="cache_lookup_failed",
                message=f"Cached real {label} is unavailable.",
                detail=relative_path,
                recoverable=True,
                can_resume=True,
            )
        actual = _sha256(path)
        if actual != expected_sha256:
            raise RuntimeStageExecutionError(
                stage_id="cache_lookup",
                code="cache_lookup_failed",
                message=f"Cached real {label} hash does not match the pinned artifact hash.",
                detail=f"{relative_path}: expected {expected_sha256}, got {actual}",
                recoverable=True,
                can_resume=True,
            )


class LockedThesisDemoCacheLookupAdapter(CachedRealDemoCacheLookupAdapter):
    def __init__(
        self,
        *,
        artifact_store: FilesystemArtifactBodyStore,
        scenario: _CachedRealScenario = LOCKED_THESIS_DEMO_SCENARIO,
        project_root: Path | None = None,
    ) -> None:
        super().__init__(
            artifact_store=artifact_store,
            scenarios=(scenario,),
            project_root=project_root,
            fallback=None,
        )

    def _cache_mode(self) -> str:
        return LOCKED_THESIS_DEMO_CACHE_MODE

    def _demo_fixture_scope(self) -> str:
        return "locked_thesis_demo_fixture_only"


class CachedFirstLiveConfirmationCacheLookupAdapter(CachedRealDemoCacheLookupAdapter):
    def __init__(
        self,
        *,
        artifact_store: FilesystemArtifactBodyStore,
        scenarios: tuple[_CachedRealScenario, ...] = CACHED_REAL_DEMO_SCENARIOS,
        project_root: Path | None = None,
    ) -> None:
        super().__init__(
            artifact_store=artifact_store,
            scenarios=scenarios,
            project_root=project_root,
            fallback=None,
        )

    def lookup(self, run_view: dict[str, Any]) -> dict[str, Any]:
        scenario = _resolve_scenario(run_view["filing_intent"], self._scenarios)
        if scenario is None:
            return _blocked_cache_lookup_result(
                "cached_first_live_confirmation_scenario_unavailable",
                "Cached-first live confirmation is limited to the locked thesis demo scenarios.",
            )
        mismatch = _confirmed_source_identity_mismatch(run_view, scenario)
        if mismatch is not None:
            return _blocked_cache_lookup_result(
                "cached_first_live_confirmation_source_mismatch",
                mismatch,
            )
        result = super().lookup(run_view)
        result["source_confirmation_candidate_overrides"] = (
            _source_confirmation_candidate_overrides_for_locked_artifacts(scenario)
        )
        return result

    def _cache_mode(self) -> str:
        return CACHED_FIRST_LIVE_CONFIRMATION_CACHE_MODE

    def _demo_fixture_scope(self) -> str:
        return "cached_first_live_confirmation"


def _unavailable_source_confirmation(code: str, message: str) -> dict[str, Any]:
    warning = {
        "code": code,
        "severity": "warning",
        "message": message,
        "stage_id": "source_discovery",
        "artifact_refs": [],
    }
    return {
        "status": "partially_rejected",
        "confirmable": False,
        "hitl_boundary": HITL_BOUNDARY,
        "slots": [
            {
                "role": "target",
                "status": "unavailable",
                "candidate": None,
                "rejection": None,
                "warnings": [copy.deepcopy(warning)],
            },
            {
                "role": "prior_year_same_quarter",
                "status": "unavailable",
                "candidate": None,
                "rejection": None,
                "warnings": [copy.deepcopy(warning)],
            },
        ],
        "package_warnings": [warning],
    }


def _resolve_scenario(
    filing_intent: dict[str, Any],
    scenarios: tuple[_CachedRealScenario, ...],
) -> _CachedRealScenario | None:
    for scenario in scenarios:
        if (
            str(filing_intent.get("company_identifier", "")).upper() == scenario.ticker
            and filing_intent.get("target_fiscal_year") == scenario.target_fiscal_year
            and filing_intent.get("target_quarter") == scenario.target_quarter
            and filing_intent.get("report_basis_preference") == scenario.public_report_basis
        ):
            return scenario
    return None


def _blocked_cache_lookup_result(code: str, message: str) -> dict[str, Any]:
    warning = {
        "code": code,
        "severity": "warning",
        "message": message,
        "stage_id": "cache_lookup",
        "artifact_refs": [],
    }
    return {
        "outcome": "invalid_blocked",
        "reusable_report_memory_refs": [],
        "warnings": [warning],
    }


def _confirmed_source_identity_mismatch(
    run_view: dict[str, Any],
    scenario: _CachedRealScenario,
) -> str | None:
    slots = {
        slot.get("role"): slot
        for slot in run_view.get("source_confirmation", {}).get("slots", [])
        if isinstance(slot, dict)
    }
    checks = (
        ("target", scenario.target),
        ("prior_year_same_quarter", scenario.prior_year),
    )
    for role, expected in checks:
        slot = slots.get(role)
        candidate = slot.get("candidate") if isinstance(slot, dict) else None
        if not isinstance(candidate, dict):
            return f"Missing confirmed source candidate for {role}."
        audit = candidate.get("audit_references")
        if not isinstance(audit, dict):
            return f"Confirmed source candidate for {role} is missing audit references."
        actual_fingerprint = audit.get("source_document_fingerprint_sha256")
        if actual_fingerprint != expected.source_sha256:
            return (
                f"Confirmed source fingerprint for {role} does not match the locked cached artifact."
            )
    return None


def _source_confirmation_candidate_overrides_for_locked_artifacts(
    scenario: _CachedRealScenario,
) -> dict[str, dict[str, Any]]:
    return {
        "target": _source_confirmation_candidate_override(scenario.target),
        "prior_year_same_quarter": _source_confirmation_candidate_override(scenario.prior_year),
    }


def _source_confirmation_candidate_override(source: _CachedSourceSpec) -> dict[str, Any]:
    return {
        "source_document_id": source.source_document_id,
        "source_url": source.url,
        "source_name": source.source_name,
        "file_size_bytes": source.file_size_bytes,
        "page_count": source.page_count,
        "audit_references": {
            "canonical_source_document_id": source.source_document_id,
            "linked_canonical_source_document_id": source.source_document_id,
            "source_document_fingerprint_sha256": source.source_sha256,
        },
    }


def _account_code(standard_account: str) -> str:
    return {
        "revenue": "10",
        "trade_receivables": "131",
        "net_profit": "60",
        "operating_cash_flow": "20",
    }.get(standard_account, standard_account.upper())


def _account_label(standard_account: str) -> str:
    return {
        "revenue": "Revenue",
        "trade_receivables": "Trade receivables",
        "net_profit": "Net profit",
        "operating_cash_flow": "Operating cash flow",
    }.get(standard_account, standard_account.replace("_", " ").title())


def _ocr_page_segments(ocr_artifact: dict[str, Any]) -> list[dict[str, Any]]:
    raw_html = ocr_artifact.get("raw_html")
    if not isinstance(raw_html, str) or not raw_html:
        return []
    try:
        content = json.loads(raw_html).get("content")
    except json.JSONDecodeError:
        content = raw_html
    if not isinstance(content, str) or not content:
        return []
    matches = list(re.finditer(r"<h2>Page\s+(\d+)</h2>", content))
    segments: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        segments.append(
            {
                "page_number": int(match.group(1)),
                "html": content[match.end() : end],
            }
        )
    return segments


def _source_navigation_for_value(
    value: Any,
    page_segments: list[dict[str, Any]],
    *,
    standard_account: str | None = None,
) -> dict[str, Any]:
    formatted_values = _format_source_value_variants(value)
    if not formatted_values:
        return {"page_number": None, "source_excerpt": None}
    candidates = _source_row_candidates_for_value_and_account(
        formatted_values,
        page_segments,
        standard_account=standard_account,
    )
    primary_matches = [candidate for candidate in candidates if candidate["is_primary_statement_row"]]
    matches = primary_matches or candidates
    if len(matches) != 1:
        return {"page_number": None, "source_excerpt": None}
    return {
        "page_number": matches[0]["page_number"],
        "source_excerpt": _bounded_source_row_excerpt(matches[0]["row_html"], matches[0]["formatted_value"]),
    }


def _source_row_candidates_for_value_and_account(
    formatted_values: list[str],
    page_segments: list[dict[str, Any]],
    *,
    standard_account: str | None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    labels = _account_label_synonyms(standard_account)
    account_code = _account_code(standard_account) if standard_account else None
    for segment in page_segments:
        for row_match in re.finditer(r"<tr\b[^>]*>.*?</tr>", segment["html"], flags=re.IGNORECASE | re.DOTALL):
            row_html = row_match.group(0)
            formatted_value = next((candidate for candidate in formatted_values if candidate in row_html), None)
            if formatted_value is None:
                continue
            row_text = _source_row_text(row_html)
            normalized_row_text = _normalize_source_text(row_text)
            label_matched = not labels or any(label in normalized_row_text for label in labels)
            code_matched = (
                bool(account_code)
                and re.search(rf"(?:^|\s){re.escape(account_code)}(?:\s|$)", normalized_row_text) is not None
            )
            if not label_matched and not code_matched:
                continue
            candidates.append(
                {
                    "page_number": segment["page_number"],
                    "row_html": row_html,
                    "formatted_value": formatted_value,
                    "is_primary_statement_row": label_matched and code_matched,
                }
            )
    return candidates


def _format_source_value_variants(value: Any) -> list[str]:
    if isinstance(value, int):
        absolute_value = abs(value)
        comma_value = f"{absolute_value:,}"
        dot_value = comma_value.replace(",", ".")
        variants = [comma_value, dot_value]
        if value < 0:
            variants.extend([f"({comma_value})", f"({dot_value})"])
        return list(dict.fromkeys(variants))
    return []


def _account_label_synonyms(standard_account: str | None) -> list[str]:
    labels = {
        "revenue": [
            "doanh thu thuan ve ban hang va cung cap dich vu",
            "doanh thu thuan ban hang va cung cap dich vu",
        ],
        "trade_receivables": [
            "phai thu ngan han cua khach hang",
            "phai thu cua khach hang ngan han",
            "phai thu khach hang",
        ],
        "net_profit": [
            "loi nhuan sau thue thu nhap doanh nghiep",
            "loi nhuan ke toan sau thue tndn",
        ],
        "operating_cash_flow": [
            "luu chuyen tien thuan tu hoat dong kinh doanh",
        ],
    }
    return labels.get(standard_account or "", [])


def _source_row_text(row_html: str) -> str:
    text = re.sub(r"<br\s*/?>", " ", row_html, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_source_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = "".join(character for character in normalized if not unicodedata.combining(character))
    return re.sub(r"\s+", " ", ascii_text.lower()).strip()


def _bounded_source_row_excerpt(page_html: str, formatted_value: str) -> str | None:
    match = re.search(
        rf"<tr\b[^>]*>.*?{re.escape(formatted_value)}.*?</tr>",
        page_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    source = match.group(0) if match else formatted_value
    text = re.sub(r"<br\s*/?>", " ", source, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None
    return text[:500]


def _local_artifact_sha256(source: _CachedSourceSpec) -> str:
    return source.local_artifact_sha256 or source.source_sha256


def _basis_clue_vi(value: str) -> str:
    normalized = value.strip().lower()
    if normalized == "hop nhat":
        return "hợp nhất"
    if normalized == "cong ty me":
        return "công ty mẹ"
    if normalized == "rieng":
        return "riêng lẻ"
    return value


def _require_equal(actual: Any, expected: Any, field: str) -> None:
    if actual != expected:
        raise RuntimeStageExecutionError(
            stage_id="cache_lookup",
            code="cache_lookup_failed",
            message="Cached real cache artifact identity did not match the pinned scenario.",
            detail=f"{field}: expected {expected!r}, got {actual!r}",
            recoverable=True,
            can_resume=True,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
