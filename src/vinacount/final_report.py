from __future__ import annotations

import json
import copy
import re
from pathlib import Path
from typing import Any

from vinacount.report_model import CompanyReportSet, ReportMemory


def build_final_report(
    report_set: CompanyReportSet,
    candidates: list[dict[str, Any]],
    tool_findings: list[dict[str, Any]],
    detector_assessments: list[dict[str, Any]],
    gating_records: list[dict[str, Any]],
    audit_records: list[dict[str, Any]] | None = None,
    report_generation_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if audit_records is None:
        audit_records = []
    generation_context = _final_report_generation_context(report_set, report_generation_context)
    source_lineage_by_report_id = _source_lineage_by_report_id(report_set)
    source_navigation_by_ref_id = _source_navigation_by_ref_id(report_set)
    candidates_by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    tool_findings_by_id = {finding["tool_result_id"]: finding for finding in tool_findings}
    supported = [
        assessment for assessment in detector_assessments if assessment["support_level"] == "supported"
    ]
    weak = [
        assessment
        for assessment in detector_assessments
        if assessment["support_level"] == "weakly_supported"
    ]
    grouped_findings = _group_supported_assessments(
        supported,
        candidates_by_id,
        tool_findings_by_id,
        generation_context,
        source_lineage_by_report_id,
        source_navigation_by_ref_id,
    )
    weak_or_limited = [
        _weak_signal_item(
            assessment,
            candidates_by_id,
            tool_findings_by_id,
            source_lineage_by_report_id,
            source_navigation_by_ref_id,
            generation_context["report_language"],
        )
        for assessment in weak
    ]
    reviewed_candidate_audit = [
        _reviewed_candidate_audit_item(
            assessment,
            candidates_by_id,
            tool_findings_by_id,
            source_lineage_by_report_id,
            source_navigation_by_ref_id,
            generation_context["report_language"],
        )
        for assessment in detector_assessments
    ]
    limitations = _final_report_limitations(
        report_set,
        detector_assessments,
        gating_records,
        generation_context,
    )
    coverage_limitations = _final_report_coverage_limitations(
        report_set.target,
        report_language=generation_context["report_language"],
    )
    overall = _final_report_overall_assessment(
        grouped_findings,
        weak_or_limited,
        limitations,
        generation_context,
    )
    report = {
        "report_id": f"FINAL_{report_set.target.report_id}",
        "report_basis": generation_context["report_basis"],
        "target_report_id": report_set.target.report_id,
        "metadata": _final_report_metadata(
            report_set.target,
            report_language=generation_context["report_language"],
        ),
        "executive_summary": _final_report_executive_summary(
            grouped_findings,
            weak_or_limited,
            limitations,
            report_language=generation_context["report_language"],
        ),
        "overall_assessment": overall,
        "grouped_findings": grouped_findings,
        "weak_or_limited_signals": weak_or_limited,
        "reviewed_candidate_audit": reviewed_candidate_audit,
        "insufficient_evidence_or_data_gaps": [
            item
            for item in reviewed_candidate_audit
            if item["support_level"] == "insufficient_evidence"
        ],
        "method_and_scope": generation_context["method_and_scope"],
        "limitations": limitations,
        "coverage_limitations": coverage_limitations,
        "audit_trail": {
            "candidate_ids": [candidate["candidate_id"] for candidate in candidates],
            "tool_result_ids": [finding["tool_result_id"] for finding in tool_findings],
            "assessment_ids": [assessment["assessment_id"] for assessment in detector_assessments],
            "gating_record_count": len(gating_records),
        },
    }
    validate_final_report(report)
    audit_records.append(
        {
            "report_id": report["report_id"],
            "target_report_id": report_set.target.report_id,
            "finding_count": len(grouped_findings),
            "weak_signal_count": len(weak_or_limited),
            "reviewed_candidate_count": len(reviewed_candidate_audit),
            "report_generation_basis": generation_context["report_basis"],
            "source_evidence_mode": generation_context["mode"],
            "report_generation_mode": generation_context["report_generation_mode"],
        }
    )
    return report


def write_final_report(output_path: Path, final_report: dict[str, Any]) -> dict[str, Path]:
    json_path = output_path / f"{final_report['report_id']}.json"
    markdown_path = output_path / f"{final_report['report_id']}.md"
    json_path.write_text(json.dumps(final_report, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(render_final_report_markdown(final_report))
    return {"json": json_path, "markdown": markdown_path}


def validate_final_report(report: dict[str, Any]) -> None:
    required_fields = {
        "report_id",
        "report_basis",
        "target_report_id",
        "metadata",
        "executive_summary",
        "overall_assessment",
        "grouped_findings",
        "weak_or_limited_signals",
        "reviewed_candidate_audit",
        "insufficient_evidence_or_data_gaps",
        "method_and_scope",
        "limitations",
        "coverage_limitations",
        "audit_trail",
    }
    missing_fields = required_fields - report.keys()
    if missing_fields:
        raise ValueError(f"FinalReport is missing fields: {sorted(missing_fields)}")
    required_metadata = {
        "report_basis",
        "filing_status",
        "canonical_source_document_id",
        "business_context_tags",
    }
    missing_metadata = required_metadata - report["metadata"].keys()
    if missing_metadata:
        raise ValueError(f"FinalReport metadata is missing fields: {sorted(missing_metadata)}")
    for finding in report["grouped_findings"]:
        if not set(finding["support_levels"]) <= {"supported"}:
            raise ValueError("FinalReport grouped findings require supported detector assessments")
    for item in report["weak_or_limited_signals"]:
        if item["support_level"] != "weakly_supported":
            raise ValueError("FinalReport weak section requires weakly_supported assessments")
        if item["final_severity"] == "high":
            raise ValueError("FinalReport must not assign high severity to weakly supported findings")
    _reject_prohibited_final_report_text(report)


def render_final_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Financial Reporting Risk-Signal Review",
        "",
        "## Report Metadata",
        "",
        "| Field | Value |",
        "|---|---|",
    ]
    for key in [
        "company_name",
        "period",
        "report_profile",
        "report_basis",
        "filing_status",
        "canonical_source_document_id",
        "business_context_tags",
        "report_assurance_type",
        "currency",
        "unit",
    ]:
        value = report["metadata"].get(key)
        if isinstance(value, list):
            value = ", ".join(value)
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Executive Summary", ""])
    lines.extend(f"- {item}" for item in report["executive_summary"])
    lines.extend(["", "## Overall Assessment", ""])
    for key, value in report["overall_assessment"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Key Grouped Risk Signals Requiring Human Review", ""])
    if report["grouped_findings"]:
        for finding in report["grouped_findings"]:
            lines.extend(
                [
                    f"### {finding['finding_id']}: {finding['title']}",
                    "",
                    f"- Risk categories: {', '.join(finding['risk_categories'])}",
                    f"- Support levels: {', '.join(finding['support_levels'])}",
                    f"- Final severity: {finding['final_severity']}",
                    f"- Candidate IDs: {', '.join(finding['candidate_ids'])}",
                    f"- Assessment IDs: {', '.join(finding['assessment_ids'])}",
                    f"- Tool refs: {', '.join(finding['tool_refs'])}",
                    f"- Evidence refs: {', '.join(ref['ref_id'] for ref in finding['evidence_refs'])}",
                    "",
                    finding["summary"],
                    "",
                ]
            )
            if finding.get("why_this_matters"):
                lines.extend(["Why this matters:", "", finding["why_this_matters"], ""])
    else:
        lines.append("No supported grouped risk signals were identified from detector-reviewed evidence.")
    lines.extend(["", "## Weak or Limited Risk Signals", ""])
    if report["weak_or_limited_signals"]:
        for item in report["weak_or_limited_signals"]:
            lines.extend(
                [
                    f"- {item['assessment_id']}: {item['risk_category']} ({item['final_severity']})",
                    f"  Evidence refs: {', '.join(ref['ref_id'] for ref in item['evidence_refs'])}",
                ]
            )
    else:
        lines.append("No weakly supported risk signals were identified.")
    lines.extend(["", "## Reviewed Candidate Audit Section", ""])
    for item in report["reviewed_candidate_audit"]:
        lines.append(
            f"- {item['candidate_id']} | {item['risk_category']} | {item['support_level']} | {item['assessment_id']}"
        )
    lines.extend(["", "## Insufficient Evidence / Data Gaps", ""])
    if report["insufficient_evidence_or_data_gaps"]:
        for item in report["insufficient_evidence_or_data_gaps"]:
            lines.append(f"- {item['candidate_id']}: provided evidence was insufficient for assessment.")
    else:
        lines.append("No detector-reviewed candidate had insufficient visible evidence.")
    lines.extend(["", "## Method and Scope", ""])
    lines.append(report["method_and_scope"]["input_scope"])
    lines.append(report["method_and_scope"]["evidence_scope"])
    lines.append(report["method_and_scope"]["reporting_rule"])
    report_assembly = report["method_and_scope"].get("report_assembly")
    if report_assembly:
        lines.append(report_assembly)
    excluded_scope = report["method_and_scope"].get("excluded_scope")
    if excluded_scope:
        lines.extend(["", "Excluded scope:"])
        lines.extend(f"- {item}" for item in excluded_scope)
    lines.extend(["", "## Limitations and Run Status Notes", ""])
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    if report.get("coverage_limitations"):
        lines.extend(["", "## Data Coverage Limitations", ""])
        for limitation in report["coverage_limitations"]:
            lines.append(f"- {limitation['label']}: {limitation['summary']}")
    lines.extend(["", "## Audit Trail", ""])
    for key, value in report["audit_trail"].items():
        if isinstance(value, list):
            value = ", ".join(value)
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def _final_report_metadata(report_memory: ReportMemory, *, report_language: str = "vi") -> dict[str, Any]:
    metadata = report_memory.metadata
    return {
        "company_name": metadata["company_name"],
        "period": metadata["period"],
        "report_period_type": metadata["report_period_type"],
        "report_profile": metadata["report_profile"],
        "report_basis": metadata["report_basis"],
        "business_context_tags": metadata["business_context_tags"],
        "insurance_subprofile": metadata.get("insurance_subprofile"),
        "report_assurance_type": metadata["report_assurance_type"],
        "currency": metadata["currency"],
        "unit": metadata["unit"],
        "filing_status": metadata["filing_status"],
        "canonical_source_document_id": metadata["canonical_source_document_id"],
        "report_language": report_language,
    }


def _source_lineage_by_report_id(report_set: CompanyReportSet) -> dict[str, dict[str, str]]:
    return {
        report_set.target.report_id: {
            "source_document_id": report_set.target.metadata["canonical_source_document_id"],
            "source_slot_role": "target",
        },
        report_set.prior_year.report_id: {
            "source_document_id": report_set.prior_year.metadata["canonical_source_document_id"],
            "source_slot_role": "prior_year_same_quarter",
        },
    }


def _source_navigation_by_ref_id(report_set: CompanyReportSet) -> dict[str, dict[str, Any]]:
    navigation: dict[str, dict[str, Any]] = {}
    for report_memory in [report_set.target, report_set.prior_year]:
        for table in report_memory.raw.get("tables", []):
            for row in table.get("rows", []):
                for cell in row.get("cells", []):
                    ref_id = f"{report_memory.report_id}:{cell.get('cell_id')}"
                    navigation[ref_id] = {
                        "page_number": cell.get("source_page_number"),
                        "source_excerpt": cell.get("source_excerpt"),
                    }
    return navigation


def _final_report_generation_context(
    report_set: CompanyReportSet,
    overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    metadata = report_set.target.metadata
    report_language = _report_language(overrides)
    tags = set(metadata.get("business_context_tags", []))
    extraction_method = str(metadata.get("extraction_method") or "")
    source_file = str(metadata.get("source_file") or "")
    mode = (overrides or {}).get("source_mode")
    if mode is None:
        if (
            "vietstock_cached_real_ocr" in tags
            or extraction_method.startswith("cached_real_ocr")
        ):
            mode = "cached_real_vietstock_ocr"
        elif extraction_method == "runtime_demo_report_memory_v1":
            mode = "runtime_demo_report_memory"
        elif "fixture" in extraction_method or "fixtures/" in source_file:
            mode = "fixture"
        else:
            mode = "report_memory"

    contexts = {
        "cached_real_vietstock_ocr": {
            "report_basis": "cached_real_vietstock_ocr_detector_assessments",
            "input_scope": (
                "Cached real Vietstock quarterly financial statement ReportMemory artifacts "
                "for the selected company, period, and filing basis."
            ),
            "evidence_scope": (
                "Structured table, note, variance explanation, source provenance, tool, "
                "candidate, and detector assessment artifacts derived from cached OCR replay."
            ),
            "first_limitation": (
                "The report is limited to cached real Vietstock OCR evidence and structured "
                "ReportMemory replay for the selected filing pair."
            ),
            "finding_limitation": (
                "The finding is a detector-reviewed risk signal based only on structured "
                "evidence replayed from cached real OCR artifacts."
            ),
            "confidence_summary": (
                "Based on deterministic detector assessments over cached real Vietstock "
                "OCR-derived evidence."
            ),
            "skipped_checks_prefix": (
                "Some checks were not executed because required cached evidence context was "
                "unavailable: "
            ),
            "excluded_scope": [
                (
                    "No live source download or live OCR call was performed during this run; "
                    "pinned Vietstock source and OCR artifacts were reused from cache."
                ),
                (
                    "No raw OCR text, full filing body, or local artifact path is included "
                    "in this public report."
                ),
                "No external market, news, enforcement, macro, or analyst data.",
                (
                    "The selected report synthesis model is recorded for run provenance, "
                    "but this report body is deterministically assembled from detector "
                    "assessments, tool findings, and aggregation rules."
                ),
            ],
        },
        "runtime_demo_report_memory": {
            "report_basis": "runtime_demo_report_memory_detector_assessments",
            "input_scope": (
                "Runtime demo ReportMemory artifacts generated after source confirmation "
                "for the selected company, period, and filing basis."
            ),
            "evidence_scope": (
                "Structured table, tool, candidate, and detector assessment artifacts from "
                "the runtime demo adapter."
            ),
            "first_limitation": (
                "The report is limited to runtime demo ReportMemory evidence assembled for "
                "workflow verification."
            ),
            "finding_limitation": (
                "The finding is a detector-reviewed risk signal based only on runtime demo "
                "ReportMemory evidence."
            ),
            "confidence_summary": (
                "Based on deterministic detector assessments over runtime demo evidence."
            ),
            "skipped_checks_prefix": (
                "Some checks were not executed because required runtime demo context was "
                "unavailable: "
            ),
            "excluded_scope": [
                "No live OCR extraction was performed by the runtime demo adapter.",
                "No raw OCR text, full filing body, or local artifact path is included in this public report.",
                "No external market, news, enforcement, macro, or analyst data.",
                (
                    "The selected report synthesis model is recorded for run provenance, "
                    "but this report body is deterministically assembled from detector "
                    "assessments, tool findings, and aggregation rules."
                ),
            ],
        },
        "fixture": {
            "report_basis": "fixture_detector_assessments",
            "input_scope": (
                "Fixture-based quarterly report profiles used by Vinacount development and "
                "regression tests."
            ),
            "evidence_scope": (
                "Structured table, note, variance explanation, tool, candidate, and detector "
                "assessment artifacts."
            ),
            "first_limitation": "The report is limited to fixture-based report evidence.",
            "finding_limitation": (
                "The finding is a detector-reviewed risk signal based only on provided "
                "fixture evidence."
            ),
            "confidence_summary": (
                "Based on deterministic detector assessments over fixture-visible evidence."
            ),
            "skipped_checks_prefix": (
                "Some checks were not executed because required fixture context was unavailable: "
            ),
            "excluded_scope": [
                "No real PDF or OCR extraction.",
                "No source fetching.",
                "No external market, news, enforcement, macro, or analyst data.",
                "Optional API LLM detector adapter is disabled unless explicit fixture configuration is present.",
            ],
        },
        "report_memory": {
            "report_basis": "report_memory_detector_assessments",
            "input_scope": (
                "ReportMemory artifacts for the selected company, period, and filing basis."
            ),
            "evidence_scope": (
                "Structured table, note, variance explanation, tool, candidate, and detector "
                "assessment artifacts available to the runtime."
            ),
            "first_limitation": "The report is limited to available ReportMemory evidence.",
            "finding_limitation": (
                "The finding is a detector-reviewed risk signal based only on available "
                "ReportMemory evidence."
            ),
            "confidence_summary": (
                "Based on deterministic detector assessments over available ReportMemory evidence."
            ),
            "skipped_checks_prefix": (
                "Some checks were not executed because required evidence context was unavailable: "
            ),
            "excluded_scope": [
                "No raw OCR text, full filing body, or local artifact path is included in this public report.",
                "No external market, news, enforcement, macro, or analyst data.",
                (
                    "The selected report synthesis model is recorded for run provenance, "
                    "but this report body is deterministically assembled from detector "
                    "assessments, tool findings, and aggregation rules."
                ),
            ],
        },
    }
    if mode not in contexts:
        mode = "report_memory"
    context = dict(contexts[mode])
    context["mode"] = mode
    context["report_language"] = report_language
    if report_language == "vi":
        context.update(_vietnamese_generation_context(mode))
    report_generation_mode = (overrides or {}).get("report_generation_mode", "deterministic_template")
    if report_generation_mode == "deterministic_template":
        report_assembly = _localized(
            report_language,
            vi=(
                "Chế độ mẫu xác định dùng cho phát triển/kiểm thử: cấu trúc và nội dung báo cáo "
                "được tổng hợp từ đánh giá detector, kết quả công cụ và quy tắc tổng hợp."
            ),
            en=(
                "Explicit development/smoke deterministic template mode: report structure and prose "
                "are deterministically assembled from detector assessments, tool findings, and aggregation rules."
            ),
        )
    else:
        report_assembly = _localized(
            report_language,
            vi=(
                "Cấu trúc và các trường kiểm soát do backend xác định; mô hình tổng hợp báo cáo "
                "chỉ soạn các phần mô tả theo ngôn ngữ báo cáo từ những nhận định được phép."
            ),
            en=(
                "Report structure and control fields are deterministically assembled by the backend; "
                "the selected report synthesis model may write bounded narrative slots only."
            ),
        )
    context["report_generation_mode"] = report_generation_mode
    context["method_and_scope"] = {
        "input_scope": context["input_scope"],
        "evidence_scope": context["evidence_scope"],
        "reporting_rule": _localized(
            report_language,
            vi="Kết luận cuối chỉ dựa trên các đánh giá detector có mức hỗ trợ hoặc hỗ trợ hạn chế.",
            en="Final findings are based only on supported or weakly supported detector assessments.",
        ),
        "report_assembly": report_assembly,
        "report_generation_mode": report_generation_mode,
        "excluded_scope": list(context["excluded_scope"]),
    }
    if overrides:
        report_synthesis_model = overrides.get("report_synthesis_model")
        if isinstance(report_synthesis_model, dict) and report_synthesis_model.get("id"):
            context["method_and_scope"]["report_synthesis_model"] = {
                "model_id": report_synthesis_model["id"],
                "provider": report_synthesis_model.get("provider"),
                "invoked_for_report_generation": False,
            }
    return context


def _report_language(overrides: dict[str, Any] | None) -> str:
    language = (overrides or {}).get("report_language", "vi")
    return language if language in {"vi", "en"} else "vi"


def _localized(report_language: str, *, vi: str, en: str) -> str:
    return vi if report_language == "vi" else en


def _vietnamese_generation_context(mode: str) -> dict[str, Any]:
    contexts = {
        "cached_real_vietstock_ocr": {
            "input_scope": (
                "Báo cáo sử dụng các hiện vật ReportMemory từ BCTC quý trên Vietstock đã được lưu "
                "cho công ty, kỳ và loại báo cáo được chọn."
            ),
            "evidence_scope": (
                "Phạm vi bằng chứng gồm bảng số liệu có cấu trúc, thuyết minh, giải trình biến động, "
                "nguồn tài liệu, kết quả công cụ, ứng viên rủi ro và đánh giá detector từ OCR đã lưu."
            ),
            "first_limitation": (
                "Báo cáo chỉ giới hạn trong bằng chứng OCR Vietstock đã lưu và ReportMemory có cấu trúc "
                "cho cặp báo cáo được chọn."
            ),
            "finding_limitation": (
                "Tín hiệu này là kết quả rà soát bởi detector và chỉ dựa trên bằng chứng có cấu trúc "
                "được tái dựng từ OCR đã lưu."
            ),
            "confidence_summary": "Dựa trên đánh giá detector xác định trên bằng chứng OCR Vietstock đã lưu.",
            "skipped_checks_prefix": "Một số kiểm tra không chạy vì thiếu ngữ cảnh bằng chứng đã lưu: ",
            "excluded_scope": [
                "Lần chạy này không tải nguồn trực tiếp hoặc gọi OCR trực tiếp; nguồn Vietstock và OCR đã ghim được tái sử dụng từ bộ nhớ đệm.",
                "Báo cáo công khai không bao gồm OCR thô, toàn văn hồ sơ hoặc đường dẫn hiện vật cục bộ.",
                "Không sử dụng dữ liệu thị trường, tin tức, thực thi pháp luật, vĩ mô hoặc phân tích bên ngoài.",
                "Mô hình tổng hợp báo cáo được ghi nhận để truy vết, nhưng thân báo cáo được lắp ráp từ đánh giá detector, kết quả công cụ và quy tắc tổng hợp.",
            ],
        },
        "runtime_demo_report_memory": {
            "input_scope": "Báo cáo sử dụng ReportMemory demo runtime sau khi xác nhận nguồn cho hồ sơ được chọn.",
            "evidence_scope": "Phạm vi bằng chứng gồm bảng có cấu trúc, kết quả công cụ, ứng viên rủi ro và đánh giá detector từ adapter demo runtime.",
            "first_limitation": "Báo cáo chỉ giới hạn trong bằng chứng ReportMemory demo runtime phục vụ kiểm tra quy trình.",
            "finding_limitation": "Tín hiệu này là kết quả rà soát bởi detector và chỉ dựa trên bằng chứng ReportMemory demo runtime.",
            "confidence_summary": "Dựa trên đánh giá detector xác định trên bằng chứng demo runtime.",
            "skipped_checks_prefix": "Một số kiểm tra không chạy vì thiếu ngữ cảnh demo runtime: ",
            "excluded_scope": [
                "Adapter demo runtime không thực hiện OCR trực tiếp.",
                "Báo cáo công khai không bao gồm OCR thô, toàn văn hồ sơ hoặc đường dẫn hiện vật cục bộ.",
                "Không sử dụng dữ liệu thị trường, tin tức, thực thi pháp luật, vĩ mô hoặc phân tích bên ngoài.",
                "Mô hình tổng hợp báo cáo được ghi nhận để truy vết, nhưng thân báo cáo được lắp ráp từ đánh giá detector, kết quả công cụ và quy tắc tổng hợp.",
            ],
        },
        "fixture": {
            "input_scope": "Báo cáo sử dụng hồ sơ fixture theo quý phục vụ phát triển và kiểm thử hồi quy Vinacount.",
            "evidence_scope": "Phạm vi bằng chứng gồm bảng có cấu trúc, thuyết minh, giải trình biến động, kết quả công cụ, ứng viên rủi ro và đánh giá detector.",
            "first_limitation": "Báo cáo chỉ giới hạn trong bằng chứng fixture.",
            "finding_limitation": "Tín hiệu này là kết quả rà soát bởi detector và chỉ dựa trên bằng chứng fixture đã cung cấp.",
            "confidence_summary": "Dựa trên đánh giá detector xác định trên bằng chứng fixture.",
            "skipped_checks_prefix": "Một số kiểm tra không chạy vì thiếu ngữ cảnh fixture: ",
            "excluded_scope": [
                "Không sử dụng PDF thật hoặc OCR.",
                "Không tải nguồn.",
                "Không sử dụng dữ liệu thị trường, tin tức, thực thi pháp luật, vĩ mô hoặc phân tích bên ngoài.",
                "Adapter detector LLM qua API chỉ bật khi có cấu hình fixture rõ ràng.",
            ],
        },
        "report_memory": {
            "input_scope": "Báo cáo sử dụng ReportMemory cho công ty, kỳ và loại báo cáo được chọn.",
            "evidence_scope": "Phạm vi bằng chứng gồm bảng có cấu trúc, thuyết minh, giải trình biến động, kết quả công cụ, ứng viên rủi ro và đánh giá detector có sẵn trong runtime.",
            "first_limitation": "Báo cáo chỉ giới hạn trong bằng chứng ReportMemory hiện có.",
            "finding_limitation": "Tín hiệu này là kết quả rà soát bởi detector và chỉ dựa trên bằng chứng ReportMemory hiện có.",
            "confidence_summary": "Dựa trên đánh giá detector xác định trên bằng chứng ReportMemory hiện có.",
            "skipped_checks_prefix": "Một số kiểm tra không chạy vì thiếu ngữ cảnh bằng chứng: ",
            "excluded_scope": [
                "Báo cáo công khai không bao gồm OCR thô, toàn văn hồ sơ hoặc đường dẫn hiện vật cục bộ.",
                "Không sử dụng dữ liệu thị trường, tin tức, thực thi pháp luật, vĩ mô hoặc phân tích bên ngoài.",
                "Mô hình tổng hợp báo cáo được ghi nhận để truy vết, nhưng thân báo cáo được lắp ráp từ đánh giá detector, kết quả công cụ và quy tắc tổng hợp.",
            ],
        },
    }
    return contexts.get(mode, contexts["report_memory"])


def _group_supported_assessments(
    supported_assessments: list[dict[str, Any]],
    candidates_by_id: dict[str, dict[str, Any]],
    tool_findings_by_id: dict[str, dict[str, Any]],
    generation_context: dict[str, Any],
    source_lineage_by_report_id: dict[str, dict[str, str]],
    source_navigation_by_ref_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for assessment in supported_assessments:
        candidate = candidates_by_id[assessment["candidate_id"]]
        key = tuple(candidate["linked_tool_result_ids"][:1] or [assessment["risk_category"]])
        groups.setdefault(key, []).append(assessment)

    findings = []
    for index, assessments in enumerate(groups.values(), start=1):
        candidate_ids = [assessment["candidate_id"] for assessment in assessments]
        candidates = [candidates_by_id[candidate_id] for candidate_id in candidate_ids]
        tool_refs = _dedupe_strings(
            tool_result_id
            for candidate in candidates
            for tool_result_id in candidate["linked_tool_result_ids"]
        )
        evidence_refs = _dedupe_evidence_refs(
            (
                ref
                for assessment in assessments
                for ref in _evidence_refs_for_assessment(
                    assessment,
                    candidates_by_id,
                    tool_findings_by_id,
                )
            ),
            source_lineage_by_report_id,
            source_navigation_by_ref_id,
            generation_context["report_language"],
            exclude_coverage_limitations=True,
        )
        final_severity = _final_severity(assessment["severity"] for assessment in assessments)
        primary = _primary_assessment(assessments, candidates_by_id)
        findings.append(
            {
                "finding_id": f"F-{index:03d}",
                "title": _risk_category_title(primary["risk_category"], generation_context["report_language"]),
                "primary_risk_category": primary["risk_category"],
                "risk_categories": _dedupe_strings(
                    assessment["risk_category"] for assessment in assessments
                ),
                "support_levels": _dedupe_strings(
                    assessment["support_level"] for assessment in assessments
                ),
                "final_severity": final_severity,
                "human_review_recommendation": "recommended",
                "assessment_ids": [assessment["assessment_id"] for assessment in assessments],
                "candidate_ids": candidate_ids,
                "tool_refs": tool_refs,
                "evidence_refs": evidence_refs,
                "summary": _finding_summary(
                    primary,
                    candidates_by_id,
                    tool_findings_by_id,
                    generation_context["report_language"],
                ),
                "why_this_matters": _why_this_matters(
                    primary["risk_category"],
                    generation_context["report_language"],
                ),
                "limitations": [generation_context["finding_limitation"]],
            }
        )
    return findings


def _weak_signal_item(
    assessment: dict[str, Any],
    candidates_by_id: dict[str, dict[str, Any]],
    tool_findings_by_id: dict[str, dict[str, Any]],
    source_lineage_by_report_id: dict[str, dict[str, str]],
    source_navigation_by_ref_id: dict[str, dict[str, Any]],
    report_language: str = "vi",
) -> dict[str, Any]:
    candidate = candidates_by_id[assessment["candidate_id"]]
    tool_refs = candidate["linked_tool_result_ids"]
    return {
        "assessment_id": assessment["assessment_id"],
        "candidate_id": assessment["candidate_id"],
        "risk_category": assessment["risk_category"],
        "support_level": assessment["support_level"],
        "final_severity": _cap_weak_final_severity(assessment["severity"]),
        "tool_refs": tool_refs,
        "evidence_refs": _dedupe_evidence_refs(
            _evidence_refs_for_assessment(assessment, candidates_by_id, tool_findings_by_id),
            source_lineage_by_report_id,
            source_navigation_by_ref_id,
            report_language,
            exclude_coverage_limitations=True,
        ),
        "summary": _finding_summary(assessment, candidates_by_id, tool_findings_by_id, report_language),
        "missing_or_limited_context": _localized(
            report_language,
            vi="Đánh giá detector chỉ có mức hỗ trợ hạn chế, nên tín hiệu này cần được chuyên gia xem xét thận trọng.",
            en="The detector assessment was weakly supported, so this remains a possible signal for human review.",
        ),
    }


def _reviewed_candidate_audit_item(
    assessment: dict[str, Any],
    candidates_by_id: dict[str, dict[str, Any]],
    tool_findings_by_id: dict[str, dict[str, Any]],
    source_lineage_by_report_id: dict[str, dict[str, str]],
    source_navigation_by_ref_id: dict[str, dict[str, Any]],
    report_language: str = "vi",
) -> dict[str, Any]:
    candidate = candidates_by_id[assessment["candidate_id"]]
    tool_refs = candidate["linked_tool_result_ids"]
    return {
        "candidate_id": assessment["candidate_id"],
        "assessment_id": assessment["assessment_id"],
        "risk_category": assessment["risk_category"],
        "support_level": assessment["support_level"],
        "detector_severity": assessment["severity"],
        "candidate_priority": candidate["priority"],
        "tool_refs": tool_refs,
        "evidence_refs": _dedupe_evidence_refs(
            _evidence_refs_for_assessment(assessment, candidates_by_id, tool_findings_by_id),
            source_lineage_by_report_id,
            source_navigation_by_ref_id,
            report_language,
        ),
        "status": _reviewed_candidate_status(assessment["support_level"]),
        "rationale_short": assessment["rationale_short"],
    }


def _final_report_limitations(
    report_set: CompanyReportSet,
    detector_assessments: list[dict[str, Any]],
    gating_records: list[dict[str, Any]],
    generation_context: dict[str, Any],
) -> list[str]:
    report_language = generation_context["report_language"]
    limitations = [
        generation_context["first_limitation"],
        _localized(
            report_language,
            vi="Công cụ, kiểm tra bị bỏ qua, bản ghi gating và ghi chú lần chạy không được dùng làm kết luận rủi ro cuối.",
            en="Tools, skipped checks, gating records, and run notes are not used as final risk claims.",
        ),
        _localized(
            report_language,
            vi="Không sử dụng nguồn bên ngoài, dữ liệu thị trường, tin tức, thực thi pháp luật, vĩ mô hoặc phân tích.",
            en="External source, market, news, enforcement, macro, and analyst context are excluded.",
        ),
    ]
    if report_set.target.metadata["report_assurance_type"] != "audited":
        limitations.append(
            _localized(
                report_language,
                vi=f"Bối cảnh đảm bảo của báo cáo mục tiêu là {report_set.target.metadata['report_assurance_type']}.",
                en=f"The target filing assurance context is {report_set.target.metadata['report_assurance_type']}.",
            )
        )
    limitations.extend(report_set.target.metadata.get("extraction_limitations", []))
    if _final_report_coverage_limitations(report_set.target, report_language=report_language):
        limitations.append(
            _localized(
                report_language,
                vi="Một số bề mặt dữ liệu chưa được bao phủ đầy đủ; xem mục giới hạn dữ liệu.",
                en="Some evidence surfaces are not fully covered; see the data coverage limitations section.",
            )
        )
    skipped_material = [
        record["tool_name"]
        for record in gating_records
        if record["status"] in {"disabled_missing_context", "skipped_missing_disclosure_context"}
    ]
    if skipped_material:
        limitations.append(
            generation_context["skipped_checks_prefix"]
            + ", ".join(skipped_material)
            + "."
        )
    if any(assessment["support_level"] == "insufficient_evidence" for assessment in detector_assessments):
        limitations.append(
            _localized(
                report_language,
                vi="Ít nhất một ứng viên đã rà soát không có đủ bằng chứng hiển thị.",
                en="At least one reviewed candidate had insufficient visible evidence.",
            )
        )
    return limitations


def _final_report_coverage_limitations(
    report_memory: ReportMemory,
    *,
    report_language: str = "vi",
) -> list[dict[str, Any]]:
    limitations = []
    limitation_states = {
        "absent_in_source",
        "not_applicable",
        "unsupported_by_extraction_path",
        "not_extracted_yet",
        "ambiguous_failed_closed",
    }
    for status in report_memory.raw.get("evidence_surface_status", []):
        if not isinstance(status, dict) or status.get("state") not in limitation_states:
            continue
        surface = status.get("surface")
        state = status.get("state")
        raw_message = status.get("message")
        if report_language == "vi":
            message = _surface_limitation_message(surface, state)
        elif not isinstance(raw_message, str) or not raw_message.strip():
            message = f"{surface} evidence surface status is {state}."
        else:
            message = raw_message
        limitation_id = f"{surface}:{state}"
        if any(item["limitation_id"] == limitation_id for item in limitations):
            continue
        limitations.append(
            {
                "limitation_id": limitation_id,
                "surface": surface,
                "state": state,
                "label": _surface_limitation_label(surface, report_language),
                "summary": message,
            }
        )
    return limitations


def _surface_limitation_label(surface: Any, report_language: str = "vi") -> str:
    if report_language == "vi":
        return {
            "notes": "Thuyết minh",
            "variance_explanations": "Giải trình biến động",
            "related_party_evidence": "Bên liên quan",
            "accounting_policy_evidence": "Chính sách kế toán",
            "extraction_quality": "Chất lượng trích xuất",
        }.get(surface, str(surface))
    return {
        "notes": "Notes",
        "variance_explanations": "Variance explanations",
        "related_party_evidence": "Related parties",
        "accounting_policy_evidence": "Accounting policies",
        "extraction_quality": "Extraction quality",
    }.get(surface, str(surface))


def _surface_limitation_message(surface: Any, state: Any) -> str:
    surface_labels = {
        "notes": "Thuyết minh",
        "variance_explanations": "Giải trình biến động",
        "related_party_evidence": "Bằng chứng bên liên quan",
        "accounting_policy_evidence": "Bằng chứng chính sách kế toán",
        "extraction_quality": "Chất lượng trích xuất",
    }
    state_labels = {
        "absent_in_source": "không có trong nguồn",
        "not_applicable": "không áp dụng",
        "unsupported_by_extraction_path": "chưa được hỗ trợ bởi luồng trích xuất",
        "not_extracted_yet": "chưa được trích xuất",
        "ambiguous_failed_closed": "không đủ rõ ràng nên bị dừng an toàn",
    }
    return f"{surface_labels.get(surface, str(surface))} có trạng thái {state_labels.get(state, str(state))}."


def _final_report_overall_assessment(
    grouped_findings: list[dict[str, Any]],
    weak_or_limited: list[dict[str, Any]],
    limitations: list[str],
    generation_context: dict[str, Any],
) -> dict[str, Any]:
    highest_severity = _final_severity(
        [finding["final_severity"] for finding in grouped_findings]
        + [item["final_severity"] for item in weak_or_limited]
    )
    if grouped_findings:
        status = "risk_signals_identified"
    elif weak_or_limited:
        status = "weak_or_limited_risk_signals_only"
    elif any(
        "insufficient visible evidence" in limitation or "không có đủ bằng chứng" in limitation
        for limitation in limitations
    ):
        status = "insufficient_evidence_for_overall_assessment"
    else:
        status = "no_material_irregularity_signal_identified"
    return {
        "overall_review_status": status,
        "primary_risk_category": grouped_findings[0]["primary_risk_category"] if grouped_findings else None,
        "secondary_risk_categories": _dedupe_strings(
            category
            for finding in grouped_findings[1:]
            for category in finding["risk_categories"]
        ),
        "highest_severity": highest_severity,
        "human_review_recommended": bool(grouped_findings or weak_or_limited),
        "confidence_summary": generation_context["confidence_summary"],
    }


def _final_report_executive_summary(
    grouped_findings: list[dict[str, Any]],
    weak_or_limited: list[dict[str, Any]],
    limitations: list[str],
    report_language: str = "vi",
) -> list[str]:
    if report_language == "vi":
        return [
            f"Hệ thống xác định {len(grouped_findings)} nhóm tín hiệu rủi ro được hỗ trợ bởi bằng chứng đã thẩm định.",
            f"Hệ thống xác định {len(weak_or_limited)} tín hiệu có mức hỗ trợ hạn chế cần xem xét thận trọng.",
            "Khuyến nghị chuyên gia rà soát các tín hiệu được trình bày trong báo cáo."
            if grouped_findings or weak_or_limited
            else "Không xác định được tín hiệu rủi ro cuối cùng từ bằng chứng đã cung cấp.",
            limitations[0],
        ]
    return [
        f"The system identified {len(grouped_findings)} supported grouped risk signal(s).",
        f"The system identified {len(weak_or_limited)} weakly supported possible risk signal(s).",
        "Human review is recommended for supported and weakly supported detector-reviewed items."
        if grouped_findings or weak_or_limited
        else "No detector-reviewed final risk signal was identified from the provided evidence.",
        limitations[0],
    ]


def _primary_assessment(
    assessments: list[dict[str, Any]],
    candidates_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    priority_rank = {"high": 0, "medium": 1, "low": 2}
    severity_rank = {"high": 0, "medium": 1, "low": 2, "unknown": 3}
    return sorted(
        assessments,
        key=lambda assessment: (
            severity_rank[assessment["severity"]],
            priority_rank[candidates_by_id[assessment["candidate_id"]]["priority"]],
            assessment["assessment_id"],
        ),
    )[0]


def _final_severity(severities: Any) -> str:
    severity_values = list(severities)
    for severity in ["high", "medium", "low"]:
        if severity in severity_values:
            return severity
    return "unknown"


def _cap_weak_final_severity(severity: str) -> str:
    if severity == "high":
        return "medium"
    return severity


def _finding_summary(
    assessment: dict[str, Any],
    candidates_by_id: dict[str, dict[str, Any]],
    tool_findings_by_id: dict[str, dict[str, Any]],
    report_language: str = "vi",
) -> str:
    candidate = candidates_by_id[assessment["candidate_id"]]
    tool_summaries = [
        tool_findings_by_id[tool_result_id]["finding_summary"]
        for tool_result_id in candidate["linked_tool_result_ids"]
        if tool_result_id in tool_findings_by_id
    ]
    if tool_summaries:
        if report_language == "vi":
            return _vietnamese_risk_summary(assessment["risk_category"])
        suffix = _localized(
            report_language,
            vi=" Đánh giá detector cho thấy nội dung này cần được chuyên gia xem xét.",
            en=" The detector assessment indicates this warrants human review.",
        )
        return " ".join(tool_summaries) + suffix
    if report_language == "vi":
        return "Đánh giá detector ghi nhận tín hiệu cần được chuyên gia xem xét trong phạm vi bằng chứng hiện có."
    return assessment["rationale_short"]


def _vietnamese_risk_summary(risk_category: str) -> str:
    summaries = {
        "revenue_income_recognition_risk": "Tín hiệu rủi ro liên quan đến chất lượng doanh thu được ghi nhận từ bằng chứng báo cáo.",
        "receivables_credit_quality_risk": "Tín hiệu rủi ro liên quan đến chất lượng khoản phải thu được ghi nhận từ bằng chứng báo cáo.",
        "asset_quality_valuation_risk": "Tín hiệu rủi ro liên quan đến định giá tài sản được ghi nhận từ bằng chứng báo cáo.",
        "inventory_cost_asset_flow_risk": "Tín hiệu rủi ro liên quan đến luân chuyển hàng tồn kho được ghi nhận từ bằng chứng báo cáo.",
        "expense_liability_understatement_risk": "Tín hiệu rủi ro liên quan đến chi phí hoặc nợ phải trả được ghi nhận từ bằng chứng báo cáo.",
        "earnings_cashflow_mismatch": "Tín hiệu rủi ro liên quan đến chênh lệch lợi nhuận và dòng tiền được ghi nhận từ bằng chứng báo cáo.",
        "disclosure_inconsistency_or_obfuscation": "Tín hiệu rủi ro liên quan đến tính nhất quán của thuyết minh được ghi nhận từ bằng chứng báo cáo.",
        "related_party_disclosure_risk": "Tín hiệu rủi ro liên quan đến thuyết minh bên liên quan được ghi nhận từ bằng chứng báo cáo.",
    }
    return summaries.get(risk_category, "Tín hiệu rủi ro kế toán được ghi nhận từ bằng chứng báo cáo.")


def _risk_category_title(risk_category: str, report_language: str = "vi") -> str:
    if report_language == "vi":
        labels = {
            "revenue_income_recognition_risk": "Tín hiệu rủi ro chất lượng doanh thu",
            "receivables_credit_quality_risk": "Tín hiệu rủi ro chất lượng khoản phải thu",
            "asset_quality_valuation_risk": "Tín hiệu rủi ro định giá tài sản",
            "inventory_cost_asset_flow_risk": "Tín hiệu rủi ro luân chuyển hàng tồn kho",
            "expense_liability_understatement_risk": "Tín hiệu rủi ro ghi nhận thiếu chi phí hoặc nợ phải trả",
            "earnings_cashflow_mismatch": "Tín hiệu rủi ro chênh lệch lợi nhuận và dòng tiền",
            "disclosure_inconsistency_or_obfuscation": "Tín hiệu rủi ro nhất quán thuyết minh",
            "related_party_disclosure_risk": "Tín hiệu rủi ro thuyết minh bên liên quan",
        }
        return labels.get(risk_category, "Tín hiệu rủi ro kế toán")
    labels = {
        "revenue_income_recognition_risk": "Revenue quality risk signal",
        "receivables_credit_quality_risk": "Credit quality risk signal",
        "asset_quality_valuation_risk": "Asset valuation risk signal",
        "inventory_cost_asset_flow_risk": "Trading-book flow risk signal",
        "expense_liability_understatement_risk": "Provision movement risk signal",
        "earnings_cashflow_mismatch": "Earnings and cash-flow risk signal",
        "disclosure_inconsistency_or_obfuscation": "Disclosure consistency risk signal",
        "related_party_disclosure_risk": "Related-party disclosure risk signal",
    }
    return labels.get(risk_category, "Accounting risk signal")


def _why_this_matters(risk_category: str, report_language: str = "vi") -> str:
    if report_language == "vi":
        messages = {
            "revenue_income_recognition_risk": (
                "Doanh thu tăng cùng các tín hiệu về chất lượng thu tiền hoặc khoản phải thu "
                "có thể làm tăng ưu tiên rà soát về khả năng chuyển đổi doanh thu thành tiền."
            ),
            "receivables_credit_quality_risk": (
                "Khoản phải thu tăng hoặc suy giảm chất lượng thu tiền có thể ảnh hưởng đến "
                "khả năng thu hồi và chất lượng lợi nhuận được báo cáo."
            ),
            "asset_quality_valuation_risk": (
                "Giá trị tài sản cần được rà soát khi bằng chứng cho thấy khả năng ghi nhận "
                "hoặc đánh giá lại có thể ảnh hưởng đến chất lượng bảng cân đối."
            ),
            "inventory_cost_asset_flow_risk": (
                "Biến động hàng tồn kho hoặc dòng giá vốn có thể ảnh hưởng đến biên lợi nhuận "
                "và chất lượng tài sản lưu động được báo cáo."
            ),
            "expense_liability_understatement_risk": (
                "Tín hiệu về chi phí hoặc nợ phải trả có thể ảnh hưởng đến mức lợi nhuận và "
                "nghĩa vụ được trình bày trong kỳ."
            ),
            "earnings_cashflow_mismatch": (
                "Khoảng cách giữa lợi nhuận và dòng tiền kinh doanh có thể làm tăng ưu tiên "
                "rà soát chất lượng lợi nhuận và biến động vốn lưu động."
            ),
            "disclosure_inconsistency_or_obfuscation": (
                "Thuyết minh thiếu nhất quán có thể làm giảm khả năng đối chiếu bằng chứng "
                "và cần được đọc cùng các số liệu liên quan."
            ),
            "related_party_disclosure_risk": (
                "Giao dịch hoặc số dư với bên liên quan cần được rà soát vì có thể ảnh hưởng "
                "đến khả năng hiểu đầy đủ quan hệ và điều kiện giao dịch."
            ),
        }
        return messages.get(
            risk_category,
            "Tín hiệu này có thể ảnh hưởng đến cách chuyên gia ưu tiên rà soát báo cáo.",
        )
    messages = {
        "revenue_income_recognition_risk": (
            "Revenue growth paired with cash-collection or receivables-quality signals can raise "
            "the review priority for whether reported revenue converts into cash."
        ),
        "receivables_credit_quality_risk": (
            "Receivables growth or collection-quality signals can affect recoverability and reported earnings quality."
        ),
        "asset_quality_valuation_risk": (
            "Asset valuation signals can affect balance-sheet quality and may warrant manual review."
        ),
        "inventory_cost_asset_flow_risk": (
            "Inventory or cost-flow signals can affect margin quality and reported current assets."
        ),
        "expense_liability_understatement_risk": (
            "Expense or liability signals can affect reported profit and obligations for the period."
        ),
        "earnings_cashflow_mismatch": (
            "A gap between earnings and operating cash flow can raise the review priority for earnings quality and working-capital movements."
        ),
        "disclosure_inconsistency_or_obfuscation": (
            "Disclosure consistency signals can affect the analyst's ability to reconcile the filing evidence."
        ),
        "related_party_disclosure_risk": (
            "Related-party balances or transactions may require manual review to understand relationships and transaction terms."
        ),
    }
    return messages.get(risk_category, "This signal can affect how an analyst prioritizes manual report review.")


def _reviewed_candidate_status(support_level: str) -> str:
    statuses = {
        "supported": "included_in_grouped_findings",
        "weakly_supported": "included_in_weak_or_limited_signals",
        "not_supported": "retained_for_audit_only",
        "insufficient_evidence": "retained_as_data_gap",
    }
    return statuses[support_level]


def _evidence_refs_for_assessment(
    assessment: dict[str, Any],
    candidates_by_id: dict[str, dict[str, Any]],
    tool_findings_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    candidate = candidates_by_id[assessment["candidate_id"]]
    refs = list(assessment["cited_evidence_refs"])
    for tool_result_id in candidate["linked_tool_result_ids"]:
        tool_finding = tool_findings_by_id.get(tool_result_id)
        if tool_finding:
            refs.extend(tool_finding.get("evidence_refs", []))
    return refs


def _dedupe_strings(values: Any) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _dedupe_evidence_refs(
    evidence_refs: Any,
    source_lineage_by_report_id: dict[str, dict[str, str]],
    source_navigation_by_ref_id: dict[str, dict[str, Any]],
    report_language: str = "vi",
    *,
    exclude_coverage_limitations: bool = False,
) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for evidence_ref in evidence_refs:
        if exclude_coverage_limitations and _is_coverage_limitation_ref(evidence_ref):
            continue
        key = (evidence_ref["evidence_ref_type"], evidence_ref["ref_id"], evidence_ref["role"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(
            _enrich_evidence_ref(
                evidence_ref,
                source_lineage_by_report_id,
                source_navigation_by_ref_id,
                report_language,
            )
        )
    return deduped


def _is_coverage_limitation_ref(evidence_ref: dict[str, Any]) -> bool:
    ref_id = str(evidence_ref.get("ref_id") or "")
    return ref_id.startswith("LIMIT_") or ":LIMIT_" in ref_id


def _enrich_evidence_ref(
    evidence_ref: dict[str, Any],
    source_lineage_by_report_id: dict[str, dict[str, str]],
    source_navigation_by_ref_id: dict[str, dict[str, Any]],
    report_language: str = "vi",
) -> dict[str, Any]:
    enriched = copy.deepcopy(evidence_ref)
    enriched["summary"] = _evidence_ref_summary(enriched, report_language)
    enriched.setdefault("source_document_id", None)
    enriched.setdefault("source_slot_role", None)
    enriched.setdefault("page_number", None)
    enriched.setdefault("source_excerpt", None)
    enriched.setdefault("geometry", None)
    if evidence_ref.get("evidence_ref_type") not in {
        "table_cell",
        "table_row",
        "note_span",
        "variance_explanation_span",
        "accounting_policy_note_span",
        "related_party_note_span",
    }:
        return enriched
    report_id = str(evidence_ref.get("ref_id", "")).split(":", 1)[0]
    lineage = source_lineage_by_report_id.get(report_id)
    if lineage:
        enriched["source_document_id"] = lineage["source_document_id"]
        enriched["source_slot_role"] = lineage["source_slot_role"]
    navigation = source_navigation_by_ref_id.get(str(evidence_ref.get("ref_id", "")))
    if navigation:
        enriched["page_number"] = navigation.get("page_number")
        enriched["source_excerpt"] = navigation.get("source_excerpt")
        enriched["summary"] = _evidence_ref_summary(enriched, report_language)
    return enriched


def _evidence_ref_summary(evidence_ref: dict[str, Any], report_language: str = "vi") -> str:
    ref_id = str(evidence_ref.get("ref_id") or "")
    ref_type = str(evidence_ref.get("evidence_ref_type") or "")
    if ref_type == "tool_result":
        return _tool_ref_summary(ref_id, report_language)
    if ref_type == "rule":
        return _rule_ref_summary(ref_id, report_language)
    if ref_type in {"table_cell", "table_row"}:
        return _table_ref_summary(ref_id, report_language)
    if ref_type.endswith("note_span") or ref_type == "note":
        return _localized(report_language, vi="Đoạn thuyết minh trong báo cáo", en="Report note excerpt")
    if ref_type == "variance_explanation_span":
        return _localized(report_language, vi="Đoạn giải trình biến động", en="Variance explanation excerpt")
    return ref_id


def _tool_ref_summary(ref_id: str, report_language: str) -> str:
    account = _account_label_from_ref(ref_id, report_language)
    period = _period_label_from_ref(ref_id)
    if "REV_GROWTH" in ref_id:
        return _localized(
            report_language,
            vi=f"Tăng trưởng doanh thu {period}".strip(),
            en=f"Revenue growth {period}".strip(),
        )
    if "REC_VS_REV" in ref_id:
        return _localized(
            report_language,
            vi=f"So sánh phải thu và doanh thu {period}".strip(),
            en=f"Receivables versus revenue {period}".strip(),
        )
    if account:
        return _localized(report_language, vi=f"Kiểm tra {account} {period}".strip(), en=f"{account} check {period}".strip())
    return _localized(report_language, vi="Kết quả kiểm tra công cụ", en="Tool check result")


def _rule_ref_summary(ref_id: str, report_language: str) -> str:
    labels = {
        "RULE_REVENUE_RECEIVABLES_GROWTH": (
            "Quy tắc: Tăng trưởng doanh thu so với khoản phải thu",
            "Rule: Revenue growth versus receivables",
        ),
        "RULE_EARNINGS_CASHFLOW_MISMATCH": (
            "Quy tắc: Chênh lệch lợi nhuận và dòng tiền",
            "Rule: Earnings and cash-flow mismatch",
        ),
    }
    vi, en = labels.get(ref_id, (f"Quy tắc: {ref_id}", f"Rule: {ref_id}"))
    return _localized(report_language, vi=vi, en=en)


def _table_ref_summary(ref_id: str, report_language: str) -> str:
    report_id, _, cell_id = ref_id.partition(":")
    account = _account_label_from_ref(cell_id, report_language) or _localized(
        report_language,
        vi="Chỉ tiêu báo cáo",
        en="Report line item",
    )
    period = _period_label_from_ref(ref_id)
    return f"{account} {period}".strip()


def _account_label_from_ref(ref_id: str, report_language: str) -> str | None:
    labels = {
        "REVENUE": ("Doanh thu", "Revenue"),
        "TRADE_RECEIVABLES": ("Phải thu khách hàng", "Trade receivables"),
        "PROFIT_AFTER_TAX": ("Lợi nhuận sau thuế", "Profit after tax"),
        "OPERATING_CASH_FLOW": ("Dòng tiền kinh doanh", "Operating cash flow"),
        "FVTPL": ("Tài sản FVTPL", "FVTPL assets"),
        "MARGIN": ("Cho vay ký quỹ", "Margin lending"),
    }
    for token, (vi, en) in labels.items():
        if token in ref_id:
            return _localized(report_language, vi=vi, en=en)
    return None


def _period_label_from_ref(ref_id: str) -> str:
    match = re.search(r"(20\d{2})_Q([1-4])", ref_id)
    if match is None:
        return ""
    return f"Q{match.group(2)} {match.group(1)}"


def _reject_prohibited_final_report_text(report: dict[str, Any]) -> None:
    text = json.dumps(report).lower()
    for prohibited in ["fraud", "manipulat", "conceal", "intent", "illegal", "legal misstatement"]:
        if prohibited in text:
            raise ValueError("FinalReport contains prohibited legal or misconduct wording")
    model_authority_terms = [
        "controls workflow",
        "workflow orchestration",
        "stage orchestration",
        "controls finding",
        "controls severity",
        "finding severity",
        "finding inclusion",
        "support level",
    ]
    if "report synthesis model" in text and any(term in text for term in model_authority_terms):
        raise ValueError(
            "FinalReport contains prohibited report synthesis model authority wording"
        )
