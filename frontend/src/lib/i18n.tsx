"use client";

import { createContext, useContext, useState, useCallback, type ReactNode } from "react";

export type Locale = "en" | "vi";

export const STRINGS: Record<string, { en: string; vi: string }> = {
  // Header
  "header.brand": { en: "Vinacount", vi: "Vinacount" },
  "header.demo_badge": { en: "Controlled Demo", vi: "Demo có kiểm soát" },

  // Phases
  "phase.finding_sources": { en: "Finding sources", vi: "Tìm nguồn" },
  "phase.confirming_sources": { en: "Confirming sources", vi: "Xác nhận nguồn" },
  "phase.analyzing": { en: "Analyzing filing", vi: "Phân tích báo cáo" },
  "phase.generating_report": { en: "Generating report", vi: "Tạo báo cáo" },

  // Phase activity labels
  "activity.finding_sources": { en: "Searching for filing documents...", vi: "Đang tìm kiếm tài liệu..." },
  "activity.confirming_sources": { en: "Waiting for your confirmation", vi: "Chờ xác nhận của bạn" },
  "activity.analyzing": { en: "Analyzing financial statements...", vi: "Đang phân tích báo cáo tài chính..." },
  "activity.generating_report": { en: "Writing risk-signal report...", vi: "Đang viết báo cáo tín hiệu rủi ro..." },

  // Runtime stage labels
  "stage.source_discovery": { en: "Find source documents", vi: "Tìm tài liệu nguồn" },
  "stage.source_confirmation": { en: "Confirm selected sources", vi: "Xác nhận nguồn đã chọn" },
  "stage.cache_lookup": { en: "Check reusable analysis memory", vi: "Kiểm tra dữ liệu phân tích đã lưu" },
  "stage.extraction": { en: "Build filing memory", vi: "Tạo dữ liệu báo cáo" },
  "stage.tool_analysis": { en: "Run accounting checks", vi: "Chạy kiểm tra kế toán" },
  "stage.detector_assessment": { en: "Assess evidence packets", vi: "Đánh giá gói bằng chứng" },
  "stage.aggregation": { en: "Aggregate risk signals", vi: "Tổng hợp tín hiệu rủi ro" },
  "stage.report_generation": { en: "Assemble final report", vi: "Tạo báo cáo cuối" },

  // Progress labels
  "progress.analysis_step": { en: "Analysis step", vi: "Bước phân tích" },
  "progress.detector_packets": { en: "Evidence packets assessed", vi: "Gói bằng chứng đã đánh giá" },
  "progress.items_processed": { en: "Items processed", vi: "Mục đã xử lý" },
  "progress.live_extraction_runs": { en: "Live extractions", vi: "Trích xuất trực tiếp" },
  "progress.report_memory_artifacts_reused": { en: "Filing memory reused", vi: "Dữ liệu báo cáo tái sử dụng" },
  "progress.raw_ocr_cache_hits": { en: "Cached extractions", vi: "Trích xuất từ bộ nhớ đệm" },
  "progress.raw_ocr_cache_misses": { en: "New extractions", vi: "Trích xuất mới" },
  "progress.cache_entries_checked": { en: "Cache entries checked", vi: "Mục đã kiểm tra" },
  "progress.cache_hits": { en: "Cache matches", vi: "Khớp bộ nhớ đệm" },
  "progress.source_slots_checked": { en: "Source slots checked", vi: "Vị trí nguồn đã kiểm tra" },
  "progress.reusable_report_memory_artifacts": { en: "Reusable filing memory", vi: "Dữ liệu báo cáo tái sử dụng được" },
  "progress.cache_misses": { en: "Cache misses", vi: "Không khớp bộ nhớ đệm" },

  // Chat input
  "input.placeholder": { en: "e.g. NKG Q3 2021 consolidated", vi: "VD: NKG Q3 2021 hợp nhất" },
  "input.disabled_confirm": { en: "Confirm sources to continue", vi: "Xác nhận nguồn để tiếp tục" },
  "input.disabled_running": { en: "Analysis in progress...", vi: "Đang phân tích..." },
  "input.submit": { en: "Analyze", vi: "Phân tích" },
  "input.parse_error": {
    en: "Use a stock ticker such as NKG, plus quarter, year, and basis.",
    vi: "Vui lòng dùng mã chứng khoán như NKG, kèm quý, năm và loại báo cáo.",
  },

  // Confirmation card
  "confirm.title": { en: "I'll analyze this filing:", vi: "Tôi sẽ phân tích báo cáo này:" },
  "confirm.company": { en: "Company", vi: "Công ty" },
  "confirm.period": { en: "Period", vi: "Kỳ" },
  "confirm.basis": { en: "Basis", vi: "Loại" },
  "confirm.consolidated": { en: "Consolidated", vi: "Hợp nhất" },
  "confirm.separate": { en: "Separate", vi: "Riêng lẻ" },
  "confirm.cancel": { en: "Cancel", vi: "Hủy" },
  "confirm.begin": { en: "Begin Analysis", vi: "Bắt đầu phân tích" },
  "confirm.confirmed": { en: "Analysis confirmed", vi: "Đã xác nhận phân tích" },

  // Filing intent draft
  "draft.title": { en: "Confirm your request", vi: "Xác nhận yêu cầu" },
  "draft.ticker_label": { en: "Ticker", vi: "Mã CK" },
  "draft.quarter_label": { en: "Quarter", vi: "Quý" },
  "draft.year_label": { en: "Year", vi: "Năm" },
  "draft.basis_label": { en: "Report basis", vi: "Loại báo cáo" },
  "draft.consolidated": { en: "Consolidated", vi: "Hợp nhất" },
  "draft.separate": { en: "Separate", vi: "Riêng lẻ" },
  "draft.begin": { en: "Begin analysis", vi: "Bắt đầu phân tích" },
  "draft.dismiss": { en: "Edit request", vi: "Sửa yêu cầu" },
  "draft.ticker_placeholder": { en: "e.g. NKG", vi: "VD: NKG" },
  "draft.year_placeholder": { en: "e.g. 2024", vi: "VD: 2024" },
  "draft.select": { en: "Select...", vi: "Chọn..." },

  // Source confirmation
  "source.target": { en: "Target", vi: "Mục tiêu" },
  "source.prior_year": { en: "Prior Year", vi: "Năm trước" },
  "source.confirm_sources": { en: "Confirm Sources", vi: "Xác nhận nguồn" },
  "source.reject": { en: "Reject source", vi: "Từ chối nguồn" },
  "source.retry": { en: "Retry source", vi: "Tìm lại nguồn" },
  "source.view_on_vietstock": { en: "View on Vietstock", vi: "Xem trên Vietstock" },
  "source.confirmed_summary": { en: "Sources confirmed", vi: "Đã xác nhận nguồn" },

  // Source slot basis labels
  "source.basis.consolidated": { en: "Consolidated", vi: "Hợp nhất" },
  "source.basis.separate": { en: "Separate", vi: "Riêng lẻ" },
  "source.basis.parent": { en: "Parent", vi: "Công ty mẹ" },

  // Source slot statuses
  "source.status.pending_discovery": { en: "Discovering", vi: "Đang tìm" },
  "source.status.ready_for_review": { en: "Ready for review", vi: "Sẵn sàng xem" },
  "source.status.rejected": { en: "Rejected", vi: "Đã từ chối" },
  "source.status.retrying_discovery": { en: "Retrying", vi: "Đang thử lại" },
  "source.status.locked": { en: "Confirmed", vi: "Đã xác nhận" },
  "source.status.unavailable": { en: "Unavailable", vi: "Không khả dụng" },

  // Source slot card labels
  "source.doc_identity": { en: "Document identity", vi: "Nhận dạng tài liệu" },
  "source.pages": { en: "pages", vi: "trang" },
  "source.searchable": { en: "Searchable", vi: "Có thể tìm kiếm" },
  "source.rejected_label": { en: "Rejected", vi: "Đã từ chối" },
  "source.filing_status.original": { en: "Original", vi: "Bản gốc" },
  "source.filing_status.reviewed": { en: "Reviewed", vi: "Đã soát xét" },
  "source.filing_status.audited": { en: "Audited", vi: "Đã kiểm toán" },
  "source.filing_status.amended": { en: "Amended", vi: "Đã điều chỉnh" },
  "source.filing_status.unaudited": { en: "Unaudited", vi: "Chưa soát xét" },
  "source.origin_label": { en: "Source", vi: "Nguồn" },
  "source.zip_package_note": {
    en: "Selected PDF from Vietstock package",
    vi: "PDF được chọn từ gói Vietstock",
  },
  "source.zip_member_file": { en: "Selected file", vi: "Tệp được chọn" },
  "source.unavailable_detail": {
    en: "This source could not be found. You can retry discovery or stop the analysis.",
    vi: "Không tìm thấy nguồn này. Bạn có thể tìm lại hoặc dừng phân tích.",
  },
  "source.select_candidate": {
    en: "Multiple documents found. Select the correct source:",
    vi: "Tìm thấy nhiều tài liệu. Chọn nguồn phù hợp:",
  },
  "source.pick_candidate": {
    en: "Select",
    vi: "Chọn",
  },
  "source.preview_pdf": {
    en: "Preview",
    vi: "Xem trước",
  },

  // Locale switch
  "locale.switch_label": {
    en: "Language",
    vi: "Ngôn ngữ",
  },

  // Source discovery warning codes
  "warning.vietstock_package_member_selected": {
    en: "A PDF was selected from a Vietstock package",
    vi: "Một PDF đã được chọn từ gói Vietstock",
  },
  "warning.missing_exact_source": {
    en: "Exact source match not found",
    vi: "Không tìm thấy nguồn khớp chính xác",
  },
  "warning.basis_mismatch": {
    en: "Report basis does not match request",
    vi: "Loại báo cáo không khớp với yêu cầu",
  },
  "warning.token_failure": {
    en: "Source authentication failed",
    vi: "Xác thực nguồn thất bại",
  },
  "warning.api_parse_failure": {
    en: "Source response could not be parsed",
    vi: "Không thể phân tích phản hồi từ nguồn",
  },
  "warning.vietstock_download_failed": {
    en: "Vietstock download failed",
    vi: "Tải từ Vietstock thất bại",
  },
  "warning.source_quality_skipped_zip_member": {
    en: "A ZIP member was skipped during quality check",
    vi: "Một thành phần ZIP đã bị bỏ qua khi kiểm tra chất lượng",
  },
  "warning.source_quality_evidence_inconclusive": {
    en: "Source quality evidence is inconclusive",
    vi: "Bằng chứng chất lượng nguồn chưa rõ ràng",
  },
  "warning.source_quality_evidence_extraction_failed": {
    en: "Could not extract source quality evidence",
    vi: "Không thể trích xuất bằng chứng chất lượng nguồn",
  },
  "warning.wrong_company": {
    en: "Company does not match the filing intent",
    vi: "Công ty không khớp với yêu cầu phân tích",
  },
  "warning.wrong_period": {
    en: "Period does not match the filing intent",
    vi: "Kỳ báo cáo không khớp với yêu cầu phân tích",
  },
  "warning.wrong_basis": {
    en: "Report basis does not match the filing intent",
    vi: "Loại báo cáo không khớp với yêu cầu phân tích",
  },
  "warning.wrong_language": {
    en: "Language does not match expectations",
    vi: "Ngôn ngữ không khớp với kỳ vọng",
  },
  "warning.not_full_financial_statement": {
    en: "Document is not a full financial statement",
    vi: "Tài liệu không phải báo cáo tài chính đầy đủ",
  },
  "warning.ambiguous_financial_statement_candidates": {
    en: "Multiple financial statement candidates found",
    vi: "Tìm thấy nhiều ứng viên báo cáo tài chính",
  },
  "warning.ambiguous_reviewed_supersession_order": {
    en: "Review supersession order is ambiguous",
    vi: "Thứ tự ưu tiên soát xét không rõ ràng",
  },
  "warning.searchable_identity_not_confirmed": {
    en: "Searchable version identity not confirmed",
    vi: "Chưa xác nhận nhận dạng phiên bản có thể tìm kiếm",
  },
  "warning.inconclusive_reviewed_full_fs_identity": {
    en: "Full financial statement identity is inconclusive",
    vi: "Nhận dạng báo cáo tài chính đầy đủ chưa rõ ràng",
  },
  "warning.corrected_value_resolution_required": {
    en: "Corrected value resolution is required",
    vi: "Cần giải quyết giá trị đã điều chỉnh",
  },

  // Filing cache lookup warning codes
  "warning.filing_cache_lookup_report_memory_reusable": {
    en: "Validated filing memory reused; extraction skipped",
    vi: "Dữ liệu báo cáo đã xác nhận được tái sử dụng; bỏ qua trích xuất",
  },
  "warning.filing_cache_lookup_source_only": {
    en: "Source-bound cached data reused; filing memory rebuilt",
    vi: "Dữ liệu nguồn đã lưu được tái sử dụng; dữ liệu báo cáo được tái tạo",
  },
  "warning.filing_cache_lookup_stale_rebuild_required": {
    en: "Cached filing memory outdated; rebuilt from cached source data",
    vi: "Dữ liệu báo cáo đã lưu lỗi thời; tái tạo từ dữ liệu nguồn",
  },
  "warning.filing_cache_lookup_incomplete_source_pair": {
    en: "Incomplete source pair in cache; extraction will proceed",
    vi: "Thiếu cặp nguồn trong bộ nhớ đệm; tiếp tục trích xuất",
  },
  "warning.filing_cache_lookup_incomplete_report_memory_pair": {
    en: "Incomplete filing memory pair in cache; extraction will proceed",
    vi: "Thiếu cặp dữ liệu báo cáo trong bộ nhớ đệm; tiếp tục trích xuất",
  },
  "warning.filing_cache_lookup_miss": {
    en: "No reusable data found; extraction will proceed",
    vi: "Không tìm thấy dữ liệu tái sử dụng; tiếp tục trích xuất",
  },
  "warning.cache_reused_report_memory_artifacts": {
    en: "Filing memory artifacts reused from validated cache",
    vi: "Hiện vật dữ liệu báo cáo được tái sử dụng từ bộ nhớ đệm",
  },
  "warning.raw_ocr_cache_activity": {
    en: "Cached source extraction data used during rebuild",
    vi: "Dữ liệu trích xuất nguồn đã lưu được sử dụng khi tái tạo",
  },

  // Filing cache lookup: recent-source refresh and supersession warning codes (#186)
  "warning.filing_cache_lookup_recent_source_refresh_unavailable": {
    en: "Source refresh unavailable; cached analysis cannot be reused",
    vi: "Không thể làm mới nguồn; dữ liệu phân tích đã lưu không thể tái sử dụng",
  },
  "warning.filing_cache_lookup_recent_source_refresh_invalid": {
    en: "Source refresh returned invalid data; cached analysis cannot be reused",
    vi: "Dữ liệu làm mới nguồn không hợp lệ; dữ liệu phân tích đã lưu không thể tái sử dụng",
  },
  "warning.filing_cache_lookup_ambiguous_source_requires_resolution": {
    en: "Source selection is ambiguous and requires resolution before analysis can continue",
    vi: "Nguồn tài liệu không xác định rõ ràng, cần giải quyết trước khi tiếp tục phân tích",
  },
  "warning.filing_cache_lookup_refreshed_source_metadata_missing": {
    en: "Required source metadata is missing; reuse safety could not be established",
    vi: "Thiếu dữ liệu nguồn bắt buộc; không thể xác minh tính an toàn để tái sử dụng",
  },
  "warning.filing_cache_lookup_refreshed_source_fingerprint_changed": {
    en: "Source document has changed; rebuilding analysis from current data",
    vi: "Tài liệu nguồn đã thay đổi; đang tái tạo phân tích từ dữ liệu hiện tại",
  },
  "warning.filing_cache_lookup_refreshed_source_identity_changed": {
    en: "Source document identity has changed; rebuilding analysis from current data",
    vi: "Nhận dạng tài liệu nguồn đã thay đổi; đang tái tạo phân tích từ dữ liệu hiện tại",
  },
  "warning.filing_cache_lookup_refreshed_source_status_changed": {
    en: "Source filing status has changed; rebuilding analysis from current data",
    vi: "Trạng thái tài liệu nguồn đã thay đổi; đang tái tạo phân tích từ dữ liệu hiện tại",
  },
  "warning.filing_cache_lookup_refreshed_source_superseded": {
    en: "Source document has been superseded by a newer filing; rebuilding analysis",
    vi: "Tài liệu nguồn đã bị thay thế bởi bản mới hơn; đang tái tạo phân tích",
  },
  "warning.filing_cache_lookup_refreshed_source_corrected": {
    en: "Source document has been corrected; rebuilding analysis from corrected data",
    vi: "Tài liệu nguồn đã được điều chỉnh; đang tái tạo phân tích từ dữ liệu đã điều chỉnh",
  },
  "warning.filing_cache_lookup_invalid_blocked": {
    en: "Reuse safety could not be established; processing was stopped",
    vi: "Không thể xác minh tính an toàn để tái sử dụng; xử lý đã bị dừng",
  },

  // Source slot role labels for cache-lookup warning context
  "source_role.target": {
    en: "target period",
    vi: "kỳ mục tiêu",
  },
  "source_role.prior_year": {
    en: "prior-year comparison period",
    vi: "kỳ đối chiếu cùng kỳ năm trước",
  },

  // Completion card
  "complete.title": { en: "Analysis complete", vi: "Phân tích hoàn tất" },
  "complete.signals_found": { en: "risk signals identified", vi: "tín hiệu rủi ro" },
  "complete.view_report": { en: "Open report", vi: "Mở báo cáo" },
  "complete.elapsed": { en: "elapsed", vi: "thời gian" },

  // Failed card
  "failed.title": { en: "Analysis paused", vi: "Phân tích tạm dừng" },
  "failed.non_recoverable": { en: "Analysis failed", vi: "Phân tích thất bại" },
  "failed.resume": { en: "Resume analysis", vi: "Tiếp tục phân tích" },

  // Cancelled card
  "cancelled.title": { en: "Analysis cancelled", vi: "Đã hủy phân tích" },

  // Report reader
  "report.title": {
    en: "Financial Reporting Risk-Signal Review",
    vi: "Rà soát tín hiệu rủi ro báo cáo tài chính",
  },
  "report.triage_banner": {
    en: "This report identifies evidence-backed risk signals for manual review. It does not conclude fraud, misstatement, or irregularity.",
    vi: "Báo cáo này xác định các tín hiệu rủi ro dựa trên bằng chứng để rà soát thủ công. Báo cáo không kết luận gian lận, sai sót, hay bất thường.",
  },
  "report.context_boundary": {
    en: "No peer, sector, or macroeconomic comparison. No external market, news, or enforcement data. Unaudited interim filings may differ from audited full-year results.",
    vi: "Không so sánh với doanh nghiệp cùng ngành, ngành, hoặc kinh tế vĩ mô. Không sử dụng dữ liệu thị trường, tin tức, hoặc thực thi bên ngoài. Báo cáo tài chính giữa niên độ chưa kiểm toán có thể khác với kết quả cả năm đã kiểm toán.",
  },
  "report.review_steps_title": {
    en: "Suggested manual review steps",
    vi: "Các bước rà soát thủ công gợi ý",
  },
  "report.review_steps_note": {
    en: "These are common follow-up checks for the risk categories identified in this report, not conclusions or audit procedures.",
    vi: "Đây là các kiểm tra theo dõi phổ biến cho các loại rủi ro được xác định trong báo cáo này, không phải kết luận hay thủ tục kiểm toán.",
  },
  "report.findings": { en: "Findings", vi: "Kết quả" },
  "report.close": { en: "Close", vi: "Đóng" },
  "report.back": { en: "Back to thread", vi: "Quay lại" },
  "report.executive_summary": { en: "Summary", vi: "Tóm tắt" },
  "report.overall_assessment": { en: "Overall Assessment", vi: "Đánh giá tổng thể" },
  "report.risk_signals": { en: "Risk Signals", vi: "Tín hiệu rủi ro" },
  "report.weak_signals": { en: "Weak Signals", vi: "Tín hiệu yếu" },
  "report.method_scope": { en: "Method & Scope", vi: "Phương pháp & Phạm vi" },
  "report.limitations": { en: "Limitations", vi: "Giới hạn" },
  "report.coverage_unavailable": {
    en: "Evidence not available for analysis",
    vi: "Bằng chứng chưa được phân tích",
  },
  "report.human_review": { en: "Human review recommended", vi: "Cần xem xét thủ công" },
  "report.severity": { en: "Severity", vi: "Mức độ" },
  "report.support": { en: "Support", vi: "Độ tin cậy" },
  "report.evidence": { en: "Evidence", vi: "Bằng chứng" },
  "report.copy": { en: "Copy report", vi: "Sao chép" },
  "report.copied": { en: "Copied", vi: "Đã sao chép" },
  "report.print": { en: "Print", vi: "In" },

  "severity_tip.high": {
    en: "Material quantitative divergence with clear evidence. Review with priority.",
    vi: "Chênh lệch định lượng trọng yếu với bằng chứng rõ ràng. Cần ưu tiên rà soát.",
  },
  "severity_tip.medium": {
    en: "Quantitative signal with clear evidence, but additional context needed before concluding.",
    vi: "Tín hiệu có bằng chứng định lượng rõ ràng nhưng cần thêm bối cảnh trước khi kết luận.",
  },
  "severity_tip.low": {
    en: "Minor quantitative signal. May warrant review if other signals are present.",
    vi: "Tín hiệu định lượng nhỏ. Có thể cần rà soát nếu có tín hiệu khác đi kèm.",
  },
  "support_tip.supported": {
    en: "Corroborated by multiple structured evidence sources from the filing.",
    vi: "Được xác nhận bởi nhiều nguồn bằng chứng có cấu trúc từ báo cáo tài chính.",
  },
  "support_tip.weakly_supported": {
    en: "Limited evidence available. Signal needs further verification before acting.",
    vi: "Bằng chứng hạn chế. Tín hiệu cần xác minh thêm trước khi hành động.",
  },
  "report.no_report": {
    en: "Report will appear here when analysis completes.",
    vi: "Báo cáo sẽ hiển thị ở đây khi phân tích hoàn tất.",
  },
  "report.generated": { en: "Generated", vi: "Ngày tạo" },
  "report.source_label": { en: "Source", vi: "Nguồn" },
  "report.deterministic_template": { en: "Deterministic template", vi: "Mẫu xác định" },
  "report.input_scope": { en: "Input scope", vi: "Phạm vi đầu vào" },
  "report.evidence_scope": { en: "Evidence scope", vi: "Phạm vi bằng chứng" },
  "report.excluded_scope": { en: "Excluded", vi: "Loại trừ" },
  "report.reporting_rule": { en: "Reporting rule", vi: "Quy tắc báo cáo" },
  "report.candidates": { en: "Candidates", vi: "Ứng viên" },
  "report.assessments": { en: "Assessments", vi: "Đánh giá" },
  "report.tool_results": { en: "Tool results", vi: "Kết quả công cụ" },
  "report.gating_records": { en: "Gating records", vi: "Bản ghi kiểm soát" },
  "report.why_this_matters": { en: "Why this matters", vi: "Tại sao điều này quan trọng" },
  "report.contradicting_evidence": { en: "Contradicting evidence", vi: "Bằng chứng trái chiều" },
  "report.missing_evidence": { en: "Missing evidence", vi: "Bằng chứng còn thiếu" },
  "report.evidence_limitation": { en: "Evidence limitation", vi: "Giới hạn bằng chứng" },
  "report.evidence_ref": { en: "Evidence ref", vi: "Mã tham chiếu bằng chứng" },
  "report.model_versions": { en: "Model versions", vi: "Phiên bản mô hình" },
  "report.not_extracted": { en: "Not extracted", vi: "Chưa trích xuất" },
  "report.unsupported_by_extraction": { en: "Unsupported by extraction", vi: "Chưa được hỗ trợ bởi bước trích xuất" },

  // Review coverage
  "review.title": { en: "Review coverage", vi: "Phạm vi xem xét" },
  "review.explanation": {
    en: "The system reviewed all candidates. Only supported or weak signals become findings.",
    vi: "Hệ thống đã xem xét tất cả ứng viên. Chỉ tín hiệu được hỗ trợ hoặc yếu mới thành phát hiện.",
  },
  "review.insufficient_evidence_expanded": {
    en: "Most candidates had insufficient evidence for assessment.",
    vi: "Phần lớn ứng viên thiếu bằng chứng để đánh giá.",
  },
  "review.insufficient_evidence": { en: "Insufficient evidence", vi: "Thiếu bằng chứng" },
  "review.not_supported": { en: "Not supported", vi: "Không hỗ trợ" },
  "review.candidates_label": { en: "candidates", vi: "ứng viên" },
  "review.source_unavailable": { en: "Source unavailable", vi: "Nguồn không khả dụng" },

  // Status
  "status.created": { en: "Created", vi: "Đã tạo" },
  "status.discovering_sources": { en: "Discovering", vi: "Đang tìm" },
  "status.awaiting_source_confirmation": { en: "Awaiting Confirmation", vi: "Chờ xác nhận" },
  "status.analyzing": { en: "Analyzing", vi: "Đang phân tích" },
  "status.failed": { en: "Failed", vi: "Thất bại" },
  "status.completed": { en: "Completed", vi: "Hoàn tất" },
  "status.cancelled": { en: "Cancelled", vi: "Đã hủy" },

  // Actions
  "action.stop": { en: "Stop", vi: "Dừng" },
  "action.resume": { en: "Resume analysis", vi: "Tiếp tục phân tích" },
  "action.retry": { en: "Retry", vi: "Thử lại" },

  // Validation errors
  "validation.title": { en: "Could not start analysis", vi: "Không thể bắt đầu phân tích" },
  "validation.runtime_error_title": { en: "Could not load analysis", vi: "Không thể tải phân tích" },
  "validation.dismiss": { en: "Dismiss", vi: "Đóng" },
  "validation.field.company_identifier": { en: "Company identifier", vi: "Mã công ty" },
  "validation.field.target_fiscal_year": { en: "Fiscal year", vi: "Năm tài chính" },
  "validation.field.target_quarter": { en: "Quarter", vi: "Quý" },
  "validation.field.report_basis_preference": { en: "Report basis", vi: "Loại báo cáo" },

  // Connection / loading
  "loading.submitting": { en: "Starting analysis...", vi: "Đang bắt đầu phân tích..." },
  "loading.report": { en: "Loading report...", vi: "Đang tải báo cáo..." },
  "error.connection": { en: "Connection error. Retrying...", vi: "Lỗi kết nối. Đang thử lại..." },

  // Mode toggle
  "mode.live": { en: "Live", vi: "Trực tiếp" },
  "mode.demo": { en: "Demo", vi: "Demo" },

  // Model selector
  "model.label": { en: "Synthesis model", vi: "Mô hình tổng hợp" },
  "model.aria_label": {
    en: "Report synthesis model",
    vi: "Mô hình tổng hợp báo cáo",
  },
  "model.tooltip": {
    en: "Writes report narrative. Does not affect findings, evidence, or risk assessments.",
    vi: "Mô hình viết phần mô tả. Không ảnh hưởng đến phát hiện, bằng chứng, hoặc đánh giá rủi ro.",
  },

  // Header actions
  "header.new_analysis": { en: "New analysis", vi: "Phân tích mới" },

  // Empty state
  "empty.title": { en: "Vinacount Analysis", vi: "Phân tích Vinacount" },
  "empty.subtitle": {
    en: "Enter a company ticker, quarter, and year to scan a Vietnamese quarterly filing for accounting irregularity risk signals.",
    vi: "Nhập mã chứng khoán, quý và năm để quét báo cáo tài chính quý về tín hiệu rủi ro bất thường kế toán.",
  },
  "empty.method_note": {
    en: "Automated extraction, rule-based detectors, and structured aggregation produce a bounded risk-signal report.",
    vi: "Trích xuất tự động, bộ phát hiện dựa trên quy tắc, và tổng hợp có cấu trúc tạo ra báo cáo tín hiệu rủi ro có phạm vi xác định.",
  },
  "empty.example_prefix": { en: "Try:", vi: "Thử:" },

  // User message
  "user.analyze": { en: "Analyze", vi: "Phân tích" },
  "user.consolidated": { en: "consolidated", vi: "hợp nhất" },
  "user.separate": { en: "separate", vi: "riêng lẻ" },

  // Report loading
  "report.loading": { en: "Loading report...", vi: "Đang tải báo cáo..." },

  // Evidence pane
  "evidence.page_abbrev": { en: "p.", vi: "tr." },
  "evidence.close": { en: "Close", vi: "Đóng" },
  "evidence.view_source": { en: "View source", vi: "Xem nguồn" },
  "evidence.open_document": { en: "Open source PDF", vi: "Mở PDF nguồn" },
  "evidence.source_excerpt": { en: "Source excerpt", vi: "Trích xuất nguồn" },
  "evidence.page": { en: "Page", vi: "Trang" },
  "evidence.page_unavailable": {
    en: "Page location unavailable for this evidence.",
    vi: "Không có thông tin vị trí trang cho bằng chứng này.",
  },
  "evidence.document_level_location": {
    en: "This evidence is linked to the source document, but this extraction artifact does not include page-level location.",
    vi: "Bằng chứng này được liên kết với tài liệu nguồn, nhưng hiện vật trích xuất này chưa có vị trí theo trang.",
  },
  "evidence.document_unavailable": {
    en: "Source document is unavailable for display.",
    vi: "Tài liệu nguồn không khả dụng để hiển thị.",
  },
  "evidence.no_source_document": {
    en: "No source document linked to this evidence.",
    vi: "Không có tài liệu nguồn liên kết với bằng chứng này.",
  },
  "evidence.target": { en: "Target filing", vi: "Báo cáo mục tiêu" },
  "evidence.prior_year": { en: "Prior year filing", vi: "Báo cáo năm trước" },
  "evidence.loading_failed": {
    en: "Could not load the source document.",
    vi: "Không thể tải tài liệu nguồn.",
  },
  "evidence.loading": {
    en: "Loading source document...",
    vi: "Đang tải tài liệu nguồn...",
  },
  "evidence.open_new_tab": {
    en: "Open PDF",
    vi: "Mở PDF",
  },
  "evidence.viewer_fallback": {
    en: "If the browser downloads this PDF or leaves the preview blank, open it in a new tab.",
    vi: "Nếu trình duyệt tải tệp xuống hoặc vùng xem bị trống, hãy mở PDF trong tab mới.",
  },

  // Error codes — user-facing localized messages
  "error.report_synthesis_unavailable": {
    en: "Report generation paused: the synthesis service is temporarily unavailable.",
    vi: "Tạo báo cáo tạm dừng: dịch vụ tổng hợp tạm thời không khả dụng.",
  },
  "error.report_narrative_invalid": {
    en: "The generated report narrative did not pass validation.",
    vi: "Nội dung báo cáo được tạo không vượt qua kiểm tra hợp lệ.",
  },
  "error.report_claim_validation_failed": {
    en: "One or more claims in the report could not be verified against evidence.",
    vi: "Một hoặc nhiều nhận định trong báo cáo không thể xác minh bằng chứng.",
  },
  "error.final_report_invalid": {
    en: "The final report structure is invalid. Please retry.",
    vi: "Cấu trúc báo cáo cuối không hợp lệ. Vui lòng thử lại.",
  },
  "error.detector_timeout": {
    en: "Evidence assessment timed out.",
    vi: "Đánh giá bằng chứng hết thời gian chờ.",
  },
  "error.detector_contract_invalid": {
    en: "An evidence assessment returned an invalid result.",
    vi: "Một đánh giá bằng chứng trả về kết quả không hợp lệ.",
  },
  "error.source_discovery_unavailable": {
    en: "Could not find source documents at this time.",
    vi: "Không thể tìm tài liệu nguồn lúc này.",
  },
  "error.source_package_unavailable": {
    en: "The confirmed source package is no longer available.",
    vi: "Gói tài liệu nguồn đã xác nhận không còn khả dụng.",
  },
  "error.cache_lookup_failed": {
    en: "Could not retrieve previously saved analysis data.",
    vi: "Không thể truy xuất dữ liệu phân tích đã lưu.",
  },
  "error.extraction_failed": {
    en: "Failed to extract data from the source documents.",
    vi: "Không thể trích xuất dữ liệu từ tài liệu nguồn.",
  },
  "error.source_artifact_unreachable": {
    en: "A confirmed source file became unavailable or failed verification.",
    vi: "Tệp nguồn đã xác nhận không còn khả dụng hoặc không qua kiểm tra.",
  },
  "error.ocr_config_missing": {
    en: "OCR provider configuration is incomplete. Please contact support.",
    vi: "Cấu hình nhà cung cấp OCR chưa đầy đủ. Vui lòng liên hệ hỗ trợ.",
  },
  "error.ocr_provider_failed": {
    en: "The OCR provider could not process the confirmed filing.",
    vi: "Nhà cung cấp OCR không thể xử lý báo cáo đã xác nhận.",
  },
  "error.raw_extraction_invalid": {
    en: "OCR output could not be converted into analysis-ready filing memory.",
    vi: "Kết quả OCR không thể chuyển đổi thành dữ liệu phân tích.",
  },
  "error.report_memory_build_failed": {
    en: "Failed to build the filing memory from extracted data.",
    vi: "Không thể tạo bộ nhớ báo cáo từ dữ liệu đã trích xuất.",
  },
  "error.tool_analysis_failed": {
    en: "One or more accounting checks failed to complete.",
    vi: "Một hoặc nhiều kiểm tra kế toán không hoàn thành.",
  },
  "error.aggregation_failed": {
    en: "Could not aggregate risk signals into findings.",
    vi: "Không thể tổng hợp tín hiệu rủi ro thành phát hiện.",
  },
  "error.internal_error": {
    en: "An unexpected error occurred. Please try again.",
    vi: "Đã xảy ra lỗi không mong muốn. Vui lòng thử lại.",
  },
  "error.filing_intent_invalid": {
    en: "The filing request could not be validated.",
    vi: "Yêu cầu phân tích báo cáo không hợp lệ.",
  },
  "error.source_identity_mismatch_after_confirmation": {
    en: "Source document identity changed after confirmation.",
    vi: "Nhận dạng tài liệu nguồn thay đổi sau khi xác nhận.",
  },
  "error.audit_bundle_failed": {
    en: "Could not assemble the audit bundle.",
    vi: "Không thể tạo gói kiểm toán.",
  },

  // Stage labels for error context
  "error_stage.source_discovery": { en: "finding sources", vi: "tìm nguồn" },
  "error_stage.source_confirmation": { en: "confirming sources", vi: "xác nhận nguồn" },
  "error_stage.cache_lookup": { en: "checking analysis memory", vi: "kiểm tra bộ nhớ đệm" },
  "error_stage.extraction": { en: "extracting data", vi: "trích xuất dữ liệu" },
  "error_stage.tool_analysis": { en: "running checks", vi: "phân tích công cụ" },
  "error_stage.detector_assessment": { en: "assessing evidence", vi: "đánh giá bằng chứng" },
  "error_stage.aggregation": { en: "aggregating results", vi: "tổng hợp kết quả" },
  "error_stage.report_generation": { en: "generating report", vi: "tạo báo cáo" },

  // Stage timeline status labels
  "stage_status.active": { en: "in progress", vi: "đang chạy" },
  "stage_status.failed": { en: "failed", vi: "thất bại" },
  "stage_status.skipped": { en: "skipped", vi: "bỏ qua" },
  "stage_status.cancelled": { en: "cancelled", vi: "đã hủy" },

  // Stage timeline heading
  "timeline.heading": { en: "Analysis Progress", vi: "Tiến trình phân tích" },

  // Run terminal state headings
  "terminal.failed_recoverable": { en: "Analysis paused", vi: "Phân tích tạm dừng" },
  "terminal.failed_non_recoverable": { en: "Analysis failed", vi: "Phân tích thất bại" },
  "terminal.cancelled": { en: "Analysis cancelled", vi: "Đã hủy phân tích" },
  "terminal.cancelled_detail": {
    en: "This analysis was stopped before completion.",
    vi: "Phân tích đã dừng trước khi hoàn tất.",
  },
  "terminal.completed": { en: "Analysis complete", vi: "Phân tích hoàn tất" },
  "terminal.report_ready": {
    en: "The risk-signal report is ready for review.",
    vi: "Báo cáo tín hiệu rủi ro sẵn sàng để xem xét.",
  },
  "terminal.view_report": { en: "Open report", vi: "Mở báo cáo" },
  "terminal.resume": { en: "Resume analysis", vi: "Tiếp tục phân tích" },
  "terminal.during": { en: "During", vi: "Trong bước" },

  // Generic error fallback
  "error.generic": {
    en: "This step could not be completed.",
    vi: "Không thể hoàn tất bước này.",
  },

  // Rejection reason codes
  "rejection.wrong_company": { en: "Wrong company", vi: "Sai công ty" },
  "rejection.wrong_period": { en: "Wrong period", vi: "Sai kỳ báo cáo" },
  "rejection.wrong_basis": { en: "Wrong basis", vi: "Sai loại báo cáo" },
  "rejection.wrong_filing_status": { en: "Wrong filing status", vi: "Sai trạng thái báo cáo" },
  "rejection.wrong_language": { en: "Wrong language", vi: "Sai ngôn ngữ" },
  "rejection.not_full_financial_statement": { en: "Not a full financial statement", vi: "Không phải báo cáo tài chính đầy đủ" },
  "rejection.source_unreadable": { en: "Source unreadable", vi: "Nguồn không đọc được" },
  "rejection.other": { en: "Other", vi: "Lý do khác" },
};

interface LocaleContextValue {
  locale: Locale;
  setLocale: (locale: Locale) => void;
  t: (key: string) => string;
}

const LocaleContext = createContext<LocaleContextValue | null>(null);

export function LocaleProvider({ children }: { children: ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>("vi");

  const setLocale = useCallback((l: Locale) => {
    setLocaleState(l);
  }, []);

  const t = useCallback(
    (key: string): string => {
      const entry = STRINGS[key];
      if (!entry) return key;
      return entry[locale];
    },
    [locale]
  );

  return (
    <LocaleContext.Provider value={{ locale, setLocale, t }}>
      {children}
    </LocaleContext.Provider>
  );
}

export function useLocale(): LocaleContextValue {
  const ctx = useContext(LocaleContext);
  if (!ctx) throw new Error("useLocale must be used within LocaleProvider");
  return ctx;
}

export function resolveWarningText(
  code: string,
  backendMessage: string,
  t: (key: string) => string,
  sourceSlotRole?: string | null,
): string {
  const i18nKey = `warning.${code}`;
  const resolved = t(i18nKey);
  const base = resolved !== i18nKey ? resolved : backendMessage;

  if (sourceSlotRole) {
    const roleKey =
      sourceSlotRole === "prior_year_same_quarter"
        ? "source_role.prior_year"
        : `source_role.${sourceSlotRole}`;
    const roleLabel = t(roleKey);
    if (roleLabel !== roleKey) {
      return `${base} (${roleLabel})`;
    }
  }

  return base;
}
