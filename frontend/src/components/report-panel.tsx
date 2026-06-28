"use client";

import {
  ArrowLeft,
  ShieldAlert,
  ShieldCheck,
  AlertTriangle,
  Info,
  FileText,
  ListChecks,
  ChevronRight,
  Eye,
  Copy,
  Check,
  Printer,
} from "lucide-react";
import { useState, useEffect, useCallback, useRef } from "react";
import { serializeReportToText } from "@/lib/report-text";
import { useLocale } from "@/lib/i18n";
import { useWorkspace } from "@/lib/workspace-store";
import type {
  CanonicalFinalReport,
  EvidenceRef,
  ReviewedCandidate,
  ReportLimitation,
  SourceConfirmation,
} from "@/types/runtime";
import type { Locale } from "@/lib/i18n";

// ---------------------------------------------------------------------------
// Evidence item normalization — accepts backend or canonical doc shape
// ---------------------------------------------------------------------------

interface NormalizedEvidence {
  evidence_type: string;
  evidence_ref: string;
  evidence_role: string;
  summary: string;
  source_document_id: string | null;
  source_slot_role: "target" | "prior_year_same_quarter" | null;
  page_number: number | null;
  source_excerpt: string | null;
  geometry: unknown | null;
}

function normalizeEvidenceItem(raw: Record<string, unknown>): NormalizedEvidence {
  return {
    evidence_type: ((raw.evidence_type ?? raw.evidence_ref_type) as string) ?? "",
    evidence_ref: ((raw.evidence_ref ?? raw.ref_id) as string) ?? "",
    evidence_role: ((raw.evidence_role ?? raw.role) as string) ?? "",
    summary: ((raw.summary) as string) ?? "",
    source_document_id: (raw.source_document_id as string | null) ?? null,
    source_slot_role: (raw.source_slot_role as "target" | "prior_year_same_quarter" | null) ?? null,
    page_number: (raw.page_number as number | null) ?? null,
    source_excerpt: (raw.source_excerpt as string | null) ?? null,
    geometry: raw.geometry ?? null,
  };
}

function normalizeEvidenceList(raw: unknown): NormalizedEvidence[] {
  if (!Array.isArray(raw)) return [];
  return raw.map((item) => normalizeEvidenceItem(item as Record<string, unknown>));
}

function toBackendEvidenceRef(n: NormalizedEvidence): EvidenceRef {
  return {
    evidence_ref_type: n.evidence_type,
    ref_id: n.evidence_ref,
    role: n.evidence_role,
    source_document_id: n.source_document_id,
    source_slot_role: n.source_slot_role,
    page_number: n.page_number,
    source_excerpt: n.source_excerpt,
    geometry: n.geometry,
  };
}

// ---------------------------------------------------------------------------
// Report JSON accessors — support both canonical and legacy field names
// ---------------------------------------------------------------------------

type ReportJson = CanonicalFinalReport["report_json"];

function getReportMetadata(json: ReportJson) {
  const raw = (json.report_metadata ?? json.metadata) as Record<string, unknown> | undefined;
  if (!raw) return undefined;
  return {
    company_name: (raw.company_name as string) ?? "",
    ticker: (raw.ticker as string) ?? "",
    period: (raw.period as string) ?? "",
    report_basis: (raw.report_basis as string) ?? "",
    filing_status: (raw.filing_status ?? raw.report_assurance_type) as string ?? "",
    source_origin: (raw.source_origin ?? raw.source_name) as string | undefined,
  };
}

function getMethodAndScope(json: ReportJson) {
  const raw = (json.method_and_scope ?? json.method_scope) as Record<string, unknown> | undefined;
  if (!raw) return undefined;
  return {
    method_summary: (raw.method_summary ?? raw.input_scope) as string | undefined,
    evidence_scope: (raw.evidence_scope ?? (raw.included_evidence_scope as string[] | undefined)?.join(", ")) as string | undefined,
    excluded_scope: (raw.excluded_scope ?? raw.excluded_context) as string[] | undefined,
    reporting_rule: (raw.reporting_rule ?? raw.final_report_role) as string | undefined,
    report_generation_mode: raw.report_generation_mode as string | undefined,
    report_synthesis_model: raw.report_synthesis_model as {
      id?: string;
      label?: string;
      provider?: string;
      selection?: string;
      invoked_for_report_generation?: boolean;
    } | undefined,
  };
}

function getOverallAssessment(json: ReportJson) {
  return json.overall_assessment as {
    overall_review_status: string;
    highest_severity: string;
    primary_risk_category: string | null;
    secondary_risk_categories: string[];
    human_review_recommended: boolean;
    confidence_summary: string;
    num_supported_findings?: number;
    num_weakly_supported_signals?: number;
    num_not_supported_candidates?: number;
    num_insufficient_evidence_items?: number;
  } | undefined;
}

function getExecutiveSummary(json: ReportJson): string[] {
  return (json.executive_summary as string[]) ?? [];
}

interface NormalizedFinding {
  finding_id: string;
  risk_category: string;
  finding_title: string;
  support_level: string;
  severity: string;
  human_review_recommendation: string;
  summary: string;
  why_this_matters: string;
  supporting_evidence: NormalizedEvidence[];
  contradicting_evidence: NormalizedEvidence[];
  missing_evidence: NormalizedEvidence[];
  limitations: string[];
}

function getGroupedFindings(json: ReportJson): NormalizedFinding[] {
  const raw = json.grouped_findings as unknown[] | undefined;
  if (!raw) return [];
  return raw.map((f) => {
    const item = f as Record<string, unknown>;
    return {
      finding_id: (item.finding_id as string) ?? "",
      risk_category: ((item.risk_category ?? item.primary_risk_category) as string) ?? "",
      finding_title: ((item.finding_title ?? item.title) as string) ?? "",
      support_level: ((item.support_level ?? (item.support_levels as string[] | undefined)?.[0]) as string) ?? "supported",
      severity: ((item.severity ?? item.final_severity) as string) ?? "unknown",
      human_review_recommendation: (item.human_review_recommendation as string) ?? "",
      summary: (item.summary as string) ?? "",
      why_this_matters: (item.why_this_matters as string) ?? "",
      supporting_evidence: normalizeEvidenceList(item.supporting_evidence ?? item.evidence_refs),
      contradicting_evidence: normalizeEvidenceList(item.contradicting_evidence),
      missing_evidence: normalizeEvidenceList(item.missing_evidence),
      limitations: (item.limitations as string[]) ?? [],
    };
  });
}

interface NormalizedWeakSignal {
  item_id: string;
  risk_category: string;
  support_level: string;
  severity: string;
  summary: string;
  available_evidence: NormalizedEvidence[];
  missing_or_limited_evidence: NormalizedEvidence[];
  human_review_recommendation: string;
}

function getWeakSignals(json: ReportJson): NormalizedWeakSignal[] {
  const raw = json.weak_or_limited_signals as unknown[] | undefined;
  if (!raw) return [];
  return raw.map((s) => {
    const item = s as Record<string, unknown>;
    return {
      item_id: (item.item_id as string) ?? "",
      risk_category: (item.risk_category as string) ?? "",
      support_level: (item.support_level as string) ?? "weakly_supported",
      severity: ((item.severity ?? item.final_severity) as string) ?? "low",
      summary: (item.summary as string) ?? "",
      available_evidence: normalizeEvidenceList(item.available_evidence ?? item.evidence_refs),
      missing_or_limited_evidence: normalizeEvidenceList(item.missing_or_limited_evidence),
      human_review_recommendation: (item.human_review_recommendation as string) ?? "",
    };
  });
}

function getReviewedCandidates(json: ReportJson): ReviewedCandidate[] {
  return (json.reviewed_candidate_audit as ReviewedCandidate[]) ?? [];
}

const EVIDENCE_REF_PATTERN = /\s*Evidence ref:\s*(\S+?)\.?$/;

function classifyLimitationType(text: string): string {
  if (EVIDENCE_REF_PATTERN.test(text)) return "evidence_surface";
  return "scope";
}

function getLimitations(json: ReportJson): ReportLimitation[] {
  const raw = json.limitations as unknown[];
  if (!raw) return [];
  if (raw.length > 0 && typeof raw[0] === "string") {
    return (raw as string[]).map((d, i) => ({
      limitation_id: `LIM-${i + 1}`,
      limitation_type: classifyLimitationType(d),
      description: d,
    }));
  }
  return raw as ReportLimitation[];
}

interface CoverageLimitation {
  limitation_id: string;
  surface: string;
  state: string;
  label: string;
  summary: string;
}

function getCoverageLimitations(json: ReportJson): CoverageLimitation[] {
  const raw = json.coverage_limitations as unknown[] | undefined;
  if (!Array.isArray(raw)) return [];
  return raw.map((item) => {
    const r = item as Record<string, unknown>;
    return {
      limitation_id: (r.limitation_id as string) ?? "",
      surface: (r.surface as string) ?? "",
      state: (r.state as string) ?? "",
      label: (r.label as string) ?? "",
      summary: (r.summary as string) ?? "",
    };
  });
}

// ---------------------------------------------------------------------------
// Localization helpers
// ---------------------------------------------------------------------------

const RISK_CATEGORY_LABELS: Record<string, { en: string; vi: string }> = {
  revenue_income_recognition_risk: {
    en: "Revenue quality risk",
    vi: "Rủi ro chất lượng doanh thu",
  },
  receivables_credit_quality_risk: {
    en: "Receivables credit quality risk",
    vi: "Rủi ro chất lượng khoản phải thu",
  },
  asset_quality_valuation_risk: {
    en: "Asset valuation risk",
    vi: "Rủi ro định giá tài sản",
  },
  inventory_cost_asset_flow_risk: {
    en: "Inventory cost and flow risk",
    vi: "Rủi ro luân chuyển hàng tồn kho",
  },
  expense_liability_understatement_risk: {
    en: "Expense and liability understatement risk",
    vi: "Rủi ro ghi nhận thiếu chi phí hoặc nợ phải trả",
  },
  earnings_cashflow_mismatch: {
    en: "Earnings and cash-flow mismatch",
    vi: "Chênh lệch lợi nhuận và dòng tiền",
  },
  disclosure_inconsistency_or_obfuscation: {
    en: "Disclosure consistency risk",
    vi: "Rủi ro nhất quán thuyết minh",
  },
  related_party_disclosure_risk: {
    en: "Related-party disclosure risk",
    vi: "Rủi ro công bố bên liên quan",
  },
};

const TITLE_LABELS_VI: Record<string, string> = {
  "Revenue quality risk signal": "Tín hiệu rủi ro chất lượng doanh thu",
  "Credit quality risk signal": "Tín hiệu rủi ro chất lượng khoản phải thu",
  "Asset valuation risk signal": "Tín hiệu rủi ro định giá tài sản",
  "Trading-book flow risk signal": "Tín hiệu rủi ro luân chuyển hàng tồn kho",
  "Provision movement risk signal": "Tín hiệu rủi ro biến động dự phòng",
  "Earnings and cash-flow risk signal": "Tín hiệu rủi ro lợi nhuận và dòng tiền",
  "Disclosure consistency risk signal": "Tín hiệu rủi ro nhất quán thuyết minh",
  "Accounting risk signal": "Tín hiệu rủi ro kế toán",
};

const ENUM_LABELS_VI: Record<string, string> = {
  high: "Cao",
  medium: "Trung bình",
  low: "Thấp",
  unknown: "Chưa rõ",
  none: "Không",
  supported: "Có bằng chứng",
  weakly_supported: "Bằng chứng yếu",
  insufficient_evidence: "Thiếu bằng chứng",
  not_supported: "Chưa được hỗ trợ",
  human_review_recommended: "Cần rà soát thủ công",
  no_final_risk_signal: "Không có tín hiệu rủi ro cuối",
  consolidated: "Hợp nhất",
  separate: "Riêng lẻ",
  parent: "Công ty mẹ",
  original: "Bản gốc",
  reviewed: "Đã soát xét",
  unaudited: "Chưa soát xét",
  audited: "Đã kiểm toán",
  risk_signals_identified: "Tín hiệu rủi ro được xác định",
  weak_or_limited_risk_signals_only: "Chỉ tín hiệu rủi ro yếu hoặc hạn chế",
  no_material_irregularity_signal_identified: "Không phát hiện tín hiệu bất thường trọng yếu",
  insufficient_evidence_for_overall_assessment: "Thiếu bằng chứng để đánh giá tổng thể",
};

const EVIDENCE_TYPE_LABELS_VI: Record<string, string> = {
  tool_result: "kiểm tra",
  rule: "quy tắc",
  table_cell: "ô bảng",
  table_row: "hàng bảng",
  note_span: "thuyết minh",
  detector_assessment: "đánh giá",
  candidate_risk: "ứng viên",
  aggregation_decision: "tổng hợp",
};

// ---------------------------------------------------------------------------
// Risk-category review steps (deterministic, frontend-only)
// ---------------------------------------------------------------------------

const REVIEW_STEPS: Record<string, { en: string[]; vi: string[] }> = {
  revenue_income_recognition_risk: {
    en: [
      "Check days sales outstanding (DSO) and receivables aging schedule",
      "Review allowance for doubtful debts relative to receivables growth",
      "Compare gross margin trend against revenue growth",
      "Check whether revenue growth is concentrated in a few customers",
      "Review related-party sales and receivables",
      "Compare Q-on-Q and interim figures with audited full-year results when available",
    ],
    vi: [
      "Kiểm tra số ngày thu tiền bình quân (DSO) và bảng phân tích tuổi nợ phải thu",
      "Rà soát dự phòng nợ phải thu khó đòi so với tăng trưởng khoản phải thu",
      "So sánh xu hướng biên lợi nhuận gộp với tăng trưởng doanh thu",
      "Kiểm tra liệu tăng trưởng doanh thu có tập trung ở một số khách hàng",
      "Rà soát doanh thu và khoản phải thu với bên liên quan",
      "So sánh số liệu quý và giữa niên độ với kết quả cả năm đã kiểm toán khi có",
    ],
  },
  receivables_credit_quality_risk: {
    en: [
      "Review receivables aging and concentration by customer",
      "Check adequacy of allowance for doubtful debts",
      "Compare receivables days outstanding with industry norms",
      "Review related-party receivables and intercompany balances",
    ],
    vi: [
      "Rà soát tuổi nợ phải thu và mức độ tập trung theo khách hàng",
      "Kiểm tra mức đầy đủ của dự phòng nợ khó đòi",
      "So sánh số ngày phải thu với chuẩn ngành",
      "Rà soát khoản phải thu bên liên quan và số dư liên công ty",
    ],
  },
  asset_quality_valuation_risk: {
    en: [
      "Review asset revaluation or impairment testing methodology",
      "Check whether fair-value estimates rely on observable inputs",
      "Compare asset turnover ratios with prior periods",
    ],
    vi: [
      "Rà soát phương pháp đánh giá lại hoặc kiểm tra suy giảm tài sản",
      "Kiểm tra liệu ước tính giá trị hợp lý có dựa trên dữ liệu quan sát được",
      "So sánh tỷ lệ vòng quay tài sản với các kỳ trước",
    ],
  },
  inventory_cost_asset_flow_risk: {
    en: [
      "Review inventory turnover and aging by category",
      "Check inventory write-down adequacy and methodology",
      "Compare cost-of-goods-sold margin trend with revenue growth",
    ],
    vi: [
      "Rà soát vòng quay và tuổi hàng tồn kho theo danh mục",
      "Kiểm tra mức đầy đủ và phương pháp trích lập dự phòng giảm giá hàng tồn kho",
      "So sánh xu hướng biên giá vốn với tăng trưởng doanh thu",
    ],
  },
  expense_liability_understatement_risk: {
    en: [
      "Review accrued expenses and provisions for completeness",
      "Check unusual decreases in payables or accruals relative to activity",
      "Compare expense ratios with prior periods and industry norms",
    ],
    vi: [
      "Rà soát chi phí dồn tích và dự phòng về tính đầy đủ",
      "Kiểm tra giảm bất thường trong phải trả hoặc dồn tích so với hoạt động",
      "So sánh tỷ lệ chi phí với các kỳ trước và chuẩn ngành",
    ],
  },
  earnings_cashflow_mismatch: {
    en: [
      "Review the operating cash-flow bridge (profit to cash)",
      "Check receivables, inventory, and payables movements individually",
      "Assess whether working-capital cycle explains the divergence",
      "Compare interim cash-flow pattern with audited full-year results when available",
    ],
    vi: [
      "Rà soát cầu nối dòng tiền hoạt động (lợi nhuận sang tiền)",
      "Kiểm tra biến động khoản phải thu, hàng tồn kho, và phải trả riêng lẻ",
      "Đánh giá liệu chu kỳ vốn lưu động có giải thích được sự lệch pha",
      "So sánh mô hình dòng tiền giữa niên độ với kết quả cả năm đã kiểm toán khi có",
    ],
  },
  disclosure_inconsistency_or_obfuscation: {
    en: [
      "Cross-check key figures between notes, statements, and supplementary disclosures",
      "Review accounting policy changes and their quantitative impact",
      "Check whether note disclosures are consistent with statement line items",
    ],
    vi: [
      "Đối chiếu các chỉ tiêu chính giữa thuyết minh, báo cáo, và công bố bổ sung",
      "Rà soát thay đổi chính sách kế toán và ảnh hưởng định lượng",
      "Kiểm tra liệu thuyết minh có nhất quán với các khoản mục trên báo cáo",
    ],
  },
  related_party_disclosure_risk: {
    en: [
      "Review related-party transaction volumes and pricing terms",
      "Check whether related-party balances are separately disclosed",
      "Assess whether related-party activity is consistent with business rationale",
    ],
    vi: [
      "Rà soát khối lượng giao dịch và điều khoản giá với bên liên quan",
      "Kiểm tra liệu số dư bên liên quan có được trình bày riêng",
      "Đánh giá liệu hoạt động bên liên quan có phù hợp với logic kinh doanh",
    ],
  },
};

function labelFromSnake(value: string): string {
  return value.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

// Rewrite awkward Vietnamese signed-multiple cash-flow phrasing at display layer.
// "âm gấp -1,86 lần lợi nhuận sau thuế dương" →
// "âm, với độ lớn tương đương 1,86 lần lợi nhuận sau thuế dương"
function cleanDisplayText(text: string): string {
  return text.replace(
    /âm gấp -?([\d,.]+) lần/g,
    "âm, với độ lớn tương đương $1 lần",
  );
}

function formatRiskCategory(cat: string, locale: Locale): string {
  const entry = RISK_CATEGORY_LABELS[cat];
  if (entry) return locale === "vi" ? entry.vi : entry.en;
  return labelFromSnake(cat);
}

function formatSeverity(severity: string, locale: Locale): string {
  if (locale === "vi") return ENUM_LABELS_VI[severity] ?? labelFromSnake(severity);
  return severity.charAt(0).toUpperCase() + severity.slice(1);
}

function formatSupportLevel(level: string, locale: Locale): string {
  if (locale === "vi") return ENUM_LABELS_VI[level] ?? labelFromSnake(level);
  return labelFromSnake(level);
}

function formatReviewStatus(status: string, locale: Locale): string {
  if (locale === "vi") return ENUM_LABELS_VI[status] ?? labelFromSnake(status);
  return labelFromSnake(status);
}

function formatEnum(value: string, locale: Locale): string {
  if (locale === "vi") return ENUM_LABELS_VI[value] ?? value;
  return value;
}

function formatFindingTitle(title: string, locale: Locale): string {
  if (locale === "vi") return TITLE_LABELS_VI[title] ?? title;
  return title;
}

function formatEvidenceType(type: string, locale: Locale): string {
  if (locale === "vi") return EVIDENCE_TYPE_LABELS_VI[type] ?? type;
  return type;
}

// ---------------------------------------------------------------------------
// Badge components
// ---------------------------------------------------------------------------

function SeverityBadge({ severity }: { severity: string }) {
  const { t, locale } = useLocale();
  const styles: Record<string, string> = {
    high: "bg-destructive/10 text-destructive",
    medium: "bg-warning-color/15 text-warning-foreground",
    low: "bg-muted text-muted-foreground",
  };
  const tooltipKey = `severity_tip.${severity}`;
  const tooltip = t(tooltipKey);
  const hasTooltip = tooltip !== tooltipKey;
  return (
    <span
      className={`inline-flex items-center rounded-md px-2 py-0.5 text-xs font-medium cursor-help decoration-dotted underline underline-offset-2 decoration-current/30 ${styles[severity] ?? styles.low}`}
      title={hasTooltip ? tooltip : undefined}
      aria-label={hasTooltip ? `${formatSeverity(severity, locale)}: ${tooltip}` : undefined}
    >
      {formatSeverity(severity, locale)}
    </span>
  );
}

function SupportBadge({ level }: { level: string }) {
  const { t, locale } = useLocale();
  const isSupported = level === "supported";
  const tooltipKey = `support_tip.${level}`;
  const tooltip = t(tooltipKey);
  const hasTooltip = tooltip !== tooltipKey;
  return (
    <span
      className={`inline-flex items-center rounded-md px-2 py-0.5 text-xs font-medium cursor-help decoration-dotted underline underline-offset-2 decoration-current/30 ${isSupported ? "bg-primary/10 text-primary" : "bg-muted text-muted-foreground"}`}
      title={hasTooltip ? tooltip : undefined}
      aria-label={hasTooltip ? `${formatSupportLevel(level, locale)}: ${tooltip}` : undefined}
    >
      {formatSupportLevel(level, locale)}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Collapsible section
// ---------------------------------------------------------------------------

function Collapsible({
  title,
  icon: Icon,
  defaultOpen = false,
  count,
  countDetail,
  children,
}: {
  title: string;
  icon: React.ComponentType<{ className?: string }>;
  defaultOpen?: boolean;
  count?: number;
  countDetail?: string;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section>
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 w-full py-2.5 text-left group rounded-md hover:bg-muted/50 transition-colors focus-visible:ring-2 focus-visible:ring-ring/50"
        aria-expanded={open}
      >
        <Icon className="size-4 text-muted-foreground shrink-0" />
        <span className="text-base font-semibold text-foreground flex-1">
          {title}
          {count != null && (
            <span className="ml-1.5 text-xs font-normal text-muted-foreground">({count})</span>
          )}
        </span>
        {countDetail && (
          <span className="text-xs text-muted-foreground mr-2">{countDetail}</span>
        )}
        <ChevronRight
          className={`size-3.5 text-muted-foreground transition-transform duration-200 ${open ? "rotate-90" : ""}`}
          style={{ transitionTimingFunction: "var(--ease-out-expo)" }}
        />
      </button>
      <div className="collapse-grid" data-open={open}>
        <div>
          <div className="pl-6 pb-2">{children}</div>
        </div>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Loading skeleton
// ---------------------------------------------------------------------------

function ReportSkeleton() {
  return (
    <div className="space-y-6 animate-pulse max-w-3xl mx-auto px-6 py-8">
      <div className="space-y-3 text-center">
        <div className="h-5 w-3/4 rounded bg-muted mx-auto" />
        <div className="h-3 w-1/2 rounded bg-muted mx-auto" />
        <div className="h-3 w-1/3 rounded bg-muted mx-auto" />
      </div>
      <div className="rounded-lg border border-border bg-card p-5 space-y-3">
        <div className="h-4 w-2/3 rounded bg-muted" />
        <div className="h-3 w-full rounded bg-muted" />
        <div className="h-3 w-5/6 rounded bg-muted" />
      </div>
      <div className="space-y-3">
        <div className="h-3 w-1/4 rounded bg-muted" />
        <div className="h-3 w-full rounded bg-muted" />
        <div className="h-3 w-4/5 rounded bg-muted" />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Exported report reader — replaces the thread when open
// ---------------------------------------------------------------------------

export function ReportReader() {
  const { t, locale } = useLocale();
  const { reportReaderOpen, closeReportReader, selectedEvidence, closeEvidence, report, run } = useWorkspace();
  const backButtonRef = useRef<HTMLButtonElement>(null);
  const [copied, setCopied] = useState(false);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      if (selectedEvidence) {
        closeEvidence();
      } else {
        closeReportReader();
      }
    },
    [selectedEvidence, closeEvidence, closeReportReader],
  );

  useEffect(() => {
    if (!reportReaderOpen) return;
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [reportReaderOpen, handleKeyDown]);

  useEffect(() => {
    if (reportReaderOpen) {
      backButtonRef.current?.focus();
    }
  }, [reportReaderOpen]);

  function handleCopy() {
    if (!report) return;
    const text = serializeReportToText(report, locale, run?.source_confirmation);
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }

  function handlePrint() {
    window.print();
  }

  if (!reportReaderOpen) return null;

  return (
    <div className="flex flex-col flex-1 min-w-0 min-h-0">
      {/* Top bar */}
      <div className="border-b border-border bg-background shrink-0 print:hidden">
        <div className="max-w-3xl mx-auto px-6 py-3 flex items-center gap-3">
          <button
            ref={backButtonRef}
            onClick={closeReportReader}
            className="inline-flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-sm text-muted-foreground hover:text-foreground hover:bg-muted transition-colors focus-visible:ring-2 focus-visible:ring-ring/50"
            aria-label={t("report.back")}
          >
            <ArrowLeft className="size-4" />
            <span className="hidden sm:inline">{t("report.back")}</span>
          </button>

          <div className="ml-auto flex items-center gap-1.5">
            {report && (
              <button
                onClick={handleCopy}
                className="inline-flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-sm text-muted-foreground hover:text-foreground hover:bg-muted transition-colors focus-visible:ring-2 focus-visible:ring-ring/50"
                aria-label={t("report.copy")}
              >
                {copied ? (
                  <>
                    <Check className="size-3.5 text-success" />
                    <span className="hidden sm:inline">{t("report.copied")}</span>
                  </>
                ) : (
                  <>
                    <Copy className="size-3.5" />
                    <span className="hidden sm:inline">{t("report.copy")}</span>
                  </>
                )}
              </button>
            )}
            <button
              onClick={handlePrint}
              className="inline-flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-sm text-muted-foreground hover:text-foreground hover:bg-muted transition-colors focus-visible:ring-2 focus-visible:ring-ring/50"
              aria-label={t("report.print")}
            >
              <Printer className="size-3.5" />
              <span className="hidden sm:inline">{t("report.print")}</span>
            </button>
          </div>
        </div>
      </div>

      {/* Scrollable content */}
      <main className="flex-1 overflow-y-auto">
        <ReportBody />
      </main>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Report body — renders the report_json
// ---------------------------------------------------------------------------

function ReportBody() {
  const { t, locale } = useLocale();
  const { report, run } = useWorkspace();

  const isLoading = run?.status === "completed" && run?.final_report?.available && !report;

  if (isLoading) return <ReportSkeleton />;

  if (!report) {
    return (
      <div className="flex items-center justify-center h-32">
        <p className="text-sm text-muted-foreground">{t("report.no_report")}</p>
      </div>
    );
  }

  const json = report.report_json;
  const metadata = getReportMetadata(json);
  const methodScope = getMethodAndScope(json);
  const overall = getOverallAssessment(json);
  const executive = getExecutiveSummary(json);
  const findings = getGroupedFindings(json);
  const weakSignals = getWeakSignals(json);
  const reviewedCandidates = getReviewedCandidates(json);
  const allLimitations = getLimitations(json);
  const scopeLimitations = allLimitations.filter(
    (l) => l.limitation_type === "scope" && /\s/.test(l.description),
  );
  const coverageLimitations = getCoverageLimitations(json);

  const nonPromotedCandidates = reviewedCandidates.filter(
    (c) => c.status === "retained_for_audit_only" || c.status === "retained_as_data_gap",
  );
  const insufficientItems = nonPromotedCandidates.filter(
    (c) => c.support_level === "insufficient_evidence",
  );
  const notSupportedItems = nonPromotedCandidates.filter(
    (c) => c.support_level !== "insufficient_evidence",
  );

  const isInsufficientOverall = overall?.overall_review_status === "insufficient_evidence_for_overall_assessment";

  // Collect unique risk categories from findings + weak signals for review steps
  const reportRiskCategories = [
    ...new Set([
      ...findings.map((f) => f.risk_category),
      ...weakSignals.map((w) => w.risk_category),
    ].filter(Boolean)),
  ];

  return (
    <div className="max-w-3xl mx-auto px-6 py-10">
      {/* Report title + triage banner */}
      <div className="space-y-4 mb-8">
        <h1 className="text-lg font-bold text-foreground tracking-tight text-center" style={{ textWrap: "balance" }}>
          {t("report.title")}
        </h1>
        <p className="text-xs text-muted-foreground text-center leading-relaxed max-w-lg mx-auto">
          {t("report.triage_banner")}
        </p>
      </div>

      {/* Provenance + Overall: tight group */}
      <div className="space-y-5">
        <ProvenanceHeader
          metadata={metadata}
          methodScope={methodScope}
          generatedAt={report.generated_at}
          sourceConfirmation={run?.source_confirmation}
        />
        {overall && <OverallAssessmentCard overall={overall} />}
      </div>

      {/* Executive summary */}
      {executive.length > 0 && (
        <section className="mt-10 space-y-2.5">
          <h2 className="text-base font-semibold text-foreground">{t("report.executive_summary")}</h2>
          <ul className="space-y-2">
            {executive.map((item, i) => (
              <li
                key={i}
                className="text-sm text-foreground/85 leading-relaxed pl-4 relative before:absolute before:left-0 before:top-[9px] before:size-1.5 before:rounded-full before:bg-muted-foreground/40 break-words"
              >
                {item}
              </li>
            ))}
          </ul>
        </section>
      )}

      {/* Risk signals — major section */}
      {findings.length > 0 && (
        <section className="mt-10 pt-8 border-t border-border space-y-4">
          <div className="flex items-center gap-2">
            <ShieldAlert className="size-4 text-destructive" />
            <h2 className="text-base font-semibold text-foreground">{t("report.risk_signals")}</h2>
          </div>
          {findings.map((f) => (
            <FindingCard key={f.finding_id} finding={f} />
          ))}
        </section>
      )}

      {/* Weak signals */}
      {weakSignals.length > 0 && (
        <section className="mt-8 space-y-4">
          <div className="flex items-center gap-2">
            <AlertTriangle className="size-4 text-warning-foreground" />
            <h2 className="text-base font-semibold text-foreground">{t("report.weak_signals")}</h2>
          </div>
          {weakSignals.map((s, i) => (
            <WeakSignalCard key={s.item_id || i} signal={s} />
          ))}
        </section>
      )}

      {/* Review coverage */}
      {nonPromotedCandidates.length > 0 && (
        <div className="mt-8">
          <ReviewCoverageSection
            insufficientItems={insufficientItems}
            notSupportedItems={notSupportedItems}
            totalCount={nonPromotedCandidates.length}
            defaultOpen={isInsufficientOverall}
            isInsufficientOverall={isInsufficientOverall}
          />
        </div>
      )}

      {/* Review steps — deterministic per risk category */}
      {reportRiskCategories.length > 0 && (
        <section className="mt-8 space-y-3">
          <div className="flex items-center gap-2">
            <ListChecks className="size-4 text-muted-foreground" />
            <h2 className="text-base font-semibold text-foreground">{t("report.review_steps_title")}</h2>
          </div>
          <p className="text-xs text-muted-foreground leading-relaxed">
            {t("report.review_steps_note")}
          </p>
          {reportRiskCategories.map((cat) => {
            const steps = REVIEW_STEPS[cat];
            if (!steps) return null;
            const localized = locale === "vi" ? steps.vi : steps.en;
            return (
              <div key={cat} className="space-y-1.5">
                <p className="text-sm font-medium text-foreground">
                  {formatRiskCategory(cat, locale)}
                </p>
                <ul className="space-y-1 pl-4">
                  {localized.map((step, i) => (
                    <li
                      key={i}
                      className="text-sm text-foreground/80 leading-relaxed relative before:absolute before:left-[-12px] before:top-[9px] before:size-1 before:rounded-full before:bg-muted-foreground/40 break-words"
                    >
                      {step}
                    </li>
                  ))}
                </ul>
              </div>
            );
          })}
        </section>
      )}

      {/* Context boundary */}
      <section className="mt-8 rounded-lg border border-border bg-muted/30 p-4">
        <p className="text-xs text-muted-foreground leading-relaxed">
          {t("report.context_boundary")}
        </p>
      </section>

      {/* Methodology & Limitations — secondary group */}
      <div className="mt-10 pt-8 border-t border-border space-y-6">
        {methodScope && (
          <Collapsible title={t("report.method_scope")} icon={FileText}>
            <MethodScopeBlock method={methodScope} />
          </Collapsible>
        )}

        {(coverageLimitations.length > 0 || scopeLimitations.length > 0) && (
          <section className="space-y-2">
            <h2 className="text-base font-semibold text-foreground">{t("report.limitations")}</h2>
            {coverageLimitations.length > 0 && (
              <p className="text-sm text-muted-foreground leading-relaxed">
                {t("report.coverage_unavailable")}{": "}
                {coverageLimitations.map((cl) => cl.label).join(", ")}.
              </p>
            )}
            {scopeLimitations.length > 0 && (
              <ul className="space-y-1">
                {scopeLimitations.map((l) => (
                  <li key={l.limitation_id} className="text-sm text-muted-foreground leading-relaxed break-words">
                    {l.description}
                  </li>
                ))}
              </ul>
            )}
          </section>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Provenance header
// ---------------------------------------------------------------------------

function deriveSourceName(
  sourceConfirmation: SourceConfirmation | null | undefined,
  metadata: ReturnType<typeof getReportMetadata>,
): string {
  if (sourceConfirmation?.slots) {
    const targetSlot = sourceConfirmation.slots.find((s) => s.role === "target");
    const candidate = targetSlot?.candidate;
    if (candidate) {
      const name = candidate.source_name || candidate.source_origin;
      if (name) return name;
    }
  }
  return metadata?.source_origin ?? "";
}

function ProvenanceHeader({
  metadata,
  methodScope,
  generatedAt,
  sourceConfirmation,
}: {
  metadata: ReturnType<typeof getReportMetadata>;
  methodScope: ReturnType<typeof getMethodAndScope>;
  generatedAt: string;
  sourceConfirmation?: SourceConfirmation | null;
}) {
  const { t, locale } = useLocale();

  const companyName = metadata?.company_name ?? "";
  const ticker = metadata?.ticker ?? "";
  const period = metadata?.period ?? "";
  const basis = metadata?.report_basis ?? "";
  const filingStatus = metadata?.filing_status ?? "";
  const sourceOrigin = deriveSourceName(sourceConfirmation, metadata);

  const date = new Date(generatedAt);
  const formattedDate = date.toLocaleDateString("vi-VN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  });

  const isDeterministic = methodScope?.report_generation_mode === "deterministic_template";
  const modelInvoked = methodScope?.report_synthesis_model?.invoked_for_report_generation === true;
  const modelLabel = methodScope?.report_synthesis_model?.label ?? "";

  let synthesisDisplay = "";
  if (isDeterministic) {
    synthesisDisplay = t("report.deterministic_template");
  } else if (modelInvoked && modelLabel) {
    synthesisDisplay = modelLabel;
  }

  return (
    <div className="text-center space-y-1.5 pb-2">
      <h1 className="text-xl font-bold text-foreground tracking-tight break-words" style={{ textWrap: "balance" }}>
        {companyName}
        {ticker && <span className="text-muted-foreground font-normal ml-1.5">({ticker})</span>}
      </h1>
      <div className="flex flex-wrap items-center justify-center gap-x-2 gap-y-1 text-sm text-muted-foreground">
        {period && <span>{period}</span>}
        {basis && (
          <>
            <span aria-hidden>&middot;</span>
            <span>{formatEnum(basis, locale)}</span>
          </>
        )}
        {filingStatus && (
          <>
            <span aria-hidden>&middot;</span>
            <span>{formatEnum(filingStatus, locale)}</span>
          </>
        )}
      </div>
      <p className="text-xs text-muted-foreground">
        {t("report.generated")}: {formattedDate}
      </p>
      {(sourceOrigin || synthesisDisplay) && (
        <p className="text-xs text-muted-foreground">
          {t("report.source_label")}:{" "}
          {[sourceOrigin, synthesisDisplay].filter(Boolean).join(" · ")}
        </p>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Overall assessment card
// ---------------------------------------------------------------------------

function overallStatusSurface(status: string): { card: string; icon: React.ReactNode } {
  switch (status) {
    case "risk_signals_identified":
      return {
        card: "border-destructive/20 bg-destructive/5",
        icon: <ShieldAlert className="size-4 text-destructive shrink-0" />,
      };
    case "weak_or_limited_risk_signals_only":
      return {
        card: "border-warning-color/25 bg-warning-color/5",
        icon: <AlertTriangle className="size-4 text-warning-foreground shrink-0" />,
      };
    case "no_material_irregularity_signal_identified":
      return {
        card: "border-success/20 bg-success/5",
        icon: <ShieldCheck className="size-4 text-success shrink-0" />,
      };
    default:
      return {
        card: "border-border bg-card",
        icon: <Info className="size-4 text-muted-foreground shrink-0" />,
      };
  }
}

function OverallAssessmentCard({
  overall,
}: {
  overall: NonNullable<ReturnType<typeof getOverallAssessment>>;
}) {
  const { t, locale } = useLocale();
  const surface = overallStatusSurface(overall.overall_review_status);

  return (
    <div className={`rounded-lg border p-5 space-y-3 ${surface.card}`}>
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          {surface.icon}
          <h2 className="text-base font-bold text-foreground">{t("report.overall_assessment")}</h2>
        </div>
        <SeverityBadge severity={overall.highest_severity} />
      </div>

      <p className="text-sm font-semibold text-foreground">
        {formatReviewStatus(overall.overall_review_status, locale)}
      </p>

      {overall.primary_risk_category && (
        <p className="text-sm text-muted-foreground break-words">
          {formatRiskCategory(overall.primary_risk_category, locale)}
        </p>
      )}

      {overall.secondary_risk_categories.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {overall.secondary_risk_categories.map((cat) => (
            <span
              key={cat}
              className="inline-flex items-center rounded-md bg-muted px-2 py-0.5 text-xs text-muted-foreground"
            >
              {formatRiskCategory(cat, locale)}
            </span>
          ))}
        </div>
      )}

      {overall.human_review_recommended && (
        <div className="flex items-center gap-1.5 pt-0.5">
          <ShieldCheck className="size-3.5 text-primary shrink-0" />
          <span className="text-sm font-medium text-primary">{t("report.human_review")}</span>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Finding card
// ---------------------------------------------------------------------------

function findingSeveritySurface(severity: string): string {
  switch (severity) {
    case "high":
      return "border-destructive/15 bg-card";
    case "medium":
      return "border-warning-color/20 bg-card";
    default:
      return "border-border bg-card";
  }
}

function FindingCard({ finding }: { finding: NormalizedFinding }) {
  const { t, locale } = useLocale();
  const [evidenceExpanded, setEvidenceExpanded] = useState(false);
  const allEvidence = finding.supporting_evidence;
  const filingVisibleCount = allEvidence.filter((r) => r.source_document_id).length;

  return (
    <div className={`rounded-lg border p-5 space-y-3 ${findingSeveritySurface(finding.severity)}`}>
      <div className="flex items-start justify-between gap-3">
        <div className="space-y-0.5 min-w-0 flex-1">
          <p className="text-sm font-semibold text-foreground break-words">
            {formatFindingTitle(finding.finding_title, locale)}
          </p>
          <p className="text-xs text-muted-foreground">
            {formatRiskCategory(finding.risk_category, locale)}
          </p>
        </div>
        <div className="flex items-center gap-1.5 shrink-0">
          <SeverityBadge severity={finding.severity} />
          <SupportBadge level={finding.support_level} />
        </div>
      </div>

      <p className="text-sm text-foreground/85 leading-relaxed break-words">{cleanDisplayText(finding.summary)}</p>

      {finding.why_this_matters && (
        <div className="space-y-1">
          <p className="text-xs font-medium text-foreground/70">{t("report.why_this_matters")}</p>
          <p className="text-sm text-foreground/75 leading-relaxed break-words">{finding.why_this_matters}</p>
        </div>
      )}

      {finding.limitations.length > 0 && (
        <p className="text-sm text-muted-foreground leading-relaxed italic break-words">
          {finding.limitations[0]}
        </p>
      )}

      {/* Contradicting evidence */}
      {finding.contradicting_evidence.length > 0 && (
        <div className="space-y-1.5">
          <p className="text-xs font-medium text-foreground/70">{t("report.contradicting_evidence")}</p>
          <EvidenceRefList refs={finding.contradicting_evidence} findingTitle={formatFindingTitle(finding.finding_title, locale)} />
        </div>
      )}

      {/* Evidence toggle */}
      {allEvidence.length > 0 && (
        <>
          <button
            onClick={() => setEvidenceExpanded(!evidenceExpanded)}
            className="flex items-center gap-1.5 rounded-md text-sm text-muted-foreground hover:text-foreground transition-colors focus-visible:ring-2 focus-visible:ring-ring/50"
            aria-expanded={evidenceExpanded}
          >
            <ChevronRight
              className={`size-3.5 transition-transform duration-200 ${evidenceExpanded ? "rotate-90" : ""}`}
              style={{ transitionTimingFunction: "var(--ease-out-expo)" }}
            />
            {t("report.evidence")} ({allEvidence.length})
            {filingVisibleCount > 0 && <Eye className="size-3.5 text-primary ml-0.5" />}
          </button>
          <div className="collapse-grid" data-open={evidenceExpanded}>
            <div>
              <EvidenceRefList refs={allEvidence} findingTitle={formatFindingTitle(finding.finding_title, locale)} />
            </div>
          </div>
        </>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Weak signal card
// ---------------------------------------------------------------------------

function WeakSignalCard({ signal }: { signal: NormalizedWeakSignal }) {
  const { t, locale } = useLocale();
  const [expanded, setExpanded] = useState(false);
  const allEvidence = signal.available_evidence;
  const filingVisibleCount = allEvidence.filter((r) => r.source_document_id).length;

  return (
    <div className="rounded-lg border border-border/60 bg-card/50 p-5 space-y-2.5">
      <div className="flex items-start justify-between gap-3">
        <p className="text-sm text-foreground/85 break-words min-w-0 flex-1">
          {formatRiskCategory(signal.risk_category, locale)}
        </p>
        <div className="flex items-center gap-1.5 shrink-0">
          <SeverityBadge severity={signal.severity} />
          <SupportBadge level={signal.support_level} />
        </div>
      </div>
      <p className="text-sm text-foreground/80 leading-relaxed break-words">{cleanDisplayText(signal.summary)}</p>

      {allEvidence.length > 0 && (
        <>
          <button
            onClick={() => setExpanded(!expanded)}
            className="flex items-center gap-1.5 rounded-md text-sm text-muted-foreground hover:text-foreground transition-colors focus-visible:ring-2 focus-visible:ring-ring/50"
            aria-expanded={expanded}
          >
            <ChevronRight
              className={`size-3.5 transition-transform duration-200 ${expanded ? "rotate-90" : ""}`}
              style={{ transitionTimingFunction: "var(--ease-out-expo)" }}
            />
            {t("report.evidence")} ({allEvidence.length})
            {filingVisibleCount > 0 && <Eye className="size-3.5 text-primary ml-0.5" />}
          </button>
          <div className="collapse-grid" data-open={expanded}>
            <div>
              <EvidenceRefList refs={allEvidence} findingTitle={formatRiskCategory(signal.risk_category, locale)} />
            </div>
          </div>
        </>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Evidence ref list — renders each evidence item with page-level nav
// ---------------------------------------------------------------------------

function EvidenceRefList({ refs, findingTitle }: { refs: NormalizedEvidence[]; findingTitle?: string }) {
  const { t, locale } = useLocale();
  const { openEvidence, selectedEvidence } = useWorkspace();

  return (
    <div className="space-y-2 pt-0.5">
      {refs.map((ref, i) => {
        const hasSource = !!ref.source_document_id;
        const hasPage = ref.page_number != null;
        const isActive =
          selectedEvidence?.evidenceRef &&
          ref.evidence_ref &&
          selectedEvidence.evidenceRef.ref_id === ref.evidence_ref;

        const backendRef = toBackendEvidenceRef(ref);

        return (
          <div key={i} className="flex items-start gap-2 text-xs text-muted-foreground">
            <span className="shrink-0 rounded bg-muted px-1.5 py-0.5 font-mono text-xs">
              {formatEvidenceType(ref.evidence_type, locale)}
            </span>
            <span className={`break-all leading-relaxed flex-1 ${ref.summary ? "" : "font-mono text-muted-foreground/70"}`}>
              {ref.summary || ref.evidence_ref}
            </span>
            {hasSource && hasPage ? (
              <button
                onClick={() => openEvidence(backendRef, findingTitle ? { findingTitle } : undefined)}
                className={`shrink-0 inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-xs font-medium transition-colors focus-visible:ring-2 focus-visible:ring-ring/50 ${
                  isActive
                    ? "bg-primary text-primary-foreground"
                    : "bg-primary/10 text-primary hover:bg-primary/20"
                }`}
                aria-label={`${t("evidence.page")} ${ref.page_number}`}
              >
                {isActive && <span className="size-1.5 rounded-full bg-current" />}
                {t("evidence.page_abbrev")}{ref.page_number} &rarr;
              </button>
            ) : hasSource ? (
              <button
                onClick={() => openEvidence(backendRef, findingTitle ? { findingTitle } : undefined)}
                className="shrink-0 inline-flex items-center gap-1 rounded-md bg-muted px-2 py-0.5 text-xs font-medium text-muted-foreground hover:bg-muted/80 hover:text-foreground transition-colors focus-visible:ring-2 focus-visible:ring-ring/50"
                aria-label={t("evidence.open_document")}
                title={t("evidence.open_document")}
              >
                <Eye className="size-3" />
                PDF
              </button>
            ) : (
              <span className="shrink-0 size-3 flex items-center justify-center text-muted-foreground/40" aria-hidden><span className="block w-2.5 h-px bg-current" /></span>
            )}
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Review coverage section
// ---------------------------------------------------------------------------

function ReviewCoverageSection({
  insufficientItems,
  notSupportedItems,
  totalCount,
  defaultOpen,
  isInsufficientOverall,
}: {
  insufficientItems: ReviewedCandidate[];
  notSupportedItems: ReviewedCandidate[];
  totalCount: number;
  defaultOpen: boolean;
  isInsufficientOverall: boolean;
}) {
  const { t, locale } = useLocale();

  const segments: string[] = [];
  segments.push(`${totalCount} ${t("review.candidates_label")}`);
  if (notSupportedItems.length > 0) {
    segments.push(`${notSupportedItems.length} ${t("review.not_supported")}`);
  }
  if (insufficientItems.length > 0) {
    segments.push(`${insufficientItems.length} ${t("review.insufficient_evidence")}`);
  }
  const countDetail = segments.join(" · ");

  return (
    <Collapsible title={t("review.title")} icon={ListChecks} defaultOpen={defaultOpen} countDetail={countDetail}>
      <p className="text-sm text-muted-foreground mb-3">
        {t("review.explanation")}
        {isInsufficientOverall && (
          <span className="block mt-1">{t("review.insufficient_evidence_expanded")}</span>
        )}
      </p>

      {insufficientItems.length > 0 && (
        <div className="mb-3">
          <p className="text-xs font-semibold text-foreground/70 mb-1.5">{t("review.insufficient_evidence")}</p>
          <div className="space-y-1.5">
            {insufficientItems.map((c) => (
              <div key={c.candidate_id} className="text-sm text-muted-foreground leading-relaxed">
                <span className="font-medium text-foreground/80">
                  {formatRiskCategory(c.risk_category, locale)}
                </span>
                {c.rationale_short && <span className="ml-1"> · {c.rationale_short}</span>}
                {c.suggested_human_review && (
                  <span className="block text-xs text-primary mt-0.5">{c.suggested_human_review}</span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {notSupportedItems.length > 0 && (
        <div>
          <p className="text-xs font-semibold text-foreground/70 mb-1.5">{t("review.not_supported")}</p>
          <div className="space-y-1.5">
            {notSupportedItems.map((c) => (
              <div key={c.candidate_id} className="text-sm text-muted-foreground leading-relaxed">
                <span className="font-medium text-foreground/80">
                  {formatRiskCategory(c.risk_category, locale)}
                </span>
                {c.rationale_short && <span className="ml-1"> · {c.rationale_short}</span>}
              </div>
            ))}
          </div>
        </div>
      )}
    </Collapsible>
  );
}

// ---------------------------------------------------------------------------
// Method & Scope
// ---------------------------------------------------------------------------

function MethodScopeBlock({
  method,
}: {
  method: NonNullable<ReturnType<typeof getMethodAndScope>>;
}) {
  const { t } = useLocale();

  return (
    <div className="space-y-3">
      {method.method_summary && (
        <div className="space-y-1">
          <p className="text-xs font-medium text-foreground/80">{t("report.input_scope")}</p>
          <p className="text-sm text-muted-foreground leading-relaxed break-words">{method.method_summary}</p>
        </div>
      )}
      {method.evidence_scope && (
        <div className="space-y-1">
          <p className="text-xs font-medium text-foreground/80">{t("report.evidence_scope")}</p>
          <p className="text-sm text-muted-foreground leading-relaxed break-words">{method.evidence_scope}</p>
        </div>
      )}
      {method.excluded_scope && method.excluded_scope.length > 0 && (
        <div className="space-y-1">
          <p className="text-xs font-medium text-foreground/80">{t("report.excluded_scope")}</p>
          <ul className="space-y-0.5">
            {method.excluded_scope.map((s, i) => (
              <li key={i} className="text-sm text-muted-foreground leading-relaxed break-words">{s}</li>
            ))}
          </ul>
        </div>
      )}
      {method.reporting_rule && (
        <div className="space-y-1">
          <p className="text-xs font-medium text-foreground/80">{t("report.reporting_rule")}</p>
          <p className="text-sm text-muted-foreground leading-relaxed break-words">{method.reporting_rule}</p>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Legacy export for backwards compat — old ReportPanel is replaced
// ---------------------------------------------------------------------------

export function ReportPanel() {
  return null;
}
