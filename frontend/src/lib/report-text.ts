import type { CanonicalFinalReport, ReviewedCandidate, SourceConfirmation } from "@/types/runtime";
import type { Locale } from "@/lib/i18n";

type ReportJson = CanonicalFinalReport["report_json"];

// ---------------------------------------------------------------------------
// JSON accessors (mirrored from report-panel — kept minimal, no UI concerns)
// ---------------------------------------------------------------------------

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
      label?: string;
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
  } | undefined;
}

function getExecutiveSummary(json: ReportJson): string[] {
  return (json.executive_summary as string[]) ?? [];
}

interface Finding {
  finding_title: string;
  risk_category: string;
  severity: string;
  support_level: string;
  summary: string;
  why_this_matters: string;
  supporting_evidence: Evidence[];
  contradicting_evidence: Evidence[];
  limitations: string[];
}

interface Evidence {
  evidence_type: string;
  summary: string;
  page_number: number | null;
  source_slot_role: string | null;
  source_document_id: string | null;
}

function normalizeEvidence(raw: unknown): Evidence[] {
  if (!Array.isArray(raw)) return [];
  return raw.map((item: Record<string, unknown>) => ({
    evidence_type: ((item.evidence_type ?? item.evidence_ref_type) as string) ?? "",
    summary: (item.summary as string) ?? "",
    page_number: (item.page_number as number | null) ?? null,
    source_slot_role: (item.source_slot_role as string | null) ?? null,
    source_document_id: (item.source_document_id as string | null) ?? null,
  }));
}

function getFindings(json: ReportJson): Finding[] {
  const raw = json.grouped_findings as Record<string, unknown>[] | undefined;
  if (!raw) return [];
  return raw.map((f) => ({
    finding_title: ((f.finding_title ?? f.title) as string) ?? "",
    risk_category: ((f.risk_category ?? f.primary_risk_category) as string) ?? "",
    severity: ((f.severity ?? f.final_severity) as string) ?? "",
    support_level: ((f.support_level ?? (f.support_levels as string[] | undefined)?.[0]) as string) ?? "",
    summary: (f.summary as string) ?? "",
    why_this_matters: (f.why_this_matters as string) ?? "",
    supporting_evidence: normalizeEvidence(f.supporting_evidence ?? f.evidence_refs),
    contradicting_evidence: normalizeEvidence(f.contradicting_evidence),
    limitations: (f.limitations as string[]) ?? [],
  }));
}

interface WeakSignal {
  risk_category: string;
  severity: string;
  support_level: string;
  summary: string;
  available_evidence: Evidence[];
}

function getWeakSignals(json: ReportJson): WeakSignal[] {
  const raw = json.weak_or_limited_signals as Record<string, unknown>[] | undefined;
  if (!raw) return [];
  return raw.map((s) => ({
    risk_category: (s.risk_category as string) ?? "",
    severity: ((s.severity ?? s.final_severity) as string) ?? "",
    support_level: (s.support_level as string) ?? "",
    summary: (s.summary as string) ?? "",
    available_evidence: normalizeEvidence(s.available_evidence ?? s.evidence_refs),
  }));
}

function getReviewedCandidates(json: ReportJson): ReviewedCandidate[] {
  return (json.reviewed_candidate_audit as ReviewedCandidate[]) ?? [];
}

interface CoverageLimitation {
  label: string;
  summary: string;
}

function getCoverageLimitations(json: ReportJson): CoverageLimitation[] {
  const raw = json.coverage_limitations as Record<string, unknown>[] | undefined;
  if (!Array.isArray(raw)) return [];
  return raw.map((r) => ({
    label: (r.label as string) ?? "",
    summary: (r.summary as string) ?? "",
  }));
}

function getLimitations(json: ReportJson): string[] {
  const raw = json.limitations as unknown[];
  if (!raw) return [];
  if (raw.length > 0 && typeof raw[0] === "string") {
    return (raw as string[]).filter((d) => /\s/.test(d));
  }
  return (raw as { description: string }[]).map((l) => l.description).filter((d) => /\s/.test(d));
}

// ---------------------------------------------------------------------------
// Label formatting (locale-aware, no React dependency)
// ---------------------------------------------------------------------------

const LABELS_VI: Record<string, string> = {
  high: "Cao",
  medium: "Trung bình",
  low: "Thấp",
  unknown: "Chưa rõ",
  supported: "Có bằng chứng",
  weakly_supported: "Bằng chứng yếu",
  insufficient_evidence: "Thiếu bằng chứng",
  not_supported: "Chưa được hỗ trợ",
  risk_signals_identified: "Tín hiệu rủi ro được xác định",
  weak_or_limited_risk_signals_only: "Chỉ tín hiệu rủi ro yếu hoặc hạn chế",
  no_material_irregularity_signal_identified: "Không phát hiện tín hiệu bất thường trọng yếu",
  insufficient_evidence_for_overall_assessment: "Thiếu bằng chứng để đánh giá tổng thể",
  consolidated: "Hợp nhất",
  separate: "Riêng lẻ",
};

function label(value: string, locale: Locale): string {
  if (locale === "vi") return LABELS_VI[value] ?? snakeToTitle(value);
  return snakeToTitle(value);
}

function snakeToTitle(s: string): string {
  return s.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

const RISK_CAT_LABELS: Record<string, { en: string; vi: string }> = {
  revenue_income_recognition_risk: { en: "Revenue quality risk", vi: "Rủi ro chất lượng doanh thu" },
  receivables_credit_quality_risk: { en: "Receivables credit quality risk", vi: "Rủi ro chất lượng khoản phải thu" },
  asset_quality_valuation_risk: { en: "Asset valuation risk", vi: "Rủi ro định giá tài sản" },
  inventory_cost_asset_flow_risk: { en: "Inventory cost and flow risk", vi: "Rủi ro luân chuyển hàng tồn kho" },
  expense_liability_understatement_risk: { en: "Expense and liability understatement risk", vi: "Rủi ro ghi nhận thiếu chi phí hoặc nợ phải trả" },
  earnings_cashflow_mismatch: { en: "Earnings and cash-flow mismatch", vi: "Chênh lệch lợi nhuận và dòng tiền" },
  disclosure_inconsistency_or_obfuscation: { en: "Disclosure consistency risk", vi: "Rủi ro nhất quán thuyết minh" },
  related_party_disclosure_risk: { en: "Related-party disclosure risk", vi: "Rủi ro công bố bên liên quan" },
};

function riskCatLabel(cat: string, locale: Locale): string {
  const entry = RISK_CAT_LABELS[cat];
  if (entry) return locale === "vi" ? entry.vi : entry.en;
  return snakeToTitle(cat);
}

// ---------------------------------------------------------------------------
// Heading strings
// ---------------------------------------------------------------------------

const H = {
  en: {
    title: "VINACOUNT FINANCIAL REPORTING RISK-SIGNAL REVIEW",
    overall: "OVERALL ASSESSMENT",
    summary: "SUMMARY",
    riskSignals: "RISK SIGNALS",
    weakSignals: "WEAK SIGNALS",
    reviewCoverage: "REVIEW COVERAGE",
    methodScope: "METHOD & SCOPE",
    limitations: "LIMITATIONS",
    status: "Status",
    severity: "Highest severity",
    primaryRisk: "Primary risk",
    humanReview: "Human review recommended",
    whyMatters: "Why this matters",
    contradicting: "Contradicting evidence",
    evidence: "Evidence",
    inputScope: "Input scope",
    evidenceScope: "Evidence scope",
    excluded: "Excluded",
    reportingRule: "Reporting rule",
    generated: "Generated",
    source: "Source",
    coverageUnavailable: "Coverage unavailable",
    insufficientEvidence: "Insufficient evidence",
    notSupported: "Not supported",
    candidates: "candidates",
    page: "p.",
    targetFiling: "Target",
    priorYear: "Prior year",
  },
  vi: {
    title: "RÀ SOÁT TÍN HIỆU RỦI RO BÁO CÁO TÀI CHÍNH VINACOUNT",
    overall: "ĐÁNH GIÁ TỔNG THỂ",
    summary: "TÓM TẮT",
    riskSignals: "TÍN HIỆU RỦI RO",
    weakSignals: "TÍN HIỆU YẾU",
    reviewCoverage: "PHẠM VI XEM XÉT",
    methodScope: "PHƯƠNG PHÁP & PHẠM VI",
    limitations: "GIỚI HẠN",
    status: "Trạng thái",
    severity: "Mức độ cao nhất",
    primaryRisk: "Rủi ro chính",
    humanReview: "Cần rà soát thủ công",
    whyMatters: "Tại sao điều này quan trọng",
    contradicting: "Bằng chứng trái chiều",
    evidence: "Bằng chứng",
    inputScope: "Phạm vi đầu vào",
    evidenceScope: "Phạm vi bằng chứng",
    excluded: "Loại trừ",
    reportingRule: "Quy tắc báo cáo",
    generated: "Ngày tạo",
    source: "Nguồn",
    coverageUnavailable: "Chưa có phạm vi",
    insufficientEvidence: "Thiếu bằng chứng",
    notSupported: "Không hỗ trợ",
    candidates: "ứng viên",
    page: "tr.",
    targetFiling: "Mục tiêu",
    priorYear: "Năm trước",
  },
} as const;

// ---------------------------------------------------------------------------
// Serializer
// ---------------------------------------------------------------------------

function line(s: string) { return s + "\n"; }
function heading(s: string) { return line(s) + line("=".repeat(s.length)); }
function subheading(s: string) { return "\n" + line(s) + line("-".repeat(s.length)); }
function bullet(s: string) { return line(`  • ${s}`); }
function separator() { return line(""); }

function formatEvidence(ev: Evidence[], locale: Locale, h: (typeof H)[keyof typeof H]): string {
  let out = "";
  for (const e of ev) {
    const parts: string[] = [];
    if (e.summary) {
      parts.push(e.summary);
    }
    if (e.page_number != null) {
      const slot = e.source_slot_role === "target"
        ? h.targetFiling
        : e.source_slot_role === "prior_year_same_quarter"
          ? h.priorYear
          : "";
      parts.push(`(${h.page}${e.page_number}${slot ? ` ${slot}` : ""})`);
    }
    out += bullet(parts.join(" ") || e.evidence_type);
  }
  return out;
}

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

export function serializeReportToText(
  report: CanonicalFinalReport,
  locale: Locale,
  sourceConfirmation?: SourceConfirmation | null,
): string {
  const json = report.report_json;
  const h = locale === "vi" ? H.vi : H.en;
  let out = "";

  // Title block
  out += heading(h.title);
  const metadata = getReportMetadata(json);
  if (metadata) {
    const nameLine = metadata.company_name + (metadata.ticker ? ` (${metadata.ticker})` : "");
    out += line(nameLine);
    const details = [
      metadata.period,
      metadata.report_basis ? label(metadata.report_basis, locale) : "",
      metadata.filing_status ? label(metadata.filing_status, locale) : "",
    ].filter(Boolean);
    if (details.length) out += line(details.join(" · "));
  }

  const date = new Date(report.generated_at);
  const formattedDate = date.toLocaleDateString("vi-VN", {
    year: "numeric", month: "2-digit", day: "2-digit",
  });
  out += line(`${h.generated}: ${formattedDate}`);

  const methodScope = getMethodAndScope(json);
  const sourceOrigin = deriveSourceName(sourceConfirmation, metadata);
  const isDeterministic = methodScope?.report_generation_mode === "deterministic_template";
  const modelLabel = methodScope?.report_synthesis_model?.invoked_for_report_generation
    ? methodScope.report_synthesis_model.label ?? ""
    : "";
  const synthParts = [
    sourceOrigin,
    isDeterministic ? (locale === "vi" ? "Mẫu xác định" : "Deterministic template") : modelLabel,
  ].filter(Boolean);
  if (synthParts.length) out += line(`${h.source}: ${synthParts.join(" · ")}`);

  // Triage disclaimer
  out += separator();
  out += line(locale === "vi"
    ? "Báo cáo này xác định các tín hiệu rủi ro dựa trên bằng chứng để rà soát thủ công. Báo cáo không kết luận gian lận, sai sót, hay bất thường."
    : "This report identifies evidence-backed risk signals for manual review. It does not conclude fraud, misstatement, or irregularity.");

  // Overall assessment
  const overall = getOverallAssessment(json);
  if (overall) {
    out += subheading(h.overall);
    out += line(`${h.status}: ${label(overall.overall_review_status, locale)}`);
    out += line(`${h.severity}: ${label(overall.highest_severity, locale)}`);
    if (overall.primary_risk_category) {
      out += line(`${h.primaryRisk}: ${riskCatLabel(overall.primary_risk_category, locale)}`);
    }
    if (overall.human_review_recommended) {
      out += line(h.humanReview);
    }
  }

  // Executive summary
  const executive = getExecutiveSummary(json);
  if (executive.length > 0) {
    out += subheading(h.summary);
    for (const item of executive) {
      out += bullet(item);
    }
  }

  // Risk signals
  const findings = getFindings(json);
  if (findings.length > 0) {
    out += subheading(h.riskSignals);
    for (const f of findings) {
      out += separator();
      out += line(`[${label(f.severity, locale).toUpperCase()} | ${label(f.support_level, locale)}] ${f.finding_title}`);
      out += line(f.summary);
      if (f.why_this_matters) {
        out += separator();
        out += line(`${h.whyMatters}: ${f.why_this_matters}`);
      }
      if (f.contradicting_evidence.length > 0) {
        out += separator();
        out += line(`${h.contradicting}:`);
        out += formatEvidence(f.contradicting_evidence, locale, h);
      }
      if (f.supporting_evidence.length > 0) {
        out += separator();
        out += line(`${h.evidence}:`);
        out += formatEvidence(f.supporting_evidence, locale, h);
      }
      if (f.limitations.length > 0) {
        out += separator();
        for (const lim of f.limitations) {
          out += bullet(lim);
        }
      }
    }
  }

  // Weak signals
  const weakSignals = getWeakSignals(json);
  if (weakSignals.length > 0) {
    out += subheading(h.weakSignals);
    for (const s of weakSignals) {
      out += separator();
      out += line(`[${label(s.severity, locale).toUpperCase()} | ${label(s.support_level, locale)}] ${riskCatLabel(s.risk_category, locale)}`);
      out += line(s.summary);
      if (s.available_evidence.length > 0) {
        out += separator();
        out += line(`${h.evidence}:`);
        out += formatEvidence(s.available_evidence, locale, h);
      }
    }
  }

  // Review coverage
  const reviewedCandidates = getReviewedCandidates(json);
  const nonPromoted = reviewedCandidates.filter(
    (c) => c.status === "retained_for_audit_only" || c.status === "retained_as_data_gap",
  );
  if (nonPromoted.length > 0) {
    const insufficient = nonPromoted.filter((c) => c.support_level === "insufficient_evidence");
    const notSupported = nonPromoted.filter((c) => c.support_level !== "insufficient_evidence");
    out += subheading(h.reviewCoverage);
    out += line(`${nonPromoted.length} ${h.candidates}`);
    if (insufficient.length > 0) {
      out += separator();
      out += line(`${h.insufficientEvidence}:`);
      for (const c of insufficient) {
        const parts = [riskCatLabel(c.risk_category, locale)];
        if (c.rationale_short) parts.push(c.rationale_short);
        out += bullet(parts.join(" — "));
      }
    }
    if (notSupported.length > 0) {
      out += separator();
      out += line(`${h.notSupported}:`);
      for (const c of notSupported) {
        const parts = [riskCatLabel(c.risk_category, locale)];
        if (c.rationale_short) parts.push(c.rationale_short);
        out += bullet(parts.join(" — "));
      }
    }
  }

  // Method & scope
  if (methodScope) {
    out += subheading(h.methodScope);
    if (methodScope.method_summary) out += line(`${h.inputScope}: ${methodScope.method_summary}`);
    if (methodScope.evidence_scope) out += line(`${h.evidenceScope}: ${methodScope.evidence_scope}`);
    if (methodScope.excluded_scope?.length) out += line(`${h.excluded}: ${methodScope.excluded_scope.join(", ")}`);
    if (methodScope.reporting_rule) out += line(`${h.reportingRule}: ${methodScope.reporting_rule}`);
  }

  // Coverage limitations + scope limitations
  const coverageLims = getCoverageLimitations(json);
  const scopeLims = getLimitations(json);
  if (coverageLims.length > 0 || scopeLims.length > 0) {
    out += subheading(h.limitations);
    if (coverageLims.length > 0) {
      out += line(`${h.coverageUnavailable}: ${coverageLims.map((l) => l.label).join(", ")}`);
    }
    for (const lim of scopeLims) {
      out += bullet(lim);
    }
  }

  return out.trimEnd() + "\n";
}
