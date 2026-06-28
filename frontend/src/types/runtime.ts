export type RunStatus =
  | "created"
  | "discovering_sources"
  | "awaiting_source_confirmation"
  | "analyzing"
  | "failed"
  | "completed"
  | "cancelled";

export const TERMINAL_STATUSES: ReadonlySet<RunStatus> = new Set([
  "failed",
  "completed",
  "cancelled",
]);

export type StageId =
  | "source_discovery"
  | "source_confirmation"
  | "cache_lookup"
  | "extraction"
  | "tool_analysis"
  | "detector_assessment"
  | "aggregation"
  | "report_generation";

export const STAGE_ORDER: readonly StageId[] = [
  "source_discovery",
  "source_confirmation",
  "cache_lookup",
  "extraction",
  "tool_analysis",
  "detector_assessment",
  "aggregation",
  "report_generation",
];

export type StageStatus =
  | "pending"
  | "active"
  | "completed"
  | "failed"
  | "skipped"
  | "cancelled";

export type ReportBasisPreference = "consolidated" | "separate";

export interface FilingIntent {
  company_identifier: string;
  company_name_vi: string | null;
  target_fiscal_year: number;
  target_quarter: 1 | 2 | 3 | 4;
  report_basis_preference: ReportBasisPreference;
}

export interface StageProgress {
  processed: number;
  total: number;
}

export interface StageCounts {
  [key: string]: number;
}

export type WarningSeverity = "info" | "warning" | "limitation";

export interface RuntimeWarning {
  code: string;
  severity: WarningSeverity;
  message: string;
  stage_id: StageId | null;
  source_slot_role: SourceSlotRole | null;
  artifact_refs: ArtifactRef[];
}

export interface Stage {
  stage_id: StageId;
  status: StageStatus;
  started_at: string | null;
  completed_at: string | null;
  summary: string | null;
  progress: StageProgress | null;
  counts: StageCounts | null;
  warnings: RuntimeWarning[];
}

export type ActionName =
  | "confirm_sources"
  | "reject_source"
  | "retry_source_discovery"
  | "select_source_candidate"
  | "stop_run"
  | "resume_run"
  | "open_final_report"
  | "download_developer_audit_bundle";

export interface AllowedAction {
  action: ActionName;
  method: string;
  href: string;
  scope: Record<string, unknown> | null;
}

export type SourceSlotRole = "target" | "prior_year_same_quarter";

export type SourceSlotStatus =
  | "pending_discovery"
  | "ready_for_review"
  | "rejected"
  | "retrying_discovery"
  | "locked"
  | "unavailable";

export type SourceConfirmationStatus =
  | "not_started"
  | "ready_for_review"
  | "partially_rejected"
  | "retrying"
  | "confirmed"
  | "stopped";

export type RejectionReasonCode =
  | "wrong_company"
  | "wrong_period"
  | "wrong_basis"
  | "wrong_filing_status"
  | "wrong_language"
  | "not_full_financial_statement"
  | "source_unreadable"
  | "other";

export interface FirstPageIdentity {
  visible_company_name: string | null;
  visible_period: string | null;
  visible_basis_clue: string | null;
}

export interface ArtifactRef {
  artifact_id: string;
  kind: string;
  sha256?: string;
}

export interface SourceCandidate {
  source_document_id: string;
  company_name_vi: string | null;
  ticker: string | null;
  period_label: string;
  quarter: number;
  fiscal_year: number;
  report_basis: string;
  filing_status: string;
  document_type: string;
  language: string;
  source_origin: string;
  source_name: string;
  source_url: string | null;
  is_searchable_version: boolean;
  file_size_bytes: number | null;
  page_count: number | null;
  visible_filing_label: string | null;
  first_page_identity: FirstPageIdentity | null;
  classification_evidence: string[];
  audit_references: Record<string, unknown> | null;
}

export interface SourceSlotRejection {
  reason_code: RejectionReasonCode;
  message: string;
  comment: string | null;
}

export interface SourceSlot {
  role: SourceSlotRole;
  status: SourceSlotStatus;
  candidate: SourceCandidate | null;
  candidate_documents: SourceCandidate[] | null;
  rejection: SourceSlotRejection | null;
  warnings: RuntimeWarning[];
}

export interface SourceConfirmation {
  status: SourceConfirmationStatus;
  confirmable: boolean;
  hitl_boundary: string;
  slots: SourceSlot[];
  package_warnings: RuntimeWarning[];
}

export interface FinalReportMeta {
  available: boolean;
  report_id: string | null;
  generated_at: string | null;
  format: string | null;
  href: string | null;
}

export type ReportSynthesisModelSelection = "default" | "user_selected";

export interface ReportSynthesisModelConfig {
  id: "deepseek-v4-flash" | "deepseek-v4-pro";
  label: string;
  provider: string;
  selection: ReportSynthesisModelSelection;
}

export interface RuntimeConfig {
  report_synthesis_model: ReportSynthesisModelConfig;
}

export type ErrorCode =
  | "filing_intent_invalid"
  | "source_discovery_unavailable"
  | "source_package_unavailable"
  | "source_identity_mismatch_after_confirmation"
  | "source_artifact_unreachable"
  | "cache_lookup_failed"
  | "extraction_failed"
  | "ocr_config_missing"
  | "ocr_provider_failed"
  | "raw_extraction_invalid"
  | "report_memory_build_failed"
  | "tool_analysis_failed"
  | "detector_timeout"
  | "detector_contract_invalid"
  | "aggregation_failed"
  | "report_synthesis_unavailable"
  | "report_narrative_invalid"
  | "report_claim_validation_failed"
  | "final_report_invalid"
  | "audit_bundle_failed"
  | "internal_error";

export interface RuntimeError {
  code: ErrorCode;
  message: string;
  detail: string | null;
  stage_id: StageId | null;
  recoverable: boolean;
  can_resume: boolean;
  artifact_refs: ArtifactRef[];
}

export interface RuntimeRunView {
  schema_version: string;
  run_id: string;
  created_at: string;
  updated_at: string;
  status: RunStatus;
  recoverable: boolean;
  can_resume: boolean;
  elapsed_seconds: number;
  filing_intent: FilingIntent;
  runtime_config: RuntimeConfig;
  source_confirmation: SourceConfirmation | null;
  stages: Stage[];
  current_stage: StageId | null;
  warnings: RuntimeWarning[];
  allowed_actions: AllowedAction[];
  final_report: FinalReportMeta | null;
  error: RuntimeError | null;
}

export interface FilingIntentValidationError {
  error: {
    code: "filing_intent_invalid";
    message: string;
    field_errors: Record<string, string>;
  };
}

export interface EvidenceRef {
  evidence_ref_type: string;
  ref_id: string;
  role: string;
  source_document_id: string | null;
  source_slot_role: SourceSlotRole | null;
  page_number: number | null;
  source_excerpt: string | null;
  geometry: unknown | null;
}

export type OverallReviewStatus =
  | "risk_signals_identified"
  | "weak_or_limited_risk_signals_only"
  | "no_material_irregularity_signal_identified"
  | "insufficient_evidence_for_overall_assessment";

export type SupportLevel =
  | "supported"
  | "weakly_supported"
  | "not_supported"
  | "insufficient_evidence";

export type Severity = "low" | "medium" | "high" | "unknown" | "none";

export interface ReportEvidenceItem {
  evidence_ref: string;
  evidence_type: string;
  evidence_role: string;
  source_section: string;
  summary: string;
  source_document_id: string | null;
  source_slot_role: SourceSlotRole | null;
  page_number: number | null;
  source_excerpt: string | null;
  geometry: unknown | null;
}

export interface GroupedFinding {
  finding_id: string;
  risk_category: string;
  finding_title: string;
  support_level: string;
  severity: string;
  human_review_recommendation: string;
  summary: string;
  why_this_matters: string;
  supporting_evidence: ReportEvidenceItem[];
  contradicting_evidence: ReportEvidenceItem[];
  missing_evidence: ReportEvidenceItem[];
  related_candidate_ids: string[];
  related_detector_assessment_ids: string[];
  related_tool_result_ids: string[];
  limitations: string[];
}

export interface WeakSignalItem {
  item_id: string;
  risk_category: string;
  support_level: string;
  severity: string;
  summary: string;
  available_evidence: ReportEvidenceItem[];
  missing_or_limited_evidence: ReportEvidenceItem[];
  human_review_recommendation: string;
  related_candidate_ids: string[];
  related_detector_assessment_ids: string[];
}

export type ReviewedCandidateStatus =
  | "retained_for_audit_only"
  | "retained_as_data_gap"
  | "promoted_to_finding"
  | "promoted_to_weak_signal";

export interface ReviewedCandidate {
  candidate_id: string;
  risk_category: string;
  assessment_id: string;
  support_level: string;
  detector_severity: string;
  candidate_priority: string;
  tool_refs: string[];
  evidence_refs: string[];
  status: ReviewedCandidateStatus;
  rationale_short: string;
  suggested_human_review?: string;
}

export interface ReportLimitation {
  limitation_id: string;
  limitation_type: string;
  description: string;
}

export interface CanonicalFinalReport {
  schema_version: string;
  run_id: string;
  report_id: string;
  generated_at: string;
  report_language?: string;
  report_json: Record<string, unknown>;
  report_markdown: string;
  artifact_refs: ArtifactRef[];
}

export interface FixtureMeta {
  scenario: string;
  step: number;
  step_label: string;
  next_step_fixture: string | null;
  transition_sequence: string[];
}
