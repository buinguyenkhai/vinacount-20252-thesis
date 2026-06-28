import {
  FileText,
  Check,
  X,
  RefreshCw,
  AlertTriangle,
  Search,
  Lock,
  ExternalLink,
  Package,
  Info,
  Eye,
} from "lucide-react";
import type {
  SourceSlot,
  SourceSlotRole,
  SourceSlotStatus,
  SourceCandidate,
  RuntimeWarning,
} from "@/types/runtime";
import { useLocale } from "@/lib/i18n";
import { getSourceDocumentPdfUrl } from "@/lib/runtime-api";
import { Button } from "@/components/ui/button";

function SlotStatusIndicator({ status }: { status: SourceSlotStatus }) {
  const { t } = useLocale();
  const label = t(`source.status.${status}`);

  switch (status) {
    case "pending_discovery":
      return (
        <span className="inline-flex items-center gap-1 text-xs text-muted-foreground">
          <Search className="size-3" />
          {label}
        </span>
      );
    case "ready_for_review":
      return (
        <span className="inline-flex items-center gap-1 text-xs text-primary">
          <FileText className="size-3" />
          {label}
        </span>
      );
    case "rejected":
      return (
        <span className="inline-flex items-center gap-1 text-xs text-destructive">
          <X className="size-3" />
          {label}
        </span>
      );
    case "retrying_discovery":
      return (
        <span className="inline-flex items-center gap-1 text-xs text-warning-foreground">
          <RefreshCw className="size-3 animate-spin" />
          {label}
        </span>
      );
    case "locked":
      return (
        <span className="inline-flex items-center gap-1 text-xs text-success">
          <Lock className="size-3" />
          {label}
        </span>
      );
    case "unavailable":
      return (
        <span className="inline-flex items-center gap-1 text-xs text-destructive/70">
          <AlertTriangle className="size-3" />
          {label}
        </span>
      );
  }
}

function SlotWarningBanner({ warning }: { warning: RuntimeWarning }) {
  const isInfo = warning.severity === "info";

  return (
    <div
      className={`flex items-start gap-2 rounded-lg px-3 py-1.5 text-xs ${
        isInfo
          ? "bg-muted text-muted-foreground"
          : "bg-warning-color/10 text-warning-foreground"
      }`}
    >
      {isInfo ? (
        <Info className="size-3 mt-0.5 shrink-0" />
      ) : (
        <AlertTriangle className="size-3 mt-0.5 shrink-0" />
      )}
      <span>{warning.message}</span>
    </div>
  );
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function auditRefString(
  auditRefs: Record<string, unknown> | null,
  key: string,
): string | null {
  const val = auditRefs?.[key];
  return typeof val === "string" ? val : null;
}

function isZipPackage(
  auditRefs: Record<string, unknown> | null,
): boolean {
  return auditRefString(auditRefs, "source_container_type") === "zip";
}

function CandidateDetail({
  candidate,
  zipPackage,
  zipMemberFilename,
  filingStatusLabel,
}: {
  candidate: SourceCandidate;
  zipPackage: boolean;
  zipMemberFilename: string | null;
  filingStatusLabel: string | null;
}) {
  const { t } = useLocale();

  return (
    <div className="space-y-2.5">
      <div className="space-y-1">
        <p className="text-sm font-medium text-foreground break-words">
          {candidate.visible_filing_label ?? candidate.period_label}
        </p>
        {candidate.company_name_vi && (
          <p className="text-xs text-muted-foreground break-words">
            {candidate.company_name_vi}
            {candidate.ticker && (
              <span className="ml-1">({candidate.ticker})</span>
            )}
          </p>
        )}
      </div>

      <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
        <span>{t(`source.basis.${candidate.report_basis}`)}</span>
        {filingStatusLabel && <span>{filingStatusLabel}</span>}
        <span>{candidate.language.toUpperCase()}</span>
        {candidate.page_count != null && candidate.page_count > 0 && (
          <span>
            {candidate.page_count} {t("source.pages")}
          </span>
        )}
        {candidate.file_size_bytes != null && candidate.file_size_bytes > 0 && (
          <span>{formatBytes(candidate.file_size_bytes)}</span>
        )}
        {candidate.is_searchable_version && (
          <span className="inline-flex items-center gap-0.5">
            <Check className="size-3 text-success" />
            {t("source.searchable")}
          </span>
        )}
      </div>

      {candidate.source_name && (
        <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <span>
            {t("source.origin_label")}:{" "}
            {candidate.source_url ? (
              <a
                href={candidate.source_url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary hover:underline focus-visible:ring-2 focus-visible:ring-ring/50 rounded-sm"
              >
                {candidate.source_name}
                <ExternalLink className="size-2.5 inline ml-0.5" />
              </a>
            ) : (
              candidate.source_name
            )}
          </span>
        </div>
      )}

      {zipPackage && (
        <div className="rounded-lg bg-muted px-3 py-2 space-y-0.5">
          <div className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
            <Package className="size-3" />
            {t("source.zip_package_note")}
          </div>
          {zipMemberFilename && (
            <p className="text-xs text-foreground break-all">
              {t("source.zip_member_file")}: {zipMemberFilename}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

function CandidatePicker({
  candidates,
  runId,
  disabled,
  onSelect,
}: {
  candidates: SourceCandidate[];
  runId: string | null;
  disabled: boolean;
  onSelect: (sourceDocumentId: string) => void;
}) {
  const { t } = useLocale();

  return (
    <div className="space-y-2">
      <p className="text-xs text-muted-foreground">
        {t("source.select_candidate")}
      </p>
      <div className="space-y-2">
        {candidates.map((doc) => (
          <div
            key={doc.source_document_id}
            className="rounded-lg border border-border bg-background p-3 space-y-1.5 hover:border-primary/40 transition-colors"
          >
            <p className="text-sm font-medium text-foreground break-words">
              {doc.visible_filing_label ?? doc.period_label}
            </p>
            <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-xs text-muted-foreground">
              <span>{t(`source.basis.${doc.report_basis}`)}</span>
              <span>{doc.language.toUpperCase()}</span>
              {doc.page_count != null && doc.page_count > 0 && (
                <span>{doc.page_count} {t("source.pages")}</span>
              )}
              {doc.file_size_bytes != null && doc.file_size_bytes > 0 && (
                <span>{formatBytes(doc.file_size_bytes)}</span>
              )}
              {doc.is_searchable_version && (
                <span className="inline-flex items-center gap-0.5">
                  <Check className="size-3 text-success" />
                  {t("source.searchable")}
                </span>
              )}
            </div>
            <div className="flex items-center gap-2 pt-1">
              <Button
                size="sm"
                variant="outline"
                onClick={() => onSelect(doc.source_document_id)}
                disabled={disabled}
              >
                <Check className="size-3" data-icon="inline-start" />
                {t("source.pick_candidate")}
              </Button>
              {runId && (
                <a
                  href={getSourceDocumentPdfUrl(runId, doc.source_document_id)}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors rounded-sm focus-visible:ring-2 focus-visible:ring-ring/50"
                >
                  <Eye className="size-3" />
                  {t("source.preview_pdf")}
                </a>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export function SourceSlotCard({
  slot,
  runId = null,
  canReject = false,
  canRetry = false,
  actionPending = false,
  onReject,
  onRetry,
  onSelectCandidate,
}: {
  slot: SourceSlot;
  runId?: string | null;
  canReject?: boolean;
  canRetry?: boolean;
  actionPending?: boolean;
  onReject?: (role: SourceSlotRole) => void;
  onRetry?: (role: SourceSlotRole) => void;
  onSelectCandidate?: (role: SourceSlotRole, sourceDocumentId: string) => void;
}) {
  const { t } = useLocale();
  const { candidate, rejection } = slot;

  const roleLabel =
    slot.role === "target" ? t("source.target") : t("source.prior_year");

  const surfaceClass =
    slot.status === "rejected"
      ? "border-destructive/25 bg-destructive/5"
      : slot.status === "locked"
        ? "border-success/25 bg-success/5"
        : slot.status === "unavailable"
          ? "border-destructive/15 bg-destructive/5"
          : "border-border bg-card";

  const zipPackage = candidate ? isZipPackage(candidate.audit_references) : false;
  const zipMemberFilename = candidate
    ? auditRefString(candidate.audit_references, "selected_package_member_filename")
    : null;

  const filingStatusLabel = candidate
    ? t(`source.filing_status.${candidate.filing_status}`)
    : null;
  const hasFilingStatusI18n =
    filingStatusLabel !== `source.filing_status.${candidate?.filing_status}`;

  return (
    <div className={`rounded-xl border ${surfaceClass} p-4 space-y-3`}>
      <div className="flex items-center justify-between gap-2">
        <span className="text-sm font-semibold truncate">{roleLabel}</span>
        <SlotStatusIndicator status={slot.status} />
      </div>

      {candidate ? (
        <CandidateDetail candidate={candidate} zipPackage={zipPackage} zipMemberFilename={zipMemberFilename} filingStatusLabel={hasFilingStatusI18n ? filingStatusLabel : null} />
      ) : slot.candidate_documents && slot.candidate_documents.length > 0 ? (
        <CandidatePicker
          candidates={slot.candidate_documents}
          runId={runId}
          disabled={actionPending}
          onSelect={(docId) => onSelectCandidate?.(slot.role, docId)}
        />
      ) : slot.status === "unavailable" ? (
        <div className="py-1 space-y-1">
          <p className="text-xs text-destructive/80 font-medium">
            {t("source.status.unavailable")}
          </p>
          <p className="text-xs text-muted-foreground">
            {t("source.unavailable_detail")}
          </p>
        </div>
      ) : (
        <div className="space-y-1.5">
          <div className="h-4 w-3/4 rounded bg-muted animate-pulse" />
          <div className="h-3 w-1/2 rounded bg-muted animate-pulse" />
          <div className="h-3 w-2/3 rounded bg-muted animate-pulse" />
        </div>
      )}

      {slot.warnings.length > 0 && (
        <div className="space-y-1.5">
          {slot.warnings.map((w, i) => (
            <SlotWarningBanner key={i} warning={w} />
          ))}
        </div>
      )}

      {rejection && (
        <div className="rounded-lg border border-destructive/20 bg-destructive/5 px-3 py-2 space-y-1">
          <p className="text-xs font-medium text-destructive">
            {t("source.rejected_label")}: {t(`rejection.${rejection.reason_code}`)}
          </p>
          <p className="text-xs text-foreground break-words">{rejection.message}</p>
          {rejection.comment && (
            <p className="text-xs text-muted-foreground italic break-words">
              {rejection.comment}
            </p>
          )}
        </div>
      )}

      {(canReject || canRetry) && (
        <div className="flex items-center gap-2 border-t border-border pt-3">
          {canReject && (
            <Button
              size="sm"
              variant="outline"
              onClick={() => onReject?.(slot.role)}
              disabled={actionPending}
            >
              <X className="size-3" data-icon="inline-start" />
              {t("source.reject")}
            </Button>
          )}
          {canRetry && (
            <Button
              size="sm"
              variant="outline"
              onClick={() => onRetry?.(slot.role)}
              disabled={actionPending}
            >
              <RefreshCw className="size-3" data-icon="inline-start" />
              {t("source.retry")}
            </Button>
          )}
        </div>
      )}
    </div>
  );
}
