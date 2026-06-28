"use client";

import { useEffect, useState } from "react";
import { X, FileText, AlertCircle } from "lucide-react";
import { useLocale } from "@/lib/i18n";
import { useWorkspace } from "@/lib/workspace-store";
import { getSourceDocumentPdfUrl } from "@/lib/runtime-api";

export function EvidencePane() {
  const { t } = useLocale();
  const { selectedEvidence } = useWorkspace();

  return (
    <>
      {/* Desktop pane */}
      <aside
        role="complementary"
        aria-label={t("evidence.view_source")}
        className={`hidden lg:flex flex-col border-l border-border bg-background motion-reduce:transition-none overflow-hidden ${
          selectedEvidence ? "w-[50%] opacity-100" : "w-0 opacity-0"
        }`}
        style={{
          willChange: selectedEvidence ? "width, opacity" : "auto",
          transitionProperty: "width, opacity",
          transitionDuration: "250ms",
          transitionTimingFunction: "var(--ease-out-expo)",
        }}
      >
        {selectedEvidence && <PaneContent />}
      </aside>

      {/* Mobile overlay */}
      {selectedEvidence && (
        <div className="lg:hidden fixed inset-0 z-50 flex flex-col bg-background">
          <PaneContent />
        </div>
      )}
    </>
  );
}

function PaneContent() {
  const { t } = useLocale();
  const { selectedEvidence, closeEvidence } = useWorkspace();

  if (!selectedEvidence) return null;

  const { evidenceRef, runId, findingTitle } = selectedEvidence;
  const hasDocument = !!evidenceRef.source_document_id;
  const hasPage = evidenceRef.page_number != null;
  const pdfUrl = hasDocument
    ? getSourceDocumentPdfUrl(
        runId,
        evidenceRef.source_document_id!,
        evidenceRef.page_number,
      )
    : null;

  const slotLabel =
    evidenceRef.source_slot_role === "target"
      ? t("evidence.target")
      : evidenceRef.source_slot_role === "prior_year_same_quarter"
        ? t("evidence.prior_year")
        : null;

  return (
    <div className="flex flex-col h-full">
      <div className="border-b border-border px-4 py-3 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-2 min-w-0">
          <FileText className="size-3.5 text-muted-foreground shrink-0" />
          <span className="text-xs font-medium text-foreground truncate">
            {slotLabel ?? t("evidence.view_source")}
          </span>
          {hasPage && (
            <span className="text-xs text-muted-foreground tabular-nums shrink-0">
              {t("evidence.page")} {evidenceRef.page_number}
            </span>
          )}
        </div>
        {pdfUrl && (
          <a
            href={pdfUrl}
            target="_blank"
            rel="noreferrer"
            className="ml-3 shrink-0 rounded-md border border-border px-2 py-1 text-xs font-medium text-muted-foreground hover:bg-muted hover:text-foreground transition-colors focus-visible:ring-2 focus-visible:ring-ring/50"
          >
            {t("evidence.open_new_tab")}
          </a>
        )}
        <button
          onClick={closeEvidence}
          className="flex items-center justify-center size-8 rounded-lg hover:bg-muted transition-colors shrink-0 focus-visible:ring-2 focus-visible:ring-ring/50"
          aria-label={t("evidence.close")}
        >
          <X className="size-4" />
        </button>
      </div>

      {findingTitle && (
        <div className="border-b border-border px-4 py-2 bg-muted/30 shrink-0">
          <p className="text-xs text-foreground/70 leading-snug truncate">
            {findingTitle}
          </p>
        </div>
      )}

      <div className="flex-1 min-h-0 flex flex-col">
        {hasDocument ? (
          <PdfViewer
            runId={runId}
            sourceDocumentId={evidenceRef.source_document_id!}
            pageNumber={evidenceRef.page_number}
          />
        ) : (
          <NoDocumentFallback />
        )}
      </div>

      {!hasPage && hasDocument && (
        <div className="border-t border-border px-4 py-2.5">
          <p className="text-xs text-muted-foreground leading-relaxed">
            {t("evidence.page_unavailable")}
          </p>
        </div>
      )}

      {evidenceRef.source_excerpt && (
        <div className="border-t border-border px-4 py-3 shrink-0">
          <p className="text-xs font-medium text-foreground/80 mb-1.5">
            {t("evidence.source_excerpt")}
          </p>
          <p className="text-xs text-foreground/85 leading-relaxed font-mono break-words whitespace-pre-wrap">
            {evidenceRef.source_excerpt}
          </p>
        </div>
      )}
    </div>
  );
}

function PdfViewer({
  runId,
  sourceDocumentId,
  pageNumber,
}: {
  runId: string;
  sourceDocumentId: string;
  pageNumber: number | null;
}) {
  const { t } = useLocale();
  const directPdfUrl = getSourceDocumentPdfUrl(runId, sourceDocumentId, pageNumber);
  const fetchUrl = getSourceDocumentPdfUrl(runId, sourceDocumentId);
  const loadKey = `${fetchUrl}#${pageNumber ?? ""}`;
  const [pdfState, setPdfState] = useState<{
    key: string;
    status: "loading" | "ready" | "error";
    viewerUrl: string | null;
  }>({ key: "", status: "loading", viewerUrl: null });

  useEffect(() => {
    const controller = new AbortController();
    let objectUrl: string | null = null;

    async function loadPdf() {
      try {
        const response = await fetch(fetchUrl, { signal: controller.signal });
        const contentType = response.headers.get("content-type") ?? "";
        if (!response.ok || !contentType.includes("application/pdf")) {
          throw new Error("Source document response was not a PDF.");
        }
        const blob = await response.blob();
        objectUrl = URL.createObjectURL(blob);
        if (!controller.signal.aborted) {
          setPdfState({
            key: loadKey,
            status: "ready",
            viewerUrl: pageNumber != null ? `${objectUrl}#page=${pageNumber}` : objectUrl,
          });
        }
      } catch {
        if (!controller.signal.aborted) {
          setPdfState({ key: loadKey, status: "error", viewerUrl: null });
        }
      }
    }

    loadPdf();

    return () => {
      controller.abort();
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
      }
    };
  }, [fetchUrl, loadKey, pageNumber]);

  const status = pdfState.key === loadKey ? pdfState.status : "loading";
  const viewerUrl = pdfState.key === loadKey ? pdfState.viewerUrl : null;

  if (status === "error") {
    return (
      <div className="flex-1 flex flex-col items-center justify-center gap-3 px-6">
        <AlertCircle className="size-8 text-muted-foreground/60" />
        <p className="text-xs text-muted-foreground text-center leading-relaxed">
          {t("evidence.loading_failed")}
        </p>
        <a
          href={directPdfUrl}
          target="_blank"
          rel="noreferrer"
          className="rounded-md border border-border px-3 py-1.5 text-xs font-medium text-foreground hover:bg-muted transition-colors focus-visible:ring-2 focus-visible:ring-ring/50"
        >
          {t("evidence.open_new_tab")}
        </a>
      </div>
    );
  }

  if (status === "loading" || !viewerUrl) {
    return (
      <div className="flex-1 flex flex-col gap-3 p-4">
        <div className="h-4 w-40 rounded bg-muted animate-pulse" />
        <div className="flex-1 rounded-md bg-muted/60 animate-pulse" />
        <p className="text-xs text-muted-foreground">
          {t("evidence.loading")}
        </p>
      </div>
    );
  }

  return (
    <div className="flex-1 min-h-0 flex flex-col">
      <iframe
        key={viewerUrl}
        src={viewerUrl}
        className="flex-1 w-full border-0 bg-white"
        title={`Source document ${sourceDocumentId}`}
        onError={() => setPdfState({ key: loadKey, status: "error", viewerUrl: null })}
      />
      <div className="border-t border-border px-4 py-2">
        <p className="text-[11px] leading-relaxed text-muted-foreground">
          {t("evidence.viewer_fallback")}
        </p>
      </div>
    </div>
  );
}

function NoDocumentFallback() {
  const { t } = useLocale();

  return (
    <div className="flex-1 flex flex-col items-center justify-center gap-3 px-6">
      <FileText className="size-8 text-muted-foreground/60" />
      <p className="text-xs text-muted-foreground text-center leading-relaxed">
        {t("evidence.no_source_document")}
      </p>
    </div>
  );
}
