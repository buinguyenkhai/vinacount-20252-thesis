"use client";

import { useState, useCallback, useEffect } from "react";
import { useWorkspace } from "@/lib/workspace-store";
import { useLiveRun } from "@/lib/live-run-store";
import { useFixtureStore } from "@/lib/fixture-store";
import { useLocale } from "@/lib/i18n";
import { parseFilingIntent, type ParsedFields } from "@/lib/filing-intent-parser";
import { ThreadContainer } from "./thread/thread-container";
import { ChatInput } from "./chat-input";
import { ReportReader } from "./report-panel";
import { EvidencePane } from "./evidence-pane";
import { FixturePlayer } from "./fixture-player";
import { ValidationBanner } from "./validation-banner";
import { FilingIntentDraft } from "./filing-intent-draft";
import type { ReportBasisPreference } from "@/types/runtime";

type ExampleRun = {
  key: "nkg" | "hap" | "nhc";
  runId: string;
  labels: Record<"en" | "vi", string>;
};

const TRY_EXAMPLE_MODE =
  process.env.NEXT_PUBLIC_TRY_EXAMPLE_MODE ??
  (process.env.NEXT_PUBLIC_SEALED_DEMO_REPLAY === "0" ? "live_draft" : "sealed_replay");

const EXAMPLES: ExampleRun[] = [
  {
    key: "nkg",
    runId: "run_qaqc_nkg_2021_q3_consolidated_report_memory_reuse",
    labels: {
      vi: "NKG Q3 2021 hợp nhất",
      en: "NKG Q3 2021 consolidated",
    },
  },
  {
    key: "hap",
    runId: "run_qaqc_hap_2024_q2_consolidated_report_memory_reuse",
    labels: {
      vi: "HAP Q2 2024 hợp nhất",
      en: "HAP Q2 2024 consolidated",
    },
  },
  {
    key: "nhc",
    runId: "run_qaqc_nhc_2024_q2_separate_report_memory_reuse",
    labels: {
      vi: "NHC Q2 2024 riêng lẻ",
      en: "NHC Q2 2024 separate",
    },
  },
];

const EXAMPLE_LABELS_BY_TEXT = Object.fromEntries(
  EXAMPLES.flatMap((example) =>
    Object.values(example.labels).map((label) => [label, example]),
  ),
);

const LEGACY_EXAMPLE_LABELS: Record<string, string[]> = {
  vi: EXAMPLES.map((example) => example.labels.vi),
  en: EXAMPLES.map((example) => example.labels.en),
};

export function WorkspaceLayout() {
  const { mode, run, modelId, modelUserSelected, reportReaderOpen } = useWorkspace();
  const { t, locale } = useLocale();
  const liveRun = useLiveRun();
  const fixture = useFixtureStore();

  const [draftFields, setDraftFields] = useState<ParsedFields | null>(null);
  const [draftText, setDraftText] = useState("");

  const hasRun =
    run !== null && (run.status !== "created" || run.stages.length > 0);

  const handleDraft = useCallback((fields: ParsedFields, text: string) => {
    setDraftFields(fields);
    setDraftText(text);
  }, []);

  const handleDraftDismiss = useCallback(() => {
    setDraftFields(null);
    setDraftText("");
  }, []);

  useEffect(() => {
    if (hasRun && draftFields) {
      setDraftFields(null);
      setDraftText("");
    }
  }, [hasRun, draftFields]);

  const handleDraftConfirm = useCallback(
    (fields: {
      company_identifier: string;
      target_quarter: 1 | 2 | 3 | 4;
      target_fiscal_year: number;
      report_basis_preference: ReportBasisPreference;
    }) => {
      if (mode === "fixture") {
        setDraftFields(null);
        setDraftText("");
        fixture.resetHappyPath();
        return;
      }

      liveRun.submitFilingIntent({
        ...fields,
        report_language: locale,
        ...(modelUserSelected ? { report_synthesis_model_id: modelId } : {}),
      });
    },
    [mode, fixture, liveRun, modelId, modelUserSelected, locale],
  );

  function handleExample(text: string) {
    if (mode === "fixture") {
      fixture.resetHappyPath();
      return;
    }

    const example = EXAMPLE_LABELS_BY_TEXT[text];
    if (example) {
      if (TRY_EXAMPLE_MODE === "sealed_replay") {
        setDraftFields(null);
        setDraftText("");
        liveRun.loadSealedDemoRun(example.runId);
        return;
      }
      if (TRY_EXAMPLE_MODE === "cached_first_live_confirmation") {
        const result = parseFilingIntent(text);
        if (!result.ok) return;
        const fields = result.fields;
        if (
          !fields.company_identifier ||
          !fields.target_quarter ||
          !fields.target_fiscal_year ||
          !fields.report_basis_preference
        ) {
          return;
        }
        setDraftFields(null);
        setDraftText("");
        liveRun.submitFilingIntent({
          company_identifier: fields.company_identifier,
          target_quarter: fields.target_quarter,
          target_fiscal_year: fields.target_fiscal_year,
          report_basis_preference: fields.report_basis_preference,
          report_language: locale,
          ...(modelUserSelected ? { report_synthesis_model_id: modelId } : {}),
        });
        return;
      }
    }

    const result = parseFilingIntent(text);
    if (!result.ok) return;
    setDraftFields(result.fields);
    setDraftText(text);
  }

  // Report reader replaces the thread view
  if (reportReaderOpen && hasRun) {
    return (
      <div className="flex flex-1 min-h-0">
        <ReportReader />
        <EvidencePane />
      </div>
    );
  }

  // Draft view: user typed something, draft card is shown for confirmation
  if (draftFields && !hasRun) {
    return (
      <div className="flex flex-1 min-h-0">
        <div className="flex flex-col flex-1 min-w-0">
          <div className="flex-1 overflow-y-auto">
            <div className="max-w-3xl mx-auto w-full flex flex-col gap-4 py-6 px-4 sm:px-6">
              <div className="flex justify-end">
                <div className="max-w-[85%] rounded-2xl rounded-br-md bg-primary px-4 py-2.5 text-sm text-primary-foreground break-words">
                  {draftText}
                </div>
              </div>
              <FilingIntentDraft
                initialFields={draftFields}
                onConfirm={handleDraftConfirm}
                onDismiss={handleDraftDismiss}
                disabled={liveRun.phase === "submitting"}
              />
            </div>
          </div>
          <div className="shrink-0 border-t border-border bg-background px-4 sm:px-6 py-3">
            <div className="max-w-2xl mx-auto space-y-2">
              <ValidationBanner />
              {mode === "fixture" && <FixturePlayer />}
              <ChatInput onDraft={handleDraft} hasDraft />
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Thread view or empty state
  return (
    <div className="flex flex-1 min-h-0">
      <div className="flex flex-col flex-1 min-w-0">
        {hasRun ? (
          <>
            <div className="flex-1 overflow-y-auto">
              <ThreadContainer />
            </div>
            <div className="shrink-0 border-t border-border bg-background px-4 sm:px-6 py-3">
              <div className="max-w-2xl mx-auto space-y-2">
                <ValidationBanner />
                {mode === "fixture" && <FixturePlayer />}
                <ChatInput onDraft={handleDraft} />
              </div>
            </div>
          </>
        ) : (
          <div className="flex-1 flex flex-col items-center justify-center px-6">
            <div className="w-full max-w-2xl space-y-6">
              <div className="text-center space-y-2.5">
                <h1 className="text-2xl font-bold text-foreground tracking-tight" style={{ textWrap: "balance" }}>
                  {t("empty.title")}
                </h1>
                <p className="text-sm text-muted-foreground max-w-md mx-auto leading-relaxed">
                  {t("empty.subtitle")}
                </p>
                <p className="text-xs text-muted-foreground max-w-sm mx-auto leading-relaxed">
                  {t("empty.method_note")}
                </p>
              </div>
              <div className="space-y-2">
                <ValidationBanner />
                {mode === "fixture" && <FixturePlayer />}
                <ChatInput onDraft={handleDraft} />
              </div>
              <div className="flex flex-wrap items-center justify-center gap-2">
                <span className="text-xs text-muted-foreground">
                  {t("empty.example_prefix")}
                </span>
                {(LEGACY_EXAMPLE_LABELS[locale] ?? LEGACY_EXAMPLE_LABELS.vi).map((ex) => (
                  <button
                    key={ex}
                    onClick={() => handleExample(ex)}
                    className="rounded-full border border-border bg-card px-3.5 py-1.5 text-xs text-foreground hover:bg-muted transition-colors focus-visible:ring-2 focus-visible:ring-ring/50"
                  >
                    {ex}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
