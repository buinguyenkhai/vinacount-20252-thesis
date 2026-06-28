"use client";

import { AlertCircle, RotateCcw, Info } from "lucide-react";
import type { RuntimeError, RuntimeWarning, AllowedAction } from "@/types/runtime";
import { useLocale, resolveWarningText } from "@/lib/i18n";
import { useWorkspace } from "@/lib/workspace-store";
import { useLiveRun } from "@/lib/live-run-store";
import { Button } from "@/components/ui/button";

interface FailedCardProps {
  error: RuntimeError;
  allowedActions: AllowedAction[];
  warnings?: RuntimeWarning[];
}

export function FailedCard({ error, allowedActions, warnings = [] }: FailedCardProps) {
  const { t, locale } = useLocale();
  const { mode } = useWorkspace();
  const liveRun = useLiveRun();
  const canResume = allowedActions.some((a) => a.action === "resume_run");

  const localizedMessage = t(`error.${error.code}`);
  const hasLocalized = localizedMessage !== `error.${error.code}`;
  const displayMessage = hasLocalized
    ? localizedMessage
    : locale === "vi" ? t("error.generic") : (error.message || t("error.generic"));

  const stageLabel = error.stage_id
    ? t(`error_stage.${error.stage_id}`)
    : null;
  const showStage = stageLabel && stageLabel !== `error_stage.${error.stage_id}`;

  const contextWarnings =
    error.code === "cache_lookup_failed"
      ? warnings.filter((w) => w.stage_id === "cache_lookup")
      : [];

  function handleResume() {
    if (mode === "live") {
      liveRun.resumeRun();
    }
  }

  return (
    <div className="max-w-md rounded-2xl rounded-bl-md border border-destructive/30 bg-destructive/5 p-4 space-y-3">
      <div className="flex items-center gap-2">
        <AlertCircle className="size-5 text-destructive" />
        <p className="text-sm font-semibold text-foreground">
          {error.recoverable
            ? t("failed.title")
            : t("failed.non_recoverable")}
        </p>
      </div>
      <p className="text-xs text-foreground break-words">{displayMessage}</p>
      {contextWarnings.length > 0 && (
        <div className="space-y-1.5">
          {contextWarnings.map((w, i) => (
            <div
              key={i}
              className="flex items-start gap-2 text-xs text-foreground/80 break-words"
            >
              <Info className="size-3 mt-0.5 shrink-0 text-destructive/60" />
              <span>{resolveWarningText(w.code, w.message, t, w.source_slot_role)}</span>
            </div>
          ))}
        </div>
      )}
      {showStage && (
        <p className="text-xs text-muted-foreground">
          {t("terminal.during")}: {stageLabel}
        </p>
      )}
      {canResume && (
        <Button
          size="sm"
          variant="outline"
          onClick={handleResume}
          disabled={mode === "live" && liveRun.actionPending}
        >
          <RotateCcw className="size-3.5" data-icon="inline-start" />
          {t("failed.resume")}
        </Button>
      )}
    </div>
  );
}
