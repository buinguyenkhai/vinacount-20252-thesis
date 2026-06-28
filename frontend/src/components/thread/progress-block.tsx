"use client";

import { useState, useEffect } from "react";
import { Check, Loader2, Circle, Square, Play, AlertTriangle } from "lucide-react";
import type {
  SourceConfirmation,
  AllowedAction,
  RuntimeWarning,
  SourceSlotRole,
  StageId,
} from "@/types/runtime";
import type { PhaseState } from "@/lib/phase-mapping";
import { useLocale, resolveWarningText } from "@/lib/i18n";
import { useWorkspace } from "@/lib/workspace-store";
import { useLiveRun } from "@/lib/live-run-store";
import { useFixtureStore } from "@/lib/fixture-store";
import { SourceSlotCard } from "@/components/source-slot-card";
import { RunWarnings } from "@/components/run-warnings";
import { Button } from "@/components/ui/button";

interface ProgressBlockProps {
  phases: PhaseState[];
  sourceConfirmation: SourceConfirmation | null;
  sourceConfirmationActive: boolean;
  allowedActions: AllowedAction[];
  warnings: RuntimeWarning[];
  elapsedSeconds: number;
}

function PhaseStatusIcon({ status }: { status: string }) {
  switch (status) {
    case "completed":
      return <Check className="size-4 text-success" />;
    case "active":
      return <Loader2 className="size-4 text-primary animate-spin" />;
    case "failed":
      return <Circle className="size-4 text-destructive fill-destructive" />;
    default:
      return <Circle className="size-4 text-muted-foreground/30" />;
  }
}

function formatElapsed(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return s > 0 ? `${m}m ${s}s` : `${m}m`;
}

function progressLabelForStage(
  stageId: StageId | null,
  t: (key: string) => string,
): string {
  if (stageId === "detector_assessment") {
    return t("progress.detector_packets");
  }
  return t("progress.items_processed");
}

export function ProgressBlock({
  phases,
  sourceConfirmation,
  sourceConfirmationActive,
  allowedActions,
  warnings,
  elapsedSeconds,
}: ProgressBlockProps) {
  const { t } = useLocale();
  const { mode, run } = useWorkspace();
  const liveRun = useLiveRun();
  const fixture = useFixtureStore();

  const isActive = phases.some((p) => p.status === "active");
  const [localElapsed, setLocalElapsed] = useState(elapsedSeconds);

  useEffect(() => {
    setLocalElapsed(elapsedSeconds);
  }, [elapsedSeconds]);

  useEffect(() => {
    if (!isActive) return;
    const id = setInterval(() => setLocalElapsed((s) => s + 1), 1000);
    return () => clearInterval(id);
  }, [isActive]);

  const sourceConfirmed =
    sourceConfirmation?.status === "confirmed" ||
    sourceConfirmation?.status === "stopped";

  const hasStopAction = allowedActions.some((a) => a.action === "stop_run");
  const hasResumeAction = allowedActions.some((a) => a.action === "resume_run");
  const rejectSlots = sourceSlotsForAction(allowedActions, "reject_source");
  const retrySlots = sourceSlotsForAction(allowedActions, "retry_source_discovery");

  function handleConfirmSources() {
    if (mode === "live") {
      liveRun.confirmSources();
    } else {
      fixture.next();
    }
  }

  function handleStop() {
    if (mode === "live") {
      liveRun.stopRun();
    }
  }

  function handleResume() {
    if (mode === "live") {
      liveRun.resumeRun();
    }
  }

  function handleRejectSource(role: SourceSlotRole) {
    if (mode === "live") {
      liveRun.rejectSource(role, "other");
    }
  }

  function handleRetrySource(role: SourceSlotRole) {
    if (mode === "live") {
      liveRun.retrySourceDiscovery(role);
    }
  }

  function handleSelectCandidate(role: SourceSlotRole, sourceDocumentId: string) {
    if (mode === "live") {
      liveRun.selectSourceCandidate(role, sourceDocumentId);
    }
  }

  return (
    <div className="max-w-lg rounded-2xl rounded-bl-md border border-border bg-card p-4 space-y-3" aria-live="polite" aria-atomic="false">
      <div className="space-y-2.5">
        {phases.map((phase) => (
          <div key={phase.id} className="flex items-start gap-2.5">
            <div className="mt-0.5">
              <PhaseStatusIcon status={phase.status} />
            </div>
            <div className="min-w-0 flex-1">
              <p
                className={`text-sm ${
                  phase.status === "active"
                    ? "text-foreground font-medium"
                    : phase.status === "completed"
                      ? "text-muted-foreground"
                      : phase.status === "failed"
                        ? "text-destructive"
                        : "text-muted-foreground/70"
                }`}
              >
                {t(`phase.${phase.id}`)}
              </p>
              {phase.status === "active" && (
                <p className="text-xs text-muted-foreground mt-0.5">
                  {t(`activity.${phase.id}`)}
                </p>
              )}
              {phase.status === "active" && phase.progress && (
                <div className="mt-1.5 space-y-1">
                  <div className="h-1.5 w-full rounded-full bg-muted overflow-hidden">
                    <div
                      className="h-full rounded-full bg-primary transition-all duration-300"
                      style={{
                        width: `${(phase.progress.processed / phase.progress.total) * 100}%`,
                      }}
                    />
                  </div>
                  <p className="text-xs text-muted-foreground tabular-nums">
                    {progressLabelForStage(phase.activeStageId, t)}:{" "}
                    {phase.progress.processed}/{phase.progress.total}
                  </p>
                </div>
              )}
              {phase.id === "analyzing" &&
                phase.status === "active" &&
                phase.stageCount > 1 &&
                phase.activeStageOrdinal &&
                phase.activeStageId && (
                  <p className="text-xs text-muted-foreground mt-0.5">
                    {t("progress.analysis_step")}{" "}
                    <span className="tabular-nums">
                      {phase.activeStageOrdinal}/{phase.stageCount}
                    </span>
                    : {t(`stage.${phase.activeStageId}`)}
                  </p>
                )}
            </div>
          </div>
        ))}
      </div>

      {sourceConfirmationActive &&
        sourceConfirmation &&
        sourceConfirmation.slots.length > 0 && (
          <div className="border-t border-border pt-3 space-y-3">
            <div className="grid gap-3">
              {sourceConfirmation.slots.map((slot) => (
                <SourceSlotCard
                  key={slot.role}
                  slot={slot}
                  runId={run?.run_id}
                  canReject={mode === "live" && rejectSlots.has(slot.role)}
                  canRetry={mode === "live" && retrySlots.has(slot.role)}
                  actionPending={liveRun.actionPending}
                  onReject={handleRejectSource}
                  onRetry={handleRetrySource}
                  onSelectCandidate={handleSelectCandidate}
                />
              ))}
            </div>
            {sourceConfirmation.package_warnings.length > 0 && (
              <div className="space-y-1.5">
                {sourceConfirmation.package_warnings.map((w, i) => (
                  <div
                    key={i}
                    className={`flex items-start gap-2 rounded-lg px-3 py-1.5 text-xs ${
                      w.severity === "info"
                        ? "bg-muted text-muted-foreground"
                        : "bg-warning-color/10 text-warning-foreground"
                    }`}
                  >
                    <AlertTriangle className="size-3 mt-0.5 shrink-0" />
                    <span>{resolveWarningText(w.code, w.message, t, w.source_slot_role)}</span>
                  </div>
                ))}
              </div>
            )}
            {sourceConfirmation.confirmable &&
              allowedActions.some((a) => a.action === "confirm_sources") && (
                <div className="flex items-center gap-2">
                  <Button
                    size="sm"
                    onClick={handleConfirmSources}
                    disabled={liveRun.actionPending}
                  >
                    <Check className="size-3.5" data-icon="inline-start" />
                    {t("source.confirm_sources")}
                  </Button>
                </div>
              )}
          </div>
        )}

      {!sourceConfirmationActive && sourceConfirmed && sourceConfirmation && (
        <div className="flex items-center gap-2 text-xs text-muted-foreground border-t border-border pt-2">
          <Check className="size-3.5 text-success" />
          <span>{t("source.confirmed_summary")}</span>
        </div>
      )}

      {(hasStopAction || hasResumeAction) && (
        <div className="flex items-center gap-2 border-t border-border pt-3">
          {hasStopAction && (
            <Button
              size="sm"
              variant="destructive"
              onClick={handleStop}
              disabled={liveRun.actionPending}
            >
              <Square className="size-3" data-icon="inline-start" />
              {t("action.stop")}
            </Button>
          )}
          {hasResumeAction && (
            <Button
              size="sm"
              variant="outline"
              onClick={handleResume}
              disabled={liveRun.actionPending}
            >
              <Play className="size-3" data-icon="inline-start" />
              {t("action.resume")}
            </Button>
          )}
        </div>
      )}

      {warnings.length > 0 && (
        <div className="border-t border-border pt-3">
          <RunWarnings warnings={warnings} />
        </div>
      )}

      {localElapsed > 0 && (
        <p className="text-xs text-muted-foreground tabular-nums">
          {formatElapsed(localElapsed)}
        </p>
      )}
    </div>
  );
}

function sourceSlotsForAction(
  actions: AllowedAction[],
  actionName: "reject_source" | "retry_source_discovery",
): Set<SourceSlotRole> {
  const action = actions.find((item) => item.action === actionName);
  const slots = action?.scope?.source_slots;
  if (!Array.isArray(slots)) {
    return new Set();
  }
  return new Set(
    slots.filter(
      (slot): slot is SourceSlotRole =>
        slot === "target" || slot === "prior_year_same_quarter",
    ),
  );
}
