"use client";

import { AlertTriangle, Info } from "lucide-react";
import type { RuntimeWarning } from "@/types/runtime";
import { useLocale, resolveWarningText } from "@/lib/i18n";

export function RunWarnings({ warnings }: { warnings: RuntimeWarning[] }) {
  const { t } = useLocale();

  if (warnings.length === 0) return null;

  return (
    <section className="space-y-2">
      {warnings.map((w, i) => {
        const isInfo = w.severity === "info";
        const stageLabel = w.stage_id
          ? t(`error_stage.${w.stage_id}`)
          : null;
        const showStage = stageLabel && stageLabel !== `error_stage.${w.stage_id}`;
        const displayMessage = resolveWarningText(w.code, w.message, t, w.source_slot_role);

        return (
          <div
            key={i}
            className={`flex items-start gap-2 rounded-lg px-3 py-2 text-xs ${
              isInfo
                ? "bg-muted text-muted-foreground"
                : "bg-warning-color/10 text-warning-foreground"
            }`}
          >
            {isInfo ? (
              <Info className="size-3.5 mt-0.5 shrink-0" />
            ) : (
              <AlertTriangle className="size-3.5 mt-0.5 shrink-0" />
            )}
            <div>
              <span>{displayMessage}</span>
              {showStage && (
                <span className="ml-1.5 text-muted-foreground">
                  ({stageLabel})
                </span>
              )}
            </div>
          </div>
        );
      })}
    </section>
  );
}
