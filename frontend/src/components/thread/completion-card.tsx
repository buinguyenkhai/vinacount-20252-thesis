"use client";

import { CheckCircle2, ArrowRight } from "lucide-react";
import type { FilingIntent, FinalReportMeta, AllowedAction } from "@/types/runtime";
import { useLocale } from "@/lib/i18n";
import { useWorkspace } from "@/lib/workspace-store";
import { Button } from "@/components/ui/button";

interface CompletionCardProps {
  intent: FilingIntent;
  finalReport: FinalReportMeta;
  elapsedSeconds: number;
  allowedActions?: AllowedAction[];
}

function formatElapsed(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return s > 0 ? `${m}m ${s}s` : `${m}m`;
}

export function CompletionCard({
  intent,
  elapsedSeconds,
}: CompletionCardProps) {
  const { t } = useLocale();
  const { openReportReader } = useWorkspace();

  return (
    <div className="max-w-md rounded-2xl rounded-bl-md border border-success/30 bg-success/5 p-4 space-y-3">
      <div className="flex items-center gap-2">
        <CheckCircle2 className="size-5 text-success" />
        <p className="text-sm font-semibold text-foreground">
          {t("complete.title")}
        </p>
      </div>
      <p className="text-xs text-muted-foreground break-words">
        {intent.company_name_vi ?? intent.company_identifier}
        {intent.company_name_vi && (
          <span className="ml-1">({intent.company_identifier})</span>
        )}{" "}
        Q{intent.target_quarter} {intent.target_fiscal_year} &middot;{" "}
        {formatElapsed(elapsedSeconds)} {t("complete.elapsed")}
      </p>
      <Button size="sm" onClick={openReportReader}>
        {t("complete.view_report")}
        <ArrowRight className="size-3.5" data-icon="inline-end" />
      </Button>
    </div>
  );
}
