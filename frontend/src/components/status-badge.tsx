"use client";

import type { RunStatus } from "@/types/runtime";
import { useLocale } from "@/lib/i18n";

const STYLES: Record<RunStatus, string> = {
  created: "bg-muted text-muted-foreground",
  discovering_sources: "bg-primary/10 text-primary",
  awaiting_source_confirmation: "bg-warning-color/15 text-warning-foreground",
  analyzing: "bg-primary/10 text-primary",
  failed: "bg-destructive/10 text-destructive",
  completed: "bg-success/15 text-success",
  cancelled: "bg-muted text-muted-foreground",
};

const STATUS_I18N_KEYS: Record<RunStatus, string> = {
  created: "status.created",
  discovering_sources: "status.discovering_sources",
  awaiting_source_confirmation: "status.awaiting_source_confirmation",
  analyzing: "status.analyzing",
  failed: "status.failed",
  completed: "status.completed",
  cancelled: "status.cancelled",
};

export function StatusBadge({ status }: { status: RunStatus }) {
  const { t } = useLocale();
  return (
    <span
      className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${STYLES[status]}`}
    >
      {t(STATUS_I18N_KEYS[status])}
    </span>
  );
}
