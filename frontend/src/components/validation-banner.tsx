"use client";

import { AlertCircle, X } from "lucide-react";
import { useLocale } from "@/lib/i18n";
import { useLiveRun } from "@/lib/live-run-store";
import { useWorkspace } from "@/lib/workspace-store";

export function ValidationBanner() {
  const { t } = useLocale();
  const { mode } = useWorkspace();
  const { validationError, error, clearValidationError, clearRun } = useLiveRun();

  if (mode !== "live" || (!validationError && !error)) return null;

  if (error && !validationError) {
    return (
      <div className="rounded-xl border border-destructive/25 bg-destructive/5 px-4 py-3">
        <div className="flex items-start gap-2">
          <AlertCircle className="size-4 text-destructive shrink-0 mt-0.5" />
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-foreground">
              {t("validation.runtime_error_title")}
            </p>
            <p className="text-xs text-muted-foreground mt-0.5">{error}</p>
          </div>
          <button
            onClick={clearRun}
            className="flex items-center justify-center size-6 rounded-lg hover:bg-destructive/10 transition-colors shrink-0 focus-visible:ring-2 focus-visible:ring-ring/50"
            aria-label={t("validation.dismiss")}
          >
            <X className="size-3.5 text-muted-foreground" />
          </button>
        </div>
      </div>
    );
  }

  if (!validationError) return null;

  const { message, field_errors } = validationError.error;

  return (
    <div className="rounded-xl border border-destructive/25 bg-destructive/5 px-4 py-3 space-y-1.5">
      <div className="flex items-start gap-2">
        <AlertCircle className="size-4 text-destructive shrink-0 mt-0.5" />
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-foreground">
            {t("validation.title")}
          </p>
          {message && (
            <p className="text-xs text-muted-foreground mt-0.5">{message}</p>
          )}
        </div>
        <button
          onClick={clearValidationError}
          className="flex items-center justify-center size-6 rounded-lg hover:bg-destructive/10 transition-colors shrink-0 focus-visible:ring-2 focus-visible:ring-ring/50"
          aria-label={t("validation.dismiss")}
        >
          <X className="size-3.5 text-muted-foreground" />
        </button>
      </div>
      {field_errors && Object.keys(field_errors).length > 0 && (
        <div className="pl-6 space-y-0.5">
          {Object.entries(field_errors).map(([field, msg]) => {
            const fieldLabel = t(`validation.field.${field}`);
            const hasLocalizedLabel = fieldLabel !== `validation.field.${field}`;
            return (
              <p key={field} className="text-xs text-destructive">
                <span className="font-medium">
                  {hasLocalizedLabel ? fieldLabel : field}
                </span>: {msg}
              </p>
            );
          })}
        </div>
      )}
    </div>
  );
}
