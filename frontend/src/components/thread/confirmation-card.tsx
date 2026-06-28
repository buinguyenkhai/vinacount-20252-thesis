"use client";

import { Check } from "lucide-react";
import type { FilingIntent } from "@/types/runtime";
import { useLocale } from "@/lib/i18n";
import { useWorkspace } from "@/lib/workspace-store";
import { useFixtureStore } from "@/lib/fixture-store";
import { Button } from "@/components/ui/button";

interface ConfirmationCardProps {
  intent: FilingIntent;
  confirmed: boolean;
}

export function ConfirmationCard({ intent, confirmed }: ConfirmationCardProps) {
  const { t } = useLocale();
  const { mode } = useWorkspace();
  const fixture = useFixtureStore();

  if (confirmed) {
    return (
      <div className="flex items-center gap-2 text-xs text-muted-foreground px-1">
        <Check className="size-3.5 text-success" />
        <span>{t("confirm.confirmed")}</span>
      </div>
    );
  }

  const basisLabel =
    intent.report_basis_preference === "consolidated"
      ? t("confirm.consolidated")
      : t("confirm.separate");

  function handleBegin() {
    if (mode === "fixture") {
      fixture.next();
    }
    // In live mode, confirmation card at "created" status means the run was
    // just created and will automatically proceed - no action needed from user.
    // The backend starts the run workflow.
  }

  return (
    <div className="max-w-md rounded-2xl rounded-bl-md border border-border bg-card p-4 space-y-3">
      <p className="text-sm text-foreground">{t("confirm.title")}</p>
      <div className="grid grid-cols-[auto_1fr] gap-x-6 gap-y-1 text-sm">
        <span className="text-muted-foreground">{t("confirm.company")}</span>
        <span className="text-foreground">
          {intent.company_name_vi ?? intent.company_identifier}
          {intent.company_name_vi && (
            <span className="ml-1.5 text-muted-foreground">
              ({intent.company_identifier})
            </span>
          )}
        </span>
        <span className="text-muted-foreground">{t("confirm.period")}</span>
        <span className="text-foreground">
          Q{intent.target_quarter} {intent.target_fiscal_year}
        </span>
        <span className="text-muted-foreground">{t("confirm.basis")}</span>
        <span className="text-foreground">{basisLabel}</span>
      </div>
      {mode === "fixture" && (
        <div className="flex items-center gap-2 pt-1">
          <Button size="sm" onClick={handleBegin}>
            {t("confirm.begin")}
          </Button>
          <Button variant="ghost" size="sm" disabled>
            {t("confirm.cancel")}
          </Button>
        </div>
      )}
    </div>
  );
}
