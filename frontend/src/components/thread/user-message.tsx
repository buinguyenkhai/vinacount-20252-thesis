"use client";

import type { FilingIntent } from "@/types/runtime";
import { useLocale } from "@/lib/i18n";

export function UserMessage({ intent }: { intent: FilingIntent }) {
  const { t } = useLocale();
  const basisLabel =
    intent.report_basis_preference === "consolidated"
      ? t("user.consolidated")
      : t("user.separate");

  return (
    <div className="flex justify-end">
      <div className="max-w-[85%] rounded-2xl rounded-br-md bg-primary px-4 py-2.5 text-sm text-primary-foreground break-words">
        {t("user.analyze")} {intent.company_identifier} Q
        {intent.target_quarter} {intent.target_fiscal_year} {basisLabel}
      </div>
    </div>
  );
}
