"use client";

import { useState } from "react";
import { useLocale } from "@/lib/i18n";
import { Button } from "@/components/ui/button";
import type { ParsedFields } from "@/lib/filing-intent-parser";
import type { ReportBasisPreference } from "@/types/runtime";

interface FilingIntentDraftProps {
  initialFields: ParsedFields;
  onConfirm: (fields: {
    company_identifier: string;
    target_quarter: 1 | 2 | 3 | 4;
    target_fiscal_year: number;
    report_basis_preference: ReportBasisPreference;
  }) => void;
  onDismiss: () => void;
  disabled?: boolean;
}

export function FilingIntentDraft({
  initialFields,
  onConfirm,
  onDismiss,
  disabled = false,
}: FilingIntentDraftProps) {
  const { t } = useLocale();
  const [ticker, setTicker] = useState(initialFields.company_identifier ?? "");
  const [quarter, setQuarter] = useState<string>(
    initialFields.target_quarter?.toString() ?? "",
  );
  const [year, setYear] = useState(
    initialFields.target_fiscal_year?.toString() ?? "",
  );
  const [basis, setBasis] = useState<string>(
    initialFields.report_basis_preference ?? "",
  );

  const yearNum = Number(year);
  const isComplete =
    ticker.trim().length >= 2 &&
    ticker.trim().length <= 5 &&
    quarter !== "" &&
    year.length === 4 &&
    yearNum >= 2000 &&
    yearNum <= 2099 &&
    basis !== "";

  function handleConfirm() {
    if (!isComplete || disabled) return;
    onConfirm({
      company_identifier: ticker.trim().toUpperCase(),
      target_quarter: Number(quarter) as 1 | 2 | 3 | 4,
      target_fiscal_year: yearNum,
      report_basis_preference: basis as ReportBasisPreference,
    });
  }

  const fieldClass =
    "w-full rounded-lg border border-input bg-background px-2.5 py-1.5 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring/30 focus:border-ring disabled:opacity-50";

  return (
    <div className="max-w-md rounded-2xl rounded-bl-md border border-border bg-card p-4 space-y-3">
      <p className="text-sm font-semibold text-foreground">
        {t("draft.title")}
      </p>
      <div className="grid grid-cols-2 gap-3">
        <label className="space-y-1">
          <span className="text-xs font-medium text-muted-foreground">
            {t("draft.ticker_label")}
          </span>
          <input
            type="text"
            value={ticker}
            onChange={(e) => setTicker(e.target.value.replace(/[^a-zA-Z]/g, "").slice(0, 5))}
            placeholder={t("draft.ticker_placeholder")}
            disabled={disabled}
            className={fieldClass}
          />
        </label>
        <label className="space-y-1">
          <span className="text-xs font-medium text-muted-foreground">
            {t("draft.quarter_label")}
          </span>
          <select
            value={quarter}
            onChange={(e) => setQuarter(e.target.value)}
            disabled={disabled}
            className={fieldClass}
          >
            <option value="">{t("draft.select")}</option>
            <option value="1">Q1</option>
            <option value="2">Q2</option>
            <option value="3">Q3</option>
            <option value="4">Q4</option>
          </select>
        </label>
        <label className="space-y-1">
          <span className="text-xs font-medium text-muted-foreground">
            {t("draft.year_label")}
          </span>
          <input
            type="text"
            inputMode="numeric"
            value={year}
            onChange={(e) => setYear(e.target.value.replace(/\D/g, "").slice(0, 4))}
            placeholder={t("draft.year_placeholder")}
            disabled={disabled}
            className={fieldClass}
          />
        </label>
        <label className="space-y-1">
          <span className="text-xs font-medium text-muted-foreground">
            {t("draft.basis_label")}
          </span>
          <select
            value={basis}
            onChange={(e) => setBasis(e.target.value)}
            disabled={disabled}
            className={fieldClass}
          >
            <option value="">{t("draft.select")}</option>
            <option value="consolidated">{t("draft.consolidated")}</option>
            <option value="separate">{t("draft.separate")}</option>
          </select>
        </label>
      </div>
      <div className="flex items-center gap-2">
        <Button size="sm" onClick={handleConfirm} disabled={!isComplete || disabled}>
          {t("draft.begin")}
        </Button>
        <Button size="sm" variant="ghost" onClick={onDismiss} disabled={disabled}>
          {t("draft.dismiss")}
        </Button>
      </div>
    </div>
  );
}
