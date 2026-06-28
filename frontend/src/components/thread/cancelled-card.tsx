"use client";

import { Ban } from "lucide-react";
import { useLocale } from "@/lib/i18n";

export function CancelledCard() {
  const { t } = useLocale();
  return (
    <div className="flex items-center gap-2 px-1 text-xs text-muted-foreground">
      <Ban className="size-3.5" />
      <span>{t("cancelled.title")}</span>
    </div>
  );
}
