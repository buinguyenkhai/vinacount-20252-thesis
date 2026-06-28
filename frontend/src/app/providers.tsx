"use client";

import type { ReactNode } from "react";
import { FixtureStoreProvider } from "@/lib/fixture-store";
import { LiveRunProvider } from "@/lib/live-run-store";
import { LocaleProvider } from "@/lib/i18n";
import { WorkspaceProvider } from "@/lib/workspace-store";

export function Providers({ children }: { children: ReactNode }) {
  return (
    <FixtureStoreProvider>
      <LiveRunProvider>
        <LocaleProvider>
          <WorkspaceProvider>{children}</WorkspaceProvider>
        </LocaleProvider>
      </LiveRunProvider>
    </FixtureStoreProvider>
  );
}
