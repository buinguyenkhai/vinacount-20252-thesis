import type {
  RuntimeRunView,
  CanonicalFinalReport,
  FilingIntentValidationError,
  ReportBasisPreference,
  RejectionReasonCode,
} from "@/types/runtime";

const API_BASE = process.env.NEXT_PUBLIC_RUNTIME_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(
    public status: number,
    public body: unknown,
  ) {
    super(`API error ${status}`);
    this.name = "ApiError";
  }
}

export function isValidationError(
  body: unknown,
): body is FilingIntentValidationError {
  return (
    typeof body === "object" &&
    body !== null &&
    "error" in body &&
    typeof (body as FilingIntentValidationError).error === "object" &&
    (body as FilingIntentValidationError).error.code === "filing_intent_invalid"
  );
}

async function request<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });

  const body = res.headers.get("content-type")?.includes("application/json")
    ? await res.json()
    : await res.text();

  if (!res.ok) {
    throw new ApiError(res.status, body);
  }

  return body as T;
}

export interface CreateRunRequest {
  company_identifier: string;
  target_fiscal_year: number;
  target_quarter: 1 | 2 | 3 | 4;
  report_basis_preference: ReportBasisPreference;
  report_synthesis_model_id?: string;
  report_language?: "vi" | "en";
}

export function createRun(
  payload: CreateRunRequest,
): Promise<RuntimeRunView> {
  return request<RuntimeRunView>("/runtime-runs", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getRun(runId: string): Promise<RuntimeRunView> {
  return request<RuntimeRunView>(`/runtime-runs/${encodeURIComponent(runId)}`);
}

export interface ConfirmSourcesBody {
  action: "confirm_sources";
}

export interface RejectSourceBody {
  action: "reject_source";
  slot_role: string;
  reason_code: RejectionReasonCode;
  comment?: string;
}

export interface RetrySourceDiscoveryBody {
  action: "retry_source_discovery";
  slot_role: string;
}

export interface SelectSourceCandidateBody {
  action: "select_source_candidate";
  slot_role: string;
  source_document_id: string;
}

export type SourceConfirmationBody =
  | ConfirmSourcesBody
  | RejectSourceBody
  | RetrySourceDiscoveryBody
  | SelectSourceCandidateBody;

export function submitSourceConfirmation(
  runId: string,
  body: SourceConfirmationBody,
): Promise<RuntimeRunView> {
  return request<RuntimeRunView>(
    `/runtime-runs/${encodeURIComponent(runId)}/source-confirmation`,
    { method: "POST", body: JSON.stringify(body) },
  );
}

export function resumeRun(runId: string): Promise<RuntimeRunView> {
  return request<RuntimeRunView>(
    `/runtime-runs/${encodeURIComponent(runId)}/actions/resume`,
    { method: "POST" },
  );
}

export function stopRun(runId: string): Promise<RuntimeRunView> {
  return request<RuntimeRunView>(
    `/runtime-runs/${encodeURIComponent(runId)}/actions/stop`,
    { method: "POST" },
  );
}

export function getReport(
  runId: string,
): Promise<CanonicalFinalReport> {
  return request<CanonicalFinalReport>(
    `/runtime-runs/${encodeURIComponent(runId)}/report`,
  );
}

export function getSourceDocumentPdfUrl(
  runId: string,
  sourceDocumentId: string,
  pageNumber?: number | null,
): string {
  const base = `${API_BASE}/runtime-runs/${encodeURIComponent(runId)}/source-documents/${encodeURIComponent(sourceDocumentId)}/pdf`;
  return pageNumber != null ? `${base}#page=${pageNumber}` : base;
}
