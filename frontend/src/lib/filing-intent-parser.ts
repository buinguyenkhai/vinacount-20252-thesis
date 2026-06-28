import type { ReportBasisPreference } from "@/types/runtime";

export interface ParsedFields {
  company_identifier?: string;
  target_quarter?: 1 | 2 | 3 | 4;
  target_fiscal_year?: number;
  report_basis_preference?: ReportBasisPreference;
}

export type ParseResult =
  | { ok: true; fields: ParsedFields }
  | { ok: false; message: string };

const QUARTER_PATTERN = /(?:q(?:u[ýy])?\s*)([1-4])/i;
const YEAR_PATTERN = /(?:^|\s|n[aă]m\s*)(20\d{2})(?:\s|$)/i;
const TICKER_PATTERN = /(?:^|\s)([A-Z]{2,5})(?=\s|$)/;
const CONSOLIDATED_PATTERN =
  /(?:h[oợ]p\s*nh[aấ]t|consolidated)/i;
const SEPARATE_PATTERN =
  /(?:ri[eê]ng\s*l[eẻ]|separate)/i;

export function parseFilingIntent(input: string): ParseResult {
  const trimmed = input.trim();
  if (!trimmed) {
    return { ok: false, message: "empty_input" };
  }

  const fields: ParsedFields = {};

  const tickerMatch = trimmed.match(TICKER_PATTERN);
  if (tickerMatch) {
    fields.company_identifier = tickerMatch[1];
  } else {
    const words = trimmed.split(/\s+/);
    const tickerWord = words.find(
      (w) => /^[a-zA-Z]{2,5}$/.test(w) && !/^(analyze|hello|world|the|and|for|ph[aâ]n|t[ií]ch|cho|qu[yý]|n[aă]m|ri[eê]ng|h[oợ]p|nh[aấ]t)$/i.test(w)
    );
    if (tickerWord) {
      fields.company_identifier = tickerWord.toUpperCase();
    }
  }

  const quarterMatch = trimmed.match(QUARTER_PATTERN);
  if (quarterMatch) {
    fields.target_quarter = Number(quarterMatch[1]) as 1 | 2 | 3 | 4;
  }

  const yearMatch = trimmed.match(YEAR_PATTERN);
  if (yearMatch) {
    fields.target_fiscal_year = Number(yearMatch[1]);
  }

  if (CONSOLIDATED_PATTERN.test(trimmed)) {
    fields.report_basis_preference = "consolidated";
  } else if (SEPARATE_PATTERN.test(trimmed)) {
    fields.report_basis_preference = "separate";
  }

  if (!fields.company_identifier && !fields.target_quarter && !fields.target_fiscal_year) {
    return { ok: false, message: "no_fields_extracted" };
  }

  return { ok: true, fields };
}
