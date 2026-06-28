"""Bounded Final Report Synthesis Model contract and deterministic merge."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from vinacount.env_loader import load_dotenv_if_available
from vinacount.final_report import validate_final_report


REPORT_NARRATIVE_DRAFT_SCHEMA_VERSION = "report_narrative_draft.v2"
REPORT_SYNTHESIS_PROMPT_VERSION = "report_synthesis_prompt.v3"
REPORT_SYNTHESIS_SCHEMA_VERSION = "report_synthesis_schema.v1"
REPORT_SYNTHESIS_DECODING_VERSION = "deepseek_json_object.temperature_0.local_schema_validation.v1"
DEEPSEEK_REPORT_SYNTHESIS_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"
ALLOWED_REPORT_SYNTHESIS_MODEL_IDS = {"deepseek-v4-flash", "deepseek-v4-pro"}

ReportSynthesisTransport = Callable[..., dict[str, Any]]


class ReportSynthesisAdapter(Protocol):
    def synthesize(
        self,
        *,
        model_id: str,
        request: dict[str, Any],
        response_schema: dict[str, Any],
    ) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class DeepSeekReportSynthesisAdapter:
    api_key: str
    timeout_seconds: float = 60.0
    max_tokens: int = 4096
    transport: ReportSynthesisTransport = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.transport is None:
            object.__setattr__(self, "transport", _post_deepseek_report_synthesis)

    def synthesize(
        self,
        *,
        model_id: str,
        request: dict[str, Any],
        response_schema: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.api_key.strip():
            raise ValueError(f"{DEEPSEEK_API_KEY_ENV} is required for model report synthesis")
        if model_id not in ALLOWED_REPORT_SYNTHESIS_MODEL_IDS:
            raise ValueError("Unknown Final Report Synthesis Model")
        provider_user_payload = {
            "request": request,
            "response_schema": response_schema,
        }
        provider_request = {
            "model": model_id,
            "messages": [
                {
                    "role": "system",
                    "content": _report_synthesis_system_prompt(),
                },
                {
                    "role": "user",
                    "content": json.dumps(provider_user_payload, ensure_ascii=False, sort_keys=True),
                },
            ],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"},
        }
        started = time.monotonic()
        response = self.transport(
            url=DEEPSEEK_REPORT_SYNTHESIS_URL,
            payload=provider_request,
            timeout_seconds=self.timeout_seconds,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        latency_ms = round((time.monotonic() - started) * 1000)
        try:
            message = response["choices"][0]["message"]
            content = message["content"]
            draft = content if isinstance(content, dict) else json.loads(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("report_synthesis_provider_response_invalid") from error
        if not isinstance(draft, dict):
            raise ValueError("report_synthesis_provider_response_invalid")
        return {
            "draft": draft,
            "provider": "deepseek",
            "model_id": response.get("model") or model_id,
            "latency_ms": latency_ms,
            "usage": copy.deepcopy(response.get("usage")),
            "request_body": provider_request,
            "response_body": {
                "id": response.get("id"),
                "model": response.get("model") or model_id,
                "content": draft,
                "finish_reason": response.get("choices", [{}])[0].get("finish_reason"),
                "usage": copy.deepcopy(response.get("usage")),
            },
        }


def make_deepseek_report_synthesis_adapter_from_env() -> DeepSeekReportSynthesisAdapter:
    load_dotenv_if_available()
    return DeepSeekReportSynthesisAdapter(api_key=os.environ.get(DEEPSEEK_API_KEY_ENV, ""))


def build_report_synthesis_request(
    report: dict[str, Any],
    tool_findings: list[dict[str, Any]],
    *,
    report_language: str | None = None,
) -> dict[str, Any]:
    """Build the provider-safe projection of the immutable report decision skeleton."""
    language = _report_language(report_language or report.get("metadata", {}).get("report_language"))
    findings_by_id = {finding["tool_result_id"]: finding for finding in tool_findings}
    slots: list[dict[str, Any]] = []
    claims: list[dict[str, Any]] = []

    for finding in report["grouped_findings"]:
        scope_id = finding["finding_id"]
        slot_id = f"grouped_finding:{scope_id}:summary"
        slot_claims = _claims_for_report_item(
            item=finding,
            finding_scope_id=scope_id,
            slot_id=slot_id,
            tool_findings_by_id=findings_by_id,
        )
        if not slot_claims:
            raise ValueError(f"No filing-lineaged Permitted Report Claim exists for {scope_id}")
        claims.extend(slot_claims)
        slots.append(
            {
                "slot_id": slot_id,
                "finding_scope_id": scope_id,
                "section": "grouped_findings",
                "required_material_claim_ids": [
                    claim["claim_id"]
                    for claim in slot_claims
                    if claim["claim_kind"] == "material_contradicting_evidence"
                ],
            }
        )

    for item in report["weak_or_limited_signals"]:
        scope_id = f"WEAK:{item['assessment_id']}"
        slot_id = f"weak_signal:{item['assessment_id']}:summary"
        slot_claims = _claims_for_report_item(
            item=item,
            finding_scope_id=scope_id,
            slot_id=slot_id,
            tool_findings_by_id=findings_by_id,
        )
        if not slot_claims:
            raise ValueError(f"No filing-lineaged Permitted Report Claim exists for {scope_id}")
        claims.extend(slot_claims)
        slots.append(
            {
                "slot_id": slot_id,
                "finding_scope_id": scope_id,
                "section": "weak_or_limited_signals",
                "required_material_claim_ids": [
                    claim["claim_id"]
                    for claim in slot_claims
                    if claim["claim_kind"] == "material_contradicting_evidence"
                ],
            }
        )

    return {
        "prompt_version": REPORT_SYNTHESIS_PROMPT_VERSION,
        "schema_version": REPORT_SYNTHESIS_SCHEMA_VERSION,
        "decoding_version": REPORT_SYNTHESIS_DECODING_VERSION,
        "report_language": language,
        "narrative_guidance": {
            "purpose": (
                "Write conservative financial-reporting risk-signal narrative for analyst review. "
                "The report identifies evidence-backed review priorities; it does not conclude fraud, "
                "misstatement, intent, or proven irregularity."
            ),
            "grouped_finding_requirements": [
                "Fill summary text from the permitted claims only.",
                "Fill why_this_matters with one short economic-significance explanation.",
                "Use risk signal / manual review wording.",
                "Include absolute figures alongside percentages only when an exact absolute figure appears in a permitted claim proposition.",
                "Do not invent values, peer/sector/macro context, or external news/market facts.",
            ],
            "limitations_to_respect": [
                "No peer, sector, macro, market, news, enforcement, or analyst data unless present in permitted claims.",
                "Unaudited or reviewed interim filing scope is a context boundary, not a standalone risk finding.",
                "Recommendations must remain human-follow-up review language, not conclusions.",
            ],
        },
        "report_decision_skeleton": {
            "report_id": report["report_id"],
            "target_report_id": report["target_report_id"],
            "overall_review_status": report["overall_assessment"]["overall_review_status"],
            "grouped_findings": [
                {
                    "finding_id": finding["finding_id"],
                    "primary_risk_category": finding["primary_risk_category"],
                    "risk_categories": copy.deepcopy(finding["risk_categories"]),
                    "support_levels": copy.deepcopy(finding["support_levels"]),
                    "final_severity": finding["final_severity"],
                }
                for finding in report["grouped_findings"]
            ],
            "weak_signals": [
                {
                    "finding_scope_id": f"WEAK:{item['assessment_id']}",
                    "risk_category": item["risk_category"],
                    "support_level": item["support_level"],
                    "final_severity": item["final_severity"],
                }
                for item in report["weak_or_limited_signals"]
            ],
        },
        "narrative_slots": slots,
        "permitted_claims": claims,
    }


def report_narrative_draft_json_schema(*, report_language: str = "vi") -> dict[str, Any]:
    language = _report_language(report_language)
    return {
        "type": "object",
        "properties": {
            "schema_version": {"const": REPORT_NARRATIVE_DRAFT_SCHEMA_VERSION},
            "language": {"const": language},
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "slot_id": {"type": "string"},
                        "finding_scope_id": {"type": "string"},
                        "text": {"type": "string", "minLength": 1},
                        "claim_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "uniqueItems": True,
                        },
                        "why_this_matters": {"type": "string", "minLength": 1},
                    },
                    "required": ["slot_id", "finding_scope_id", "text", "claim_ids"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["schema_version", "language", "items"],
        "additionalProperties": False,
    }


def merge_validated_report_narrative(
    report: dict[str, Any],
    request: dict[str, Any],
    draft: Any,
    *,
    model_id: str,
    provider: str,
) -> dict[str, Any]:
    items = _validate_draft(request, draft)
    merged = copy.deepcopy(report)
    grouped_by_id = {finding["finding_id"]: finding for finding in merged["grouped_findings"]}
    weak_by_scope = {
        f"WEAK:{item['assessment_id']}": item
        for item in merged["weak_or_limited_signals"]
    }
    for item in items:
        scope_id = item["finding_scope_id"]
        if scope_id in grouped_by_id:
            grouped_by_id[scope_id]["summary"] = item["text"]
            grouped_by_id[scope_id]["why_this_matters"] = item["why_this_matters"]
        else:
            weak_by_scope[scope_id]["summary"] = item["text"]

    merged["executive_summary"] = _vietnamese_executive_summary(merged)
    method = merged["method_and_scope"]
    method["report_assembly"] = (
        "Cấu trúc và các trường kiểm soát do backend xác định; mô hình tổng hợp báo cáo "
        "chỉ soạn phần mô tả tiếng Việt từ các nhận định được phép."
    )
    method["excluded_scope"] = [
        item
        for item in method.get("excluded_scope", [])
        if "selected report synthesis model is recorded" not in item.lower()
    ]
    method["report_synthesis_model"] = {
        "model_id": model_id,
        "provider": provider,
        "invoked_for_report_generation": True,
    }
    validate_final_report(merged)
    return merged


def report_synthesis_output_hash(draft: dict[str, Any]) -> str:
    body = json.dumps(draft, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _claims_for_report_item(
    *,
    item: dict[str, Any],
    finding_scope_id: str,
    slot_id: str,
    tool_findings_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    lineages = _filing_lineages(item.get("evidence_refs", []))
    claims = []
    for index, tool_ref in enumerate(item.get("tool_refs", []), start=1):
        tool_finding = tool_findings_by_id.get(tool_ref)
        if not tool_finding:
            continue
        tool_lineages = _lineages_for_tool_finding(tool_finding, lineages)
        if not tool_lineages:
            continue
        proposition = str(tool_finding.get("finding_summary") or "").strip()
        if not proposition:
            continue
        metric = tool_finding.get("metric") if isinstance(tool_finding.get("metric"), dict) else {}
        claims.append(
            {
                "claim_id": f"CLAIM_{_claim_token(finding_scope_id)}_{index:03d}",
                "finding_scope_id": finding_scope_id,
                "slot_id": slot_id,
                "claim_kind": "supporting",
                "proposition": proposition,
                "allowed_numbers": _dedupe_strings(
                    _number_tokens(proposition)
                    + _number_tokens(json.dumps(metric, ensure_ascii=False, sort_keys=True))
                    + _evidence_ref_number_tokens(item.get("evidence_refs", []))
                ),
                "allowed_periods": [
                    value
                    for value in [metric.get("period_current"), metric.get("period_comparison")]
                    if isinstance(value, str) and value
                ],
                "allowed_categories": copy.deepcopy(item.get("risk_categories") or [item.get("risk_category")]),
                "allowed_severities": [item["final_severity"]],
                "filing_source_lineage": tool_lineages,
            }
        )

    if not claims and lineages:
        proposition = str(item.get("summary") or "").strip()
        if proposition:
            claims.append(
                {
                    "claim_id": f"CLAIM_{_claim_token(finding_scope_id)}_001",
                    "finding_scope_id": finding_scope_id,
                    "slot_id": slot_id,
                    "claim_kind": "supporting",
                    "proposition": proposition,
                    "allowed_numbers": _dedupe_strings(
                        _number_tokens(proposition)
                        + _evidence_ref_number_tokens(item.get("evidence_refs", []))
                    ),
                    "allowed_periods": [],
                    "allowed_categories": copy.deepcopy(item.get("risk_categories") or [item.get("risk_category")]),
                    "allowed_severities": [item["final_severity"]],
                    "filing_source_lineage": lineages,
                }
            )

    contradiction_refs = [
        lineage for lineage in lineages if lineage.get("role") == "contradicting"
    ]
    if contradiction_refs:
        claims.append(
            {
                "claim_id": f"CLAIM_{_claim_token(finding_scope_id)}_CONTRA_001",
                "finding_scope_id": finding_scope_id,
                "slot_id": slot_id,
                "claim_kind": "material_contradicting_evidence",
                "proposition": "Bằng chứng đối nghịch này phải được trình bày như một yếu tố hạn chế.",
                "allowed_numbers": [],
                "allowed_periods": [],
                "allowed_categories": copy.deepcopy(item.get("risk_categories") or [item.get("risk_category")]),
                "allowed_severities": [item["final_severity"]],
                "filing_source_lineage": contradiction_refs,
            }
        )
    return claims


def _filing_lineages(evidence_refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for ref in evidence_refs:
        source_document_id = ref.get("source_document_id")
        source_slot_role = ref.get("source_slot_role")
        if not source_document_id or not source_slot_role:
            continue
        key = (ref.get("evidence_ref_type"), ref.get("ref_id"), source_document_id, source_slot_role, ref.get("role"))
        if key in seen:
            continue
        seen.add(key)
        result.append(
            {
                "evidence_ref_type": ref.get("evidence_ref_type"),
                "ref_id": ref.get("ref_id"),
                "role": ref.get("role"),
                "source_document_id": source_document_id,
                "source_slot_role": source_slot_role,
            }
        )
    return result


def _lineages_for_tool_finding(
    tool_finding: dict[str, Any],
    available_lineages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    keys = {
        (ref.get("evidence_ref_type"), ref.get("ref_id"))
        for ref in tool_finding.get("evidence_refs", [])
    }
    return [
        copy.deepcopy(lineage)
        for lineage in available_lineages
        if (lineage.get("evidence_ref_type"), lineage.get("ref_id")) in keys
    ]


def _validate_draft(request: dict[str, Any], draft: Any) -> list[dict[str, Any]]:
    if not isinstance(draft, dict) or set(draft) != {"schema_version", "language", "items"}:
        raise ValueError("Report Narrative Draft fields do not match the schema")
    language = _report_language(request.get("report_language"))
    if draft["schema_version"] != REPORT_NARRATIVE_DRAFT_SCHEMA_VERSION or draft["language"] != language:
        raise ValueError("Report Narrative Draft must match the requested report language")
    if not isinstance(draft["items"], list):
        raise ValueError("Report Narrative Draft items must be an array")

    slots = {slot["slot_id"]: slot for slot in request["narrative_slots"]}
    claims = {claim["claim_id"]: claim for claim in request["permitted_claims"]}
    if {item.get("slot_id") for item in draft["items"] if isinstance(item, dict)} != set(slots):
        raise ValueError("Report Narrative Draft must fill every backend-owned narrative slot exactly once")
    if len(draft["items"]) != len(slots):
        raise ValueError("Report Narrative Draft contains duplicate or additional narrative items")

    for item in draft["items"]:
        required_item_fields = {"slot_id", "finding_scope_id", "text", "claim_ids"}
        optional_item_fields = {"why_this_matters"}
        if (
            not isinstance(item, dict)
            or not required_item_fields <= set(item)
            or set(item) - required_item_fields - optional_item_fields
        ):
            raise ValueError("Report Narrative Draft item fields do not match the schema")
        slot = slots[item["slot_id"]]
        if item["finding_scope_id"] != slot["finding_scope_id"]:
            raise ValueError("Report Narrative Draft cannot add, remove, or regroup findings")
        text = item["text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Report Narrative Draft narrative must be non-empty text")
        if language == "vi" and not _looks_vietnamese(text):
            raise ValueError("Report Narrative Draft narrative must be non-empty Vietnamese text")
        claim_ids = item["claim_ids"]
        if not isinstance(claim_ids, list) or not claim_ids or len(claim_ids) != len(set(claim_ids)):
            raise ValueError("Every narrative item must cite one or more unique Permitted Report Claims")
        cited_claims = []
        for claim_id in claim_ids:
            claim = claims.get(claim_id)
            if claim is None:
                raise ValueError("Report Narrative Draft cites an unknown Permitted Report Claim")
            if claim["finding_scope_id"] != item["finding_scope_id"]:
                raise ValueError("Report Narrative Draft cites a cross-finding Permitted Report Claim")
            cited_claims.append(claim)
        if not set(slot["required_material_claim_ids"]) <= set(claim_ids):
            raise ValueError("Report Narrative Draft omits material contradicting evidence")
        _validate_narrative_facts(text, cited_claims)
        why_this_matters = item.get("why_this_matters")
        if slot["section"] == "grouped_findings":
            if not isinstance(why_this_matters, str) or not why_this_matters.strip():
                raise ValueError("Report Narrative Draft must fill why_this_matters for grouped findings")
            if language == "vi" and not _looks_vietnamese(why_this_matters):
                raise ValueError("Report Narrative Draft why_this_matters must be non-empty Vietnamese text")
            _validate_narrative_facts(why_this_matters, cited_claims)
        elif why_this_matters is not None and (
            not isinstance(why_this_matters, str) or not why_this_matters.strip()
        ):
            raise ValueError("Report Narrative Draft why_this_matters must be non-empty text when provided")
    return copy.deepcopy(draft["items"])


def _validate_narrative_facts(text: str, claims: list[dict[str, Any]]) -> None:
    normalized = text.lower()
    prohibited = [
        "fraud",
        "manipulat",
        "conceal",
        "intentional",
        "illegal",
        "legal misstatement",
        "gian lận",
        "lừa đảo",
        "thao túng",
        "che giấu",
        "cố ý",
        "bất hợp pháp",
        "vi phạm pháp luật",
        "chắc chắn",
        "chứng minh",
        "gây ra",
        "nguyên nhân",
    ]
    if any(term in normalized for term in prohibited):
        raise ValueError("Report Narrative Draft contains unsupported accusation, certainty, or causation language")

    allowed_numbers = {number for claim in claims for number in claim["allowed_numbers"]}
    for number in _number_tokens(text):
        if number not in allowed_numbers:
            raise ValueError("Report Narrative Draft contains an unmatched number")
    allowed_periods = {period.lower() for claim in claims for period in claim["allowed_periods"]}
    for period in re.findall(r"\b(?:19|20)\d{2}[- /]?q[1-4]\b", normalized, flags=re.IGNORECASE):
        if period.lower().replace(" ", "") not in {value.replace(" ", "") for value in allowed_periods}:
            raise ValueError("Report Narrative Draft contains an unmatched period")
    allowed_severities = {severity for claim in claims for severity in claim["allowed_severities"]}
    severity_terms = {
        "high": ["high", "cao"],
        "medium": ["medium", "trung bình"],
        "low": ["low", "thấp"],
        "unknown": ["unknown", "không xác định"],
    }
    mentioned_severities = {
        severity
        for severity, terms in severity_terms.items()
        if any(re.search(rf"(?<!\w){re.escape(term)}(?!\w)", normalized) for term in terms)
    }
    if not mentioned_severities <= allowed_severities:
        raise ValueError("Report Narrative Draft contains an unmatched severity")
    allowed_categories = {category for claim in claims for category in claim["allowed_categories"] if category}
    category_terms = {
        "revenue_income_recognition_risk": [
            "revenue_income_recognition_risk",
            "ghi nhận doanh thu",
            "chất lượng doanh thu",
        ],
        "receivables_credit_quality_risk": [
            "receivables_credit_quality_risk",
            "chất lượng tín dụng",
            "rủi ro khoản phải thu",
        ],
        "asset_quality_valuation_risk": [
            "asset_quality_valuation_risk",
            "định giá tài sản",
        ],
        "inventory_cost_asset_flow_risk": [
            "inventory_cost_asset_flow_risk",
            "luân chuyển hàng tồn kho",
        ],
        "expense_liability_understatement_risk": [
            "expense_liability_understatement_risk",
            "ghi nhận thiếu chi phí",
            "ghi nhận thiếu nợ phải trả",
        ],
        "earnings_cashflow_mismatch": [
            "earnings_cashflow_mismatch",
            "chênh lệch lợi nhuận và dòng tiền",
        ],
        "disclosure_inconsistency_or_obfuscation": [
            "disclosure_inconsistency_or_obfuscation",
            "không nhất quán trong thuyết minh",
        ],
        "related_party_disclosure_risk": [
            "related_party_disclosure_risk",
            "bên liên quan",
            "công bố bên liên quan",
        ],
    }
    mentioned_categories = {
        category
        for category, terms in category_terms.items()
        if any(term in normalized for term in terms)
    }
    if not mentioned_categories <= allowed_categories:
        raise ValueError("Report Narrative Draft contains an unmatched category")


def _number_tokens(text: str) -> list[str]:
    tokens = []
    for token in re.findall(r"(?<![\w.,])-?\d+(?:[.,]\d+)*(?!\w)(?![.,]\d)", text):
        canonical = _canonical_number_token(token)
        tokens.append(canonical)
        if canonical.startswith("-") and len(canonical) > 1:
            tokens.append(canonical[1:])
    return _dedupe_strings(tokens)


def _canonical_number_token(token: str) -> str:
    if "." in token and "," in token:
        return token.replace(".", "").replace(",", ".")
    if "," in token:
        groups = token.split(",")
        if len(groups) == 2 and len(groups[1]) in {1, 2}:
            return token.replace(",", ".")
        if len(groups) > 1 and all(len(group) == 3 for group in groups[1:]):
            return token.replace(",", "")
        return token.replace(",", ".")
    if "." in token:
        groups = token.split(".")
        if len(groups) > 2 and all(len(group) == 3 for group in groups[1:]):
            return token.replace(".", "")
        if len(groups) == 2 and len(groups[1]) == 3 and len(groups[0]) <= 3:
            return token.replace(".", "")
    return token


def _evidence_ref_number_tokens(evidence_refs: list[dict[str, Any]]) -> list[str]:
    tokens: list[str] = []
    for ref in evidence_refs:
        for field in ["summary", "source_excerpt"]:
            value = ref.get(field)
            if isinstance(value, str):
                tokens.extend(_number_tokens(value))
    return _dedupe_strings(tokens)


def _dedupe_strings(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _looks_vietnamese(text: str) -> bool:
    return bool(re.search(r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]", text.lower()))


def _report_language(value: Any) -> str:
    return value if value in {"vi", "en"} else "vi"


def _claim_token(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_")


def _vietnamese_executive_summary(report: dict[str, Any]) -> list[str]:
    grouped = len(report["grouped_findings"])
    weak = len(report["weak_or_limited_signals"])
    return [
        f"Hệ thống xác định {grouped} nhóm tín hiệu rủi ro được hỗ trợ bởi bằng chứng đã thẩm định.",
        f"Hệ thống xác định {weak} tín hiệu có mức hỗ trợ hạn chế cần xem xét thận trọng.",
        (
            "Khuyến nghị chuyên gia rà soát các tín hiệu được trình bày trong báo cáo."
            if grouped or weak
            else "Không xác định được tín hiệu rủi ro cuối cùng từ bằng chứng đã cung cấp."
        ),
    ]


def _report_synthesis_system_prompt() -> str:
    return (
        "Bạn là Mô hình Tổng hợp Báo cáo của Vinacount. Chỉ điền các ô mô tả "
        "trong Report Narrative Draft bằng đúng report_language của request và trích dẫn "
        "Permitted Report Claim thuộc đúng finding. "
        "Với grouped_findings, điền cả text và why_this_matters: text tóm tắt tín hiệu, "
        "why_this_matters giải thích ngắn ý nghĩa kinh tế hoặc lý do cần rà soát thủ công. "
        "Dùng ngôn ngữ tín hiệu rủi ro, ưu tiên rà soát và cần chuyên gia xem xét; "
        "không viết như kết luận sai sót, gian lận, thao túng, bất thường đã được chứng minh "
        "hoặc hành vi có chủ ý. Chỉ nêu số tuyệt đối khi proposition của Permitted Report Claim "
        "ghi rõ số đó; không dùng allowed_numbers như nguồn để viết thêm số liệu. Nếu proposition "
        "chỉ nêu tỷ lệ, chỉ dùng tỷ lệ. Không bịa số còn thiếu. Nêu giới hạn phạm vi bằng ngôn ngữ thận trọng nếu "
        "phù hợp với claim: không có so sánh ngành, vĩ mô, thị trường, tin tức hoặc nguồn ngoài. "
        "Không thêm, xóa hoặc nhóm lại finding; không thay đổi category, support, severity, "
        "evidence, method, scope hoặc limitation. Không suy diễn số liệu, kỳ báo cáo, quan hệ "
        "nhân quả, mức độ chắc chắn, gian lận hay kết luận pháp lý. Chỉ trả về JSON đúng schema."
    )


def _post_deepseek_report_synthesis(
    *,
    url: str,
    payload: dict[str, Any],
    timeout_seconds: float,
    headers: dict[str, str],
) -> dict[str, Any]:
    provider_request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(provider_request, timeout=timeout_seconds) as response:
            response_text = response.read().decode("utf-8", errors="replace")
    except (TimeoutError, socket.timeout) as error:
        raise TimeoutError("report_synthesis_provider_timeout") from error
    except urllib.error.URLError as error:
        if isinstance(error.reason, (TimeoutError, socket.timeout)):
            raise TimeoutError("report_synthesis_provider_timeout") from error
        raise OSError("report_synthesis_provider_transport_error") from error
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise ValueError("report_synthesis_provider_response_invalid") from error
    if not isinstance(parsed, dict):
        raise ValueError("report_synthesis_provider_response_invalid")
    return parsed
