from __future__ import annotations

import contextvars
import json
import os
import socket
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Callable

from vinacount.analysis_chain_tracer import OfflineDetectorAdapter
from vinacount.detector_contract import (
    ALLOWED_EVIDENCE_REF_TYPES,
    ALLOWED_SUPPORT_LEVELS,
    DetectorAdapter,
    DetectorAssessment,
    DetectorPacket,
    detector_assessment_json_schema,
    enrich_detector_packet_evidence_roles,
    normalize_json_content,
    parse_and_validate_detector_assessment,
    validate_detector_packet,
    visible_packet_evidence_ids,
)
from vinacount.env_loader import load_dotenv_if_available


DEFAULT_RUNTIME_DETECTOR_MODE = "deterministic_local"
DEFAULT_RUNTIME_DETECTOR_VERSION = "offline_detector_adapter.v1"
API_LLM_BASELINE_MODE = "api_llm_baseline"
SFT_VLLM_MODE = "sft_vllm"
RUNTIME_DETECTOR_MODE_ENV = "VINACOUNT_RUNTIME_DETECTOR_MODE"
DEFAULT_API_LLM_MODEL = "deepseek-v4-flash"
DEFAULT_API_LLM_PROMPT_VERSION = "api_llm_detector_baseline_prompt_v1_evidence_guard"
DEFAULT_SFT_VLLM_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_SFT_VLLM_MODEL = "vinacount-qwen-lora-detector-v1"
DEFAULT_SFT_VLLM_MAX_TOKENS = 1024
DEFAULT_SFT_VLLM_TIMEOUT_SECONDS = 180.0
DEFAULT_SFT_DETECTOR_LABEL = "Qwen3.5-4B-Detector-LoRA-Guard"
DEFAULT_SFT_SYSTEM_PROMPT_VERSION = "v2_evidence_bundle"
DEFAULT_SFT_CONTRACT_GUARD_VERSION = "deterministic_contract_guard_v2"
SFT_VLLM_BASE_URL_ENV = "VINACOUNT_SFT_VLLM_BASE_URL"
SFT_VLLM_MODEL_ENV = "VINACOUNT_SFT_VLLM_MODEL"
SFT_VLLM_API_KEY_ENV = "VINACOUNT_SFT_VLLM_API_KEY"
SFT_VLLM_MAX_TOKENS_ENV = "VINACOUNT_SFT_VLLM_MAX_TOKENS"
SFT_VLLM_TIMEOUT_SECONDS_ENV = "VINACOUNT_SFT_VLLM_TIMEOUT_SECONDS"
DEEPSEEK_CHAT_COMPLETIONS_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"

DETECTOR_SFT_SYSTEM_PROMPT_V1 = (
    "You are a specialized accounting irregularity risk-signal detector for Vietnamese financial reports.\n\n"
    "Your task is to assess whether the provided DetectorPacket supports the given CandidateRisk category.\n\n"
    "Return only a valid DetectorAssessment JSON object.\n\n"
    "Use only evidence provided in the packet.\n"
    "Reference only provided evidence IDs.\n"
    "Do not use outside knowledge.\n"
    "Do not use hidden injection metadata.\n"
    "Do not claim fraud, manipulation, intent, concealment, or legal misstatement.\n"
    "Use risk-signal language only.\n"
    "Keep rationale_short to 1-3 sentences."
)
DETECTOR_SFT_SYSTEM_PROMPT_V2_EVIDENCE_BUNDLE = (
    f"{DETECTOR_SFT_SYSTEM_PROMPT_V1}\n\n"
    "Calibrate support from the complete visible evidence bundle, not from trigger magnitude alone.\n"
    "A tool finding's strength describes the magnitude of that trigger; it does not by itself establish "
    "that the candidate is fully supported.\n"
    "Use supported when the packet contains direct, coherent, and sufficient evidence, normally through "
    "multiple aligned signals, one signal plus independent corroboration, or a category-specific decisive "
    "item identified by a visible rule.\n"
    "Use weakly_supported when a meaningful signal is present but isolated, single-source, incomplete, "
    "mixed, or missing an important corroborating component. A strong isolated signal may therefore be "
    "weakly_supported.\n"
    "Use not_supported when visible evidence contradicts or fails to support the candidate. Use "
    "insufficient_evidence when the packet lacks the evidence needed to assess it."
)
DETECTOR_SFT_SYSTEM_PROMPTS = {
    "v1": DETECTOR_SFT_SYSTEM_PROMPT_V1,
    "v2_evidence_bundle": DETECTOR_SFT_SYSTEM_PROMPT_V2_EVIDENCE_BUNDLE,
}

AdapterFactory = Callable[[], DetectorAdapter]


class RuntimeDetectorTimeoutError(TimeoutError):
    pass


class RuntimeDetectorTransportError(RuntimeError):
    pass


class RuntimeDetectorProviderResponseError(ValueError):
    pass


class RuntimeDetectorInvalidJsonError(ValueError):
    pass


class RuntimeDetectorGuardError(ValueError):
    pass


@dataclass(frozen=True)
class RuntimeDetectorModeSpec:
    mode: str
    version: str
    provider: str
    demo_safe: bool
    adapter_factory: AdapterFactory

    def select(self, selection: str) -> RuntimeDetectorModeSelection:
        return RuntimeDetectorModeSelection(
            mode=self.mode,
            version=self.version,
            provider=self.provider,
            selection=selection,
            demo_safe=self.demo_safe,
            adapter=self.adapter_factory(),
        )


@dataclass(frozen=True)
class RuntimeDetectorModeSelection:
    mode: str
    version: str
    provider: str
    selection: str
    demo_safe: bool
    adapter: DetectorAdapter | None = None

    def audit_metadata(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "version": self.version,
            "provider": self.provider,
            "selection": self.selection,
            "demo_safe": self.demo_safe,
        }


class RuntimeDetectorModeRegistry:
    def __init__(self, specs: list[RuntimeDetectorModeSpec] | None = None) -> None:
        mode_specs = specs if specs is not None else default_runtime_detector_mode_specs()
        self._specs = {spec.mode: spec for spec in mode_specs}
        if len(self._specs) != len(mode_specs):
            raise ValueError("Runtime detector mode registry contains duplicate mode ids.")

    def resolve(self, mode: str | None = None) -> RuntimeDetectorModeSelection:
        mode_id = mode or DEFAULT_RUNTIME_DETECTOR_MODE
        if mode_id not in self._specs:
            raise ValueError(f"Unknown runtime detector mode: {mode_id}")
        selection = "backend_config" if mode is not None else "backend_default"
        return self._specs[mode_id].select(selection)


def default_runtime_detector_mode_selection() -> RuntimeDetectorModeSelection:
    return RuntimeDetectorModeRegistry().resolve()


def runtime_detector_mode_from_environment() -> str | None:
    load_dotenv_if_available()
    value = os.environ.get(RUNTIME_DETECTOR_MODE_ENV)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def default_runtime_detector_mode_specs() -> list[RuntimeDetectorModeSpec]:
    load_dotenv_if_available()
    return [
        RuntimeDetectorModeSpec(
            mode=DEFAULT_RUNTIME_DETECTOR_MODE,
            version=DEFAULT_RUNTIME_DETECTOR_VERSION,
            provider="local",
            demo_safe=True,
            adapter_factory=OfflineDetectorAdapter,
        ),
        RuntimeDetectorModeSpec(
            mode=API_LLM_BASELINE_MODE,
            version=f"{DEFAULT_API_LLM_PROMPT_VERSION}.{DEFAULT_API_LLM_MODEL}",
            provider="deepseek",
            demo_safe=False,
            adapter_factory=_api_llm_baseline_adapter_from_env,
        ),
        RuntimeDetectorModeSpec(
            mode=SFT_VLLM_MODE,
            version=(
                f"{DEFAULT_SFT_DETECTOR_LABEL}.{DEFAULT_SFT_VLLM_MODEL}."
                f"{DEFAULT_SFT_SYSTEM_PROMPT_VERSION}.{DEFAULT_SFT_CONTRACT_GUARD_VERSION}"
            ),
            provider="vllm",
            demo_safe=False,
            adapter_factory=lambda: SftVllmRuntimeDetectorAdapter(
                base_url=os.environ.get(SFT_VLLM_BASE_URL_ENV, DEFAULT_SFT_VLLM_BASE_URL),
                model=os.environ.get(SFT_VLLM_MODEL_ENV, DEFAULT_SFT_VLLM_MODEL),
                api_key=_optional_env(SFT_VLLM_API_KEY_ENV),
                max_tokens=_positive_int_env(SFT_VLLM_MAX_TOKENS_ENV, DEFAULT_SFT_VLLM_MAX_TOKENS),
                timeout_seconds=_positive_float_env(
                    SFT_VLLM_TIMEOUT_SECONDS_ENV,
                    DEFAULT_SFT_VLLM_TIMEOUT_SECONDS,
                ),
            ),
        ),
    ]


@dataclass(frozen=True)
class ApiLlmRuntimeDetectorAdapter:
    api_key: str
    model: str = DEFAULT_API_LLM_MODEL
    prompt_version: str = DEFAULT_API_LLM_PROMPT_VERSION
    timeout_seconds: float = 30.0
    temperature: float = 0.0

    def __call__(self, packet: DetectorPacket) -> DetectorAssessment:
        packet_payload = _contract_dict(packet)
        request_payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _api_detector_system_prompt(self.prompt_version)},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "detector_packet": packet_payload,
                            "detector_assessment_output_contract": _detector_assessment_output_contract(packet_payload),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                },
            ],
            "temperature": self.temperature,
            "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"},
        }
        content = _post_chat_completion(
            url=DEEPSEEK_CHAT_COMPLETIONS_URL,
            payload=request_payload,
            timeout_seconds=self.timeout_seconds,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        return _assessment_from_completion(content, packet)


@dataclass(frozen=True)
class SftVllmRuntimeDetectorAdapter:
    base_url: str = DEFAULT_SFT_VLLM_BASE_URL
    model: str = DEFAULT_SFT_VLLM_MODEL
    system_prompt_version: str = DEFAULT_SFT_SYSTEM_PROMPT_VERSION
    timeout_seconds: float = DEFAULT_SFT_VLLM_TIMEOUT_SECONDS
    max_tokens: int = DEFAULT_SFT_VLLM_MAX_TOKENS
    temperature: float = 0.0
    api_key: str | None = None
    _last_contract_guard_actions: contextvars.ContextVar[tuple[str, ...]] = field(
        default_factory=lambda: contextvars.ContextVar("sft_vllm_last_contract_guard_actions", default=()),
        init=False,
        repr=False,
        compare=False,
    )

    @property
    def last_contract_guard_actions(self) -> tuple[str, ...]:
        return self._last_contract_guard_actions.get()

    def __call__(self, packet: DetectorPacket) -> DetectorAssessment:
        self._last_contract_guard_actions.set(())
        packet_payload = canonical_sft_detector_packet_payload(packet)
        prompt = render_qwen_no_think_messages(
            [
                {"role": "system", "content": _sft_system_prompt(self.system_prompt_version)},
                {"role": "user", "content": canonical_sft_detector_packet_json(packet_payload)},
            ]
        )
        request_payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        content = _post_completion(
            url=f"{self.base_url.rstrip('/')}/completions",
            payload=request_payload,
            timeout_seconds=self.timeout_seconds,
            headers=headers,
        )
        assessment, guard_actions = _guarded_assessment_from_completion(content, packet)
        self._last_contract_guard_actions.set(guard_actions)
        return assessment


def injected_runtime_detector_mode_registry(
    adapter: DetectorAdapter,
    *,
    mode: str | None = None,
) -> RuntimeDetectorModeRegistry:
    mode_id = mode or DEFAULT_RUNTIME_DETECTOR_MODE
    return RuntimeDetectorModeRegistry(
        [
            RuntimeDetectorModeSpec(
                mode=mode_id,
                version="injected_detector_adapter.v1",
                provider="local",
                demo_safe=mode_id == DEFAULT_RUNTIME_DETECTOR_MODE,
                adapter_factory=lambda: adapter,
            )
        ]
    )


def _api_llm_baseline_adapter_from_env() -> ApiLlmRuntimeDetectorAdapter:
    load_dotenv_if_available()
    api_key = os.environ.get(DEEPSEEK_API_KEY_ENV)
    if not api_key:
        raise ValueError(f"{API_LLM_BASELINE_MODE} requires {DEEPSEEK_API_KEY_ENV} during backend setup")
    return ApiLlmRuntimeDetectorAdapter(api_key=api_key)


def _optional_env(name: str) -> str | None:
    load_dotenv_if_available()
    value = os.environ.get(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _positive_int_env(name: str, default: int) -> int:
    load_dotenv_if_available()
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _positive_float_env(name: str, default: float) -> float:
    load_dotenv_if_available()
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive number") from error
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive number")
    return parsed


def canonical_sft_detector_packet_payload(packet: DetectorPacket | dict[str, Any]) -> dict[str, Any]:
    packet_data = _contract_dict(packet)
    if not isinstance(packet_data, dict):
        raise ValueError("invalid_detector_packet")
    canonical_fields = DetectorPacket.__dataclass_fields__
    canonical = {key: packet_data[key] for key in canonical_fields if key in packet_data and packet_data[key] is not None}
    canonical = enrich_detector_packet_evidence_roles(canonical)
    validate_detector_packet(canonical)
    return canonical


def canonical_sft_detector_packet_json(packet: DetectorPacket | dict[str, Any]) -> str:
    return json.dumps(
        canonical_sft_detector_packet_payload(packet),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def render_qwen_no_think_messages(messages: list[dict[str, str]]) -> str:
    if len(messages) != 2 or [message.get("role") for message in messages] != ["system", "user"]:
        raise ValueError("sft_detector_messages_invalid")
    system = messages[0]["content"].strip()
    user = messages[1]["content"].strip()
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def _post_chat_completion(
    *,
    url: str,
    payload: dict[str, Any],
    timeout_seconds: float,
    headers: dict[str, str] | None = None,
) -> str:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            **(headers or {}),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_text = response.read().decode("utf-8", errors="replace")
    except (TimeoutError, socket.timeout) as error:
        raise RuntimeDetectorTimeoutError("runtime_detector_timeout") from error
    except urllib.error.URLError as error:
        if isinstance(error.reason, (TimeoutError, socket.timeout)):
            raise RuntimeDetectorTimeoutError("runtime_detector_timeout") from error
        raise RuntimeDetectorTransportError("runtime_detector_transport_error") from error
    try:
        response_payload = json.loads(response_text)
        content = response_payload["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
        raise RuntimeDetectorProviderResponseError("detector_provider_response_invalid") from error
    if not isinstance(content, str) or not content.strip():
        raise RuntimeDetectorProviderResponseError("detector_provider_response_invalid")
    return content


def _post_completion(
    *,
    url: str,
    payload: dict[str, Any],
    timeout_seconds: float,
    headers: dict[str, str] | None = None,
) -> str:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_text = response.read().decode("utf-8", errors="replace")
    except (TimeoutError, socket.timeout) as error:
        raise RuntimeDetectorTimeoutError("runtime_detector_timeout") from error
    except urllib.error.URLError as error:
        if isinstance(error.reason, (TimeoutError, socket.timeout)):
            raise RuntimeDetectorTimeoutError("runtime_detector_timeout") from error
        raise RuntimeDetectorTransportError("runtime_detector_transport_error") from error
    try:
        response_payload = json.loads(response_text)
        content = response_payload["choices"][0]["text"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as error:
        raise RuntimeDetectorProviderResponseError("detector_provider_response_invalid") from error
    if not isinstance(content, str) or not content.strip():
        raise RuntimeDetectorProviderResponseError("detector_provider_response_invalid")
    return content


def _assessment_from_completion(content: str, packet: DetectorPacket) -> DetectorAssessment:
    parsed = parse_and_validate_detector_assessment(content, packet)
    return DetectorAssessment(**parsed)


def _guarded_assessment_from_completion(
    content: str,
    packet: DetectorPacket,
) -> tuple[DetectorAssessment, tuple[str, ...]]:
    packet_payload = _contract_dict(packet)
    try:
        assessment = _assessment_dict_from_completion(content)
    except ValueError as error:
        raise RuntimeDetectorInvalidJsonError(str(error)) from error
    guard_actions = apply_sft_detector_contract_guard(assessment, packet_payload)
    guarded_content = json.dumps(assessment, ensure_ascii=False, sort_keys=True)
    try:
        parsed = parse_and_validate_detector_assessment(guarded_content, packet)
    except ValueError as error:
        raise RuntimeDetectorGuardError(str(error)) from error
    return DetectorAssessment(**parsed), tuple(guard_actions)


def _assessment_dict_from_completion(content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(normalize_json_content(content))
    except json.JSONDecodeError as error:
        raise ValueError("invalid_json") from error
    if not isinstance(parsed, dict):
        raise ValueError("wrong_top_level_structure")
    if parsed.get("type") == "DetectorAssessment" and isinstance(parsed.get("data"), dict):
        parsed = parsed["data"]
    return parsed


def _sft_system_prompt(version: str) -> str:
    try:
        return DETECTOR_SFT_SYSTEM_PROMPTS[version]
    except KeyError as error:
        raise ValueError(f"Unknown SFT detector system prompt version: {version}") from error


def apply_sft_detector_contract_guard(assessment: dict[str, Any], packet: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    actions.extend(_repair_legacy_decision_support_level(assessment))
    actions.extend(_repair_missing_identity_fields(assessment, packet))
    actions.extend(_repair_missing_validated_signals(assessment, packet))
    actions.extend(_strip_auxiliary_assessment_fields(assessment))
    actions.extend(_strip_auxiliary_evidence_ref_fields(assessment))
    actions.extend(_strip_auxiliary_signal_fields(assessment))
    actions.extend(_repair_unambiguous_evidence_ref_ids(assessment, packet))
    actions.extend(_repair_unambiguous_evidence_ref_types(assessment, packet))
    actions.extend(_remove_invalid_optional_tool_result_ids(assessment, packet))
    return sorted(set(actions))


def _repair_legacy_decision_support_level(assessment: dict[str, Any]) -> list[str]:
    if "support_level" in assessment:
        return []
    decision = assessment.get("decision")
    if decision not in ALLOWED_SUPPORT_LEVELS:
        return []
    assessment["support_level"] = decision
    return ["repaired_legacy_decision_support_level"]


def _repair_missing_identity_fields(assessment: dict[str, Any], packet: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    if not isinstance(assessment.get("assessment_id"), str) or not assessment.get("assessment_id"):
        packet_id = packet.get("packet_id")
        if isinstance(packet_id, str) and packet_id:
            assessment["assessment_id"] = f"ASSESS_{packet_id}"
            actions.append("repaired_missing_assessment_id")
    for field_name in ("packet_id", "candidate_id", "report_id", "risk_category"):
        if field_name == "risk_category":
            value = packet.get("task", {}).get("risk_category")
        else:
            value = packet.get(field_name)
        if not isinstance(value, str) or not value:
            continue
        if assessment.get(field_name) == value:
            continue
        was_missing = field_name not in assessment
        assessment[field_name] = value
        actions.append(f"repaired_missing_{field_name}" if was_missing else f"repaired_{field_name}_identity")
    return sorted(actions)


def _repair_missing_validated_signals(assessment: dict[str, Any], packet: dict[str, Any]) -> list[str]:
    if "validated_signals" in assessment:
        return []
    support_level = assessment.get("support_level")
    if support_level not in ALLOWED_SUPPORT_LEVELS:
        return []
    cited_evidence_refs = assessment.get("cited_evidence_refs")
    if not isinstance(cited_evidence_refs, list) or not cited_evidence_refs:
        return []
    signal_ids = assessment.get("supporting_signal_ids")
    if not isinstance(signal_ids, list) or not all(isinstance(signal_id, str) for signal_id in signal_ids):
        signal_ids = packet.get("candidate_summary", {}).get("supporting_signal_ids", [])
    visible_signal_ids = set(packet.get("candidate_summary", {}).get("supporting_signal_ids", []))
    visible_signal_ids.update(
        finding.get("signal_id")
        for finding in packet.get("tool_findings", [])
        if isinstance(finding, dict) and isinstance(finding.get("signal_id"), str)
    )
    normalized_signal_ids = [signal_id for signal_id in signal_ids if signal_id in visible_signal_ids]
    if not normalized_signal_ids:
        return []
    status = {
        "supported": "validated",
        "weakly_supported": "partially_validated",
        "not_supported": "rejected",
        "insufficient_evidence": "not_assessable",
    }[support_level]
    assessment["validated_signals"] = [
        {
            "signal_id": signal_id,
            "status": status,
            "support_level": support_level,
            "cited_evidence_refs": list(cited_evidence_refs),
        }
        for signal_id in normalized_signal_ids
    ]
    return ["repaired_missing_validated_signals"]


def _strip_auxiliary_assessment_fields(assessment: dict[str, Any]) -> list[str]:
    allowed_keys = {
        "assessment_id",
        "packet_id",
        "candidate_id",
        "report_id",
        "risk_category",
        "support_level",
        "confidence",
        "severity",
        "validated_signals",
        "cited_evidence_refs",
        "rationale_short",
    }
    extra_keys = sorted(set(assessment) - allowed_keys)
    if not extra_keys:
        return []
    for key in extra_keys:
        assessment.pop(key, None)
    return ["stripped_assessment_aux_fields"]


def _strip_auxiliary_evidence_ref_fields(assessment: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    allowed_ref_keys = {"evidence_ref_type", "ref_id", "role"}
    for evidence_ref in _iter_assessment_evidence_refs(assessment):
        extra_keys = sorted(set(evidence_ref) - allowed_ref_keys)
        if not extra_keys:
            continue
        for key in extra_keys:
            evidence_ref.pop(key, None)
        actions.append("stripped_evidence_ref_aux_fields")
    return sorted(set(actions))


def _strip_auxiliary_signal_fields(assessment: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    allowed_signal_keys = {"signal_id", "status", "support_level", "tool_result_id", "cited_evidence_refs"}
    for signal in _iter_assessment_signals(assessment):
        extra_keys = sorted(set(signal) - allowed_signal_keys)
        if not extra_keys:
            continue
        for key in extra_keys:
            signal.pop(key, None)
        actions.append("stripped_signal_aux_fields")
    return sorted(set(actions))


def _repair_unambiguous_evidence_ref_ids(assessment: dict[str, Any], packet: dict[str, Any]) -> list[str]:
    visible_ids = visible_packet_evidence_ids(packet)
    actions: list[str] = []
    for evidence_ref in _iter_assessment_evidence_refs(assessment):
        ref_id = evidence_ref.get("ref_id")
        if not isinstance(ref_id, str) or ref_id in visible_ids:
            continue
        local_id = ref_id.rsplit(":", 1)[-1]
        matches = sorted(visible_id for visible_id in visible_ids if visible_id.endswith(f":{local_id}"))
        if len(matches) != 1:
            continue
        evidence_ref["ref_id"] = matches[0]
        actions.append("repaired_evidence_ref_id")
    return sorted(set(actions))


def _repair_unambiguous_evidence_ref_types(assessment: dict[str, Any], packet: dict[str, Any]) -> list[str]:
    evidence_type_by_ref_id = _evidence_type_by_ref_id(packet)
    actions: list[str] = []
    for evidence_ref in _iter_assessment_evidence_refs(assessment):
        evidence_ref_type = evidence_ref.get("evidence_ref_type")
        if evidence_ref_type in ALLOWED_EVIDENCE_REF_TYPES:
            continue
        inferred_type = evidence_type_by_ref_id.get(evidence_ref.get("ref_id"))
        if inferred_type is None:
            continue
        evidence_ref["evidence_ref_type"] = inferred_type
        actions.append("repaired_evidence_ref_type")
    return sorted(set(actions))


def _remove_invalid_optional_tool_result_ids(assessment: dict[str, Any], packet: dict[str, Any]) -> list[str]:
    visible_tool_result_ids = {
        finding.get("tool_result_id")
        for finding in packet.get("tool_findings", [])
        if isinstance(finding, dict) and finding.get("tool_result_id")
    }
    actions: list[str] = []
    for signal in _iter_assessment_signals(assessment):
        tool_result_id = signal.get("tool_result_id")
        if not tool_result_id or tool_result_id in visible_tool_result_ids:
            continue
        signal.pop("tool_result_id", None)
        actions.append("removed_invalid_optional_tool_result_id")
    return sorted(set(actions))


def _iter_assessment_signals(assessment: dict[str, Any]) -> list[dict[str, Any]]:
    signals = assessment.get("validated_signals", [])
    if not isinstance(signals, list):
        return []
    return [signal for signal in signals if isinstance(signal, dict)]


def _iter_assessment_evidence_refs(assessment: dict[str, Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    cited_refs = assessment.get("cited_evidence_refs", [])
    if isinstance(cited_refs, list):
        refs.extend(ref for ref in cited_refs if isinstance(ref, dict))
    for signal in _iter_assessment_signals(assessment):
        signal_refs = signal.get("cited_evidence_refs", [])
        if isinstance(signal_refs, list):
            refs.extend(ref for ref in signal_refs if isinstance(ref, dict))
    return refs


def _evidence_type_by_ref_id(packet: dict[str, Any]) -> dict[str, str]:
    evidence_types: dict[str, str] = {}
    for rule in packet.get("rules", []):
        if isinstance(rule, dict) and rule.get("rule_id"):
            evidence_types[rule["rule_id"]] = "rule"
    for finding in packet.get("tool_findings", []):
        if not isinstance(finding, dict):
            continue
        if finding.get("tool_result_id"):
            evidence_types[finding["tool_result_id"]] = "tool_result"
        for ref in finding.get("evidence_refs", []):
            if (
                isinstance(ref, dict)
                and ref.get("ref_id")
                and ref.get("evidence_ref_type") in ALLOWED_EVIDENCE_REF_TYPES
            ):
                evidence_types[ref["ref_id"]] = ref["evidence_ref_type"]
    for row in packet.get("relevant_table_rows", []):
        if not isinstance(row, dict):
            continue
        report_id = row.get("report_id")
        if not report_id:
            continue
        if row.get("row_id"):
            evidence_types[f"{report_id}:{row['row_id']}"] = "table_row"
        if row.get("local_evidence_id"):
            evidence_types[f"{report_id}:{row['local_evidence_id']}"] = "table_row"
        values = row.get("values", {})
        value_iter = values.values() if isinstance(values, dict) else values if isinstance(values, list) else []
        for value in value_iter:
            if isinstance(value, dict) and value.get("cell_id"):
                evidence_types[f"{report_id}:{value['cell_id']}"] = "table_cell"
    for note in packet.get("relevant_notes", []):
        if not isinstance(note, dict):
            continue
        report_id = note.get("report_id")
        if not report_id:
            continue
        if note.get("note_id"):
            evidence_types[f"{report_id}:{note['note_id']}"] = "note"
        if note.get("local_evidence_id"):
            evidence_types[f"{report_id}:{note['local_evidence_id']}"] = "note"
    for span in packet.get("relevant_variance_explanations", []):
        if not isinstance(span, dict):
            continue
        report_id = span.get("report_id")
        if not report_id:
            continue
        if span.get("span_id"):
            evidence_types[f"{report_id}:{span['span_id']}"] = "variance_explanation_span"
        if span.get("local_evidence_id"):
            evidence_types[f"{report_id}:{span['local_evidence_id']}"] = "variance_explanation_span"
    return evidence_types


def _api_detector_system_prompt(prompt_version: str) -> str:
    if prompt_version != DEFAULT_API_LLM_PROMPT_VERSION:
        raise ValueError(f"unsupported runtime API detector prompt version: {prompt_version}")
    return (
        "You are a specialized accounting irregularity risk-signal detector. "
        "Return one DetectorAssessment JSON object only. Use only the detector packet evidence, "
        "copy identity fields exactly, cite only visible evidence IDs, and avoid fraud or legal-misstatement language."
    )


def _detector_assessment_output_contract(packet: dict[str, Any]) -> dict[str, Any]:
    schema = detector_assessment_json_schema()
    signal_schema = schema["properties"]["validated_signals"]["items"]
    evidence_ref_schema = schema["properties"]["cited_evidence_refs"]["items"]
    return {
        "return_only": "one DetectorAssessment JSON object; no markdown, array, or wrapper object",
        "top_level_keys_exact": list(schema["required"]),
        "additional_top_level_keys_allowed": False,
        "copy_identity_fields_exactly": {
            "packet_id": packet["packet_id"],
            "candidate_id": packet["candidate_id"],
            "report_id": packet["report_id"],
            "risk_category": packet["task"]["risk_category"],
        },
        "allowed_values": {
            "support_level": schema["properties"]["support_level"]["enum"],
            "severity": schema["properties"]["severity"]["enum"],
            "validated_signals.status": signal_schema["properties"]["status"]["enum"],
            "evidence_ref.evidence_ref_type": evidence_ref_schema["properties"]["evidence_ref_type"]["enum"],
            "evidence_ref.role": evidence_ref_schema["properties"]["role"]["enum"],
        },
        "confidence": "number only, between 0 and 1",
        "rationale_short": "1-3 concise sentences using risk-signal language only",
    }


def _contract_dict(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    return value
