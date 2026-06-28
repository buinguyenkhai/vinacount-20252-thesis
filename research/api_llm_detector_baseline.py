from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from research.detector_contract_validation import (
    ALLOWED_EVIDENCE_REF_ROLES,
    ALLOWED_EVIDENCE_REF_TYPES,
    ALLOWED_SEVERITIES,
    ALLOWED_SIGNAL_STATUSES,
    ALLOWED_SUPPORT_LEVELS,
    detector_assessment_json_schema as _shared_detector_assessment_json_schema,
    parse_and_validate_detector_assessment,
)

DEFAULT_RELEASE_DIR = Path("data/real_manual/combined_real_manual_validation_release")
DEFAULT_OUTPUT_ROOT = Path("artifacts/api_llm_detector_baseline")
PROMPT_VERSION_BASE = "api_llm_detector_baseline_prompt_v1"
PROMPT_VERSION_EVIDENCE_GUARD = "api_llm_detector_baseline_prompt_v1_evidence_guard"
PROMPT_VERSION_SUPPORT_CALIBRATED = "api_llm_detector_baseline_prompt_v1_support_calibrated"
DEFAULT_PROMPT_VERSION = PROMPT_VERSION_EVIDENCE_GUARD
DEFAULT_TEMPERATURE = 0.0
SUPPORTED_PROMPT_VERSIONS = {
    PROMPT_VERSION_BASE,
    PROMPT_VERSION_EVIDENCE_GUARD,
    PROMPT_VERSION_SUPPORT_CALIBRATED,
}
DEFAULT_SWEEP_PRESET = "deepseek_v4_flash_prompt_config_v1"
SWEEP_PRESETS: dict[str, list[dict[str, Any]]] = {
    DEFAULT_SWEEP_PRESET: [
        {
            "variant_id": "v1_temp0_0",
            "prompt_version": PROMPT_VERSION_BASE,
            "temperature": 0.0,
        },
        {
            "variant_id": "v1_temp0_2",
            "prompt_version": PROMPT_VERSION_BASE,
            "temperature": 0.2,
        },
        {
            "variant_id": "evidence_guard_temp0_0",
            "prompt_version": PROMPT_VERSION_EVIDENCE_GUARD,
            "temperature": 0.0,
        },
        {
            "variant_id": "support_calibrated_temp0_0",
            "prompt_version": PROMPT_VERSION_SUPPORT_CALIBRATED,
            "temperature": 0.0,
        },
    ]
}
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"
DEEPSEEK_CHAT_COMPLETIONS_URL = "https://api.deepseek.com/chat/completions"
OPENROUTER_LOCKED_PROVIDER_MODELS = {
    "deepseek/deepseek-v4-flash",
    "deepseek/deepseek-v4-flash-20260423",
}
DEEPSEEK_DIRECT_MODELS = {
    "deepseek-v4-flash",
    "deepseek-v4-pro",
}
DEEPSEEK_DIRECT_MAX_ATTEMPTS = 3
DEEPSEEK_DIRECT_RETRY_INITIAL_SLEEP_SECONDS = 1.0
DEEPSEEK_DIRECT_RETRYABLE_HTTP_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
OPENROUTER_LOCKED_PROVIDER_ROUTING = {
    "order": ["deepseek", "alibaba"],
    "only": ["deepseek", "alibaba"],
    "allow_fallbacks": False,
    "require_parameters": True,
}


@dataclass(frozen=True)
class ApiLlmDetectorBaselineResult:
    status: str
    run_dir: Path
    manifest_path: Path
    predictions_path: Path
    metrics_path: Path
    invalid_responses_path: Path
    records_loaded: int
    predictions_written: int
    invalid_response_count: int
    errors: list[str]


@dataclass(frozen=True)
class ApiLlmDetectorSweepResult:
    status: str
    sweep_dir: Path
    manifest_path: Path
    results_path: Path
    variant_count: int
    errors: list[str]


@dataclass(frozen=True)
class ApiLlmDetectorRequest:
    example_id: str
    prompt_version: str
    provider: str
    model: str
    temperature: float
    detector_packet: dict


@dataclass(frozen=True)
class ApiLlmDetectorResponse:
    content: str
    provider_metadata: dict | None = None


class ApiLlmDetectorClient(Protocol):
    def complete(self, request: ApiLlmDetectorRequest) -> ApiLlmDetectorResponse:
        ...


@dataclass(frozen=True)
class OpenRouterApiLlmDetectorClient:
    api_key: str
    timeout_seconds: float = 30.0

    def complete(self, request: ApiLlmDetectorRequest) -> ApiLlmDetectorResponse:
        payload = {
            "model": request.model,
            "messages": [
                {"role": "system", "content": _detector_system_prompt(request.prompt_version)},
                {
                    "role": "user",
                    "content": json.dumps(request.detector_packet, ensure_ascii=False, sort_keys=True),
                },
            ],
            "temperature": request.temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "DetectorAssessment",
                    "strict": True,
                    "schema": _detector_assessment_json_schema(),
                },
            },
        }
        provider_routing = _openrouter_provider_routing(request.model)
        if provider_routing:
            payload["provider"] = provider_routing
        http_request = urllib.request.Request(
            OPENROUTER_CHAT_COMPLETIONS_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/buinguyenkhai/vinacount",
                "X-Title": "vinacount-api-llm-detector-baseline",
            },
            method="POST",
        )
        with urllib.request.urlopen(http_request, timeout=self.timeout_seconds) as response:
            response_text = response.read().decode("utf-8", errors="replace")
        try:
            response_payload = json.loads(response_text)
        except json.JSONDecodeError as error:
            preview = response_text[max(0, error.pos - 200) : error.pos + 500].strip()
            raise RuntimeError(
                "openrouter_non_json_response "
                f"json_error={error.msg!r} position={error.pos} body_preview={preview!r}"
            ) from error
        try:
            content = response_payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError(
                "openrouter_unexpected_response_shape "
                f"body_preview={json.dumps(response_payload, ensure_ascii=False)[:1000]!r}"
            ) from error
        return ApiLlmDetectorResponse(
            content=content,
            provider_metadata={
                "provider": request.provider,
                "model": request.model,
                "prompt_version": request.prompt_version,
                "temperature": request.temperature,
                "provider_routing": provider_routing,
            },
        )


@dataclass(frozen=True)
class DeepSeekApiLlmDetectorClient:
    api_key: str
    timeout_seconds: float = 30.0

    def complete(self, request: ApiLlmDetectorRequest) -> ApiLlmDetectorResponse:
        payload = {
            "model": request.model,
            "messages": [
                {"role": "system", "content": _detector_system_prompt(request.prompt_version)},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "detector_packet": request.detector_packet,
                            "detector_assessment_output_contract": _detector_assessment_output_contract(
                                request.detector_packet
                            ),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                },
            ],
            "temperature": request.temperature,
            "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"},
        }
        http_request = urllib.request.Request(
            DEEPSEEK_CHAT_COMPLETIONS_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        response_text = _deepseek_urlopen_text_with_retry(http_request, timeout_seconds=self.timeout_seconds)
        try:
            response_payload = json.loads(response_text)
        except json.JSONDecodeError as error:
            preview = response_text[max(0, error.pos - 200) : error.pos + 500].strip()
            raise RuntimeError(
                "deepseek_non_json_response "
                f"json_error={error.msg!r} position={error.pos} body_preview={preview!r}"
            ) from error
        try:
            content = response_payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError(
                "deepseek_unexpected_response_shape "
                f"body_preview={json.dumps(response_payload, ensure_ascii=False)[:1000]!r}"
            ) from error
        usage = response_payload.get("usage") if isinstance(response_payload.get("usage"), dict) else {}
        return ApiLlmDetectorResponse(
            content=content,
            provider_metadata={
                "provider": request.provider,
                "strategy": "direct_deepseek_chat_completions_v1",
                "request_model": request.model,
                "response_model": response_payload.get("model"),
                "prompt_version": request.prompt_version,
                "temperature": request.temperature,
                "thinking": payload["thinking"],
                "response_format": payload["response_format"],
                "system_fingerprint": response_payload.get("system_fingerprint"),
                "usage": usage,
                "cache_usage": {
                    "prompt_cache_hit_tokens": usage.get("prompt_cache_hit_tokens"),
                    "prompt_cache_miss_tokens": usage.get("prompt_cache_miss_tokens"),
                },
            },
        )


def _deepseek_urlopen_text_with_retry(
    request: urllib.request.Request,
    *,
    timeout_seconds: float,
) -> str:
    last_error: BaseException | None = None
    for attempt in range(1, DEEPSEEK_DIRECT_MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                return response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as error:
            body_preview = error.read().decode("utf-8", errors="replace")[:1000]
            if error.code not in DEEPSEEK_DIRECT_RETRYABLE_HTTP_STATUS:
                raise RuntimeError(
                    "deepseek_http_error "
                    f"status={error.code} reason={error.reason!r} body_preview={body_preview!r}"
                ) from error
            last_error = RuntimeError(
                "deepseek_retryable_http_error "
                f"status={error.code} reason={error.reason!r} body_preview={body_preview!r}"
            )
        except urllib.error.URLError as error:
            last_error = error
        if attempt < DEEPSEEK_DIRECT_MAX_ATTEMPTS:
            time.sleep(DEEPSEEK_DIRECT_RETRY_INITIAL_SLEEP_SECONDS * attempt)
    raise RuntimeError(
        "deepseek_request_failed "
        f"attempts={DEEPSEEK_DIRECT_MAX_ATTEMPTS} last_error={last_error!r}"
    ) from last_error


def _openrouter_provider_routing(model: str) -> dict[str, Any] | None:
    if model not in OPENROUTER_LOCKED_PROVIDER_MODELS:
        return None
    return {
        "order": list(OPENROUTER_LOCKED_PROVIDER_ROUTING["order"]),
        "only": list(OPENROUTER_LOCKED_PROVIDER_ROUTING["only"]),
        "allow_fallbacks": OPENROUTER_LOCKED_PROVIDER_ROUTING["allow_fallbacks"],
        "require_parameters": OPENROUTER_LOCKED_PROVIDER_ROUTING["require_parameters"],
    }


def _provider_routing_for_run(provider: str, model: str) -> dict[str, Any] | None:
    if provider != "openrouter":
        return None
    return _openrouter_provider_routing(model)


def _validate_live_provider_model(provider: str, model: str) -> None:
    if provider == "openrouter":
        return
    if provider == "deepseek":
        if model not in DEEPSEEK_DIRECT_MODELS:
            raise ValueError(f"live direct DeepSeek baseline model is not approved: {model}")
        return
    raise ValueError(f"live mode currently supports provider=openrouter or provider=deepseek, got {provider}")


def _validate_prompt_version(prompt_version: str) -> None:
    if prompt_version not in SUPPORTED_PROMPT_VERSIONS:
        raise ValueError(f"unsupported prompt_version: {prompt_version}")


def _detector_assessment_json_schema() -> dict[str, Any]:
    return _shared_detector_assessment_json_schema()


def _detector_assessment_output_contract(packet: dict[str, Any]) -> dict[str, Any]:
    schema = _detector_assessment_json_schema()
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
        "confidence": "number only, between 0 and 1; do not use string labels such as high, medium, or low",
        "validated_signals_item": {
            "required_keys": list(signal_schema["required"]),
            "optional_keys": ["tool_result_id"],
            "additional_keys_allowed": False,
        },
        "evidence_ref_item": {
            "keys_exact": list(evidence_ref_schema["required"]),
            "ref_id_rule": "Use only IDs visible in the detector_packet.",
        },
        "rationale_short": "1-3 concise sentences using risk-signal language only",
    }


@dataclass(frozen=True)
class ReleaseGoldFakeDetectorClient:
    responses_by_example_id: dict[str, dict[str, Any]]

    def complete(self, request: ApiLlmDetectorRequest) -> ApiLlmDetectorResponse:
        return ApiLlmDetectorResponse(
            content=json.dumps(self.responses_by_example_id[request.example_id], ensure_ascii=False)
        )


def run_api_llm_detector_baseline(
    release_dir: Path | str = DEFAULT_RELEASE_DIR,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    *,
    mode: Literal["dry_run", "fake", "live"],
    provider: str = "openrouter",
    model: str | None = None,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    temperature: float = DEFAULT_TEMPERATURE,
    client: ApiLlmDetectorClient | None = None,
    run_id: str | None = None,
    limit: int | None = None,
    example_ids: list[str] | tuple[str, ...] | None = None,
) -> ApiLlmDetectorBaselineResult:
    if mode not in {"dry_run", "fake", "live"}:
        raise ValueError("mode must be dry_run, fake, or live")
    _validate_prompt_version(prompt_version)
    if mode != "dry_run" and not model:
        raise ValueError("model is required for api LLM detector baseline runs")
    model = model or "dry_run_no_model"
    if mode == "live":
        _validate_live_provider_model(provider, model)
    source_root = Path(release_dir)
    run_dir = Path(output_root) / (run_id or _default_run_id(provider, model))
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = _read_json(source_root / "manifest.json")
    records = _read_jsonl(source_root / "validation.jsonl")
    records = _select_records(records, example_ids=example_ids, limit=limit)
    if mode == "fake" and client is None:
        client = ReleaseGoldFakeDetectorClient(
            {
                record["example_id"]: record["output"]["data"]
                for record in records
            }
        )
    if mode == "live" and client is None:
        client = _default_live_client(provider)

    predictions = []
    invalid_responses: list[dict] = []
    raw_invalid_responses: list[dict] = []
    for record in records:
        packet = record["input"]["data"]
        gold = record["output"]["data"]
        prediction_record = _base_prediction_record(record, packet, gold)
        if mode == "dry_run":
            prediction_record["prediction_status"] = "dry_run"
            prediction_record["schema_valid"] = False
            prediction_record["evidence_valid"] = False
        else:
            request = ApiLlmDetectorRequest(
                example_id=record["example_id"],
                prompt_version=prompt_version,
                provider=provider,
                model=model,
                temperature=temperature,
                detector_packet=packet,
            )
            response = client.complete(request) if client is not None else None
            raw_content = response.content if response else ""
            if response and response.provider_metadata:
                prediction_record["provider_metadata"] = response.provider_metadata
            try:
                assessment = _parse_and_validate_prediction(raw_content, packet)
            except ValueError as error:
                prediction_record["prediction_status"] = "invalid"
                prediction_record["schema_valid"] = False
                prediction_record["evidence_valid"] = False
                prediction_record["invalid_reason_codes"] = [str(error)]
                invalid_responses.append(
                    {
                        "example_id": record["example_id"],
                        "packet_id": packet["packet_id"],
                        "candidate_id": packet["candidate_id"],
                        "failure_stage": "response_validation",
                        "reason_codes": [str(error)],
                        "raw_response_debug_path": "debug/raw_invalid_responses.jsonl",
                    }
                )
                raw_invalid_responses.append(
                    {
                        "example_id": record["example_id"],
                        "packet_id": packet["packet_id"],
                        "candidate_id": packet["candidate_id"],
                        "raw_response_text": raw_content,
                        "reason_codes": [str(error)],
                        "non_canonical": True,
                        "debug_only": True,
                        "not_labels": True,
                        "not_training_data": True,
                        "not_detector_visible_data": True,
                    }
                )
            else:
                prediction_record["prediction_status"] = "accepted"
                prediction_record["schema_valid"] = True
                prediction_record["evidence_valid"] = True
                prediction_record["prediction"] = assessment
                prediction_record["predicted_support_level"] = assessment["support_level"]
        predictions.append(prediction_record)

    metrics = _score_predictions(predictions, mode=mode)
    run_manifest = {
        "run_id": run_dir.name,
        "mode": mode,
        "provider": provider,
        "model": model,
        "prompt_version": prompt_version,
        "decoding_config": {"temperature": temperature},
        "provider_routing": _provider_routing_for_run(provider, model),
        "source_release": {
            "path": str(source_root),
            "release_name": manifest.get("release_name"),
            "release_build_id": manifest.get("release_build_id"),
            "dataset_version": manifest.get("dataset_version"),
            "num_examples_total": manifest.get("num_examples_total"),
            "num_examples_by_split": manifest.get("num_examples_by_split"),
        },
        "selection": {
            "limit": limit,
            "example_ids": list(example_ids) if example_ids else None,
            "records_loaded": len(records),
        },
        "artifact_policy": {
            "canonical_prediction_artifact": "predictions.jsonl",
            "canonical_metrics_artifact": "metrics.json",
            "canonical_invalid_response_artifact": "invalid_responses.jsonl",
            "raw_invalid_responses_debug_artifact": "debug/raw_invalid_responses.jsonl",
            "raw_valid_responses_canonical": False,
            "raw_responses_are_labels": False,
            "predictions_are_evaluation_outputs_only": True,
            "not_gold_labels": True,
            "not_training_data": True,
            "usable_for_training": False,
        },
        "reporting_policy": {
            "corpus_description": "Pilot Real-World Development/Evaluation Corpus",
            "protected_test_claims_allowed": False,
            "final_performance_claims_allowed": False,
            "fraud_or_legal_misstatement_claims_allowed": False,
        },
        "artifacts": {
            "predictions": "predictions.jsonl",
            "metrics": "metrics.json",
            "invalid_responses": "invalid_responses.jsonl",
        },
    }

    manifest_path = run_dir / "manifest.json"
    predictions_path = run_dir / "predictions.jsonl"
    metrics_path = run_dir / "metrics.json"
    invalid_responses_path = run_dir / "invalid_responses.jsonl"
    raw_invalid_responses_path = run_dir / "debug" / "raw_invalid_responses.jsonl"
    raw_invalid_responses_path.parent.mkdir(parents=True, exist_ok=True)

    _write_json(manifest_path, run_manifest)
    _write_jsonl(predictions_path, predictions)
    _write_json(metrics_path, metrics)
    _write_jsonl(invalid_responses_path, invalid_responses)
    _write_jsonl(raw_invalid_responses_path, raw_invalid_responses)

    return ApiLlmDetectorBaselineResult(
        status="passed",
        run_dir=run_dir,
        manifest_path=manifest_path,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        invalid_responses_path=invalid_responses_path,
        records_loaded=len(records),
        predictions_written=len(predictions),
        invalid_response_count=len(invalid_responses),
        errors=[],
    )


def run_api_llm_detector_baseline_sweep(
    release_dir: Path | str = DEFAULT_RELEASE_DIR,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    *,
    mode: Literal["dry_run", "fake", "live"],
    provider: str = "openrouter",
    model: str | None = None,
    sweep_id: str,
    preset: str = DEFAULT_SWEEP_PRESET,
    variants: list[dict[str, Any]] | None = None,
    limit: int | None = None,
    example_ids: list[str] | tuple[str, ...] | None = None,
) -> ApiLlmDetectorSweepResult:
    if not model:
        model = "dry_run_no_model" if mode == "dry_run" else None
    if not model:
        raise ValueError("model is required for api LLM detector baseline sweeps")
    selected_variants = variants or SWEEP_PRESETS[preset]
    _validate_sweep_variant_ids(selected_variants)

    output_root = Path(output_root)
    sweep_dir = output_root / sweep_id
    sweep_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for variant in selected_variants:
        variant_id = variant["variant_id"]
        child_run_id = f"{sweep_id}__{variant_id}"
        try:
            _validate_sweep_variant_config(variant)
            child_result = run_api_llm_detector_baseline(
                release_dir=release_dir,
                output_root=output_root,
                mode=mode,
                provider=provider,
                model=model,
                prompt_version=variant["prompt_version"],
                temperature=variant["temperature"],
                run_id=child_run_id,
                limit=limit,
                example_ids=example_ids,
            )
        except Exception as error:
            results.append(
                {
                    "variant_id": variant_id,
                    "run_id": child_run_id,
                    "run_dir": str(output_root / child_run_id),
                    "status": "failed",
                    "prompt_version": variant.get("prompt_version"),
                    "decoding_config": {"temperature": variant.get("temperature")},
                    "records_loaded": 0,
                    "metrics": {},
                    "error": str(error),
                }
            )
            continue
        metrics = _read_json(child_result.metrics_path)
        results.append(
            {
                "variant_id": variant_id,
                "run_id": child_run_id,
                "run_dir": str(child_result.run_dir),
                "status": child_result.status,
                "prompt_version": variant["prompt_version"],
                "decoding_config": {"temperature": variant["temperature"]},
                "records_loaded": child_result.records_loaded,
                "metrics": metrics,
            }
        )

    passed_results = [result for result in results if result["status"] == "passed"]
    recommended = max(passed_results, key=_sweep_rank_key) if passed_results else None
    manifest_path = sweep_dir / "sweep_manifest.json"
    results_path = sweep_dir / "sweep_results.jsonl"
    sweep_manifest = {
        "sweep_id": sweep_id,
        "mode": mode,
        "provider": provider,
        "model": model,
        "preset": preset,
        "provider_routing": _provider_routing_for_run(provider, model),
        "selection": {
            "limit": limit,
            "example_ids": list(example_ids) if example_ids else None,
        },
        "variants": selected_variants,
        "recommended_variant_id": recommended["variant_id"] if recommended else None,
        "status": "passed" if len(passed_results) == len(results) else "completed_with_errors",
        "ranking_policy": {
            "primary": "num_valid_predictions",
            "secondary": [
                "support_level_exact_match_count",
                "false_positive_rejection.correct_or_conservative",
                "insufficient_evidence_detection.true_positive",
            ],
            "diagnostic_only": True,
            "not_final_performance_evidence": True,
        },
        "artifact_policy": _evaluation_artifact_policy(),
        "reporting_policy": _reporting_policy(),
        "artifacts": {
            "sweep_manifest": "sweep_manifest.json",
            "sweep_results": "sweep_results.jsonl",
            "variant_runs": "sibling directories named <sweep_id>__<variant_id>",
        },
    }
    _write_json(manifest_path, sweep_manifest)
    _write_jsonl(results_path, results)
    return ApiLlmDetectorSweepResult(
        status="passed" if len(passed_results) == len(results) else "completed_with_errors",
        sweep_dir=sweep_dir,
        manifest_path=manifest_path,
        results_path=results_path,
        variant_count=len(results),
        errors=[result["error"] for result in results if result["status"] == "failed"],
    )


def _validate_sweep_variant_ids(variants: list[dict[str, Any]]) -> None:
    seen = set()
    for variant in variants:
        if "variant_id" not in variant:
            raise ValueError("sweep variant missing variant_id")
        if variant["variant_id"] in seen:
            raise ValueError(f"duplicate sweep variant_id: {variant['variant_id']}")
        seen.add(variant["variant_id"])


def _validate_sweep_variant_config(variant: dict[str, Any]) -> None:
    for required_field in ["prompt_version", "temperature"]:
        if required_field not in variant:
            raise ValueError(f"sweep variant missing {required_field}")
    _validate_prompt_version(variant["prompt_version"])
    if not isinstance(variant["temperature"], int | float):
        raise ValueError(f"invalid temperature for sweep variant: {variant['variant_id']}")


def _sweep_rank_key(result: dict[str, Any]) -> tuple[int, int, int, int, int]:
    metrics = result["metrics"]
    false_positive = metrics.get("false_positive_rejection", {})
    insufficient = metrics.get("insufficient_evidence_detection", {})
    return (
        metrics.get("num_valid_predictions", 0),
        metrics.get("support_level_exact_match_count", 0),
        false_positive.get("correct_or_conservative", 0),
        insufficient.get("true_positive", 0),
        -metrics.get("num_invalid_responses", 0),
    )


def _evaluation_artifact_policy() -> dict[str, Any]:
    return {
        "canonical_prediction_artifact": "predictions.jsonl",
        "canonical_metrics_artifact": "metrics.json",
        "canonical_invalid_response_artifact": "invalid_responses.jsonl",
        "raw_invalid_responses_debug_artifact": "debug/raw_invalid_responses.jsonl",
        "raw_valid_responses_canonical": False,
        "raw_responses_are_labels": False,
        "predictions_are_evaluation_outputs_only": True,
        "not_gold_labels": True,
        "not_training_data": True,
        "usable_for_training": False,
    }


def _reporting_policy() -> dict[str, Any]:
    return {
        "corpus_description": "Pilot Real-World Development/Evaluation Corpus",
        "protected_test_claims_allowed": False,
        "final_performance_claims_allowed": False,
        "fraud_or_legal_misstatement_claims_allowed": False,
    }


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _select_records(
    records: list[dict[str, Any]],
    *,
    example_ids: list[str] | tuple[str, ...] | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    selected = records
    if example_ids:
        requested = set(example_ids)
        selected = [record for record in selected if record["example_id"] in requested]
        missing = requested - {record["example_id"] for record in selected}
        if missing:
            raise ValueError(f"example_ids not found in release: {sorted(missing)}")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        selected = selected[:limit]
    return selected


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _base_prediction_record(record: dict[str, Any], packet: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    return {
        "example_id": record["example_id"],
        "packet_id": packet["packet_id"],
        "candidate_id": packet["candidate_id"],
        "gold_support_level": gold["support_level"],
        "model_visible_input": {
            "type": "DetectorPacket",
            "data": packet,
        },
    }


def _parse_and_validate_prediction(content: str, packet: dict[str, Any]) -> dict[str, Any]:
    return parse_and_validate_detector_assessment(content, packet)


def _normalize_json_content(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```json\n") and stripped.endswith("\n```"):
        return stripped[len("```json\n") : -len("\n```")].strip()
    if stripped.startswith("```\n") and stripped.endswith("\n```"):
        return stripped[len("```\n") : -len("\n```")].strip()
    return content


def _validate_detector_assessment_prediction(assessment: dict[str, Any], packet: dict[str, Any]) -> None:
    if not isinstance(assessment, dict):
        raise ValueError("schema_mismatch")
    _validate_detector_packet(packet)
    required_fields = {
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
    missing = required_fields - assessment.keys()
    if missing:
        raise ValueError("schema_mismatch")
    if set(assessment.keys()) != required_fields:
        raise ValueError("schema_mismatch")
    if assessment["packet_id"] != packet["packet_id"]:
        raise ValueError("identity_mismatch")
    if assessment["candidate_id"] != packet["candidate_id"]:
        raise ValueError("identity_mismatch")
    if assessment["report_id"] != packet["report_id"]:
        raise ValueError("identity_mismatch")
    if assessment["risk_category"] != packet["task"]["risk_category"]:
        raise ValueError("risk_category_mismatch")
    if assessment["support_level"] not in ALLOWED_SUPPORT_LEVELS:
        raise ValueError("invalid_support_level")
    if assessment["severity"] not in ALLOWED_SEVERITIES:
        raise ValueError("invalid_severity")
    confidence = assessment["confidence"]
    if not isinstance(confidence, int | float) or isinstance(confidence, bool) or confidence < 0 or confidence > 1:
        raise ValueError("invalid_confidence")
    if not isinstance(assessment["validated_signals"], list) or not assessment["validated_signals"]:
        raise ValueError("schema_mismatch")
    if not isinstance(assessment["cited_evidence_refs"], list):
        raise ValueError("schema_mismatch")
    if not isinstance(assessment["rationale_short"], str) or not assessment["rationale_short"].strip():
        raise ValueError("schema_mismatch")
    if _sentence_count(assessment["rationale_short"]) > 3:
        raise ValueError("rationale_too_long")
    if _contains_prohibited_text(assessment):
        raise ValueError("prohibited_risk_language")
    visible_ids = _visible_packet_evidence_ids(packet)
    for evidence_ref in assessment["cited_evidence_refs"]:
        _validate_evidence_ref(evidence_ref, visible_ids)
    visible_signal_ids = {
        finding.get("signal_id")
        for finding in packet.get("tool_findings", [])
        if finding.get("signal_id")
    }
    visible_signal_ids.update(
        signal_id
        for rule in packet.get("rules", [])
        for signal_id in rule.get("related_signal_ids", [])
    )
    visible_signal_ids.update(packet.get("candidate_summary", {}).get("supporting_signal_ids", []))
    visible_tool_result_ids = {
        finding.get("tool_result_id")
        for finding in packet.get("tool_findings", [])
        if finding.get("tool_result_id")
    }
    for signal in assessment["validated_signals"]:
        if not isinstance(signal, dict):
            raise ValueError("schema_mismatch")
        required_signal_fields = {"signal_id", "status", "support_level", "cited_evidence_refs"}
        optional_signal_fields = {"tool_result_id"}
        signal_fields = set(signal.keys())
        if not required_signal_fields <= signal_fields or not signal_fields <= required_signal_fields | optional_signal_fields:
            raise ValueError("schema_mismatch")
        if signal["status"] not in ALLOWED_SIGNAL_STATUSES:
            raise ValueError("invalid_signal_status")
        if signal["support_level"] not in ALLOWED_SUPPORT_LEVELS:
            raise ValueError("invalid_support_level")
        if not isinstance(signal["cited_evidence_refs"], list) or not signal["cited_evidence_refs"]:
            raise ValueError("schema_mismatch")
        if signal.get("signal_id") and signal["signal_id"] not in visible_signal_ids:
            raise ValueError("invalid_evidence_ids")
        if signal.get("tool_result_id") and signal["tool_result_id"] not in visible_tool_result_ids:
            raise ValueError("invalid_evidence_ids")
        for evidence_ref in signal.get("cited_evidence_refs", []):
            _validate_evidence_ref(evidence_ref, visible_ids)


def _validate_evidence_ref(evidence_ref: Any, visible_ids: set[str]) -> None:
    if not isinstance(evidence_ref, dict):
        raise ValueError("schema_mismatch")
    if set(evidence_ref.keys()) != {"evidence_ref_type", "ref_id", "role"}:
        raise ValueError("schema_mismatch")
    if evidence_ref["evidence_ref_type"] not in ALLOWED_EVIDENCE_REF_TYPES:
        raise ValueError("invalid_evidence_ref_type")
    if evidence_ref["role"] not in ALLOWED_EVIDENCE_REF_ROLES:
        raise ValueError("invalid_evidence_ref_role")
    if evidence_ref["ref_id"] not in visible_ids:
        raise ValueError("invalid_evidence_ids")


def _visible_packet_evidence_ids(packet: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    ids.update(rule["rule_id"] for rule in packet.get("rules", []) if rule.get("rule_id"))
    for finding in packet.get("tool_findings", []):
        if finding.get("tool_result_id"):
            ids.add(finding["tool_result_id"])
        for ref in finding.get("evidence_refs", []):
            if ref.get("ref_id"):
                ids.add(ref["ref_id"])
    for row in packet.get("relevant_table_rows", []):
        report_id = row.get("report_id")
        for key in ("row_id", "local_evidence_id"):
            if row.get(key):
                if report_id:
                    ids.add(f"{report_id}:{row[key]}")
        for value in row.get("values", {}).values():
            if isinstance(value, dict) and value.get("cell_id"):
                if report_id:
                    ids.add(f"{report_id}:{value['cell_id']}")
    for note in packet.get("relevant_notes", []):
        report_id = note.get("report_id")
        for key in ("note_id", "local_evidence_id"):
            if note.get(key):
                if report_id:
                    ids.add(f"{report_id}:{note[key]}")
    for span in packet.get("relevant_variance_explanations", []):
        report_id = span.get("report_id")
        for key in ("span_id", "local_evidence_id"):
            if span.get(key):
                if report_id:
                    ids.add(f"{report_id}:{span[key]}")
    return ids


def _validate_detector_packet(packet: dict[str, Any]) -> None:
    required_fields = {
        "packet_id",
        "candidate_id",
        "report_id",
        "task",
        "metadata",
        "candidate_summary",
        "relevant_table_rows",
        "relevant_notes",
        "relevant_variance_explanations",
        "tool_findings",
        "rules",
        "constraints",
    }
    missing = required_fields - packet.keys()
    if missing:
        raise ValueError("invalid_detector_packet")
    if packet["task"].get("risk_category") in {"no_material_irregularity_signal", "insufficient_evidence"}:
        raise ValueError("invalid_detector_packet")
    if _contains_prohibited_detector_visible_payload(packet):
        raise ValueError("invalid_detector_packet")


def _contains_prohibited_detector_visible_payload(value: Any) -> bool:
    prohibited_keys = {
        "raw_ocr_text",
        "full_raw_ocr_text",
        "raw_tables",
        "raw_pdf_coordinates",
        "raw_coordinates",
        "coordinates",
        "bbox",
        "bounding_box",
        "cache_record_id",
        "cache_key",
        "source_file_sha256",
        "normalized_text_hash",
        "table_content_hash",
        "hidden_injection_details",
        "hidden_reasoning",
        "chain_of_thought",
        "hidden_chain_of_thought",
        "omitted_evidence_ids",
        "external_context",
        "outside_context",
    }
    if isinstance(value, dict):
        return any(
            key in prohibited_keys or _contains_prohibited_detector_visible_payload(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_prohibited_detector_visible_payload(item) for item in value)
    return False


def _score_predictions(predictions: list[dict[str, Any]], *, mode: str) -> dict[str, Any]:
    valid_predictions = [
        prediction
        for prediction in predictions
        if prediction.get("prediction_status") == "accepted"
    ]
    invalid_count = sum(1 for prediction in predictions if prediction.get("prediction_status") == "invalid")
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    exact_match_count = 0
    false_positive_total = 0
    false_positive_correct_or_conservative = 0
    insufficient_gold_total = 0
    insufficient_predicted_total = 0
    insufficient_true_positive = 0
    for prediction in predictions:
        gold = prediction["gold_support_level"]
        if gold == "insufficient_evidence":
            insufficient_gold_total += 1
        if gold in {"not_supported", "insufficient_evidence"}:
            false_positive_total += 1
            if prediction.get("prediction_status") == "invalid":
                continue
        if prediction.get("prediction_status") != "accepted":
            continue
        predicted = prediction["predicted_support_level"]
        confusion[gold][predicted] += 1
        if gold == predicted:
            exact_match_count += 1
        if gold in {"not_supported", "insufficient_evidence"} and predicted in {
            "not_supported",
            "insufficient_evidence",
            "weakly_supported",
        }:
            false_positive_correct_or_conservative += 1
        if predicted == "insufficient_evidence":
            insufficient_predicted_total += 1
        if gold == "insufficient_evidence" and predicted == "insufficient_evidence":
            insufficient_true_positive += 1
    invalid_gold_false_positive_total = sum(
        1
        for prediction in predictions
        if prediction.get("prediction_status") == "invalid"
        and prediction["gold_support_level"] in {"not_supported", "insufficient_evidence"}
    )
    invalid_gold_insufficient_total = sum(
        1
        for prediction in predictions
        if prediction.get("prediction_status") == "invalid"
        and prediction["gold_support_level"] == "insufficient_evidence"
    )
    return {
        "mode": mode,
        "num_examples": len(predictions),
        "num_attempted": 0 if mode == "dry_run" else len(predictions),
        "num_valid_predictions": len(valid_predictions),
        "num_invalid_responses": invalid_count,
        "schema_valid_count": sum(1 for prediction in predictions if prediction.get("schema_valid")),
        "evidence_id_valid_count": sum(1 for prediction in predictions if prediction.get("evidence_valid")),
        "support_level_exact_match_count": exact_match_count,
        "false_positive_rejection": {
            "correct_or_conservative": false_positive_correct_or_conservative,
            "invalid_gold_count": invalid_gold_false_positive_total,
            "total": false_positive_total,
        },
        "insufficient_evidence_detection": {
            "true_positive": insufficient_true_positive,
            "predicted_total": insufficient_predicted_total,
            "gold_total": insufficient_gold_total,
            "invalid_gold_count": invalid_gold_insufficient_total,
        },
        "accusation_language_violation_count": sum(
            1
            for prediction in predictions
            if "prohibited_risk_language" in prediction.get("invalid_reason_codes", [])
        ),
        "outside_packet_reasoning_violation_count": sum(
            1
            for prediction in predictions
            if "outside_packet_reasoning" in prediction.get("invalid_reason_codes", [])
        ),
        "support_level_confusion_matrix_counts": {
            gold: dict(predicted_counts)
            for gold, predicted_counts in sorted(confusion.items())
        },
    }


def _sentence_count(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    count = 0
    for index, char in enumerate(stripped):
        if char not in ".!?":
            continue
        previous_char = stripped[index - 1] if index > 0 else ""
        next_char = stripped[index + 1] if index + 1 < len(stripped) else ""
        if char == "." and previous_char.isdigit() and next_char.isdigit():
            continue
        count += 1
    if count == 0 or stripped[-1] not in ".!?":
        count += 1
    return count


def _contains_prohibited_text(value: Any) -> bool:
    text = json.dumps(value, ensure_ascii=False).lower()
    prohibited = ["fraud", "manipulat", "conceal", "intent", "illegal", "legal misstatement"]
    return any(term in text for term in prohibited)


def _default_live_client(provider: str) -> ApiLlmDetectorClient:
    _load_dotenv_if_available()
    if provider == "openrouter":
        api_key = os.environ.get(OPENROUTER_API_KEY_ENV)
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required for live mode")
        return OpenRouterApiLlmDetectorClient(api_key=api_key)
    if provider == "deepseek":
        api_key = os.environ.get(DEEPSEEK_API_KEY_ENV)
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is required for DeepSeek live mode")
        return DeepSeekApiLlmDetectorClient(api_key=api_key)
    raise ValueError(f"live mode currently supports provider=openrouter or provider=deepseek, got {provider}")


def _load_dotenv_if_available() -> None:
    dotenv_path = Path(".env")
    if not dotenv_path.exists():
        return
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


def _detector_system_prompt(prompt_version: str = DEFAULT_PROMPT_VERSION) -> str:
    _validate_prompt_version(prompt_version)
    base_prompt = (
        "You are a specialized accounting irregularity risk-signal detector for Vietnamese financial reports. "
        "Return a single JSON object with these top-level keys: assessment_id, packet_id, candidate_id, "
        "report_id, risk_category, support_level, confidence, severity, validated_signals, "
        "cited_evidence_refs, rationale_short. "
        "Do not wrap the object in assessment, type, data, or any other envelope. "
        "Do not use decision, decision_rationale, signal_ids, or evidence_ids_used fields. "
        "Use only evidence provided in the DetectorPacket. "
        "Use exact ref_id strings from DetectorPacket.tool_findings[].evidence_refs, DetectorPacket.rules[].rule_id, "
        "DetectorPacket.tool_findings[].tool_result_id, and canonical packet evidence IDs formed as "
        "report_id:cell_id, report_id:row_id, report_id:note_id, or report_id:span_id from visible packet rows, "
        "notes, and variance explanations. "
        "Do not use local_evidence_id, row_id, note_id, span_id, or cell_id by itself. "
        "Use only these evidence_ref_type values: table_cell, table_row, note, note_span, "
        "variance_explanation_span, accounting_policy_note_span, related_party_note_span, tool_result, rule. "
        "Use only these evidence roles: supporting, contradicting, refuting, context, missing_required_context. "
        "Use only these validated_signals status values: validated, partially_validated, rejected, not_assessable. "
        "Confidence must be a number between 0 and 1, never a string label. "
        "Do not use outside knowledge. "
        "Do not use hidden metadata. "
        "Do not claim fraud, manipulation, intent, concealment, or legal misstatement. "
        "Use risk-signal language only. "
        "Keep rationale_short to 1-3 sentences."
    )
    evidence_guard = (
        " For every cited_evidence_refs item, ref_id must be exactly one of: "
        "DetectorPacket.tool_findings[].evidence_refs[].ref_id, DetectorPacket.rules[].rule_id, "
        "DetectorPacket.tool_findings[].tool_result_id, or a canonical report-prefixed evidence ID from visible "
        "packet rows, notes, or variance explanations. Never put signal_id, candidate_summary, "
        "metric_name, row_id, note_id, span_id, local_evidence_id, cell_id, report_id, or tool_name by itself into ref_id. "
        "Use fewer cited_evidence_refs rather than inventing refs."
    )
    support_calibration = (
        " Use supported only when packet-visible evidence directly and completely supports the candidate signal. "
        "Use weakly_supported when evidence points toward the signal but is partial, indirect, ambiguous, "
        "or mainly tool-derived without enough corroborating packet evidence. "
        "Use insufficient_evidence when required packet evidence is missing. "
        "Use not_supported when packet-visible evidence contradicts the signal."
    )
    if prompt_version == PROMPT_VERSION_BASE:
        return base_prompt
    if prompt_version == PROMPT_VERSION_EVIDENCE_GUARD:
        return base_prompt + evidence_guard
    if prompt_version == PROMPT_VERSION_SUPPORT_CALIBRATED:
        return base_prompt + evidence_guard + support_calibration
    raise ValueError(f"unsupported prompt_version: {prompt_version}")


def _default_run_id(provider: str, model: str) -> str:
    return f"{provider}__{_slug(model)}"


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value.strip().lower()).strip("_")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the API LLM detector baseline.")
    parser.add_argument("--release-dir", default=str(DEFAULT_RELEASE_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--mode", choices=["dry_run", "fake", "live"], required=True)
    parser.add_argument("--provider", default="openrouter")
    parser.add_argument("--model")
    parser.add_argument("--prompt-version", default=DEFAULT_PROMPT_VERSION)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--run-id")
    parser.add_argument("--sweep-id")
    parser.add_argument("--sweep-preset", default=DEFAULT_SWEEP_PRESET, choices=sorted(SWEEP_PRESETS))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--example-id", action="append", dest="example_ids")
    args = parser.parse_args(argv)
    if args.sweep_id:
        sweep_result = run_api_llm_detector_baseline_sweep(
            release_dir=args.release_dir,
            output_root=args.output_root,
            mode=args.mode,
            provider=args.provider,
            model=args.model,
            sweep_id=args.sweep_id,
            preset=args.sweep_preset,
            limit=args.limit,
            example_ids=args.example_ids,
        )
        print(
            json.dumps(
                {"status": sweep_result.status, "sweep_dir": str(sweep_result.sweep_dir)},
                sort_keys=True,
            )
        )
        return 0 if sweep_result.status == "passed" else 1
    result = run_api_llm_detector_baseline(
        release_dir=args.release_dir,
        output_root=args.output_root,
        mode=args.mode,
        provider=args.provider,
        model=args.model,
        prompt_version=args.prompt_version,
        temperature=args.temperature,
        run_id=args.run_id,
        limit=args.limit,
        example_ids=args.example_ids,
    )
    print(json.dumps({"status": result.status, "run_dir": str(result.run_dir)}, sort_keys=True))
    return 0 if result.status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
