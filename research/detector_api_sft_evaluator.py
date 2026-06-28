from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from research.api_llm_detector_baseline import (
    ApiLlmDetectorClient,
    ApiLlmDetectorRequest,
    ApiLlmDetectorResponse,
    DEFAULT_PROMPT_VERSION,
    DEFAULT_TEMPERATURE,
    DEEPSEEK_DIRECT_MODELS,
    PROMPT_VERSION_EVIDENCE_GUARD,
    SUPPORTED_PROMPT_VERSIONS,
    _default_live_client,
    _validate_live_provider_model,
    _validate_prompt_version,
)
from research.detector_contract_validation import (
    parse_and_validate_detector_assessment,
)
from research.detector_sft_evaluator import (
    SUPPORTED_EVAL_SPLITS,
    _base_prediction_record,
    _canonical_json,
    _error_reason_code,
    _score_predictions,
    _write_json,
    _write_jsonl,
)
from research.detector_sft_trainer import (
    CANONICAL_SFT_JSONL,
    load_detector_sft_dataset,
)


DEFAULT_OUTPUT_ROOT = Path("artifacts/detector_ablation/deepseek_api_sft_validation")
DEFAULT_PROVIDER = "deepseek"
DEFAULT_MODEL = "deepseek-v4-flash"


@dataclass(frozen=True)
class Wave4DetectorApiSftEvalResult:
    status: str
    output_root: Path
    predictions_path: Path
    metrics_path: Path
    manifest_path: Path
    invalid_responses_path: Path
    records_loaded: int
    predictions_written: int
    invalid_response_count: int
    errors: list[str]


@dataclass(frozen=True)
class SftGoldFakeDetectorClient:
    responses_by_example_id: dict[str, dict[str, Any]]

    def complete(self, request: ApiLlmDetectorRequest) -> ApiLlmDetectorResponse:
        return ApiLlmDetectorResponse(
            content=json.dumps(self.responses_by_example_id[request.example_id], ensure_ascii=False),
            provider_metadata={
                "provider": request.provider,
                "strategy": "sft_gold_fake_client",
                "request_model": request.model,
                "prompt_version": request.prompt_version,
                "temperature": request.temperature,
            },
        )


def run_detector_api_sft_evaluation(
    *,
    sft_jsonl: Path | str = CANONICAL_SFT_JSONL,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    mode: Literal["fake", "live"] = "live",
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    prompt_version: str = PROMPT_VERSION_EVIDENCE_GUARD,
    temperature: float = DEFAULT_TEMPERATURE,
    split: Literal["validation", "test"] = "validation",
    limit: int | None = None,
    client: ApiLlmDetectorClient | None = None,
    allow_noncanonical_sft_jsonl: bool = False,
) -> Wave4DetectorApiSftEvalResult:
    if mode not in {"fake", "live"}:
        raise ValueError("mode must be fake or live")
    if provider != DEFAULT_PROVIDER:
        raise ValueError("API-on-SFT evaluation currently supports provider=deepseek only")
    _validate_live_provider_model(provider, model)
    _validate_prompt_version(prompt_version)
    if split not in SUPPORTED_EVAL_SPLITS:
        raise ValueError(f"split must be one of {sorted(SUPPORTED_EVAL_SPLITS)}")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided")

    output_path = Path(output_root)
    predictions_path = output_path / "predictions.jsonl"
    metrics_path = output_path / "metrics.json"
    manifest_path = output_path / "manifest.json"
    invalid_responses_path = output_path / "invalid_responses.jsonl"
    raw_invalid_responses_path = output_path / "debug" / "raw_invalid_responses.jsonl"

    dataset = load_detector_sft_dataset(
        sft_jsonl,
        allow_noncanonical_sft_jsonl=allow_noncanonical_sft_jsonl,
    )
    rows = dataset.validation_rows if split == "validation" else dataset.test_rows
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"no rows available for split={split}")

    gold_by_example_id: dict[str, dict[str, Any]] = {
        row.example_id: json.loads(row.messages[2]["content"])
        for row in rows
    }
    if mode == "fake" and client is None:
        client = SftGoldFakeDetectorClient(gold_by_example_id)
    if mode == "live" and client is None:
        client = _default_live_client(provider)

    predictions: list[dict[str, Any]] = []
    invalid_responses: list[dict[str, Any]] = []
    raw_invalid_responses: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        packet = json.loads(row.messages[1]["content"])
        gold = gold_by_example_id[row.example_id]
        prediction_record = _base_prediction_record(row.example_id, split, packet, gold)
        request = ApiLlmDetectorRequest(
            example_id=row.example_id,
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
            assessment = parse_and_validate_detector_assessment(raw_content, packet)
        except Exception as error:
            reason_code = _error_reason_code(error)
            prediction_record["prediction_status"] = "invalid"
            prediction_record["schema_valid"] = False
            prediction_record["evidence_valid"] = False
            prediction_record["invalid_reason_codes"] = [reason_code]
            invalid_responses.append(
                {
                    "example_id": row.example_id,
                    "packet_id": packet["packet_id"],
                    "candidate_id": packet["candidate_id"],
                    "failure_stage": "response_validation",
                    "reason_codes": [reason_code],
                    "raw_response_debug_path": "debug/raw_invalid_responses.jsonl",
                }
            )
            raw_invalid_responses.append(
                {
                    "example_id": row.example_id,
                    "packet_id": packet["packet_id"],
                    "candidate_id": packet["candidate_id"],
                    "raw_response_text": raw_content,
                    "reason_codes": [reason_code],
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
            prediction_record["predicted_severity"] = assessment["severity"]
            prediction_record["support_level_exact_match"] = assessment["support_level"] == gold["support_level"]
            prediction_record["severity_exact_match"] = assessment["severity"] == gold["severity"]
            prediction_record["assessment_exact_match"] = _canonical_json(assessment) == _canonical_json(gold)
        predictions.append(prediction_record)
        if index % 25 == 0:
            print(json.dumps({"evaluated": index, "total": len(rows)}, sort_keys=True), flush=True)

    metrics = _score_predictions(predictions, split=split)
    manifest = {
        "status": "passed",
        "variant": _api_variant_id(model),
        "model_variant": _api_variant_id(model),
        "mode": mode,
        "provider": provider,
        "model": model,
        "prompt_version": prompt_version,
        "decoding_config": {"temperature": temperature},
        "source_jsonl": str(sft_jsonl),
        "split": split,
        "limit": limit,
        "records_loaded": len(rows),
        "predictions": "predictions.jsonl",
        "metrics": "metrics.json",
        "invalid_responses": "invalid_responses.jsonl",
        "evaluation_policy": {
            "selection_split": split,
            "gold_source": "assistant message in official detector_sft_chat.jsonl",
            "test_split_used_for_training_or_selection": False,
            "predictions_are_evaluation_outputs_only": True,
            "synthetic_validation_evidence_only": True,
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
    }
    output_path.mkdir(parents=True, exist_ok=True)
    raw_invalid_responses_path.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(predictions_path, predictions)
    _write_json(metrics_path, metrics)
    _write_json(manifest_path, manifest)
    _write_jsonl(invalid_responses_path, invalid_responses)
    _write_jsonl(raw_invalid_responses_path, raw_invalid_responses)
    return Wave4DetectorApiSftEvalResult(
        status="passed",
        output_root=output_path,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        manifest_path=manifest_path,
        invalid_responses_path=invalid_responses_path,
        records_loaded=len(rows),
        predictions_written=len(predictions),
        invalid_response_count=len(invalid_responses),
        errors=[],
    )


def _api_variant_id(model: str) -> str:
    return "api_deepseek_" + model.replace("/", "_").replace("-", "_").replace(".", "_")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate DeepSeek API models on Wave 4 detector SFT validation/test rows."
    )
    parser.add_argument("--sft-jsonl", default=str(CANONICAL_SFT_JSONL))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--mode", choices=["fake", "live"], default="live")
    parser.add_argument("--model", choices=sorted(DEEPSEEK_DIRECT_MODELS), default=DEFAULT_MODEL)
    parser.add_argument("--prompt-version", choices=sorted(SUPPORTED_PROMPT_VERSIONS), default=DEFAULT_PROMPT_VERSION)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--split", choices=sorted(SUPPORTED_EVAL_SPLITS), default="validation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow-noncanonical-sft-jsonl", action="store_true")
    args = parser.parse_args(argv)
    result = run_detector_api_sft_evaluation(
        sft_jsonl=args.sft_jsonl,
        output_root=args.output_root,
        mode=args.mode,
        model=args.model,
        prompt_version=args.prompt_version,
        temperature=args.temperature,
        split=args.split,
        limit=args.limit,
        allow_noncanonical_sft_jsonl=args.allow_noncanonical_sft_jsonl,
    )
    print(
        json.dumps(
            {
                "status": result.status,
                "output_root": str(result.output_root),
                "records_loaded": result.records_loaded,
                "invalid_response_count": result.invalid_response_count,
                "errors": result.errors,
            },
            sort_keys=True,
        )
    )
    return 0 if result.status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
