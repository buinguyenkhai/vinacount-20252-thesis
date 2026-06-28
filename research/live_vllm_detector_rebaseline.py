from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from research.detector_hybrid_evaluator import _hybrid_prediction
from research.detector_sft_evaluator import (
    _load_evaluation_rows,
    _score_predictions,
)
from vinacount.detector_contract import parse_and_validate_detector_assessment
from vinacount.env_loader import load_dotenv_if_available
from vinacount.runtime_detector_modes import (
    DEFAULT_SFT_CONTRACT_GUARD_VERSION,
    DEFAULT_SFT_DETECTOR_LABEL,
    DEFAULT_SFT_SYSTEM_PROMPT_VERSION,
    DEFAULT_SFT_VLLM_MODEL,
    SFT_VLLM_API_KEY_ENV,
    SFT_VLLM_BASE_URL_ENV,
    SFT_VLLM_MODEL_ENV,
    _optional_env,
    _sft_system_prompt,
    canonical_sft_detector_packet_json,
    canonical_sft_detector_packet_payload,
    render_qwen_no_think_messages,
)


BASE_MODEL_ID = "unsloth/Qwen3.5-4B"
DEFAULT_BASE_URL = "http://127.0.0.1:18000/v1"
DEFAULT_OUTPUT_ROOT = Path("artifacts/live_vllm_detector_rebaseline")
DEFAULT_SYNTHETIC_SOURCE = Path("data/detector_sft/synthetic_detector_dataset/detector_sft_chat.jsonl")
DEFAULT_CALIBRATED_SOURCE = Path("artifacts/detector_corroboration_calibration/sft_v3_balanced/detector_sft_chat.jsonl")
DEFAULT_REAL_MANUAL_SOURCE = Path("data/real_manual/combined_real_manual_validation_release/validation.jsonl")


def run_live_vllm_detector_rebaseline(
    *,
    source_jsonl: Path,
    source_format: str,
    split: str,
    output_root: Path,
    model_alias: str,
    checkpoint_dir: Path | None,
    checkpoint_identity: str,
    evaluation_role: str,
    base_url: str,
    api_key: str | None,
    vllm_version: str | None,
    max_tokens: int,
    batch_size: int,
    timeout_seconds: float,
    system_prompt_version: str,
) -> dict[str, Any]:
    rows, source_split_counts = _load_evaluation_rows(
        sft_jsonl=source_jsonl,
        split=split,
        source_format=source_format,
        allow_noncanonical_sft_jsonl=True,
        system_prompt_version=system_prompt_version,
        packet_evidence_role_enrichment=True,
    )
    if not rows:
        raise ValueError(f"no rows available for {source_jsonl} split={split}")
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")

    output_root.mkdir(parents=True, exist_ok=True)
    raw_path = output_root / "raw_completions_private.jsonl"
    unguarded_path = output_root / "predictions_unguarded_private.jsonl"
    guarded_path = output_root / "predictions_guarded_private.jsonl"
    unguarded_metrics_path = output_root / "metrics_unguarded.json"
    guarded_metrics_path = output_root / "metrics_guarded.json"
    manifest_path = output_root / "manifest.json"
    summary_path = output_root / "summary_public_safe.json"

    client = _VllmCompletionsClient(
        base_url=base_url,
        model=model_alias,
        api_key=api_key,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
    )
    prompts: list[str] = []
    source_records: list[dict[str, Any]] = []
    for row in rows:
        packet = canonical_sft_detector_packet_payload(json.loads(row.messages[1]["content"]))
        gold = json.loads(row.messages[2]["content"])
        prompt = render_qwen_no_think_messages(
            [
                {"role": "system", "content": _sft_system_prompt(system_prompt_version)},
                {"role": "user", "content": canonical_sft_detector_packet_json(packet)},
            ]
        )
        prompts.append(prompt)
        source_records.append(_source_record(row, packet=packet, gold=gold, split=split))

    raw_records: list[dict[str, Any]] = []
    for start in range(0, len(prompts), batch_size):
        completions = client.complete_batch(prompts[start : start + batch_size])
        for offset, raw_completion in enumerate(completions):
            source_record = source_records[start + offset]
            raw_records.append(
                {
                    "example_id": source_record["example_id"],
                    "split": source_record["split"],
                    "packet_id": source_record["packet_id"],
                    "risk_category": source_record["risk_category"],
                    "gold_support_level": source_record["gold_support_level"],
                    "gold_severity": source_record["gold_severity"],
                    "model_visible_input": source_record["model_visible_input"],
                    "raw_completion": raw_completion,
                }
            )
        print(json.dumps({"completed": min(start + batch_size, len(prompts)), "total": len(prompts)}), flush=True)

    unguarded_predictions = [_unguarded_prediction(record) for record in raw_records]
    guarded_predictions = [_hybrid_prediction(record) for record in raw_records]
    unguarded_metrics = _augment_metrics(
        _score_predictions(unguarded_predictions, split=split),
        predictions=unguarded_predictions,
    )
    guarded_metrics = _augment_metrics(
        _score_predictions(guarded_predictions, split=split),
        predictions=guarded_predictions,
    )
    checkpoint_manifest = _checkpoint_manifest(checkpoint_dir)
    manifest = {
        "status": "passed",
        "issue": 182,
        "evaluation_name": output_root.name,
        "evaluation_role": evaluation_role,
        "source_jsonl": str(source_jsonl),
        "source_format": source_format,
        "source_split_counts": source_split_counts,
        "split": split,
        "num_examples": len(rows),
        "base_model": BASE_MODEL_ID,
        "checkpoint_identity": checkpoint_identity,
        "checkpoint": checkpoint_manifest,
        "served_model_alias": model_alias,
        "vllm_version": vllm_version or "unknown",
        "dtype_precision": "fp16/bfloat16 permissioned Triton/vLLM host",
        "context_length": 8192,
        "endpoint": {
            "mode": "/v1/completions",
            "base_url": "redacted_private_tunnel",
            "api_key_recorded": False,
        },
        "prompt": {
            "system_prompt_version": system_prompt_version,
            "qwen_no_thinking_rendering": "explicit_chat_prompt_with_empty_think_block",
            "enable_thinking": False,
        },
        "packet_representation": "compact_canonical_detector_packet_json_with_report_set_id",
        "decoding": {
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "batch_size": batch_size,
            "timeout_seconds": timeout_seconds,
            "provider_retries": 0,
        },
        "guard_policy": {
            "version": DEFAULT_SFT_CONTRACT_GUARD_VERSION,
            "semantic_label_overrides": False,
            "guarded_and_unguarded_derived_from_same_raw_completions": True,
        },
        "private_artifacts": {
            "raw_completions": raw_path.name,
            "unguarded_predictions": unguarded_path.name,
            "guarded_predictions": guarded_path.name,
        },
        "public_artifacts": {
            "unguarded_metrics": unguarded_metrics_path.name,
            "guarded_metrics": guarded_metrics_path.name,
            "summary": summary_path.name,
        },
        "public_safety": {
            "no_private_endpoint": True,
            "no_api_key": True,
            "no_prompts": True,
            "no_raw_ocr": True,
            "no_provider_bodies_in_public_summary": True,
        },
    }
    summary = {
        "evaluation_name": output_root.name,
        "evaluation_role": evaluation_role,
        "model": f"{DEFAULT_SFT_DETECTOR_LABEL} ({checkpoint_identity})",
        "served_model_alias": model_alias,
        "split": split,
        "num_examples": len(rows),
        "unguarded": _compact_metric_summary(unguarded_metrics),
        "guarded": _compact_metric_summary(guarded_metrics),
    }

    _write_jsonl(raw_path, raw_records)
    _write_jsonl(unguarded_path, unguarded_predictions)
    _write_jsonl(guarded_path, guarded_predictions)
    _write_json(unguarded_metrics_path, unguarded_metrics)
    _write_json(guarded_metrics_path, guarded_metrics)
    _write_json(manifest_path, manifest)
    _write_json(summary_path, summary)
    return {
        "status": "passed",
        "output_root": str(output_root),
        "manifest": str(manifest_path),
        "summary": summary,
    }


class _VllmCompletionsClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None,
        max_tokens: int,
        timeout_seconds: float,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds

    def complete_batch(self, prompts: list[str]) -> list[str]:
        payload = {
            "model": self.model,
            "prompt": prompts,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8", errors="replace"))
        except urllib.error.URLError as error:
            raise RuntimeError(f"vllm_completion_failed: {error}") from error
        choices = body.get("choices") if isinstance(body, dict) else None
        if not isinstance(choices, list):
            raise ValueError("vllm_completions_response_invalid")
        completions: list[str | None] = [None] * len(prompts)
        for choice in choices:
            if isinstance(choice, dict) and isinstance(choice.get("index"), int):
                completions[choice["index"]] = choice.get("text", "")
        if any(completion is None for completion in completions):
            raise ValueError("vllm_completions_response_missing_choice")
        return [completion or "" for completion in completions]


def _source_record(row: Any, *, packet: dict[str, Any], gold: dict[str, Any], split: str) -> dict[str, Any]:
    record = {
        "example_id": row.example_id,
        "split": split,
        "packet_id": packet["packet_id"],
        "risk_category": packet["task"]["risk_category"],
        "gold_support_level": gold["support_level"],
        "gold_severity": gold["severity"],
        "model_visible_input": {"type": "DetectorPacket", "data": packet},
    }
    for key in ("report_profile", "source_type", "company_key", "period_key"):
        value = row.metadata.get(key) if isinstance(row.metadata, dict) else None
        if value is not None:
            record[key] = value
    return record


def _unguarded_prediction(source_record: dict[str, Any]) -> dict[str, Any]:
    prediction = {
        "example_id": source_record["example_id"],
        "split": source_record["split"],
        "packet_id": source_record["packet_id"],
        "risk_category": source_record["risk_category"],
        "gold_support_level": source_record["gold_support_level"],
        "gold_severity": source_record["gold_severity"],
        "prediction_status": "not_attempted",
        "schema_valid": False,
        "evidence_valid": False,
        "model_visible_input": source_record["model_visible_input"],
        "raw_completion": source_record.get("raw_completion"),
        "guard_actions": [],
    }
    raw_completion = source_record.get("raw_completion")
    if not isinstance(raw_completion, str) or not raw_completion.strip():
        prediction["prediction_status"] = "invalid"
        prediction["invalid_reason_codes"] = ["missing_raw_completion"]
        return prediction
    try:
        assessment = parse_and_validate_detector_assessment(raw_completion, source_record["model_visible_input"]["data"])
    except Exception as error:
        prediction["prediction_status"] = "invalid"
        prediction["invalid_reason_codes"] = [_error_reason_code(error)]
        return prediction
    prediction["prediction_status"] = "accepted"
    prediction["schema_valid"] = True
    prediction["evidence_valid"] = True
    prediction["prediction"] = assessment
    prediction["predicted_support_level"] = assessment["support_level"]
    prediction["predicted_severity"] = assessment["severity"]
    prediction["support_level_exact_match"] = assessment["support_level"] == source_record["gold_support_level"]
    prediction["severity_exact_match"] = assessment["severity"] == source_record["gold_severity"]
    prediction["assessment_exact_match"] = False
    return prediction


def _error_reason_code(error: Exception) -> str:
    message = str(error).strip()
    return message or error.__class__.__name__


def _augment_metrics(metrics: dict[str, Any], *, predictions: list[dict[str, Any]]) -> dict[str, Any]:
    metrics["guard_action_counts"] = dict(
        sorted(Counter(action for prediction in predictions for action in prediction.get("guard_actions", [])).items())
    )
    scenario_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for prediction in predictions:
        family = _scenario_family(prediction)
        if not family:
            continue
        scenario_counts[family]["total"] += 1
        if prediction.get("support_level_exact_match"):
            scenario_counts[family]["support_exact"] += 1
        if prediction.get("prediction_status") == "accepted":
            scenario_counts[family]["valid"] += 1
    if scenario_counts:
        metrics["scenario_family_analysis"] = {key: dict(value) for key, value in sorted(scenario_counts.items())}
    return metrics


def _scenario_family(prediction: dict[str, Any]) -> str | None:
    packet = prediction.get("model_visible_input", {}).get("data", {})
    candidate_summary = packet.get("candidate_summary") if isinstance(packet, dict) else None
    if isinstance(candidate_summary, dict) and isinstance(candidate_summary.get("scenario_family"), str):
        return candidate_summary["scenario_family"]
    return prediction.get("risk_category")


def _compact_metric_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    weak = metrics.get("weakly_supported_detection", {})
    return {
        "valid": f"{metrics.get('num_valid_predictions')}/{metrics.get('num_examples')}",
        "support_exact": f"{metrics.get('support_level_exact_match_count')}/{metrics.get('num_examples')}",
        "support_accuracy": metrics.get("support_level_exact_match_rate"),
        "support_macro_f1": metrics.get("support_level_macro_f1"),
        "weak_f1": weak.get("f1"),
        "weak_recall": weak.get("recall"),
        "false_positive_rejection": metrics.get("false_positive_rejection", {}).get("rate"),
        "severity_exact": f"{metrics.get('severity_exact_match_count')}/{metrics.get('num_examples')}",
    }


def _checkpoint_manifest(checkpoint_dir: Path | None) -> dict[str, Any]:
    if checkpoint_dir is None:
        return {"identity_hash_sha256": None, "files_hashed": []}
    if not checkpoint_dir.exists():
        return {"identity_hash_sha256": None, "missing": True, "files_hashed": []}
    selected_files = [
        path
        for path in sorted(checkpoint_dir.iterdir())
        if path.is_file()
        and path.name
        in {
            "adapter_config.json",
            "adapter_model.safetensors",
            "tokenizer_config.json",
            "trainer_state.json",
            "README.md",
        }
    ]
    digest = hashlib.sha256()
    file_records = []
    for path in selected_files:
        file_hash = _sha256_file(path)
        digest.update(path.name.encode("utf-8"))
        digest.update(file_hash.encode("ascii"))
        file_records.append({"name": path.name, "sha256": file_hash, "bytes": path.stat().st_size})
    return {
        "identity_hash_sha256": digest.hexdigest() if file_records else None,
        "files_hashed": file_records,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _query_vllm_version(base_url: str, api_key: str | None, timeout_seconds: float) -> str | None:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        base_url.rstrip("/").removesuffix("/v1") + "/version",
        headers=headers,
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8", errors="replace"))
    except Exception:
        return None
    version = body.get("version") if isinstance(body, dict) else None
    return version if isinstance(version, str) else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run issue #182 live-vLLM detector rebaseline.")
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--source-format", choices=["sft_chat", "detector_record"], required=True)
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-identity", required=True)
    parser.add_argument("--evaluation-role", required=True)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--system-prompt-version", default=DEFAULT_SFT_SYSTEM_PROMPT_VERSION)
    args = parser.parse_args(argv)

    load_dotenv_if_available()
    base_url = args.base_url or os.environ.get(SFT_VLLM_BASE_URL_ENV, DEFAULT_BASE_URL)
    api_key = _optional_env(SFT_VLLM_API_KEY_ENV)
    result = run_live_vllm_detector_rebaseline(
        source_jsonl=args.source_jsonl,
        source_format=args.source_format,
        split=args.split,
        output_root=args.output_root,
        model_alias=args.model or os.environ.get(SFT_VLLM_MODEL_ENV, DEFAULT_SFT_VLLM_MODEL),
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_identity=args.checkpoint_identity,
        evaluation_role=args.evaluation_role,
        base_url=base_url,
        api_key=api_key,
        vllm_version=_query_vllm_version(base_url, api_key, timeout_seconds=10),
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        timeout_seconds=args.timeout_seconds,
        system_prompt_version=args.system_prompt_version,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
