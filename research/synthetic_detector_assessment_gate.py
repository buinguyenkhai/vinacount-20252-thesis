from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import multiprocessing
import os
import queue
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, TextIO

from research.detector_contract_validation import (
    detector_assessment_json_schema,
    parse_and_validate_detector_assessment,
    validate_detector_packet,
    visible_packet_evidence_ids,
)

DEFAULT_OUTPUT_ROOT = Path("artifacts/synthetic_detector_assessment_gate")
DEFAULT_BEST_OF_N = 4
ARTIFACT_CONTRACT_VERSION = "synthetic_detector_assessment_gate_v1"
VALID_MODES = {"dry_run", "fake", "live"}
VALID_JUDGE_DECISIONS = {"keep", "revise", "reject", "needs_human_review"}
VALID_JUDGE_STRATEGIES = {"non_batched_v1", "batched_per_packet_v1"}
PROMOTION_JUDGE_SCORE_THRESHOLD = 0.90
PROMOTION_WEAK_TARGET_MATCH_JUDGE_SCORE_THRESHOLD = 0.70
PROMOTION_HARD_JUDGE_DIMENSIONS = {
    "schema_correctness",
    "evidence_grounding",
    "risk_category_consistency",
    "validated_signal_correctness",
    "risk_language_compliance",
    "hidden_metadata_independence",
}
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"
DEEPSEEK_CHAT_COMPLETIONS_URL = "https://api.deepseek.com/chat/completions"
APPROVED_ISSUE_149_RAW_INPUT = Path(
    "artifacts/synthetic_raw_release_candidate/raw/synthetic_injected_raw.jsonl"
)
APPROVED_ISSUE_149_RAW_INPUT_SHA256 = "0cf3010b45fe93a45c8fc4ffe67c14b8d13d16ae57207c266c2c97f9dc4b66ac"
APPROVED_ISSUE_150_FULL_RUN_RECORDS = 8
LIVE_RUN_PURPOSES = {
    "capability_preflight",
    "temperature_probe",
    "approved_live",
    "approved_issue151_canary",
    "approved_issue151_standard",
    "approved_corroboration_calibration",
    "provider_smoke",
}
VALID_THINKING_MODES = {"disabled", "enabled"}
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
    "order": ["morph", "akashml", "deepinfra", "parasail"],
    "only": ["morph", "akashml", "deepinfra", "parasail"],
    "allow_fallbacks": True,
    "require_parameters": True,
}
OPENROUTER_PROVIDER_ROTATION_PASSES = 4
OPENROUTER_TEACHER_MAX_TOKENS = 2048
OPENROUTER_JUDGE_MAX_TOKENS = 2048
DEEPSEEK_TEACHER_MAX_TOKENS = 2048
DEEPSEEK_JUDGE_MAX_TOKENS = 2048
APPROVED_ISSUE151_STANDARD_RECORDS = 100
APPROVED_ISSUE151_STANDARD_RECORD_OPTIONS = {80, 100, 200}
APPROVED_CORROBORATION_CALIBRATION_RECORDS = 76
DEFAULT_TEACHER_PROMPT_VERSION = "synthetic_detector_assessment_teacher_v2"
DEFAULT_JUDGE_PROMPT_VERSION = "synthetic_detector_assessment_judge_v2"
LEGACY_TEACHER_PROMPT_VERSION = "synthetic_detector_assessment_teacher_v1"
LEGACY_JUDGE_PROMPT_VERSION = "synthetic_detector_assessment_judge_v1"
WEAKCAL_TEACHER_PROMPT_VERSION = "synthetic_detector_assessment_teacher_v2_weakcal"
WEAKCAL_JUDGE_PROMPT_VERSION = "synthetic_detector_assessment_judge_v2_weakcal"
CORROBORATION_TEACHER_PROMPT_VERSION = "synthetic_detector_assessment_teacher_v3_corroboration"
CORROBORATION_JUDGE_PROMPT_VERSION = "synthetic_detector_assessment_judge_v3_corroboration"
CORROBORATION_STRICT_TEACHER_PROMPT_VERSION = "synthetic_detector_assessment_teacher_v4_corroboration_strict"
CORROBORATION_STRICT_JUDGE_PROMPT_VERSION = "synthetic_detector_assessment_judge_v4_corroboration_strict"
CORROBORATION_BALANCED_TEACHER_PROMPT_VERSION = (
    "synthetic_detector_assessment_teacher_v5_corroboration_balanced"
)
CORROBORATION_BALANCED_JUDGE_PROMPT_VERSION = (
    "synthetic_detector_assessment_judge_v5_corroboration_balanced"
)


@dataclass(frozen=True)
class SyntheticAssessmentGateResult:
    status: str
    run_dir: Path
    manifest_path: Path
    candidates_path: Path
    filtered_path: Path
    non_kept_path: Path
    metrics_path: Path
    records_loaded: int
    candidates_written: int
    promoted_count: int
    errors: list[str]


@dataclass(frozen=True)
class SyntheticAssessmentGateResponse:
    content: str
    provider_metadata: dict | None = None


@dataclass(frozen=True)
class SyntheticAssessmentTeacherRequest:
    example_id: str
    packet: dict[str, Any]
    best_of_n: int
    provider: str
    model: str
    prompt_version: str
    seed: int | None
    temperature: float
    thinking: str
    top_p: float | None = None
    repetition_penalty: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    candidate_seeds: tuple[int | None, ...] = ()
    progress: Callable[[str], None] | None = None


@dataclass(frozen=True)
class SyntheticAssessmentJudgeRequest:
    example_id: str
    packet: dict[str, Any]
    assessment: dict[str, Any]
    provider: str
    model: str
    rubric_version: str
    temperature: float | None
    thinking: str
    prompt_version: str = DEFAULT_JUDGE_PROMPT_VERSION
    progress: Callable[[str], None] | None = None


@dataclass(frozen=True)
class SyntheticAssessmentBatchedJudgeRequest:
    example_id: str
    packet: dict[str, Any]
    candidates: list[dict[str, Any]]
    provider: str
    model: str
    rubric_version: str
    temperature: float | None
    thinking: str
    prompt_version: str = DEFAULT_JUDGE_PROMPT_VERSION
    progress: Callable[[str], None] | None = None


class SyntheticAssessmentTeacherClient(Protocol):
    def generate(self, request: SyntheticAssessmentTeacherRequest) -> list[SyntheticAssessmentGateResponse]:
        ...


class SyntheticAssessmentJudgeClient(Protocol):
    def judge(self, request: SyntheticAssessmentJudgeRequest) -> dict[str, Any]:
        ...

    def judge_batch(self, request: SyntheticAssessmentBatchedJudgeRequest) -> dict[str, Any]:
        ...


def run_synthetic_detector_assessment_gate(
    *,
    input_jsonl: Path | str,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    mode: Literal["dry_run", "fake", "live"],
    provider: str = "openrouter",
    teacher_model: str | None = None,
    judge_model: str | None = None,
    teacher_prompt_version: str = DEFAULT_TEACHER_PROMPT_VERSION,
    judge_prompt_version: str = DEFAULT_JUDGE_PROMPT_VERSION,
    judge_rubric_version: str = "synthetic_detector_assessment_judge_rubric_v1",
    best_of_n: int = DEFAULT_BEST_OF_N,
    run_id: str | None = None,
    limit: int | None = None,
    example_ids: list[str] | tuple[str, ...] | None = None,
    seed: int | None = None,
    base_seed: int | None = None,
    run_purpose: str = "development",
    approved_input_sha256: str | None = None,
    teacher_temperature: float = 0.2,
    teacher_top_p: float | None = None,
    teacher_repetition_penalty: float | None = None,
    teacher_frequency_penalty: float | None = None,
    teacher_presence_penalty: float | None = None,
    teacher_thinking: str = "disabled",
    judge_temperature: float | None = 0.0,
    judge_thinking: str = "disabled",
    judge_strategy: str = "non_batched_v1",
    judge_replay_candidates_jsonl: Path | str | None = None,
    live_timeout_seconds: float = 60.0,
    teacher_concurrency: int = 1,
    progress_stream: TextIO | None = None,
    teacher_client: SyntheticAssessmentTeacherClient | None = None,
    judge_client: SyntheticAssessmentJudgeClient | None = None,
) -> SyntheticAssessmentGateResult:
    if mode not in VALID_MODES:
        raise ValueError("mode must be dry_run, fake, or live")
    if judge_strategy not in VALID_JUDGE_STRATEGIES:
        raise ValueError(f"judge_strategy must be one of: {sorted(VALID_JUDGE_STRATEGIES)}")
    if best_of_n < 1 or best_of_n > 8:
        raise ValueError("best_of_n must be between 1 and 8")
    if teacher_concurrency < 1 or teacher_concurrency > best_of_n:
        raise ValueError("teacher_concurrency must be between 1 and best_of_n")
    if mode != "dry_run" and (not teacher_model or not judge_model):
        raise ValueError("teacher_model and judge_model are required outside dry_run")
    _validate_prompt_versions(teacher_prompt_version=teacher_prompt_version, judge_prompt_version=judge_prompt_version)
    input_path = Path(input_jsonl)
    if mode == "live":
        _validate_live_input_lock(
            input_path,
            run_purpose=run_purpose,
            approved_input_sha256=approved_input_sha256,
        )
        _validate_live_model_lock(provider, teacher_model, judge_model)
    _validate_decoding_config(
        teacher_temperature=teacher_temperature,
        teacher_top_p=teacher_top_p,
        teacher_repetition_penalty=teacher_repetition_penalty,
        teacher_frequency_penalty=teacher_frequency_penalty,
        teacher_presence_penalty=teacher_presence_penalty,
        teacher_thinking=teacher_thinking,
        judge_temperature=judge_temperature,
        judge_thinking=judge_thinking,
    )
    output_root_path = Path(output_root)
    run_dir = output_root_path / (run_id or f"{mode}_{_slug(input_path.stem)}")
    run_dir.mkdir(parents=True, exist_ok=True)

    records = _read_jsonl(input_path)
    records = _select_records(records, example_ids=example_ids, limit=limit)
    _validate_raw_input_records(records)
    if mode == "live":
        _validate_live_run_shape(
            run_purpose,
            records_loaded=len(records),
            best_of_n=best_of_n,
            judge_replay_candidates_jsonl=judge_replay_candidates_jsonl,
        )

    progress = _progress_writer(progress_stream)

    if mode == "live":
        if provider == "openrouter":
            teacher_client = teacher_client or OpenRouterSyntheticAssessmentTeacherClient(
                timeout_seconds=live_timeout_seconds,
                max_concurrency=teacher_concurrency,
            )
            judge_client = judge_client or OpenRouterSyntheticAssessmentJudgeClient(timeout_seconds=live_timeout_seconds)
        elif provider == "deepseek":
            teacher_client = teacher_client or DeepSeekSyntheticAssessmentTeacherClient(
                timeout_seconds=live_timeout_seconds,
                max_concurrency=teacher_concurrency,
            )
            judge_client = judge_client or DeepSeekSyntheticAssessmentJudgeClient(timeout_seconds=live_timeout_seconds)
        else:
            raise ValueError(f"unsupported live provider: {provider}")
    else:
        teacher_client = teacher_client or _DeterministicFakeTeacherClient()
        judge_client = judge_client or _DeterministicFakeJudgeClient()

    candidate_rows: list[dict[str, Any]] = []
    filtered_records: list[dict[str, Any]] = []
    non_kept_rows: list[dict[str, Any]] = []
    judge_api_calls = 0
    judge_strategy_effective_counts: Counter[str] = Counter()
    effective_base_seed = base_seed if base_seed is not None else seed
    replay_candidates_by_example_id = _load_judge_replay_candidates(judge_replay_candidates_jsonl)

    if mode != "dry_run":
        for raw_record_index, record in enumerate(records):
            progress(
                f"record {raw_record_index + 1}/{len(records)} "
                f"example_id={record['example_id']} run_purpose={run_purpose}"
            )
            packet = record["input"]["data"]
            record_candidate_rows: list[dict[str, Any]] = []
            judged_candidates: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
            replay_rows = replay_candidates_by_example_id.get(record["example_id"], [])
            if replay_candidates_by_example_id:
                progress(
                    f"teacher replay record={raw_record_index + 1}/{len(records)} "
                    f"candidates={len(replay_rows)}"
                )
                response_specs = [
                    (
                        row["candidate_index"],
                        SyntheticAssessmentGateResponse(
                            content=json.dumps(row["assessment"], ensure_ascii=False, sort_keys=True)
                        ),
                        row.get("candidate_seed"),
                        row.get("teacher_temperature", teacher_temperature),
                        row.get("teacher_top_p", teacher_top_p),
                        row.get("teacher_repetition_penalty", teacher_repetition_penalty),
                        row.get("teacher_frequency_penalty", teacher_frequency_penalty),
                        row.get("teacher_presence_penalty", teacher_presence_penalty),
                        True,
                    )
                    for row in replay_rows
                ]
            else:
                candidate_seeds = _candidate_seeds(
                    effective_base_seed,
                    raw_record_index=raw_record_index,
                    best_of_n=best_of_n,
                )
                progress(
                    f"teacher batch start record={raw_record_index + 1}/{len(records)} "
                    f"candidates={best_of_n} temperature={teacher_temperature} "
                    f"top_p={teacher_top_p} repetition_penalty={teacher_repetition_penalty} "
                    f"frequency_penalty={teacher_frequency_penalty} presence_penalty={teacher_presence_penalty} "
                    f"seeds={list(candidate_seeds)}"
                )
                responses = teacher_client.generate(
                    SyntheticAssessmentTeacherRequest(
                        example_id=record["example_id"],
                        packet=packet,
                        best_of_n=best_of_n,
                        provider=provider,
                        model=teacher_model or "dry_run_no_teacher_model",
                        prompt_version=teacher_prompt_version,
                        seed=candidate_seeds[0],
                        temperature=teacher_temperature,
                        top_p=teacher_top_p,
                        repetition_penalty=teacher_repetition_penalty,
                        frequency_penalty=teacher_frequency_penalty,
                        presence_penalty=teacher_presence_penalty,
                        thinking=teacher_thinking,
                        candidate_seeds=candidate_seeds,
                        progress=progress,
                    )
                )
                progress(
                    f"teacher batch done record={raw_record_index + 1}/{len(records)} "
                    f"candidates={len(responses[:best_of_n])}"
                )
                response_specs = [
                    (
                        index,
                        response,
                        candidate_seeds[index],
                        teacher_temperature,
                        teacher_top_p,
                        teacher_repetition_penalty,
                        teacher_frequency_penalty,
                        teacher_presence_penalty,
                        False,
                    )
                    for index, response in enumerate(responses[:best_of_n])
                ]
            for (
                index,
                response,
                candidate_seed,
                candidate_temperature,
                candidate_top_p,
                candidate_repetition_penalty,
                candidate_frequency_penalty,
                candidate_presence_penalty,
                replayed_candidate,
            ) in response_specs:
                row = _candidate_base_row(
                    record,
                    index,
                    packet,
                    candidate_seed=candidate_seed,
                    teacher_temperature=candidate_temperature,
                    teacher_top_p=candidate_top_p,
                    teacher_repetition_penalty=candidate_repetition_penalty,
                    teacher_frequency_penalty=candidate_frequency_penalty,
                    teacher_presence_penalty=candidate_presence_penalty,
                    replayed_candidate=replayed_candidate,
                )
                if response.provider_metadata:
                    row["teacher_provider_metadata"] = response.provider_metadata
                try:
                    assessment = parse_and_validate_detector_assessment(response.content, packet)
                except ValueError as error:
                    progress(
                        f"deterministic validation failed record={raw_record_index + 1}/{len(records)} "
                        f"candidate={index + 1}/{len(response_specs)} reason={error}"
                    )
                    row.update(
                        {
                            "assessment": None,
                            "parse_status": "invalid",
                            "deterministic_valid": False,
                            "validation_reason_codes": [str(error)],
                            "judge_decision": None,
                            "judge_score": None,
                            "judge_reason_codes": [],
                            "selected_for_promotion": False,
                        }
                    )
                    candidate_rows.append(row)
                    record_candidate_rows.append(row)
                    continue

                row.update(
                    {
                        "assessment": assessment,
                        "assessment_support_level": assessment.get("support_level"),
                        "assessment_risk_category": assessment.get("risk_category"),
                        "parse_status": "parsed",
                        "deterministic_valid": True,
                        "validation_reason_codes": [],
                        "judge_decision": None,
                        "judge_score": None,
                        "judge_reason_codes": [],
                        "selected_for_promotion": False,
                    }
                )
                candidate_rows.append(row)
                record_candidate_rows.append(row)
                judged_candidates.append((row, assessment, {}))

            if judged_candidates:
                if judge_strategy == "batched_per_packet_v1":
                    judge_batch = getattr(judge_client, "judge_batch", None)
                    if callable(judge_batch):
                        progress(
                            f"batched judge start record={raw_record_index + 1}/{len(records)} "
                            f"candidates={len(judged_candidates)}"
                        )
                        batch_result = judge_batch(
                            SyntheticAssessmentBatchedJudgeRequest(
                                example_id=record["example_id"],
                                packet=packet,
                                candidates=[
                                    {"candidate_index": row["candidate_index"], "assessment": assessment}
                                    for row, assessment, _ in judged_candidates
                                ],
                                provider=provider,
                                model=judge_model or "dry_run_no_judge_model",
                                rubric_version=judge_rubric_version,
                                temperature=judge_temperature,
                                thinking=judge_thinking,
                                prompt_version=judge_prompt_version,
                                progress=progress,
                            )
                        )
                        judge_api_calls += 1
                        judge_strategy_effective_counts["batched_per_packet_v1"] += 1
                        _apply_batched_judge_result(judged_candidates, batch_result)
                        progress(
                            f"batched judge done record={raw_record_index + 1}/{len(records)} "
                            f"selected_candidate_index={batch_result.get('selected_candidate_index')}"
                        )
                    else:
                        progress(
                            f"batched judge unavailable record={raw_record_index + 1}/{len(records)} "
                            "falling_back=non_batched_v1"
                        )
                        judge_api_calls += _judge_candidates_individually(
                            judged_candidates,
                            judge_client,
                            record=record,
                            packet=packet,
                            provider=provider,
                            judge_model=judge_model,
                            judge_rubric_version=judge_rubric_version,
                            judge_temperature=judge_temperature,
                            judge_thinking=judge_thinking,
                            judge_prompt_version=judge_prompt_version,
                            progress=progress,
                            raw_record_index=raw_record_index,
                            total_records=len(records),
                        )
                        judge_strategy_effective_counts["non_batched_v1"] += 1
                else:
                    judge_api_calls += _judge_candidates_individually(
                        judged_candidates,
                        judge_client,
                        record=record,
                        packet=packet,
                        provider=provider,
                        judge_model=judge_model,
                        judge_rubric_version=judge_rubric_version,
                        judge_temperature=judge_temperature,
                        judge_thinking=judge_thinking,
                        judge_prompt_version=judge_prompt_version,
                        progress=progress,
                        raw_record_index=raw_record_index,
                        total_records=len(records),
                    )
                    judge_strategy_effective_counts["non_batched_v1"] += 1

            selected_candidate = _select_promoted_candidate(judged_candidates)
            for row, assessment, judge_result in judged_candidates:
                if selected_candidate is not None and row is selected_candidate[0]:
                    row["selected_for_promotion"] = True
                    filtered = _filtered_record(
                        record,
                        assessment,
                        judge_result,
                        provider=provider,
                        teacher_model=teacher_model,
                        judge_model=judge_model,
                        teacher_temperature=teacher_temperature,
                        teacher_top_p=teacher_top_p,
                        teacher_repetition_penalty=teacher_repetition_penalty,
                        teacher_frequency_penalty=teacher_frequency_penalty,
                        teacher_presence_penalty=teacher_presence_penalty,
                        teacher_thinking=teacher_thinking,
                        judge_temperature=judge_temperature,
                        judge_thinking=judge_thinking,
                        best_of_n=best_of_n,
                        base_seed=effective_base_seed,
                    )
                    _validate_filtered_record(filtered)
                    filtered_records.append(filtered)
            for row in record_candidate_rows:
                if row.get("selected_for_promotion"):
                    continue
                if not row.get("deterministic_valid"):
                    non_kept_rows.append(_non_kept_row(row, "reject", row.get("validation_reason_codes", [])))
                    continue
                decision = row.get("judge_decision")
                disposition = decision if decision in VALID_JUDGE_DECISIONS and decision != "keep" else "not_selected"
                reason_codes = list(row.get("judge_reason_codes", []))
                reason_codes.extend(_promotion_rejection_reason_codes(row))
                non_kept_rows.append(_non_kept_row(row, disposition, sorted(set(reason_codes))))

    manifest = _manifest(
        run_dir=run_dir,
        input_path=input_path,
        mode=mode,
        best_of_n=best_of_n,
        provider=provider,
        teacher_model=teacher_model,
        judge_model=judge_model,
        teacher_prompt_version=teacher_prompt_version,
        judge_prompt_version=judge_prompt_version,
        judge_rubric_version=judge_rubric_version,
        base_seed=effective_base_seed,
        run_purpose=run_purpose,
        teacher_temperature=teacher_temperature,
        teacher_top_p=teacher_top_p,
        teacher_repetition_penalty=teacher_repetition_penalty,
        teacher_frequency_penalty=teacher_frequency_penalty,
        teacher_presence_penalty=teacher_presence_penalty,
        teacher_thinking=teacher_thinking,
        judge_temperature=judge_temperature,
        judge_thinking=judge_thinking,
        judge_strategy=judge_strategy,
        input_records=len(records),
        promoted_records=len(filtered_records),
        judge_replay_candidates_jsonl=Path(judge_replay_candidates_jsonl) if judge_replay_candidates_jsonl else None,
    )
    metrics = _metrics(
        records,
        candidate_rows,
        filtered_records,
        non_kept_rows,
        mode=mode,
        judge_api_calls=judge_api_calls,
        judge_strategy=judge_strategy,
        judge_strategy_effective_counts=judge_strategy_effective_counts,
    )
    errors = _run_errors(mode=mode, run_purpose=run_purpose, metrics=metrics)

    manifest_path = run_dir / "manifest.json"
    candidates_path = run_dir / "candidates.jsonl"
    filtered_path = run_dir / "filtered.jsonl"
    non_kept_path = run_dir / "non_kept.jsonl"
    metrics_path = run_dir / "metrics.json"
    _write_json(manifest_path, manifest)
    _write_jsonl(candidates_path, candidate_rows)
    _write_jsonl(filtered_path, filtered_records)
    _write_jsonl(non_kept_path, non_kept_rows)
    _write_json(metrics_path, metrics)

    return SyntheticAssessmentGateResult(
        status="failed" if errors else "passed",
        run_dir=run_dir,
        manifest_path=manifest_path,
        candidates_path=candidates_path,
        filtered_path=filtered_path,
        non_kept_path=non_kept_path,
        metrics_path=metrics_path,
        records_loaded=len(records),
        candidates_written=len(candidate_rows),
        promoted_count=len(filtered_records),
        errors=errors,
    )


class _DeterministicFakeTeacherClient:
    def generate(self, request: SyntheticAssessmentTeacherRequest) -> list[SyntheticAssessmentGateResponse]:
        packet = request.packet
        finding = next((item for item in packet.get("tool_findings", []) if item.get("tool_result_id")), None)
        supporting_signal_ids = packet.get("candidate_summary", {}).get("supporting_signal_ids") or ["signal"]
        signal_id = finding.get("signal_id") if finding else supporting_signal_ids[0]
        tool_result_id = finding.get("tool_result_id") if finding else None
        evidence_ref = {"evidence_ref_type": "rule", "ref_id": packet["rules"][0]["rule_id"], "role": "context"}
        if tool_result_id:
            evidence_ref = {"evidence_ref_type": "tool_result", "ref_id": tool_result_id, "role": "supporting"}
        signal = {
            "signal_id": signal_id,
            "status": "validated",
            "support_level": "supported",
            "cited_evidence_refs": [evidence_ref],
        }
        if tool_result_id:
            signal["tool_result_id"] = tool_result_id
        assessment = {
            "assessment_id": f"ASSESS_{packet['packet_id']}_FAKE_001",
            "packet_id": packet["packet_id"],
            "candidate_id": packet["candidate_id"],
            "report_id": packet["report_id"],
            "risk_category": packet["task"]["risk_category"],
            "support_level": "supported",
            "confidence": 0.8,
            "severity": "medium",
            "validated_signals": [signal],
            "cited_evidence_refs": [evidence_ref],
            "rationale_short": "The cited packet evidence supports the candidate risk signal.",
        }
        return [
            SyntheticAssessmentGateResponse(content=json.dumps(assessment, ensure_ascii=False))
            for _ in range(request.best_of_n)
        ]


@dataclass(frozen=True)
class OpenRouterSyntheticAssessmentTeacherClient:
    timeout_seconds: float = 60.0
    max_concurrency: int = 1

    def generate(self, request: SyntheticAssessmentTeacherRequest) -> list[SyntheticAssessmentGateResponse]:
        candidate_seeds = request.candidate_seeds or (request.seed,) * request.best_of_n
        if self.max_concurrency <= 1 or len(candidate_seeds) <= 1:
            return [
                self._generate_one(request, index=index, candidate_seed=candidate_seed, total=len(candidate_seeds))
                for index, candidate_seed in enumerate(candidate_seeds)
            ]

        responses_by_index: dict[int, SyntheticAssessmentGateResponse] = {}
        max_workers = min(self.max_concurrency, len(candidate_seeds))
        if request.progress:
            request.progress(f"teacher parallel start candidates={len(candidate_seeds)} max_workers={max_workers}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._generate_one,
                    request,
                    index=index,
                    candidate_seed=candidate_seed,
                    total=len(candidate_seeds),
                ): index
                for index, candidate_seed in enumerate(candidate_seeds)
            }
            for future in concurrent.futures.as_completed(futures):
                index = futures[future]
                responses_by_index[index] = future.result()
        if request.progress:
            request.progress(f"teacher parallel done candidates={len(candidate_seeds)} max_workers={max_workers}")
        return [responses_by_index[index] for index in range(len(candidate_seeds))]

    def _generate_one(
        self,
        request: SyntheticAssessmentTeacherRequest,
        *,
        index: int,
        candidate_seed: int | None,
        total: int,
    ) -> SyntheticAssessmentGateResponse:
        if request.progress:
            request.progress(
                f"teacher call start candidate={index + 1}/{total} "
                f"seed={candidate_seed} timeout_seconds={self.timeout_seconds}"
            )
        started = time.monotonic()
        content, provider_metadata = self._complete_once(
            request,
            candidate_seed=candidate_seed,
            call_label=f"teacher candidate={index + 1}/{total} seed={candidate_seed}",
        )
        elapsed_seconds = time.monotonic() - started
        if request.progress:
            request.progress(
                f"teacher call done candidate={index + 1}/{total} "
                f"seed={candidate_seed} elapsed_seconds={elapsed_seconds:.1f}"
            )
        return SyntheticAssessmentGateResponse(content=content, provider_metadata=provider_metadata)

    def _complete_once(
        self,
        request: SyntheticAssessmentTeacherRequest,
        *,
        candidate_seed: int | None,
        call_label: str,
    ) -> tuple[str, dict[str, Any]]:
        payload = {
            "model": request.model,
            "max_tokens": OPENROUTER_TEACHER_MAX_TOKENS,
            "messages": [
                {"role": "system", "content": _teacher_system_prompt(request.prompt_version)},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "detector_packet": request.packet,
                            "visible_id_guide": _teacher_visible_id_guide(request.packet),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "DetectorAssessment",
                    "strict": True,
                    "schema": detector_assessment_json_schema(),
                },
            },
        }
        payload.update(
            _openrouter_decoding_payload(
                request.temperature,
                request.thinking,
                seed=candidate_seed,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty,
            )
        )
        return _openrouter_chat_content_with_provider_rotation(
            payload,
            provider=request.provider,
            model=request.model,
            timeout_seconds=self.timeout_seconds,
            progress=request.progress,
            call_label=call_label,
        )


@dataclass(frozen=True)
class OpenRouterSyntheticAssessmentJudgeClient:
    timeout_seconds: float = 60.0

    def judge(self, request: SyntheticAssessmentJudgeRequest) -> dict[str, Any]:
        payload = {
            "model": request.model,
            "max_tokens": OPENROUTER_JUDGE_MAX_TOKENS,
            "messages": [
                {"role": "system", "content": _judge_system_prompt(request.prompt_version)},
                {
                    "role": "user",
                    "content": json.dumps(
                        {"detector_packet": request.packet, "candidate_assessment": request.assessment},
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "SyntheticAssessmentJudgeDecision",
                    "strict": True,
                    "schema": _judge_json_schema(),
                },
            },
        }
        payload.update(_openrouter_decoding_payload(request.temperature, request.thinking))
        progress = getattr(request, "progress", None)
        content, provider_metadata = _openrouter_chat_content_with_provider_rotation(
            payload,
            provider=request.provider,
            model=request.model,
            timeout_seconds=self.timeout_seconds,
            progress=progress,
            call_label=f"judge example_id={request.example_id}",
            content_validator=_validate_openrouter_json_content,
        )
        result = json.loads(content)
        result["_provider_metadata"] = provider_metadata
        return result

    def judge_batch(self, request: SyntheticAssessmentBatchedJudgeRequest) -> dict[str, Any]:
        payload = {
            "model": request.model,
            "max_tokens": OPENROUTER_JUDGE_MAX_TOKENS,
            "messages": [
                {"role": "system", "content": _batched_judge_system_prompt(request.prompt_version)},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "example_id": request.example_id,
                            "packet": request.packet,
                            "candidates": request.candidates,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "SyntheticAssessmentBatchedJudgeDecision",
                    "strict": True,
                    "schema": _batched_judge_json_schema(),
                },
            },
        }
        payload.update(_openrouter_decoding_payload(request.temperature, request.thinking))
        content, provider_metadata = _openrouter_chat_content_with_provider_rotation(
            payload,
            provider=request.provider,
            model=request.model,
            timeout_seconds=self.timeout_seconds,
            progress=request.progress,
            call_label=f"batched judge example_id={request.example_id}",
            content_validator=_validate_openrouter_json_content,
        )
        result = json.loads(content)
        result["_provider_metadata"] = provider_metadata
        return result


@dataclass(frozen=True)
class DeepSeekSyntheticAssessmentTeacherClient:
    timeout_seconds: float = 60.0
    max_concurrency: int = 1

    def generate(self, request: SyntheticAssessmentTeacherRequest) -> list[SyntheticAssessmentGateResponse]:
        candidate_seeds = request.candidate_seeds or (request.seed,) * request.best_of_n
        if self.max_concurrency <= 1 or len(candidate_seeds) <= 1:
            return [
                self._generate_one(request, index=index, candidate_seed=candidate_seed, total=len(candidate_seeds))
                for index, candidate_seed in enumerate(candidate_seeds)
            ]

        responses_by_index: dict[int, SyntheticAssessmentGateResponse] = {}
        max_workers = min(self.max_concurrency, len(candidate_seeds))
        if request.progress:
            request.progress(f"teacher parallel start candidates={len(candidate_seeds)} max_workers={max_workers}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._generate_one,
                    request,
                    index=index,
                    candidate_seed=candidate_seed,
                    total=len(candidate_seeds),
                ): index
                for index, candidate_seed in enumerate(candidate_seeds)
            }
            for future in concurrent.futures.as_completed(futures):
                index = futures[future]
                responses_by_index[index] = future.result()
        if request.progress:
            request.progress(f"teacher parallel done candidates={len(candidate_seeds)} max_workers={max_workers}")
        return [responses_by_index[index] for index in range(len(candidate_seeds))]

    def _generate_one(
        self,
        request: SyntheticAssessmentTeacherRequest,
        *,
        index: int,
        candidate_seed: int | None,
        total: int,
    ) -> SyntheticAssessmentGateResponse:
        if request.progress:
            request.progress(
                f"teacher call start candidate={index + 1}/{total} "
                f"seed={candidate_seed} timeout_seconds={self.timeout_seconds}"
            )
        started = time.monotonic()
        payload = {
            "model": request.model,
            "max_tokens": DEEPSEEK_TEACHER_MAX_TOKENS,
            "messages": [
                {"role": "system", "content": _teacher_system_prompt(request.prompt_version)},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "detector_packet": request.packet,
                            "visible_id_guide": _teacher_visible_id_guide(request.packet),
                            "detector_assessment_output_contract": _detector_assessment_output_contract(
                                request.packet
                            ),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                },
            ],
            "response_format": {"type": "json_object"},
        }
        payload.update(
            _deepseek_decoding_payload(
                request.temperature,
                request.thinking,
                top_p=request.top_p,
                frequency_penalty=request.frequency_penalty,
                presence_penalty=request.presence_penalty,
            )
        )
        content, provider_metadata = _deepseek_chat_content(payload, timeout_seconds=self.timeout_seconds)
        elapsed_seconds = time.monotonic() - started
        provider_metadata["elapsed_seconds"] = round(elapsed_seconds, 3)
        provider_metadata["candidate_seed_requested"] = candidate_seed
        if request.repetition_penalty is not None:
            provider_metadata["unsupported_decoding_parameters_omitted"] = ["repetition_penalty"]
        if request.progress:
            request.progress(
                f"teacher call done candidate={index + 1}/{total} "
                f"seed={candidate_seed} elapsed_seconds={elapsed_seconds:.1f}"
            )
        return SyntheticAssessmentGateResponse(content=content, provider_metadata=provider_metadata)


@dataclass(frozen=True)
class DeepSeekSyntheticAssessmentJudgeClient:
    timeout_seconds: float = 60.0

    def judge(self, request: SyntheticAssessmentJudgeRequest) -> dict[str, Any]:
        payload = {
            "model": request.model,
            "max_tokens": DEEPSEEK_JUDGE_MAX_TOKENS,
            "messages": [
                {"role": "system", "content": _judge_system_prompt(request.prompt_version)},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "detector_packet": request.packet,
                            "candidate_assessment": request.assessment,
                            "judge_output_contract": _json_output_contract_from_schema(_judge_json_schema()),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                },
            ],
            "response_format": {"type": "json_object"},
        }
        payload.update(_deepseek_decoding_payload(request.temperature, request.thinking))
        content, provider_metadata = _deepseek_chat_content(payload, timeout_seconds=self.timeout_seconds)
        _validate_openrouter_json_content(content)
        result = json.loads(content)
        result["_provider_metadata"] = provider_metadata
        return result

    def judge_batch(self, request: SyntheticAssessmentBatchedJudgeRequest) -> dict[str, Any]:
        payload = {
            "model": request.model,
            "max_tokens": DEEPSEEK_JUDGE_MAX_TOKENS,
            "messages": [
                {"role": "system", "content": _batched_judge_system_prompt(request.prompt_version)},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "example_id": request.example_id,
                            "packet": request.packet,
                            "candidates": request.candidates,
                            "batched_judge_output_contract": _json_output_contract_from_schema(
                                _batched_judge_json_schema()
                            ),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                },
            ],
            "response_format": {"type": "json_object"},
        }
        payload.update(_deepseek_decoding_payload(request.temperature, request.thinking))
        content, provider_metadata = _deepseek_chat_content(payload, timeout_seconds=self.timeout_seconds)
        _validate_openrouter_json_content(content)
        result = json.loads(content)
        result["_provider_metadata"] = provider_metadata
        return result


class _DeterministicFakeJudgeClient:
    def judge(self, request: SyntheticAssessmentJudgeRequest) -> dict[str, Any]:
        return {
            "decision": "keep",
            "score": 1.0,
            "reason_codes": [],
            "dimension_scores": {
                "schema_correctness": 5,
                "evidence_grounding": 5,
                "support_level_correctness": 5,
                "risk_category_consistency": 5,
                "validated_signal_correctness": 5,
                "rationale_quality": 5,
                "conservativeness": 5,
                "risk_language_compliance": 5,
                "hidden_metadata_independence": 5,
            },
        }


def _teacher_system_prompt(prompt_version: str = DEFAULT_TEACHER_PROMPT_VERSION) -> str:
    return (
        "Return exactly one DetectorAssessment JSON object with these top-level keys: assessment_id, "
        "packet_id, candidate_id, report_id, risk_category, support_level, confidence, severity, "
        "validated_signals, cited_evidence_refs, rationale_short. Do not wrap the object in "
        "assessment, type, data, result, output, or any other envelope. Do not add secondary risk "
        "categories. Risk category must match DetectorPacket.task.risk_category. Use only evidence "
        "provided in the DetectorPacket. Use exact ref_id strings from DetectorPacket.tool_findings[]."
        "evidence_refs, DetectorPacket.rules[].rule_id, DetectorPacket.tool_findings[].tool_result_id, "
        "and canonical packet evidence IDs formed as report_id:cell_id, report_id:row_id, "
        "report_id:note_id, or report_id:span_id from visible packet rows, notes, and variance "
        "explanations. Do not use local_evidence_id, row_id, note_id, span_id, cell_id, signal_id, "
        "candidate_summary, metric_name, report_id, or tool_name by itself as an evidence ref_id. "
        "Use only these evidence_ref_type values: table_cell, table_row, note, note_span, "
        "variance_explanation_span, accounting_policy_note_span, related_party_note_span, "
        "tool_result, rule. Use only these evidence roles: supporting, contradicting, refuting, "
        "context, missing_required_context. Each validated_signals item must have signal_id, status, "
        "support_level, cited_evidence_refs, and may include tool_result_id only when it is visible in "
        "the packet. Use only these validated_signals status values: validated, partially_validated, "
        "rejected, not_assessable. Use only supported, weakly_supported, not_supported, or "
        "insufficient_evidence. Use only low, medium, high, or unknown severity. Confidence must be "
        "between 0 and 1. Keep rationale_short to 1-3 concise sentences. Do not use outside "
        "knowledge or hidden injection metadata. Do not claim fraud, manipulation, concealment, "
        "intent, misconduct, or legal misstatement. Use risk-signal language only. If the packet has "
        "no tool_findings or no supporting_signal_ids, normally return insufficient_evidence with "
        "status not_assessable; cite visible rule or table IDs as missing_required_context and omit "
        "tool_result_id unless a visible tool_result_id exists. "
        f"{_support_level_calibration_prompt(prompt_version)}"
    )


def _support_level_calibration_prompt(prompt_version: str) -> str:
    base_prompt = (
        "Calibrate support_level to the visible evidence, not to the safest possible label: use "
        "supported when visible tool findings, cited table values, or notes strongly and materially "
        "satisfy the configured risk condition. Do not downgrade a strongly satisfied condition to "
        "weakly_supported, not_supported, or insufficient_evidence merely because the task is "
        "synthetic or because conservatism is preferred. Use weakly_supported when the evidence "
        "supports the direction of the risk signal but the threshold margin is modest, corroborating "
        "notes or disclosures are missing, or only part of the condition is satisfied. Use "
        "not_supported when flag is false, the finding_summary says the configured condition did not "
        "hold, the calculation contradicts the risk claim, or a signal_id appears without supporting "
        "values. Use insufficient_evidence only when required visible evidence is absent or the "
        "packet cannot assess the configured condition. When fields conflict, prefer the "
        "finding_summary, flag, threshold, and metric_value over the candidate_summary wording, and "
        "do not upgrade to supported from signal_id alone."
    )
    if prompt_version in {
        DEFAULT_TEACHER_PROMPT_VERSION,
        DEFAULT_JUDGE_PROMPT_VERSION,
        LEGACY_TEACHER_PROMPT_VERSION,
        LEGACY_JUDGE_PROMPT_VERSION,
    }:
        return base_prompt
    if prompt_version in {WEAKCAL_TEACHER_PROMPT_VERSION, WEAKCAL_JUDGE_PROMPT_VERSION}:
        return (
            f"{base_prompt} Weak-support calibration: use weakly_supported, not supported, when "
            "visible evidence points in the risk direction but is borderline, partial, single-source, "
            "missing corroborating note context, or has only a modest threshold margin. Do not collapse "
            "a positive-but-limited signal into insufficient_evidence when cited visible evidence is "
            "available, and do not promote it to supported unless the packet shows strong materiality "
            "and corroboration. In validated_signals, weakly_supported signals should normally use "
            "status partially_validated when the visible finding supports the direction but the evidence is limited."
        )
    if prompt_version in {
        CORROBORATION_TEACHER_PROMPT_VERSION,
        CORROBORATION_JUDGE_PROMPT_VERSION,
    }:
        return (
            f"{base_prompt} Evidence-bundle calibration: distinguish trigger magnitude from support "
            "sufficiency. A tool finding's strength field describes the magnitude of that trigger, "
            "not the completeness of the packet's evidence. A strong isolated quantitative signal "
            "with no independent corroborating signal, note, or mechanism should normally be "
            "weakly_supported. Use supported when the complete visible evidence bundle is direct, "
            "coherent, and sufficient, normally because multiple independent signals align, one "
            "signal is independently corroborated, or a visible category-specific rule identifies "
            "one evidence item as decisive. Do not infer corroboration from repeated references to "
            "the same calculation."
        )
    if prompt_version in {
        CORROBORATION_STRICT_TEACHER_PROMPT_VERSION,
        CORROBORATION_STRICT_JUDGE_PROMPT_VERSION,
    }:
        return (
            f"{base_prompt} Strict evidence-bundle calibration: trigger magnitude and support "
            "sufficiency are different judgments. A tool finding's strength field describes only "
            "the magnitude of that trigger. For earnings_cashflow_mismatch, the profit row and "
            "operating-cash-flow row are inputs to one primary trigger; they do not independently "
            "corroborate that trigger. If the packet shows exactly one independent signal and no "
            "second tool finding, note, variance explanation, or explicit corroboration signal, "
            "return weakly_supported even when the trigger is strong, the ratio is extreme, and "
            "the table rows directly prove the mismatch. Return supported only when a second "
            "independent visible signal or note/mechanism corroborates the primary trigger. Reject "
            "any assessment that calls the primary trigger's own input rows independent "
            "corroboration."
        )
    if prompt_version in {
        CORROBORATION_BALANCED_TEACHER_PROMPT_VERSION,
        CORROBORATION_BALANCED_JUDGE_PROMPT_VERSION,
    }:
        return (
            f"{base_prompt} Balanced evidence-bundle calibration: trigger magnitude and support "
            "sufficiency are different judgments. For earnings_cashflow_mismatch, the profit row "
            "and operating-cash-flow row are inputs to one primary trigger; they do not independently "
            "corroborate that trigger. If the packet shows exactly one independent signal and no "
            "second tool finding, note, variance explanation, or explicit corroboration signal, "
            "return weakly_supported even when the trigger is strong. A second tool finding counts "
            "as independent corroboration only when it has a distinct signal_id, a distinct "
            "tool_result_id, and cites at least one non-primary-trigger evidence item such as a "
            "linked balance row, note, or variance explanation. When such a second visible signal "
            "directly corroborates cash-conversion pressure, supported is appropriate; do not require "
            "a separate narrative note. Reject any assessment that calls the primary trigger's own "
            "input rows independent corroboration."
        )
    raise ValueError(f"unsupported synthetic detector assessment prompt version: {prompt_version}")


def _teacher_visible_id_guide(packet: dict[str, Any]) -> dict[str, Any]:
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
    return {
        "allowed_evidence_ref_ids": sorted(visible_packet_evidence_ids(packet)),
        "allowed_signal_ids": sorted(visible_signal_ids),
        "allowed_tool_result_ids": sorted(visible_tool_result_ids),
        "insufficient_evidence_guidance": (
            "When allowed_tool_result_ids is empty or candidate_summary.supporting_signal_ids is empty, "
            "use support_level insufficient_evidence, signal status not_assessable, no tool_result_id, "
            "and cite only allowed_evidence_ref_ids with role missing_required_context or context."
        ),
    }


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
        "validated_signals_item": {
            "required_keys": list(signal_schema["required"]),
            "optional_keys": ["tool_result_id"],
            "additional_keys_allowed": False,
        },
        "evidence_ref_item": {
            "keys_exact": list(evidence_ref_schema["required"]),
            "ref_id_rule": "Use only IDs listed in visible_id_guide.allowed_evidence_ref_ids.",
        },
        "rationale_short": "1-3 concise sentences using risk-signal language only",
    }


def _json_output_contract_from_schema(schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "return_only": "one JSON object; no markdown, array, or wrapper object",
        "top_level_keys_exact": list(schema.get("required", [])),
        "additional_top_level_keys_allowed": bool(schema.get("additionalProperties", True)),
        "schema": schema,
    }


def _judge_system_prompt(prompt_version: str = DEFAULT_JUDGE_PROMPT_VERSION) -> str:
    return (
        "Judge one deterministic-valid candidate DetectorAssessment against the DetectorPacket. "
        "Return JSON with decision, score, reason_codes, and dimension_scores only. Use a 1-5 "
        "integer scale for every dimension_score, where 1 means unusable, 3 means borderline, "
        "and 5 means perfect. Allowed decisions are keep, revise, reject, and needs_human_review. "
        "Keep only if schema correctness, evidence grounding, risk-language compliance, "
        "hidden-metadata independence, support-level correctness, and rationale quality are 5. "
        "For support_level insufficient_evidence, treat evidence grounding as 5 when the assessment "
        "uses only visible packet evidence IDs with role missing_required_context or context, uses "
        "status not_assessable, and omits tool_result_id when no visible tool_result_id exists. "
        f"{_support_level_calibration_prompt(prompt_version)} "
        "Reject any dependence on outside knowledge, hidden injection metadata, invented evidence, "
        "wrong risk category, prohibited language, or unsupported support-level promotion."
    )


def _judge_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": sorted(VALID_JUDGE_DECISIONS)},
            "score": {"type": "number", "minimum": 0, "maximum": 1},
            "reason_codes": {"type": "array", "items": {"type": "string"}},
            "dimension_scores": {
                "type": "object",
                "properties": {
                    key: {"type": "integer", "minimum": 1, "maximum": 5}
                    for key in [
                        "schema_correctness",
                        "evidence_grounding",
                        "support_level_correctness",
                        "risk_category_consistency",
                        "validated_signal_correctness",
                        "rationale_quality",
                        "conservativeness",
                        "risk_language_compliance",
                        "hidden_metadata_independence",
                    ]
                },
                "required": [
                    "schema_correctness",
                    "evidence_grounding",
                    "support_level_correctness",
                    "risk_category_consistency",
                    "validated_signal_correctness",
                    "rationale_quality",
                    "conservativeness",
                    "risk_language_compliance",
                    "hidden_metadata_independence",
                ],
                "additionalProperties": False,
            },
        },
        "required": ["decision", "score", "reason_codes", "dimension_scores"],
        "additionalProperties": False,
    }


def _batched_judge_system_prompt(prompt_version: str = DEFAULT_JUDGE_PROMPT_VERSION) -> str:
    return (
        "Judge all deterministic-valid candidate DetectorAssessment objects against the single "
        "DetectorPacket. Return JSON with candidate_results and selected_candidate_index only. "
        "Each candidate result must include candidate_index, decision, score, reason_codes, and "
        "dimension_scores. Use a 1-5 integer scale for every dimension_score, where 1 means "
        "unusable, 3 means borderline, and 5 means perfect. Allowed decisions are keep, revise, "
        "reject, and needs_human_review. Select at most one candidate_index. Keep only if schema "
        "correctness, evidence grounding, risk-language compliance, hidden-metadata independence, "
        "support-level correctness, and rationale quality are 5. Reject any dependence on outside "
        "knowledge, hidden injection metadata, invented evidence, wrong risk category, prohibited "
        "language, or unsupported support-level promotion. For support_level insufficient_evidence, "
        "treat evidence grounding as 5 when the assessment uses only visible packet evidence IDs "
        "with role missing_required_context or context, uses status not_assessable, and omits "
        "tool_result_id when no visible tool_result_id exists. "
        f"{_support_level_calibration_prompt(prompt_version)} When multiple candidates are otherwise keepable, "
        "select the candidate whose support_level is best calibrated to the visible evidence. Do not "
        "prefer a weaker support_level when another keepable candidate more accurately reflects a "
        "strongly satisfied configured condition."
    )


def _batched_judge_json_schema() -> dict[str, Any]:
    dimension_schema = _judge_json_schema()["properties"]["dimension_scores"]
    return {
        "type": "object",
        "properties": {
            "candidate_results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate_index": {"type": "integer", "minimum": 0},
                        "decision": {"type": "string", "enum": sorted(VALID_JUDGE_DECISIONS)},
                        "score": {"type": "number", "minimum": 0, "maximum": 1},
                        "reason_codes": {"type": "array", "items": {"type": "string"}},
                        "dimension_scores": dimension_schema,
                    },
                    "required": [
                        "candidate_index",
                        "decision",
                        "score",
                        "reason_codes",
                        "dimension_scores",
                    ],
                    "additionalProperties": False,
                },
            },
            "selected_candidate_index": {"type": ["integer", "null"], "minimum": 0},
        },
        "required": ["candidate_results", "selected_candidate_index"],
        "additionalProperties": False,
    }


def _openrouter_chat_content_with_provider_rotation(
    payload: dict[str, Any],
    *,
    provider: str,
    model: str,
    timeout_seconds: float,
    progress: Callable[[str], None] | None,
    call_label: str,
    content_validator: Callable[[str], None] | None = None,
) -> tuple[str, dict[str, Any]]:
    locked_provider_names = list(_locked_openrouter_provider_routing(provider, model)["order"])
    provider_names = locked_provider_names * OPENROUTER_PROVIDER_ROTATION_PASSES
    attempts: list[dict[str, Any]] = []
    last_error: BaseException | None = None
    for attempt_index, provider_name in enumerate(provider_names):
        attempt_payload = dict(payload)
        attempt_payload["provider"] = _single_openrouter_provider_routing(provider, model, provider_name)
        if progress:
            progress(
                f"openrouter attempt start {call_label} provider={provider_name} "
                f"attempt={attempt_index + 1}/{len(provider_names)} timeout_seconds={timeout_seconds}"
            )
        started = time.monotonic()
        try:
            content = _openrouter_chat_content_with_hard_timeout(
                attempt_payload,
                timeout_seconds=timeout_seconds,
                provider_name=provider_name,
            )
        except (TimeoutError, urllib.error.HTTPError, urllib.error.URLError, OSError) as error:
            elapsed_seconds = time.monotonic() - started
            attempts.append(
                {
                    "provider": provider_name,
                    "status": "failed",
                    "elapsed_seconds": round(elapsed_seconds, 3),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            last_error = error
            if progress:
                progress(
                    f"openrouter attempt failed {call_label} provider={provider_name} "
                    f"elapsed_seconds={elapsed_seconds:.1f} error_type={type(error).__name__}"
                )
            continue
        if content_validator:
            try:
                content_validator(content)
            except ValueError as error:
                elapsed_seconds = time.monotonic() - started
                attempts.append(
                    {
                        "provider": provider_name,
                        "status": "failed",
                        "elapsed_seconds": round(elapsed_seconds, 3),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                last_error = error
                if progress:
                    progress(
                        f"openrouter attempt failed {call_label} provider={provider_name} "
                        f"elapsed_seconds={elapsed_seconds:.1f} error_type={type(error).__name__}"
                    )
                continue
        elapsed_seconds = time.monotonic() - started
        attempts.append(
            {
                "provider": provider_name,
                "status": "succeeded",
                "elapsed_seconds": round(elapsed_seconds, 3),
            }
        )
        if progress:
            progress(
                f"openrouter attempt done {call_label} provider={provider_name} "
                f"elapsed_seconds={elapsed_seconds:.1f}"
            )
        return content, {"strategy": "client_side_provider_rotation_v1", "attempts": attempts}
    raise RuntimeError(
        f"OpenRouter request failed after {OPENROUTER_PROVIDER_ROTATION_PASSES} approved-provider passes: "
        f"{', '.join(locked_provider_names)}"
    ) from last_error


def _validate_openrouter_json_content(content: str) -> None:
    if not isinstance(content, str):
        raise ValueError(f"OpenRouter content must be a string, got {type(content).__name__}")
    try:
        json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(
            "malformed OpenRouter JSON content: "
            f"{error.msg} at line {error.lineno} column {error.colno} "
            f"content_chars={len(content)}"
        ) from error


def _openrouter_chat_content(payload: dict[str, Any], *, timeout_seconds: float) -> str:
    _load_dotenv_if_available()
    api_key = os.environ.get(OPENROUTER_API_KEY_ENV)
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is required for live mode")
    request = urllib.request.Request(
        OPENROUTER_CHAT_COMPLETIONS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/buinguyenkhai/vinacount",
            "X-Title": "vinacount-synthetic-detector-assessment-gate",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        response_payload = json.loads(response.read().decode("utf-8", errors="replace"))
    content = response_payload["choices"][0]["message"].get("content")
    if not isinstance(content, str):
        raise OSError(f"OpenRouter message content was {type(content).__name__}")
    return content


def _deepseek_chat_content(payload: dict[str, Any], *, timeout_seconds: float) -> tuple[str, dict[str, Any]]:
    _load_dotenv_if_available()
    api_key = os.environ.get(DEEPSEEK_API_KEY_ENV)
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is required for DeepSeek live mode")
    request = urllib.request.Request(
        DEEPSEEK_CHAT_COMPLETIONS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    response_text = _deepseek_urlopen_text_with_retry(request, timeout_seconds=timeout_seconds)
    try:
        response_payload = json.loads(response_text)
    except json.JSONDecodeError as error:
        preview = response_text[max(0, error.pos - 200) : error.pos + 500].strip()
        raise RuntimeError(
            "deepseek_non_json_response "
            f"json_error={error.msg!r} position={error.pos} body_preview={preview!r}"
        ) from error
    try:
        content = response_payload["choices"][0]["message"].get("content")
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(
            "deepseek_unexpected_response_shape "
            f"body_preview={json.dumps(response_payload, ensure_ascii=False)[:1000]!r}"
        ) from error
    if not isinstance(content, str):
        raise OSError(f"DeepSeek message content was {type(content).__name__}")
    usage = response_payload.get("usage") if isinstance(response_payload.get("usage"), dict) else {}
    provider_metadata = {
        "provider": "deepseek",
        "strategy": "direct_deepseek_chat_completions_v1",
        "request_model": payload.get("model"),
        "response_model": response_payload.get("model"),
        "system_fingerprint": response_payload.get("system_fingerprint"),
        "response_format": payload.get("response_format"),
        "thinking": payload.get("thinking"),
        "usage": usage,
        "cache_usage": {
            "prompt_cache_hit_tokens": usage.get("prompt_cache_hit_tokens"),
            "prompt_cache_miss_tokens": usage.get("prompt_cache_miss_tokens"),
        },
    }
    return content, provider_metadata


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
        except (urllib.error.URLError, TimeoutError, http.client.HTTPException, OSError) as error:
            last_error = error
        if attempt < DEEPSEEK_DIRECT_MAX_ATTEMPTS:
            time.sleep(DEEPSEEK_DIRECT_RETRY_INITIAL_SLEEP_SECONDS * attempt)
    raise RuntimeError(
        "deepseek_request_failed "
        f"attempts={DEEPSEEK_DIRECT_MAX_ATTEMPTS} last_error={last_error!r}"
    ) from last_error


def _openrouter_chat_content_with_hard_timeout(
    payload: dict[str, Any],
    *,
    timeout_seconds: float,
    provider_name: str,
    worker_target: Any = None,
) -> str:
    if timeout_seconds <= 0:
        return _openrouter_chat_content(payload, timeout_seconds=timeout_seconds)

    worker_target = worker_target or _openrouter_chat_content_worker
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=worker_target,
        args=(payload, timeout_seconds, result_queue),
    )
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)
        raise TimeoutError(
            f"OpenRouter provider attempt exceeded {timeout_seconds:.1f}s hard timeout provider={provider_name}"
        )

    try:
        result = result_queue.get_nowait()
    except queue.Empty as error:
        raise OSError(f"OpenRouter provider attempt exited without a response provider={provider_name}") from error
    if result["status"] == "ok":
        return result["content"]
    error_type = result.get("error_type", "RuntimeError")
    message = result.get("error", "OpenRouter provider attempt failed")
    if error_type == "ValueError":
        raise ValueError(message)
    if error_type == "TimeoutError":
        raise TimeoutError(message)
    raise OSError(f"{error_type}: {message}")


def _openrouter_chat_content_worker(
    payload: dict[str, Any],
    timeout_seconds: float,
    result_queue: multiprocessing.Queue,
) -> None:
    try:
        result_queue.put({"status": "ok", "content": _openrouter_chat_content(payload, timeout_seconds=timeout_seconds)})
    except BaseException as error:
        result_queue.put({"status": "error", "error_type": type(error).__name__, "error": str(error)})


def _progress_writer(progress_stream: TextIO | None) -> Callable[[str], None]:
    if progress_stream is None:
        return lambda message: None

    def write(message: str) -> None:
        progress_stream.write(f"[synthetic-gate] {message}\n")
        progress_stream.flush()

    return write


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


def _candidate_base_row(
    record: dict[str, Any],
    index: int,
    packet: dict[str, Any],
    *,
    candidate_seed: int | None,
    teacher_temperature: float,
    teacher_top_p: float | None,
    teacher_repetition_penalty: float | None,
    teacher_frequency_penalty: float | None,
    teacher_presence_penalty: float | None,
    replayed_candidate: bool,
) -> dict[str, Any]:
    return {
        "input_example_id": record["example_id"],
        "candidate_index": index,
        "packet_id": packet["packet_id"],
        "candidate_id": packet["candidate_id"],
        "target_support_level": record.get("metadata", {})
        .get("generation_metadata", {})
        .get("target_support_level"),
        "target_risk_category": record.get("metadata", {})
        .get("generation_metadata", {})
        .get("target_risk_category"),
        "candidate_seed": candidate_seed,
        "teacher_temperature": teacher_temperature,
        "teacher_top_p": teacher_top_p,
        "teacher_repetition_penalty": teacher_repetition_penalty,
        "teacher_frequency_penalty": teacher_frequency_penalty,
        "teacher_presence_penalty": teacher_presence_penalty,
        "replayed_candidate": replayed_candidate,
    }


def _filtered_record(
    record: dict[str, Any],
    assessment: dict[str, Any],
    judge_result: dict[str, Any],
    *,
    provider: str,
    teacher_model: str | None,
    judge_model: str | None,
    teacher_temperature: float,
    teacher_top_p: float | None,
    teacher_repetition_penalty: float | None,
    teacher_frequency_penalty: float | None,
    teacher_presence_penalty: float | None,
    teacher_thinking: str,
    judge_temperature: float | None,
    judge_thinking: str,
    best_of_n: int,
    base_seed: int | None,
) -> dict[str, Any]:
    filtered = {
        "example_id": record["example_id"].replace("RAW", "FILTERED"),
        "dataset_version": record.get("dataset_version", "tdf_v1.0.0"),
        "source_type": "synthetic_injected_filtered",
        "input": record["input"],
        "output": {"type": "DetectorAssessment", "data": assessment},
        "metadata": dict(record.get("metadata", {})),
    }
    filtered["metadata"]["risk_category"] = assessment["risk_category"]
    filtered["metadata"]["support_level"] = assessment["support_level"]
    filtered["metadata"]["severity"] = assessment["severity"]
    filtered["metadata"].setdefault("report_profile", record["input"]["data"].get("metadata", {}).get("report_profile"))
    filtered["metadata"].setdefault("language", record["input"]["data"].get("metadata", {}).get("language"))
    filtered["metadata"]["validation_metadata"] = {
        "validated": True,
        "validation_pipeline_version": "detector_contract_validation_v1",
        "schema_valid": True,
        "evidence_ids_valid": True,
        "risk_category_valid": True,
        "support_level_valid": True,
        "severity_valid": True,
        "confidence_valid": True,
        "rationale_length_valid": True,
        "risk_language_valid": True,
        "no_outside_knowledge": True,
        "no_hidden_metadata_leakage": True,
        "judge_score": judge_result.get("score"),
        "judge_decision": "keep",
        "rejection_reasons": [],
    }
    generation_metadata = dict(filtered["metadata"].get("generation_metadata", {}))
    generation_metadata.update(
        {
            "teacher_model": teacher_model,
            "teacher_provider_route": _provider_routing_for_run(provider, teacher_model),
            "judge_model": judge_model,
            "judge_provider_route": _provider_routing_for_run(provider, judge_model),
            "generation_parameters": {
                "best_of_n": best_of_n,
                "base_seed": base_seed,
                "teacher_temperature": teacher_temperature,
                "teacher_top_p": teacher_top_p,
                "teacher_repetition_penalty": teacher_repetition_penalty,
                "teacher_frequency_penalty": teacher_frequency_penalty,
                "teacher_presence_penalty": teacher_presence_penalty,
                "teacher_thinking": teacher_thinking,
                "judge_temperature": judge_temperature,
                "judge_thinking": judge_thinking,
            },
        }
    )
    filtered["metadata"]["generation_metadata"] = generation_metadata
    return filtered


def _non_kept_row(candidate_row: dict[str, Any], disposition: str, reason_codes: list[str]) -> dict[str, Any]:
    return {
        "input_example_id": candidate_row["input_example_id"],
        "candidate_index": candidate_row["candidate_index"],
        "packet_id": candidate_row["packet_id"],
        "candidate_id": candidate_row["candidate_id"],
        "assessment_or_parse_status": candidate_row.get("assessment") or candidate_row.get("parse_status"),
        "disposition": disposition,
        "reason_codes": reason_codes,
    }


def _validate_filtered_record(record: dict[str, Any]) -> None:
    if record.get("source_type") != "synthetic_injected_filtered":
        raise ValueError("promoted record must use source_type synthetic_injected_filtered")
    if record.get("input", {}).get("type") != "DetectorPacket":
        raise ValueError("promoted record input wrapper type must be DetectorPacket")
    if record.get("output", {}).get("type") != "DetectorAssessment":
        raise ValueError("promoted record output wrapper type must be DetectorAssessment")
    packet = record["input"].get("data")
    validate_detector_packet(packet)
    parse_and_validate_detector_assessment(json.dumps(record["output"].get("data"), ensure_ascii=False), packet)


def _select_promoted_candidate(
    judged_candidates: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    if any("batched_judge_selected" in candidate[0] for candidate in judged_candidates):
        batched_selected = [
            candidate
            for candidate in judged_candidates
            if candidate[0].get("batched_judge_selected") is True
        ]
        if not batched_selected:
            return None
        eligible_selected = [
            candidate for candidate in batched_selected if not _promotion_rejection_reason_codes(candidate[0])
        ]
        if not eligible_selected:
            return _best_eligible_candidate(judged_candidates)
        selected = eligible_selected[0]
        best_target_match = _best_target_support_match(judged_candidates)
        if best_target_match is not None and not _target_support_level_match(selected):
            return best_target_match
        return selected
    return _best_eligible_candidate(judged_candidates)


def _best_eligible_candidate(
    judged_candidates: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    eligible = [candidate for candidate in judged_candidates if not _promotion_rejection_reason_codes(candidate[0])]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda candidate: (
            _target_support_level_match(candidate),
            _target_risk_category_match(candidate),
            candidate[0].get("judge_score", 0),
            sum(candidate[0].get("judge_dimension_scores", {}).values()),
            -candidate[0]["candidate_index"],
        ),
    )


def _best_target_support_match(
    judged_candidates: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    eligible_matches = [
        candidate
        for candidate in judged_candidates
        if not _promotion_rejection_reason_codes(candidate[0]) and _target_support_level_match(candidate)
    ]
    if not eligible_matches:
        return None
    return max(
        eligible_matches,
        key=lambda candidate: (
            _target_risk_category_match(candidate),
            candidate[0].get("judge_score", 0),
            sum(candidate[0].get("judge_dimension_scores", {}).values()),
            -candidate[0]["candidate_index"],
        ),
    )


def _target_support_level_match(candidate: tuple[dict[str, Any], dict[str, Any], dict[str, Any]]) -> int:
    row, assessment, _ = candidate
    return int(bool(row.get("target_support_level") and assessment.get("support_level") == row.get("target_support_level")))


def _target_risk_category_match(candidate: tuple[dict[str, Any], dict[str, Any], dict[str, Any]]) -> int:
    row, assessment, _ = candidate
    return int(bool(row.get("target_risk_category") and assessment.get("risk_category") == row.get("target_risk_category")))


def _apply_batched_judge_result(
    judged_candidates: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
    batch_result: dict[str, Any],
) -> None:
    selected_candidate_index = batch_result.get("selected_candidate_index")
    provider_metadata = batch_result.get("_provider_metadata")
    results_by_index = {
        result.get("candidate_index"): result
        for result in batch_result.get("candidate_results", [])
        if isinstance(result, dict)
    }
    for position, (row, assessment, _) in enumerate(judged_candidates):
        judge_result = results_by_index.get(row["candidate_index"])
        if not isinstance(judge_result, dict):
            judge_result = {
                "decision": "reject",
                "score": 0.0,
                "reason_codes": ["missing_batched_judge_result"],
                "dimension_scores": {},
            }
        _apply_judge_result(row, judge_result)
        if provider_metadata:
            row["judge_provider_metadata"] = provider_metadata
        row["batched_judge_selected"] = row["candidate_index"] == selected_candidate_index
        judged_candidates[position] = (row, assessment, judge_result)


def _judge_candidates_individually(
    judged_candidates: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
    judge_client: SyntheticAssessmentJudgeClient,
    *,
    record: dict[str, Any],
    packet: dict[str, Any],
    provider: str,
    judge_model: str | None,
    judge_rubric_version: str,
    judge_temperature: float | None,
    judge_thinking: str,
    judge_prompt_version: str,
    progress: Callable[[str], None],
    raw_record_index: int,
    total_records: int,
) -> int:
    judge_api_calls = 0
    for candidate_position, (row, assessment, _) in enumerate(judged_candidates):
        progress(
            f"judge start record={raw_record_index + 1}/{total_records} "
            f"candidate={candidate_position + 1}/{len(judged_candidates)}"
        )
        judge_result = judge_client.judge(
            SyntheticAssessmentJudgeRequest(
                example_id=record["example_id"],
                packet=packet,
                assessment=assessment,
                provider=provider,
                model=judge_model or "dry_run_no_judge_model",
                rubric_version=judge_rubric_version,
                temperature=judge_temperature,
                thinking=judge_thinking,
                prompt_version=judge_prompt_version,
                progress=progress,
            )
        )
        judge_api_calls += 1
        _apply_judge_result(row, judge_result)
        judged_candidates[candidate_position] = (row, assessment, judge_result)
        progress(
            f"judge done record={raw_record_index + 1}/{total_records} "
            f"candidate={candidate_position + 1}/{len(judged_candidates)} "
            f"decision={row.get('judge_decision')} score={row.get('judge_score')}"
        )
    return judge_api_calls


def _apply_judge_result(row: dict[str, Any], judge_result: dict[str, Any]) -> None:
    decision = judge_result.get("decision")
    if decision not in VALID_JUDGE_DECISIONS:
        decision = "reject"
    row.update(
        {
            "judge_decision": decision,
            "judge_score": judge_result.get("score"),
            "judge_reason_codes": list(judge_result.get("reason_codes", [])),
        }
    )
    if judge_result.get("dimension_scores"):
        row["judge_dimension_scores"] = judge_result["dimension_scores"]
    if judge_result.get("_provider_metadata"):
        row["judge_provider_metadata"] = judge_result["_provider_metadata"]


def _promotion_rejection_reason_codes(candidate_row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if candidate_row.get("judge_decision") != "keep":
        reasons.append(f"judge_decision_{candidate_row.get('judge_decision') or 'missing'}")
    score = candidate_row.get("judge_score")
    score_threshold = _promotion_judge_score_threshold(candidate_row)
    if not isinstance(score, int | float) or isinstance(score, bool) or score < score_threshold:
        reasons.append("judge_score_below_promotion_threshold")
    dimension_scores = candidate_row.get("judge_dimension_scores", {})
    if any(dimension_scores.get(dimension) != 5 for dimension in PROMOTION_HARD_JUDGE_DIMENSIONS):
        reasons.append("judge_hard_dimension_below_promotion_threshold")
    return reasons


def _promotion_judge_score_threshold(candidate_row: dict[str, Any]) -> float:
    if (
        candidate_row.get("target_support_level") == "weakly_supported"
        and candidate_row.get("assessment_support_level") == "weakly_supported"
    ):
        return PROMOTION_WEAK_TARGET_MATCH_JUDGE_SCORE_THRESHOLD
    return PROMOTION_JUDGE_SCORE_THRESHOLD


def _validate_raw_input_records(records: list[dict[str, Any]]) -> None:
    for record in records:
        if record.get("source_type") != "synthetic_injected_raw":
            raise ValueError("input_jsonl must contain only synthetic_injected_raw records")
        if record.get("input", {}).get("type") != "DetectorPacket":
            raise ValueError("input wrapper type must be DetectorPacket")
        if "output" in record:
            raise ValueError("synthetic_injected_raw gate input must not already contain output")
        validate_detector_packet(record["input"].get("data"))


def _select_records(records: list[dict[str, Any]], *, example_ids: list[str] | tuple[str, ...] | None, limit: int | None) -> list[dict[str, Any]]:
    selected = records
    if example_ids:
        wanted = set(example_ids)
        selected = [record for record in selected if record.get("example_id") in wanted]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        selected = selected[:limit]
    return selected


def _manifest(
    *,
    run_dir: Path,
    input_path: Path,
    mode: str,
    best_of_n: int,
    provider: str,
    teacher_model: str | None,
    judge_model: str | None,
    teacher_prompt_version: str,
    judge_prompt_version: str,
    judge_rubric_version: str,
    base_seed: int | None,
    run_purpose: str,
    teacher_temperature: float,
    teacher_top_p: float | None,
    teacher_repetition_penalty: float | None,
    teacher_frequency_penalty: float | None,
    teacher_presence_penalty: float | None,
    teacher_thinking: str,
    judge_temperature: float | None,
    judge_thinking: str,
    judge_strategy: str,
    input_records: int,
    promoted_records: int,
    judge_replay_candidates_jsonl: Path | None,
) -> dict[str, Any]:
    approved_live_input_records_required = APPROVED_ISSUE_150_FULL_RUN_RECORDS
    if run_purpose == "approved_issue151_canary":
        approved_live_input_records_required = 25
    elif run_purpose == "approved_issue151_standard":
        approved_live_input_records_required = input_records
    elif run_purpose == "approved_corroboration_calibration":
        approved_live_input_records_required = APPROVED_CORROBORATION_CALIBRATION_RECORDS
    trainable_labels_approved = False
    if mode == "live" and run_purpose == "approved_live":
        trainable_labels_approved = (
            input_records == APPROVED_ISSUE_150_FULL_RUN_RECORDS
            and promoted_records == APPROVED_ISSUE_150_FULL_RUN_RECORDS
        )
    elif mode == "live" and run_purpose == "approved_issue151_standard":
        trainable_labels_approved = (
            input_records in APPROVED_ISSUE151_STANDARD_RECORD_OPTIONS
            and promoted_records > 0
            and judge_replay_candidates_jsonl is None
        )
    elif mode == "live" and run_purpose == "approved_corroboration_calibration":
        trainable_labels_approved = (
            input_records == APPROVED_CORROBORATION_CALIBRATION_RECORDS
            and promoted_records > 0
            and judge_replay_candidates_jsonl is None
        )
    return {
        "run_id": run_dir.name,
        "input_jsonl_path": str(input_path),
        "input_jsonl_sha256": _file_sha256(input_path),
        "mode": mode,
        "best_of_n": best_of_n,
        "judge_strategy": judge_strategy,
        "run_purpose": run_purpose,
        "downstream_split_builder_allowed": trainable_labels_approved,
        "trainable_labels_approved": trainable_labels_approved,
        "approved_live_input_records_required": approved_live_input_records_required,
        "input_records": input_records,
        "promoted_records": promoted_records,
        "judge_replay_candidates_jsonl_path": str(judge_replay_candidates_jsonl) if judge_replay_candidates_jsonl else None,
        "judge_replay_candidates_jsonl_sha256": (
            _file_sha256(judge_replay_candidates_jsonl) if judge_replay_candidates_jsonl else None
        ),
        "base_seed": base_seed,
        "seed_policy": "base_plus_raw_record_index_times_best_of_n_plus_candidate_index_v1",
        "teacher": {
            "provider": provider,
            "model": teacher_model,
            "prompt_version": teacher_prompt_version,
            "provider_routing": _provider_routing_for_run(provider, teacher_model),
            "decoding_config": {
                "base_seed": base_seed,
                "temperature": teacher_temperature,
                "top_p": teacher_top_p,
                "repetition_penalty": teacher_repetition_penalty,
                "frequency_penalty": teacher_frequency_penalty,
                "presence_penalty": teacher_presence_penalty,
                "thinking": teacher_thinking,
            },
        },
        "judge": {
            "provider": provider,
            "model": judge_model,
            "prompt_version": judge_prompt_version,
            "rubric_version": judge_rubric_version,
            "provider_routing": _provider_routing_for_run(provider, judge_model),
            "decoding_config": {
                "temperature": judge_temperature,
                "thinking": judge_thinking,
            },
        },
        "same_model_teacher_judge": bool(teacher_model and judge_model and teacher_model == judge_model),
        "validation_pipeline_version": "detector_contract_validation_v1",
        "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "artifact_policy": {
            "canonical_artifacts": [
                "manifest.json",
                "candidates.jsonl",
                "filtered.jsonl",
                "non_kept.jsonl",
                "metrics.json",
            ],
            "raw_model_responses_stored": False,
            "raw_prompts_stored": False,
            "chain_of_thought_stored": False,
        },
    }


def _metrics(
    records: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    filtered_records: list[dict[str, Any]],
    non_kept_rows: list[dict[str, Any]],
    *,
    mode: str,
    judge_api_calls: int,
    judge_strategy: str,
    judge_strategy_effective_counts: Counter[str],
) -> dict[str, Any]:
    deterministic_valid_rows = [row for row in candidate_rows if row.get("deterministic_valid")]
    return {
        "mode": mode,
        "input_records": len(records),
        "input_target_support_levels": dict(
            Counter(record["metadata"]["generation_metadata"]["target_support_level"] for record in records)
        ),
        "input_risk_categories": dict(Counter(record["metadata"]["risk_category"] for record in records)),
        "input_report_profiles": dict(Counter(record["metadata"].get("report_profile") for record in records)),
        "input_base_groups": dict(
            Counter(record["metadata"]["split_metadata"]["derived_from_group_key"] for record in records)
        ),
        "input_synthetic_groups": dict(Counter(record["metadata"]["split_metadata"]["group_key"] for record in records)),
        "input_source_groups": dict(
            Counter(
                record["metadata"]["split_metadata"].get("source_group_key")
                for record in records
                if record["metadata"]["split_metadata"].get("source_group_key")
            )
        ),
        "input_scenario_ids": dict(
            Counter(record["metadata"]["generation_metadata"]["injection_scenario_id"] for record in records)
        ),
        "generated_candidates": len(candidate_rows),
        "teacher_calls": sum(1 for row in candidate_rows if not row.get("replayed_candidate")),
        "judge_calls": judge_api_calls,
        "judge_strategy": judge_strategy,
        "judge_strategy_effective": dict(judge_strategy_effective_counts),
        "teacher_temperatures": dict(Counter(str(row.get("teacher_temperature")) for row in candidate_rows)),
        "teacher_top_ps": dict(Counter(str(row.get("teacher_top_p")) for row in candidate_rows)),
        "teacher_repetition_penalties": dict(
            Counter(str(row.get("teacher_repetition_penalty")) for row in candidate_rows)
        ),
        "teacher_frequency_penalties": dict(
            Counter(str(row.get("teacher_frequency_penalty")) for row in candidate_rows)
        ),
        "teacher_presence_penalties": dict(
            Counter(str(row.get("teacher_presence_penalty")) for row in candidate_rows)
        ),
        "deterministic_valid_candidates": len(deterministic_valid_rows),
        "deterministic_valid_rate": (len(deterministic_valid_rows) / len(candidate_rows)) if candidate_rows else 0.0,
        "materially_distinct_deterministic_valid_candidates": len(
            {
                hashlib.sha256(
                    json.dumps(row["assessment"], ensure_ascii=False, sort_keys=True).encode("utf-8")
                ).hexdigest()
                for row in deterministic_valid_rows
            }
        ),
        "promoted_records": len(filtered_records),
        "non_kept_records": len(non_kept_rows),
        "non_kept_dispositions": dict(Counter(row.get("disposition") for row in non_kept_rows)),
        "validation_reason_codes": dict(Counter(code for row in candidate_rows for code in row.get("validation_reason_codes", []))),
        "judge_decisions": dict(Counter(row.get("judge_decision") for row in candidate_rows if row.get("judge_decision"))),
        "support_levels": dict(Counter(record["metadata"]["support_level"] for record in filtered_records)),
        "risk_categories": dict(Counter(record["metadata"]["risk_category"] for record in filtered_records)),
        "report_profiles": dict(Counter(record["metadata"].get("report_profile") for record in filtered_records)),
        "promoted_target_support_levels": dict(
            Counter(record["metadata"]["generation_metadata"]["target_support_level"] for record in filtered_records)
        ),
        "promoted_scenario_ids": dict(
            Counter(record["metadata"]["generation_metadata"]["injection_scenario_id"] for record in filtered_records)
        ),
        "promoted_base_groups": dict(
            Counter(record["metadata"]["split_metadata"]["derived_from_group_key"] for record in filtered_records)
        ),
        "promoted_synthetic_groups": dict(
            Counter(record["metadata"]["split_metadata"]["group_key"] for record in filtered_records)
        ),
        "promoted_source_groups": dict(
            Counter(
                record["metadata"]["split_metadata"].get("source_group_key")
                for record in filtered_records
                if record["metadata"]["split_metadata"].get("source_group_key")
            )
        ),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _candidate_seeds(base_seed: int | None, *, raw_record_index: int, best_of_n: int) -> tuple[int | None, ...]:
    if base_seed is None:
        return (None,) * best_of_n
    return tuple(base_seed + (raw_record_index * best_of_n) + candidate_index for candidate_index in range(best_of_n))


def _load_judge_replay_candidates(path: Path | str | None) -> dict[str, list[dict[str, Any]]]:
    if path is None:
        return {}
    candidates_by_example_id: dict[str, list[dict[str, Any]]] = {}
    for row in _read_jsonl(Path(path)):
        if row.get("deterministic_valid") is not True or not isinstance(row.get("assessment"), dict):
            raise ValueError("judge replay accepts only parsed deterministic-valid candidate assessments")
        if not isinstance(row.get("candidate_index"), int) or not row.get("input_example_id"):
            raise ValueError("judge replay candidate is missing identity metadata")
        candidates_by_example_id.setdefault(row["input_example_id"], []).append(row)
    for rows in candidates_by_example_id.values():
        rows.sort(key=lambda row: row["candidate_index"])
    return candidates_by_example_id


def _validate_live_input_lock(input_path: Path, *, run_purpose: str, approved_input_sha256: str | None) -> None:
    if run_purpose not in LIVE_RUN_PURPOSES:
        raise ValueError(f"live mode run_purpose must be one of: {sorted(LIVE_RUN_PURPOSES)}")
    if run_purpose in {
        "approved_issue151_canary",
        "approved_issue151_standard",
        "approved_corroboration_calibration",
    }:
        if not approved_input_sha256:
            raise ValueError(f"{run_purpose} requires an approved input SHA-256")
        if _file_sha256(input_path) != approved_input_sha256:
            raise ValueError(f"{run_purpose} input SHA-256 mismatch")
        return
    if run_purpose == "provider_smoke":
        if not approved_input_sha256:
            raise ValueError("provider_smoke requires an approved input SHA-256")
        if _file_sha256(input_path) != approved_input_sha256:
            raise ValueError("provider_smoke input SHA-256 mismatch")
        return
    if input_path.resolve() != APPROVED_ISSUE_149_RAW_INPUT.resolve():
        raise ValueError("live mode requires the approved #149 synthetic_injected_raw input path")
    if approved_input_sha256 != APPROVED_ISSUE_149_RAW_INPUT_SHA256:
        raise ValueError("live mode requires the approved #149 synthetic_injected_raw SHA-256")
    if _file_sha256(input_path) != APPROVED_ISSUE_149_RAW_INPUT_SHA256:
        raise ValueError("approved #149 synthetic_injected_raw content SHA-256 mismatch")


def _validate_live_model_lock(provider: str, teacher_model: str | None, judge_model: str | None) -> None:
    if provider == "openrouter":
        for role, model in [("teacher", teacher_model), ("judge", judge_model)]:
            if model not in OPENROUTER_LOCKED_PROVIDER_MODELS:
                raise ValueError(f"live {role} model has no locked OpenRouter provider route: {model}")
        return
    if provider == "deepseek":
        for role, model in [("teacher", teacher_model), ("judge", judge_model)]:
            if model not in DEEPSEEK_DIRECT_MODELS:
                raise ValueError(f"live {role} model is not an approved direct DeepSeek model: {model}")
        return
    raise ValueError(f"live mode currently supports provider=openrouter or provider=deepseek, got {provider}")


def _validate_live_run_shape(
    run_purpose: str,
    *,
    records_loaded: int,
    best_of_n: int,
    judge_replay_candidates_jsonl: Path | str | None,
) -> None:
    if run_purpose == "capability_preflight" and (records_loaded != 1 or best_of_n != 1):
        raise ValueError("capability_preflight requires exactly one input record and best_of_n=1")
    if run_purpose == "temperature_probe" and records_loaded != 1:
        raise ValueError("temperature_probe requires exactly one input record")
    if run_purpose == "provider_smoke":
        if records_loaded < 1 or records_loaded > 64 or best_of_n > DEFAULT_BEST_OF_N:
            raise ValueError("provider_smoke requires 1-64 input records and best_of_n<=4")
        if judge_replay_candidates_jsonl is not None:
            raise ValueError("provider_smoke does not accept judge replay candidates")
    if run_purpose == "approved_live":
        if records_loaded != APPROVED_ISSUE_150_FULL_RUN_RECORDS or best_of_n != DEFAULT_BEST_OF_N:
            raise ValueError("approved_live requires the full eight-record #149 input and best_of_n=4")
        if judge_replay_candidates_jsonl is not None:
            raise ValueError("approved_live does not accept judge replay candidates")
    if run_purpose == "approved_issue151_canary":
        if records_loaded != 25 or best_of_n != DEFAULT_BEST_OF_N:
            raise ValueError("approved_issue151_canary requires exactly 25 input records and best_of_n=4")
        if judge_replay_candidates_jsonl is not None:
            raise ValueError("approved_issue151_canary does not accept judge replay candidates")
    if run_purpose == "approved_issue151_standard":
        if records_loaded not in APPROVED_ISSUE151_STANDARD_RECORD_OPTIONS or best_of_n != DEFAULT_BEST_OF_N:
            raise ValueError(
                "approved_issue151_standard requires "
                f"{sorted(APPROVED_ISSUE151_STANDARD_RECORD_OPTIONS)} input records and best_of_n=4"
            )
        if judge_replay_candidates_jsonl is not None:
            raise ValueError("approved_issue151_standard does not accept judge replay candidates")
    if run_purpose == "approved_corroboration_calibration":
        if (
            records_loaded != APPROVED_CORROBORATION_CALIBRATION_RECORDS
            or best_of_n != DEFAULT_BEST_OF_N
        ):
            raise ValueError(
                "approved_corroboration_calibration requires exactly "
                f"{APPROVED_CORROBORATION_CALIBRATION_RECORDS} input records and best_of_n=4"
            )
        if judge_replay_candidates_jsonl is not None:
            raise ValueError("approved_corroboration_calibration does not accept judge replay candidates")


def _validate_prompt_versions(*, teacher_prompt_version: str, judge_prompt_version: str) -> None:
    _teacher_system_prompt(teacher_prompt_version)
    _judge_system_prompt(judge_prompt_version)


def _validate_decoding_config(
    *,
    teacher_temperature: float,
    teacher_top_p: float | None,
    teacher_repetition_penalty: float | None,
    teacher_frequency_penalty: float | None,
    teacher_presence_penalty: float | None,
    teacher_thinking: str,
    judge_temperature: float | None,
    judge_thinking: str,
) -> None:
    if teacher_thinking not in VALID_THINKING_MODES or judge_thinking not in VALID_THINKING_MODES:
        raise ValueError(f"thinking mode must be one of: {sorted(VALID_THINKING_MODES)}")
    if teacher_thinking == "enabled":
        raise ValueError("teacher thinking must be disabled so temperature remains effective")
    if judge_thinking == "enabled" and judge_temperature is not None:
        raise ValueError("judge temperature must be omitted when judge thinking is enabled")
    if teacher_temperature < 0 or teacher_temperature > 2:
        raise ValueError("teacher_temperature must be between 0 and 2")
    if teacher_top_p is not None and (teacher_top_p <= 0 or teacher_top_p > 1):
        raise ValueError("teacher_top_p must be > 0 and <= 1")
    if teacher_repetition_penalty is not None and (
        teacher_repetition_penalty <= 0 or teacher_repetition_penalty > 2
    ):
        raise ValueError("teacher_repetition_penalty must be > 0 and <= 2")
    if teacher_frequency_penalty is not None and (
        teacher_frequency_penalty < -2 or teacher_frequency_penalty > 2
    ):
        raise ValueError("teacher_frequency_penalty must be between -2 and 2")
    if teacher_presence_penalty is not None and (
        teacher_presence_penalty < -2 or teacher_presence_penalty > 2
    ):
        raise ValueError("teacher_presence_penalty must be between -2 and 2")
    if judge_temperature is not None and (judge_temperature < 0 or judge_temperature > 2):
        raise ValueError("judge_temperature must be between 0 and 2")


def _locked_openrouter_provider_routing(provider: str, model: str) -> dict[str, Any]:
    if provider != "openrouter" or model not in OPENROUTER_LOCKED_PROVIDER_MODELS:
        raise ValueError(f"model has no locked OpenRouter provider route: {provider}/{model}")
    return {
        "order": list(OPENROUTER_LOCKED_PROVIDER_ROUTING["order"]),
        "only": list(OPENROUTER_LOCKED_PROVIDER_ROUTING["only"]),
        "allow_fallbacks": OPENROUTER_LOCKED_PROVIDER_ROUTING["allow_fallbacks"],
        "require_parameters": OPENROUTER_LOCKED_PROVIDER_ROUTING["require_parameters"],
    }


def _single_openrouter_provider_routing(provider: str, model: str, provider_name: str) -> dict[str, Any]:
    locked_route = _locked_openrouter_provider_routing(provider, model)
    if provider_name not in locked_route["only"]:
        raise ValueError(f"provider is not in the locked OpenRouter provider route: {provider_name}")
    return {
        "order": [provider_name],
        "only": [provider_name],
        "allow_fallbacks": False,
        "require_parameters": locked_route["require_parameters"],
    }


def _provider_routing_for_run(provider: str, model: str | None) -> dict[str, Any] | None:
    if provider != "openrouter" or model not in OPENROUTER_LOCKED_PROVIDER_MODELS:
        return None
    return _locked_openrouter_provider_routing(provider, model)


def _openrouter_decoding_payload(
    temperature: float | None,
    thinking: str,
    *,
    seed: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"reasoning": {"enabled": thinking == "enabled"}}
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if repetition_penalty is not None:
        payload["repetition_penalty"] = repetition_penalty
    if frequency_penalty is not None:
        payload["frequency_penalty"] = frequency_penalty
    if presence_penalty is not None:
        payload["presence_penalty"] = presence_penalty
    if seed is not None:
        payload["seed"] = seed
    return payload


def _deepseek_decoding_payload(
    temperature: float | None,
    thinking: str,
    *,
    top_p: float | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"thinking": {"type": thinking}}
    if thinking == "enabled":
        return payload
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if frequency_penalty is not None:
        payload["frequency_penalty"] = frequency_penalty
    if presence_penalty is not None:
        payload["presence_penalty"] = presence_penalty
    return payload


def _run_errors(*, mode: str, run_purpose: str, metrics: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if mode == "live" and run_purpose == "capability_preflight":
        if metrics["teacher_calls"] != 1:
            errors.append("capability_preflight must complete exactly one teacher call")
        if metrics["judge_calls"] != 1:
            errors.append("capability_preflight must complete exactly one judge call")
    return errors


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value.strip().lower()).strip("_")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Label and judge staged synthetic DetectorPacket records.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--mode", choices=sorted(VALID_MODES), required=True)
    parser.add_argument("--provider", default="openrouter")
    parser.add_argument("--teacher-model")
    parser.add_argument("--judge-model")
    parser.add_argument("--teacher-prompt-version", default=DEFAULT_TEACHER_PROMPT_VERSION)
    parser.add_argument("--judge-prompt-version", default=DEFAULT_JUDGE_PROMPT_VERSION)
    parser.add_argument("--run-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--example-id", action="append", dest="example_ids")
    parser.add_argument("--best-of-n", type=int, default=DEFAULT_BEST_OF_N)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--base-seed", type=int)
    parser.add_argument("--run-purpose", default="development")
    parser.add_argument("--approved-input-sha256")
    parser.add_argument("--teacher-temperature", type=float, default=0.2)
    parser.add_argument("--teacher-top-p", type=float)
    parser.add_argument("--teacher-repetition-penalty", type=float)
    parser.add_argument("--teacher-frequency-penalty", type=float)
    parser.add_argument("--teacher-presence-penalty", type=float)
    parser.add_argument("--teacher-thinking", choices=sorted(VALID_THINKING_MODES), default="disabled")
    parser.add_argument("--judge-temperature", type=float)
    parser.add_argument("--judge-thinking", choices=sorted(VALID_THINKING_MODES), default="disabled")
    parser.add_argument("--judge-strategy", choices=sorted(VALID_JUDGE_STRATEGIES), default="non_batched_v1")
    parser.add_argument("--judge-replay-candidates-jsonl", type=Path)
    parser.add_argument("--live-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--teacher-concurrency", type=int, default=1)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args(argv)
    result = run_synthetic_detector_assessment_gate(
        input_jsonl=args.input_jsonl,
        output_root=args.output_root,
        mode=args.mode,
        provider=args.provider,
        teacher_model=args.teacher_model,
        judge_model=args.judge_model,
        teacher_prompt_version=args.teacher_prompt_version,
        judge_prompt_version=args.judge_prompt_version,
        run_id=args.run_id,
        limit=args.limit,
        example_ids=args.example_ids,
        best_of_n=args.best_of_n,
        seed=args.seed,
        base_seed=args.base_seed,
        run_purpose=args.run_purpose,
        approved_input_sha256=args.approved_input_sha256,
        teacher_temperature=args.teacher_temperature,
        teacher_top_p=args.teacher_top_p,
        teacher_repetition_penalty=args.teacher_repetition_penalty,
        teacher_frequency_penalty=args.teacher_frequency_penalty,
        teacher_presence_penalty=args.teacher_presence_penalty,
        teacher_thinking=args.teacher_thinking,
        judge_temperature=(
            args.judge_temperature
            if args.judge_temperature is not None or args.judge_thinking == "enabled"
            else 0.0
        ),
        judge_thinking=args.judge_thinking,
        judge_strategy=args.judge_strategy,
        judge_replay_candidates_jsonl=args.judge_replay_candidates_jsonl,
        live_timeout_seconds=args.live_timeout_seconds,
        teacher_concurrency=args.teacher_concurrency,
        progress_stream=None if args.no_progress else sys.stderr,
    )
    print(json.dumps({"status": result.status, "run_dir": str(result.run_dir)}, sort_keys=True))
    return 0 if result.status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
