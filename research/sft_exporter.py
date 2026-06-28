from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research.detector_contract_validation import (
    contains_hidden_injection_leakage,
    contains_outside_packet_reasoning,
    contains_prohibited_detector_visible_payload,
    normalize_json_content,
    validate_detector_assessment,
    validate_detector_packet,
)


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
    "Keep rationale_short to 1–3 sentences."
)
DETECTOR_SFT_SYSTEM_PROMPT_V2 = (
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
    "v2_evidence_bundle": DETECTOR_SFT_SYSTEM_PROMPT_V2,
}
DEFAULT_DETECTOR_SFT_SYSTEM_PROMPT_VERSION = "v1"
DETECTOR_SFT_SYSTEM_PROMPT = DETECTOR_SFT_SYSTEM_PROMPT_V1

DEFAULT_OUTPUT_FILENAME = "detector_sft_chat.jsonl"
SPLIT_FILES = ("train", "validation", "test", "holdout", "excluded")
REAL_MANUAL_SOURCE_TYPE = "human_gold_real_report"


@dataclass(frozen=True)
class Wave4SftExportResult:
    status: str
    output_jsonl: Path
    records_exported: int
    errors: list[str]


def run_sft_exporter(
    split_release_dir: Path | str,
    *,
    output_jsonl: Path | str | None = None,
    system_prompt_version: str = DEFAULT_DETECTOR_SFT_SYSTEM_PROMPT_VERSION,
) -> Wave4SftExportResult:
    release_dir = Path(split_release_dir)
    output_path = Path(output_jsonl) if output_jsonl is not None else release_dir / DEFAULT_OUTPUT_FILENAME
    errors: list[str] = []
    chat_records: list[dict[str, Any]] = []

    try:
        manifest = _read_json(release_dir / "manifest.json")
        allowed_splits, allowed_source_types, excluded_source_types = _export_policy_from_manifest(manifest)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return Wave4SftExportResult("failed", output_path, 0, [str(error)])

    for split in allowed_splits:
        if split not in SPLIT_FILES:
            errors.append(f"manifest sft_export_allowed_splits contains unsupported split: {split}")
            continue
        split_path = release_dir / f"{split}.jsonl"
        if not split_path.exists():
            errors.append(f"{split}.jsonl is required by manifest sft_export_allowed_splits")
            continue
        for line_number, line in enumerate(split_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if _should_skip_default_export(record, split, excluded_source_types):
                    continue
                chat_records.append(
                    _record_to_chat(
                        record,
                        split=split,
                        allowed_source_types=allowed_source_types,
                        excluded_source_types=excluded_source_types,
                        system_prompt_version=system_prompt_version,
                    )
                )
            except (json.JSONDecodeError, ValueError) as error:
                errors.append(f"{split}.jsonl:{line_number}: {error}")

    if errors:
        return Wave4SftExportResult("failed", output_path, 0, errors)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in chat_records),
        encoding="utf-8",
    )
    return Wave4SftExportResult("passed", output_path, len(chat_records), [])


def _export_policy_from_manifest(manifest: dict[str, Any]) -> tuple[list[str], set[str], set[str]]:
    if manifest.get("sft_export_ready") is not True:
        raise ValueError("split release manifest is not SFT-export-ready")
    if manifest.get("integrity_checks", {}).get("validator_status") != "passed":
        raise ValueError("split release manifest integrity_checks.validator_status must be passed")
    allowed_splits = manifest.get("sft_export_allowed_splits")
    allowed_source_types = manifest.get("sft_export_default_source_types")
    excluded_source_types = manifest.get("sft_export_excluded_source_types_by_default", [])
    if not isinstance(allowed_splits, list) or not allowed_splits:
        raise ValueError("manifest sft_export_allowed_splits must be a non-empty list")
    if not isinstance(allowed_source_types, list) or not allowed_source_types:
        raise ValueError("manifest sft_export_default_source_types must be a non-empty list")
    if not isinstance(excluded_source_types, list):
        raise ValueError("manifest sft_export_excluded_source_types_by_default must be a list")
    unsupported_default_splits = sorted(set(allowed_splits) - {"train", "validation", "test"})
    if unsupported_default_splits:
        raise ValueError(f"unsupported default SFT split in manifest: {unsupported_default_splits}")
    if REAL_MANUAL_SOURCE_TYPE in set(allowed_source_types):
        raise ValueError(f"{REAL_MANUAL_SOURCE_TYPE} cannot be exported by default")
    return allowed_splits, set(allowed_source_types), set(excluded_source_types)


def _should_skip_default_export(record: dict[str, Any], split: str, excluded_source_types: set[str]) -> bool:
    source_type = record.get("source_type")
    if source_type not in excluded_source_types:
        return False
    if source_type == REAL_MANUAL_SOURCE_TYPE and split == "train":
        raise ValueError(f"{REAL_MANUAL_SOURCE_TYPE} cannot be exported by default")
    if split == "train":
        raise ValueError(f"source_type is excluded from default SFT export: {source_type}")
    return True


def _record_to_chat(
    record: dict[str, Any],
    *,
    split: str,
    allowed_source_types: set[str],
    excluded_source_types: set[str],
    system_prompt_version: str = DEFAULT_DETECTOR_SFT_SYSTEM_PROMPT_VERSION,
) -> dict[str, Any]:
    _validate_record_source_policy(record, split, allowed_source_types, excluded_source_types)
    packet = record.get("input", {}).get("data")
    assessment = record.get("output", {}).get("data")
    if record.get("input", {}).get("type") != "DetectorPacket":
        raise ValueError("input wrapper type must be DetectorPacket")
    if record.get("output", {}).get("type") != "DetectorAssessment":
        raise ValueError("output wrapper type must be DetectorAssessment")
    if not isinstance(packet, dict):
        raise ValueError("input.data must be a DetectorPacket object")
    if not isinstance(assessment, dict):
        raise ValueError("output.data must be a flattened DetectorAssessment object")

    _validate_detector_visible_payload("DetectorPacket", packet)
    _validate_detector_visible_payload("DetectorAssessment", assessment)
    try:
        validate_detector_packet(packet)
    except ValueError as error:
        raise ValueError(f"input.data must be a valid DetectorPacket JSON object: {error}") from error
    try:
        validate_detector_assessment(assessment, packet)
    except ValueError as error:
        raise ValueError(f"output.data must be valid flattened DetectorAssessment JSON: {error}") from error

    assistant_content = _json_content(assessment)
    _validate_assistant_content_is_json_only(assistant_content)
    user_content = _json_content(packet)
    _validate_message_content("user", user_content)
    _validate_message_content("assistant", assistant_content)

    return {
        "messages": [
            {"role": "system", "content": detector_sft_system_prompt(system_prompt_version)},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": _chat_metadata(record, packet, split),
    }


def _validate_record_source_policy(
    record: dict[str, Any],
    split: str,
    allowed_source_types: set[str],
    excluded_source_types: set[str],
) -> None:
    source_type = record.get("source_type")
    if source_type == REAL_MANUAL_SOURCE_TYPE:
        raise ValueError(f"{REAL_MANUAL_SOURCE_TYPE} cannot be exported by default")
    if source_type in excluded_source_types:
        raise ValueError(f"source_type is excluded from default SFT export: {source_type}")
    if source_type not in allowed_source_types:
        raise ValueError(f"source_type is not allowed for default SFT export: {source_type}")
    split_metadata = record.get("metadata", {}).get("split_metadata", {})
    if split_metadata.get("split") != split:
        raise ValueError("metadata split must match the JSONL split file")
    if split_metadata.get("usable_for_training") is not True:
        raise ValueError("record is not marked usable_for_training")
    if split_metadata.get("exclusion_reason") is not None:
        raise ValueError("excluded records cannot be exported for default SFT")


def _validate_detector_visible_payload(label: str, payload: Any) -> None:
    if contains_prohibited_detector_visible_payload(payload):
        raise ValueError(f"{label} contains prohibited hidden or raw metadata")
    if contains_hidden_injection_leakage(payload):
        raise ValueError(f"{label} contains hidden injection leakage")
    if contains_outside_packet_reasoning(payload):
        raise ValueError(f"{label} contains outside-packet reasoning")
    text = json.dumps(payload, ensure_ascii=False).casefold()
    if "final report" in text or "final_report" in text:
        raise ValueError(f"{label} contains final report text or references")


def _validate_assistant_content_is_json_only(content: str) -> None:
    if normalize_json_content(content) != content:
        raise ValueError("assistant content must not contain markdown fences")
    json.loads(content)


def _validate_message_content(role: str, content: str) -> None:
    parsed = json.loads(content)
    if contains_prohibited_detector_visible_payload(parsed):
        raise ValueError(f"{role} message contains prohibited hidden or raw metadata")
    if contains_hidden_injection_leakage(parsed) or contains_outside_packet_reasoning(parsed):
        raise ValueError(f"{role} message contains non-detector-visible leakage")


def _chat_metadata(record: dict[str, Any], packet: dict[str, Any], split: str) -> dict[str, Any]:
    metadata = record.get("metadata", {})
    split_metadata = metadata.get("split_metadata", {})
    chat_metadata = {
        "example_id": record.get("example_id"),
        "dataset_version": record.get("dataset_version"),
        "source_type": record.get("source_type"),
        "split": split,
        "risk_category": metadata.get("risk_category"),
        "support_level": metadata.get("support_level"),
        "report_profile": metadata.get("report_profile"),
        "report_id": packet.get("report_id"),
    }
    for field in [
        "company_key",
        "period_key",
        "group_key",
        "source_file_sha256",
        "normalized_text_hash",
        "table_content_hash",
        "derived_from_report_artifact_id",
        "derived_from_source_document_id",
        "decontamination_group_id",
        "packet_content_hash",
        "assessment_rationale_hash",
    ]:
        if split_metadata.get(field) is not None:
            chat_metadata[field] = split_metadata[field]
    return chat_metadata


def _json_content(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def detector_sft_system_prompt(version: str = DEFAULT_DETECTOR_SFT_SYSTEM_PROMPT_VERSION) -> str:
    try:
        return DETECTOR_SFT_SYSTEM_PROMPTS[version]
    except KeyError as error:
        raise ValueError(
            f"unsupported detector SFT system prompt version: {version}; "
            f"expected one of {sorted(DETECTOR_SFT_SYSTEM_PROMPTS)}"
        ) from error


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Wave 4 Detector Split Release records as chat-style SFT JSONL.")
    parser.add_argument("split_release_dir", type=Path)
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument(
        "--system-prompt-version",
        choices=sorted(DETECTOR_SFT_SYSTEM_PROMPTS),
        default=DEFAULT_DETECTOR_SFT_SYSTEM_PROMPT_VERSION,
    )
    args = parser.parse_args()

    result = run_sft_exporter(
        args.split_release_dir,
        output_jsonl=args.output_jsonl,
        system_prompt_version=args.system_prompt_version,
    )
    print(
        json.dumps(
            {
                "status": result.status,
                "output_jsonl": str(result.output_jsonl),
                "records_exported": result.records_exported,
                "errors": result.errors,
            },
            sort_keys=True,
        )
    )
    raise SystemExit(0 if result.status == "passed" else 1)


if __name__ == "__main__":
    main()
