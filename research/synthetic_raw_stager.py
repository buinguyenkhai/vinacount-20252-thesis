from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from research.synthetic_raw_candidate_validation import (
    validate_synthetic_raw_candidate,
)


ARTIFACT_CONTRACT_VERSION = "synthetic_injected_raw_staging_v1"
DEFAULT_OUTPUT_DIR = Path("data/synthetic/detector_assessment_gate")
RAW_JSONL_FILENAME = "synthetic_injected_raw.jsonl"
@dataclass(frozen=True)
class Wave4SyntheticRawStagingResult:
    status: str
    output_dir: Path
    output_jsonl: Path
    manifest_path: Path
    metrics_path: Path
    records_written: int
    errors: list[str]


def run_synthetic_raw_stager(
    *,
    input_jsonl: Path | str,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    run_id: str = "synthetic_raw_staging",
) -> Wave4SyntheticRawStagingResult:
    input_path = Path(input_jsonl)
    output_dir_path = Path(output_dir)
    output_jsonl = output_dir_path / RAW_JSONL_FILENAME
    manifest_path = output_dir_path / "manifest.json"
    metrics_path = output_dir_path / "metrics.json"

    errors: list[str] = []
    raw_records: list[dict[str, Any]] = []
    try:
        staged_records = _read_jsonl(input_path)
    except (OSError, json.JSONDecodeError) as error:
        return Wave4SyntheticRawStagingResult(
            "failed", output_dir_path, output_jsonl, manifest_path, metrics_path, 0, [str(error)]
        )

    for index, record in enumerate(staged_records, start=1):
        try:
            raw_records.append(_validate_and_canonicalize_record(record))
        except ValueError as error:
            errors.append(f"input:{index}: {error}")

    if errors:
        return Wave4SyntheticRawStagingResult("failed", output_dir_path, output_jsonl, manifest_path, metrics_path, 0, errors)

    output_dir_path.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_jsonl, raw_records)
    manifest = _manifest(
        run_id=run_id,
        input_path=input_path,
        output_jsonl=output_jsonl,
        records=raw_records,
    )
    metrics = _metrics(raw_records)
    _write_json(manifest_path, manifest)
    _write_json(metrics_path, metrics)
    return Wave4SyntheticRawStagingResult(
        "passed",
        output_dir_path,
        output_jsonl,
        manifest_path,
        metrics_path,
        len(raw_records),
        [],
    )


def _validate_and_canonicalize_record(record: dict[str, Any]) -> dict[str, Any]:
    validate_synthetic_raw_candidate(record)
    packet = record["input"]["data"]
    metadata = record["metadata"]

    canonical = {
        "example_id": record.get("example_id"),
        "dataset_version": record.get("dataset_version", "tdf_v1.0.0"),
        "source_type": "synthetic_injected_raw",
        "input": {"type": "DetectorPacket", "data": packet},
        "metadata": metadata,
    }
    if not canonical["example_id"]:
        raise ValueError("example_id is required")
    return canonical


def _manifest(*, run_id: str, input_path: Path, output_jsonl: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "created_by": "research.synthetic_raw_stager",
        "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
        "input_jsonl_path": str(input_path),
        "input_jsonl_sha256": _file_sha256(input_path),
        "output_jsonl": str(output_jsonl),
        "records_written": len(records),
        "source_type": "synthetic_injected_raw",
        "artifact_policy": {
            "detector_assessments_stored": False,
            "raw_model_responses_stored": False,
            "raw_prompts_stored": False,
            "chain_of_thought_stored": False,
        },
    }


def _metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records_written": len(records),
        "source_types": dict(Counter(record["source_type"] for record in records)),
        "risk_categories": dict(Counter(record["metadata"]["generation_metadata"]["target_risk_category"] for record in records)),
        "target_support_levels": dict(Counter(record["metadata"]["generation_metadata"]["target_support_level"] for record in records)),
        "report_profiles": dict(Counter(record["input"]["data"].get("metadata", {}).get("report_profile") for record in records)),
        "decontamination_groups": len({record["metadata"]["split_metadata"]["group_key"] for record in records}),
        "source_groups": dict(
            Counter(
                record["metadata"]["split_metadata"].get("source_group_key")
                for record in records
                if record["metadata"]["split_metadata"].get("source_group_key")
            )
        ),
        "base_groups": dict(Counter(record["metadata"]["split_metadata"]["derived_from_group_key"] for record in records)),
        "synthetic_groups": dict(Counter(record["metadata"]["split_metadata"]["group_key"] for record in records)),
        "scenario_ids": dict(Counter(record["metadata"]["generation_metadata"]["injection_scenario_id"] for record in records)),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise json.JSONDecodeError(f"line {line_number}: {error.msg}", error.doc, error.pos) from error
    return rows


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records), encoding="utf-8")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage synthetic DetectorPacket records as synthetic_injected_raw gate input.")
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path)
    parser.add_argument("--run-id", default="synthetic_raw_staging")
    args = parser.parse_args()

    result = run_synthetic_raw_stager(
        input_jsonl=args.input_jsonl,
        output_dir=args.output_dir,
        run_id=args.run_id,
    )
    print(
        json.dumps(
            {
                "status": result.status,
                "output_dir": str(result.output_dir),
                "output_jsonl": str(result.output_jsonl),
                "records_written": result.records_written,
                "errors": result.errors,
            },
            sort_keys=True,
        )
    )
    raise SystemExit(0 if result.status == "passed" else 1)


if __name__ == "__main__":
    main()
