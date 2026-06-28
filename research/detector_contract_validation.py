from __future__ import annotations

import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from vinacount.detector_contract import (
    ALLOWED_EVIDENCE_REF_ROLES,
    ALLOWED_EVIDENCE_REF_TYPES,
    ALLOWED_SEVERITIES,
    ALLOWED_SIGNAL_STATUSES,
    ALLOWED_SUPPORT_LEVELS,
    DETECTOR_PACKET_CAPS,
    DetectorAdapter,
    contains_hidden_injection_leakage,
    contains_outside_packet_reasoning,
    contains_prohibited_detector_visible_payload,
    contains_prohibited_text,
    detector_assessment_json_schema,
    enrich_detector_packet_evidence_roles,
    normalize_json_content,
    parse_and_validate_detector_assessment,
    sentence_count,
    validate_detector_assessment,
    validate_detector_packet,
    validate_evidence_ref,
    visible_packet_evidence_ids,
)


__all__ = [
    "ALLOWED_EVIDENCE_REF_ROLES",
    "ALLOWED_EVIDENCE_REF_TYPES",
    "ALLOWED_SEVERITIES",
    "ALLOWED_SIGNAL_STATUSES",
    "ALLOWED_SUPPORT_LEVELS",
    "DETECTOR_PACKET_CAPS",
    "DetectorAdapter",
    "contains_hidden_injection_leakage",
    "contains_outside_packet_reasoning",
    "contains_prohibited_detector_visible_payload",
    "contains_prohibited_text",
    "detector_assessment_json_schema",
    "enrich_detector_packet_evidence_roles",
    "normalize_json_content",
    "parse_and_validate_detector_assessment",
    "sentence_count",
    "validate_detector_assessment",
    "validate_detector_packet",
    "validate_evidence_ref",
    "visible_packet_evidence_ids",
]
