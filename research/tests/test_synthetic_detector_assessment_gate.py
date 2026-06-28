import io
import hashlib
import http.client
import json
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from research.synthetic_detector_assessment_gate import (
    APPROVED_CORROBORATION_CALIBRATION_RECORDS,
    APPROVED_ISSUE_149_RAW_INPUT,
    APPROVED_ISSUE_149_RAW_INPUT_SHA256,
    CORROBORATION_BALANCED_TEACHER_PROMPT_VERSION,
    CORROBORATION_STRICT_TEACHER_PROMPT_VERSION,
    DEEPSEEK_CHAT_COMPLETIONS_URL,
    DeepSeekSyntheticAssessmentJudgeClient,
    DeepSeekSyntheticAssessmentTeacherClient,
    OpenRouterSyntheticAssessmentJudgeClient,
    OpenRouterSyntheticAssessmentTeacherClient,
    SyntheticAssessmentGateResponse,
    SyntheticAssessmentBatchedJudgeRequest,
    SyntheticAssessmentJudgeRequest,
    SyntheticAssessmentTeacherRequest,
    _openrouter_chat_content_with_hard_timeout,
    _select_promoted_candidate,
    _support_level_calibration_prompt,
    _validate_live_run_shape,
    run_synthetic_detector_assessment_gate,
)


class SyntheticDetectorAssessmentGateTest(unittest.TestCase):
    def test_corroboration_prompt_distinguishes_trigger_magnitude_from_support_sufficiency(self):
        prompt = _support_level_calibration_prompt(CORROBORATION_STRICT_TEACHER_PROMPT_VERSION)

        self.assertIn("trigger magnitude", prompt)
        self.assertIn("do not independently", prompt)
        self.assertIn("exactly one independent signal", prompt)

    def test_balanced_corroboration_prompt_defines_independent_second_signal(self):
        prompt = _support_level_calibration_prompt(CORROBORATION_BALANCED_TEACHER_PROMPT_VERSION)

        self.assertIn("distinct signal_id", prompt)
        self.assertIn("non-primary-trigger evidence item", prompt)
        self.assertIn("do not require a separate narrative note", prompt)

    def test_approved_corroboration_calibration_has_a_locked_run_shape(self):
        _validate_live_run_shape(
            "approved_corroboration_calibration",
            records_loaded=APPROVED_CORROBORATION_CALIBRATION_RECORDS,
            best_of_n=4,
            judge_replay_candidates_jsonl=None,
        )

        with self.assertRaisesRegex(ValueError, "exactly 76"):
            _validate_live_run_shape(
                "approved_corroboration_calibration",
                records_loaded=75,
                best_of_n=4,
                judge_replay_candidates_jsonl=None,
            )

    def test_cli_dry_run_writes_only_five_canonical_artifacts_without_model_outputs(self):
        raw_record = _raw_record()

        with tempfile.TemporaryDirectory() as temp_dir:
            input_jsonl = Path(temp_dir) / "synthetic_injected_raw.jsonl"
            _write_jsonl(input_jsonl, [raw_record])
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "research.synthetic_detector_assessment_gate",
                    "--input-jsonl",
                    str(input_jsonl),
                    "--output-root",
                    str(Path(temp_dir) / "gate"),
                    "--mode",
                    "dry_run",
                    "--run-id",
                    "cli-dry-run",
                ],
                cwd=Path.cwd(),
                text=True,
                capture_output=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            run_dir = Path(temp_dir) / "gate" / "cli-dry-run"
            self.assertEqual(
                sorted(path.name for path in run_dir.iterdir()),
                ["candidates.jsonl", "filtered.jsonl", "manifest.json", "metrics.json", "non_kept.jsonl"],
            )
            self.assertEqual(_read_jsonl(run_dir / "candidates.jsonl"), [])
            self.assertEqual(_read_jsonl(run_dir / "filtered.jsonl"), [])
            self.assertEqual(_read_jsonl(run_dir / "non_kept.jsonl"), [])
            manifest = _read_json(run_dir / "manifest.json")
            self.assertFalse(manifest["artifact_policy"]["raw_model_responses_stored"])
            self.assertFalse(manifest["artifact_policy"]["raw_prompts_stored"])
            self.assertFalse(manifest["artifact_policy"]["chain_of_thought_stored"])

    def test_fake_mode_promotes_only_deterministic_valid_keep_records(self):
        raw_record = _raw_record()
        assessment = _valid_assessment()

        with tempfile.TemporaryDirectory() as temp_dir:
            input_jsonl = Path(temp_dir) / "synthetic_injected_raw.jsonl"
            _write_jsonl(input_jsonl, [raw_record])

            result = run_synthetic_detector_assessment_gate(
                input_jsonl=input_jsonl,
                output_root=Path(temp_dir) / "gate",
                mode="fake",
                provider="openrouter",
                teacher_model="fake-teacher",
                judge_model="fake-judge",
                teacher_client=FakeTeacherClient({raw_record["example_id"]: [assessment]}),
                judge_client=FakeJudgeClient({assessment["assessment_id"]: ("keep", 0.94, [])}),
                run_id="fake-keep",
            )

            self.assertEqual(result.status, "passed", result.errors)
            run_dir = Path(temp_dir) / "gate" / "fake-keep"
            self.assertEqual(result.run_dir, run_dir)
            self.assertEqual(
                sorted(path.name for path in run_dir.iterdir()),
                ["candidates.jsonl", "filtered.jsonl", "manifest.json", "metrics.json", "non_kept.jsonl"],
            )

            manifest = _read_json(run_dir / "manifest.json")
            self.assertEqual(manifest["mode"], "fake")
            self.assertEqual(manifest["best_of_n"], 4)
            self.assertEqual(manifest["teacher"]["model"], "fake-teacher")
            self.assertEqual(manifest["judge"]["model"], "fake-judge")

            candidates = _read_jsonl(run_dir / "candidates.jsonl")
            self.assertEqual(len(candidates), 1)
            self.assertTrue(candidates[0]["deterministic_valid"])
            self.assertEqual(candidates[0]["judge_decision"], "keep")
            self.assertTrue(candidates[0]["selected_for_promotion"])

            filtered = _read_jsonl(run_dir / "filtered.jsonl")
            self.assertEqual(len(filtered), 1)
            self.assertEqual(filtered[0]["source_type"], "synthetic_injected_filtered")
            self.assertEqual(filtered[0]["output"]["data"], assessment)
            self.assertEqual(filtered[0]["metadata"]["generation_metadata"]["teacher_model"], "fake-teacher")
            self.assertEqual(filtered[0]["metadata"]["generation_metadata"]["judge_model"], "fake-judge")
            self.assertEqual(filtered[0]["metadata"]["validation_metadata"]["judge_decision"], "keep")

            self.assertEqual(_read_jsonl(run_dir / "non_kept.jsonl"), [])
            metrics = _read_json(run_dir / "metrics.json")
            self.assertEqual(metrics["input_records"], 1)
            self.assertEqual(metrics["generated_candidates"], 1)
            self.assertEqual(metrics["promoted_records"], 1)

    def test_fake_mode_keeps_revise_reject_needs_human_review_and_invalid_candidates_out_of_filtered(self):
        raw_record = _raw_record()
        keep = _valid_assessment("ASSESS_KEEP")
        revise = _valid_assessment("ASSESS_REVISE")
        reject = _valid_assessment("ASSESS_REJECT")
        human = _valid_assessment("ASSESS_HUMAN")
        invalid = _valid_assessment("ASSESS_INVALID")
        invalid["cited_evidence_refs"] = [
            {"evidence_ref_type": "tool_result", "ref_id": "INVENTED_TOOL_REF", "role": "supporting"}
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            input_jsonl = Path(temp_dir) / "synthetic_injected_raw.jsonl"
            _write_jsonl(input_jsonl, [raw_record])

            result = run_synthetic_detector_assessment_gate(
                input_jsonl=input_jsonl,
                output_root=Path(temp_dir) / "gate",
                mode="fake",
                provider="openrouter",
                teacher_model="fake-teacher",
                judge_model="fake-judge",
                best_of_n=5,
                teacher_client=FakeTeacherClient({raw_record["example_id"]: [keep, revise, reject, human, invalid]}),
                judge_client=FakeJudgeClient(
                    {
                        "ASSESS_KEEP": ("keep", 0.95, []),
                        "ASSESS_REVISE": ("revise", 0.72, ["borderline_support_level"]),
                        "ASSESS_REJECT": ("reject", 0.1, ["ungrounded_rationale"]),
                        "ASSESS_HUMAN": ("needs_human_review", 0.61, ["profile_nuance"]),
                    }
                ),
                run_id="fake-non-kept",
            )

            self.assertEqual(result.status, "passed", result.errors)
            run_dir = Path(temp_dir) / "gate" / "fake-non-kept"
            filtered = _read_jsonl(run_dir / "filtered.jsonl")
            self.assertEqual([record["output"]["data"]["assessment_id"] for record in filtered], ["ASSESS_KEEP"])

            non_kept = _read_jsonl(run_dir / "non_kept.jsonl")
            self.assertEqual(
                [record["disposition"] for record in non_kept],
                ["revise", "reject", "needs_human_review", "reject"],
            )
            self.assertEqual(non_kept[-1]["reason_codes"], ["invalid_evidence_ids"])

            candidates = _read_jsonl(run_dir / "candidates.jsonl")
            self.assertEqual(len(candidates), 5)
            self.assertFalse(candidates[-1]["deterministic_valid"])
            self.assertIsNone(candidates[-1]["judge_decision"])

            metrics = _read_json(run_dir / "metrics.json")
            self.assertEqual(metrics["judge_decisions"], {"keep": 1, "needs_human_review": 1, "reject": 1, "revise": 1})
            self.assertEqual(metrics["validation_reason_codes"], {"invalid_evidence_ids": 1})

    def test_input_source_type_guard_aborts_before_teacher_generation(self):
        real_record = _raw_record()
        real_record["source_type"] = "human_gold_real_report"
        teacher = CountingTeacherClient()

        with tempfile.TemporaryDirectory() as temp_dir:
            input_jsonl = Path(temp_dir) / "mixed.jsonl"
            _write_jsonl(input_jsonl, [real_record])

            with self.assertRaisesRegex(ValueError, "synthetic_injected_raw"):
                run_synthetic_detector_assessment_gate(
                    input_jsonl=input_jsonl,
                    output_root=Path(temp_dir) / "gate",
                    mode="fake",
                    provider="openrouter",
                    teacher_model="fake-teacher",
                    judge_model="fake-judge",
                    teacher_client=teacher,
                    judge_client=FakeJudgeClient({}),
                    run_id="bad-source",
                )

            self.assertEqual(teacher.call_count, 0)
            self.assertFalse((Path(temp_dir) / "gate" / "bad-source" / "non_kept.jsonl").exists())

    def test_live_mode_rejects_unapproved_input_before_teacher_generation(self):
        teacher = CountingTeacherClient()

        with tempfile.TemporaryDirectory() as temp_dir:
            input_jsonl = Path(temp_dir) / "synthetic_injected_raw.jsonl"
            _write_jsonl(input_jsonl, [_raw_record()])

            with self.assertRaisesRegex(ValueError, "approved #149 synthetic_injected_raw"):
                run_synthetic_detector_assessment_gate(
                    input_jsonl=input_jsonl,
                    output_root=Path(temp_dir) / "gate",
                    mode="live",
                    run_purpose="approved_live",
                    approved_input_sha256="not-the-approved-digest",
                    provider="openrouter",
                    teacher_model="deepseek/deepseek-v4-flash",
                    judge_model="deepseek/deepseek-v4-flash",
                    teacher_client=teacher,
                    judge_client=FakeJudgeClient({}),
                    run_id="unapproved-live-input",
                )

            self.assertEqual(teacher.call_count, 0)

    def test_live_standard_accepts_hundred_record_batch_shape(self):
        raw_record = _raw_record()
        records = [raw_record for _ in range(100)]
        assessment = _valid_assessment()

        with tempfile.TemporaryDirectory() as temp_dir:
            input_jsonl = Path(temp_dir) / "synthetic_injected_raw.jsonl"
            _write_jsonl(input_jsonl, records)

            result = run_synthetic_detector_assessment_gate(
                input_jsonl=input_jsonl,
                output_root=Path(temp_dir) / "gate",
                mode="live",
                run_purpose="approved_issue151_standard",
                approved_input_sha256=_file_sha256(input_jsonl),
                provider="openrouter",
                teacher_model="deepseek/deepseek-v4-flash",
                judge_model="deepseek/deepseek-v4-flash",
                teacher_client=FakeTeacherClient({raw_record["example_id"]: [assessment]}),
                judge_client=FakeJudgeClient({assessment["assessment_id"]: ("keep", 0.94, [])}),
                best_of_n=4,
                run_id="issue151-standard-shape",
            )

            self.assertEqual(result.status, "passed", result.errors)
            manifest = _read_json(result.manifest_path)
            self.assertEqual("approved_issue151_standard", manifest["run_purpose"])
            self.assertEqual(100, manifest["input_records"])
            self.assertEqual(100, manifest["approved_live_input_records_required"])
            self.assertTrue(manifest["downstream_split_builder_allowed"])
            self.assertTrue(manifest["trainable_labels_approved"])

    def test_live_standard_accepts_two_hundred_record_batch_shape(self):
        raw_record = _raw_record()
        records = [raw_record for _ in range(200)]
        assessment = _valid_assessment()

        with tempfile.TemporaryDirectory() as temp_dir:
            input_jsonl = Path(temp_dir) / "synthetic_injected_raw.jsonl"
            _write_jsonl(input_jsonl, records)

            result = run_synthetic_detector_assessment_gate(
                input_jsonl=input_jsonl,
                output_root=Path(temp_dir) / "gate",
                mode="live",
                run_purpose="approved_issue151_standard",
                approved_input_sha256=_file_sha256(input_jsonl),
                provider="openrouter",
                teacher_model="deepseek/deepseek-v4-flash",
                judge_model="deepseek/deepseek-v4-flash",
                teacher_client=FakeTeacherClient({raw_record["example_id"]: [assessment]}),
                judge_client=FakeJudgeClient({assessment["assessment_id"]: ("keep", 0.94, [])}),
                best_of_n=4,
                run_id="issue151-standard-200-shape",
            )

            self.assertEqual(result.status, "passed", result.errors)
            manifest = _read_json(result.manifest_path)
            self.assertEqual("approved_issue151_standard", manifest["run_purpose"])
            self.assertEqual(200, manifest["input_records"])
            self.assertEqual(200, manifest["approved_live_input_records_required"])
            self.assertTrue(manifest["downstream_split_builder_allowed"])
            self.assertTrue(manifest["trainable_labels_approved"])

    def test_live_standard_accepts_eighty_record_tail_batch_shape(self):
        raw_record = _raw_record()
        records = [raw_record for _ in range(80)]
        assessment = _valid_assessment()

        with tempfile.TemporaryDirectory() as temp_dir:
            input_jsonl = Path(temp_dir) / "synthetic_injected_raw.jsonl"
            _write_jsonl(input_jsonl, records)

            result = run_synthetic_detector_assessment_gate(
                input_jsonl=input_jsonl,
                output_root=Path(temp_dir) / "gate",
                mode="live",
                run_purpose="approved_issue151_standard",
                approved_input_sha256=_file_sha256(input_jsonl),
                provider="openrouter",
                teacher_model="deepseek/deepseek-v4-flash",
                judge_model="deepseek/deepseek-v4-flash",
                teacher_client=FakeTeacherClient({raw_record["example_id"]: [assessment]}),
                judge_client=FakeJudgeClient({assessment["assessment_id"]: ("keep", 0.94, [])}),
                best_of_n=4,
                run_id="issue151-standard-tail-shape",
            )

            self.assertEqual(result.status, "passed", result.errors)
            manifest = _read_json(result.manifest_path)
            self.assertEqual("approved_issue151_standard", manifest["run_purpose"])
            self.assertEqual(80, manifest["input_records"])
            self.assertEqual(80, manifest["approved_live_input_records_required"])
            self.assertTrue(manifest["downstream_split_builder_allowed"])
            self.assertTrue(manifest["trainable_labels_approved"])

    def test_live_issue151_standard_manifest_blocks_downstream_when_no_records_promote(self):
        raw_record = _raw_record()
        records = [raw_record for _ in range(80)]

        with tempfile.TemporaryDirectory() as temp_dir:
            input_jsonl = Path(temp_dir) / "synthetic_injected_raw.jsonl"
            _write_jsonl(input_jsonl, records)

            result = run_synthetic_detector_assessment_gate(
                input_jsonl=input_jsonl,
                output_root=Path(temp_dir) / "gate",
                mode="live",
                run_purpose="approved_issue151_standard",
                approved_input_sha256=_file_sha256(input_jsonl),
                provider="openrouter",
                teacher_model="deepseek/deepseek-v4-flash",
                judge_model="deepseek/deepseek-v4-flash",
                teacher_client=CountingTeacherClient(),
                judge_client=FakeJudgeClient({}),
                best_of_n=4,
                run_id="issue151-standard-no-promotions",
            )

            manifest = _read_json(result.manifest_path)
            self.assertEqual("approved_issue151_standard", manifest["run_purpose"])
            self.assertEqual(80, manifest["input_records"])
            self.assertEqual(0, manifest["promoted_records"])
            self.assertFalse(manifest["downstream_split_builder_allowed"])
            self.assertFalse(manifest["trainable_labels_approved"])

    def test_live_provider_smoke_accepts_explicit_hash_without_trainable_approval(self):
        raw_record = _raw_record()
        records = [raw_record, raw_record]
        assessment = _valid_assessment()

        with tempfile.TemporaryDirectory() as temp_dir:
            input_jsonl = Path(temp_dir) / "synthetic_injected_raw.jsonl"
            _write_jsonl(input_jsonl, records)

            result = run_synthetic_detector_assessment_gate(
                input_jsonl=input_jsonl,
                output_root=Path(temp_dir) / "gate",
                mode="live",
                run_purpose="provider_smoke",
                approved_input_sha256=_file_sha256(input_jsonl),
                provider="deepseek",
                teacher_model="deepseek-v4-pro",
                judge_model="deepseek-v4-pro",
                teacher_client=FakeTeacherClient({raw_record["example_id"]: [assessment]}),
                judge_client=FakeJudgeClient({assessment["assessment_id"]: ("keep", 0.94, [])}),
                best_of_n=4,
                run_id="deepseek-provider-smoke",
            )

            self.assertEqual(result.status, "passed", result.errors)
            manifest = _read_json(result.manifest_path)
            self.assertEqual("provider_smoke", manifest["run_purpose"])
            self.assertEqual("deepseek", manifest["teacher"]["provider"])
            self.assertFalse(manifest["trainable_labels_approved"])
            self.assertFalse(manifest["downstream_split_builder_allowed"])

    @mock.patch("research.synthetic_detector_assessment_gate.urllib.request.urlopen")
    def test_openrouter_teacher_request_uses_locked_route_and_explicit_decoding(self, mock_urlopen):
        mock_urlopen.return_value.__enter__.return_value.read.return_value = json.dumps(
            {"choices": [{"message": {"content": json.dumps(_valid_assessment())}}]}
        ).encode("utf-8")

        with mock.patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            OpenRouterSyntheticAssessmentTeacherClient(timeout_seconds=0).generate(
                SyntheticAssessmentTeacherRequest(
                    example_id="SYN_RAW_000001",
                    packet=_valid_packet(),
                    best_of_n=1,
                    provider="openrouter",
                    model="deepseek/deepseek-v4-flash",
                    prompt_version="synthetic_detector_assessment_teacher_v1",
                    seed=150,
                    temperature=0.1,
                    thinking="disabled",
                )
            )

        payload = json.loads(mock_urlopen.call_args.args[0].data)
        self.assertEqual(
            payload["provider"],
            {
                "allow_fallbacks": False,
                "only": ["morph"],
                "order": ["morph"],
                "require_parameters": True,
            },
        )
        self.assertEqual(payload["temperature"], 0.1)
        self.assertNotIn("top_p", payload)
        self.assertNotIn("repetition_penalty", payload)
        self.assertNotIn("frequency_penalty", payload)
        self.assertNotIn("presence_penalty", payload)
        self.assertEqual(payload["seed"], 150)
        self.assertEqual(payload["reasoning"], {"enabled": False})
        self.assertEqual(payload["max_tokens"], 2048)
        user_payload = json.loads(payload["messages"][1]["content"])
        self.assertEqual(user_payload["detector_packet"]["packet_id"], "PACKET_FPT_2024_Q3_001")
        self.assertIn("TOOL_REV_GROWTH_1", user_payload["visible_id_guide"]["allowed_evidence_ref_ids"])
        self.assertIn("TOOL_REV_GROWTH_1", user_payload["visible_id_guide"]["allowed_tool_result_ids"])
        self.assertIn("revenue_growth_high", user_payload["visible_id_guide"]["allowed_signal_ids"])
        self.assertEqual(mock_urlopen.call_args.kwargs["timeout"], 0)

    @mock.patch("research.synthetic_detector_assessment_gate.urllib.request.urlopen")
    def test_openrouter_teacher_request_guides_insufficient_evidence_without_invented_tool_ids(self, mock_urlopen):
        mock_urlopen.return_value.__enter__.return_value.read.return_value = json.dumps(
            {"choices": [{"message": {"content": json.dumps(_valid_assessment())}}]}
        ).encode("utf-8")
        packet = _valid_packet_without_tool_findings()

        with mock.patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            OpenRouterSyntheticAssessmentTeacherClient(timeout_seconds=0).generate(
                SyntheticAssessmentTeacherRequest(
                    example_id="SYN_RAW_000001",
                    packet=packet,
                    best_of_n=1,
                    provider="openrouter",
                    model="deepseek/deepseek-v4-flash",
                    prompt_version="synthetic_detector_assessment_teacher_v1",
                    seed=150,
                    temperature=0.1,
                    thinking="disabled",
                )
            )

        payload = json.loads(mock_urlopen.call_args.args[0].data)
        system_prompt = payload["messages"][0]["content"]
        user_payload = json.loads(payload["messages"][1]["content"])
        guide = user_payload["visible_id_guide"]
        self.assertEqual(payload["max_tokens"], 2048)
        self.assertIn("return insufficient_evidence", system_prompt)
        self.assertIn("Calibrate support_level to the visible evidence", system_prompt)
        self.assertIn("Do not downgrade a strongly satisfied condition", system_prompt)
        self.assertIn("flag is false", system_prompt)
        self.assertIn("do not upgrade to supported from signal_id alone", system_prompt)
        self.assertNotIn("Weak-support calibration", system_prompt)
        self.assertEqual(guide["allowed_tool_result_ids"], [])
        self.assertIn("RULE_REV_GROWTH", guide["allowed_evidence_ref_ids"])
        self.assertIn("FPT_2024_Q3_SYN_REV_001:ROW_REVENUE", guide["allowed_evidence_ref_ids"])
        self.assertIn("revenue_growth_high", guide["allowed_signal_ids"])
        self.assertIn("no tool_result_id", guide["insufficient_evidence_guidance"])

    @mock.patch("research.synthetic_detector_assessment_gate.urllib.request.urlopen")
    def test_openrouter_teacher_weakcal_prompt_version_adds_borderline_guidance(self, mock_urlopen):
        mock_urlopen.return_value.__enter__.return_value.read.return_value = json.dumps(
            {"choices": [{"message": {"content": json.dumps(_valid_assessment())}}]}
        ).encode("utf-8")

        with mock.patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            OpenRouterSyntheticAssessmentTeacherClient(timeout_seconds=0).generate(
                SyntheticAssessmentTeacherRequest(
                    example_id="SYN_RAW_000001",
                    packet=_valid_packet(),
                    best_of_n=1,
                    provider="openrouter",
                    model="deepseek/deepseek-v4-flash",
                    prompt_version="synthetic_detector_assessment_teacher_v2_weakcal",
                    seed=150,
                    temperature=0.6,
                    thinking="disabled",
                )
            )

        payload = json.loads(mock_urlopen.call_args.args[0].data)
        system_prompt = payload["messages"][0]["content"]
        self.assertIn("Weak-support calibration", system_prompt)
        self.assertIn("borderline, partial, single-source", system_prompt)
        self.assertIn("Do not collapse a positive-but-limited signal into insufficient_evidence", system_prompt)

    @mock.patch("research.synthetic_detector_assessment_gate.urllib.request.urlopen")
    def test_openrouter_teacher_request_accepts_optional_sampling_penalties(self, mock_urlopen):
        mock_urlopen.return_value.__enter__.return_value.read.return_value = json.dumps(
            {"choices": [{"message": {"content": json.dumps(_valid_assessment())}}]}
        ).encode("utf-8")

        with mock.patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            OpenRouterSyntheticAssessmentTeacherClient(timeout_seconds=0).generate(
                SyntheticAssessmentTeacherRequest(
                    example_id="SYN_RAW_000001",
                    packet=_valid_packet(),
                    best_of_n=1,
                    provider="openrouter",
                    model="deepseek/deepseek-v4-flash",
                    prompt_version="synthetic_detector_assessment_teacher_v1",
                    seed=150,
                    temperature=0.6,
                    top_p=0.9,
                    repetition_penalty=1.05,
                    frequency_penalty=0.1,
                    presence_penalty=0.05,
                    thinking="disabled",
                )
            )

        payload = json.loads(mock_urlopen.call_args.args[0].data)
        self.assertEqual(payload["temperature"], 0.6)
        self.assertEqual(payload["top_p"], 0.9)
        self.assertEqual(payload["repetition_penalty"], 1.05)
        self.assertEqual(payload["frequency_penalty"], 0.1)
        self.assertEqual(payload["presence_penalty"], 0.05)

    @mock.patch("research.synthetic_detector_assessment_gate.urllib.request.urlopen")
    def test_deepseek_teacher_request_uses_direct_json_object_and_records_cache_usage(self, mock_urlopen):
        mock_urlopen.return_value.__enter__.return_value.read.return_value = json.dumps(
            {
                "model": "deepseek-v4-pro",
                "system_fingerprint": "fp-test",
                "choices": [{"message": {"content": json.dumps(_valid_assessment())}}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                    "prompt_cache_hit_tokens": 80,
                    "prompt_cache_miss_tokens": 20,
                },
            }
        ).encode("utf-8")

        with mock.patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
            responses = DeepSeekSyntheticAssessmentTeacherClient(timeout_seconds=0).generate(
                SyntheticAssessmentTeacherRequest(
                    example_id="SYN_RAW_000001",
                    packet=_valid_packet(),
                    best_of_n=1,
                    provider="deepseek",
                    model="deepseek-v4-pro",
                    prompt_version="synthetic_detector_assessment_teacher_v2_weakcal",
                    seed=150,
                    temperature=0.6,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    thinking="disabled",
                )
            )

        request = mock_urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, DEEPSEEK_CHAT_COMPLETIONS_URL)
        self.assertEqual(payload["model"], "deepseek-v4-pro")
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["temperature"], 0.6)
        self.assertEqual(payload["top_p"], 0.9)
        self.assertNotIn("repetition_penalty", payload)
        self.assertNotIn("seed", payload)
        self.assertEqual(payload["max_tokens"], 2048)
        user_payload = json.loads(payload["messages"][1]["content"])
        self.assertEqual(user_payload["detector_packet"], _valid_packet())
        self.assertIn("visible_id_guide", user_payload)
        output_contract = user_payload["detector_assessment_output_contract"]
        self.assertEqual(output_contract["copy_identity_fields_exactly"]["packet_id"], _valid_packet()["packet_id"])
        self.assertIn("assessment_id", output_contract["top_level_keys_exact"])
        self.assertFalse(output_contract["additional_top_level_keys_allowed"])
        self.assertIn("Authorization", request.headers)
        metadata = responses[0].provider_metadata
        self.assertEqual(metadata["strategy"], "direct_deepseek_chat_completions_v1")
        self.assertEqual(metadata["request_model"], "deepseek-v4-pro")
        self.assertEqual(metadata["response_model"], "deepseek-v4-pro")
        self.assertEqual(metadata["system_fingerprint"], "fp-test")
        self.assertEqual(metadata["cache_usage"]["prompt_cache_hit_tokens"], 80)
        self.assertEqual(metadata["cache_usage"]["prompt_cache_miss_tokens"], 20)
        self.assertEqual(metadata["unsupported_decoding_parameters_omitted"], ["repetition_penalty"])

    @mock.patch("research.synthetic_detector_assessment_gate.time.sleep")
    @mock.patch("research.synthetic_detector_assessment_gate.urllib.request.urlopen")
    def test_deepseek_teacher_retries_transient_transport_error(self, mock_urlopen, mock_sleep):
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            {
                "model": "deepseek-v4-pro",
                "choices": [{"message": {"content": json.dumps(_valid_assessment())}}],
                "usage": {},
            }
        ).encode("utf-8")
        mock_urlopen.side_effect = [urllib.error.URLError("temporary"), response]

        with mock.patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
            responses = DeepSeekSyntheticAssessmentTeacherClient(timeout_seconds=0).generate(
                SyntheticAssessmentTeacherRequest(
                    example_id="SYN_RAW_000001",
                    packet=_valid_packet(),
                    best_of_n=1,
                    provider="deepseek",
                    model="deepseek-v4-pro",
                    prompt_version="synthetic_detector_assessment_teacher_v2_weakcal",
                    seed=150,
                    temperature=0.6,
                    thinking="disabled",
                )
            )

        self.assertEqual(mock_urlopen.call_count, 2)
        self.assertEqual(mock_sleep.call_args.args[0], 1.0)
        self.assertEqual(json.loads(responses[0].content)["assessment_id"], _valid_assessment()["assessment_id"])

    @mock.patch("research.synthetic_detector_assessment_gate.time.sleep")
    @mock.patch("research.synthetic_detector_assessment_gate.urllib.request.urlopen")
    def test_deepseek_teacher_retries_incomplete_chunked_response(self, mock_urlopen, mock_sleep):
        incomplete_response = mock.MagicMock()
        incomplete_response.__enter__.return_value.read.side_effect = http.client.IncompleteRead(b'{"partial":')
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            {
                "model": "deepseek-v4-pro",
                "choices": [{"message": {"content": json.dumps(_valid_assessment())}}],
                "usage": {},
            }
        ).encode("utf-8")
        mock_urlopen.side_effect = [incomplete_response, response]

        with mock.patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
            responses = DeepSeekSyntheticAssessmentTeacherClient(timeout_seconds=0).generate(
                SyntheticAssessmentTeacherRequest(
                    example_id="SYN_RAW_000001",
                    packet=_valid_packet(),
                    best_of_n=1,
                    provider="deepseek",
                    model="deepseek-v4-pro",
                    prompt_version="synthetic_detector_assessment_teacher_v2_weakcal",
                    seed=150,
                    temperature=0.6,
                    thinking="disabled",
                )
            )

        self.assertEqual(mock_urlopen.call_count, 2)
        self.assertEqual(mock_sleep.call_args.args[0], 1.0)
        self.assertEqual(json.loads(responses[0].content)["assessment_id"], _valid_assessment()["assessment_id"])

    @mock.patch("research.synthetic_detector_assessment_gate.urllib.request.urlopen")
    def test_deepseek_batched_judge_request_uses_json_object_and_cache_metadata(self, mock_urlopen):
        judge_result = {
            "selected_candidate_index": 0,
            "decision": "keep",
            "score": 0.95,
            "reason_codes": ["correct_support_level"],
            "candidate_decisions": [
                {
                    "candidate_index": 0,
                    "decision": "keep",
                    "score": 0.95,
                    "reason_codes": ["correct_support_level"],
                    "dimension_scores": {},
                }
            ],
        }
        mock_urlopen.return_value.__enter__.return_value.read.return_value = json.dumps(
            {
                "model": "deepseek-v4-pro",
                "choices": [{"message": {"content": json.dumps(judge_result)}}],
                "usage": {
                    "prompt_tokens": 70,
                    "completion_tokens": 10,
                    "total_tokens": 80,
                    "prompt_cache_hit_tokens": 60,
                    "prompt_cache_miss_tokens": 10,
                },
            }
        ).encode("utf-8")

        with mock.patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}):
            result = DeepSeekSyntheticAssessmentJudgeClient(timeout_seconds=0).judge_batch(
                SyntheticAssessmentBatchedJudgeRequest(
                    example_id="SYN_RAW_000001",
                    packet=_valid_packet(),
                    candidates=[{"candidate_index": 0, "assessment": _valid_assessment()}],
                    provider="deepseek",
                    model="deepseek-v4-pro",
                    rubric_version="synthetic_detector_assessment_judge_rubric_v1",
                    temperature=0.0,
                    thinking="disabled",
                    prompt_version="synthetic_detector_assessment_judge_v2_weakcal",
                )
            )

        payload = json.loads(mock_urlopen.call_args.args[0].data)
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["temperature"], 0.0)
        user_payload = json.loads(payload["messages"][1]["content"])
        self.assertIn("batched_judge_output_contract", user_payload)
        self.assertIn("candidate_results", user_payload["batched_judge_output_contract"]["top_level_keys_exact"])
        self.assertEqual(result["selected_candidate_index"], 0)
        self.assertEqual(result["_provider_metadata"]["cache_usage"]["prompt_cache_hit_tokens"], 60)

    @mock.patch("research.synthetic_detector_assessment_gate.urllib.request.urlopen")
    def test_openrouter_teacher_rotates_to_next_locked_provider_after_timeout(self, mock_urlopen):
        def _response(content: dict):
            response = mock.MagicMock()
            response.__enter__.return_value.read.return_value = json.dumps(
                {"choices": [{"message": {"content": json.dumps(content)}}]}
            ).encode("utf-8")
            return response

        mock_urlopen.side_effect = [
            TimeoutError("provider timed out"),
            _response(_valid_assessment()),
        ]
        progress_messages: list[str] = []

        with mock.patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            responses = OpenRouterSyntheticAssessmentTeacherClient(timeout_seconds=0).generate(
                SyntheticAssessmentTeacherRequest(
                    example_id="SYN_RAW_000001",
                    packet=_valid_packet(),
                    best_of_n=1,
                    provider="openrouter",
                    model="deepseek/deepseek-v4-flash",
                    prompt_version="synthetic_detector_assessment_teacher_v1",
                    seed=150,
                    temperature=0.1,
                    thinking="disabled",
                    progress=progress_messages.append,
                )
            )

        self.assertEqual(len(responses), 1)
        self.assertEqual(mock_urlopen.call_count, 2)
        first_payload = json.loads(mock_urlopen.call_args_list[0].args[0].data)
        second_payload = json.loads(mock_urlopen.call_args_list[1].args[0].data)
        self.assertEqual(first_payload["provider"]["only"], ["morph"])
        self.assertEqual(second_payload["provider"]["only"], ["akashml"])
        self.assertEqual(
            responses[0].provider_metadata,
            {
                "strategy": "client_side_provider_rotation_v1",
                "attempts": [
                    {
                        "provider": "morph",
                        "status": "failed",
                        "elapsed_seconds": mock.ANY,
                        "error_type": "TimeoutError",
                        "error": "provider timed out",
                    },
                    {
                        "provider": "akashml",
                        "status": "succeeded",
                        "elapsed_seconds": mock.ANY,
                    },
                ],
            },
        )
        self.assertTrue(any("provider=morph" in message and "failed" in message for message in progress_messages))
        self.assertTrue(any("provider=akashml" in message and "done" in message for message in progress_messages))

    @mock.patch("research.synthetic_detector_assessment_gate.urllib.request.urlopen")
    def test_openrouter_teacher_rotation_loops_for_two_bounded_passes(self, mock_urlopen):
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            {"choices": [{"message": {"content": json.dumps(_valid_assessment())}}]}
        ).encode("utf-8")
        mock_urlopen.side_effect = [
            TimeoutError("morph timed out"),
            TimeoutError("akashml timed out"),
            TimeoutError("deepinfra timed out"),
            TimeoutError("parasail timed out"),
            response,
        ]

        with mock.patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            OpenRouterSyntheticAssessmentTeacherClient(timeout_seconds=0).generate(
                SyntheticAssessmentTeacherRequest(
                    example_id="SYN_RAW_000001",
                    packet=_valid_packet(),
                    best_of_n=1,
                    provider="openrouter",
                    model="deepseek/deepseek-v4-flash",
                    prompt_version="synthetic_detector_assessment_teacher_v1",
                    seed=150,
                    temperature=0.1,
                    thinking="disabled",
                )
            )

        attempted_providers = [
            json.loads(call.args[0].data)["provider"]["only"][0] for call in mock_urlopen.call_args_list
        ]
        self.assertEqual(attempted_providers, ["morph", "akashml", "deepinfra", "parasail", "morph"])

    def test_temperature_probe_records_derived_candidate_seeds_and_blocks_downstream_use(self):
        raw_record = _raw_record()
        teacher = CapturingTeacherClient(
            [_valid_assessment(f"ASSESS_SEED_{index}") for index in range(4)]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            input_jsonl = Path(temp_dir) / "synthetic_injected_raw.jsonl"
            _write_jsonl(input_jsonl, [raw_record])

            result = run_synthetic_detector_assessment_gate(
                input_jsonl=input_jsonl,
                output_root=Path(temp_dir) / "gate",
                mode="fake",
                run_purpose="temperature_probe",
                provider="openrouter",
                teacher_model="fake-teacher",
                judge_model="fake-judge",
                base_seed=150,
                teacher_client=teacher,
                judge_client=FakeJudgeClient(
                    {f"ASSESS_SEED_{index}": ("keep", 0.95, []) for index in range(4)}
                ),
                run_id="fake-temperature-probe",
            )

            manifest = _read_json(result.manifest_path)
            self.assertEqual(manifest["run_purpose"], "temperature_probe")
            self.assertFalse(manifest["downstream_split_builder_allowed"])
            self.assertFalse(manifest["trainable_labels_approved"])
            self.assertEqual(manifest["base_seed"], 150)
            self.assertEqual(
                manifest["seed_policy"],
                "base_plus_raw_record_index_times_best_of_n_plus_candidate_index_v1",
            )
            self.assertEqual(teacher.candidate_seeds, [[150, 151, 152, 153]])
            self.assertEqual(
                [candidate["candidate_seed"] for candidate in _read_jsonl(result.candidates_path)],
                [150, 151, 152, 153],
            )
            metrics = _read_json(result.metrics_path)
            self.assertEqual(metrics["teacher_calls"], 4)
            self.assertEqual(metrics["judge_calls"], 4)
            self.assertEqual(metrics["teacher_temperatures"], {"0.2": 4})
            self.assertEqual(metrics["materially_distinct_deterministic_valid_candidates"], 4)
            self.assertEqual(metrics["promoted_scenario_ids"], {"INJ_REV_REC_001": 1})
            self.assertEqual(metrics["non_kept_dispositions"], {"not_selected": 3})

    def test_temperature_probe_emits_progress_for_teacher_and_judge_boundaries(self):
        raw_record = _raw_record()
        teacher = CapturingTeacherClient(
            [_valid_assessment(f"ASSESS_PROGRESS_{index}") for index in range(4)]
        )
        progress_stream = io.StringIO()

        with tempfile.TemporaryDirectory() as temp_dir:
            input_jsonl = Path(temp_dir) / "synthetic_injected_raw.jsonl"
            _write_jsonl(input_jsonl, [raw_record])

            run_synthetic_detector_assessment_gate(
                input_jsonl=input_jsonl,
                output_root=Path(temp_dir) / "gate",
                mode="fake",
                run_purpose="temperature_probe",
                provider="openrouter",
                teacher_model="fake-teacher",
                judge_model="fake-judge",
                best_of_n=4,
                base_seed=150,
                teacher_client=teacher,
                judge_client=FakeJudgeClient(
                    {f"ASSESS_PROGRESS_{index}": ("keep", 0.95, []) for index in range(4)}
                ),
                run_id="fake-progress",
                progress_stream=progress_stream,
            )

        progress = progress_stream.getvalue()
        self.assertIn("[synthetic-gate] record 1/1", progress)
        self.assertIn("[synthetic-gate] teacher batch start record=1/1 candidates=4", progress)
        self.assertIn("[synthetic-gate] teacher batch done record=1/1 candidates=4", progress)
        self.assertIn("[synthetic-gate] judge start record=1/1 candidate=1/4", progress)
        self.assertIn("[synthetic-gate] judge done record=1/1 candidate=4/4 decision=keep score=0.95", progress)

    def test_promotion_selects_highest_scoring_threshold_eligible_keep_candidate(self):
        raw_record = _raw_record()
        below_threshold = _valid_assessment("ASSESS_BELOW_THRESHOLD")
        selected = _valid_assessment("ASSESS_SELECTED")

        with tempfile.TemporaryDirectory() as temp_dir:
            input_jsonl = Path(temp_dir) / "synthetic_injected_raw.jsonl"
            _write_jsonl(input_jsonl, [raw_record])

            result = run_synthetic_detector_assessment_gate(
                input_jsonl=input_jsonl,
                output_root=Path(temp_dir) / "gate",
                mode="fake",
                provider="openrouter",
                teacher_model="fake-teacher",
                judge_model="fake-judge",
                teacher_client=FakeTeacherClient(
                    {raw_record["example_id"]: [below_threshold, selected]}
                ),
                judge_client=FakeJudgeClient(
                    {
                        "ASSESS_BELOW_THRESHOLD": ("keep", 0.89, []),
                        "ASSESS_SELECTED": ("keep", 0.95, []),
                    }
                ),
                run_id="threshold-selection",
            )

            filtered = _read_jsonl(result.filtered_path)
            self.assertEqual([record["output"]["data"]["assessment_id"] for record in filtered], ["ASSESS_SELECTED"])
            candidates = _read_jsonl(result.candidates_path)
            self.assertEqual([candidate["selected_for_promotion"] for candidate in candidates], [False, True])
            non_kept = _read_jsonl(result.non_kept_path)
            self.assertEqual(non_kept[0]["disposition"], "not_selected")
            self.assertIn("judge_score_below_promotion_threshold", non_kept[0]["reason_codes"])

    def test_promotion_prefers_target_support_match_among_eligible_candidates(self):
        conservative_row = {
            "candidate_index": 0,
            "target_support_level": "supported",
            "target_risk_category": "revenue_income_recognition_risk",
            "judge_decision": "keep",
            "judge_score": 0.99,
            "judge_dimension_scores": _perfect_dimension_scores(),
        }
        target_match_row = {
            "candidate_index": 1,
            "target_support_level": "supported",
            "target_risk_category": "revenue_income_recognition_risk",
            "judge_decision": "keep",
            "judge_score": 0.95,
            "judge_dimension_scores": _perfect_dimension_scores(),
        }
        conservative_assessment = {
            "support_level": "insufficient_evidence",
            "risk_category": "revenue_income_recognition_risk",
        }
        target_match_assessment = {
            "support_level": "supported",
            "risk_category": "revenue_income_recognition_risk",
        }

        selected = _select_promoted_candidate(
            [
                (conservative_row, conservative_assessment, {"score": 0.99}),
                (target_match_row, target_match_assessment, {"score": 0.95}),
            ]
        )

        self.assertIsNotNone(selected)
        self.assertIs(selected[0], target_match_row)

    def test_promotion_allows_lower_score_for_target_matched_weak_candidate(self):
        weak_target_match_row = {
            "candidate_index": 0,
            "target_support_level": "weakly_supported",
            "target_risk_category": "revenue_income_recognition_risk",
            "assessment_support_level": "weakly_supported",
            "assessment_risk_category": "revenue_income_recognition_risk",
            "judge_decision": "keep",
            "judge_score": 0.70,
            "judge_dimension_scores": _perfect_dimension_scores(),
        }

        selected = _select_promoted_candidate(
            [
                (
                    weak_target_match_row,
                    {"support_level": "weakly_supported", "risk_category": "revenue_income_recognition_risk"},
                    {"score": 0.70},
                )
            ]
        )

        self.assertIsNotNone(selected)
        self.assertIs(selected[0], weak_target_match_row)

    def test_promotion_keeps_default_score_threshold_for_non_weak_target_match(self):
        supported_target_row = {
            "candidate_index": 0,
            "target_support_level": "supported",
            "target_risk_category": "revenue_income_recognition_risk",
            "assessment_support_level": "supported",
            "assessment_risk_category": "revenue_income_recognition_risk",
            "judge_decision": "keep",
            "judge_score": 0.89,
            "judge_dimension_scores": _perfect_dimension_scores(),
        }

        selected = _select_promoted_candidate(
            [
                (
                    supported_target_row,
                    {"support_level": "supported", "risk_category": "revenue_income_recognition_risk"},
                    {"score": 0.89},
                )
            ]
        )

        self.assertIsNone(selected)

    def test_promotion_keeps_hard_dimension_gate_for_lower_score_weak_candidate(self):
        weak_target_match_row = {
            "candidate_index": 0,
            "target_support_level": "weakly_supported",
            "target_risk_category": "revenue_income_recognition_risk",
            "assessment_support_level": "weakly_supported",
            "assessment_risk_category": "revenue_income_recognition_risk",
            "judge_decision": "keep",
            "judge_score": 0.75,
            "judge_dimension_scores": {**_perfect_dimension_scores(), "evidence_grounding": 4},
        }

        selected = _select_promoted_candidate(
            [
                (
                    weak_target_match_row,
                    {"support_level": "weakly_supported", "risk_category": "revenue_income_recognition_risk"},
                    {"score": 0.75},
                )
            ]
        )

        self.assertIsNone(selected)

    def test_batched_judge_invalid_selection_falls_back_to_target_support_match(self):
        rejected_selected_row = {
            "candidate_index": 0,
            "target_support_level": "supported",
            "target_risk_category": "revenue_income_recognition_risk",
            "batched_judge_selected": True,
            "judge_decision": "reject",
            "judge_score": 0.99,
            "judge_dimension_scores": _perfect_dimension_scores(),
        }
        target_match_row = {
            "candidate_index": 1,
            "target_support_level": "supported",
            "target_risk_category": "revenue_income_recognition_risk",
            "batched_judge_selected": False,
            "judge_decision": "keep",
            "judge_score": 0.95,
            "judge_dimension_scores": _perfect_dimension_scores(),
        }

        selected = _select_promoted_candidate(
            [
                (
                    rejected_selected_row,
                    {"support_level": "supported", "risk_category": "revenue_income_recognition_risk"},
                    {"score": 0.99},
                ),
                (
                    target_match_row,
                    {"support_level": "supported", "risk_category": "revenue_income_recognition_risk"},
                    {"score": 0.95},
                ),
            ]
        )

        self.assertIsNotNone(selected)
        self.assertIs(selected[0], target_match_row)

    def test_batched_judge_eligible_selection_falls_back_to_target_support_match(self):
        selected_mismatch_row = {
            "candidate_index": 0,
            "target_support_level": "weakly_supported",
            "target_risk_category": "revenue_income_recognition_risk",
            "batched_judge_selected": True,
            "judge_decision": "keep",
            "judge_score": 0.99,
            "judge_dimension_scores": _perfect_dimension_scores(),
        }
        target_match_row = {
            "candidate_index": 1,
            "target_support_level": "weakly_supported",
            "target_risk_category": "revenue_income_recognition_risk",
            "batched_judge_selected": False,
            "judge_decision": "keep",
            "judge_score": 0.95,
            "judge_dimension_scores": _perfect_dimension_scores(),
        }

        selected = _select_promoted_candidate(
            [
                (
                    selected_mismatch_row,
                    {"support_level": "supported", "risk_category": "revenue_income_recognition_risk"},
                    {"score": 0.99},
                ),
                (
                    target_match_row,
                    {"support_level": "weakly_supported", "risk_category": "revenue_income_recognition_risk"},
                    {"score": 0.95},
                ),
            ]
        )

        self.assertIsNotNone(selected)
        self.assertIs(selected[0], target_match_row)

    def test_batched_judge_target_support_match_keeps_selected_candidate(self):
        selected_target_match_row = {
            "candidate_index": 0,
            "target_support_level": "weakly_supported",
            "target_risk_category": "revenue_income_recognition_risk",
            "batched_judge_selected": True,
            "judge_decision": "keep",
            "judge_score": 0.95,
            "judge_dimension_scores": _perfect_dimension_scores(),
        }
        higher_scoring_mismatch_row = {
            "candidate_index": 1,
            "target_support_level": "weakly_supported",
            "target_risk_category": "revenue_income_recognition_risk",
            "batched_judge_selected": False,
            "judge_decision": "keep",
            "judge_score": 0.99,
            "judge_dimension_scores": _perfect_dimension_scores(),
        }

        selected = _select_promoted_candidate(
            [
                (
                    selected_target_match_row,
                    {"support_level": "weakly_supported", "risk_category": "revenue_income_recognition_risk"},
                    {"score": 0.95},
                ),
                (
                    higher_scoring_mismatch_row,
                    {"support_level": "supported", "risk_category": "revenue_income_recognition_risk"},
                    {"score": 0.99},
                ),
            ]
        )

        self.assertIsNotNone(selected)
        self.assertIs(selected[0], selected_target_match_row)

    def test_live_capability_preflight_manifest_records_route_decoding_and_downstream_block(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_synthetic_detector_assessment_gate(
                input_jsonl=APPROVED_ISSUE_149_RAW_INPUT,
                output_root=Path(temp_dir) / "gate",
                mode="live",
                run_purpose="capability_preflight",
                approved_input_sha256=APPROVED_ISSUE_149_RAW_INPUT_SHA256,
                provider="openrouter",
                teacher_model="deepseek/deepseek-v4-flash",
                judge_model="deepseek/deepseek-v4-flash",
                limit=1,
                best_of_n=1,
                base_seed=150,
                teacher_temperature=0.0,
                teacher_thinking="disabled",
                judge_temperature=0.0,
                judge_thinking="disabled",
                teacher_client=CountingTeacherClient(),
                judge_client=FakeJudgeClient({}),
                run_id="capability-preflight-manifest",
            )

            manifest = _read_json(result.manifest_path)
            expected_route = {
                "allow_fallbacks": True,
                "only": ["morph", "akashml", "deepinfra", "parasail"],
                "order": ["morph", "akashml", "deepinfra", "parasail"],
                "require_parameters": True,
            }
            self.assertEqual(manifest["teacher"]["provider_routing"], expected_route)
            self.assertEqual(manifest["judge"]["provider_routing"], expected_route)
            self.assertEqual(manifest["teacher"]["decoding_config"]["temperature"], 0.0)
            self.assertEqual(manifest["teacher"]["decoding_config"]["thinking"], "disabled")
            self.assertEqual(manifest["judge"]["decoding_config"]["temperature"], 0.0)
            self.assertEqual(manifest["judge"]["decoding_config"]["thinking"], "disabled")
            self.assertFalse(manifest["downstream_split_builder_allowed"])
            self.assertFalse(manifest["trainable_labels_approved"])

    def test_live_mode_rejects_model_without_locked_provider_route_before_teacher_generation(self):
        teacher = CountingTeacherClient()

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "no locked OpenRouter provider route"):
                run_synthetic_detector_assessment_gate(
                    input_jsonl=APPROVED_ISSUE_149_RAW_INPUT,
                    output_root=Path(temp_dir) / "gate",
                    mode="live",
                    run_purpose="capability_preflight",
                    approved_input_sha256=APPROVED_ISSUE_149_RAW_INPUT_SHA256,
                    provider="openrouter",
                    teacher_model="unlocked-teacher",
                    judge_model="deepseek/deepseek-v4-flash",
                    limit=1,
                    best_of_n=1,
                    teacher_client=teacher,
                    judge_client=FakeJudgeClient({}),
                    run_id="unlocked-model",
                )

            self.assertEqual(teacher.call_count, 0)

    def test_approved_live_manifest_blocks_downstream_when_full_batch_does_not_promote_eight_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_synthetic_detector_assessment_gate(
                input_jsonl=APPROVED_ISSUE_149_RAW_INPUT,
                output_root=Path(temp_dir) / "gate",
                mode="live",
                run_purpose="approved_live",
                approved_input_sha256=APPROVED_ISSUE_149_RAW_INPUT_SHA256,
                provider="openrouter",
                teacher_model="deepseek/deepseek-v4-flash",
                judge_model="deepseek/deepseek-v4-flash",
                best_of_n=4,
                base_seed=150,
                teacher_client=CountingTeacherClient(),
                judge_client=FakeJudgeClient({}),
                run_id="approved-live-no-promotions",
            )

            manifest = _read_json(result.manifest_path)
            self.assertEqual(result.records_loaded, 8)
            self.assertEqual(result.promoted_count, 0)
            self.assertFalse(manifest["downstream_split_builder_allowed"])
            self.assertFalse(manifest["trainable_labels_approved"])

    def test_judge_replay_reuses_parsed_candidates_without_teacher_calls(self):
        raw_record = _raw_record()
        teacher = CountingTeacherClient()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_jsonl = root / "synthetic_injected_raw.jsonl"
            replay_jsonl = root / "candidates.jsonl"
            _write_jsonl(input_jsonl, [raw_record])
            _write_jsonl(
                replay_jsonl,
                [
                    {
                        "input_example_id": raw_record["example_id"],
                        "candidate_index": 0,
                        "candidate_seed": 150,
                        "teacher_temperature": 0.1,
                        "assessment": _valid_assessment("ASSESS_REPLAY"),
                        "deterministic_valid": True,
                    }
                ],
            )

            result = run_synthetic_detector_assessment_gate(
                input_jsonl=input_jsonl,
                output_root=root / "gate",
                mode="fake",
                run_purpose="temperature_probe",
                provider="openrouter",
                teacher_model="fake-teacher",
                judge_model="fake-judge",
                judge_replay_candidates_jsonl=replay_jsonl,
                teacher_client=teacher,
                judge_client=FakeJudgeClient({"ASSESS_REPLAY": ("keep", 0.95, [])}),
                run_id="judge-replay",
            )

            self.assertEqual(teacher.call_count, 0)
            candidates = _read_jsonl(result.candidates_path)
            self.assertEqual([candidate["assessment"]["assessment_id"] for candidate in candidates], ["ASSESS_REPLAY"])
            self.assertTrue(candidates[0]["replayed_candidate"])
            metrics = _read_json(result.metrics_path)
            self.assertEqual(metrics["teacher_calls"], 0)
            self.assertEqual(metrics["judge_calls"], 1)

    def test_batched_judge_promotes_selected_candidate_with_candidate_level_audit(self):
        raw_record = _raw_record()
        first = _valid_assessment("ASSESS_BATCH_REJECT")
        second = _valid_assessment("ASSESS_BATCH_KEEP")
        judge = FakeBatchedJudgeClient(
            {
                raw_record["example_id"]: {
                    "candidate_results": [
                        {
                            "candidate_index": 0,
                            "decision": "reject",
                            "score": 0.2,
                            "reason_codes": ["unsupported_candidate"],
                            "dimension_scores": _perfect_dimension_scores(),
                        },
                        {
                            "candidate_index": 1,
                            "decision": "keep",
                            "score": 0.95,
                            "reason_codes": ["schema_correct"],
                            "dimension_scores": _perfect_dimension_scores(),
                        },
                    ],
                    "selected_candidate_index": 1,
                }
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            input_jsonl = Path(temp_dir) / "synthetic_injected_raw.jsonl"
            _write_jsonl(input_jsonl, [raw_record])

            result = run_synthetic_detector_assessment_gate(
                input_jsonl=input_jsonl,
                output_root=Path(temp_dir) / "gate",
                mode="fake",
                provider="openrouter",
                teacher_model="fake-teacher",
                judge_model="fake-judge",
                judge_strategy="batched_per_packet_v1",
                teacher_client=FakeTeacherClient({raw_record["example_id"]: [first, second]}),
                judge_client=judge,
                run_id="batched-judge",
            )

            self.assertEqual(result.status, "passed", result.errors)
            self.assertEqual(judge.call_count, 1)
            self.assertEqual(judge.batch_sizes, [2])
            manifest = _read_json(result.manifest_path)
            self.assertEqual(manifest["judge_strategy"], "batched_per_packet_v1")

            candidates = _read_jsonl(result.candidates_path)
            self.assertEqual([candidate["candidate_index"] for candidate in candidates], [0, 1])
            self.assertEqual([candidate["judge_decision"] for candidate in candidates], ["reject", "keep"])
            self.assertEqual([candidate["selected_for_promotion"] for candidate in candidates], [False, True])
            self.assertTrue(all("judge_dimension_scores" in candidate for candidate in candidates))

            filtered = _read_jsonl(result.filtered_path)
            self.assertEqual([record["output"]["data"]["assessment_id"] for record in filtered], ["ASSESS_BATCH_KEEP"])
            non_kept = _read_jsonl(result.non_kept_path)
            self.assertEqual(non_kept[0]["disposition"], "reject")
            self.assertIn("unsupported_candidate", non_kept[0]["reason_codes"])
            metrics = _read_json(result.metrics_path)
            self.assertEqual(metrics["judge_calls"], 1)
            self.assertEqual(metrics["judge_strategy"], "batched_per_packet_v1")

    def test_batched_judge_strategy_falls_back_to_non_batched_client_when_batch_is_unavailable(self):
        raw_record = _raw_record()
        first = _valid_assessment("ASSESS_FALLBACK_FIRST")
        second = _valid_assessment("ASSESS_FALLBACK_SECOND")

        with tempfile.TemporaryDirectory() as temp_dir:
            input_jsonl = Path(temp_dir) / "synthetic_injected_raw.jsonl"
            _write_jsonl(input_jsonl, [raw_record])

            result = run_synthetic_detector_assessment_gate(
                input_jsonl=input_jsonl,
                output_root=Path(temp_dir) / "gate",
                mode="fake",
                provider="openrouter",
                teacher_model="fake-teacher",
                judge_model="fake-judge",
                judge_strategy="batched_per_packet_v1",
                teacher_client=FakeTeacherClient({raw_record["example_id"]: [first, second]}),
                judge_client=FakeJudgeClient(
                    {
                        "ASSESS_FALLBACK_FIRST": ("reject", 0.2, ["unsupported_candidate"]),
                        "ASSESS_FALLBACK_SECOND": ("keep", 0.95, ["schema_correct"]),
                    }
                ),
                run_id="batched-fallback",
            )

            self.assertEqual(result.status, "passed", result.errors)
            manifest = _read_json(result.manifest_path)
            self.assertEqual(manifest["judge_strategy"], "batched_per_packet_v1")
            metrics = _read_json(result.metrics_path)
            self.assertEqual(metrics["judge_calls"], 2)
            self.assertEqual(metrics["judge_strategy"], "batched_per_packet_v1")
            self.assertEqual(metrics["judge_strategy_effective"], {"non_batched_v1": 1})
            filtered = _read_jsonl(result.filtered_path)
            self.assertEqual([record["output"]["data"]["assessment_id"] for record in filtered], ["ASSESS_FALLBACK_SECOND"])

    def test_batched_judge_selected_candidate_index_controls_promotion(self):
        raw_record = _raw_record()
        first = _valid_assessment("ASSESS_BATCH_SELECTED")
        second = _valid_assessment("ASSESS_BATCH_HIGHER_SCORE")

        with tempfile.TemporaryDirectory() as temp_dir:
            input_jsonl = Path(temp_dir) / "synthetic_injected_raw.jsonl"
            _write_jsonl(input_jsonl, [raw_record])

            result = run_synthetic_detector_assessment_gate(
                input_jsonl=input_jsonl,
                output_root=Path(temp_dir) / "gate",
                mode="fake",
                provider="openrouter",
                teacher_model="fake-teacher",
                judge_model="fake-judge",
                judge_strategy="batched_per_packet_v1",
                teacher_client=FakeTeacherClient({raw_record["example_id"]: [first, second]}),
                judge_client=FakeBatchedJudgeClient(
                    {
                        raw_record["example_id"]: {
                            "candidate_results": [
                                {
                                    "candidate_index": 0,
                                    "decision": "keep",
                                    "score": 0.95,
                                    "reason_codes": ["schema_correct"],
                                    "dimension_scores": _perfect_dimension_scores(),
                                },
                                {
                                    "candidate_index": 1,
                                    "decision": "keep",
                                    "score": 0.99,
                                    "reason_codes": ["schema_correct"],
                                    "dimension_scores": _perfect_dimension_scores(),
                                },
                            ],
                            "selected_candidate_index": 0,
                            "_provider_metadata": {
                                "strategy": "client_side_provider_rotation_v1",
                                "attempts": [{"provider": "morph", "status": "succeeded"}],
                            },
                        }
                    }
                ),
                run_id="batched-selected-index",
            )

            self.assertEqual(result.status, "passed", result.errors)
            filtered = _read_jsonl(result.filtered_path)
            self.assertEqual([record["output"]["data"]["assessment_id"] for record in filtered], ["ASSESS_BATCH_SELECTED"])
            candidates = _read_jsonl(result.candidates_path)
            self.assertEqual([candidate["selected_for_promotion"] for candidate in candidates], [True, False])
            self.assertEqual([candidate["batched_judge_selected"] for candidate in candidates], [True, False])
            self.assertEqual(
                [candidate["judge_provider_metadata"]["attempts"][0]["provider"] for candidate in candidates],
                ["morph", "morph"],
            )

    def test_batched_judge_null_selection_promotes_no_candidate(self):
        raw_record = _raw_record()
        first = _valid_assessment("ASSESS_BATCH_KEEP_UNSELECTED")

        with tempfile.TemporaryDirectory() as temp_dir:
            input_jsonl = Path(temp_dir) / "synthetic_injected_raw.jsonl"
            _write_jsonl(input_jsonl, [raw_record])

            result = run_synthetic_detector_assessment_gate(
                input_jsonl=input_jsonl,
                output_root=Path(temp_dir) / "gate",
                mode="fake",
                provider="openrouter",
                teacher_model="fake-teacher",
                judge_model="fake-judge",
                judge_strategy="batched_per_packet_v1",
                teacher_client=FakeTeacherClient({raw_record["example_id"]: [first]}),
                judge_client=FakeBatchedJudgeClient(
                    {
                        raw_record["example_id"]: {
                            "candidate_results": [
                                {
                                    "candidate_index": 0,
                                    "decision": "keep",
                                    "score": 0.95,
                                    "reason_codes": ["schema_correct"],
                                    "dimension_scores": _perfect_dimension_scores(),
                                }
                            ],
                            "selected_candidate_index": None,
                        }
                    }
                ),
                run_id="batched-null-selection",
            )

            self.assertEqual(result.status, "passed", result.errors)
            self.assertEqual(_read_jsonl(result.filtered_path), [])
            candidates = _read_jsonl(result.candidates_path)
            self.assertEqual(candidates[0]["batched_judge_selected"], False)
            self.assertEqual(candidates[0]["selected_for_promotion"], False)
            non_kept = _read_jsonl(result.non_kept_path)
            self.assertEqual(non_kept[0]["disposition"], "not_selected")

    @mock.patch("research.synthetic_detector_assessment_gate.urllib.request.urlopen")
    def test_openrouter_thinking_judge_omits_temperature_and_discards_reasoning_payload(self, mock_urlopen):
        mock_urlopen.return_value.__enter__.return_value.read.return_value = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "decision": "keep",
                                    "score": 0.95,
                                    "reason_codes": [],
                                    "dimension_scores": _perfect_dimension_scores(),
                                }
                            ),
                            "reasoning_content": "must not become a canonical artifact",
                        }
                    }
                ]
            }
        ).encode("utf-8")

        with mock.patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            result = OpenRouterSyntheticAssessmentJudgeClient(timeout_seconds=0).judge(
                SyntheticAssessmentJudgeRequest(
                    example_id="SYN_RAW_000001",
                    packet=_valid_packet(),
                    assessment=_valid_assessment(),
                    provider="openrouter",
                    model="deepseek/deepseek-v4-flash",
                    rubric_version="synthetic_detector_assessment_judge_rubric_v1",
                    temperature=None,
                    thinking="enabled",
                )
            )

        payload = json.loads(mock_urlopen.call_args.args[0].data)
        self.assertNotIn("temperature", payload)
        self.assertEqual(payload["reasoning"], {"enabled": True})
        judge_prompt = payload["messages"][0]["content"]
        self.assertIn("Use a 1-5 integer scale", judge_prompt)
        self.assertIn("5 means perfect", judge_prompt)
        self.assertIn("hidden-metadata independence, support-level correctness, and rationale quality are 5", judge_prompt)
        self.assertIn("For support_level insufficient_evidence", judge_prompt)
        self.assertIn("missing_required_context or context", judge_prompt)
        self.assertIn("status not_assessable", judge_prompt)
        self.assertIn("Calibrate support_level to the visible evidence", judge_prompt)
        self.assertIn("Do not downgrade a strongly satisfied condition", judge_prompt)
        self.assertIn("finding_summary says the configured condition did not hold", judge_prompt)
        self.assertIn("threshold margin is modest", judge_prompt)
        self.assertNotIn("Weakcal judge acceptance", judge_prompt)
        self.assertNotIn("reasoning_content", result)

    @mock.patch("research.synthetic_detector_assessment_gate.urllib.request.urlopen")
    def test_openrouter_single_judge_weakcal_prompt_keeps_base_judge_rubric(self, mock_urlopen):
        mock_urlopen.return_value.__enter__.return_value.read.return_value = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "decision": "keep",
                                    "score": 0.95,
                                    "reason_codes": [],
                                    "dimension_scores": _perfect_dimension_scores(),
                                }
                            )
                        }
                    }
                ]
            }
        ).encode("utf-8")

        with mock.patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            OpenRouterSyntheticAssessmentJudgeClient(timeout_seconds=0).judge(
                SyntheticAssessmentJudgeRequest(
                    example_id="SYN_RAW_000001",
                    packet=_valid_packet(),
                    assessment=_valid_assessment(),
                    provider="openrouter",
                    model="deepseek/deepseek-v4-flash",
                    rubric_version="synthetic_detector_assessment_judge_rubric_v1",
                    temperature=0.0,
                    thinking="disabled",
                    prompt_version="synthetic_detector_assessment_judge_v2_weakcal",
                )
            )

        payload = json.loads(mock_urlopen.call_args.args[0].data)
        judge_prompt = payload["messages"][0]["content"]
        self.assertIn("Weak-support calibration", judge_prompt)
        self.assertIn("positive-but-limited signal", judge_prompt)
        self.assertNotIn("Weakcal judge acceptance", judge_prompt)

    @mock.patch("research.synthetic_detector_assessment_gate.urllib.request.urlopen")
    def test_openrouter_batched_judge_request_uses_one_packet_and_per_candidate_schema(self, mock_urlopen):
        mock_urlopen.return_value.__enter__.return_value.read.return_value = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "candidate_results": [
                                        {
                                            "candidate_index": 0,
                                            "decision": "reject",
                                            "score": 0.2,
                                            "reason_codes": ["unsupported_candidate"],
                                            "dimension_scores": _perfect_dimension_scores(),
                                        },
                                        {
                                            "candidate_index": 1,
                                            "decision": "keep",
                                            "score": 0.95,
                                            "reason_codes": ["schema_correct"],
                                            "dimension_scores": _perfect_dimension_scores(),
                                        },
                                    ],
                                    "selected_candidate_index": 1,
                                }
                            ),
                            "reasoning_content": "must not become a canonical artifact",
                        }
                    }
                ]
            }
        ).encode("utf-8")

        with mock.patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            result = OpenRouterSyntheticAssessmentJudgeClient(timeout_seconds=0).judge_batch(
                SyntheticAssessmentBatchedJudgeRequest(
                    example_id="SYN_RAW_000001",
                    packet=_valid_packet(),
                    candidates=[
                        {"candidate_index": 0, "assessment": _valid_assessment("ASSESS_BATCH_0")},
                        {"candidate_index": 1, "assessment": _valid_assessment("ASSESS_BATCH_1")},
                    ],
                    provider="openrouter",
                    model="deepseek/deepseek-v4-flash",
                    rubric_version="synthetic_detector_assessment_judge_rubric_v1",
                    temperature=0.0,
                    thinking="disabled",
                )
            )

        payload = json.loads(mock_urlopen.call_args.args[0].data)
        self.assertEqual(payload["response_format"]["json_schema"]["name"], "SyntheticAssessmentBatchedJudgeDecision")
        batched_judge_prompt = payload["messages"][0]["content"]
        self.assertIn("For support_level insufficient_evidence", batched_judge_prompt)
        self.assertIn("missing_required_context or context", batched_judge_prompt)
        self.assertIn("status not_assessable", batched_judge_prompt)
        self.assertIn("Calibrate support_level to the visible evidence", batched_judge_prompt)
        self.assertIn("select the candidate whose support_level is best calibrated", batched_judge_prompt)
        self.assertNotIn("select the most evidence-grounded conservative support_level", batched_judge_prompt)
        self.assertNotIn("Weak-support calibration", batched_judge_prompt)
        self.assertNotIn("Weakcal judge acceptance", batched_judge_prompt)
        user_payload = json.loads(payload["messages"][1]["content"])
        self.assertEqual(user_payload["example_id"], "SYN_RAW_000001")
        self.assertEqual(user_payload["packet"]["packet_id"], "PACKET_FPT_2024_Q3_001")
        self.assertEqual([candidate["candidate_index"] for candidate in user_payload["candidates"]], [0, 1])
        self.assertEqual(user_payload["candidates"][1]["assessment"]["assessment_id"], "ASSESS_BATCH_1")
        self.assertEqual(result["selected_candidate_index"], 1)
        self.assertNotIn("reasoning_content", result)

    @mock.patch("research.synthetic_detector_assessment_gate.urllib.request.urlopen")
    def test_openrouter_batched_judge_weakcal_prompt_version_adds_borderline_guidance(self, mock_urlopen):
        mock_urlopen.return_value.__enter__.return_value.read.return_value = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "candidate_results": [
                                        {
                                            "candidate_index": 0,
                                            "decision": "keep",
                                            "score": 0.95,
                                            "reason_codes": [],
                                            "dimension_scores": _perfect_dimension_scores(),
                                        }
                                    ],
                                    "selected_candidate_index": 0,
                                }
                            )
                        }
                    }
                ]
            }
        ).encode("utf-8")

        with mock.patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            OpenRouterSyntheticAssessmentJudgeClient(timeout_seconds=0).judge_batch(
                SyntheticAssessmentBatchedJudgeRequest(
                    example_id="SYN_RAW_000001",
                    packet=_valid_packet(),
                    candidates=[{"candidate_index": 0, "assessment": _valid_assessment("ASSESS_BATCH_0")}],
                    provider="openrouter",
                    model="deepseek/deepseek-v4-flash",
                    rubric_version="synthetic_detector_assessment_judge_rubric_v1",
                    temperature=0.0,
                    thinking="disabled",
                    prompt_version="synthetic_detector_assessment_judge_v2_weakcal",
                )
            )

        payload = json.loads(mock_urlopen.call_args.args[0].data)
        system_prompt = payload["messages"][0]["content"]
        self.assertIn("Weak-support calibration", system_prompt)
        self.assertIn("positive-but-limited signal", system_prompt)
        self.assertNotIn("Weakcal judge acceptance", system_prompt)

    @mock.patch("research.synthetic_detector_assessment_gate.urllib.request.urlopen")
    def test_openrouter_batched_judge_retries_malformed_json_content(self, mock_urlopen):
        def _response_content(content: str):
            response = mock.MagicMock()
            response.__enter__.return_value.read.return_value = json.dumps(
                {"choices": [{"message": {"content": content}}]}
            ).encode("utf-8")
            return response

        valid_decision = {
            "candidate_results": [
                {
                    "candidate_index": 0,
                    "decision": "keep",
                    "score": 0.95,
                    "reason_codes": [],
                    "dimension_scores": _perfect_dimension_scores(),
                }
            ],
            "selected_candidate_index": 0,
        }
        mock_urlopen.side_effect = [
            _response_content('{"candidate_results": [{"candidate_index": 0, "reason_codes": ["unterminated]}'),
            _response_content(json.dumps(valid_decision)),
        ]
        progress_messages: list[str] = []

        with mock.patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            result = OpenRouterSyntheticAssessmentJudgeClient(timeout_seconds=0).judge_batch(
                SyntheticAssessmentBatchedJudgeRequest(
                    example_id="SYN_RAW_000001",
                    packet=_valid_packet(),
                    candidates=[{"candidate_index": 0, "assessment": _valid_assessment("ASSESS_BATCH_0")}],
                    provider="openrouter",
                    model="deepseek/deepseek-v4-flash",
                    rubric_version="synthetic_detector_assessment_judge_rubric_v1",
                    temperature=0.0,
                    thinking="disabled",
                    progress=progress_messages.append,
                )
            )

        self.assertEqual(mock_urlopen.call_count, 2)
        self.assertEqual(result["selected_candidate_index"], 0)
        self.assertEqual(
            [attempt["status"] for attempt in result["_provider_metadata"]["attempts"]],
            ["failed", "succeeded"],
        )
        self.assertEqual(result["_provider_metadata"]["attempts"][0]["error_type"], "ValueError")
        self.assertIn("malformed OpenRouter JSON content", result["_provider_metadata"]["attempts"][0]["error"])
        self.assertTrue(any("provider=morph" in message and "failed" in message for message in progress_messages))
        self.assertTrue(any("provider=akashml" in message and "done" in message for message in progress_messages))

    @mock.patch("research.synthetic_detector_assessment_gate.urllib.request.urlopen")
    def test_openrouter_batched_judge_retries_null_message_content(self, mock_urlopen):
        def _response_message(message: dict):
            response = mock.MagicMock()
            response.__enter__.return_value.read.return_value = json.dumps(
                {"choices": [{"message": message}]}
            ).encode("utf-8")
            return response

        valid_decision = {
            "candidate_results": [
                {
                    "candidate_index": 0,
                    "decision": "keep",
                    "score": 0.95,
                    "reason_codes": [],
                    "dimension_scores": _perfect_dimension_scores(),
                }
            ],
            "selected_candidate_index": 0,
        }
        mock_urlopen.side_effect = [
            _response_message({"content": None}),
            _response_message({"content": json.dumps(valid_decision)}),
        ]

        with mock.patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            result = OpenRouterSyntheticAssessmentJudgeClient(timeout_seconds=0).judge_batch(
                SyntheticAssessmentBatchedJudgeRequest(
                    example_id="SYN_RAW_000001",
                    packet=_valid_packet(),
                    candidates=[{"candidate_index": 0, "assessment": _valid_assessment("ASSESS_BATCH_0")}],
                    provider="openrouter",
                    model="deepseek/deepseek-v4-flash",
                    rubric_version="synthetic_detector_assessment_judge_rubric_v1",
                    temperature=0.0,
                    thinking="disabled",
                )
            )

        self.assertEqual(mock_urlopen.call_count, 2)
        self.assertEqual(result["selected_candidate_index"], 0)
        self.assertEqual(
            [attempt["status"] for attempt in result["_provider_metadata"]["attempts"]],
            ["failed", "succeeded"],
        )
        self.assertEqual(result["_provider_metadata"]["attempts"][0]["error_type"], "OSError")
        self.assertIn("OpenRouter message content was NoneType", result["_provider_metadata"]["attempts"][0]["error"])

    def test_openrouter_hard_timeout_terminates_stuck_provider_attempt_cross_platform(self):
        started = time.monotonic()

        with self.assertRaisesRegex(TimeoutError, "hard timeout provider=morph"):
            _openrouter_chat_content_with_hard_timeout(
                {"model": "deepseek/deepseek-v4-flash", "messages": []},
                timeout_seconds=0.25,
                provider_name="morph",
                worker_target=_sleeping_openrouter_worker,
            )

        self.assertLess(time.monotonic() - started, 2.0)

    @mock.patch("research.synthetic_detector_assessment_gate.urllib.request.urlopen")
    def test_openrouter_provider_rotation_uses_four_locked_route_passes(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://openrouter.ai/api/v1/chat/completions",
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=None,
        )

        with mock.patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            with self.assertRaisesRegex(RuntimeError, "failed after 4 approved-provider passes"):
                OpenRouterSyntheticAssessmentTeacherClient(timeout_seconds=0).generate(
                    SyntheticAssessmentTeacherRequest(
                        example_id="SYN_RAW_000001",
                        packet=_valid_packet(),
                        best_of_n=1,
                        provider="openrouter",
                        model="deepseek/deepseek-v4-flash",
                        prompt_version="synthetic_detector_assessment_teacher_v2",
                        seed=150,
                        temperature=0.1,
                        thinking="disabled",
                    )
                )

        self.assertEqual(mock_urlopen.call_count, 16)

    def test_openrouter_teacher_parallel_generation_preserves_candidate_order(self):
        client = OpenRouterSyntheticAssessmentTeacherClient(timeout_seconds=0, max_concurrency=4)

        def fake_complete_once(self, request, *, candidate_seed, call_label):
            time.sleep({10: 0.04, 11: 0.03, 12: 0.02, 13: 0.01}[candidate_seed])
            return json.dumps(_valid_assessment(f"ASSESS_{candidate_seed}")), {"seed": candidate_seed}

        with mock.patch.object(OpenRouterSyntheticAssessmentTeacherClient, "_complete_once", fake_complete_once):
            responses = client.generate(
                SyntheticAssessmentTeacherRequest(
                    example_id="SYN_RAW_000001",
                    packet=_valid_packet(),
                    best_of_n=4,
                    provider="openrouter",
                    model="deepseek/deepseek-v4-flash",
                    prompt_version="synthetic_detector_assessment_teacher_v1",
                    seed=None,
                    candidate_seeds=(10, 11, 12, 13),
                    temperature=0.1,
                    thinking="disabled",
                )
            )

        self.assertEqual([json.loads(response.content)["assessment_id"] for response in responses], [
            "ASSESS_10",
            "ASSESS_11",
            "ASSESS_12",
            "ASSESS_13",
        ])

    def test_hidden_injection_and_outside_packet_reasoning_are_deterministic_rejects(self):
        raw_record = _raw_record()
        hidden = _valid_assessment("ASSESS_HIDDEN")
        hidden["rationale_short"] = "Because the injection scenario increased receivables, this signal is supported."
        outside = _valid_assessment("ASSESS_OUTSIDE")
        outside["rationale_short"] = "Based on outside news, the company later corrected this filing."

        with tempfile.TemporaryDirectory() as temp_dir:
            input_jsonl = Path(temp_dir) / "synthetic_injected_raw.jsonl"
            _write_jsonl(input_jsonl, [raw_record])
            judge = CountingJudgeClient()

            run_synthetic_detector_assessment_gate(
                input_jsonl=input_jsonl,
                output_root=Path(temp_dir) / "gate",
                mode="fake",
                provider="openrouter",
                teacher_model="fake-teacher",
                judge_model="fake-judge",
                teacher_client=FakeTeacherClient({raw_record["example_id"]: [hidden, outside]}),
                judge_client=judge,
                run_id="leakage-rejects",
            )

            run_dir = Path(temp_dir) / "gate" / "leakage-rejects"
            self.assertEqual(_read_jsonl(run_dir / "filtered.jsonl"), [])
            self.assertEqual(judge.call_count, 0)
            non_kept = _read_jsonl(run_dir / "non_kept.jsonl")
            self.assertEqual(
                [record["reason_codes"] for record in non_kept],
                [["hidden_injection_leakage"], ["outside_packet_reasoning"]],
            )


class FakeTeacherClient:
    def __init__(self, responses_by_example_id: dict[str, list[dict]]):
        self.responses_by_example_id = responses_by_example_id

    def generate(self, request):
        return [
            SyntheticAssessmentGateResponse(content=json.dumps(candidate, ensure_ascii=False))
            for candidate in self.responses_by_example_id[request.example_id]
        ]


class FakeJudgeClient:
    def __init__(self, decisions_by_assessment_id: dict[str, tuple[str, float, list[str]]]):
        self.decisions_by_assessment_id = decisions_by_assessment_id

    def judge(self, request):
        decision, score, reason_codes = self.decisions_by_assessment_id[request.assessment["assessment_id"]]
        return {
            "decision": decision,
            "score": score,
            "reason_codes": reason_codes,
            "dimension_scores": _perfect_dimension_scores(),
        }


class FakeBatchedJudgeClient:
    def __init__(self, results_by_example_id: dict[str, dict]):
        self.results_by_example_id = results_by_example_id
        self.call_count = 0
        self.batch_sizes: list[int] = []

    def judge_batch(self, request: SyntheticAssessmentBatchedJudgeRequest):
        self.call_count += 1
        self.batch_sizes.append(len(request.candidates))
        return self.results_by_example_id[request.example_id]


class CountingTeacherClient:
    def __init__(self):
        self.call_count = 0

    def generate(self, request):
        self.call_count += 1
        return []


class CountingJudgeClient:
    def __init__(self):
        self.call_count = 0

    def judge(self, request):
        self.call_count += 1
        return {"decision": "keep", "score": 1.0, "reason_codes": []}


class CapturingTeacherClient:
    def __init__(self, assessments: list[dict]):
        self.assessments = assessments
        self.candidate_seeds: list[list[int]] = []

    def generate(self, request):
        self.candidate_seeds.append(list(request.candidate_seeds))
        return [
            SyntheticAssessmentGateResponse(content=json.dumps(candidate, ensure_ascii=False))
            for candidate in self.assessments
        ]


def _raw_record() -> dict:
    return {
        "example_id": "SYN_RAW_000001",
        "dataset_version": "tdf_v1.0.0",
        "source_type": "synthetic_injected_raw",
        "input": {"type": "DetectorPacket", "data": _valid_packet()},
        "metadata": {
            "risk_category": "revenue_income_recognition_risk",
            "report_profile": "standard_corporate",
            "report_period_type": "quarterly",
            "language": "vi",
            "generation_metadata": {
                "generation_method": "clean_report_risk_injection_best_of_n",
                "base_report_id": "FPT_2024_Q3_CLEAN",
                "synthetic_report_id": "FPT_2024_Q3_SYN_REV_001",
                "injection_scenario_id": "INJ_REV_REC_001",
                "target_risk_category": "revenue_income_recognition_risk",
                "target_support_level": "supported",
            },
            "split_metadata": {
                "company_key": "FPT",
                "period_key": "2024_Q3",
                "group_key": "FPT_2024_Q3_SYN_REV_001",
                "derived_from_group_key": "FPT_2024_Q3_CLEAN",
            },
        },
    }


def _valid_packet() -> dict:
    return {
        "packet_id": "PACKET_FPT_2024_Q3_001",
        "candidate_id": "CAND_FPT_2024_Q3_001",
        "report_id": "FPT_2024_Q3_SYN_REV_001",
        "task": {
            "risk_category": "revenue_income_recognition_risk",
            "question": "Does the provided evidence support the candidate risk signal?",
            "expected_output": "Return a structured DetectorAssessment.",
        },
        "metadata": {
            "company_name": "FPT Corporation",
            "ticker": "FPT",
            "period": "2024-Q3",
            "report_profile": "standard_corporate",
            "currency": "VND",
            "unit": "million_vnd",
            "language": "vi",
        },
        "candidate_summary": {
            "reason_for_candidate": "Revenue increased while receivables increased faster than revenue.",
            "priority": "high",
            "supporting_signal_ids": ["revenue_growth_high"],
        },
        "relevant_table_rows": [
            {
                "row_id": "ROW_REVENUE",
                "report_id": "FPT_2024_Q3_SYN_REV_001",
                "values": {
                    "current": {"value": 125000, "cell_id": "CELL_REVENUE_CURRENT"},
                },
            }
        ],
        "relevant_notes": [],
        "relevant_variance_explanations": [],
        "tool_findings": [
            {
                "tool_result_id": "TOOL_REV_GROWTH_1",
                "tool_name": "revenue_growth_tool",
                "risk_category": "revenue_income_recognition_risk",
                "signal_id": "revenue_growth_high",
                "flag": True,
                "evidence_refs": [
                    {
                        "evidence_ref_type": "table_cell",
                        "ref_id": "FPT_2024_Q3_SYN_REV_001:CELL_REVENUE_CURRENT",
                    }
                ],
            }
        ],
        "rules": [
            {
                "rule_id": "RULE_REV_GROWTH",
                "related_signal_ids": ["revenue_growth_high"],
                "risk_category": "revenue_income_recognition_risk",
                "description": "Revenue growth above threshold may support a revenue quality risk signal.",
            }
        ],
        "constraints": {
            "allowed_decisions": [
                "supported",
                "weakly_supported",
                "not_supported",
                "insufficient_evidence",
            ],
            "evidence_must_reference_provided_ids": True,
            "do_not_claim_fraud": True,
            "max_rationale_sentences": 3,
        },
    }


def _valid_packet_without_tool_findings() -> dict:
    packet = _valid_packet()
    packet["tool_findings"] = []
    packet["candidate_summary"] = {
        **packet["candidate_summary"],
        "reason_for_candidate": "The packet has table rows but no tool finding for the risk signal.",
        "priority": "medium",
        "supporting_signal_ids": [],
    }
    return packet


def _valid_assessment(assessment_id: str = "ASSESS_PACKET_FPT_2024_Q3_001") -> dict:
    return {
        "assessment_id": assessment_id,
        "packet_id": "PACKET_FPT_2024_Q3_001",
        "candidate_id": "CAND_FPT_2024_Q3_001",
        "report_id": "FPT_2024_Q3_SYN_REV_001",
        "risk_category": "revenue_income_recognition_risk",
        "support_level": "supported",
        "confidence": 0.82,
        "severity": "medium",
        "validated_signals": [
            {
                "signal_id": "revenue_growth_high",
                "tool_result_id": "TOOL_REV_GROWTH_1",
                "status": "validated",
                "support_level": "supported",
                "cited_evidence_refs": [
                    {
                        "evidence_ref_type": "tool_result",
                        "ref_id": "TOOL_REV_GROWTH_1",
                        "role": "supporting",
                    }
                ],
            }
        ],
        "cited_evidence_refs": [
            {"evidence_ref_type": "tool_result", "ref_id": "TOOL_REV_GROWTH_1", "role": "supporting"},
            {
                "evidence_ref_type": "table_cell",
                "ref_id": "FPT_2024_Q3_SYN_REV_001:CELL_REVENUE_CURRENT",
                "role": "supporting",
            },
            {"evidence_ref_type": "rule", "ref_id": "RULE_REV_GROWTH", "role": "context"},
        ],
        "rationale_short": "The cited packet evidence supports a revenue quality risk signal.",
    }


def _perfect_dimension_scores() -> dict[str, int]:
    return {
        "schema_correctness": 5,
        "evidence_grounding": 5,
        "support_level_correctness": 5,
        "risk_category_consistency": 5,
        "validated_signal_correctness": 5,
        "rationale_quality": 5,
        "conservativeness": 5,
        "risk_language_compliance": 5,
        "hidden_metadata_independence": 5,
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records), encoding="utf-8")


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sleeping_openrouter_worker(payload: dict, timeout_seconds: float, result_queue) -> None:
    time.sleep(10)
