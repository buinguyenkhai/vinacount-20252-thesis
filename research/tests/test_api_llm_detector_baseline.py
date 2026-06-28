import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from unittest import mock
from pathlib import Path

from research.api_llm_detector_baseline import (
    ApiLlmDetectorResponse,
    DEEPSEEK_CHAT_COMPLETIONS_URL,
    DeepSeekApiLlmDetectorClient,
    OpenRouterApiLlmDetectorClient,
    run_api_llm_detector_baseline,
    run_api_llm_detector_baseline_sweep,
)


RELEASE_DIR = Path("data/real_manual/combined_real_manual_validation_release")


class ApiLlmDetectorBaselineTest(unittest.TestCase):
    def test_dry_run_writes_diagnostic_artifacts_without_mutating_source_release(self):
        before_hashes = _file_hashes(RELEASE_DIR, ["manifest.json", "validation.jsonl"])

        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_api_llm_detector_baseline(
                release_dir=RELEASE_DIR,
                output_root=Path(temp_dir) / "api_llm_detector_baseline",
                mode="dry_run",
                provider="openrouter",
                model="dry-run-model",
                run_id="dry-run-test",
            )

            self.assertEqual(result.status, "passed", result.errors)
            self.assertEqual(result.records_loaded, 14)
            self.assertEqual(result.predictions_written, 14)
            self.assertEqual(result.invalid_response_count, 0)

            run_dir = Path(temp_dir) / "api_llm_detector_baseline" / "dry-run-test"
            self.assertEqual(result.run_dir, run_dir)
            self.assertTrue((run_dir / "manifest.json").exists())
            self.assertTrue((run_dir / "predictions.jsonl").exists())
            self.assertTrue((run_dir / "metrics.json").exists())
            self.assertTrue((run_dir / "invalid_responses.jsonl").exists())

            manifest = _read_json(run_dir / "manifest.json")
            self.assertEqual(manifest["mode"], "dry_run")
            self.assertEqual(manifest["provider"], "openrouter")
            self.assertEqual(manifest["model"], "dry-run-model")
            self.assertEqual(manifest["prompt_version"], "api_llm_detector_baseline_prompt_v1_evidence_guard")
            self.assertEqual(manifest["decoding_config"], {"temperature": 0.0})
            self.assertEqual(manifest["source_release"]["release_name"], "combined_real_manual_validation_release")
            self.assertEqual(manifest["source_release"]["release_build_id"], "WAVE4_COMBINED_1DB1E64FF22E075D")
            self.assertEqual(manifest["source_release"]["num_examples_total"], 14)
            self.assertEqual(manifest["artifact_policy"]["canonical_prediction_artifact"], "predictions.jsonl")
            self.assertFalse(manifest["artifact_policy"]["raw_valid_responses_canonical"])
            self.assertTrue(manifest["artifact_policy"]["predictions_are_evaluation_outputs_only"])
            self.assertTrue(manifest["artifact_policy"]["not_gold_labels"])
            self.assertTrue(manifest["artifact_policy"]["not_training_data"])
            self.assertEqual(
                manifest["reporting_policy"]["corpus_description"],
                "Pilot Real-World Development/Evaluation Corpus",
            )
            self.assertFalse(manifest["reporting_policy"]["protected_test_claims_allowed"])
            self.assertFalse(manifest["reporting_policy"]["final_performance_claims_allowed"])

            predictions = _read_jsonl(run_dir / "predictions.jsonl")
            self.assertEqual(len(predictions), 14)
            self.assertEqual({record["prediction_status"] for record in predictions}, {"dry_run"})
            self.assertTrue(all(record["model_visible_input"]["type"] == "DetectorPacket" for record in predictions))
            self.assertTrue(all("output" not in record["model_visible_input"] for record in predictions))
            self.assertTrue(all("metadata" not in record["model_visible_input"] for record in predictions))

            metrics = _read_json(run_dir / "metrics.json")
            self.assertEqual(metrics["num_examples"], 14)
            self.assertEqual(metrics["num_attempted"], 0)
            self.assertEqual(metrics["num_valid_predictions"], 0)
            self.assertEqual(metrics["num_invalid_responses"], 0)

            self.assertEqual(_read_jsonl(run_dir / "invalid_responses.jsonl"), [])

        self.assertEqual(_file_hashes(RELEASE_DIR, ["manifest.json", "validation.jsonl"]), before_hashes)

    def test_fake_client_valid_responses_become_accepted_predictions_and_counts(self):
        release_records = _read_jsonl(RELEASE_DIR / "validation.jsonl")
        responses_by_example_id = {
            record["example_id"]: record["output"]["data"]
            for record in release_records
        }
        client = FakeClient(responses_by_example_id)

        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_api_llm_detector_baseline(
                release_dir=RELEASE_DIR,
                output_root=Path(temp_dir) / "api_llm_detector_baseline",
                mode="fake",
                provider="openrouter",
                model="fake-model",
                client=client,
                run_id="fake-valid-test",
            )

            self.assertEqual(result.status, "passed", result.errors)
            self.assertEqual(result.records_loaded, 14)
            self.assertEqual(result.predictions_written, 14)
            self.assertEqual(result.invalid_response_count, 0)
            self.assertEqual(len(client.requests), 14)
            self.assertEqual(client.requests[0].detector_packet, release_records[0]["input"]["data"])

            run_dir = Path(temp_dir) / "api_llm_detector_baseline" / "fake-valid-test"
            predictions = _read_jsonl(run_dir / "predictions.jsonl")
            self.assertEqual({record["prediction_status"] for record in predictions}, {"accepted"})
            self.assertTrue(all(record["schema_valid"] for record in predictions))
            self.assertTrue(all(record["evidence_valid"] for record in predictions))
            self.assertTrue(
                all(
                    record["prediction"]["support_level"] == record["gold_support_level"]
                    for record in predictions
                )
            )

            metrics = _read_json(run_dir / "metrics.json")
            self.assertEqual(metrics["num_examples"], 14)
            self.assertEqual(metrics["num_attempted"], 14)
            self.assertEqual(metrics["num_valid_predictions"], 14)
            self.assertEqual(metrics["num_invalid_responses"], 0)
            self.assertEqual(metrics["support_level_exact_match_count"], 14)
            self.assertEqual(
                metrics["support_level_confusion_matrix_counts"],
                {
                    "insufficient_evidence": {"insufficient_evidence": 2},
                    "not_supported": {"not_supported": 1},
                    "supported": {"supported": 6},
                    "weakly_supported": {"weakly_supported": 5},
                },
            )

    def test_provider_metadata_is_written_to_prediction_rows(self):
        release_record = _read_jsonl(RELEASE_DIR / "validation.jsonl")[0]
        provider_metadata = {
            "provider": "deepseek",
            "strategy": "direct_deepseek_chat_completions_v1",
            "usage": {"total_tokens": 123},
            "cache_usage": {"prompt_cache_hit_tokens": 10, "prompt_cache_miss_tokens": 20},
        }
        client = MetadataFakeClient(
            {release_record["example_id"]: release_record["output"]["data"]},
            provider_metadata=provider_metadata,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            run_api_llm_detector_baseline(
                release_dir=RELEASE_DIR,
                output_root=Path(temp_dir) / "api_llm_detector_baseline",
                mode="fake",
                provider="deepseek",
                model="fake-model",
                client=client,
                example_ids=[release_record["example_id"]],
                run_id="metadata-test",
            )

            prediction = _read_jsonl(
                Path(temp_dir) / "api_llm_detector_baseline" / "metadata-test" / "predictions.jsonl"
            )[0]
            self.assertEqual(prediction["provider_metadata"], provider_metadata)

    def test_decimal_heavy_two_sentence_rationale_counts_as_valid(self):
        release_record = _read_jsonl(RELEASE_DIR / "validation.jsonl")[0]
        assessment = dict(release_record["output"]["data"])
        assessment["rationale_short"] = (
            "The packet shows profit after tax of VND 5.46 billion, unrealized FVTPL gains of VND "
            "126.77 billion, and the tool value of 2321.14% while operating cash flow is VND "
            "-2.58 trillion. This supports the earnings and cash-flow mismatch risk signal."
        )
        client = FakeClient({release_record["example_id"]: assessment})

        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_api_llm_detector_baseline(
                release_dir=RELEASE_DIR,
                output_root=Path(temp_dir) / "api_llm_detector_baseline",
                mode="fake",
                provider="openrouter",
                model="fake-model",
                client=client,
                example_ids=[release_record["example_id"]],
                run_id="decimal-rationale-test",
            )

            self.assertEqual(result.status, "passed", result.errors)
            self.assertEqual(result.invalid_response_count, 0)
            run_dir = Path(temp_dir) / "api_llm_detector_baseline" / "decimal-rationale-test"
            prediction = _read_jsonl(run_dir / "predictions.jsonl")[0]
            self.assertEqual(prediction["prediction_status"], "accepted")
            self.assertTrue(prediction["schema_valid"])
            self.assertTrue(prediction["evidence_valid"])

    def test_run_manifest_records_locked_openrouter_provider_routing_for_deepseek_v4_flash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_api_llm_detector_baseline(
                release_dir=RELEASE_DIR,
                output_root=Path(temp_dir) / "api_llm_detector_baseline",
                mode="dry_run",
                provider="openrouter",
                model="deepseek/deepseek-v4-flash",
                run_id="deepseek-provider-routing-manifest-test",
                limit=1,
            )

            self.assertEqual(result.status, "passed", result.errors)
            manifest = _read_json(result.manifest_path)
            self.assertEqual(
                manifest["provider_routing"],
                {
                    "allow_fallbacks": False,
                    "only": ["deepseek", "alibaba"],
                    "order": ["deepseek", "alibaba"],
                    "require_parameters": True,
                },
            )

    def test_sweep_runs_diagnostic_variants_and_writes_summary_artifacts(self):
        variants = [
            {
                "variant_id": "v1_temp0_0",
                "prompt_version": "api_llm_detector_baseline_prompt_v1",
                "temperature": 0.0,
            },
            {
                "variant_id": "evidence_guard_temp0_2",
                "prompt_version": "api_llm_detector_baseline_prompt_v1_evidence_guard",
                "temperature": 0.2,
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_api_llm_detector_baseline_sweep(
                release_dir=RELEASE_DIR,
                output_root=Path(temp_dir) / "api_llm_detector_baseline",
                mode="fake",
                provider="openrouter",
                model="deepseek/deepseek-v4-flash",
                sweep_id="fake-sweep-test",
                variants=variants,
                limit=2,
            )

            self.assertEqual(result.status, "passed", result.errors)
            self.assertEqual(result.variant_count, 2)
            self.assertTrue(result.manifest_path.exists())
            self.assertTrue(result.results_path.exists())

            manifest = _read_json(result.manifest_path)
            self.assertEqual(manifest["sweep_id"], "fake-sweep-test")
            self.assertTrue(manifest["artifact_policy"]["predictions_are_evaluation_outputs_only"])
            self.assertTrue(manifest["ranking_policy"]["diagnostic_only"])
            self.assertEqual(
                manifest["provider_routing"],
                {
                    "allow_fallbacks": False,
                    "only": ["deepseek", "alibaba"],
                    "order": ["deepseek", "alibaba"],
                    "require_parameters": True,
                },
            )

            rows = _read_jsonl(result.results_path)
            self.assertEqual([row["variant_id"] for row in rows], ["v1_temp0_0", "evidence_guard_temp0_2"])
            self.assertEqual(rows[0]["metrics"]["num_valid_predictions"], 2)
            child_manifest = _read_json(
                Path(temp_dir)
                / "api_llm_detector_baseline"
                / "fake-sweep-test__evidence_guard_temp0_2"
                / "manifest.json"
            )
            self.assertEqual(child_manifest["prompt_version"], "api_llm_detector_baseline_prompt_v1_evidence_guard")
            self.assertEqual(child_manifest["decoding_config"], {"temperature": 0.2})

    def test_sweep_records_failed_variant_and_continues(self):
        variants = [
            {
                "variant_id": "bad_prompt",
                "prompt_version": "unsupported_prompt_version",
                "temperature": 0.0,
            },
            {
                "variant_id": "good_prompt",
                "prompt_version": "api_llm_detector_baseline_prompt_v1",
                "temperature": 0.0,
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_api_llm_detector_baseline_sweep(
                release_dir=RELEASE_DIR,
                output_root=Path(temp_dir) / "api_llm_detector_baseline",
                mode="dry_run",
                provider="openrouter",
                model="deepseek/deepseek-v4-flash",
                sweep_id="failed-variant-sweep-test",
                variants=variants,
                limit=1,
            )

            self.assertEqual(result.status, "completed_with_errors")
            rows = _read_jsonl(result.results_path)
            self.assertEqual([row["status"] for row in rows], ["failed", "passed"])
            self.assertEqual(rows[0]["variant_id"], "bad_prompt")
            self.assertIn("unsupported prompt_version", rows[0]["error"])
            self.assertEqual(rows[1]["metrics"]["num_examples"], 1)

    def test_fake_client_accepts_exact_markdown_wrapped_json_response(self):
        release_records = _read_jsonl(RELEASE_DIR / "validation.jsonl")
        responses_by_example_id = {
            record["example_id"]: record["output"]["data"]
            for record in release_records
        }
        client = MarkdownWrappedFakeClient(responses_by_example_id)

        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_api_llm_detector_baseline(
                release_dir=RELEASE_DIR,
                output_root=Path(temp_dir) / "api_llm_detector_baseline",
                mode="fake",
                provider="openrouter",
                model="fake-model",
                client=client,
                run_id="fake-markdown-test",
            )

            self.assertEqual(result.status, "passed", result.errors)
            self.assertEqual(result.invalid_response_count, 0)

            predictions = _read_jsonl(
                Path(temp_dir)
                / "api_llm_detector_baseline"
                / "fake-markdown-test"
                / "predictions.jsonl"
            )
            self.assertEqual({record["prediction_status"] for record in predictions}, {"accepted"})

    def test_fake_client_invalid_responses_are_counted_and_raw_text_is_debug_only(self):
        release_records = _read_jsonl(RELEASE_DIR / "validation.jsonl")
        client = InvalidResponseFakeClient(release_records)

        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_api_llm_detector_baseline(
                release_dir=RELEASE_DIR,
                output_root=Path(temp_dir) / "api_llm_detector_baseline",
                mode="fake",
                provider="openrouter",
                model="fake-model",
                client=client,
                run_id="fake-invalid-test",
            )

            self.assertEqual(result.status, "passed", result.errors)
            self.assertEqual(result.predictions_written, 14)
            self.assertEqual(result.invalid_response_count, 5)

            run_dir = Path(temp_dir) / "api_llm_detector_baseline" / "fake-invalid-test"
            predictions = _read_jsonl(run_dir / "predictions.jsonl")
            invalid_predictions = [
                record for record in predictions if record["prediction_status"] == "invalid"
            ]
            self.assertEqual(len(invalid_predictions), 5)
            self.assertEqual(
                [record["invalid_reason_codes"][0] for record in invalid_predictions],
                [
                    "invalid_json",
                    "schema_mismatch",
                    "identity_mismatch",
                    "invalid_evidence_ids",
                    "prohibited_risk_language",
                ],
            )

            invalid_responses = _read_jsonl(run_dir / "invalid_responses.jsonl")
            self.assertEqual(len(invalid_responses), 5)
            self.assertTrue(all("raw_response_text" not in record for record in invalid_responses))
            self.assertTrue(
                all(record["raw_response_debug_path"] == "debug/raw_invalid_responses.jsonl" for record in invalid_responses)
            )

            raw_invalid = _read_jsonl(run_dir / "debug" / "raw_invalid_responses.jsonl")
            self.assertEqual(len(raw_invalid), 5)
            self.assertTrue(all(record["non_canonical"] for record in raw_invalid))
            self.assertTrue(all(record["debug_only"] for record in raw_invalid))
            self.assertTrue(all(record["not_labels"] for record in raw_invalid))

            metrics = _read_json(run_dir / "metrics.json")
            self.assertEqual(metrics["num_examples"], 14)
            self.assertEqual(metrics["num_attempted"], 14)
            self.assertEqual(metrics["num_valid_predictions"], 9)
            self.assertEqual(metrics["num_invalid_responses"], 5)
            self.assertEqual(
                metrics["false_positive_rejection"],
                {"correct_or_conservative": 1, "invalid_gold_count": 2, "total": 3},
            )
            self.assertEqual(
                metrics["insufficient_evidence_detection"],
                {"gold_total": 2, "invalid_gold_count": 2, "predicted_total": 0, "true_positive": 0},
            )

    def test_fake_client_rejects_noncanonical_nested_schema_values(self):
        release_records = _read_jsonl(RELEASE_DIR / "validation.jsonl")
        selected_records = release_records[:4]
        client = ContractDriftFakeClient(selected_records)

        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_api_llm_detector_baseline(
                release_dir=RELEASE_DIR,
                output_root=Path(temp_dir) / "api_llm_detector_baseline",
                mode="fake",
                provider="openrouter",
                model="fake-model",
                client=client,
                run_id="contract-drift-test",
                limit=4,
            )

            self.assertEqual(result.status, "passed", result.errors)
            self.assertEqual(result.invalid_response_count, 4)

            run_dir = Path(temp_dir) / "api_llm_detector_baseline" / "contract-drift-test"
            predictions = _read_jsonl(run_dir / "predictions.jsonl")
            self.assertEqual(
                [record["invalid_reason_codes"][0] for record in predictions],
                [
                    "invalid_evidence_ref_type",
                    "invalid_evidence_ref_role",
                    "invalid_signal_status",
                    "schema_mismatch",
                ],
            )
            metrics = _read_json(run_dir / "metrics.json")
            self.assertEqual(metrics["num_valid_predictions"], 0)
            self.assertEqual(metrics["schema_valid_count"], 0)
            self.assertEqual(metrics["evidence_id_valid_count"], 0)

    def test_fake_client_requests_and_prediction_artifacts_do_not_expose_release_wrappers_or_provenance(self):
        release_records = _read_jsonl(RELEASE_DIR / "validation.jsonl")
        responses_by_example_id = {
            record["example_id"]: record["output"]["data"]
            for record in release_records
        }
        client = FakeClient(responses_by_example_id)

        with tempfile.TemporaryDirectory() as temp_dir:
            run_api_llm_detector_baseline(
                release_dir=RELEASE_DIR,
                output_root=Path(temp_dir) / "api_llm_detector_baseline",
                mode="fake",
                provider="openrouter",
                model="fake-model",
                client=client,
                run_id="prompt-visibility-test",
            )

            request_surface = json.dumps([request.detector_packet for request in client.requests], ensure_ascii=False)
            for forbidden in [
                '"output"',
                '"source_type"',
                '"split_metadata"',
                '"audit_metadata"',
                '"release_build_id"',
                '"sampling_pool"',
                '"source_file_sha256"',
                '"raw_extraction_artifacts"',
                '"api_assistance_metadata"',
            ]:
                self.assertNotIn(forbidden, request_surface)

            predictions = _read_jsonl(
                Path(temp_dir)
                / "api_llm_detector_baseline"
                / "prompt-visibility-test"
                / "predictions.jsonl"
            )
            prediction_surface = json.dumps(
                [prediction["model_visible_input"] for prediction in predictions],
                ensure_ascii=False,
            )
            self.assertNotIn('"output"', prediction_surface)
            self.assertNotIn('"split_metadata"', prediction_surface)
            self.assertNotIn('"source_file_sha256"', prediction_surface)

    def test_openrouter_client_sends_packet_only_prompt_and_json_response_request(self):
        release_record = _read_jsonl(RELEASE_DIR / "validation.jsonl")[0]
        packet = release_record["input"]["data"]
        assessment = release_record["output"]["data"]
        response_payload = {"choices": [{"message": {"content": json.dumps(assessment, ensure_ascii=False)}}]}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self):
                return json.dumps(response_payload).encode("utf-8")

        with mock.patch("urllib.request.urlopen", return_value=Response()) as urlopen:
            client = OpenRouterApiLlmDetectorClient(api_key="secret-not-logged")
            response = client.complete(
                request=type(
                    "Request",
                    (),
                    {
                        "example_id": release_record["example_id"],
                        "prompt_version": "api_llm_detector_baseline_prompt_v1",
                        "provider": "openrouter",
                        "model": "deepseek/deepseek-v4-flash",
                        "temperature": 0.2,
                        "detector_packet": packet,
                    },
                )()
            )

        self.assertEqual(response.content, response_payload["choices"][0]["message"]["content"])
        http_request = urlopen.call_args.args[0]
        payload = json.loads(http_request.data.decode("utf-8"))
        self.assertEqual(payload["model"], "deepseek/deepseek-v4-flash")
        self.assertEqual(
            payload["provider"],
            {
                "allow_fallbacks": False,
                "only": ["deepseek", "alibaba"],
                "order": ["deepseek", "alibaba"],
                "require_parameters": True,
            },
        )
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertTrue(payload["response_format"]["json_schema"]["strict"])
        schema = payload["response_format"]["json_schema"]["schema"]
        self.assertIn("assessment_id", schema["required"])
        self.assertIn("cited_evidence_refs", schema["properties"])
        evidence_ref_schema = schema["properties"]["cited_evidence_refs"]["items"]
        self.assertEqual(
            set(evidence_ref_schema["properties"]["evidence_ref_type"]["enum"]),
            {
                "accounting_policy_note_span",
                "note",
                "note_span",
                "related_party_note_span",
                "rule",
                "table_cell",
                "table_row",
                "tool_result",
                "variance_explanation_span",
            },
        )
        self.assertEqual(
            set(evidence_ref_schema["properties"]["role"]["enum"]),
            {"contradicting", "context", "missing_required_context", "refuting", "supporting"},
        )
        signal_schema = schema["properties"]["validated_signals"]["items"]
        self.assertEqual(
            set(signal_schema["properties"]["status"]["enum"]),
            {"not_assessable", "partially_validated", "rejected", "validated"},
        )
        self.assertNotIn("tool_result_id", signal_schema["required"])
        self.assertEqual(payload["temperature"], 0.2)
        system_prompt = payload["messages"][0]["content"]
        self.assertIn("Return a single JSON object with these top-level keys", system_prompt)
        self.assertIn("assessment_id", system_prompt)
        self.assertIn("cited_evidence_refs", system_prompt)
        self.assertIn("Use exact ref_id strings from DetectorPacket.tool_findings[].evidence_refs", system_prompt)
        self.assertIn("report_id:note_id", system_prompt)
        self.assertIn("Do not use local_evidence_id, row_id, note_id, span_id, or cell_id by itself", system_prompt)
        user_packet = json.loads(payload["messages"][1]["content"])
        self.assertEqual(user_packet, packet)
        payload_surface = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn('"output"', payload_surface)
        self.assertNotIn('"split_metadata"', payload_surface)
        self.assertNotIn('"source_file_sha256"', payload_surface)
        self.assertNotIn("secret-not-logged", payload_surface)

    def test_deepseek_client_sends_json_object_request_and_records_cache_usage(self):
        release_record = _read_jsonl(RELEASE_DIR / "validation.jsonl")[0]
        packet = release_record["input"]["data"]
        assessment = release_record["output"]["data"]
        response_payload = {
            "model": "deepseek-v4-pro",
            "system_fingerprint": "fp-baseline-test",
            "choices": [{"message": {"content": json.dumps(assessment, ensure_ascii=False)}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "prompt_cache_hit_tokens": 75,
                "prompt_cache_miss_tokens": 25,
            },
        }

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self):
                return json.dumps(response_payload).encode("utf-8")

        with mock.patch("urllib.request.urlopen", return_value=Response()) as urlopen:
            client = DeepSeekApiLlmDetectorClient(api_key="secret-not-logged")
            response = client.complete(
                request=type(
                    "Request",
                    (),
                    {
                        "example_id": release_record["example_id"],
                        "prompt_version": "api_llm_detector_baseline_prompt_v1_evidence_guard",
                        "provider": "deepseek",
                        "model": "deepseek-v4-pro",
                        "temperature": 0.0,
                        "detector_packet": packet,
                    },
                )()
            )

        self.assertEqual(response.content, response_payload["choices"][0]["message"]["content"])
        http_request = urlopen.call_args.args[0]
        payload = json.loads(http_request.data.decode("utf-8"))
        self.assertEqual(http_request.full_url, DEEPSEEK_CHAT_COMPLETIONS_URL)
        self.assertEqual(payload["model"], "deepseek-v4-pro")
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["temperature"], 0.0)
        self.assertNotIn("provider", payload)
        user_payload = json.loads(payload["messages"][1]["content"])
        self.assertEqual(user_payload["detector_packet"], packet)
        output_contract = user_payload["detector_assessment_output_contract"]
        self.assertEqual(output_contract["copy_identity_fields_exactly"]["packet_id"], packet["packet_id"])
        self.assertIn("assessment_id", output_contract["top_level_keys_exact"])
        self.assertIn("number only", output_contract["confidence"])
        self.assertFalse(output_contract["additional_top_level_keys_allowed"])
        self.assertIn("Confidence must be a number between 0 and 1", payload["messages"][0]["content"])
        payload_surface = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn('"output"', payload_surface)
        self.assertNotIn('"split_metadata"', payload_surface)
        self.assertNotIn('"source_file_sha256"', payload_surface)
        self.assertNotIn("secret-not-logged", payload_surface)
        self.assertEqual(response.provider_metadata["strategy"], "direct_deepseek_chat_completions_v1")
        self.assertEqual(response.provider_metadata["request_model"], "deepseek-v4-pro")
        self.assertEqual(response.provider_metadata["response_model"], "deepseek-v4-pro")
        self.assertEqual(response.provider_metadata["system_fingerprint"], "fp-baseline-test")
        self.assertEqual(response.provider_metadata["cache_usage"]["prompt_cache_hit_tokens"], 75)
        self.assertEqual(response.provider_metadata["cache_usage"]["prompt_cache_miss_tokens"], 25)

    def test_deepseek_client_retries_transient_transport_error(self):
        release_record = _read_jsonl(RELEASE_DIR / "validation.jsonl")[0]
        packet = release_record["input"]["data"]
        assessment = release_record["output"]["data"]
        response_payload = {
            "model": "deepseek-v4-pro",
            "choices": [{"message": {"content": json.dumps(assessment, ensure_ascii=False)}}],
            "usage": {},
        }

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self):
                return json.dumps(response_payload).encode("utf-8")

        with mock.patch("urllib.request.urlopen", side_effect=[urllib.error.URLError("temporary"), Response()]) as urlopen:
            with mock.patch("research.api_llm_detector_baseline.time.sleep") as sleep:
                client = DeepSeekApiLlmDetectorClient(api_key="secret-not-logged")
                response = client.complete(
                    request=type(
                        "Request",
                        (),
                        {
                            "example_id": release_record["example_id"],
                            "prompt_version": "api_llm_detector_baseline_prompt_v1_evidence_guard",
                            "provider": "deepseek",
                            "model": "deepseek-v4-pro",
                            "temperature": 0.0,
                            "detector_packet": packet,
                        },
                    )()
                )

        self.assertEqual(urlopen.call_count, 2)
        self.assertEqual(sleep.call_args.args[0], 1.0)
        self.assertEqual(response.content, response_payload["choices"][0]["message"]["content"])

    def test_openrouter_client_reports_malformed_provider_json_with_body_preview(self):
        release_record = _read_jsonl(RELEASE_DIR / "validation.jsonl")[0]
        packet = release_record["input"]["data"]
        malformed_body = (" " * 6302) + "<html>provider error</html>"

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self):
                return malformed_body.encode("utf-8")

        with mock.patch("urllib.request.urlopen", return_value=Response()):
            client = OpenRouterApiLlmDetectorClient(api_key="secret-not-logged")
            with self.assertRaisesRegex(RuntimeError, "openrouter_non_json_response"):
                client.complete(
                    request=type(
                        "Request",
                        (),
                        {
                            "example_id": release_record["example_id"],
                            "prompt_version": "api_llm_detector_baseline_prompt_v1",
                            "provider": "openrouter",
                            "model": "deepseek/deepseek-v4-flash",
                            "temperature": 0.2,
                            "detector_packet": packet,
                        },
                    )()
                )

    def test_diagnostic_scoring_reports_false_positive_rejection_and_insufficient_evidence_counts(self):
        release_records = _read_jsonl(RELEASE_DIR / "validation.jsonl")
        client = DiagnosticScoringFakeClient(release_records)

        with tempfile.TemporaryDirectory() as temp_dir:
            run_api_llm_detector_baseline(
                release_dir=RELEASE_DIR,
                output_root=Path(temp_dir) / "api_llm_detector_baseline",
                mode="fake",
                provider="openrouter",
                model="fake-model",
                client=client,
                run_id="diagnostic-scoring-test",
            )

            metrics = _read_json(
                Path(temp_dir)
                / "api_llm_detector_baseline"
                / "diagnostic-scoring-test"
                / "metrics.json"
            )

        self.assertEqual(
            metrics["false_positive_rejection"],
            {"correct_or_conservative": 2, "invalid_gold_count": 0, "total": 3},
        )
        self.assertEqual(
            metrics["insufficient_evidence_detection"],
            {"true_positive": 1, "predicted_total": 1, "gold_total": 2, "invalid_gold_count": 0},
        )
        self.assertEqual(metrics["accusation_language_violation_count"], 0)
        self.assertEqual(metrics["outside_packet_reasoning_violation_count"], 0)

    def test_live_mode_requires_openrouter_api_key_before_network_client_is_created(self):
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(
            "os.environ", {}, clear=True
        ), mock.patch(
            "research.api_llm_detector_baseline._load_dotenv_if_available",
            return_value=None,
        ):
            with self.assertRaisesRegex(ValueError, "OPENROUTER_API_KEY"):
                run_api_llm_detector_baseline(
                    release_dir=RELEASE_DIR,
                    output_root=Path(temp_dir) / "api_llm_detector_baseline",
                    mode="live",
                    provider="openrouter",
                    model="live-model",
                    run_id="live-missing-key-test",
                )

    def test_cli_dry_run_uses_public_runner_and_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "research.api_llm_detector_baseline",
                    "--mode",
                    "dry_run",
                    "--output-root",
                    str(Path(temp_dir) / "api_llm_detector_baseline"),
                    "--run-id",
                    "cli-dry-run-test",
                ],
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            run_dir = Path(temp_dir) / "api_llm_detector_baseline" / "cli-dry-run-test"
            self.assertTrue((run_dir / "manifest.json").exists())
            self.assertTrue((run_dir / "predictions.jsonl").exists())

    def test_cli_fake_mode_uses_release_gold_outputs_as_fake_responses(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "research.api_llm_detector_baseline",
                    "--mode",
                    "fake",
                    "--model",
                    "fake-cli-model",
                    "--output-root",
                    str(Path(temp_dir) / "api_llm_detector_baseline"),
                    "--run-id",
                    "cli-fake-test",
                    "--limit",
                    "2",
                ],
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            run_dir = Path(temp_dir) / "api_llm_detector_baseline" / "cli-fake-test"
            predictions = _read_jsonl(run_dir / "predictions.jsonl")
            self.assertEqual(len(predictions), 2)
            self.assertEqual({record["prediction_status"] for record in predictions}, {"accepted"})
            metrics = _read_json(run_dir / "metrics.json")
            self.assertEqual(metrics["num_attempted"], 2)
            self.assertEqual(metrics["num_valid_predictions"], 2)

    def test_cli_sweep_uses_preset_and_writes_sweep_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "research.api_llm_detector_baseline",
                    "--mode",
                    "dry_run",
                    "--model",
                    "deepseek/deepseek-v4-flash",
                    "--output-root",
                    str(Path(temp_dir) / "api_llm_detector_baseline"),
                    "--sweep-id",
                    "cli-sweep-test",
                    "--limit",
                    "1",
                ],
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["status"], "passed")
            sweep_dir = Path(temp_dir) / "api_llm_detector_baseline" / "cli-sweep-test"
            self.assertEqual(Path(payload["sweep_dir"]), sweep_dir)
            manifest = _read_json(sweep_dir / "sweep_manifest.json")
            rows = _read_jsonl(sweep_dir / "sweep_results.jsonl")
            self.assertEqual(manifest["preset"], "deepseek_v4_flash_prompt_config_v1")
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[0]["records_loaded"], 1)

    def test_live_mode_can_run_one_selected_example_with_mocked_openrouter_response(self):
        release_record = _read_jsonl(RELEASE_DIR / "validation.jsonl")[0]
        assessment = release_record["output"]["data"]
        response_payload = {"choices": [{"message": {"content": json.dumps(assessment, ensure_ascii=False)}}]}

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self):
                return json.dumps(response_payload).encode("utf-8")

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(
            "os.environ", {"OPENROUTER_API_KEY": "test-secret-not-logged"}, clear=True
        ), mock.patch("urllib.request.urlopen", return_value=Response()) as urlopen:
            result = run_api_llm_detector_baseline(
                release_dir=RELEASE_DIR,
                output_root=Path(temp_dir) / "api_llm_detector_baseline",
                mode="live",
                provider="openrouter",
                model="live-model",
                example_ids=[release_record["example_id"]],
                run_id="live-one-mocked-test",
            )

            self.assertEqual(result.status, "passed", result.errors)
            self.assertEqual(result.records_loaded, 1)
            self.assertEqual(result.predictions_written, 1)
            self.assertEqual(result.invalid_response_count, 0)
            self.assertEqual(urlopen.call_count, 1)

            run_dir = Path(temp_dir) / "api_llm_detector_baseline" / "live-one-mocked-test"
            predictions = _read_jsonl(run_dir / "predictions.jsonl")
            self.assertEqual(predictions[0]["example_id"], release_record["example_id"])
            self.assertEqual(predictions[0]["prediction_status"], "accepted")

    def test_public_runner_can_limit_or_select_examples_for_smoke_runs(self):
        release_records = _read_jsonl(RELEASE_DIR / "validation.jsonl")
        selected_example_id = release_records[3]["example_id"]
        responses_by_example_id = {
            record["example_id"]: record["output"]["data"]
            for record in release_records
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            limited_result = run_api_llm_detector_baseline(
                release_dir=RELEASE_DIR,
                output_root=Path(temp_dir) / "api_llm_detector_baseline",
                mode="fake",
                provider="openrouter",
                model="fake-model",
                client=FakeClient(responses_by_example_id),
                limit=2,
                run_id="limited-test",
            )
            selected_result = run_api_llm_detector_baseline(
                release_dir=RELEASE_DIR,
                output_root=Path(temp_dir) / "api_llm_detector_baseline",
                mode="fake",
                provider="openrouter",
                model="fake-model",
                client=FakeClient(responses_by_example_id),
                example_ids=[selected_example_id],
                run_id="selected-test",
            )

            self.assertEqual(limited_result.records_loaded, 2)
            self.assertEqual(selected_result.records_loaded, 1)
            selected_predictions = _read_jsonl(
                Path(temp_dir) / "api_llm_detector_baseline" / "selected-test" / "predictions.jsonl"
            )
            self.assertEqual(selected_predictions[0]["example_id"], selected_example_id)


class FakeClient:
    def __init__(self, responses_by_example_id: dict[str, dict]):
        self.responses_by_example_id = responses_by_example_id
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return ApiLlmDetectorResponse(
            content=json.dumps(self.responses_by_example_id[request.example_id], ensure_ascii=False)
        )


class MetadataFakeClient(FakeClient):
    def __init__(self, responses_by_example_id: dict[str, dict], *, provider_metadata: dict):
        super().__init__(responses_by_example_id)
        self.provider_metadata = provider_metadata

    def complete(self, request):
        self.requests.append(request)
        return ApiLlmDetectorResponse(
            content=json.dumps(self.responses_by_example_id[request.example_id], ensure_ascii=False),
            provider_metadata=self.provider_metadata,
        )


class MarkdownWrappedFakeClient(FakeClient):
    def complete(self, request):
        self.requests.append(request)
        content = json.dumps(self.responses_by_example_id[request.example_id], ensure_ascii=False)
        return ApiLlmDetectorResponse(content=f"```json\n{content}\n```")


class InvalidResponseFakeClient:
    def __init__(self, release_records: list[dict]):
        self.records_by_example_id = {record["example_id"]: record for record in release_records}
        self.example_ids = [record["example_id"] for record in release_records]

    def complete(self, request):
        record = self.records_by_example_id[request.example_id]
        assessment = json.loads(json.dumps(record["output"]["data"], ensure_ascii=False))
        index = self.example_ids.index(request.example_id)
        if index == 0:
            return ApiLlmDetectorResponse(content="not json")
        if index == 1:
            del assessment["support_level"]
            return ApiLlmDetectorResponse(content=json.dumps(assessment, ensure_ascii=False))
        if index == 2:
            assessment["packet_id"] = "PACKET_WRONG"
            return ApiLlmDetectorResponse(content=json.dumps(assessment, ensure_ascii=False))
        if index == 3:
            assessment["cited_evidence_refs"][0]["ref_id"] = "INVENTED_EVIDENCE_ID"
            return ApiLlmDetectorResponse(content=json.dumps(assessment, ensure_ascii=False))
        if index == 4:
            assessment["rationale_short"] = "This proves fraud."
            return ApiLlmDetectorResponse(content=json.dumps(assessment, ensure_ascii=False))
        return ApiLlmDetectorResponse(content=json.dumps(assessment, ensure_ascii=False))


class ContractDriftFakeClient:
    def __init__(self, release_records: list[dict]):
        self.records_by_example_id = {record["example_id"]: record for record in release_records}
        self.example_ids = [record["example_id"] for record in release_records]

    def complete(self, request):
        record = self.records_by_example_id[request.example_id]
        assessment = json.loads(json.dumps(record["output"]["data"], ensure_ascii=False))
        index = self.example_ids.index(request.example_id)
        if index == 0:
            assessment["cited_evidence_refs"][0]["evidence_ref_type"] = "signal_finding"
        elif index == 1:
            assessment["cited_evidence_refs"][0]["role"] = "primary_metric"
        elif index == 2:
            assessment["validated_signals"][0]["status"] = "confirmed"
        elif index == 3:
            assessment["validated_signals"] = []
        return ApiLlmDetectorResponse(content=json.dumps(assessment, ensure_ascii=False))


class DiagnosticScoringFakeClient:
    def __init__(self, release_records: list[dict]):
        self.records_by_example_id = {record["example_id"]: record for record in release_records}
        self.insufficient_seen = 0

    def complete(self, request):
        record = self.records_by_example_id[request.example_id]
        assessment = json.loads(json.dumps(record["output"]["data"], ensure_ascii=False))
        gold_support = record["output"]["data"]["support_level"]
        if gold_support == "not_supported":
            assessment["support_level"] = "supported"
        if gold_support == "insufficient_evidence":
            self.insufficient_seen += 1
            if self.insufficient_seen == 1:
                assessment["support_level"] = "insufficient_evidence"
            else:
                assessment["support_level"] = "not_supported"
        return ApiLlmDetectorResponse(content=json.dumps(assessment, ensure_ascii=False))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _file_hashes(root: Path, filenames: list[str]) -> dict[str, str]:
    return {
        filename: hashlib.sha256((root / filename).read_bytes()).hexdigest()
        for filename in filenames
    }


if __name__ == "__main__":
    unittest.main()
