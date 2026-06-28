import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from research.detector_sft_trainer import (
    ADAPTER_ID,
    BASE_MODEL_ID,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SAVE_TOTAL_LIMIT,
    DISPLAY_LABEL,
    MAX_EPOCHS,
    _apply_chat_template_for_qwen35,
    _run_live_unsloth_training,
    _training_rows_for_sft,
    load_detector_sft_dataset,
    preflight_detector_sft_token_lengths,
    run_detector_sft_training,
)


class Wave4DetectorSftTrainerTest(unittest.TestCase):
    def test_loader_partitions_official_chat_export_by_metadata_split_without_changing_messages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sft_jsonl = Path(temp_dir) / "detector_sft_chat.jsonl"
            rows = [
                _chat_row("train-1", "train"),
                _chat_row("validation-1", "validation"),
                _chat_row("test-1", "test"),
            ]
            _write_jsonl(sft_jsonl, rows)

            dataset = load_detector_sft_dataset(sft_jsonl, allow_noncanonical_sft_jsonl=True)

            self.assertEqual([row.example_id for row in dataset.train_rows], ["train-1"])
            self.assertEqual([row.example_id for row in dataset.validation_rows], ["validation-1"])
            self.assertEqual([row.example_id for row in dataset.test_rows], ["test-1"])
            self.assertEqual(dataset.split_counts, {"test": 1, "train": 1, "validation": 1})
            self.assertEqual(dataset.train_rows[0].messages, rows[0]["messages"])
            self.assertEqual(dataset.validation_rows[0].messages, rows[1]["messages"])
            self.assertEqual(dataset.test_rows[0].messages, rows[2]["messages"])

    def test_training_rows_can_oversample_only_weakly_supported_train_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sft_jsonl = Path(temp_dir) / "detector_sft_chat.jsonl"
            _write_jsonl(
                sft_jsonl,
                [
                    _chat_row("train-supported", "train", support_level="supported"),
                    _chat_row("train-weak", "train", support_level="weakly_supported"),
                    _chat_row("validation-weak", "validation", support_level="weakly_supported"),
                    _chat_row("test-weak", "test", support_level="weakly_supported"),
                ],
            )
            dataset = load_detector_sft_dataset(sft_jsonl, allow_noncanonical_sft_jsonl=True)

            training_rows = _training_rows_for_sft(dataset, weakly_supported_train_multiplier=3)

            self.assertEqual(
                [row.example_id for row in training_rows],
                ["train-supported", "train-weak", "train-weak", "train-weak"],
            )

    def test_tokenizer_preflight_hard_fails_on_train_or_validation_rows_over_sequence_cap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sft_jsonl = Path(temp_dir) / "detector_sft_chat.jsonl"
            _write_jsonl(
                sft_jsonl,
                [
                    _chat_row("train-1", "train", token_count=4),
                    _chat_row("validation-1", "validation", token_count=6),
                    _chat_row("test-1", "test", token_count=100),
                ],
            )
            dataset = load_detector_sft_dataset(sft_jsonl, allow_noncanonical_sft_jsonl=True)
            tokenizer = TokenCountingTokenizer()

            with self.assertRaisesRegex(ValueError, "validation-1.*6.*5"):
                preflight_detector_sft_token_lengths(dataset, tokenizer=tokenizer, max_seq_length=5)

            self.assertEqual(tokenizer.checked_example_ids, ["train-1", "validation-1"])

    def test_tokenizer_preflight_adapts_plain_chat_messages_for_qwen35_processor_without_mutating_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sft_jsonl = Path(temp_dir) / "detector_sft_chat.jsonl"
            rows = [
                _chat_row("train-1", "train", token_count=4),
                _chat_row("validation-1", "validation", token_count=5),
            ]
            _write_jsonl(sft_jsonl, rows)
            dataset = load_detector_sft_dataset(sft_jsonl, allow_noncanonical_sft_jsonl=True)
            processor = Qwen35TextBlockProcessor()

            preflight = preflight_detector_sft_token_lengths(dataset, tokenizer=processor, max_seq_length=10)

            self.assertEqual(preflight.max_token_count, 5)
            self.assertEqual(processor.checked_example_ids, ["train-1", "validation-1"])
            self.assertEqual(dataset.train_rows[0].messages, rows[0]["messages"])
            self.assertIsInstance(dataset.train_rows[0].messages[1]["content"], str)

    def test_qwen35_chat_template_adapter_forwards_generation_kwargs_after_text_block_fallback(self) -> None:
        processor = Qwen35TextBlockProcessor()
        messages = _chat_row("validation-1", "validation", token_count=5)["messages"][:2]

        _apply_chat_template_for_qwen35(
            processor,
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
        )

        self.assertEqual(processor.last_kwargs, {"enable_thinking": False, "return_tensors": "pt"})

    def test_dry_run_public_training_path_writes_locked_lean_artifacts_without_using_test_for_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sft_jsonl = root / "detector_sft_chat.jsonl"
            _write_jsonl(
                sft_jsonl,
                [
                    _chat_row("train-1", "train", token_count=4),
                    _chat_row("validation-1", "validation", token_count=5),
                    _chat_row("test-1", "test", token_count=100),
                ],
            )
            output_root = root / "artifacts"

            result = run_detector_sft_training(
                sft_jsonl=sft_jsonl,
                artifact_root=output_root,
                mode="dry_run",
                tokenizer=TokenCountingTokenizer(),
                allow_noncanonical_sft_jsonl=True,
            )

            self.assertEqual(result.status, "passed", result.errors)
            self.assertEqual(result.artifact_root, output_root)
            self.assertTrue((output_root / "adapter" / "DRY_RUN_ADAPTER.txt").exists())
            self.assertTrue((output_root / "training_config.json").exists())
            self.assertTrue((output_root / "data_manifest.json").exists())
            self.assertTrue((output_root / "metrics.json").exists())
            self.assertTrue((output_root / "validation_selection_summary.json").exists())
            self.assertTrue((output_root / "eval_summary.json").exists())

            training_config = _read_json(output_root / "training_config.json")
            self.assertEqual(training_config["base_model"], BASE_MODEL_ID)
            self.assertEqual(training_config["training_runtime"], "unsloth_bf16_lora")
            self.assertEqual(training_config["adapter_id"], ADAPTER_ID)
            self.assertEqual(training_config["display_label"], DISPLAY_LABEL)
            self.assertEqual(training_config["artifact_root"], str(output_root))
            self.assertEqual(training_config["max_seq_length"], 8192)
            self.assertEqual(training_config["max_epochs"], MAX_EPOCHS)
            self.assertEqual(
                training_config["model_load_config"],
                {"full_finetuning": False, "load_in_16bit": True, "load_in_4bit": False},
            )
            self.assertEqual(training_config["trainer_config"]["per_device_train_batch_size"], 4)
            self.assertEqual(training_config["trainer_config"]["per_device_eval_batch_size"], 1)
            self.assertEqual(training_config["trainer_config"]["gradient_accumulation_steps"], 2)
            self.assertEqual(training_config["trainer_config"]["learning_rate"], DEFAULT_LEARNING_RATE)
            self.assertEqual(training_config["trainer_config"]["save_strategy"], "epoch")
            self.assertEqual(training_config["trainer_config"]["eval_strategy"], "epoch")
            self.assertNotIn("eval_steps", training_config["trainer_config"])
            self.assertEqual(training_config["trainer_config"]["save_total_limit"], DEFAULT_SAVE_TOTAL_LIMIT)
            self.assertTrue(training_config["use_official_chat_messages_exactly"])
            self.assertFalse(training_config["implements_sft_vllm_runtime"])

            data_manifest = _read_json(output_root / "data_manifest.json")
            self.assertEqual(data_manifest["split_counts"], {"test": 1, "train": 1, "validation": 1})
            self.assertEqual(data_manifest["trainer_splits"], {"train": 1, "validation": 1})
            self.assertEqual(data_manifest["effective_trainer_splits"], {"train": 1, "validation": 1})
            self.assertEqual(data_manifest["train_sampling_policy"]["weakly_supported_train_multiplier"], 1)
            self.assertEqual(data_manifest["reserved_splits"], {"test": 1})
            self.assertTrue(data_manifest["source_policy"]["only_training_export"])
            self.assertTrue(data_manifest["source_policy"]["real_manual_rows_excluded"])

            selection = _read_json(output_root / "validation_selection_summary.json")
            self.assertEqual(selection["selection_split"], "validation")
            self.assertEqual(selection["selection_metric"], "eval_loss")
            self.assertEqual(selection["selected_checkpoint_source"], "dry_run_no_checkpoint")
            self.assertFalse(selection["test_split_used_for_selection"])

            eval_summary = _read_json(output_root / "eval_summary.json")
            self.assertEqual(eval_summary["synthetic_test_rows_reserved"], 1)
            self.assertEqual(eval_summary["test_split_role"], "post_training_synthetic_regression_only")
            self.assertFalse(eval_summary["runtime_inference_integration_implemented"])
            self.assertFalse(eval_summary["api_baseline_predictions_used_as_labels"])

    def test_dry_run_records_custom_overnight_experiment_hparams(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sft_jsonl = root / "detector_sft_chat.jsonl"
            _write_jsonl(
                sft_jsonl,
                [
                    _chat_row("train-1", "train"),
                    _chat_row("validation-1", "validation"),
                    _chat_row("test-1", "test"),
                ],
            )
            output_root = root / "artifacts"

            result = run_detector_sft_training(
                sft_jsonl=sft_jsonl,
                artifact_root=output_root,
                mode="dry_run",
                tokenizer=TokenCountingTokenizer(),
                allow_noncanonical_sft_jsonl=True,
                max_epochs=6,
                learning_rate=1e-4,
                eval_steps=180,
                save_total_limit=2,
                weakly_supported_train_multiplier=3,
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
            )

            self.assertEqual(result.status, "passed", result.errors)
            training_config = _read_json(output_root / "training_config.json")
            self.assertEqual(training_config["max_epochs"], 6)
            self.assertEqual(training_config["trainer_config"]["num_train_epochs"], 6)
            self.assertEqual(training_config["trainer_config"]["learning_rate"], 1e-4)
            self.assertEqual(training_config["trainer_config"]["eval_strategy"], "steps")
            self.assertEqual(training_config["trainer_config"]["save_strategy"], "steps")
            self.assertEqual(training_config["trainer_config"]["eval_steps"], 180)
            self.assertEqual(training_config["trainer_config"]["save_steps"], 180)
            self.assertEqual(training_config["trainer_config"]["save_total_limit"], 2)
            self.assertEqual(training_config["trainer_config"]["per_device_train_batch_size"], 1)
            self.assertEqual(training_config["trainer_config"]["per_device_eval_batch_size"], 1)
            self.assertEqual(training_config["train_sampling_policy"]["weakly_supported_train_multiplier"], 3)

            metrics = _read_json(output_root / "metrics.json")
            self.assertEqual(metrics["max_epochs"], 6)
            self.assertEqual(metrics["learning_rate"], 1e-4)
            self.assertEqual(metrics["eval_steps"], 180)
            self.assertEqual(metrics["weakly_supported_train_multiplier"], 3)

            selection = _read_json(output_root / "validation_selection_summary.json")
            self.assertEqual(selection["max_epochs"], 6)

    def test_public_command_runs_dry_run_smoke_and_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sft_jsonl = root / "detector_sft_chat.jsonl"
            _write_jsonl(
                sft_jsonl,
                [
                    _chat_row("train-1", "train"),
                    _chat_row("validation-1", "validation"),
                    _chat_row("test-1", "test"),
                ],
            )
            output_root = root / "artifacts"

            completed = _run_trainer_command(
                "--mode",
                "dry_run",
                "--sft-jsonl",
                str(sft_jsonl),
                "--artifact-root",
                str(output_root),
                "--allow-noncanonical-sft-jsonl",
                "--max-epochs",
                "6",
                "--learning-rate",
                "0.0001",
                "--eval-steps",
                "180",
                "--save-total-limit",
                "2",
                "--weakly-supported-train-multiplier",
                "3",
                "--per-device-train-batch-size",
                "1",
                "--per-device-eval-batch-size",
                "1",
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(completed.stdout)["status"], "passed")
            data_manifest = _read_json(output_root / "data_manifest.json")
            self.assertEqual(data_manifest["trainer_splits"], {"train": 1, "validation": 1})
            self.assertEqual(data_manifest["reserved_splits"], {"test": 1})
            training_config = _read_json(output_root / "training_config.json")
            self.assertEqual(training_config["trainer_config"]["num_train_epochs"], 6)
            self.assertEqual(training_config["trainer_config"]["learning_rate"], 1e-4)
            self.assertEqual(training_config["trainer_config"]["eval_steps"], 180)
            self.assertEqual(training_config["trainer_config"]["per_device_train_batch_size"], 1)
            self.assertEqual(training_config["trainer_config"]["per_device_eval_batch_size"], 1)
            self.assertEqual(training_config["train_sampling_policy"]["weakly_supported_train_multiplier"], 3)

    def test_live_training_preserves_text_columns_and_evaluates_trainer_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sft_jsonl = root / "detector_sft_chat.jsonl"
            _write_jsonl(
                sft_jsonl,
                [
                    _chat_row("train-1", "train"),
                    _chat_row("validation-1", "validation"),
                    _chat_row("test-1", "test"),
                ],
            )
            dataset = load_detector_sft_dataset(sft_jsonl, allow_noncanonical_sft_jsonl=True)
            calls: dict[str, object] = {}

            class FakeDataset:
                def __init__(self, rows: list[dict[str, str]]) -> None:
                    self.rows = rows

                @classmethod
                def from_list(cls, rows: list[dict[str, str]]) -> "FakeDataset":
                    return cls(rows)

            class FakeSftConfig:
                def __init__(self, **kwargs: object) -> None:
                    self.kwargs = kwargs
                    calls["config"] = self

            class FakeModel:
                def save_pretrained(self, path: Path) -> None:
                    calls["model_save_path"] = path

            class FakeTokenizer:
                def apply_chat_template(
                    self,
                    messages: list[dict[str, str]],
                    *,
                    tokenize: bool,
                    add_generation_prompt: bool,
                ) -> str:
                    self.last_add_generation_prompt = add_generation_prompt
                    if tokenize:
                        raise AssertionError("live training text rows should request rendered text")
                    return "\n".join(message["content"] for message in messages)

                def save_pretrained(self, path: Path) -> None:
                    calls["tokenizer_save_path"] = path

            class FakeFastLanguageModel:
                @staticmethod
                def get_peft_model(model: FakeModel, **kwargs: object) -> FakeModel:
                    calls["lora_kwargs"] = kwargs
                    return model

            class FakeSftTrainer:
                def __init__(
                    self,
                    *,
                    model: FakeModel,
                    processing_class: FakeTokenizer,
                    train_dataset: FakeDataset,
                    eval_dataset: FakeDataset,
                    args: FakeSftConfig,
                ) -> None:
                    self.model = model
                    self.processing_class = processing_class
                    self.train_dataset = train_dataset
                    self.eval_dataset = eval_dataset
                    self.args = args
                    self.state = SimpleNamespace(best_model_checkpoint="checkpoint-1077", best_metric=0.2188)
                    self.evaluate_calls: list[dict[str, object]] = []
                    calls["trainer"] = self

                def train(self) -> SimpleNamespace:
                    return SimpleNamespace(training_loss=0.08838)

                def evaluate(self, **kwargs: object) -> dict[str, float]:
                    self.evaluate_calls.append(kwargs)
                    if "eval_dataset" in kwargs:
                        raise AssertionError("final live eval should reuse SFTTrainer's prepared eval dataset")
                    return {"eval_loss": float("nan"), "eval_runtime": 1.0}

            fake_modules = {
                "datasets": SimpleNamespace(Dataset=FakeDataset),
                "trl": SimpleNamespace(SFTConfig=FakeSftConfig, SFTTrainer=FakeSftTrainer),
            }
            live_assets = {
                "model": FakeModel(),
                "tokenizer": FakeTokenizer(),
                "fast_language_model": FakeFastLanguageModel,
            }

            with patch.dict(sys.modules, fake_modules):
                summary = _run_live_unsloth_training(
                    dataset,
                    output_root=root / "artifacts",
                    adapter_dir=root / "artifacts" / "adapter",
                    live_assets=live_assets,
                    max_epochs=6,
                    learning_rate=1e-4,
                    eval_steps=180,
                    save_total_limit=2,
                    training_rows=[dataset.train_rows[0], dataset.train_rows[0]],
                    per_device_train_batch_size=1,
                    per_device_eval_batch_size=1,
                )

            config = calls["config"]
            trainer = calls["trainer"]
            self.assertEqual(config.kwargs["num_train_epochs"], 6)
            self.assertEqual(config.kwargs["learning_rate"], 1e-4)
            self.assertEqual(config.kwargs["eval_strategy"], "steps")
            self.assertEqual(config.kwargs["save_strategy"], "steps")
            self.assertEqual(config.kwargs["eval_steps"], 180)
            self.assertEqual(config.kwargs["save_steps"], 180)
            self.assertEqual(config.kwargs["save_total_limit"], 2)
            self.assertEqual(config.kwargs["per_device_train_batch_size"], 1)
            self.assertEqual(config.kwargs["per_device_eval_batch_size"], 1)
            self.assertFalse(config.kwargs["remove_unused_columns"])
            self.assertEqual(len(trainer.train_dataset.rows), 2)
            self.assertEqual(trainer.evaluate_calls, [{}])
            self.assertEqual(summary["best_model_checkpoint"], "checkpoint-1077")
            self.assertEqual(summary["best_eval_loss"], 0.2188)
            self.assertEqual(summary["metrics"], {"train_loss": 0.08838, "eval_loss": 0.2188})


class TokenCountingTokenizer:
    def __init__(self) -> None:
        self.checked_example_ids: list[str] = []

    def apply_chat_template(self, messages: list[dict], *, tokenize: bool, add_generation_prompt: bool) -> list[int]:
        self.checked_example_ids.append(json.loads(messages[1]["content"])["packet_id"])
        token_count = json.loads(messages[1]["content"])["token_count"]
        return list(range(token_count))


class Qwen35TextBlockProcessor:
    def __init__(self) -> None:
        self.checked_example_ids: list[str] = []
        self.last_kwargs: dict = {}

    def apply_chat_template(self, messages: list[dict], *, tokenize: bool, add_generation_prompt: bool, **kwargs):
        self.last_kwargs = kwargs
        first_content = messages[0]["content"]
        if isinstance(first_content, str):
            raise TypeError("string indices must be integers, not 'str'")
        user_text = messages[1]["content"][0]["text"]
        self.checked_example_ids.append(json.loads(user_text)["packet_id"])
        token_count = json.loads(user_text)["token_count"]
        if tokenize:
            return {"input_ids": [list(range(token_count))]}
        return "\n".join(message["content"][0]["text"] for message in messages)


def _chat_row(
    example_id: str,
    split: str,
    *,
    source_type: str = "synthetic_injected_filtered",
    token_count: int = 3,
    support_level: str = "supported",
    severity: str = "medium",
) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "locked system prompt"},
            {"role": "user", "content": json.dumps({"packet_id": example_id, "token_count": token_count}, sort_keys=True)},
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "assessment_id": example_id,
                        "evidence_refs": [],
                        "severity": severity,
                        "support_level": support_level,
                    },
                    sort_keys=True,
                ),
            },
        ],
        "metadata": {
            "example_id": example_id,
            "source_type": source_type,
            "split": split,
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_trainer_command(*args: str) -> subprocess.CompletedProcess:
    repo_root = Path(__file__).resolve().parents[3]
    pythonpath = os.pathsep.join([str(repo_root / "src"), str(repo_root)])
    return subprocess.run(
        [sys.executable, "-m", "research.detector_sft_trainer", *args],
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONPATH": pythonpath},
        check=False,
    )
