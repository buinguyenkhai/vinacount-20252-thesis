# Research Tools

Research and evaluation code for the Vinacount detector pipeline. These modules
implement the grounded synthetic data generation, label gating, dataset
management, detector training, and evaluation described in the thesis.

## Synthetic Data Pipeline

| Module | Purpose |
|---|---|
| `grounded_synthetic_packet_generator.py` | Grounded injection over clean Vietnamese quarterly report artifacts |
| `synthetic_raw_stager.py` | Stage raw synthetic candidates with metadata |
| `synthetic_raw_candidate_validation.py` | Deterministic pre-gate validation of raw candidates |
| `synthetic_detector_assessment_gate.py` | Teacher/judge label gate (best-of-N generation + filtering) |
| `detector_split_builder.py` | Decontaminated train/val/test split building |
| `sft_exporter.py` | Chat-style SFT export from split releases |

## Detector Training

| Module | Purpose |
|---|---|
| `detector_sft_trainer.py` | Unsloth bf16 LoRA fine-tuning for Qwen3.5-4B |

## Evaluation and Baselines

| Module | Purpose |
|---|---|
| `detector_sft_evaluator.py` | SFT LoRA detector evaluation |
| `detector_rule_baseline.py` | Deterministic rule-only detector baseline |
| `api_llm_detector_baseline.py` | API LLM detector baseline (DeepSeek V4 Flash/Pro) |
| `detector_hybrid_evaluator.py` | Hybrid evaluation combining multiple approaches |
| `detector_api_sft_evaluator.py` | API vs SFT comparative evaluation |
| `detector_api_ref_normalizer.py` | API reference normalization for comparison |
| `detector_ablation_summary.py` | Ablation study summary across detector variants |
| `detector_weak_calibration_decision.py` | Weak-support calibration analysis |
| `detector_weak_error_analysis.py` | Structured error analysis for weak signals |

## Calibration Datasets

| Module | Purpose |
|---|---|
| `detector_cashflow_evidence_structure_dataset.py` | Cash-flow evidence structure calibration |
| `detector_corroboration_dataset.py` | Corroboration evidence calibration |
| `detector_corroboration_decision.py` | Corroboration decision logic |
| `detector_real_shape_alignment_dataset.py` | Real-shape alignment calibration |

## Dataset Validation

| Module | Purpose |
|---|---|
| `dataset_validator.py` | Release-level dataset validation |
| `dataset_eligibility.py` | Record eligibility checks |
| `dataset_traceability.py` | Source-to-dataset traceability links |
| `detector_contract_validation.py` | DetectorPacket/DetectorAssessment contract validation |

## Supporting

| Module | Purpose |
|---|---|
| `fixture_spine.py` | Shared fixture builders and validators |
| `real_source_extraction_adapter.py` | Real source PDF extraction adapter |
| `live_vllm_detector_rebaseline.py` | Live vLLM detector rebaseline |
| `testing.py` | Test utilities |

## Data

Modules that train or evaluate detectors expect dataset files under `data/`.
Download folders 01 and 02 from the
[external artifact manifest](../SUBMISSION_ARTIFACT_MANIFEST.md) and place them
as:

```
data/
  synthetic_detector_dataset/   ← folder 01
  combined_real_manual_validation_release/   ← folder 02
```

## Running Tests

```bash
PYTHONPATH=src:. pytest research/tests/
```
