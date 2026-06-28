# External Artifact Manifest

Large artifacts that cannot fit in the repository are hosted on Google Drive.
Each folder contains a `checksums.sha256` file with per-file SHA-256 hashes.

**Drive root:** https://drive.google.com/drive/folders/10Ib0XvIiyL7zcEOGOzpNWMPOBATuzdFV

| # | Folder | Contents | Approx size |
|---|---|---|---|
| 1 | `01_synthetic_detector_dataset_release` | Train/val/test/excluded JSONL splits, SFT chat export, leakage report, manifest, metrics | ~72 MB |
| 2 | `02_real_manual_validation_dataset_release` | 14-case real/manual validation JSONL, manifest | ~180 KB |
| 3 | `03_qwen_lora_detector_adapter` | LoRA adapter weights, tokenizer, chat template, configs (base model `unsloth/Qwen3.5-4B` not included) | ~101 MB |
| 4 | `04_runtime_reproducibility_artifacts` | 8 source PDFs/ZIPs, 6 cached OCR JSONs, 3 clean structured reports for NKG/HAP/NHC scenarios | ~83 MB |
| 5 | `05_runtime_demo_evidence` | 6-run QA/QC metrics pack, runtime registry, provider cost reconciliation, per-run JSONs | ~3 MB |

## Usage

1. Clone the repo and install dependencies per `README.md`.
2. Download folders 04 and 05 into `artifacts/` to run Sealed Demo Replay or Cached-First Live Confirmation.
3. Folders 01–03 support detector training and evaluation — not required for the demo.
