# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SWIFT (Scalable lightWeight Infrastructure for Fine-Tuning) is a comprehensive framework by ModelScope for LLM and multimodal model training, inference, evaluation, quantization, and deployment. It supports 600+ text LLMs and 300+ multimodal LLMs. Current version is 4.0.0.dev0 (in development).

## Common Commands

### Installation
```bash
pip install -e .                    # Development install from source
pip install ms-swift -U             # Release install
```

### CLI Entry Point
The `swift` command is the main entry point (swift.cli.main:cli_main). Key subcommands:
```bash
swift sft      # Supervised fine-tuning
swift pt       # Pre-training
swift rlhf     # RLHF training (GRPO, DPO, KTO, CPO, PPO, etc.)
swift infer    # Inference
swift deploy   # Deployment server
swift eval     # Evaluation
swift export   # Model export/quantization
swift app      # Gradio app
swift web-ui   # Web UI
swift merge-lora  # Merge LoRA weights
swift sample   # Sampling
swift rollout  # RL rollout
```

Distributed training uses `NPROC_PER_NODE` env var (triggers torchrun internally). Config files supported via `--config <yaml_file>`.

### Megatron Entry Point
```bash
megatron sft   # Megatron-based SFT (separate CLI)
megatron pt    # Megatron-based pre-training
```

### Linting
```bash
pip install pre-commit
pre-commit run --all-files
```
Style: yapf (pep8-based), isort, flake8. Line length: 120. Double quotes converted to single quotes.

### Testing
```bash
python tests/run.py --parallel 2 --run_config tests/run_config.yaml
```

### Build
```bash
make whl       # Build wheel (python setup.py sdist bdist_wheel)
make docs      # Build Sphinx documentation
make linter    # Run linter
make test      # Run CI tests
```

### Documentation
```bash
pip install -r requirements/docs.txt
cd docs && make html
```

## Architecture

### Source Layout (swift/)

The codebase (~91k lines Python) follows a pipeline architecture:

**Arguments → Model/Tokenizer → Template → Dataset → Tuner → Trainer → Checkpoint**

Key packages:
- `arguments/` — Argument dataclasses for each command (SftArgs, RlhfArgs, InferArgs, etc.) with shared base classes in `base_args/`
- `cli/` — CLI routing; dispatches `swift <cmd>` to corresponding pipeline. `_megatron/` has separate Megatron CLI
- `pipelines/` — Execution pipelines: `train/` (sft, pretrain, rlhf), `infer/`, `export/`, `eval/`, `app/`, `sampling/`
- `model/` — Model registration, loading, and patching. `models/` has per-model implementations (~32 model families). `model_meta.py` and `register.py` define the model registry
- `template/` — Chat/prompt templates for each model family
- `dataset/` — Dataset loading, preprocessing, and packing. `preprocessor/` handles format conversion
- `tuners/` — Training adapters: LoRA, QLoRA, DoRA, Adapter, LLaMAPro, ReFT, etc. Built on top of PEFT
- `trainers/` — Core trainers extending HuggingFace Trainer. `trainer_factory.py` selects the right trainer
- `rlhf_trainers/` — RLHF-specific trainers: GRPO (major component), DPO, KTO, CPO, PPO, ORPO, GKD, reward model
- `infer_engine/` — Inference backends: transformers, vLLM, SGLang, LMDeploy
- `megatron/` — Full Megatron parallelism support (TP, PP, CP, EP) with its own arguments, models, trainers, and pipelines
- `sequence_parallel/` — Ulysses and Ring-Attention sequence parallelism
- `rewards/` — Reward models/functions for RLHF
- `rollout/` — RL rollout infrastructure
- `ray/` — Ray distributed training integration
- `ui/` — Gradio web UI components
- `utils/` — Shared utilities

### Design Patterns
- **Lazy Module Loading:** `_LazyModule` in `__init__.py` for fast imports
- **Factory Pattern:** `TrainerFactory` creates appropriate trainer based on args
- **Mixin Pattern:** `TunerMixin`, `RLHFMixin` for composable trainer functionality
- **Pipeline Pattern:** `SwiftPipeline` base class for all execution pipelines
- **Plugin System:** Extensible `tuner_plugin/`, `callbacks/`, `loss/`, `metrics/`

### Adding a New Model
Models are registered in `swift/model/`. Each model family has a file in `swift/model/models/` that defines `ModelMeta` with: model architecture, supported features, template mapping, and any required patches. Registration happens via `swift/model/register.py`.

### Adding a New Dataset
Datasets are defined in `swift/dataset/`. Custom datasets can be loaded via file paths or HuggingFace/ModelScope dataset IDs. Format preprocessors in `swift/dataset/preprocessor/` handle conversion to the internal format.

## Code Standards
- Variable naming: underscore_separated; class naming: CapitalizedWords
- Indentation: 4 spaces (no tabs)
- Line length: 120 characters
- Formatting enforced by yapf, isort, flake8 via pre-commit hooks
- Excluded from linting: thirdparty/, examples/, tests/run.py

## Key Dependencies
- transformers >=4.33, peft >=0.11, trl >=0.15, accelerate, datasets >=3.0
- PyTorch >=2.0 (recommended 2.8.0/2.9.0)
- Optional: vllm, sglang, lmdeploy, flash_attn, deepspeed, megatron-core
