# wsp-automode-0.8b

> **Note:** This is an independent, experimental **Weight-Space Programming (WSP)** research repository built around [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B). It is **not** an official release of [`Mapika/decider`](https://github.com/Mapika/decider) or [`Chiytako/decider-auto`](https://github.com/Chiytako/decider-auto).

## Research Goal

Construct and rigorously evaluate a **single-model, strictly binary (`Allow` / `Block`), `llama.cpp` (`Q8_0` GGUF) compatible** coding-agent Auto-Mode permission classifier on **`Qwen/Qwen3.5-0.8B`** **without gradient descent** (zero `.backward()`, `optimizer.step()`, or LoRA training in Tracks 1 & 2), relying instead on:

1. **Track 0 (Diagnostics & Oracle Upper Bound)**: Plain Zero-Shot baseline vs. Oracle Policy + Few-Shot upper bound, 24-layer Logit Lens, 4-bucket error taxonomy, and Plain-vs-Oracle slot activation geometry.
2. **Track 1 (Closed-Form Weight Editing)**: Context-to-Weight distillation, position-conditioned rank-1 updates (`STEER2EDIT`-style), `AlphaEdit`-style null-space projection against general `decider` replay activations, and 3-point causal sanity checks (Dose-Response, Sign-Reversal, Orthogonal Random Control).
3. **Track 2 (Agent-Driven Weight-Space Programming)**: Multi-trajectory inspect -> hypothesize -> patch -> 3-point causal check -> 3-axis regression gate loop, followed by Leave-One-Out ablation and a single locked `frozen_test` evaluation.
4. **Runtime Parity (`Q8_0` GGUF)**: Exact PyTorch (`bf16`) vs. `llama-server` (`Q8_0`) parity verification, `second_reading` option-order debiasing, and harness-trusted `Blocked action` feedback loop.

## Non-Negotiables

- **100% Gradient-Free in Tracks 1 & 2** (existing tensor values modified in place; zero architecture changes).
- **Plain Layout & Reasoning-Blind**: BOS-free, no chat template, `<|endoftext|>` right padding, 1-pass slot readout (`Answer: (`), no assistant prose or tool outputs in classifier history.
- **Zero-Leakage 3-Split Protocol**: Strict separation into `observe` / `select_dev` / `frozen_test` across project groups and scenario families, with `frozen_test` locked by SHA-256 until Track 2 concludes.
- **Privacy Hygiene**: No raw private session transcripts, local credentials, or model weights are stored in this repository.

## Hardware & Measured Phase 0 Baseline

- **GPU**: NVIDIA GeForce RTX 3050 Ti Laptop GPU (`3770.81 MB` VRAM, `bf16` enabled; `torch 2.15.0+cu132`, `transformers 5.24.0`).
- **Backbone**: `Qwen/Qwen3.5-0.8B` — 24 layers (`full_attention_interval = 4`: 18 `Qwen3_5GatedDeltaNet` layers + 6 `Qwen3_5Attention` layers at indices `3, 7, 11, 15, 19, 23`).
- **4GB VRAM Footprint (`Qwen3_5ForCausalLM` + slot-only `lm_head`)**: `1459.10 MB` loaded; `1921.61 MB` peak allocated at `4088` tokens (`1071.84 ms` eager PyTorch).

See [`PROGRESS.md`](PROGRESS.md) for current experiment status and [`docs/phase0_design.md`](docs/phase0_design.md) for full Phase 0 measurements.
