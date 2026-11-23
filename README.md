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

## Measured Results (Phases 0–4: 100% Gradient-Free)

### 1. `select_dev` (n=200) & Locked `frozen_test` (n=200, Evaluated Once Only)

| System | `% modified` (`\|\|ΔW\|\|_F`) | Split | `real_cf` AUROC (SR) | `real_cf` Recall / FPR | `consent_5p` Acc (`P2 / P3`) | Order Flip (`rc / c5`) | `long_50t` (`recap ON`) | `decider_replay` Acc / ECE |
|---|---:|---|---|---|---|---|---|---|
| **Track 0a (Zero-Shot)** | `0.00%` (`0.0`) | `select_dev` | `0.5697` | `0.0500 / 0.0000` | `0.3333` (`1.00 / 0.00`) | `0.9625 / 1.0000` | `0.5833` (`0.5833`) | `0.5333 / 0.1687` |
| **Track 0c (`oracle_chat`)** | `0.00%` (`0.0`) | `select_dev` | `0.6500` (`0.7394`) | `0.9000 / 0.7250` | `0.6250` (`0.375 / 0.875`) | `0.6125 / 0.6667` | `0.5833` (`0.5000`) | `0.5333 / 0.1687` |
| **Track 1 (`L15` SwiGLU)** | `0.001633%` (`8.0109`) | `select_dev` | `0.8512` (`0.8644`) | `0.7000 / 0.2000` | `0.7292` (`0.125 / 1.000`) | `0.1500 / 0.2708` | `0.5000` (`0.5833`) | `0.5500 / 0.1366` |
| **Track 2 (`L15+L22` WSP)** | `0.003264%` (`11.8885`) | `select_dev` | **`0.8731` (`0.8753`)** | `0.6250 / 0.1750` | **`0.7500`** (`0.375 / 0.875`) | `0.2375 / 0.1042` | `0.5000` (`0.5833`) | `0.5333 / 0.1407` |
| **Track 0a (Frozen)** | `0.00%` (`0.0`) | `frozen_test` | `0.6587` (`0.6353`) | `0.0250 / 0.0000` | `0.3958` (`0.875 / 0.125`) | `0.9875 / 0.8958` | `0.5833` (`0.6667`) | `0.5000 / 0.1989` |
| **Track 0c (Frozen)** | `0.00%` (`0.0`) | `frozen_test` | `0.6416` (`0.6600`) | `1.0000 / 0.9000` | `0.5208` (`0.375 / 0.625`) | `0.2750 / 0.2292` | `0.5000` (`0.5000`) | `0.5000 / 0.1989` |
| **Track 1 (Frozen)** | `0.001633%` (`8.0109`) | `frozen_test` | `0.8213` (`0.8241`) | `0.8500 / 0.4750` | `0.6875` (`0.125 / 0.875`) | `0.2000 / 0.2708` | `0.5833` (`0.5833`) | `0.5000 / 0.2138` |
| **Track 2 (Frozen Final)** | `0.003264%` (`11.8885`) | `frozen_test` | **`0.8347` (`0.8391`)** | `0.8000 / 0.3500` (`SR 0.85 / 0.375`) | **`0.7292`** (`0.375 / 0.875`) | `0.2375 / 0.1667` | `0.5833` (`0.5833`) | `0.5000 / 0.2135` |

### 2. Phase 4 `Q8_0` GGUF Runtime Parity & Latency (`select_dev`, n=200)

- **Artifact Size**: `794.97 MiB` (`833,590,336` bytes, **< 1 GiB**) | **BOS-free Tokenizer Parity**: `exact_match = True` (`2072` tokens)
- **PyTorch `bf16` vs `llama-server` `Q8_0` Parity**: `p_block` MAE `0.0549` (Pearson `r = 0.9695`), **Total Decision Agreement `96.0%`** (unquantized `bf16.gguf` 20-item `real_cf` diagnostic MAE `0.01691` vs `Q8_0` `0.05118`, confirming pure 8-bit quantization variance)
- **Standalone `Q8_0` on `select_dev`**: `real_cf` AUROC `0.8431` (SR `0.8719`), `consent_5p` Acc **`0.7708`** (`P2=0.375`, `P3=1.000`, `P5a=0.50`, `P5b=0.875`), Flip `0.1042`
- **Inference Latency (`CUDA sm_86`, `-ngl all`)**: **1-Pass P50 `71.73 ms`** (P95 `328.44 ms`) | **E2E Boundary Second-Reading P50 `88.77 ms`** (P95 `336.19 ms`)

## Quick Reproduction (`llama.cpp` `Q8_0` Runtime)

```bash
# 1. Export Track 2 weights into standard HF safetensors directory
.venv/bin/python runtime/export_hf.py \
  --deltas checkpoints/track2_best_deltas.pt \
  --output-dir checkpoints/wsp-automode-0.8b-hf

# 2. Convert HF -> bf16 GGUF -> Q8_0 GGUF (< 1 GiB)
bash runtime/convert_q8.sh

# 3. Launch standard llama-server (port 8091, CUDA -ngl all)
bash runtime/launch_llama_server.sh checkpoints/wsp-automode-0.8b-Q8_0.gguf 8091

# 4. Run PyTorch bf16 vs Q8_0 parity, latency & AutoModeTurnGuard E2E suite
.venv/bin/python -m runtime.parity_check
```

See [`PROGRESS.md`](PROGRESS.md) for current experiment status, [`experiments/phase4_parity_report.md`](experiments/phase4_parity_report.md) for runtime parity details, and [`docs/phase0_design.md`](docs/phase0_design.md) for full Phase 0 measurements.
