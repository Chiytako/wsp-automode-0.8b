# wsp-automode-0.8b

> **Note:** This is an independent, experimental **Weight-Space Programming (WSP)** and **continuous Auto-Mode classifier improvement** research repository built around [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B), compatible with the [`Mapika/decider`](https://github.com/Mapika/decider) plain state-first wire layout (not an official release of `Mapika/decider`).

## Released Public Models (Hugging Face)

| Model | Track | Method | Size (`Q8_0` GGUF) | Locked `frozen_test` (`real_cf` AUROC / `consent_5p` / `replay`) | Hugging Face |
|---|---|---|---:|---|---|
| **`wsp-automode-0.8b`** | Track 2 | 100% Gradient-Free WSP (`0.003264%`, 8 SwiGLU neurons on L15+L22) | `794.97 MiB` | `0.8347` / `0.7292` / `0.5000` | [`ChiTako/wsp-automode-0.8b`](https://huggingface.co/ChiTako/wsp-automode-0.8b) |
| **`automode-0.8b`** | Track 3 Final | Weight-Space Model Soup (`0.25 * r16-2ep + 0.75 * r32-3ep`, `Q8_0` parity `100%` on `frozen_test`) | `794.97 MiB` | **`1.0000`** / **`0.9375`** / **`0.7833`** | [`ChiTako/automode-0.8b`](https://huggingface.co/ChiTako/automode-0.8b) |

## Research Progression (Phases 0–5)

1. **Track 0 (Diagnostics & Oracle Upper Bound)**: Plain Zero-Shot baseline vs. Oracle Policy + Few-Shot upper bound, 24-layer Logit Lens, 4-bucket error taxonomy, and Plain-vs-Oracle slot activation geometry.
2. **Track 1 (Closed-Form Weight Editing)**: 100% gradient-free Context-to-Weight distillation, position-conditioned rank-1 SwiGLU gating (`L15`, 4 neurons), `AlphaEdit`-style replay null-space projection, and strict 3-point causal sanity checks (Dose-Response, Sign-Reversal, Orthogonal Random Control).
3. **Track 2 (Agent-Driven Weight-Space Programming)**: Multi-trajectory inspect -> hypothesize -> patch -> 3-point causal check -> 3-axis regression gate loop (`L15` Domain + `L22` Consent, `0.003264%`), Leave-One-Out ablation, and single locked `frozen_test` evaluation.
4. **Phase 4 (`llama.cpp` `Q8_0` GGUF Parity)**: Standard HF `safetensors` export, `< 1 GiB` `Q8_0` GGUF (`794.97 MiB`), BOS-free tokenizer parity, `second_reading` order debiasing, and `AutoModeTurnGuard` blocked-action feedback loop.
5. **Track 3 / Phase 5 (Beyond Weight Editing -> Weight-Space Model Soup)**: Decoupled scalar-chain-rule VRAM-safe (`<= 2.31 GB`) LoRA training across all 24 layers (`GatedDeltaNet` state gates + `FullAttn` + MLP) with order-equivariant BCE, within-family margin separation, and bidirectional replay KL (`683`-row zero-leakage train split), combined in weight space via **`0.25 * Track 3a + 0.75 * Track 3c`** Model Soup (`||ΔW||_F = 2.77189`), achieving **`99.0%`** (`select_dev`) and **`100.0%`** (`frozen_test`) PyTorch-vs-`Q8_0` agreement at **`69–71 ms` P50**.

## Non-Negotiables

- **Single 0.8B & Strictly Binary**: Always returns `Allow` or `Block` in a single forward pass with zero Stage-2 LLM fallback and zero harness approval bypass.
- **Plain Layout & Reasoning-Blind**: BOS-free, no chat template, `<|endoftext|>` right padding, 1-pass slot readout (`Answer: (`), no assistant prose or tool outputs in classifier history.
- **Zero-Leakage 3-Split Protocol**: Strict separation into `observe` / `select_dev` / `frozen_test` across project groups and scenario families, with `frozen_test` SHA-256 locked (`Phase 5` train data uses strictly `observe` projects `[chitako-drable, pi-loopedslm]`).
- **Privacy Hygiene**: No raw private session transcripts, local credentials, or model binaries are stored in this repository.

## Hardware & Measured Results

- **GPU**: NVIDIA GeForce RTX 3050 Ti Laptop GPU (`3770.81 MB` VRAM, `bf16` enabled; `torch 2.15.0+cu132`, `transformers 5.24.0`).
- **Backbone**: `Qwen/Qwen3.5-0.8B` — 24 layers (`full_attention_interval = 4`: 18 `Qwen3_5GatedDeltaNet` layers + 6 `Qwen3_5Attention` layers at indices `3, 7, 11, 15, 19, 23`).

### 1. Locked `frozen_test` (n=200, Unseen Projects — Evaluated Once Per Finalized Checkpoint)

| System | `% modified` (`\|\|ΔW\|\|_F` / Size) | Proxy | `real_cf` AUROC (SR) | `real_cf` Recall / FPR | `consent_5p` Acc (`P2 / P3`) | `P5a / P5b` (`Flip`) | `long_50t` (`recap`) | `decider_replay` Acc / ECE |
|---|---:|---:|---|---|---|---|---|---|
| **Track 0a (Zero-Shot)** | `0.0%` (`0.0`) | `-1.25390` | `0.6587` (`0.6353`) | `0.025 / 0.000` | `0.3958` (`0.875 / 0.125`) | `1.0 / 0.0` (`0.8958`) | `0.5833` (`0.6667`) | `0.5000 / 0.1989` |
| **Track 0c (`oracle_chat`)** | `0.0%` (`0.0`) | `-0.56906` | `0.6416` (`0.6600`) | `1.000 / 0.900` | `0.5208` (`0.375 / 0.625`) | `0.625 / 0.500` (`0.2292`) | `0.5000` (`0.5000`) | `0.5000 / 0.1989` |
| **Track 1 (`L15` WSP, Gradient-Free)** | `0.001633%` (`8.0109`) | `-0.02822` | `0.8213` (`0.8241`) | `0.850 / 0.475` | `0.6875` (`0.125 / 0.875`) | `0.375 / 0.750` (`0.2708`) | `0.5833` (`0.5833`) | `0.5000 / 0.2138` |
| **Track 2 (`L15+L22` WSP, Gradient-Free)** | `0.003264%` (`11.8885`) | `0.22845` | `0.8347` (`0.8391`) | `0.800 / 0.350` | `0.7292` (`0.375 / 0.875`) | `0.500 / 0.625` (`0.1667`) | `0.5833` (`0.5833`) | `0.5000 / 0.2135` |
| **Track 3 Final (Soup `λ=0.25`, `bf16`)** | `52.787731%` (`2.7719`) | `4.35396` | **`1.0000` (`1.0000`)** | **`1.000 / 0.025`** | **`0.9375`** (`0.750 / 1.000`) | **`1.0 / 1.0` (`0.0000`)** | **`0.9167`** (`0.8333`) | **`0.7833 / 0.0828`** |
| **Track 3 Final (`Q8_0` GGUF)** | `52.787731%` (`794.97 MiB`) | **`4.36611`** | **`1.0000` (`1.0000`)** | **`1.000 / 0.025`** | **`0.9375`** (`0.750 / 1.000`) | **`1.0 / 1.0` (`0.0000`)** | **`0.9167`** (`-`) | **`0.7833 / 0.0803`** |

### 2. Development & Ablation (`select_dev`, n=200)

| System | `% modified` (`\|\|ΔW\|\|_F` / Size) | Select Score | Proxy | `real_cf` AUROC (SR) | `real_cf` Rec / FPR | `consent_5p` Acc (`P2 / P3`) | Flip (`rc / c5`) | `long_50t` (`recap`) | `decider_replay` Acc / ECE |
|---|---:|---:|---:|---|---|---|---|---|---|
| **Track 2 (`L15+L22` WSP, `bf16`)** | `0.003264%` (`11.8885`) | `1.89028` | `0.56208` | `0.8731` (`0.8753`) | `0.625 / 0.175` | `0.7500` (`0.375 / 0.875`) | `0.2375 / 0.1042` | `0.5000` (`0.5833`) | `0.5333 / 0.1407` |
| **Track 2 (`Q8_0` GGUF)** | `0.003264%` (`794.97 MiB`) | - | `0.37521` | `0.8431` (`0.8719`) | `0.600 / 0.150` | `0.7708` (`0.375 / 1.000`) | `0.2000 / 0.1042` | `0.5000` (`-`) | `0.5333 / 0.1649` |
| **Track 3a (Base LoRA r=16 2ep)** | `52.28319%` (`2.8016`) | `6.20938` | `4.33433` | `1.0000` (`1.0000`) | `1.000 / 0.025` | `0.9167` (`1.000 / 0.875`) | `0.0000 / 0.0000` | `0.9167` (`0.9167`) | **`0.9167`** `/ 0.1350` |
| **Track 3b (WSP Warm-Start r=16 2ep)** | `51.528239%` (`12.1566`) | `5.18841` | `3.51166` | `0.9994` (`0.9994`) | `0.975 / 0.000` | `0.7708` (`0.750 / 1.000`) | `0.0250 / 0.1250` | `0.7500` (`0.7500`) | `0.8333 / 0.0812` |
| **Track 3c (Base LoRA r=32 3ep)** | `54.805714%` (`3.4821`) | `6.26425` | `4.39550` | `1.0000` (`1.0000`) | `1.000 / 0.050` | `0.9375` (`1.000 / 0.875`) | `0.0125 / 0.0000` | `0.9167` (`0.8333`) | `0.8833 / 0.1050` |
| **Track 3 Final (Soup `λ=0.25`, `bf16`)** | `52.787731%` (`2.7719`) | **`6.30240`** | `4.43365` | **`1.0000` (`1.0000`)** | **`1.000 / 0.000`** | **`0.9375`** (`1.000 / 0.875`) | **`0.0000 / 0.0000`** | `0.9167` (`0.8333`) | `0.8833 / 0.0801` |
| **Track 3 Final (`Q8_0` GGUF)** | `52.787731%` (`794.97 MiB`) | - | **`4.48369`** | **`1.0000` (`1.0000`)** | **`1.000 / 0.000`** | **`0.9375`** (`1.000 / 0.875`) | **`0.0000 / 0.0000`** | **`1.0000`** (`-`) | **`0.9000 / 0.0733`** |

### 3. `llama.cpp` `Q8_0` Runtime Parity & Latency (`CUDA sm_86`, `-ngl all`)

- **Track 2 (`wsp-automode-0.8b-Q8_0.gguf`, `794.97 MiB`)**: `select_dev` BOS-free exact match `True`, `p_block` MAE `0.0549` (Pearson `r = 0.9695`), Total Agreement `96.0%`, 1-Pass P50 `71.73 ms` / E2E P50 `88.77 ms`.
- **Track 3 Final (`automode-0.8b-track3-Q8_0.gguf`, `794.97 MiB`)**:
  - `select_dev` (n=200): BOS-free exact match `True`, `p_block` MAE **`0.00256`**, Pearson **`r = 0.99982`**, Total Agreement **`99.0%`**, 1-Pass P50 **`71.26 ms`** / E2E P50 **`71.26 ms`**.
  - `frozen_test` (n=200): BOS-free exact match `True`, `p_block` MAE **`0.00253`**, Pearson **`r = 0.99957`**, Total Agreement **`100.0%` (`1.0`)**, 1-Pass P50 **`69.06 ms`** / E2E P50 **`69.06 ms`**.

## Quick Reproduction (`llama.cpp` `Q8_0` Runtime)

```bash
# 1. Export finalized weights into standard HF safetensors directory
.venv/bin/python runtime/export_hf.py \
  --deltas checkpoints/track3_final_best_deltas.pt \
  --output-dir checkpoints/automode-0.8b-track3-hf

# 2. Convert HF -> bf16 GGUF -> Q8_0 GGUF (< 1 GiB)
bash runtime/convert_q8.sh \
  checkpoints/automode-0.8b-track3-hf \
  checkpoints/automode-0.8b-track3-bf16.gguf \
  checkpoints/automode-0.8b-track3-Q8_0.gguf

# 3. Launch standard llama-server (port 8091, CUDA -ngl all)
bash runtime/launch_llama_server.sh checkpoints/automode-0.8b-track3-Q8_0.gguf 8091

# 4. Run PyTorch bf16 vs Q8_0 parity, latency & AutoModeTurnGuard E2E suite
.venv/bin/python runtime/parity_check.py \
  --deltas checkpoints/track3_final_best_deltas.pt \
  --gguf checkpoints/automode-0.8b-track3-Q8_0.gguf \
  --split select_dev
```

See [`PROGRESS.md`](PROGRESS.md) for the complete Phases 0–5 log and artifact index, [`experiments/phase5_frozen_comparison.md`](experiments/phase5_frozen_comparison.md) for the full frozen comparison, and [`docs/phase0_design.md`](docs/phase0_design.md) for architecture details.
