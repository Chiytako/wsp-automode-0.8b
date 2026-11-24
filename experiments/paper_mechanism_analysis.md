# Paper Phase A Mechanistic Analysis (`paper_mechanism_analysis`)

## 1. Multi-Bit Quantization Degradation Curve (`select_dev`, n=200)

| Model | Format | Size (`MiB`) | `p_block` MAE | Pearson `r` | Agreement (200) | `real_cf` AUROC | `consent_5p` Acc | `long_50t` Acc | `replay` Acc | Proxy | P50 (`ms`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `track2_wsp` | `Q8_0` | `794.97` | `0.0549` | `0.969498` | `0.96` | `0.8431` | `0.7708` | `0.5` | `0.5333` | `0.37521` | `71.71` |
| `track2_wsp` | `Q6_K` | `616.59` | `0.21408` | `0.878074` | `0.835` | `0.8181` | `0.6667` | `0.5` | `0.5667` | `0.10018` | `78.21` |
| `track2_wsp` | `Q5_K_M` | `567.84` | `0.25814` | `0.740936` | `0.775` | `0.7706` | `0.6667` | `0.5` | `0.5167` | `-0.37455` | `75.7` |
| `track2_wsp` | `Q4_K_M` | `521.95` | `0.22105` | `0.58167` | `0.765` | `0.7887` | `0.6875` | `0.5833` | `0.5667` | `-0.64725` | `74.04` |
| `track3_soup` | `Q8_0` | `794.97` | `0.00256` | `0.99982` | `0.99` | `1.0` | `0.9375` | `1.0` | `0.9` | `4.48369` | `72.29` |
| `track3_soup` | `Q6_K` | `616.59` | `0.00689` | `0.998374` | `0.99` | `1.0` | `0.9375` | `1.0` | `0.9` | `4.45436` | `77.4` |
| `track3_soup` | `Q5_K_M` | `567.84` | `0.01658` | `0.989804` | `0.985` | `1.0` | `0.9375` | `1.0` | `0.8833` | `4.4097` | `75.27` |
| `track3_soup` | `Q4_K_M` | `521.95` | `0.03268` | `0.974793` | `0.955` | `1.0` | `0.875` | `0.8333` | `0.8833` | `4.3092` | `75.64` |

## 2. General Language Modeling Preservation (`32` Passages)

| Model | Tokens | Mean NLL | Perplexity | Mean Next-Token `KL(Base || Model)` |
|---|---:|---:|---:|---:|
| `base_qwen35_0.8b` | `2026` | `3.15569` | `23.4693` | `0.0` |
| `mit_wsp_0.8b` | `2026` | `3.19263` | `24.3524` | `0.014648` |
| `mapika_decider_0.8b` | `2026` | `3.11334` | `22.496` | `0.272107` |
| `ours_track2_wsp` | `2026` | `3.15942` | `23.5569` | `0.004377` |
| `ours_track3_soup_bf16` | `2026` | `3.22559` | `25.1683` | `0.047377` |

### Track 2 Quiet-Neuron Selectivity (General Text vs. Auto-Mode `Answer: (` Slots)
- **L15 `[3447,291,1066,917]`**: General Mean `|act|=0.0879` (`p95=0.39541`) vs. Auto-Mode Slot Mean `|act|=0.19294` (`p95=0.74609`)
- **L22 `[3394,3581,3037,2593]`**: General Mean `|act|=0.21713` (`p95=0.63672`) vs. Auto-Mode Slot Mean `|act|=0.62942` (`p95=2.87734`)

## 3. Track 3b Warm-Start Off-Manifold Interference (`n=64` Probes)

| Model | `h_15` Cos vs Base | Rel L2 Drift | T2 Gate `|SiLU|` | T2 Gate `|SwiGLU|` | Peak Train Loss |
|---|---:|---:|---:|---:|---:|
| `base_qwen35_0.8b` | `1.0` | `0.0` | `1.7785` | `1.594` | `-` |
| `ours_track2_wsp` | `1.0` | `0.0` | `1.7785` | `1.594` | `-` |
| `track3a_base_lora` | `0.390546` | `1.123611` | `4.3763` | `24.796` | `5.0943` |
| `track3b_wsp_warmstart_lora` | `0.699767` | `0.770948` | `0.4661` | `0.554` | `45.2076` |
