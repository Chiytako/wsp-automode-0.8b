# `wsp-automode-0.8b` 研究開発進捗トラッカー

> **リポジトリ**: [`Chiytako/wsp-automode-0.8b`](https://github.com/Chiytako/wsp-automode-0.8b)（[`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) 上の Auto-Mode Decider 研究実験実装であり、`Mapika/decider` の公式リリースではありません）
> **公開 Hugging Face モデル**: Track 2 無勾配 WSP [`ChiTako/wsp-automode-0.8b`](https://huggingface.co/ChiTako/wsp-automode-0.8b) ｜ Track 3 Final Weight-Space Soup [`ChiTako/automode-0.8b`](https://huggingface.co/ChiTako/automode-0.8b)

Phase 1〜4 では勾配降下法（SFT / LoRA）を使わず **重み直接編集（Context-to-Weight 焼き込み・STEER2EDIT＋Null-Space 射影・因果検証付き Weight-Space Programming）** に徹し、Phase 5 では勾配最適化と重み空間統合（**順序同変 + 対照マージン + Bidirectional Replay KL LoRA + Weight-Space Model Soup**）を組み合わせて `Qwen/Qwen3.5-0.8B` 単体での安全性・汎用 `decider` 性能および `Q8_0` 量子化耐性の極大化を達成・公開しました。

---

## フェーズ別進捗チェックリスト

- [x] **Phase 0：現状調査・実機モデル構造確認・データ設計・GitHub 連携**（2026-11-23）
  - Python 3.12 仮想環境構築（`torch 2.15.0+cu132`, `transformers 5.24.0`, `bf16` 対応 GPU 確認）
  - 実機 GPU 確認: `NVIDIA GeForce RTX 3050 Ti Laptop GPU`（VRAM `3770.81 MB`）
  - [`scripts/inspect_qwen35.py`](scripts/inspect_qwen35.py) で `Qwen/Qwen3.5-0.8B` 全 24 層構造（`linear_attention` 18 層＋`full_attention` 6 層: `3,7,11,15,19,23`）、トークナイザ整合性（`exact_concat_match_A/B = True`）、4GB VRAM ベンチマーク（`AutoModelForCausalLM` + スロット限定 `lm_head` で 4096 tok が `1921.61 MB` / `1071.84 ms`）を実測
  - [`scripts/inspect_sessions.py`](scripts/inspect_sessions.py) でローカル実セッション 23 本（888 件の非 read-only アクションチェックポイント）を監査し、プロジェクト単位で完全分離した 3 分割案を策定
  - 設計書 [`docs/phase0_design.md`](docs/phase0_design.md) 作成
- [x] **Phase 1**：`automode/view.py`・`automode/questions.py`・リークゼロ 3 分割データ（`frozen_test` SHA-256 凍結）・統一評価ハーネス・`diagnose.py` 実装 ＋ **Track 0（Zero-Shot / Oracle Prompt 上限 / 内部診断）**（2026-11-23）
  - 3 分割データ生成 (`observe` 220 件 / `select_dev` 200 件 / `frozen_test` 200 件、SHA-256 ロック済)、単体テスト 5/5 通過
  - Track 0a (Plain Zero-Shot) と Track 0b (Plain-wrapped Oracle Prompt) を `select_dev` 上で計測: `(A)` バイアスが強く Recall@0.5 が `0.05` / `0.0`、Order Flip Rate が `0.9625` / `1.0` となることを特定
  - 24 層診断 ([`experiments/track0_diagnostics.md`](experiments/track0_diagnostics.md)): 中間〜終盤層（L15/L19 Full-Attn, L22）に `P3-P2` 分離信号（L22 で `+0.360`）と L4/L22 の線形分離（`real_cf` 2-fold probe AUROC `0.760` / `0.670`）が出現する一方、L22–L23 で打ち消される現象・Oracle 形式ギャップを確認
- [x] **Phase 2**：**Track 1（Context-to-Weight ＋ STEER2EDIT ＋ 対照プローブ／解析ゲイン校正付き SwiGLU ゲート ＋ Null-Space 射影 ＋ 3 点因果サニティテスト）**（2026-11-23）
  - [`experiments/track0c_oracle_chat_dev.md`](experiments/track0c_oracle_chat_dev.md) で Chat-template Oracle 形式上限を検証（`real_cf` Second-Reading AUROC `0.7394`, `consent_5p` Acc `0.6250`, Proxy `-0.39148`）
  - [`weight_edit/causal_check.py`](weight_edit/causal_check.py) および [`weight_edit/context_edit.py`](weight_edit/context_edit.py) を実装（`.backward()` / `optimizer.step()` ゼロの完全 Gradient-Free、Replay 強化 AlphaEdit Null-Space、層別ロジット伝達ゲイン $G_\ell$ 解析校正、ファミリー内対照＋Joint Ridge、Layer Freezing スイープ $l_f \in \{8,4,3,2,1,0\}$、厳格な 3 点因果テスト＋Leave-One-Out アブレーション）、単体テスト [`tests/test_phase2_weight_edit.py`](tests/test_phase2_weight_edit.py) 含む全 **11/11** テスト通過
  - `step1b_swiglu_order_gate_L15` が厳格な 3 点因果サニティテストを全通過（Spearman=`1.0`, Sign-Reversal `-1.205 -> -5.895`, Orthogonal Random Max Gain `0.418 < 1.429`, Replay 維持）。わずか **`0.001633%`（12,283 パラメータ / 3 テンソル、`||ΔW||_F = 8.0109`）** の書き換えで `select_dev` Proxy Score が `-1.12343` → **`+0.40678`**（`real_cf` AUROC `0.5697 -> 0.8512` / SR `0.8644`, Recall@0.5 `0.05 -> 0.70` / SR `0.80`, FPR `0.20`, Order Flip `0.9625 -> 0.1500`；`consent_5p` Acc `0.3333 -> 0.7292`, `P3-P2` Margin Sep `+0.0781 -> +0.4688`；汎用 `decider_replay` Acc `0.5333 -> 0.5500`・ECE `0.1687 -> 0.1366`）へ大幅向上
- [x] **Phase 3**：**Track 2（複数軌跡 WSP 探索ループ ＋ 3 軸回帰ゲート ＋ Leave-One-Out アブレーション）** → 最後に 1 回のみ `frozen_test` 解禁（2026-11-23）
  - [`weight_edit/causal_check.py`](weight_edit/causal_check.py) に `base_zero_deltas` ゼロアーティファクト線形スケーリングを実装（旧 SwiGLU 列残差による `scale=0.25` 歪みを解消）＋ [`weight_edit/wsp_runner.py`](weight_edit/wsp_runner.py) に閉形式残差ブースティング・Priority A Attention Null-Space・Priority C DeltaNet `A_log`/`dt_bias` 減衰手術・3 軸回帰ゲートを実装（[`tests/test_phase3_wsp.py`](tests/test_phase3_wsp.py) 含め単体テスト **15/15** 通過）
  - 3 軌跡比較 ([`experiments/track2_trajectories_summary.md`](experiments/track2_trajectories_summary.md)): 3 軸回帰ゲートが `P3` 劣化案（`1.0 -> 0.75`: `traj1_iter1`, `traj2_iter2`）を自動 REVERT し、DeltaNet `A_log`/`dt_bias` 減衰手術（`gain=-0.00305`）も誠実に REVERT。勝者はゼロからの独立軌跡 `traj3_from_scratch_multilayer`（`L15` ドメイン専門 ＋ `L22` 同意専門、`0.003264%` / `24,561` パラメータ / 6 テンソル、`select_dev` Proxy `+0.56208`）
  - LOO アブレーション ([`experiments/track2_loo_ablation.json`](experiments/track2_loo_ablation.json)): `L15` 削除で Proxy `-2.15962`、`L22` 削除で `-0.22161` となり、2 ステップとも必須を確認
  - [`eval/eval_frozen_once.py`](eval/eval_frozen_once.py) で `frozen_test`（200 件・初見プロジェクト群 `carla-ja` + `home-sit`）を SHA-256 検証の上 **1 回限り**解禁評価（[`experiments/frozen_test_comparison.md`](experiments/frozen_test_comparison.md)）：`Track 2 Final` は Proxy `-1.25390` (0a) / `-0.56906` (0c) → **`+0.22845`**（`real_cf` AUROC `0.8347` / SR `0.8391`、Recall `0.80` / FPR `0.35`、`consent_5p` Acc `0.7292`・`P2=0.375 / P3=0.875`・Flip `0.1667`、`decider_replay` Acc `0.50`）を達成（評価後の再調整ゼロ）
- [x] **Phase 4**：`Q8_0` GGUF 変換・`llama-server` 構築・`bf16` vs `Q8_0` パリティ検証・`second_reading` / 差し戻しループ実測（2026-11-23）
  - [`runtime/export_hf.py`](runtime/export_hf.py) で Track 2 勝者パッチを HF `bf16` モデルへ書き出し（全 488 テンソル中 6 テンソル変更、`24,562` 要素、`||ΔW||_F = 11.88819`、`model.visual.*` 無変更）
  - [`runtime/convert_q8.sh`](runtime/convert_q8.sh) & [`runtime/launch_llama_server.sh`](runtime/launch_llama_server.sh) 実装（`llama.cpp` CUDA `sm_86` ビルド、`wsp-automode-0.8b-Q8_0.gguf` = `794.97 MiB < 1 GiB`、BOS なしトークナイザ完全一致 `exact_match = True`）
  - [`runtime/client.py`](runtime/client.py) & [`runtime/parity_check.py`](runtime/parity_check.py)：PyTorch `bf16` vs `llama-server` `Q8_0`（`select_dev` 200 件）で `p_block MAE = 0.0549`（Pearson `r = 0.9695`）、全体決定一致率 **`96.0%`**（`bf16` GGUF 20 件診断 `0.01691` により純粋な 8bit 量子化分散であることを確認）；`Q8_0` 単独で `real_cf` AUROC `0.8431`（SR `0.8719`）、`consent_5p` Acc `0.7708`；1-pass レイテンシ P50 **`71.73 ms`** / E2E P50 **`88.77 ms`**；単体テスト [`tests/test_phase4_runtime.py`](tests/test_phase4_runtime.py) **18/18** 通過
- [x] **Phase 5（Track 3+）**：リークゼロ 683 件学習データ構築・24 層フル空間 LoRA（`GatedDeltaNet` 減衰＋順序同変＋対照＋KL）・Weight-Space Model Soup・`Q8_0` パリティ・単回 `frozen_test` 評価・Hugging Face 公開（2026-11-23）
  - [`data/build_phase5_train.py`](data/build_phase5_train.py) & [`data/phase5_manifest.json`](data/phase5_manifest.json)：`observe` プロジェクト群（`chitako-drable`, `pi-loopedslm`）＋新規 180 ファミリー合成で 683 件を監査生成（`zero_leakage_vs_select_dev_and_frozen_test = true`, `base_splits_sha256_intact = true`）
  - [`train/lora_model.py`](train/lora_model.py)・[`train/losses.py`](train/losses.py)・[`train/train_track3.py`](train/train_track3.py)：全 24 層 186 線形層への決定的 `dropout=0.0` LoRA ＋ float32 `A_log`/`dt_bias` ＋ スカラー連鎖律分離ファミリー勾配を実装（RTX 3050 Ti でピーク **`2.14–2.31 GB` VRAM**）。Base LoRA (`Track 3a` r=16 2ep `select_score=6.20938` / `Track 3c` r=32 3ep `6.26425`) が WSP Warm-Start (`Track 3b` `5.18841`, 高ノルムスパースゲートと上流表現シフトの干渉) を大きく上回ることを確認
  - [`train/soup_and_calibrate.py`](train/soup_and_calibrate.py)：重み空間補間 **`Soup lam=0.25` (`0.25 * Track 3a + 0.75 * Track 3c`, `T=1.0`, `52.787731%`, `||ΔW||_F = 2.77189`)** が `select_dev` 最高 **`Select Score = 6.30240`**（`Proxy = 4.43365`, `real_cf 80/80` **AUROC `1.0` / Rec `1.0` / FPR `0.0`**, `consent_5p 0.9375`, `Flip 0.0 / 0.0`, `long_50t 0.9167` / SR `1.0`, `decider_replay 0.8833 / ECE 0.0801`）を達成
  - `automode-0.8b-track3-Q8_0.gguf`（`794.97 MiB < 1 GiB`）：滑らかな重み分散 (`||ΔW||_F = 2.77`) により `Q8_0` 耐性が劇的に向上し、`select_dev` で PyTorch `bf16` と **`99.0%` 一致**（`p_block MAE = 0.00256`, `r = 0.99982`, P50 `71.26 ms`, `Q8_0` Proxy `4.48369`, `long_50t 1.0`, `replay 0.90 / ECE 0.0733`）を記録
  - 単回ロック済 `frozen_test`（200 件・初見プロジェクト群、`experiments/phase5_frozen_comparison.md`）：PyTorch `bf16` と `Q8_0` GGUF で **200/200 (`100.0%`) 完全決定一致**（`p_block MAE = 0.00253`, `r = 0.999569`, P50 **`69.06 ms`**）、**`real_cf` AUROC `1.0000`**（Recall `1.000` / FPR `0.025` / Flip `0.000`）、**`consent_5p` Acc `0.9375`**（`P2=0.75 / P3=1.0 / P5a=1.0 / P5b=1.0` / Flip `0.0`）、**`long_50t` `0.9167`**、**`decider_replay` `0.7833 / ECE 0.0803`**（Proxy `bf16 4.35396` / `Q8_0 4.36611`）を達成
  - [`runtime/upload_hf.py`](runtime/upload_hf.py)（[`experiments/hf_upload_summary.json`](experiments/hf_upload_summary.json)）：[`ChiTako/wsp-automode-0.8b`](https://huggingface.co/ChiTako/wsp-automode-0.8b)（Track 2）および [`ChiTako/automode-0.8b`](https://huggingface.co/ChiTako/automode-0.8b)（Track 3 Final）をともに Public 公開（`bf16 safetensors` + `< 1 GiB` `Q8_0` GGUF + Model Card、全 **25/25** `pytest` 通過）

---

## 実験トラック比較サマリー

> **規約**: Phase 3 の凍結レポート（[`experiments/frozen_test_comparison.md`](experiments/frozen_test_comparison.md) 等）は不変のまま保持し、Phase 5 の単回 `frozen_test` 評価結果は [`experiments/phase5_frozen_comparison.md`](experiments/phase5_frozen_comparison.md) へ独立記録しています。

| Track | 手法概要 | `% modified params` | 対象 Split | 実ログ AUROC | 実ログ Recall@0.5 | 実ログ FPR@0.5 | 5 パターン (P2 Allow / P3 Block) | Flip Rate | 長文 50T 保持 | 汎用 `decider` Acc / ECE | `Q8_0` パリティ誤差 |
|---|---|---:|---|---|---|---|---|---|---|---|---|
| **Track 0a** | 未編集 Zero-Shot (Plain Layout) | 0.00% | `select_dev` | 0.5697 | 0.0500 | 0.0000 | Acc 0.3333 (`1.00 / 0.00`) | 0.9625 (rc) / 1.00 (c5) | 0.5833 (recap ON: 0.5833) | 0.5333 / 0.1687 | 未実行 |
| **Track 0b** | 未編集 Oracle Prompt (Plain Context) | 0.00% | `select_dev` | 0.6016 | 0.0000 | 0.0000 | Acc 0.3333 (`1.00 / 0.00`) | 1.0000 | 0.5000 (recap ON: 0.5000) | 0.5333 / 0.1687 | 未実行 |
| **Track 0c** | 未編集 Oracle Prompt (`oracle_chat` 上限) | 0.00% | `select_dev` | 0.6500 (`SR 0.7394`) | 0.9000 | 0.7250 | Acc 0.6250 (`0.375 / 0.875`) | 0.6125 (rc) / 0.6667 (c5) | 0.5833 (recap ON: 0.5000) | 0.5333 / 0.1687 | 未実行 |
| **Track 1** | 校正済 SwiGLU 順序ゲート (`L15`・4 neuron) + Replay 強化 Null-Space (Gradient-Free) | 0.001633% (`12,283`) | `select_dev` | **0.8512 (`SR 0.8644`)** | **0.7000 (`SR 0.8000`)** | 0.2000 | Acc **0.7292** (`0.125 / 1.000`) | **0.1500** (rc) / **0.2708** (c5) | 0.5000 (recap ON: 0.5833) | **0.5500 / 0.1366** | 未実行 |
| **Track 2** | 複数軌跡 WSP + 3 軸ゲート + LOO (`L15` Domain + `L22` Consent, Gradient-Free) | 0.003264% (`24,561`) | `select_dev` | **0.8731 (`SR 0.8753`)** | 0.6250 (`SR 0.7000`) | **0.1750** | Acc **0.7500** (`0.375 / 0.875`) | 0.2375 (rc) / **0.1042** (c5) | 0.5000 (recap ON: 0.5833) | 0.5333 / 0.1407 | MAE 0.0549 (bf16 GGUF 0.0169) / 96.0% 一致 / P50 71.73 ms |
| **Track 2 (Q8_0 Runtime)** | `llama-server` `Q8_0` GGUF 単独評価 (`CUDA sm_86`) | 0.003264% (`794.97 MiB`) | `select_dev` | 0.8431 (`SR 0.8719`) | 0.6000 (`SR 0.6750`) | **0.1500** | Acc **0.7708** (`0.375 / 1.000`) | **0.2000** (rc) / **0.1042** (c5) | 0.5000 | 0.5333 / 0.1649 | 794.97 MiB / 96.0% 一致 / P50 71.73 ms |
| **Track 3a (LoRA r=16 2ep)** | 24 層全線形 + DeltaNet Gate LoRA (対照＋順序同変＋Replay KL) | 52.28319% (`\|\|ΔW\|\|_F=2.80`) | `select_dev` | **1.0000** (`SR 1.0000`) | **1.0000** | 0.0250 | Acc 0.9167 (`1.000 / 0.875`) | **0.0000** | **0.9167** (recap ON: 0.9167) | **0.9167 / 0.1350** | 未実行 |
| **Track 3b (WSP Warm-Start)** | Track 2 WSP (`L15+L22`) 初期化 ＋ LoRA r=16 2ep（負の干渉実験） | 51.528239% (`\|\|ΔW\|\|_F=12.16`) | `select_dev` | 0.9994 (`SR 0.9994`) | 0.9750 | **0.0000** | Acc 0.7708 (`0.750 / 1.000`) | 0.0250 / 0.1250 | 0.7500 (recap ON: 0.7500) | 0.8333 / 0.0812 | 未実行 |
| **Track 3c (LoRA r=32 3ep)** | 高ランク 24 層 LoRA (対照＋順序同変＋Replay KL) | 54.805714% (`\|\|ΔW\|\|_F=3.48`) | `select_dev` | **1.0000** (`SR 1.0000`) | **1.0000** | 0.0500 | Acc **0.9375** (`1.000 / 0.875`) | 0.0125 / **0.0000** | **0.9167** (recap ON: 0.8333) | 0.8833 / 0.1050 | 未実行 |
| **Track 3 Final (Soup bf16)** | Weight-Space Model Soup `0.25*3a + 0.75*3c` (`T=1.0`) | 52.787731% (`\|\|ΔW\|\|_F=2.77`) | `select_dev` | **1.0000** (`SR 1.0000`) | **1.0000** | **0.0000** | Acc **0.9375** (`1.000 / 0.875`) | **0.0000** | **0.9167** (`SR 1.0`) | 0.8833 / **0.0801** | MAE 0.00256 / **99.0% 一致** / P50 71.26 ms |
| **Track 3 Final (Q8_0 Runtime)** | `llama-server` `Q8_0` GGUF 単独評価 (`CUDA sm_86`) | 52.787731% (`794.97 MiB`) | `select_dev` | **1.0000** (`SR 1.0000`) | **1.0000** | **0.0000** | Acc **0.9375** (`1.000 / 0.875`) | **0.0000** | **1.0000** | **0.9000 / 0.0733** | 794.97 MiB / 99.0% 一致 / P50 71.26 ms |
| **Track 0a (Frozen)** | 未編集 Zero-Shot (Plain Layout) | 0.00% | `frozen_test` | 0.6587 (`SR 0.6353`) | 0.0250 (`SR 0.7500`) | 0.0000 | Acc 0.3958 (`0.875 / 0.125`) | 0.9875 (rc) / 0.8958 (c5) | 0.5833 (recap ON: 0.6667) | 0.5000 / 0.1989 | 未実行 |
| **Track 0c (Frozen)** | 未編集 Oracle Prompt (`oracle_chat` 上限) | 0.00% | `frozen_test` | 0.6416 (`SR 0.6600`) | 1.0000 (`SR 0.9250`) | 0.9000 | Acc 0.5208 (`0.375 / 0.625`) | 0.2750 (rc) / 0.2292 (c5) | 0.5000 (recap ON: 0.5000) | 0.5000 / 0.1989 | 未実行 |
| **Track 1 (Frozen)** | 校正済 SwiGLU 順序ゲート (`L15`, Gradient-Free) | 0.001633% (`12,283`) | `frozen_test` | 0.8213 (`SR 0.8241`) | **0.8500** (`SR 0.8250`) | 0.4750 | Acc 0.6875 (`0.125 / 0.875`) | **0.2000** (rc) / 0.2708 (c5) | 0.5833 (recap ON: 0.5833) | 0.5000 / 0.2138 | 未実行 |
| **Track 2 (Final)** | Track 2 勝者パッチ (`L15`+`L22` SwiGLU, 凍結評価 1 回限り) | 0.003264% (`24,561`) | `frozen_test` | **0.8347 (`SR 0.8391`)** | 0.8000 (`SR 0.8500`) | **0.3500** | Acc **0.7292** (`0.375 / 0.875`) | 0.2375 (rc) / **0.1667** (c5) | 0.5833 (recap ON: 0.5833) | 0.5000 / 0.2135 | 未実行 |
| **Track 3 Final (Frozen `bf16`)** | Weight-Space Model Soup `λ=0.25` (凍結評価 1 回限り) | 52.787731% (`\|\|ΔW\|\|_F=2.77`) | `frozen_test` | **1.0000 (`SR 1.0000`)** | **1.0000** | **0.0250** | Acc **0.9375** (`0.750 / 1.000`) | **0.0000** | **0.9167** (recap ON: 0.8333) | **0.7833 / 0.0828** | MAE 0.00253 / **100.0% 一致** / P50 69.06 ms |
| **Track 3 Final (Frozen `Q8_0`)** | `llama-server` `Q8_0` GGUF (`CUDA sm_86`, 凍結評価 1 回限り) | 52.787731% (`794.97 MiB`) | `frozen_test` | **1.0000 (`SR 1.0000`)** | **1.0000** | **0.0250** | Acc **0.9375** (`0.750 / 1.000`) | **0.0000** | **0.9167** | **0.7833 / 0.0803** | **794.97 MiB / 100.0% 一致** / P50 **69.06 ms** |

---

## 実験アーティファクト索引

- [`docs/phase0_design.md`](docs/phase0_design.md) — Phase 0 アーキテクチャ・VRAM・3 分割データ・ディレクトリ設計書
- [`experiments/phase0_model_inspection.json`](experiments/phase0_model_inspection.json) — `Qwen/Qwen3.5-0.8B` 実機モジュール／トークナイザ／4GB VRAM 実測ログ
- [`experiments/phase0_sessions_summary.json`](experiments/phase0_sessions_summary.json) — 匿名化ローカルセッションログ監査（生テキスト・機密情報非含有）
- [`data/manifest.json`](data/manifest.json) — 3 分割データセット件数・SHA-256 ロックマニフェスト
- [`experiments/track0a_zeroshot_dev.md`](experiments/track0a_zeroshot_dev.md) — Track 0a Plain Zero-Shot 評価レポート
- [`experiments/track0b_oracle_dev.md`](experiments/track0b_oracle_dev.md) — Track 0b Oracle Prompt 評価レポート
- [`experiments/track0_diagnostics.md`](experiments/track0_diagnostics.md) — Track 0 24 層 Logit Lens・4 バケット誤判定・Plain vs Oracle 診断レポート
- [`experiments/track0c_oracle_aligned_dev.md`](experiments/track0c_oracle_aligned_dev.md) — Track 0c Oracle Aligned Prompt 評価レポート
- [`experiments/track0c_oracle_chat_dev.md`](experiments/track0c_oracle_chat_dev.md) — Track 0c Chat-Template Oracle 形式上限レポート
- [`experiments/track1_causal_checks.md`](experiments/track1_causal_checks.md) — Track 1 全候補 3 点因果サニティチェック（Dose-Response / Sign-Reversal / Orthogonal Random Control）
- [`experiments/track1_layer_freezing_ablation.json`](experiments/track1_layer_freezing_ablation.json) — Track 1 末尾層凍結アブレーション（$l_f \in \{8,4,3,2,1,0\}$）＆ゲイン校正ログ
- [`experiments/track1_loo_ablation.json`](experiments/track1_loo_ablation.json) — Track 1 Leave-One-Out アブレーションログ
- [`experiments/track1_patch_spec.json`](experiments/track1_patch_spec.json) — Track 1 採用パッチ仕様（`step1b_swiglu_order_gate_L15`）
- [`experiments/track1_best_dev.md`](experiments/track1_best_dev.md) — Track 1 勝者モデル `select_dev` 全指標レポート
- [`experiments/wsp_log.jsonl`](experiments/wsp_log.jsonl) — Track 2 全 3 軌跡 WSP 探索ステップ（Inspect / 仮説 / 3 点因果テスト / 3 軸回帰ゲート / KEEP・REVERT 決定）ログ
- [`experiments/track2_trajectories_summary.md`](experiments/track2_trajectories_summary.md) — Track 2 3 軌跡フル `select_dev` 比較サマリー
- [`experiments/track2_loo_ablation.json`](experiments/track2_loo_ablation.json) — Track 2 Leave-One-Out アブレーションログ
- [`experiments/track2_patch_spec.json`](experiments/track2_patch_spec.json) — Track 2 最終勝者パッチ仕様（`L15` ドメイン専門 ＋ `L22` 同意専門）
- [`experiments/track2_best_dev.md`](experiments/track2_best_dev.md) — Track 2 勝者モデル `select_dev` 全指標レポート
- [`experiments/frozen_test_comparison.md`](experiments/frozen_test_comparison.md) — `frozen_test` 1 回限り解禁・全 4 系統（Track 0a / 0c / 1 / 2）比較レポート
- [`experiments/frozen_track0a_zeroshot.md`](experiments/frozen_track0a_zeroshot.md) — `frozen_test` Track 0a (Zero-Shot) 個別レポート
- [`experiments/frozen_track0c_oracle_chat.md`](experiments/frozen_track0c_oracle_chat.md) — `frozen_test` Track 0c (`oracle_chat`) 個別レポート
- [`experiments/frozen_track1_best.md`](experiments/frozen_track1_best.md) — `frozen_test` Track 1 個別レポート
- [`experiments/frozen_track2_best.md`](experiments/frozen_track2_best.md) — `frozen_test` Track 2 最終勝者個別レポート
- [`experiments/phase4_parity_report.md`](experiments/phase4_parity_report.md) — Phase 4 `Q8_0` GGUF パリティ・`bf16` GGUF 比較・レイテンシ・E2E TurnGuard レポート
- [`experiments/phase4_parity_report.json`](experiments/phase4_parity_report.json) — Phase 4 パリティ・単独評価・E2E スモーク計測 JSON
- [`data/phase5_manifest.json`](data/phase5_manifest.json) — Phase 5 学習データ (683 件・180 ファミリー) リークゼロ検証・SHA-256 マニフェスト
- [`experiments/track3a_base_lora_best_dev.md`](experiments/track3a_base_lora_best_dev.md) — Track 3a (Base LoRA r=16, 2ep) `select_dev` 評価レポート
- [`experiments/track3b_wsp_warmstart_lora_best_dev.md`](experiments/track3b_wsp_warmstart_lora_best_dev.md) — Track 3b (WSP Warm-Start LoRA r=16, 2ep) `select_dev` 評価・干渉実験レポート
- [`experiments/track3c_r32_3ep_lora_best_dev.md`](experiments/track3c_r32_3ep_lora_best_dev.md) — Track 3c (Base LoRA r=32, 3ep) `select_dev` 評価レポート
- [`experiments/track3_dev_comparison.md`](experiments/track3_dev_comparison.md) — Track 0–3c + Model Soup + RMSNorm 温度スイープ `select_dev` 比較レポート
- [`experiments/track3_final_best_dev.md`](experiments/track3_final_best_dev.md) — Track 3 Final (`Soup lam=0.25`, `T=1.0`) `select_dev` 全指標レポート
- [`experiments/track3_final_patch_spec.json`](experiments/track3_final_patch_spec.json) — Track 3 Final Weight-Space Soup マージパッチ仕様
- [`experiments/phase5_track3_parity_dev.md`](experiments/phase5_track3_parity_dev.md) — Phase 5 Track 3 Final `Q8_0` GGUF パリティ・レイテンシ (`select_dev`)
- [`experiments/phase5_track3_parity_frozen.md`](experiments/phase5_track3_parity_frozen.md) — Phase 5 Track 3 Final 単回ロック済 `frozen_test` パリティ・単独計測レポート
- [`experiments/phase5_frozen_comparison.md`](experiments/phase5_frozen_comparison.md) — Phase 5 単回 `frozen_test` 全系統（Track 0a / 0c / 1 / 2 / 3 `bf16` / 3 `Q8_0`）比較レポート
- [`experiments/hf_upload_summary.json`](experiments/hf_upload_summary.json) — Hugging Face Public 公開（`ChiTako/wsp-automode-0.8b` & `ChiTako/automode-0.8b`）検証サマリー
