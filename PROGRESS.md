# `wsp-automode-0.8b` 研究開発進捗トラッカー

> **リポジトリ**: [`Chiytako/wsp-automode-0.8b`](https://github.com/Chiytako/wsp-automode-0.8b)（`Qwen/Qwen3.5-0.8B` 上の無勾配 Weight-Space Programming 実験実装であり、`Mapika/decider` や `Chiytako/decider-auto` の公式リリースではありません）

勾配降下法（SFT / LoRA）を使わず、**重み直接編集（Context-to-Weight 焼き込み・STEER2EDIT＋Null-Space 射影・因果検証付き Weight-Space Programming）** を主軸として、`Qwen/Qwen3.5-0.8B` 単体で動作する高精度・完全 2 値・`llama.cpp` (`Q8_0` GGUF) 互換の Auto-Mode Decider を構築・検証するプロジェクトの経過記録です。

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
- [ ] **Phase 3**：**Track 2（複数軌跡 WSP 探索ループ ＋ 3 軸回帰ゲート ＋ Leave-One-Out アブレーション）** → 最後に 1 回のみ `frozen_test` 解禁
- [ ] **Phase 4**：`Q8_0` GGUF 変換・`llama-server` 構築・`bf16` vs `Q8_0` パリティ検証・`second_reading` / 差し戻しループ実測

---

## 実験トラック比較サマリー

> **規約**: 未実行の項目は必ず `未実行` と記載し、推測値・未計測値は一切記入しません。`frozen_test` は Phase 3 終了時まで一切参照・評価しません。

| Track | 手法概要 | `% modified params` | 対象 Split | 実ログ AUROC | 実ログ Recall@0.5 | 実ログ FPR@0.5 | 5 パターン (P2 Allow / P3 Block) | Flip Rate | 長文 50T 保持 | 汎用 `decider` Acc / ECE | `Q8_0` パリティ誤差 |
|---|---|---:|---|---|---|---|---|---|---|---|---|
| **Track 0a** | 未編集 Zero-Shot (Plain Layout) | 0.00% | `select_dev` | 0.5697 | 0.0500 | 0.0000 | Acc 0.3333 (`1.00 / 0.00`) | 0.9625 (rc) / 1.00 (c5) | 0.5833 (recap ON: 0.5833) | 0.5333 / 0.1687 | 未実行 |
| **Track 0b** | 未編集 Oracle Prompt (Plain Context) | 0.00% | `select_dev` | 0.6016 | 0.0000 | 0.0000 | Acc 0.3333 (`1.00 / 0.00`) | 1.0000 | 0.5000 (recap ON: 0.5000) | 0.5333 / 0.1687 | 未実行 |
| **Track 0c** | 未編集 Oracle Prompt (`oracle_chat` 上限) | 0.00% | `select_dev` | 0.6500 (`SR 0.7394`) | 0.9000 | 0.7250 | Acc 0.6250 (`0.375 / 0.875`) | 0.6125 (rc) / 0.6667 (c5) | 0.5833 (recap ON: 0.5000) | 0.5333 / 0.1687 | 未実行 |
| **Track 1** | 校正済 SwiGLU 順序ゲート (`L15`・4 neuron) + Replay 強化 Null-Space (Gradient-Free) | 0.001633% (`12,283`) | `select_dev` | **0.8512 (`SR 0.8644`)** | **0.7000 (`SR 0.8000`)** | 0.2000 | Acc **0.7292** (`0.125 / 1.000`) | **0.1500** (rc) / **0.2708** (c5) | 0.5000 (recap ON: 0.5833) | **0.5500 / 0.1366** | 未実行 |
| **Track 2** | 複数軌跡 WSP + 3 軸ゲート + LOO アブレーション (Gradient-Free) | 未実行 | `select_dev` | 未実行 | 未実行 | 未実行 | 未実行 | 未実行 | 未実行 | 未実行 | 未実行 |
| **Track 2 (Final)** | Track 2 勝者パッチ（凍結評価 1 回限り） | 未実行 | `frozen_test` | 未実行 | 未実行 | 未実行 | 未実行 | 未実行 | 未実行 | 未実行 | 未実行 |
| **Track 3 (任意)** | 最小勾配ベースライン比較 | 未実行 | `frozen_test` | 未実行 | 未実行 | 未実行 | 未実行 | 未実行 | 未実行 | 未実行 | 未実行 |

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
