# `decider-0.8b-automode` 研究開発進捗トラッカー

勾配降下法（SFT / LoRA）を使わず、**重み直接編集（Context-to-Weight 焼き込み・STEER2EDIT＋Null-Space 射影・因果検証付き Weight-Space Programming）** を主軸として、`Qwen/Qwen3.5-0.8B` 単体で動作する高精度・完全 2 値・`llama.cpp` (`Q8_0` GGUF) 互換の Auto-Mode Decider を構築・検証するプロジェクトの経過記録です。

---

## フェーズ別進捗チェックリスト

- [x] **Phase 0：現状調査・実機モデル構造確認・データ設計・GitHub 連携**（2026-11-23）
  - Python 3.12 仮想環境構築（`torch 2.15.0+cu132`, `transformers 5.24.0`, `bf16` 対応 GPU 確認）
  - 実機 GPU 確認: `NVIDIA GeForce RTX 3050 Ti Laptop GPU`（VRAM `3770.81 MB`）
  - [`scripts/inspect_qwen35.py`](scripts/inspect_qwen35.py) で `Qwen/Qwen3.5-0.8B` 全 24 層構造（`linear_attention` 18 層＋`full_attention` 6 層: `3,7,11,15,19,23`）、トークナイザ整合性（`exact_concat_match_A/B = True`）、4GB VRAM ベンチマーク（`AutoModelForCausalLM` + スロット限定 `lm_head` で 4096 tok が `1921.61 MB` / `1071.84 ms`）を実測
  - [`scripts/inspect_sessions.py`](scripts/inspect_sessions.py) でローカル実セッション 23 本（888 件の非 read-only アクションチェックポイント）を監査し、プロジェクト単位で完全分離した 3 分割案を策定
  - 設計書 [`docs/phase0_design.md`](docs/phase0_design.md) 作成
- [ ] **Phase 1**：`automode/view.py`・`automode/questions.py`・リークゼロ 3 分割データ（`frozen_test` SHA-256 凍結）・統一評価ハーネス・`diagnose.py` 実装 ＋ **Track 0（Zero-Shot / Oracle Prompt 上限 / 内部診断）**
- [ ] **Phase 2**：**Track 1（Context-to-Weight 焼き込み ＋ STEER2EDIT ＋ Null-Space 射影 ＋ 3 点因果サニティテスト）**
- [ ] **Phase 3**：**Track 2（複数軌跡 WSP 探索ループ ＋ 3 軸回帰ゲート ＋ Leave-One-Out アブレーション）** → 最後に 1 回のみ `frozen_test` 解禁
- [ ] **Phase 4**：`Q8_0` GGUF 変換・`llama-server` 構築・`bf16` vs `Q8_0` パリティ検証・`second_reading` / 差し戻しループ実測

---

## 実験トラック比較サマリー

> **規約**: 未実行の項目は必ず `未実行` と記載し、推測値・未計測値は一切記入しません。`frozen_test` は Phase 3 終了時まで一切参照・評価しません。

| Track | 手法概要 | `% modified params` | 対象 Split | 実ログ AUROC | 実ログ Recall@0.5 | 実ログ FPR@0.5 | 5 パターン (P2 Allow / P3 Block) | Flip Rate | 長文 50T 保持 | 汎用 `decider` Acc / ECE | `Q8_0` パリティ誤差 |
|---|---|---:|---|---|---|---|---|---|---|---|---|
| **Track 0a** | 未編集 Zero-Shot (Plain Layout) | 0.00% | `select_dev` | 未実行 | 未実行 | 未実行 | 未実行 | 未実行 | 未実行 | 未実行 | 未実行 |
| **Track 0b** | 未編集 Oracle Prompt 上限 | 0.00% | `select_dev` | 未実行 | 未実行 | 未実行 | 未実行 | 未実行 | 未実行 | 未実行 | 未実行 |
| **Track 1** | Context-to-Weight + STEER2EDIT + Null-Space (Gradient-Free) | 未実行 | `select_dev` | 未実行 | 未実行 | 未実行 | 未実行 | 未実行 | 未実行 | 未実行 | 未実行 |
| **Track 2** | 複数軌跡 WSP + 3 軸ゲート + LOO アブレーション (Gradient-Free) | 未実行 | `select_dev` | 未実行 | 未実行 | 未実行 | 未実行 | 未実行 | 未実行 | 未実行 | 未実行 |
| **Track 2 (Final)** | Track 2 勝者パッチ（凍結評価 1 回限り） | 未実行 | `frozen_test` | 未実行 | 未実行 | 未実行 | 未実行 | 未実行 | 未実行 | 未実行 | 未実行 |
| **Track 3 (任意)** | 最小勾配ベースライン比較 | 未実行 | `frozen_test` | 未実行 | 未実行 | 未実行 | 未実行 | 未実行 | 未実行 | 未実行 | 未実行 |

---

## 実験アーティファクト索引

- [`docs/phase0_design.md`](docs/phase0_design.md) — Phase 0 アーキテクチャ・VRAM・3 分割データ・ディレクトリ設計書
- [`experiments/phase0_model_inspection.json`](experiments/phase0_model_inspection.json) — `Qwen/Qwen3.5-0.8B` 実機モジュール／トークナイザ／4GB VRAM 実測ログ
- [`experiments/phase0_sessions_summary.json`](experiments/phase0_sessions_summary.json) — 匿名化ローカルセッションログ監査（生テキスト・機密情報非含有）
