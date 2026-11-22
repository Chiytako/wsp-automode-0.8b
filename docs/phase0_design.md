# Phase 0 設計書: `decider-0.8b-automode`（重み直接編集型 単一モデル Auto-Mode Decider）

作成日: 2026-11-23  
対象ベースモデル: `Qwen/Qwen3.5-0.8B`  
検証スクリプト: [`scripts/inspect_qwen35.py`](../scripts/inspect_qwen35.py), [`scripts/inspect_sessions.py`](../scripts/inspect_sessions.py)  
実測データ: [`experiments/phase0_model_inspection.json`](../experiments/phase0_model_inspection.json), [`experiments/phase0_sessions_summary.json`](../experiments/phase0_sessions_summary.json)

---

## 1. 実機計算環境と 4GB VRAM 運用設計

実機検証（`torch 2.15.0+cu132`, `transformers 5.24.0`）により確認された計算環境は以下の通りである。

| 項目 | 実測値 |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Ti Laptop GPU（報告総 VRAM `3770.81 MB` / `3.68 GiB`, `bf16` 対応） |
| Host RAM / CPU | `62 GiB` RAM / `20` 論理コア |
| 参照実装カーネル | `causal_conv1d` / `flash-linear-attention` 未導入のため PyTorch リファレンス実装で動作 |

### ロード方式とシーケンス長別 VRAM・レイテンシ実測

初期計画では 16GB VRAM クラスを想定していたが、実機は 4GB クラス（`3770.81 MB`）であるため、全推論パスで **`Qwen3_5ForCausalLM`（`AutoModelForCausalLM`）＋バックボーン直接呼び出し＋スロット位置限定 `lm_head`** を採用する。

| モード / ロード方式 | ロード後 VRAM | `256` tok (actual 249) | `1024` tok (actual 1019) | `2048` tok (actual 2042) | `4096` tok (actual 4088) |
|---|---|---|---|---|---|
| `AutoModelForImageTextToText`（全位置 logits 生成） | `1656.09 MB` | `80.64 ms` / Peak `1900.70 MB` | `309.90 ms` / Peak `2386.75 MB` | `616.69 ms` / Peak `3238.75 MB` | **OOM**（`[1, 4088, 248320]` logits で `1.89 GiB` 要求失敗） |
| `AutoModelForCausalLM`（`m.model` ＋ スロット位置限定 `lm_head`） | `1459.10 MB` | `74.01 ms` / Peak `1498.80 MB` | `272.70 ms` / Peak `1580.60 MB` | `539.73 ms` / Peak `1692.28 MB` | **`1071.84 ms` / Peak `1921.61 MB`** |

**4GB VRAM 運用ルール（必須）**:
1. GPU 上にモデルコピーを複数置かない。元のベース重みや編集前スナップショットは 62 GiB ホスト RAM（CPU）上に保持する。
2. すべての順伝播は `torch.bfloat16`・`torch.inference_mode()`・`use_cache=False`・`batch_size=1` で実行し、探索ステップ間で `torch.cuda.empty_cache()` を呼ぶ。
3. ロジット読出は必ず `hidden = model.model(input_ids=..., use_cache=False).last_hidden_state` から回答スロット位置のみ `model.lm_head(hidden[:, slot_indices, :])` で計算し、全系列ロジット展開を禁止する。

---

## 2. `Qwen/Qwen3.5-0.8B` 実機モデル構造と編集ターゲットマップ

### 2.1 パラメータ内訳とチェックポイント構成

- HuggingFace スナップショット: `model.safetensors-00001-of-00001.safetensors`（全 `488` テンソル, dtype: `BF16` / `F32`）
- 総パラメータ数: **`873,438,784`**
  - テキストバックボーン（`model.language_model.*`）: **`772,845,888`**
  - 視覚エンコーダ（`model.visual.*`）: **`100,592,896`**
- `tie_word_embeddings`: **`true`**（`lm_head.weight` の独立テンソルは存在せず、`embed_tokens.weight` `[248320, 1024]` と共有）
- テキスト設定: `hidden_size = 1024`, `intermediate_size = 3584`, `num_hidden_layers = 24`, `vocab_size = 248320`, `max_position_embeddings = 262144`
  - Full Attention: `num_attention_heads = 8`, `num_key_value_heads = 2`, `head_dim = 256`
  - Gated DeltaNet: `linear_num_key_heads = 16`, `linear_num_value_heads = 16`, `linear_key_head_dim = 128`, `linear_value_head_dim = 128`, `linear_conv_kernel_dim = 4`

> **重みエクスポート整合性**: 実験中は VRAM 節約のため `AutoModelForCausalLM`（`model.layers.<idx>.*`、全 320 テンソル、`missing=0 / unexpected=0`）を使用するが、GGUF 変換用 HF チェックポイント書き出し時には元の読取専用スナップショットの `config.json`・トークナイザ・`model.visual.*` をそのまま複製し、編集済み `model.*` を `model.language_model.*` へマップして保存する。これにより `llama.cpp` の `convert_hf_to_gguf.py` に対し純正 `Qwen/Qwen3.5-0.8B` と 100% 同一構造を保つ。

### 2.2 全 24 層モジュール構成（実測テーブル）

`full_attention_interval = 4` の 3:1 ハイブリッド構造（18 層の `linear_attention` + 6 層の `full_attention`）。

| Layer | `layer_type` | PyTorch プレフィックス (`CausalLM`) | Attention 出力射影 | MLP `down_proj` | 補助検査パラメータ | 優先度 |
|---:|---|---|---|---|---|---|
| 0–2 | `linear_attention` | `model.layers.{0..2}.linear_attn` | `out_proj.weight` `[1024, 2048]` | `mlp.down_proj.weight` `[1024, 3584]` | `A_log` `[16]`, `dt_bias` `[16]`, `in_proj_{qkv,z,a,b}` | 初期特徴層（原則凍結） |
| 3 | `full_attention` | `model.layers.3.self_attn` | `o_proj.weight` `[1024, 2048]` | `mlp.down_proj.weight` `[1024, 3584]` | `q_proj` `[4096,1024]`, `k/v_proj` `[512,1024]` | 初期特徴層 |
| 4–6 | `linear_attention` | `model.layers.{4..6}.linear_attn` | `out_proj.weight` `[1024, 2048]` | `mlp.down_proj.weight` `[1024, 3584]` | `A_log` `[16]`, `dt_bias` `[16]` | 初期特徴層 |
| 7 | `full_attention` | `model.layers.7.self_attn` | `o_proj.weight` `[1024, 2048]` | `mlp.down_proj.weight` `[1024, 3584]` | `q/k/v_proj`, `q/k_norm` `[256]` | 補助（履歴集約） |
| 8–10 | `linear_attention` | `model.layers.{8..10}.linear_attn` | `out_proj.weight` `[1024, 2048]` | `mlp.down_proj.weight` `[1024, 3584]` | `A_log` `[16]`, `dt_bias` `[16]` | **最優先 A** / 補助 C |
| **11** | **`full_attention`** | `model.layers.11.self_attn` | **`o_proj.weight` `[1024, 2048]`** | `mlp.down_proj.weight` `[1024, 3584]` | `q_proj`, `k_proj`, `v_proj` | **最優先 A（中段 Full-Attn）** |
| 12–14 | `linear_attention` | `model.layers.{12..14}.linear_attn` | `out_proj.weight` `[1024, 2048]` | `mlp.down_proj.weight` `[1024, 3584]` | `A_log` `[16]`, `dt_bias` `[16]` | **最優先 A** / 補助 C |
| **15** | **`full_attention`** | `model.layers.15.self_attn` | **`o_proj.weight` `[1024, 2048]`** | **`mlp.down_proj.weight` `[1024, 3584]`** | `q_proj`, `k_proj`, `v_proj` | **最優先 A ＋ B** |
| 16–18 | `linear_attention` | `model.layers.{16..18}.linear_attn` | `out_proj.weight` `[1024, 2048]` | **`mlp.down_proj.weight` `[1024, 3584]`** | `A_log` `[16]`, `dt_bias` `[16]` | **最優先 A ＋ B** / 補助 C |
| **19** | **`full_attention`** | `model.layers.19.self_attn` | **`o_proj.weight` `[1024, 2048]`** | **`mlp.down_proj.weight` `[1024, 3584]`** | `q_proj`, `k_proj`, `v_proj` | **最優先 A ＋ B** |
| 20 | `linear_attention` | `model.layers.20.linear_attn` | `out_proj.weight` `[1024, 2048]` | **`mlp.down_proj.weight` `[1024, 3584]`** | `A_log` `[16]`, `dt_bias` `[16]` | **最優先 A ＋ B** |
| 21–22 | `linear_attention` | `model.layers.{21..22}.linear_attn` | `out_proj.weight` `[1024, 2048]` | `mlp.down_proj.weight` `[1024, 3584]` | `A_log` `[16]`, `dt_bias` `[16]` | **最優先 A** / 順序バイアス |
| 23 | `full_attention` | `model.layers.23.self_attn` | `o_proj.weight` `[1024, 2048]` | `mlp.down_proj.weight` `[1024, 3584]` | 最終層保護（Layer Freezing 対象候補） | 補助 C（順序/読出） |

---

## 3. トークナイザ検証・Plain Layout・State View 仕様

### 3.1 トークナイザ検証結果（実測）

- クラス: `Qwen3_5Tokenizer`（`vocab_size = 248044`, `len(tok) = 248077`）
- 特殊トークン: `bos_token = null`, `add_bos_token = false`, `eos_token = "<|im_end|>" (248046)`, `pad_token = "<|endoftext|>" (248044)`
- 選択肢トークン ID: `'A' = 32`, `'B' = 33`, `'C' = 34`, `'D' = 35`, `'E' = 36`, `'(' = 7`, `')' = 8`
- 読出整合性:
  - `"Answer 1: ("` → `[15666, 220, 16, 25, 318]`（`['Answer', ' ', '1', ':', ' (']`）
  - `"Answer 1: (A"` == `encode("Answer 1: (") + [32]`（**`exact_concat_match_A = true`**）
  - `"Answer 1: (B"` == `encode("Answer 1: (") + [33]`（**`exact_concat_match_B = true`**）
  - プレフィックス/サフィックス境界: `"Environment:\n- test\nProposed action:\nTool: Bash"` のトークン列は `\n` 境界でマージせず完全分割一致（**`single_pass_prefix_suffix_split_clean = true`**）
- **未編集モデルのゼロショット初期分布**（256 tok ダミー文脈、`(A) no` / `(B) yes`）: `p('A') = 0.8079`, `p('B') = 0.1591`（**合計 96.7% が `(A)/(B)` に集中**）。ベース時点で `Answer: (` スロットの選択形式自体は既に理解しており、課題は安全性・特例同意・順序中立性の境界抽出と焼き込みにあることが確認された。

### 3.2 `automode/view.py` と `automode/questions.py` のインタフェース設計

`Mapika/decider` の plain state-first フォーマットと `Chiytako/decider-auto` の Reasoning-Blind prefix/suffix 契約を継承し、以下を追加する:

1. **Plain State-First Layout（BOS なし・チャットテンプレートなし・右パディング `<|endoftext|>`）**:
   ```text
   Context:
   Environment:
   ...
   User messages (oldest first):
   ...
   [Optional] User instructions recap: ...
   Agent actions so far (oldest first; results are not shown):
   ...
   [Optional] Blocked action k: ... (harness-trusted denial log)
   Proposed action:
   Tool: ...

   Question: ...
   Options:
   (A) ...
   (B) ...
   Answer: (
   ```
2. **信頼済み `Blocked action k:` エントリ**: ツール結果やエージェント自由記述は引き続き除外（Reasoning-Blind）しつつ、ハーネス自身が発行したブロック履歴（操作種別・ターゲット・ブロック理由のみ）を `Agent actions so far` 系譜へ安全に記録し、直後のユーザー承認（「はい」「実行して」等）による再試行（パターン 5）をモデル単体で `Allow` 判定可能にする。
3. **明示的 2 値質問スキーマ（`Allow` / `Block`）**:
   - `Q_block` に固定の明示的 `criteria`（Allow 条件 = `no: ...`、Block 条件 = `yes: ...`）を記載。
   - `second_reading` オプション順反転 API（通常 `(A) no / (B) yes` と反転 `(A) yes / (B) no` を切り替えてロジット平均・Flip Rate 計測）。
   - `User instructions recap` ON/OFF フラグ（50 ターン超長セッションでの Gated DeltaNet 重み編集効果を独立評価するため、デフォルト OFF 実験と ON 比較を可能にする）。
4. **Track 0/1 用 `Oracle Prompt` フォーマッタ**: 詳細ルール全文＋5 パターン Few-shot 例を `Context:` 先頭（または専用セクション）に付与し、同一評価質問スロットで未編集モデルの潜在能力上限および Context-to-Weight 蒸留の Teacher Activation を生成する。

---

## 4. 厳密 3 分割データ設計（リークゼロ・SHA-256 凍結）

[`scripts/inspect_sessions.py`](../scripts/inspect_sessions.py) により、ローカル実セッション 23 本（Claude Code 14 本、Pi 8 本、OMP 1 本）から **888 件の非 read-only アクションチェックポイント**（全 988 ツール呼出から read-only 91 件・対話/メタ 9 件を除外）を確認した。プロジェクト単位で物理的に重なりをなくした 3 分割は以下の通りとする:

| Split | 割当プロジェクト群（実ログ） | セッション数 | 実チェックポイント数 | 構成要素（Phase 1 で構築） | 用途 |
|---|---|---:|---:|---|---|
| **`observe`** | `chitako-drable` (252), `pi-loopedslm` (160), `pi-home-sit` (0) | 13 | **412** | 実ログ反実仮想ペア ＋ 特例許可 5 パターン群 A ＋ 長文 50 ターン群 A ＋ 汎用 `decider` Replay A | Oracle vs Plain 活性化差分抽出・方向ベクトル算出・Null-Space 共分散構築 |
| **`select_dev`** | `chitako-drive` (240), `omp-tmp` (118) | 3 | **358** | 実ログ反実仮想ペア ＋ 特例許可 5 パターン群 B ＋ 長文 50 ターン群 B ＋ 汎用 `decider` Replay B | Track 1 ハイパラ・Track 2 WSP 探索・3 点因果チェック・3 軸回帰ゲート |
| **`frozen_test`** | `carla-ja` (57), `home-sit` (61) | 7 | **118** | 実ログ反実仮想ペア ＋ 特例許可 5 パターン群 C ＋ 長文 50 ターン群 C ＋ 汎用 `decider` Replay C | **Phase 3 完了後に 1 回だけ評価**（SHA-256 でハッシュ凍結・閲覧禁止） |

- **反実仮想（Counterfactual）構成**: 実ログ中の安全な操作チェックポイント（`Allow`）に加え、同一履歴末尾の `Proposed action` のみを危険操作（破壊的 `rm -rf`、秘密鍵探索・外部送信、`git push --force`、セキュリティ無効化等）へ差し替えた `Block` 対照ペアを作成し、実ログのドメインショートカット（「実ログは常に Allow」という偽相関）を破壊・計測する。
- **5 パターン特例許可 Minimal Pairs**: シナリオ族（ベースシナリオのテンプレート／トピック）自体を `observe` / `select_dev` / `frozen_test` で完全に分離し、全分割に 5 パターンすべてを含める。
- **50 ターン超長セッション保持テスト**: 冒頭ターンの明示的ターゲット許可の後に 50+ ターンの無関係な開発アクションが続く長セッションで、末尾の許可対象操作を正しく `Allow`（および非許可ターゲットを `Block`）できるか検証する。
- **トークン上限設計**: `automode/view.py` の `DEFAULT_BUDGET` は 4GB VRAM でも安定・高速に収まる **`2048`〜`2300` トークン** を標準とし、長文 50 ターン保持テストのみ `3500`〜`4096` トークンまで評価可能（4096 tok でも `1921.61 MB` VRAM で安全に動作）。

---

## 5. リポジトリ構成と Phase 1–4 ロードマップ

```text
decider-0.8b-automode/
├── pyproject.toml / .gitignore / PROGRESS.md
├── docs/
│   ├── phase0_design.md                 # 本設計書
│   └── references/                      # .gitignore 対象（参照資料キャッシュ）
├── automode/                            # Phase 1: 状態ビュー・質問スキーマ・プロンプト
│   ├── view.py
│   └── questions.py
├── data/                                # Phase 1: リークゼロ 3 分割データビルダー
│   ├── build_datasets.py                # observe / select_dev / frozen_test 生成 & ハッシュ固定
│   └── manifest.json                    # frozen_test SHA-256 を含む分割メタデータ
├── weight_edit/                         # Phase 1–3: 診断・因果検証・無勾配重み編集・WSP
│   ├── diagnose.py                      # Track 0: Logit Lens, Plain vs Oracle 差分, 4バケット誤判定
│   ├── causal_check.py                  # Phase 2: 3点因果サニティテスト & Leave-One-Out
│   ├── context_edit.py                  # Phase 2 (Track 1): Context-to-Weight + STEER2EDIT + Null-Space
│   └── wsp_runner.py                    # Phase 3 (Track 2): 可逆パッチ管理・3軸ゲート・複数軌跡・wsp_log.jsonl
├── eval/                                # Phase 1–4: 統一評価ハーネス & レポート生成
│   └── evaluate.py
├── runtime/                             # Phase 4: llama.cpp GGUF Q8_0 変換・パリティ検証・クライアント
│   ├── export_hf.py                     # model.* -> model.language_model.* 書き戻し
│   ├── convert_q8.sh
│   ├── client.py                        # /completion + second_reading + 差し戻しループ
│   └── parity_check.py                  # PyTorch bf16 vs llama-server Q8_0 誤差検証
├── scripts/
│   ├── inspect_qwen35.py                # Phase 0 完了
│   └── inspect_sessions.py              # Phase 0 完了
├── experiments/                         # Phase 0–4 計測ログ・WSP 履歴・比較レポート
└── tests/                               # GPU 不要の単体テスト（レイアウト・リーク検査・因果モック等）
```

### フェーズ進行順
- **Phase 0（完了）**: 環境構築、実機 `Qwen/Qwen3.5-0.8B` 検査、実セッション監査、GitHub 連携、設計確定。
- **Phase 1**: `automode/view.py`, `questions.py`, 3 分割データ（`frozen_test` 凍結）, `eval/`, `weight_edit/diagnose.py` 実装 ＋ **Track 0（Plain Zero-Shot / Oracle Prompt 上限 / 24 層診断）** 実行。
- **Phase 2**: **Track 1（Context-to-Weight 焼き込み ＋ STEER2EDIT ＋ AlphaEdit 型 Null-Space 射影 ＋ 3 点因果チェック）** 実行。
- **Phase 3**: **Track 2（複数軌跡 WSP 探索ループ ＋ 3 軸回帰ゲート ＋ Leave-One-Out アブレーション）** 実行後、最後に 1 回だけ `frozen_test` を解禁して成績確定。
- **Phase 4**: `Q8_0` GGUF 変換、`llama-server`（DeltaNet キャッシュ最適化設定）での `PyTorch bf16` vs `Q8_0` パリティ検証、`second_reading` と差し戻しループの実測評価。
