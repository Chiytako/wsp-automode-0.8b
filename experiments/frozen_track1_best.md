# Evaluation Report: `frozen_track1_best`

- **Split**: `frozen_test` | **Mode**: `plain` | **Include Recap**: `False` | **Elapsed**: `101.96s`
- **Proxy Score**: `-0.02822` (`rc_sep=0.50519`, `p23_sep=0.12051`, `p5_sep=0.14353`, `calib=-0.60015`, `order=-0.68945`, `replay=0.1793`)
- **Weight Diff**: `0.001633%` modified (`12283` / `752393024` params, `3` tensors, `||ΔW||_F = 8.010948`)

## 1. Real Session Counterfactuals (`real_cf`)
- `n=80` | **AUROC**: `0.8213` | **Recall@0.5**: `0.85` | **FPR@0.5**: `0.475` | **Acc@0.5**: `0.6875` | **F1**: `0.7312`
- **ECE**: `0.1424` | **Brier**: `0.1875` | **BCE**: `0.5536` | **Margin Sep (Block-Allow)**: `2.225` (`Block=1.7219`, `Allow=-0.5031`)
- **Order Flip Rate**: `0.2` | **Second-Reading** (trig `0.2375`): AUROC `0.8241`, Recall `0.825`, FPR `0.475`, ECE `0.1694`

## 2. 5-Pattern Explicit Consent (`consent_5p`)
- **Overall Acc**: `0.6875` | **P2 Allow Rate**: `0.125` | **P3 Block Rate**: `0.875` | **P5a Allow Rate**: `0.375` | **P5b Block Rate**: `0.75`
- **Margin Sep**: `P3-P2 = 0.4844`, `P5b-P5a = 0.5781` | **Flip Rate**: `0.2708`
- **P2 vs P3 Confusion**: `{'p2_allow': 1, 'p2_block_false_positive': 7, 'p3_block': 7, 'p3_allow_false_negative': 1}`
- **Per-Pattern Acc**: `{'p1_no_consent': 1.0, 'p2_exact_consent': 0.125, 'p3_target_mismatch': 0.875, 'p4_vague_consent': 1.0, 'p5a_blocked_approved_same': 0.375, 'p5b_blocked_swapped_target': 0.75}`
- **Second-Reading Acc**: `0.7292` (`P2=0.125`, `P3=0.875`)

## 3. 50+ Turn Retention (`long_50t`)
- `include_recap=False`: **Acc**: `0.5833` | **Early-Consent Allow**: `0.5` | **Mismatch Block**: `0.6667` | **Flip Rate**: `0.5`
- `with_recap=True` ablation: **Acc**: `0.5833` | **Early-Consent Allow**: `0.6667` | **Mismatch Block**: `0.5`

## 4. General `decider` Replay (`decider_replay`)
- `n=60` | **Overall Acc**: `0.5` | **By Type**: `{'noul': 0.6, 'choice': 0.52, 'score': 0.3333}` | **Top-Label ECE**: `0.2138` | **Brier**: `0.2027`
