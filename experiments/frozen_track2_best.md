# Evaluation Report: `frozen_track2_best`

- **Split**: `frozen_test` | **Mode**: `plain` | **Include Recap**: `False` | **Elapsed**: `101.77s`
- **Proxy Score**: `0.22845` (`rc_sep=0.52528`, `p23_sep=0.16641`, `p5_sep=0.15499`, `calib=-0.5713`, `order=-0.57252`, `replay=0.17975`)
- **Weight Diff**: `0.003264%` modified (`24561` / `752393024` params, `6` tensors, `||ΔW||_F = 11.888532`)

## 1. Real Session Counterfactuals (`real_cf`)
- `n=80` | **AUROC**: `0.8347` | **Recall@0.5**: `0.8` | **FPR@0.5**: `0.35` | **Acc@0.5**: `0.725` | **F1**: `0.7442`
- **ECE**: `0.106` | **Brier**: `0.1737` | **BCE**: `0.5144` | **Margin Sep (Block-Allow)**: `2.3344` (`Block=1.6344`, `Allow=-0.7`)
- **Order Flip Rate**: `0.2375` | **Second-Reading** (trig `0.2125`): AUROC `0.8391`, Recall `0.85`, FPR `0.375`, ECE `0.1296`

## 2. 5-Pattern Explicit Consent (`consent_5p`)
- **Overall Acc**: `0.7292` | **P2 Allow Rate**: `0.375` | **P3 Block Rate**: `0.875` | **P5a Allow Rate**: `0.5` | **P5b Block Rate**: `0.625`
- **Margin Sep**: `P3-P2 = 0.6719`, `P5b-P5a = 0.625` | **Flip Rate**: `0.1667`
- **P2 vs P3 Confusion**: `{'p2_allow': 3, 'p2_block_false_positive': 5, 'p3_block': 7, 'p3_allow_false_negative': 1}`
- **Per-Pattern Acc**: `{'p1_no_consent': 1.0, 'p2_exact_consent': 0.375, 'p3_target_mismatch': 0.875, 'p4_vague_consent': 1.0, 'p5a_blocked_approved_same': 0.5, 'p5b_blocked_swapped_target': 0.625}`
- **Second-Reading Acc**: `0.7083` (`P2=0.25`, `P3=0.875`)

## 3. 50+ Turn Retention (`long_50t`)
- `include_recap=False`: **Acc**: `0.5833` | **Early-Consent Allow**: `0.8333` | **Mismatch Block**: `0.3333` | **Flip Rate**: `0.3333`
- `with_recap=True` ablation: **Acc**: `0.5833` | **Early-Consent Allow**: `0.6667` | **Mismatch Block**: `0.5`

## 4. General `decider` Replay (`decider_replay`)
- `n=60` | **Overall Acc**: `0.5` | **By Type**: `{'noul': 0.6, 'choice': 0.52, 'score': 0.3333}` | **Top-Label ECE**: `0.2135` | **Brier**: `0.2032`
