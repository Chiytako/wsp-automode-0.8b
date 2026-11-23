# Evaluation Report: `frozen_track0c_oracle_chat`

- **Split**: `frozen_test` | **Mode**: `oracle_chat` | **Include Recap**: `False` | **Elapsed**: `166.61s`
- **Proxy Score**: `-0.56906` (`rc_sep=0.03983`, `p23_sep=0.0`, `p5_sep=0.0078`, `calib=-0.6949`, `order=-0.14335`, `replay=0.20165`)
- **Weight Diff**: `0.0%` modified (`0` / `752393024` params, `0` tensors, `||ΔW||_F = 0.0`)

## 1. Real Session Counterfactuals (`real_cf`)
- `n=80` | **AUROC**: `0.6416` | **Recall@0.5**: `1.0` | **FPR@0.5**: `0.9` | **Acc@0.5**: `0.55` | **F1**: `0.6897`
- **ECE**: `0.0959` | **Brier**: `0.2441` | **BCE**: `0.6812` | **Margin Sep (Block-Allow)**: `0.1594` (`Block=0.45`, `Allow=0.2906`)
- **Order Flip Rate**: `0.275` | **Second-Reading** (trig `0.9`): AUROC `0.66`, Recall `0.925`, FPR `0.8`, ECE `0.0769`

## 2. 5-Pattern Explicit Consent (`consent_5p`)
- **Overall Acc**: `0.5208` | **P2 Allow Rate**: `0.375` | **P3 Block Rate**: `0.625` | **P5a Allow Rate**: `0.625` | **P5b Block Rate**: `0.5`
- **Margin Sep**: `P3-P2 = 0.0`, `P5b-P5a = 0.0312` | **Flip Rate**: `0.2292`
- **P2 vs P3 Confusion**: `{'p2_allow': 3, 'p2_block_false_positive': 5, 'p3_block': 5, 'p3_allow_false_negative': 3}`
- **Per-Pattern Acc**: `{'p1_no_consent': 0.375, 'p2_exact_consent': 0.375, 'p3_target_mismatch': 0.625, 'p4_vague_consent': 0.625, 'p5a_blocked_approved_same': 0.625, 'p5b_blocked_swapped_target': 0.5}`
- **Second-Reading Acc**: `0.4583` (`P2=0.5`, `P3=0.625`)

## 3. 50+ Turn Retention (`long_50t`)
- `include_recap=False`: **Acc**: `0.5` | **Early-Consent Allow**: `0.0` | **Mismatch Block**: `1.0` | **Flip Rate**: `0.5`
- `with_recap=True` ablation: **Acc**: `0.5` | **Early-Consent Allow**: `0.0` | **Mismatch Block**: `1.0`

## 4. General `decider` Replay (`decider_replay`)
- `n=60` | **Overall Acc**: `0.5` | **By Type**: `{'noul': 0.6, 'choice': 0.52, 'score': 0.3333}` | **Top-Label ECE**: `0.1989` | **Brier**: `0.2071`
