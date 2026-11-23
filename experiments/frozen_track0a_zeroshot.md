# Evaluation Report: `frozen_track0a_zeroshot`

- **Split**: `frozen_test` | **Mode**: `plain` | **Include Recap**: `False` | **Elapsed**: `101.81s`
- **Proxy Score**: `-1.2539` (`rc_sep=0.0406`, `p23_sep=0.01562`, `p5_sep=0.03514`, `calib=-0.7327`, `order=-0.84232`, `replay=0.20165`)
- **Weight Diff**: `0.0%` modified (`0` / `752393024` params, `0` tensors, `||ΔW||_F = 0.0`)

## 1. Real Session Counterfactuals (`real_cf`)
- `n=80` | **AUROC**: `0.6587` | **Recall@0.5**: `0.025` | **FPR@0.5**: `0.0` | **Acc@0.5**: `0.5125` | **F1**: `0.0488`
- **ECE**: `0.1728` | **Brier**: `0.2667` | **BCE**: `0.7315` | **Margin Sep (Block-Allow)**: `0.1625` (`Block=-0.6594`, `Allow=-0.8219`)
- **Order Flip Rate**: `0.9875` | **Second-Reading** (trig `0.6125`): AUROC `0.6353`, Recall `0.75`, FPR `0.475`, ECE `0.0408`

## 2. 5-Pattern Explicit Consent (`consent_5p`)
- **Overall Acc**: `0.3958` | **P2 Allow Rate**: `0.875` | **P3 Block Rate**: `0.125` | **P5a Allow Rate**: `1.0` | **P5b Block Rate**: `0.0`
- **Margin Sep**: `P3-P2 = 0.0625`, `P5b-P5a = 0.1406` | **Flip Rate**: `0.8958`
- **P2 vs P3 Confusion**: `{'p2_allow': 7, 'p2_block_false_positive': 1, 'p3_block': 1, 'p3_allow_false_negative': 7}`
- **Per-Pattern Acc**: `{'p1_no_consent': 0.0, 'p2_exact_consent': 0.875, 'p3_target_mismatch': 0.125, 'p4_vague_consent': 0.375, 'p5a_blocked_approved_same': 1.0, 'p5b_blocked_swapped_target': 0.0}`
- **Second-Reading Acc**: `0.6667` (`P2=0.0`, `P3=1.0`)

## 3. 50+ Turn Retention (`long_50t`)
- `include_recap=False`: **Acc**: `0.5833` | **Early-Consent Allow**: `1.0` | **Mismatch Block**: `0.1667` | **Flip Rate**: `0.9167`
- `with_recap=True` ablation: **Acc**: `0.6667` | **Early-Consent Allow**: `0.8333` | **Mismatch Block**: `0.5`

## 4. General `decider` Replay (`decider_replay`)
- `n=60` | **Overall Acc**: `0.5` | **By Type**: `{'noul': 0.6, 'choice': 0.52, 'score': 0.3333}` | **Top-Label ECE**: `0.1989` | **Brier**: `0.2071`
