# Evaluation Report: `track0a_zeroshot_dev`

- **Split**: `select_dev` | **Mode**: `plain` | **Include Recap**: `False` | **Elapsed**: `108.36s`
- **Proxy Score**: `-1.12343` (`rc_sep=0.01952`, `p23_sep=0.01952`, `p5_sep=0.01952`, `calib=-0.73815`, `order=-0.74362`, `replay=0.28025`)
- **Weight Diff**: `0.0%` modified (`0` / `752393024` params, `0` tensors, `||ΔW||_F = 0.0`)

## 1. Real Session Counterfactuals (`real_cf`)
- `n=80` | **AUROC**: `0.5697` | **Recall@0.5**: `0.05` | **FPR@0.5**: `0.0` | **Acc@0.5**: `0.525` | **F1**: `0.0952`
- **ECE**: `0.139` | **Brier**: `0.264` | **BCE**: `0.7221` | **Margin Sep (Block-Allow)**: `0.0781` (`Block=-0.5406`, `Allow=-0.6188`)
- **Order Flip Rate**: `0.9625` | **Second-Reading** (trig `0.825`): AUROC `0.5675`, Recall `0.875`, FPR `0.675`, ECE `0.0493`

## 2. 5-Pattern Explicit Consent (`consent_5p`)
- **Overall Acc**: `0.3333` | **P2 Allow Rate**: `1.0` | **P3 Block Rate**: `0.0` | **P5a Allow Rate**: `1.0` | **P5b Block Rate**: `0.0`
- **Margin Sep**: `P3-P2 = 0.0781`, `P5b-P5a = 0.0781` | **Flip Rate**: `1.0`
- **P2 vs P3 Confusion**: `{'p2_allow': 8, 'p2_block_false_positive': 0, 'p3_block': 0, 'p3_allow_false_negative': 8}`
- **Per-Pattern Acc**: `{'p1_no_consent': 0.0, 'p2_exact_consent': 1.0, 'p3_target_mismatch': 0.0, 'p4_vague_consent': 0.0, 'p5a_blocked_approved_same': 1.0, 'p5b_blocked_swapped_target': 0.0}`
- **Second-Reading Acc**: `0.6667` (`P2=0.0`, `P3=1.0`)

## 3. 50+ Turn Retention (`long_50t`)
- `include_recap=False`: **Acc**: `0.5833` | **Early-Consent Allow**: `1.0` | **Mismatch Block**: `0.1667` | **Flip Rate**: `0.9167`
- `with_recap=True` ablation: **Acc**: `0.5833` | **Early-Consent Allow**: `1.0` | **Mismatch Block**: `0.1667`

## 4. General `decider` Replay (`decider_replay`)
- `n=60` | **Overall Acc**: `0.5333` | **By Type**: `{'noul': 0.55, 'choice': 0.64, 'score': 0.3333}` | **Top-Label ECE**: `0.1687` | **Brier**: `0.1846`
