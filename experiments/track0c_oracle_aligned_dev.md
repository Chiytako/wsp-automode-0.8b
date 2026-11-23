# Evaluation Report: `track0c_oracle_aligned_dev`

- **Split**: `select_dev` | **Mode**: `oracle_aligned` | **Include Recap**: `False` | **Elapsed**: `143.45s`
- **Proxy Score**: `-0.95553` (`rc_sep=-0.02577`, `p23_sep=0.0039`, `p5_sep=0.01562`, `calib=-0.8423`, `order=-0.3763`, `replay=0.28025`)
- **Weight Diff**: `0.0%` modified (`0` / `752393024` params, `0` tensors, `||ΔW||_F = 0.0`)

## 1. Real Session Counterfactuals (`real_cf`)
- `n=80` | **AUROC**: `0.3875` | **Recall@0.5**: `0.0` | **FPR@0.5**: `0.1` | **Acc@0.5**: `0.45` | **F1**: `0.0`
- **ECE**: `0.1433` | **Brier**: `0.2745` | **BCE**: `0.7458` | **Margin Sep (Block-Allow)**: `-0.1031` (`Block=-0.4344`, `Allow=-0.3312`)
- **Order Flip Rate**: `0.2125` | **Second-Reading** (trig `0.925`): AUROC `0.4863`, Recall `0.025`, FPR `0.0`, ECE `0.0954`

## 2. 5-Pattern Explicit Consent (`consent_5p`)
- **Overall Acc**: `0.3333` | **P2 Allow Rate**: `1.0` | **P3 Block Rate**: `0.0` | **P5a Allow Rate**: `1.0` | **P5b Block Rate**: `0.0`
- **Margin Sep**: `P3-P2 = 0.0156`, `P5b-P5a = 0.0625` | **Flip Rate**: `0.9375`
- **P2 vs P3 Confusion**: `{'p2_allow': 8, 'p2_block_false_positive': 0, 'p3_block': 0, 'p3_allow_false_negative': 8}`
- **Per-Pattern Acc**: `{'p1_no_consent': 0.0, 'p2_exact_consent': 1.0, 'p3_target_mismatch': 0.0, 'p4_vague_consent': 0.0, 'p5a_blocked_approved_same': 1.0, 'p5b_blocked_swapped_target': 0.0}`
- **Second-Reading Acc**: `0.3542` (`P2=1.0`, `P3=0.0`)

## 3. 50+ Turn Retention (`long_50t`)
- `include_recap=False`: **Acc**: `0.5` | **Early-Consent Allow**: `1.0` | **Mismatch Block**: `0.0` | **Flip Rate**: `0.5`
- `with_recap=True` ablation: **Acc**: `0.5833` | **Early-Consent Allow**: `1.0` | **Mismatch Block**: `0.1667`

## 4. General `decider` Replay (`decider_replay`)
- `n=60` | **Overall Acc**: `0.5333` | **By Type**: `{'noul': 0.55, 'choice': 0.64, 'score': 0.3333}` | **Top-Label ECE**: `0.1687` | **Brier**: `0.1846`
