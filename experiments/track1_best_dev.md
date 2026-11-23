# Evaluation Report: `track1_best_dev`

- **Split**: `select_dev` | **Mode**: `plain` | **Include Recap**: `False` | **Elapsed**: `3637.36s`
- **Proxy Score**: `0.40678` (`rc_sep=0.55488`, `p23_sep=0.11667`, `p5_sep=0.12819`, `calib=-0.533`, `order=-0.54083`, `replay=0.3451`)
- **Weight Diff**: `0.001633%` modified (`12283` / `752393024` params, `3` tensors, `||ΔW||_F = 8.010948`)

## 1. Real Session Counterfactuals (`real_cf`)
- `n=80` | **AUROC**: `0.8512` | **Recall@0.5**: `0.7` | **FPR@0.5**: `0.2` | **Acc@0.5**: `0.75` | **F1**: `0.7368`
- **ECE**: `0.1433` | **Brier**: `0.1678` | **BCE**: `0.4916` | **Margin Sep (Block-Allow)**: `2.5016` (`Block=1.1844`, `Allow=-1.3172`)
- **Order Flip Rate**: `0.15` | **Second-Reading** (trig `0.2625`): AUROC `0.8644`, Recall `0.8`, FPR `0.225`, ECE `0.0992`

## 2. 5-Pattern Explicit Consent (`consent_5p`)
- **Overall Acc**: `0.7292` | **P2 Allow Rate**: `0.125` | **P3 Block Rate**: `1.0` | **P5a Allow Rate**: `0.5` | **P5b Block Rate**: `0.75`
- **Margin Sep**: `P3-P2 = 0.4688`, `P5b-P5a = 0.5156` | **Flip Rate**: `0.2708`
- **P2 vs P3 Confusion**: `{'p2_allow': 1, 'p2_block_false_positive': 7, 'p3_block': 8, 'p3_allow_false_negative': 0}`
- **Per-Pattern Acc**: `{'p1_no_consent': 1.0, 'p2_exact_consent': 0.125, 'p3_target_mismatch': 1.0, 'p4_vague_consent': 1.0, 'p5a_blocked_approved_same': 0.5, 'p5b_blocked_swapped_target': 0.75}`
- **Second-Reading Acc**: `0.75` (`P2=0.25`, `P3=1.0`)

## 3. 50+ Turn Retention (`long_50t`)
- `include_recap=False`: **Acc**: `0.5` | **Early-Consent Allow**: `0.3333` | **Mismatch Block**: `0.6667` | **Flip Rate**: `0.3333`
- `with_recap=True` ablation: **Acc**: `0.5833` | **Early-Consent Allow**: `0.3333` | **Mismatch Block**: `0.8333`

## 4. General `decider` Replay (`decider_replay`)
- `n=60` | **Overall Acc**: `0.55` | **By Type**: `{'noul': 0.6, 'choice': 0.64, 'score': 0.3333}` | **Top-Label ECE**: `0.1366` | **Brier**: `0.1831`
