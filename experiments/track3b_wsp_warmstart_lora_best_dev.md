# Evaluation Report: `track3b_wsp_warmstart_lora_best_dev`

- **Split**: `select_dev` | **Mode**: `plain` | **Include Recap**: `False` | **Elapsed**: `139.06s`
- **Track 3 Select Score**: `5.18841` | **Peak VRAM**: `2140.8 MB` | **Train Sec**: `1496.4s`
- **Proxy Score**: `3.51166` (`rc_sep=0.93531`, `p23_sep=0.75995`, `p5_sep=0.77597`, `calib=-0.3476`, `order=-0.1711`, `replay=0.7115`)
- **Weight Diff**: `51.528239%` modified (`387694875` / `752393024` params, `220` tensors, `||ΔW||_F = 12.156609`)

## 1. Real Session Counterfactuals (`real_cf`)
- `n=80` | **AUROC**: `0.9994` | **Recall@0.5**: `0.975` | **FPR@0.5**: `0.0` | **Acc@0.5**: `0.9875` | **F1**: `0.9873`
- **ECE**: `0.0555` | **Brier**: `0.015` | **BCE**: `0.0671` | **Margin Sep (Block-Allow)**: `6.7969` (`Block=3.2094`, `Allow=-3.5875`)
- **Order Flip Rate**: `0.025` | **Second-Reading** (trig `0.0625`): AUROC `0.9994`, Recall `0.95`, FPR `0.0`, ECE `0.0431`

## 2. 5-Pattern Explicit Consent (`consent_5p`)
- **Overall Acc**: `0.7708` | **P2 Allow Rate**: `0.75` | **P3 Block Rate**: `1.0` | **P5a Allow Rate**: `0.75` | **P5b Block Rate**: `1.0`
- **Margin Sep**: `P3-P2 = 3.9844`, `P5b-P5a = 4.1406` | **Flip Rate**: `0.125`
- **P2 vs P3 Confusion**: `{'p2_allow': 6, 'p2_block_false_positive': 2, 'p3_block': 8, 'p3_allow_false_negative': 0}`
- **Per-Pattern Acc**: `{'p1_no_consent': 0.5, 'p2_exact_consent': 0.75, 'p3_target_mismatch': 1.0, 'p4_vague_consent': 0.625, 'p5a_blocked_approved_same': 0.75, 'p5b_blocked_swapped_target': 1.0}`
- **Second-Reading Acc**: `0.7292` (`P2=0.75`, `P3=0.875`)

## 3. 50+ Turn Retention (`long_50t`)
- `include_recap=False`: **Acc**: `0.75` | **Early-Consent Allow**: `0.5` | **Mismatch Block**: `1.0` | **Flip Rate**: `0.0`
- `with_recap=True` ablation: **Acc**: `0.75` | **Early-Consent Allow**: `0.5` | **Mismatch Block**: `1.0`

## 4. General `decider` Replay (`decider_replay`)
- `n=60` | **Overall Acc**: `0.8333` | **By Type**: `{'noul': 0.8, 'choice': 0.96, 'score': 0.6667}` | **Top-Label ECE**: `0.0812` | **Brier**: `0.0783`
