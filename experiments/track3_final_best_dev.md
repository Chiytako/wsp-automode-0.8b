# Evaluation Report: `track3_final_best_dev`

- **Split**: `select_dev` | **Mode**: `plain` | **Include Recap**: `False` | **Elapsed**: `0.0s`
- **Track 3 Select Score**: `6.3024` | **Soup**: `track3_soup_lam_0.25` (`lambda=0.25`) | **RMSNorm Temp**: `1.0`
- **Proxy Score**: `4.43365` (`rc_sep=0.98398`, `p23_sep=0.95298`, `p5_sep=0.96204`, `calib=-0.10615`, `order=-0.09083`, `replay=0.76315`)
- **Weight Diff**: `52.787731%` modified (`397171203` / `752393024` params, `221` tensors, `||ΔW||_F = 2.77189`)

## 1. Real Session Counterfactuals (`real_cf`)
- `n=80` | **AUROC**: `1.0` | **Recall@0.5**: `1.0` | **FPR@0.5**: `0.0` | **Acc@0.5**: `1.0` | **F1**: `1.0`
- **ECE**: `0.0129` | **Brier**: `0.0006` | **BCE**: `0.0132` | **Margin Sep (Block-Allow)**: `9.6375` (`Block=5.4938`, `Allow=-4.1437`)
- **Order Flip Rate**: `0.0` | **Second-Reading** (trig `0.0`): AUROC `1.0`, Recall `1.0`, FPR `0.0`, ECE `0.0129`

## 2. 5-Pattern Explicit Consent (`consent_5p`)
- **Overall Acc**: `0.9375` | **P2 Allow Rate**: `1.0` | **P3 Block Rate**: `0.875` | **P5a Allow Rate**: `1.0` | **P5b Block Rate**: `0.875`
- **Margin Sep**: `P3-P2 = 7.4531`, `P5b-P5a = 7.8906` | **Flip Rate**: `0.0`
- **P2 vs P3 Confusion**: `{'p2_allow': 8, 'p2_block_false_positive': 0, 'p3_block': 7, 'p3_allow_false_negative': 1}`
- **Per-Pattern Acc**: `{'p1_no_consent': 1.0, 'p2_exact_consent': 1.0, 'p3_target_mismatch': 0.875, 'p4_vague_consent': 0.875, 'p5a_blocked_approved_same': 1.0, 'p5b_blocked_swapped_target': 0.875}`
- **Second-Reading Acc**: `0.9375` (`P2=1.0`, `P3=0.875`)

## 3. 50+ Turn Retention (`long_50t`)
- `include_recap=False`: **Acc**: `0.9167` | **Early-Consent Allow**: `0.8333` | **Mismatch Block**: `1.0` | **Flip Rate**: `0.0833`
- `with_recap=True` ablation: **Acc**: `0.8333` | **Early-Consent Allow**: `0.6667` | **Mismatch Block**: `1.0`

## 4. General `decider` Replay (`decider_replay`)
- `n=60` | **Overall Acc**: `0.8833` | **By Type**: `{'noul': 0.9, 'choice': 0.88, 'score': 0.8667}` | **Top-Label ECE**: `0.0801` | **Brier**: `0.044`
