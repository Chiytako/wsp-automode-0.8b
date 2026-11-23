# Evaluation Report: `track2_best_dev`

- **Split**: `select_dev` | **Mode**: `plain` | **Include Recap**: `False` | **Elapsed**: `5100.09s`
- **Proxy Score**: `0.56208` (`rc_sep=0.53146`, `p23_sep=0.18533`, `p5_sep=0.18156`, `calib=-0.52565`, `order=-0.49128`, `replay=0.32225`)
- **Weight Diff**: `0.003264%` modified (`24561` / `752393024` params, `6` tensors, `||ΔW||_F = 11.888535`)

## 1. Real Session Counterfactuals (`real_cf`)
- `n=80` | **AUROC**: `0.8731` | **Recall@0.5**: `0.625` | **FPR@0.5**: `0.175` | **Acc@0.5**: `0.725` | **F1**: `0.6944`
- **ECE**: `0.1653` | **Brier**: `0.1662` | **BCE**: `0.4885` | **Margin Sep (Block-Allow)**: `2.3687` (`Block=0.775`, `Allow=-1.5938`)
- **Order Flip Rate**: `0.2375` | **Second-Reading** (trig `0.125`): AUROC `0.8753`, Recall `0.7`, FPR `0.2`, ECE `0.1529`

## 2. 5-Pattern Explicit Consent (`consent_5p`)
- **Overall Acc**: `0.75` | **P2 Allow Rate**: `0.375` | **P3 Block Rate**: `0.875` | **P5a Allow Rate**: `0.5` | **P5b Block Rate**: `0.75`
- **Margin Sep**: `P3-P2 = 0.75`, `P5b-P5a = 0.7344` | **Flip Rate**: `0.1042`
- **P2 vs P3 Confusion**: `{'p2_allow': 3, 'p2_block_false_positive': 5, 'p3_block': 7, 'p3_allow_false_negative': 1}`
- **Per-Pattern Acc**: `{'p1_no_consent': 1.0, 'p2_exact_consent': 0.375, 'p3_target_mismatch': 0.875, 'p4_vague_consent': 1.0, 'p5a_blocked_approved_same': 0.5, 'p5b_blocked_swapped_target': 0.75}`
- **Second-Reading Acc**: `0.75` (`P2=0.25`, `P3=0.875`)

## 3. 50+ Turn Retention (`long_50t`)
- `include_recap=False`: **Acc**: `0.5` | **Early-Consent Allow**: `0.3333` | **Mismatch Block**: `0.6667` | **Flip Rate**: `0.1667`
- `with_recap=True` ablation: **Acc**: `0.5833` | **Early-Consent Allow**: `0.3333` | **Mismatch Block**: `0.8333`

## 4. General `decider` Replay (`decider_replay`)
- `n=60` | **Overall Acc**: `0.5333` | **By Type**: `{'noul': 0.55, 'choice': 0.64, 'score': 0.3333}` | **Top-Label ECE**: `0.1407` | **Brier**: `0.184`
