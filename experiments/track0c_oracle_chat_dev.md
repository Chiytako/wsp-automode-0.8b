# Evaluation Report: `track0c_oracle_chat_dev`

- **Split**: `select_dev` | **Mode**: `oracle_chat` | **Include Recap**: `False` | **Elapsed**: `173.02s`
- **Proxy Score**: `-0.39148` (`rc_sep=0.04996`, `p23_sep=0.04684`, `p5_sep=0.04295`, `calib=-0.66585`, `order=-0.19403`, `replay=0.28025`)
- **Weight Diff**: `0.0%` modified (`0` / `752393024` params, `0` tensors, `||ΔW||_F = 0.0`)

## 1. Real Session Counterfactuals (`real_cf`)
- `n=80` | **AUROC**: `0.65` | **Recall@0.5**: `0.9` | **FPR@0.5**: `0.725` | **Acc@0.5**: `0.5875` | **F1**: `0.6857`
- **ECE**: `0.0882` | **Brier**: `0.2371` | **BCE**: `0.6658` | **Margin Sep (Block-Allow)**: `0.2` (`Block=0.3594`, `Allow=0.1594`)
- **Order Flip Rate**: `0.6125` | **Second-Reading** (trig `0.95`): AUROC `0.7394`, Recall `0.9`, FPR `0.725`, ECE `0.0696`

## 2. 5-Pattern Explicit Consent (`consent_5p`)
- **Overall Acc**: `0.625` | **P2 Allow Rate**: `0.375` | **P3 Block Rate**: `0.875` | **P5a Allow Rate**: `0.875` | **P5b Block Rate**: `0.5`
- **Margin Sep**: `P3-P2 = 0.1875`, `P5b-P5a = 0.1719` | **Flip Rate**: `0.6667`
- **P2 vs P3 Confusion**: `{'p2_allow': 3, 'p2_block_false_positive': 5, 'p3_block': 7, 'p3_allow_false_negative': 1}`
- **Per-Pattern Acc**: `{'p1_no_consent': 0.5, 'p2_exact_consent': 0.375, 'p3_target_mismatch': 0.875, 'p4_vague_consent': 0.625, 'p5a_blocked_approved_same': 0.875, 'p5b_blocked_swapped_target': 0.5}`
- **Second-Reading Acc**: `0.6458` (`P2=0.25`, `P3=0.875`)

## 3. 50+ Turn Retention (`long_50t`)
- `include_recap=False`: **Acc**: `0.5833` | **Early-Consent Allow**: `0.1667` | **Mismatch Block**: `1.0` | **Flip Rate**: `0.5`
- `with_recap=True` ablation: **Acc**: `0.5` | **Early-Consent Allow**: `0.0` | **Mismatch Block**: `1.0`

## 4. General `decider` Replay (`decider_replay`)
- `n=60` | **Overall Acc**: `0.5333` | **By Type**: `{'noul': 0.55, 'choice': 0.64, 'score': 0.3333}` | **Top-Label ECE**: `0.1687` | **Brier**: `0.1846`
