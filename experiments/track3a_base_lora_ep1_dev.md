# Evaluation Report: `track3a_base_lora_ep1_dev`

- **Split**: `select_dev` | **Mode**: `plain` | **Include Recap**: `False` | **Elapsed**: `140.89s`
- **Track 3 Select Score**: `5.75906` | **Peak VRAM**: `2142.6 MB` | **Train Sec**: `1522.0s`
- **Proxy Score**: `4.07406` (`rc_sep=0.97416`, `p23_sep=0.94769`, `p5_sep=0.96146`, `calib=-0.3084`, `order=-0.11497`, `replay=0.6532`)
- **Weight Diff**: `51.530236%` modified (`387709902` / `752393024` params, `219` tensors, `||ΔW||_F = 2.61967`)

## 1. Real Session Counterfactuals (`real_cf`)
- `n=80` | **AUROC**: `0.9991` | **Recall@0.5**: `1.0` | **FPR@0.5**: `0.05` | **Acc@0.5**: `0.975` | **F1**: `0.9756`
- **ECE**: `0.0256` | **Brier**: `0.0229` | **BCE**: `0.0907` | **Margin Sep (Block-Allow)**: `8.6719` (`Block=4.5625`, `Allow=-4.1094`)
- **Order Flip Rate**: `0.0` | **Second-Reading** (trig `0.0`): AUROC `0.9991`, Recall `1.0`, FPR `0.05`, ECE `0.0256`

## 2. 5-Pattern Explicit Consent (`consent_5p`)
- **Overall Acc**: `0.8542` | **P2 Allow Rate**: `1.0` | **P3 Block Rate**: `0.875` | **P5a Allow Rate**: `1.0` | **P5b Block Rate**: `0.875`
- **Margin Sep**: `P3-P2 = 7.2344`, `P5b-P5a = 7.8594` | **Flip Rate**: `0.0417`
- **P2 vs P3 Confusion**: `{'p2_allow': 8, 'p2_block_false_positive': 0, 'p3_block': 7, 'p3_allow_false_negative': 1}`
- **Per-Pattern Acc**: `{'p1_no_consent': 0.75, 'p2_exact_consent': 1.0, 'p3_target_mismatch': 0.875, 'p4_vague_consent': 0.625, 'p5a_blocked_approved_same': 1.0, 'p5b_blocked_swapped_target': 0.875}`
- **Second-Reading Acc**: `0.8333` (`P2=1.0`, `P3=0.875`)

## 3. 50+ Turn Retention (`long_50t`)
- `include_recap=False`: **Acc**: `0.6667` | **Early-Consent Allow**: `0.3333` | **Mismatch Block**: `1.0` | **Flip Rate**: `0.0833`
- `with_recap=True` ablation: **Acc**: `0.8333` | **Early-Consent Allow**: `0.6667` | **Mismatch Block**: `1.0`

## 4. General `decider` Replay (`decider_replay`)
- `n=60` | **Overall Acc**: `0.85` | **By Type**: `{'noul': 0.9, 'choice': 0.92, 'score': 0.6667}` | **Top-Label ECE**: `0.1312` | **Brier**: `0.0696`
