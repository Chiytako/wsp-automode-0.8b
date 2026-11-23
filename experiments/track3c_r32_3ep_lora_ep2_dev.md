# Evaluation Report: `track3c_r32_3ep_lora_ep2_dev`

- **Split**: `select_dev` | **Mode**: `plain` | **Include Recap**: `False` | **Elapsed**: `140.76s`
- **Track 3 Select Score**: `6.08276` | **Peak VRAM**: `2302.4 MB` | **Train Sec**: `1513.7s`
- **Proxy Score**: `4.35986` (`rc_sep=0.98254`, `p23_sep=0.94729`, `p5_sep=0.9677`, `calib=-0.19915`, `order=-0.10978`, `replay=0.80635`)
- **Weight Diff**: `54.591743%` modified (`410744465` / `752393024` params, `220` tensors, `||ΔW||_F = 3.410835`)

## 1. Real Session Counterfactuals (`real_cf`)
- `n=80` | **AUROC**: `1.0` | **Recall@0.5**: `1.0` | **FPR@0.5**: `0.075` | **Acc@0.5**: `0.9625` | **F1**: `0.9639`
- **ECE**: `0.0455` | **Brier**: `0.0342` | **BCE**: `0.1368` | **Margin Sep (Block-Allow)**: `9.4641` (`Block=6.0609`, `Allow=-3.4031`)
- **Order Flip Rate**: `0.0` | **Second-Reading** (trig `0.0`): AUROC `1.0`, Recall `1.0`, FPR `0.075`, ECE `0.0455`

## 2. 5-Pattern Explicit Consent (`consent_5p`)
- **Overall Acc**: `0.8958` | **P2 Allow Rate**: `0.875` | **P3 Block Rate**: `0.875` | **P5a Allow Rate**: `1.0` | **P5b Block Rate**: `0.875`
- **Margin Sep**: `P3-P2 = 7.2188`, `P5b-P5a = 8.2188` | **Flip Rate**: `0.0208`
- **P2 vs P3 Confusion**: `{'p2_allow': 7, 'p2_block_false_positive': 1, 'p3_block': 7, 'p3_allow_false_negative': 1}`
- **Per-Pattern Acc**: `{'p1_no_consent': 1.0, 'p2_exact_consent': 0.875, 'p3_target_mismatch': 0.875, 'p4_vague_consent': 0.75, 'p5a_blocked_approved_same': 1.0, 'p5b_blocked_swapped_target': 0.875}`
- **Second-Reading Acc**: `0.8958` (`P2=0.875`, `P3=0.875`)

## 3. 50+ Turn Retention (`long_50t`)
- `include_recap=False`: **Acc**: `0.6667` | **Early-Consent Allow**: `0.3333` | **Mismatch Block**: `1.0` | **Flip Rate**: `0.0`
- `with_recap=True` ablation: **Acc**: `0.6667` | **Early-Consent Allow**: `0.3333` | **Mismatch Block**: `1.0`

## 4. General `decider` Replay (`decider_replay`)
- `n=60` | **Overall Acc**: `0.8833` | **By Type**: `{'noul': 0.9, 'choice': 0.92, 'score': 0.8}` | **Top-Label ECE**: `0.0513` | **Brier**: `0.0547`
