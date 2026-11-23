# Evaluation Report: `track3c_r32_3ep_lora_ep3_dev`

- **Split**: `select_dev` | **Mode**: `plain` | **Include Recap**: `False` | **Elapsed**: `140.85s`
- **Track 3 Select Score**: `6.26425` | **Peak VRAM**: `2306.9 MB` | **Train Sec**: `1514.2s`
- **Proxy Score**: `4.3955` (`rc_sep=0.98103`, `p23_sep=0.95369`, `p5_sep=0.96319`, `calib=-0.12565`, `order=-0.06993`, `replay=0.7258`)
- **Weight Diff**: `54.805714%` modified (`412354372` / `752393024` params, `221` tensors, `||ΔW||_F = 3.482085`)

## 1. Real Session Counterfactuals (`real_cf`)
- `n=80` | **AUROC**: `1.0` | **Recall@0.5**: `1.0` | **FPR@0.5**: `0.05` | **Acc@0.5**: `0.975` | **F1**: `0.9756`
- **ECE**: `0.032` | **Brier**: `0.0151` | **BCE**: `0.0531` | **Margin Sep (Block-Allow)**: `9.2969` (`Block=5.5531`, `Allow=-3.7437`)
- **Order Flip Rate**: `0.0125` | **Second-Reading** (trig `0.0125`): AUROC `1.0`, Recall `1.0`, FPR `0.05`, ECE `0.0311`

## 2. 5-Pattern Explicit Consent (`consent_5p`)
- **Overall Acc**: `0.9375` | **P2 Allow Rate**: `1.0` | **P3 Block Rate**: `0.875` | **P5a Allow Rate**: `1.0` | **P5b Block Rate**: `0.875`
- **Margin Sep**: `P3-P2 = 7.4844`, `P5b-P5a = 7.9531` | **Flip Rate**: `0.0`
- **P2 vs P3 Confusion**: `{'p2_allow': 8, 'p2_block_false_positive': 0, 'p3_block': 7, 'p3_allow_false_negative': 1}`
- **Per-Pattern Acc**: `{'p1_no_consent': 1.0, 'p2_exact_consent': 1.0, 'p3_target_mismatch': 0.875, 'p4_vague_consent': 0.875, 'p5a_blocked_approved_same': 1.0, 'p5b_blocked_swapped_target': 0.875}`
- **Second-Reading Acc**: `0.9375` (`P2=1.0`, `P3=0.875`)

## 3. 50+ Turn Retention (`long_50t`)
- `include_recap=False`: **Acc**: `0.9167` | **Early-Consent Allow**: `0.8333` | **Mismatch Block**: `1.0` | **Flip Rate**: `0.0833`
- `with_recap=True` ablation: **Acc**: `0.8333` | **Early-Consent Allow**: `0.6667` | **Mismatch Block**: `1.0`

## 4. General `decider` Replay (`decider_replay`)
- `n=60` | **Overall Acc**: `0.8833` | **By Type**: `{'noul': 0.9, 'choice': 0.92, 'score': 0.8}` | **Top-Label ECE**: `0.105` | **Brier**: `0.0431`
