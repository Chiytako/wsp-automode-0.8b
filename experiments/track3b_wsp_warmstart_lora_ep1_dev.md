# Evaluation Report: `track3b_wsp_warmstart_lora_ep1_dev`

- **Split**: `select_dev` | **Mode**: `plain` | **Include Recap**: `False` | **Elapsed**: `139.24s`
- **Track 3 Select Score**: `2.76666` | **Peak VRAM**: `2133.6 MB` | **Train Sec**: `1498.9s`
- **Proxy Score**: `1.26201` (`rc_sep=0.5857`, `p23_sep=0.17779`, `p5_sep=0.12819`, `calib=-0.5187`, `order=-0.20897`, `replay=0.71625`)
- **Weight Diff**: `50.416333%` modified (`379328970` / `752393024` params, `220` tensors, `||ΔW||_F = 12.113531`)

## 1. Real Session Counterfactuals (`real_cf`)
- `n=80` | **AUROC**: `0.9759` | **Recall@0.5**: `0.95` | **FPR@0.5**: `0.075` | **Acc@0.5**: `0.9375` | **F1**: `0.9383`
- **ECE**: `0.1737` | **Brier**: `0.0768` | **BCE**: `0.2911` | **Margin Sep (Block-Allow)**: `2.6844` (`Block=1.3969`, `Allow=-1.2875`)
- **Order Flip Rate**: `0.0875` | **Second-Reading** (trig `0.2`): AUROC `0.9822`, Recall `0.9`, FPR `0.05`, ECE `0.1627`

## 2. 5-Pattern Explicit Consent (`consent_5p`)
- **Overall Acc**: `0.6667` | **P2 Allow Rate**: `0.125` | **P3 Block Rate**: `1.0` | **P5a Allow Rate**: `0.125` | **P5b Block Rate**: `1.0`
- **Margin Sep**: `P3-P2 = 0.7188`, `P5b-P5a = 0.5156` | **Flip Rate**: `0.0208`
- **P2 vs P3 Confusion**: `{'p2_allow': 1, 'p2_block_false_positive': 7, 'p3_block': 8, 'p3_allow_false_negative': 0}`
- **Per-Pattern Acc**: `{'p1_no_consent': 0.875, 'p2_exact_consent': 0.125, 'p3_target_mismatch': 1.0, 'p4_vague_consent': 0.875, 'p5a_blocked_approved_same': 0.125, 'p5b_blocked_swapped_target': 1.0}`
- **Second-Reading Acc**: `0.6667` (`P2=0.125`, `P3=1.0`)

## 3. 50+ Turn Retention (`long_50t`)
- `include_recap=False`: **Acc**: `0.5` | **Early-Consent Allow**: `0.0` | **Mismatch Block**: `1.0` | **Flip Rate**: `0.0833`
- `with_recap=True` ablation: **Acc**: `0.5` | **Early-Consent Allow**: `0.0` | **Mismatch Block**: `1.0`

## 4. General `decider` Replay (`decider_replay`)
- `n=60` | **Overall Acc**: `0.8667` | **By Type**: `{'noul': 0.8, 'choice': 1.0, 'score': 0.7333}` | **Top-Label ECE**: `0.1003` | **Brier**: `0.073`
