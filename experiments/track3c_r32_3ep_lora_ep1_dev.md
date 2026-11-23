# Evaluation Report: `track3c_r32_3ep_lora_ep1_dev`

- **Split**: `select_dev` | **Mode**: `plain` | **Include Recap**: `False` | **Elapsed**: `140.75s`
- **Track 3 Select Score**: `2.75044` | **Peak VRAM**: `2300.7 MB` | **Train Sec**: `1512.3s`
- **Proxy Score**: `1.26159` (`rc_sep=0.57898`, `p23_sep=0.22648`, `p5_sep=0.23018`, `calib=-0.4712`, `order=-0.24988`, `replay=0.5443`)
- **Weight Diff**: `52.479527%` modified (`394852301` / `752393024` params, `220` tensors, `||ΔW||_F = 2.773654`)

## 1. Real Session Counterfactuals (`real_cf`)
- `n=80` | **AUROC**: `0.9819` | **Recall@0.5**: `0.9` | **FPR@0.5**: `0.075` | **Acc@0.5**: `0.9125` | **F1**: `0.9114`
- **ECE**: `0.1542` | **Brier**: `0.0771` | **BCE**: `0.2943` | **Margin Sep (Block-Allow)**: `2.6437` (`Block=1.4281`, `Allow=-1.2156`)
- **Order Flip Rate**: `0.025` | **Second-Reading** (trig `0.15`): AUROC `0.9859`, Recall `0.9`, FPR `0.05`, ECE `0.1624`

## 2. 5-Pattern Explicit Consent (`consent_5p`)
- **Overall Acc**: `0.6458` | **P2 Allow Rate**: `0.375` | **P3 Block Rate**: `1.0` | **P5a Allow Rate**: `0.375` | **P5b Block Rate**: `1.0`
- **Margin Sep**: `P3-P2 = 0.9219`, `P5b-P5a = 0.9375` | **Flip Rate**: `0.3542`
- **P2 vs P3 Confusion**: `{'p2_allow': 3, 'p2_block_false_positive': 5, 'p3_block': 8, 'p3_allow_false_negative': 0}`
- **Per-Pattern Acc**: `{'p1_no_consent': 0.625, 'p2_exact_consent': 0.375, 'p3_target_mismatch': 1.0, 'p4_vague_consent': 0.5, 'p5a_blocked_approved_same': 0.375, 'p5b_blocked_swapped_target': 1.0}`
- **Second-Reading Acc**: `0.6458` (`P2=0.375`, `P3=1.0`)

## 3. 50+ Turn Retention (`long_50t`)
- `include_recap=False`: **Acc**: `0.5` | **Early-Consent Allow**: `0.0` | **Mismatch Block**: `1.0` | **Flip Rate**: `0.1667`
- `with_recap=True` ablation: **Acc**: `0.5` | **Early-Consent Allow**: `0.0` | **Mismatch Block**: `1.0`

## 4. General `decider` Replay (`decider_replay`)
- `n=60` | **Overall Acc**: `0.85` | **By Type**: `{'noul': 0.9, 'choice': 0.88, 'score': 0.7333}` | **Top-Label ECE**: `0.2038` | **Brier**: `0.1081`
