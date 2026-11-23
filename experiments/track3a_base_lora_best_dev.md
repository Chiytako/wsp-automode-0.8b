# Evaluation Report: `track3a_base_lora_best_dev`

- **Split**: `select_dev` | **Mode**: `plain` | **Include Recap**: `False` | **Elapsed**: `140.67s`
- **Track 3 Select Score**: `6.20938` | **Peak VRAM**: `2140.8 MB` | **Train Sec**: `1515.5s`
- **Proxy Score**: `4.33433` (`rc_sep=0.9786`, `p23_sep=0.94848`, `p5_sep=0.95902`, `calib=-0.158`, `order=-0.0715`, `replay=0.7142`)
- **Weight Diff**: `52.28319%` modified (`393375078` / `752393024` params, `221` tensors, `||ΔW||_F = 2.801579`)

## 1. Real Session Counterfactuals (`real_cf`)
- `n=80` | **AUROC**: `1.0` | **Recall@0.5**: `1.0` | **FPR@0.5**: `0.025` | **Acc@0.5**: `0.9875` | **F1**: `0.9877`
- **ECE**: `0.0192` | **Brier**: `0.0059` | **BCE**: `0.025` | **Margin Sep (Block-Allow)**: `9.0531` (`Block=4.6813`, `Allow=-4.3719`)
- **Order Flip Rate**: `0.0` | **Second-Reading** (trig `0.0125`): AUROC `1.0`, Recall `1.0`, FPR `0.025`, ECE `0.0188`

## 2. 5-Pattern Explicit Consent (`consent_5p`)
- **Overall Acc**: `0.9167` | **P2 Allow Rate**: `1.0` | **P3 Block Rate**: `0.875` | **P5a Allow Rate**: `1.0` | **P5b Block Rate**: `0.875`
- **Margin Sep**: `P3-P2 = 7.2656`, `P5b-P5a = 7.7344` | **Flip Rate**: `0.0`
- **P2 vs P3 Confusion**: `{'p2_allow': 8, 'p2_block_false_positive': 0, 'p3_block': 7, 'p3_allow_false_negative': 1}`
- **Per-Pattern Acc**: `{'p1_no_consent': 0.875, 'p2_exact_consent': 1.0, 'p3_target_mismatch': 0.875, 'p4_vague_consent': 0.875, 'p5a_blocked_approved_same': 1.0, 'p5b_blocked_swapped_target': 0.875}`
- **Second-Reading Acc**: `0.9167` (`P2=1.0`, `P3=0.875`)

## 3. 50+ Turn Retention (`long_50t`)
- `include_recap=False`: **Acc**: `0.9167` | **Early-Consent Allow**: `0.8333` | **Mismatch Block**: `1.0` | **Flip Rate**: `0.0`
- `with_recap=True` ablation: **Acc**: `0.9167` | **Early-Consent Allow**: `0.8333` | **Mismatch Block**: `1.0`

## 4. General `decider` Replay (`decider_replay`)
- `n=60` | **Overall Acc**: `0.9167` | **By Type**: `{'noul': 0.95, 'choice': 0.96, 'score': 0.8}` | **Top-Label ECE**: `0.135` | **Brier**: `0.0543`
