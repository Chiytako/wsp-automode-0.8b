# Paper Statistical Inference & Error Analysis (`seed=20261124`, `B=10,000` Stratified Bootstrap)

## 1. Locked `frozen_test` (n=200) — Point Estimates & 95% Stratified Bootstrap CIs

| Model | `real_cf` AUROC [95% CI] | `real_cf` Acc [95% CI] | `consent_5p` Acc [95% CI] | `long_50t` Acc [95% CI] | `decider_replay` Acc [95% CI] |
|---|---|---|---|---|---|
| `base_qwen35_0.8b` | `0.6587` [`0.5347`, `0.7778`] | `0.5125` [`0.5000`, `0.5375`] | `0.3958` [`0.3125`, `0.4792`] | `0.5833` [`0.5000`, `0.7500`] | `0.5000` [`0.3833`, `0.6333`] |
| `mit_wsp_0.8b` | `0.6684` [`0.5462`, `0.7853`] | `0.5125` [`0.5000`, `0.5375`] | `0.3958` [`0.3125`, `0.4792`] | `0.5833` [`0.5000`, `0.7500`] | `0.5000` [`0.3833`, `0.6167`] |
| `mapika_decider_0.8b` | `1.0000` [`1.0000`, `1.0000`] | `0.9750` [`0.9375`, `1.0000`] | `0.6667` [`0.5625`, `0.7708`] | `0.5000` [`0.3333`, `0.7500`] | `0.8333` [`0.7333`, `0.9167`] |
| `ours_track2_wsp` | `0.8347` [`0.7416`, `0.9147`] | `0.7250` [`0.6250`, `0.8125`] | `0.7292` [`0.6250`, `0.8333`] | `0.5833` [`0.3333`, `0.8333`] | `0.5000` [`0.3833`, `0.6167`] |
| `ours_track3_soup_bf16` | `1.0000` [`1.0000`, `1.0000`] | `0.9875` [`0.9625`, `1.0000`] | `0.9375` [`0.8750`, `1.0000`] | `0.9167` [`0.7500`, `1.0000`] | `0.7833` [`0.6833`, `0.8833`] |
| `ours_track3_soup_q8` | `1.0000` [`1.0000`, `1.0000`] | `0.9875` [`0.9625`, `1.0000`] | `0.9375` [`0.8750`, `1.0000`] | `0.9167` [`0.7500`, `1.0000`] | `0.7833` [`0.6833`, `0.8833`] |

## 2. Adversarial Red-Team (`n=160` + `32` Leaky) — 95% CIs

| Model | AUROC [95% CI] | Overall Acc [95% CI] | Overall ASR [95% CI] | Benign FPR [95% CI] | Blind ASR -> Leaky ASR (McNemar `p`) |
|---|---|---|---|---|---|
| `base_qwen35_0.8b` | `0.4357` [`0.3377`, `0.5346`] | `0.3312` [`0.2687`, `0.4000`] | `0.8103` [`0.7414`, `0.8793`] | `0.2955` [`0.1591`, `0.4318`] | `0.8750` -> `0.6562` (`p=0.03906`) |
| `mit_wsp_0.8b` | `0.4351` [`0.3370`, `0.5357`] | `0.3375` [`0.2687`, `0.4062`] | `0.7759` [`0.6983`, `0.8448`] | `0.3636` [`0.2273`, `0.5000`] | `0.8438` -> `0.8750` (`p=1`) |
| `mapika_decider_0.8b` | `0.7329` [`0.6271`, `0.8288`] | `0.7625` [`0.7000`, `0.8250`] | `0.1638` [`0.1034`, `0.2328`] | `0.4318` [`0.2955`, `0.5682`] | `0.1562` -> `0.6875` (`p=1.526e-05`) |
| `ours_track2_wsp` | `0.7033` [`0.6092`, `0.7905`] | `0.7250` [`0.6813`, `0.7688`] | `0.0603` [`0.0259`, `0.1034`] | `0.8409` [`0.7500`, `0.9318`] | `0.0312` -> `0.1250` (`p=0.25`) |
| `ours_track3_soup_bf16` | `0.9719` [`0.9361`, `0.9957`] | `0.9187` [`0.8812`, `0.9563`] | `0.0172` [`0.0000`, `0.0431`] | `0.2500` [`0.1364`, `0.3636`] | `0.0000` -> `0.0312` (`p=1`) |
| `ours_track3_soup_q8` | `0.9784` [`0.9534`, `0.9957`] | `0.9187` [`0.8812`, `0.9563`] | `0.0172` [`0.0000`, `0.0431`] | `0.2500` [`0.1364`, `0.3636`] | `0.0000` -> `0.0312` (`p=1`) |

## 3. Primary Pairwise Hypothesis Tests (Exact McNemar / Paired Bootstrap + Holm-Bonferroni, m=16)

Holm family (`m=16`): pairs=['ours_track2_wsp vs base_qwen35_0.8b', 'ours_track3_soup_bf16 vs base_qwen35_0.8b', 'ours_track3_soup_bf16 vs ours_track2_wsp', 'ours_track3_soup_bf16 vs mapika_decider_0.8b'], metrics=['frozen_real_cf_auroc_diff', 'frozen_consent_5p_mcnemar', 'frozen_all200_mcnemar', 'adv_acc_mcnemar']

| Comparison (`A` vs `B`) | `frozen real_cf` ΔAUROC (`p_holm`) | `frozen consent_5p` ΔAcc (`p_holm`) | `frozen All-200` ΔAcc (`p_holm`) | `adv Overall` ΔAcc (`p_holm`) |
|---|---|---|---|---|
| `ours_track2_wsp` vs `base_qwen35_0.8b` | `+0.1759` (`p=0.01`) | `+0.3333` (`p=0.02539`) | `+0.1650` (`p=0.001845`) | `+0.3937` (`p=2.681e-08`) |
| `ours_track3_soup_bf16` vs `base_qwen35_0.8b` | `+0.3413` (`p=0.0018`) | `+0.5417` (`p=9.548e-06`) | `+0.4250` (`p=2.611e-18`) | `+0.5875` (`p=1.158e-19`) |
| `ours_track3_soup_bf16` vs `ours_track2_wsp` | `+0.1653` (`p=0.0018`) | `+0.2083` (`p=0.02539`) | `+0.2600` (`p=1.264e-10`) | `+0.1937` (`p=1.48e-06`) |
| `ours_track3_soup_bf16` vs `mapika_decider_0.8b` | `+0.0000` (`p=1`) | `+0.2708` (`p=0.005859`) | `+0.0800` (`p=0.02539`) | `+0.1562` (`p=0.000413`) |

## 3b. Secondary / Reference Comparisons (Unadjusted Two-Sided Raw p-values)

| Comparison (`A` vs `B`) | `frozen real_cf` ΔAUROC (`p`) | `frozen consent_5p` ΔAcc (`p`) | `frozen replay` ΔAcc (`p`) | `frozen All-200` ΔAcc (`p`) | `adv Overall` ΔAcc (`p`) |
|---|---|---|---|---|---|
| `ours_track2_wsp` vs `mapika_decider_0.8b` | `-0.1653` (`p=0.0002`) | `+0.0625` (`p=0.6476`) | `-0.3333` (`p=8.798e-05`) | `-0.1800` (`p=4.369e-05`) | `-0.0375` (`p=0.4614`) |
| `mit_wsp_0.8b` vs `base_qwen35_0.8b` | `+0.0097` (`p=0.6417`) | `+0.0000` (`p=1`) | `+0.0000` (`p=1`) | `+0.0000` (`p=1`) | `+0.0063` (`p=1`) |
| `ours_track3_soup_bf16` vs `ours_track3_soup_q8` | `+0.0000` (`p=1`) | `+0.0000` (`p=1`) | `+0.0000` (`p=1`) | `+0.0000` (`p=1`) | `+0.0000` (`p=1`) |

## 4. Fine-Grained Error Breakdown (Summary)

- **`mit_wsp_0.8b`**: `frozen` binary errs=`73` ({'real_cf:cf_vcs_destructive:block->allow': 4, 'real_cf:cf_deletion:block->allow': 4, 'real_cf:cf_exfil:block->allow': 4, 'real_cf:cf_untrusted_exec:block->allow': 4, 'real_cf:cf_instruction_poisoning:block->allow': 4, 'real_cf:cf_network_exposure:block->allow': 4, 'real_cf:cf_prod_infra:block->allow': 4, 'real_cf:cf_ci_bypass:block->allow': 4, 'real_cf:cf_credential_exploration:block->allow': 3, 'real_cf:cf_containment_escape:block->allow': 4, 'consent_5p:p1_no_consent:block->allow': 8, 'consent_5p:p3_target_mismatch:block->allow': 7, 'consent_5p:p4_vague_consent:block->allow': 5, 'consent_5p:p5b_blocked_swapped_target:block->allow': 8, 'consent_5p:p2_exact_consent:allow->block': 1, 'long_50t:long_50t_target_mismatch:block->allow': 5}), replay errs=`30`; `adversarial` errs=`106` ({'obfuscation_evasion': 18, 'arg_prompt_injection': 21, 'scope_creep_smuggling': 15, 'benign_hard_negatives': 13, 'multilingual_paraphrase': 12, 'reasoning_blind_ablation': 27})
- **`mapika_decider_0.8b`**: `frozen` binary errs=`24` ({'real_cf:real_benign:allow->block': 2, 'consent_5p:p2_exact_consent:allow->block': 5, 'consent_5p:p5a_blocked_approved_same:allow->block': 6, 'consent_5p:p4_vague_consent:block->allow': 2, 'consent_5p:p1_no_consent:block->allow': 2, 'consent_5p:p3_target_mismatch:block->allow': 1, 'long_50t:long_50t_exact_target:allow->block': 5, 'long_50t:long_50t_target_mismatch:block->allow': 1}), replay errs=`10`; `adversarial` errs=`38` ({'obfuscation_evasion': 6, 'arg_prompt_injection': 3, 'scope_creep_smuggling': 5, 'benign_hard_negatives': 11, 'multilingual_paraphrase': 8, 'reasoning_blind_ablation': 5})
- **`ours_track2_wsp`**: `frozen` binary errs=`40` ({'real_cf:cf_vcs_destructive:block->allow': 1, 'real_cf:cf_instruction_poisoning:block->allow': 2, 'real_cf:real_benign:allow->block': 14, 'real_cf:cf_prod_infra:block->allow': 1, 'real_cf:cf_containment_escape:block->allow': 1, 'real_cf:cf_exfil:block->allow': 1, 'real_cf:cf_network_exposure:block->allow': 1, 'real_cf:cf_credential_exploration:block->allow': 1, 'consent_5p:p5b_blocked_swapped_target:block->allow': 3, 'consent_5p:p3_target_mismatch:block->allow': 1, 'consent_5p:p2_exact_consent:allow->block': 5, 'consent_5p:p5a_blocked_approved_same:allow->block': 4, 'long_50t:long_50t_exact_target:allow->block': 1, 'long_50t:long_50t_target_mismatch:block->allow': 4}), replay errs=`30`; `adversarial` errs=`44` ({'arg_prompt_injection': 5, 'scope_creep_smuggling': 1, 'benign_hard_negatives': 30, 'multilingual_paraphrase': 7, 'reasoning_blind_ablation': 1})
- **`ours_track3_soup_bf16`**: `frozen` binary errs=`5` ({'real_cf:real_benign:allow->block': 1, 'consent_5p:p4_vague_consent:block->allow': 1, 'consent_5p:p2_exact_consent:allow->block': 2, 'long_50t:long_50t_exact_target:allow->block': 1}), replay errs=`13`; `adversarial` errs=`13` ({'obfuscation_evasion': 2, 'benign_hard_negatives': 11})
- **`ours_track3_soup_q8`**: `frozen` binary errs=`5` ({'real_cf:real_benign:allow->block': 1, 'consent_5p:p4_vague_consent:block->allow': 1, 'consent_5p:p2_exact_consent:allow->block': 2, 'long_50t:long_50t_exact_target:allow->block': 1}), replay errs=`13`; `adversarial` errs=`13` ({'obfuscation_evasion': 2, 'benign_hard_negatives': 11})
