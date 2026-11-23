# Track 2 Multi-Trajectory WSP Comparison (`select_dev`, n=200)

| Trajectory | Steps Kept | `% modified` | Proxy Score | `real_cf` AUROC (Rec/FPR) | `consent_5p` Acc (`P2/P3`) | `long_50t` Acc | `decider_replay` Acc / ECE |
|---|---|---:|---:|---|---|---:|---|
| `track1_baseline` | `1` (`step1b_swiglu_order_gate_L15`) | `0.001633%` | `0.40678` | `0.8512` (`0.7 / 0.2`) | `0.7292` (`0.125 / 1.0`) | `0.5` | `0.55 / 0.1366` |
| `traj1_residual_boost_from_t1` | `2` (`step1b_swiglu_order_gate_L15, traj1_iter3_balanced_swiglu_L23`) | `0.003266%` | `0.55077` | `0.8678` (`0.625 / 0.175`) | `0.7708` (`0.25 / 1.0`) | `0.5` | `0.55 / 0.1367` |
| `traj2_deltanet_long50t_from_t1` | `2` (`step1b_swiglu_order_gate_L15, traj2_iter3_order_cal_L23`) | `0.003266%` | `0.51547` | `0.8606` (`0.625 / 0.175`) | `0.7708` (`0.25 / 1.0`) | `0.5` | `0.55 / 0.1367` |
| `traj3_from_scratch_multilayer` | `2` (`traj3_iter1_domain_spec_L15, traj3_iter2_consent_spec_L22`) | `0.003264%` | `0.56208` | `0.8731` (`0.625 / 0.175`) | `0.75` (`0.375 / 0.875`) | `0.5` | `0.5333 / 0.1407` |

**Selected Winner**: `traj3_from_scratch_multilayer` -> **After LOO Pruning**: `['traj3_iter1_domain_spec_L15', 'traj3_iter2_consent_spec_L22']`
