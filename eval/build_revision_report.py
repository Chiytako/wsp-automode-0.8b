"""Generate Markdown summary report `experiments/paper_revision_controls.md` verbatim from JSON."""

from __future__ import annotations

import json
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent.parent / "experiments"
SRC_JSON = EXP_DIR / "paper_revision_controls.json"
DST_MD = EXP_DIR / "paper_revision_controls.md"


def build_markdown() -> str:
    data = json.loads(SRC_JSON.read_text(encoding="utf-8"))
    lines: list[str] = [
        "# Paper Revision Controlled Experiments (`experiments/paper_revision_controls.json`)",
        "",
        "## R2a. K-Quant Superblock Margin Diagnostics (`select_dev`, n=200 vs. PyTorch `bf16`)",
        "",
        "| Model | Format | `p_block` MAE | Pearson `r` | Agree | `rc` Flip | Mean `Δm` (Quant−PT) | Mean `m` Allow | Mean `m` Block | `rc` Allow MAE / Block MAE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    kd = data["kquant_diagnostics"]
    for mk in ("track2_wsp", "track3_soup"):
        for q in ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"):
            e = kd[mk][q]
            ms = e["margin_stats"]
            lines.append(
                f"| `{mk}` | `{q}` | `{e['p_block_mae_normal']:.5f}` | `{e['pearson_r']:.4f}` | "
                f"`{e['agreement_total_200']:.3f}` | `{e['real_cf_flip']:.3f}` | `{ms['mean_signed_margin_shift']:+.3f}` | "
                f"`{ms['q_mean_margin_allow']:+.3f}` | `{ms['q_mean_margin_block']:+.3f}` | "
                f"`{ms['real_cf_allow_p_mae']:.4f} / {ms['real_cf_block_p_mae']:.4f}` |"
            )

    lines.extend(
        [
            "",
            "## R2b & R2d. Selective Mixed-Precision Mitigation & Causal Double-Dissociation (`select_dev`, n=200)",
            "",
            "| Config | Size (MiB) | `p_block` MAE | Pearson `r` | Agree (200) | `rc` AUROC | `rc` Rec / FPR | `rc` Flip | Proxy |",
            "|---|---:|---:|---:|---:|---:|---|---:|---:|",
        ]
    )
    t2_base = kd["track2_wsp"]
    for q in ("Q8_0", "Q4_K_M"):
        e = t2_base[q]
        lines.append(
            f"| `vanilla_{q}` | `{e['gguf_mib']:.2f}` | `{e['p_block_mae_normal']:.5f}` | `{e['pearson_r']:.4f}` | "
            f"`{e['agreement_total_200']:.3f}` | `{e['real_cf_auroc']:.4f}` | `{e['real_cf_rec']:.3f} / {e['real_cf_fpr']:.3f}` | "
            f"`{e['real_cf_flip']:.3f}` | `{e['proxy_score']:+.3f}` |"
        )
    for k, e in data["mixed_precision_mitigation"]["configs"].items():
        lines.append(
            f"| `{k}` (R2b) | `{e['gguf_mib']:.2f}` | `{e['p_block_mae_normal']:.5f}` | `{e['pearson_r']:.4f}` | "
            f"`{e['agreement_total_200']:.3f}` | `{e['real_cf_auroc']:.4f}` | `{e['real_cf_rec']:.3f} / {e['real_cf_fpr']:.3f}` | "
            f"`{e['real_cf_flip']:.3f}` | `{e['proxy_score']:+.3f}` |"
        )
    for k, e in data["kquant_double_dissociation"]["configs"].items():
        lines.append(
            f"| `{k}` (R2d) | `{e['gguf_mib']:.2f}` | `{e['p_block_mae_normal']:.5f}` | `{e['pearson_r']:.4f}` | "
            f"`{e['agreement_total_200']:.3f}` | `{e['real_cf_auroc']:.4f}` | `{e['real_cf_rec']:.3f} / {e['real_cf_fpr']:.3f}` | "
            f"`{e['real_cf_flip']:.3f}` | `{e['proxy_score']:+.3f}` |"
        )

    lines.extend(
        [
            "",
            "## R2c. Norm-Matched Multi-Neuron WSP Dispersion (`K=16` & `K=64` neurons/layer)",
            "",
            "| Variant (`K`/layer) | Mod. Params | `||ΔW||_F` | RMS | `Q8` MSE Inflation | `bf16` Proxy (`rc` AUROC) | `Q8_0` `r` (MAE) | `Q5_K_M` `r` (MAE) | `Q4_K_M` `r` (MAE / Proxy) |",
            "|---|---:|---:|---:|---:|---|---|---|---|",
        ]
    )
    for vk, v in data["multi_neuron_dispersion"].items():
        wf = v["weight_forensics"]
        bf = v["bf16_select_dev"]
        qc = v["quantization_curve"]
        lines.append(
            f"| `{vk}` ({v['neurons_per_layer']}/L) | `{wf['modified_params']}` | `{wf['total_delta_frobenius_norm']:.2f}` | "
            f"`{wf['rms_delta_per_modified_param']:.4f}` | `{wf['q8_0_forensics']['touched_block_mse_inflation_ratio']:.4f}x` | "
            f"`{bf['proxy_score']:+.3f}` (`{bf['real_cf_auroc']:.4f}`) | "
            f"`{qc['Q8_0']['pearson_r']:.4f}` (`{qc['Q8_0']['p_block_mae_normal']:.4f}`) | "
            f"`{qc['Q5_K_M']['pearson_r']:.4f}` (`{qc['Q5_K_M']['p_block_mae_normal']:.4f}`) | "
            f"`{qc['Q4_K_M']['pearson_r']:.4f}` (`{qc['Q4_K_M']['p_block_mae_normal']:.4f}` / `{qc['Q4_K_M']['proxy_score']:+.3f}`) |"
        )

    lines.extend(
        [
            "",
            "## R3. Deconfounding Gradient Controls (`frozen_test` n=200, `adversarial` n=160, `select_dev` Quant)",
            "",
            "| Control | Trainable Params | Mod. Elements (`Q8` MSE) | `frozen` Proxy | `frozen rc` AUROC (Rec/FPR) | `frozen c5p` (P2/P3) | `adv` ASR / Benign FPR (Tier-2 ASR) | `Q4_K_M` `r` (Agree / MAE) |",
            "|---|---:|---|---:|---|---|---|---|",
        ]
    )
    for ck in ("ctrl_sparse_8neuron_grad", "ctrl_lora_r1_same6", "ctrl_lora_r1_depth8"):
        c = data["deconfound_controls"][ck]
        wf = c["weight_forensics"]
        fr = c["frozen_test_summary"]
        ad = c["adversarial_summary"]
        q4 = c["quantization_curve"]["Q4_K_M"]
        lines.append(
            f"| `{ck}` | `{c['trainable_params']}` | `{wf['modified_params']}` (`{wf['q8_0_forensics']['touched_block_mse_inflation_ratio']:.4f}x`) | "
            f"`{fr['proxy_score']:+.3f}` | `{fr['real_cf_auroc']:.4f}` (`{fr['real_cf_rec']:.3f}/{fr['real_cf_fpr']:.3f}`) | "
            f"`{fr['consent_5p_acc']:.3f}` (`{fr['consent_5p_p2']:.3f}/{fr['consent_5p_p3']:.3f}`) | "
            f"`{ad['overall_asr']:.3f}` / `{ad['benign_fpr']:.3f}` (`{ad['tier2_file_edit_asr']:.3f}`) | "
            f"`{q4['pearson_r']:.4f}` (`{q4['agreement_total_200']:.3f}` / `{q4['p_block_mae_normal']:.4f}`) |"
        )

    lines.extend(
        [
            "",
            "## R4. Direct In-Situ Warm-Start Causality & Interventions (`opt_step 0..20`)",
            "",
            "| Condition | Peak 20-Grp Loss (Single Peak) | Max Grad Norm | End `h15_cos` | Live L15 `|SwiGLU|` (Step 0 -> 20) | Step-20 `select_dev` Proxy (`rc` AUROC / `c5p` Acc) |",
            "|---|---:|---:|---:|---|---|",
        ]
    )
    for wk in ("t3a_insitu_base", "t3b_insitu_full", "t3b_lowlr_5x", "t3b_freeze_L0_14", "t3b_freeze_t2gates"):
        w = data["warmstart_causality"][wk]
        snaps = w["step_snapshots"]
        max_gn = max((s["last_grad_norm"] or 0.0) for s in snaps)
        s0, s20 = snaps[0], snaps[-1]
        dv = w["step20_dev_summary"]
        lines.append(
            f"| `{wk}` | `{w['peak_recent20_loss']:.2f}` (`{w['peak_group_loss']:.2f}`) | `{max_gn:.2f}` | "
            f"`{s20['h15_cos_vs_base_mean']:.6f}` | `{s0['l15_live_swiglu_mean_abs']:.4f}` -> `{s20['l15_live_swiglu_mean_abs']:.4f}` | "
            f"`{dv['proxy_score']:+.3f}` (`{dv['real_cf_auroc']:.4f}` / `{dv['consent_5p_acc']:.4f}`) |"
        )

    if "r5_base_and_bootstrap_cis" in data:
        r5a = data["r5_base_and_bootstrap_cis"]
        lines.extend(
            [
                "",
                "## R5a. Unmodified Base GGUF Curve, Raw/Normalized `|Δm|`, and B=10,000 Bootstrap 95% CIs (`select_dev`)",
                "",
                "| Key | `bf16 |m|` | Raw `|Δm|` [95% CI] | Norm `|Δm|/|m|` [95% CI] | Pearson `r` [95% CI] | Agree [95% CI] | Signed Bias |",
                "|---|---:|---|---|---|---|---:|",
            ]
        )
        for ck, c in r5a["configs"].items():
            b = c["bootstrap_ci95"]
            mc, nc, rc, ac = b["margin_mae"], b["margin_mae_norm"], b["pearson_r"], b["agreement_total_200"]
            lines.append(
                f"| `{ck}` | `{c['pt_mean_abs_margin']:.3f}` | "
                f"`{mc['point']:.3f}` [`{mc['ci95_lo']:.3f}`, `{mc['ci95_hi']:.3f}`] | "
                f"`{nc['point']:.3f}` [`{nc['ci95_lo']:.3f}`, `{nc['ci95_hi']:.3f}`] | "
                f"`{rc['point']:.3f}` [`{rc['ci95_lo']:.3f}`, `{rc['ci95_hi']:.3f}`] | "
                f"`{ac['point']:.3f}` [`{ac['ci95_lo']:.3f}`, `{ac['ci95_hi']:.3f}`] | "
                f"`{c['signed_margin_bias']:+.3f}` |"
            )
        lines.extend(
            [
                "",
                "### Paired Bootstrap Differences (`B=10,000`)",
                "",
                "| Comparison (`A - B`) | Metric | Δ [95% CI] | Two-Sided `p` |",
                "|---|---|---|---:|",
            ]
        )
        for pd in r5a["paired_differences"]:
            lines.append(
                f"| `{pd['a']}` - `{pd['b']}` | `{pd['metric']}` | "
                f"`{pd['delta_a_minus_b']:+.4f}` [`{pd['ci95_lo']:+.4f}`, `{pd['ci95_hi']:+.4f}`] | `{pd['two_sided_p']}` |"
            )

    if "r5_activation_amplification" in data:
        r5b = data["r5_activation_amplification"]
        lines.extend(
            [
                "",
                "## R5b. Direct 4-Bit Internal `h15`/`h22` Drift & SwiGLU Gate Amplification (`select_dev` n=140, `gguf.dequantize`)",
                "",
                f"- Sanity check Track 2 BF16 GGUF vs. Native PT: `r={r5b['sanity_t2_gguf_bf16_vs_native_pt']['pearson_r_margin']:.6f}`, `mean|Δm|={r5b['sanity_t2_gguf_bf16_vs_native_pt']['mean_abs_margin_diff']:.4f}`, `max|Δm|={r5b['sanity_t2_gguf_bf16_vs_native_pt']['max_abs_margin_diff']:.4f}`",
                f"- Sanity check Track 3 BF16 GGUF vs. Native PT: `r={r5b['sanity_t3_gguf_bf16_vs_native_pt']['pearson_r_margin']:.6f}`, `max|Δm|={r5b['sanity_t3_gguf_bf16_vs_native_pt']['max_abs_margin_diff']:.4f}`",
                "",
                "| Condition | `h15` Rel-L2 (cos) | L15 Edited-4 `|Δg|` (Amp vs Untouched / vs Base) | L22 Edited-4 `|Δg|` (Amp vs Untouched / vs Base) | Raw `|Δm|` (`r_p`, `r_llama`) |",
                "|---|---|---|---|---|",
            ]
        )
        for cname, cv in r5b["conditions"].items():
            l15, l22, lg = cv["l15"], cv["l22"], cv["logits"]
            b15 = f"{l15['amplification_vs_base_Q4_same4']:.1f}x" if "amplification_vs_base_Q4_same4" in l15 else "-"
            b22 = f"{l22['amplification_vs_base_Q4_same4']:.1f}x" if "amplification_vs_base_Q4_same4" in l22 else "-"
            rl = f"{lg['pearson_r_p_vs_llama_cpp']:.3f}" if "pearson_r_p_vs_llama_cpp" in lg else "-"
            lines.append(
                f"| `{cname}` | `{l15['h_rel_l2_drift']:.3f}` (`{l15['h_cos_vs_bf16']:.4f}`) | "
                f"`{l15['edited4_mean_abs_delta']:.4f}` (`{l15['amplification_vs_untouched_mean']:.1f}x` / {b15}) | "
                f"`{l22['edited4_mean_abs_delta']:.4f}` (`{l22['amplification_vs_untouched_mean']:.1f}x` / {b22}) | "
                f"`{lg['margin_mae']:.3f}` (`{lg['pearson_r_p_block']:.3f}`, `{rl}`) |"
            )

    def _ci_row(cfg_key: str, c: dict) -> str:
        b = c["bootstrap_ci95"]
        mc, nc, rc, ac = b["margin_mae"], b["margin_mae_norm"], b["pearson_r"], b["agreement_total_200"]
        return (
            f"| `{cfg_key}` | `{c['gguf_mib']:.2f}` | "
            f"`{mc['point']:.4f}` [`{mc['ci95_lo']:.3f}`, `{mc['ci95_hi']:.3f}`] | "
            f"`{nc['point']:.4f}` | "
            f"`{rc['point']:.4f}` [`{rc['ci95_lo']:.3f}`, `{rc['ci95_hi']:.3f}`] | "
            f"`{ac['point']:.3f}` | `{c['signed_margin_bias']:+.4f}` | "
            f"`{c.get('real_cf_auroc', 0.0):.4f}` (`{c.get('real_cf_fpr', 0.0):.3f}` / `{c.get('real_cf_flip', 0.0):.3f}`) |"
        )

    if "r6a_kquant_q6_q5_investigation" in data:
        r6a = data["r6a_kquant_q6_q5_investigation"]
        lines.extend(
            [
                "",
                "## R6a. K-Quant Tensor Inventory & Q6_K/Q5_K_M Resolution (`select_dev`, n=200 Server + n=140 Probes)",
                "",
                "| Variant | Size (MiB) | Tensor Counts | Edited Tensors | `token_embd` |",
                "|---|---:|---|---|---|",
            ]
        )
        for ik, iv in r6a["gguf_tensor_inventory"].items():
            cnts = ", ".join(f"{t}={n}" for t, n in iv["tensor_type_counts"].items())
            eds = ", ".join(f"{k.replace('blk.', 'L').replace('.weight', '')}={v}" for k, v in iv["track2_edited_tensor_types"].items())
            lines.append(f"| `{ik}` | `{iv['gguf_mib']:.2f}` | {cnts} | {eds} | `{iv['token_embd_type']}` |")

        lines.extend(
            [
                "",
                "| Server Config | Size (MiB) | Raw `|Δm|` [95% CI] | Norm `|Δm|/|m|` | Pearson `r` [95% CI] | Agree | Signed Bias | `rc` AUROC (FPR / Flip) |",
                "|---|---:|---|---:|---|---:|---:|---|",
            ]
        )
        for ck, cv in r6a["server_configs"].items():
            lines.append(_ci_row(ck, cv))
        lines.append("")
        lines.extend(["### R6a Paired Bootstrap Differences", "", "| Comparison (`A - B`) | Metric | Δ [95% CI] | `p` |", "|---|---|---|---:|"])
        for pd in r6a["paired_differences"]:
            lines.append(f"| `{pd['a']}` - `{pd['b']}` | `{pd['metric']}` | `{pd['delta_a_minus_b']:+.4f}` [`{pd['ci95_lo']:+.4f}`, `{pd['ci95_hi']:+.4f}`] | `{pd['two_sided_p']}` |")

        lines.extend(
            [
                "",
                "| Dequant Probe (`n=140`) | `h15` Rel-L2 | L15 `Δg` (Mean / `|Δg|` / Rel) | L22 `Δg` (Mean / `|Δg|` / Rel) | Unt. L15 Rel | Raw `|Δm|` (`bias`) | `r_p` (`r_llama`) |",
                "|---|---:|---|---|---:|---|---|",
            ]
        )
        for dk, dv in r6a["dequant_probes"].items():
            l15, l22, lg = dv["l15"], dv["l22"], dv["logits"]
            rl = f"{lg['pearson_r_p_vs_llama_cpp']:.4f}" if lg.get("pearson_r_p_vs_llama_cpp") is not None else "-"
            lines.append(
                f"| `{dk}` | `{l15['h_rel_l2_drift']:.4f}` | "
                f"`{l15['edited4_signed_mean_delta']:+.4f}` / `{l15['edited4_mean_abs_delta']:.4f}` / `{l15['edited4_rel_shift_dg_over_g']:.4f}` | "
                f"`{l22['edited4_signed_mean_delta']:+.4f}` / `{l22['edited4_mean_abs_delta']:.4f}` / `{l22['edited4_rel_shift_dg_over_g']:.4f}` | "
                f"`{l15['untouched_rel_shift_dg_over_g']:.4f}` | `{lg['margin_mae']:.4f}` (`{lg['signed_margin_shift']:+.4f}`) | "
                f"`{lg['pearson_r_p_block']:.4f}` (`{rl}`) |"
            )

    if "r6b_jacobian_and_margin_decomp" in data:
        r6b = data["r6b_jacobian_and_margin_decomp"]
        lines.extend(
            [
                "",
                "## R6b. First-Order SwiGLU Input Jacobian Theory & Causal 8-Neuron Margin Attribution",
                "",
                "| Model / Layer | Edited `||w_gate||` | Edited `||w_up||` | Mean `||∇_h g_j||_2` (p95) | Untouched `||∇_h g_j||_2` | Ratio vs. Untouched |",
                "|---|---:|---:|---|---:|---:|",
            ]
        )
        for mk, mv in r6b["jacobian_stats"].items():
            for lk, lv in mv.items():
                lines.append(
                    f"| `{mk}` `{lk}` | `{lv['edited_wg_row_norm_mean']:.4f}` | `{lv['edited_wu_row_norm_mean']:.4f}` | "
                    f"`{lv['edited_jac_norm_mean']:.4f}` (`{lv['edited_jac_norm_p95']:.4f}`) | `{lv['untouched_jac_norm_mean']:.4f}` | `{lv['jac_ratio_vs_untouched']:.2f}x` |"
                )

        lines.extend(
            [
                "",
                "| Format | L15 Taylor `r` (MAE / Pred vs. True `|Δg|`) | L22 Taylor `r` (MAE / Pred vs. True `|Δg|`) |",
                "|---|---|---|",
            ]
        )
        for lk, lv in r6b["linearized_prediction_validation"].items():
            l15, l22 = lv["l15"], lv["l22"]
            lines.append(
                f"| `{lk}` | `{l15['pearson_r_lin_vs_true']:.4f}` (`{l15['mae_lin_vs_true']:.4f}` / `{l15['mean_abs_dg_pred']:.4f}` vs `{l15['mean_abs_dg_true']:.4f}`) | "
                f"`{l22['pearson_r_lin_vs_true']:.4f}` (`{l22['mae_lin_vs_true']:.4f}` / `{l22['mean_abs_dg_pred']:.4f}` vs `{l22['mean_abs_dg_true']:.4f}`) |"
            )

        lines.extend(
            [
                "",
                "| Format (`n=140`) | Full Q `|Δm|` (`bias`) | Inject Q L15 Ed4 (Bias) | Inject Q Ed8 `|Δm|` (`r` vs Full Q) | Rescue Ed8 -> Residual `|Δm|` (`bias`) | Drift Removed (%) |",
                "|---|---:|---|---|---|---|",
            ]
        )
        for fk, fv in r6b["causal_8neuron_margin_attribution"].items():
            lines.append(
                f"| `{fk}` | `{fv['full_q_margin_mae']:.4f}` (`{fv['full_q_signed_bias']:+.4f}`) | "
                f"`{fv['bf16_inject_q_L15_ed4_mae']:.4f}` (`{fv['bf16_inject_q_L15_ed4_bias']:+.4f}`) | "
                f"`{fv['bf16_inject_q_L15_L22_ed8_mae']:.4f}` (`{fv['bf16_inject_q_ed8_r_vs_full_q']:.4f}`) | "
                f"`{fv['q_rescue_bf16_ed8_mae']:.4f}` (`{fv['q_rescue_bf16_ed8_bias']:+.4f}`) | `{fv['pct_total_abs_m_drift_removed_by_8neu_rescue']:.1f}%` |"
            )

    if "r6c_bit_matched_controls" in data:
        r6c = data["r6c_bit_matched_controls"]
        lines.extend(
            [
                "",
                "## R6c. Byte-Budget-Matched K-Quant Allocation Controls (`select_dev`, n=200)",
                "",
                "| Config | Size (MiB) | Raw `|Δm|` [95% CI] | Norm `|Δm|/|m|` | Pearson `r` [95% CI] | Agree | Signed Bias | `rc` AUROC (FPR / Flip) |",
                "|---|---:|---|---:|---|---:|---:|---|",
            ]
        )
        for ck, cv in r6c["configs"].items():
            lines.append(_ci_row(ck, cv))
        lines.extend(["", "| Comparison (`A - B`) | Metric | Δ [95% CI] | `p` |", "|---|---|---|---:|"])
        for pd in r6c["paired_differences"]:
            lines.append(f"| `{pd['a']}` - `{pd['b']}` | `{pd['metric']}` | `{pd['delta_a_minus_b']:+.4f}` [`{pd['ci95_lo']:+.4f}`, `{pd['ci95_hi']:+.4f}`] | `{pd['two_sided_p']}` |")

    if "r6d_warmstart_fair_grad_and_backprop" in data:
        r6d = data["r6d_warmstart_fair_grad_and_backprop"]
        lines.extend(
            [
                "",
                f"## R6d. Fair Matched-Subset Gradients (`shared_L15_23`) & Backward `dL/dh15` Probe (`n_diag={r6d['n_diag_items']}`)",
                "",
                "| Condition | Total Params | `shared_L15_23` Params | Max Total Grad | Max `shared_L15_23` | Step-5 `shared_L15_23` | Max `dL/dh15_via_mlp` | Step 0 / 5 / 10 `dL/dh15` |",
                "|---|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for ck, cv in r6d["conditions"].items():
            s5 = cv["step_grad_logs"][4]["shared_L15_23_grad_norm"]
            bp_map = {b["opt_step"]: b["rms_dL_dh15_via_mlp"] for b in cv["backward_snapshots"]}
            lines.append(
                f"| `{ck}` | `{cv['n_trainable_total']}` | `{cv['n_shared_L15_23_params']}` | "
                f"`{cv['max_total_grad_norm']:.2f}` | `{cv['max_shared_L15_23_grad_norm']:.2f}` | `{s5:.2f}` | "
                f"`{cv['max_rms_dL_dh15_via_mlp']:.4f}` | `{bp_map[0]:.4f}` / `{bp_map[5]:.4f}` / `{bp_map[10]:.4f}` |"
            )

    if "r6e_multi_edit_generality" in data:
        r6e = data["r6e_multi_edit_generality"]
        t1 = r6e["track1_L15_swiglu"]
        mit = r6e["mit_wsp_down_proj_only"]
        lines.extend(
            [
                "",
                "## R6e. Multi-Edit Structural Generality (Prediction P2: Track 1 SwiGLU vs. MIT WSP `down_proj`-Only)",
                "",
                f"- **Track 1 (`L15` 4-neuron SwiGLU)**: `||∇_h g_j||_2 = {t1['jacobian_l15']['edited_jac_norm_mean']:.4f}` (`p95={t1['jacobian_l15']['edited_jac_norm_p95']:.4f}`, `{t1['jacobian_l15']['jac_ratio_vs_untouched']:.2f}x` untouched `{t1['jacobian_l15']['untouched_jac_norm_mean']:.4f}`).",
                "",
                "| Track 1 Probe (`n=140`) | `h15` Rel-L2 | L15 `Δg` (`mean` / `|Δg|` / Rel) | Raw `|Δm|` (`bias`) | `r_p` |",
                "|---|---:|---|---:|---:|",
            ]
        )
        for pk, pv in t1["dequant_probes"].items():
            l15, lg = pv["l15"], pv["logits"]
            lines.append(
                f"| `{pk}` | `{l15['h_rel_l2_drift']:.4f}` | `{l15['edited4_signed_mean_delta']:+.4f}` / `{l15['edited4_mean_abs_delta']:.4f}` / `{l15['edited4_rel_shift_dg_over_g']:.4f}` | "
                f"`{lg['margin_mae']:.4f}` (`{lg['signed_margin_shift']:+.4f}`) | `{lg['pearson_r_p_block']:.4f}` |"
            )
        lines.extend(
            [
                "",
                f"- **MIT WSP (`L15` & `L19` `down_proj`-only, 64 cols each)**: L15 `||∇_h g_j||_2 = {mit['jacobian_l15_edited64']['edited_jac_norm_mean']:.4f}` (`{mit['jacobian_l15_edited64']['jac_ratio_vs_untouched']:.2f}x`), L19 `||∇_h g_j||_2 = {mit['jacobian_l19_edited64']['edited_jac_norm_mean']:.4f}` (`{mit['jacobian_l19_edited64']['jac_ratio_vs_untouched']:.2f}x`); Q4_K_M edited `|Δg|` = `{mit['l15_edited64_mean_abs_dg_q4']:.5f}` (`rel={mit['l15_edited64_rel_shift_dg_over_g_q4']:.4f}`) and `{mit['l19_edited64_mean_abs_dg_q4']:.5f}` (`rel={mit['l19_edited64_rel_shift_dg_over_g_q4']:.4f}`); `select_dev Q4_K_M` `raw|Δm|={mit['select_dev_q4']['logits']['margin_mae']:.4f}` (`r_p={mit['select_dev_q4']['logits']['pearson_r_p_block']:.4f}`).",
                "",
                "| `GENERAL_CORPUS` (2,026 tokens) | Perplexity | Mean KL vs. BF16 | Mean Logprob L1 vs. BF16 |",
                "|---|---:|---:|---:|",
            ]
        )
        for gk, gv in mit["general_corpus_quant_parity"].items():
            lines.append(f"| `{gk}` | `{gv['perplexity']:.4f}` | `{gv['mean_kl_vs_bf16']:.6f}` | `{gv['mean_logprob_l1_vs_bf16']:.5f}` |")

    return "\n".join(lines) + "\n"


def main() -> None:
    md = build_markdown()
    DST_MD.write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
