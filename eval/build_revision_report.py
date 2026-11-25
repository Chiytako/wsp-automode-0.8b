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

    return "\n".join(lines) + "\n"


def main() -> None:
    md = build_markdown()
    DST_MD.write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
