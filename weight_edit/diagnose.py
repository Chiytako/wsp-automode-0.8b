"""Track 0 mechanistic circuit diagnostics for Qwen3.5-0.8B.

Computes:
  1. 4-bucket error taxonomy on select_dev (Keyword Reflex, Target Mismatch Blindness,
     Real-Log Allow Drift FN, Order-Swap Flip).
  2. 24-layer Logit Lens and direct sub-block (Attention/DeltaNet vs MLP) contributions.
  3. Paired Plain vs Oracle Prompt slot activation shift, cosine similarity,
     u-direction push (Block vs Allow), and 2-fold linear separability AUROC across layers 0..23.
  4. Saves JSON, Markdown, and matplotlib diagnostic plots under experiments/.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from automode.questions import Q_BLOCK_BINARY, build_oracle_prompt, build_plain_prompt
from automode.view import render_view
from eval.evaluate import load_split_rows
from eval.model_runner import QwenDeciderRunner

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = ROOT / "experiments"
PLOTS_DIR = EXPERIMENTS_DIR / "plots"


def _apply_final_rmsnorm(vec: torch.Tensor, norm_weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Exact Qwen3_5RMSNorm formula: (x / rms(x)) * (1.0 + weight)."""
    x = vec.float()
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return (x / rms) * (1.0 + norm_weight.float())


def _extract_slot_layers(runner: QwenDeciderRunner, prompt_text: str) -> dict[str, Any]:
    """Run one forward pass and capture final-slot resid, attn_out, and mlp_out across all 24 layers."""
    layers = runner.model.model.layers
    n_layers = len(layers)
    resid: list[torch.Tensor | None] = [None] * n_layers
    attn_out: list[torch.Tensor | None] = [None] * n_layers
    mlp_out: list[torch.Tensor | None] = [None] * n_layers
    handles = []

    for i, layer in enumerate(layers):
        def _make_resid_hook(idx: int):
            def _hook(_mod, _inp, out):
                t = out[0] if isinstance(out, tuple) else out
                resid[idx] = t[0, -1, :].detach().float().cpu()
            return _hook

        def _make_attn_hook(idx: int):
            def _hook(_mod, _inp, out):
                t = out[0] if isinstance(out, tuple) else out
                attn_out[idx] = t[0, -1, :].detach().float().cpu()
            return _hook

        def _make_mlp_hook(idx: int):
            def _hook(_mod, _inp, out):
                t = out[0] if isinstance(out, tuple) else out
                mlp_out[idx] = t[0, -1, :].detach().float().cpu()
            return _hook

        handles.append(layer.register_forward_hook(_make_resid_hook(i)))
        proj = layer.linear_attn.out_proj if hasattr(layer, "linear_attn") else layer.self_attn.o_proj
        handles.append(proj.register_forward_hook(_make_attn_hook(i)))
        handles.append(layer.mlp.down_proj.register_forward_hook(_make_mlp_hook(i)))

    try:
        input_ids = runner.encode_prompt(prompt_text)
        with torch.inference_mode():
            out = runner.model.model(input_ids=input_ids, use_cache=False)
            slot_h = out.last_hidden_state[:, -1:, :]
            vocab_logits = runner.model.lm_head(slot_h)[0, 0, :].float().cpu()
    finally:
        for h in handles:
            h.remove()

    id_a = runner.letter_token_ids["A"]
    id_b = runner.letter_token_ids["B"]
    return {
        "resid": torch.stack([r for r in resid if r is not None], dim=0),        # [24, 1024]
        "attn_out": torch.stack([a for a in attn_out if a is not None], dim=0),  # [24, 1024]
        "mlp_out": torch.stack([m for m in mlp_out if m is not None], dim=0),    # [24, 1024]
        "logit_no": float(vocab_logits[id_a].item()),
        "logit_yes": float(vocab_logits[id_b].item()),
        "margin": float(vocab_logits[id_b].item() - vocab_logits[id_a].item()),
    }


def _two_fold_mean_diff_auroc(feats: np.ndarray, labels: np.ndarray) -> float:
    """2-fold cross-validated mean-difference linear probe AUROC (no label leakage)."""
    if len(np.unique(labels)) < 2 or len(labels) < 6:
        return 0.5
    half = len(labels) // 2
    scores = np.zeros(len(labels), dtype=np.float64)
    for train_idx, test_idx in [(np.arange(0, half), np.arange(half, len(labels))), (np.arange(half, len(labels)), np.arange(0, half))]:
        x_tr, y_tr = feats[train_idx], labels[train_idx]
        x_te = feats[test_idx]
        if len(np.unique(y_tr)) < 2:
            return 0.5
        mu_pos = x_tr[y_tr == 1].mean(axis=0)
        mu_neg = x_tr[y_tr == 0].mean(axis=0)
        d = mu_pos - mu_neg
        nrm = np.linalg.norm(d)
        if nrm > 1e-8:
            d = d / nrm
        scores[test_idx] = x_te @ d
    return float(roc_auc_score(labels, scores))


def run_diagnostics(observe_limit: int = 148, dev_limit: int = 128) -> dict[str, Any]:
    t0 = time.perf_counter()
    runner = QwenDeciderRunner()
    norm_w = runner.model.model.norm.weight.detach().float().cpu()
    norm_eps = float(getattr(runner.model.model.norm, "variance_epsilon", 1e-6))
    w_lm = runner.model.lm_head.weight.detach().float().cpu()
    id_no = runner.letter_token_ids["A"]
    id_yes = runner.letter_token_ids["B"]
    u_vec = w_lm[id_yes] - w_lm[id_no]  # [1024]

    # ---- Part A: select_dev (4-bucket errors + 24-layer Logit Lens)
    dev_rows = [
        r for r in load_split_rows("select_dev")
        if r["subset"] in ("real_cf", "consent_5p")
    ][:dev_limit]

    buckets = {
        "bucket_1_keyword_reflex_p2_p5a_total": 0,
        "bucket_1_keyword_reflex_errors": 0,
        "bucket_2_target_mismatch_p3_p5b_total": 0,
        "bucket_2_target_mismatch_errors": 0,
        "bucket_3_real_cf_block_total": 0,
        "bucket_3_real_cf_allow_drift_fn": 0,
        "bucket_4_order_flip_total": 0,
        "bucket_4_order_flips": 0,
    }
    dev_resid = []
    dev_attn = []
    dev_mlp = []
    dev_meta = []

    for r in dev_rows:
        ctx = render_view(r, tok=runner.tokenizer, budget=2048, include_recap=False).text
        paired = runner.score_binary_paired(ctx, mode="plain", question_spec=Q_BLOCK_BINARY)
        p_norm = build_plain_prompt(ctx, question_spec=Q_BLOCK_BINARY, reverse_options=False)
        caps = _extract_slot_layers(runner, p_norm.text)

        y = 1 if r["label"] == "block" else 0
        pat = r["pattern"]
        pred = paired["pred_normal"]

        buckets["bucket_4_order_flip_total"] += 1
        if paired["flipped"]:
            buckets["bucket_4_order_flips"] += 1
        if pat in ("p2_exact_consent", "p5a_blocked_approved_same"):
            buckets["bucket_1_keyword_reflex_p2_p5a_total"] += 1
            if pred == "block":
                buckets["bucket_1_keyword_reflex_errors"] += 1
        elif pat in ("p3_target_mismatch", "p5b_blocked_swapped_target"):
            buckets["bucket_2_target_mismatch_p3_p5b_total"] += 1
            if pred == "allow":
                buckets["bucket_2_target_mismatch_errors"] += 1
        elif r["subset"] == "real_cf" and y == 1:
            buckets["bucket_3_real_cf_block_total"] += 1
            if pred == "allow":
                buckets["bucket_3_real_cf_allow_drift_fn"] += 1

        dev_resid.append(caps["resid"])
        dev_attn.append(caps["attn_out"])
        dev_mlp.append(caps["mlp_out"])
        dev_meta.append({"subset": r["subset"], "pattern": pat, "y": y})

    resid_t = torch.stack(dev_resid, dim=0)   # [N, 24, 1024]
    attn_t = torch.stack(dev_attn, dim=0)     # [N, 24, 1024]
    mlp_t = torch.stack(dev_mlp, dim=0)       # [N, 24, 1024]
    final_rms = torch.sqrt(torch.mean(resid_t[:, -1, :] ** 2, dim=-1, keepdim=True) + norm_eps)  # [N, 1]
    scale_vec = (1.0 + norm_w) / final_rms  # [N, 1024]

    n_layers = resid_t.shape[1]
    layer_types = runner.model.config.layer_types
    logit_lens_rows = []
    rc_idx = [i for i, m in enumerate(dev_meta) if m["subset"] == "real_cf"]
    rc_y = np.array([dev_meta[i]["y"] for i in rc_idx])
    p2_idx = [i for i, m in enumerate(dev_meta) if m["pattern"] == "p2_exact_consent"]
    p3_idx = [i for i, m in enumerate(dev_meta) if m["pattern"] == "p3_target_mismatch"]

    for l in range(n_layers):
        normed_r = _apply_final_rmsnorm(resid_t[:, l, :], norm_w, norm_eps)
        lens_margin = (normed_r @ u_vec).numpy()
        attn_contrib = ((attn_t[:, l, :] * scale_vec) @ u_vec).numpy()
        mlp_contrib = ((mlp_t[:, l, :] * scale_vec) @ u_vec).numpy()

        rc_sep = float(lens_margin[rc_idx][rc_y == 1].mean() - lens_margin[rc_idx][rc_y == 0].mean()) if rc_idx else 0.0
        p23_sep = float(lens_margin[p3_idx].mean() - lens_margin[p2_idx].mean()) if (p2_idx and p3_idx) else 0.0
        rc_attn_sep = float(attn_contrib[rc_idx][rc_y == 1].mean() - attn_contrib[rc_idx][rc_y == 0].mean()) if rc_idx else 0.0
        rc_mlp_sep = float(mlp_contrib[rc_idx][rc_y == 1].mean() - mlp_contrib[rc_idx][rc_y == 0].mean()) if rc_idx else 0.0
        p23_attn_sep = float(attn_contrib[p3_idx].mean() - attn_contrib[p2_idx].mean()) if (p2_idx and p3_idx) else 0.0
        p23_mlp_sep = float(mlp_contrib[p3_idx].mean() - mlp_contrib[p2_idx].mean()) if (p2_idx and p3_idx) else 0.0

        logit_lens_rows.append(
            {
                "layer": l,
                "layer_type": layer_types[l],
                "real_cf_lens_sep": round(rc_sep, 4),
                "p3_minus_p2_lens_sep": round(p23_sep, 4),
                "real_cf_attn_contrib_sep": round(rc_attn_sep, 4),
                "real_cf_mlp_contrib_sep": round(rc_mlp_sep, 4),
                "p23_attn_contrib_sep": round(p23_attn_sep, 4),
                "p23_mlp_contrib_sep": round(p23_mlp_sep, 4),
            }
        )

    # ---- Part B: observe (Paired Plain vs Oracle representation shift across 24 layers)
    obs_rows = [
        r for r in load_split_rows("observe")
        if r["subset"] in ("real_cf", "consent_5p")
    ][:observe_limit]

    obs_plain_resid = []
    obs_oracle_resid = []
    obs_meta = []

    for r in obs_rows:
        ctx = render_view(r, tok=runner.tokenizer, budget=2048, include_recap=False).text
        p_plain = build_plain_prompt(ctx, question_spec=Q_BLOCK_BINARY, reverse_options=False).text
        p_oracle = build_oracle_prompt(ctx, question_spec=Q_BLOCK_BINARY, reverse_options=False).text
        c_plain = _extract_slot_layers(runner, p_plain)
        c_oracle = _extract_slot_layers(runner, p_oracle)
        obs_plain_resid.append(c_plain["resid"])
        obs_oracle_resid.append(c_oracle["resid"])
        obs_meta.append({"subset": r["subset"], "pattern": r["pattern"], "y": 1 if r["label"] == "block" else 0})

    op_t = torch.stack(obs_plain_resid, dim=0)   # [M, 24, 1024]
    oo_t = torch.stack(obs_oracle_resid, dim=0)  # [M, 24, 1024]
    obs_y = np.array([m["y"] for m in obs_meta])
    obs_rc_idx = [i for i, m in enumerate(obs_meta) if m["subset"] == "real_cf"]
    obs_c5_p23_idx = [i for i, m in enumerate(obs_meta) if m["pattern"] in ("p2_exact_consent", "p3_target_mismatch")]
    obs_rc_y = np.array([obs_meta[i]["y"] for i in obs_rc_idx])
    obs_p23_y = np.array([obs_meta[i]["y"] for i in obs_c5_p23_idx])

    plain_vs_oracle_rows = []
    for l in range(n_layers):
        hp = op_t[:, l, :]
        ho = oo_t[:, l, :]
        diff = ho - hp
        cos = torch.nn.functional.cosine_similarity(hp, ho, dim=-1)
        l2 = torch.linalg.norm(diff, dim=-1)
        rel = l2 / torch.linalg.norm(hp, dim=-1).clamp_min(1e-6)

        normed_p = _apply_final_rmsnorm(hp, norm_w, norm_eps)
        normed_o = _apply_final_rmsnorm(ho, norm_w, norm_eps)
        u_shift = ((normed_o - normed_p) @ u_vec).numpy()

        push_blk = float(u_shift[obs_y == 1].mean()) if np.any(obs_y == 1) else 0.0
        push_alw = float(u_shift[obs_y == 0].mean()) if np.any(obs_y == 0) else 0.0

        plain_rc_auroc = _two_fold_mean_diff_auroc(hp[obs_rc_idx].numpy(), obs_rc_y)
        oracle_rc_auroc = _two_fold_mean_diff_auroc(ho[obs_rc_idx].numpy(), obs_rc_y)
        plain_p23_auroc = _two_fold_mean_diff_auroc(hp[obs_c5_p23_idx].numpy(), obs_p23_y)
        oracle_p23_auroc = _two_fold_mean_diff_auroc(ho[obs_c5_p23_idx].numpy(), obs_p23_y)

        plain_vs_oracle_rows.append(
            {
                "layer": l,
                "layer_type": layer_types[l],
                "mean_cosine": round(float(cos.mean().item()), 4),
                "mean_l2_shift": round(float(l2.mean().item()), 4),
                "mean_rel_shift": round(float(rel.mean().item()), 4),
                "oracle_u_push_block": round(push_blk, 4),
                "oracle_u_push_allow": round(push_alw, 4),
                "oracle_net_sep_shift": round(push_blk - push_alw, 4),
                "plain_rc_probe_auroc": round(plain_rc_auroc, 4),
                "oracle_rc_probe_auroc": round(oracle_rc_auroc, 4),
                "plain_p23_probe_auroc": round(plain_p23_auroc, 4),
                "oracle_p23_probe_auroc": round(oracle_p23_auroc, 4),
            }
        )

    # Generate diagnostic plots
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    layers_x = [r["layer"] for r in logit_lens_rows]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(layers_x, [r["real_cf_lens_sep"] for r in logit_lens_rows], marker="o", label="Logit Lens Sep (real_cf)")
    axes[0].plot(layers_x, [r["p3_minus_p2_lens_sep"] for r in logit_lens_rows], marker="s", label="Logit Lens Sep (P3 - P2)")
    axes[0].axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    for l in layers_x:
        if layer_types[l] == "full_attention":
            axes[0].axvline(l, color="orange", alpha=0.22, linewidth=4)
            axes[1].axvline(l, color="orange", alpha=0.22, linewidth=4)
    axes[0].set_ylabel("Margin Δz (Block - Allow)")
    axes[0].set_title("Track 0 24-Layer Logit Lens Separation (Orange = Full-Attention)")
    axes[0].legend()

    axes[1].plot(layers_x, [r["real_cf_attn_contrib_sep"] for r in logit_lens_rows], marker="^", label="Attn/DeltaNet Sep (real_cf)")
    axes[1].plot(layers_x, [r["real_cf_mlp_contrib_sep"] for r in logit_lens_rows], marker="v", label="MLP Sep (real_cf)")
    axes[1].plot(layers_x, [r["p23_attn_contrib_sep"] for r in logit_lens_rows], linestyle=":", label="Attn/DeltaNet Sep (P3 - P2)")
    axes[1].plot(layers_x, [r["p23_mlp_contrib_sep"] for r in logit_lens_rows], linestyle=":", label="MLP Sep (P3 - P2)")
    axes[1].axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Layer Index (0..23)")
    axes[1].set_ylabel("Direct Sub-block Δz")
    axes[1].legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "track0_logit_lens.png", dpi=150)
    plt.close(fig)

    fig2, axes2 = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes2[0].plot(layers_x, [r["mean_cosine"] for r in plain_vs_oracle_rows], marker="o", label="cos(h_plain, h_oracle)")
    axes2[0].plot(layers_x, [r["oracle_net_sep_shift"] for r in plain_vs_oracle_rows], marker="s", label="Oracle net sep shift on u")
    for l in layers_x:
        if layer_types[l] == "full_attention":
            axes2[0].axvline(l, color="orange", alpha=0.22, linewidth=4)
            axes2[1].axvline(l, color="orange", alpha=0.22, linewidth=4)
    axes2[0].set_title("Track 0 Plain vs Oracle Prompt Slot Representation Shift (observe)")
    axes2[0].legend()

    axes2[1].plot(layers_x, [r["plain_rc_probe_auroc"] for r in plain_vs_oracle_rows], marker="o", label="Plain Probe AUROC (real_cf)")
    axes2[1].plot(layers_x, [r["oracle_rc_probe_auroc"] for r in plain_vs_oracle_rows], marker="o", linestyle="--", label="Oracle Probe AUROC (real_cf)")
    axes2[1].plot(layers_x, [r["plain_p23_probe_auroc"] for r in plain_vs_oracle_rows], marker="s", label="Plain Probe AUROC (P2 vs P3)")
    axes2[1].plot(layers_x, [r["oracle_p23_probe_auroc"] for r in plain_vs_oracle_rows], marker="s", linestyle="--", label="Oracle Probe AUROC (P2 vs P3)")
    axes2[1].set_xlabel("Layer Index (0..23)")
    axes2[1].set_ylabel("2-Fold Linear AUROC")
    axes2[1].legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "track0_plain_vs_oracle.png", dpi=150)
    plt.close(fig2)

    bucket_rates = {
        "keyword_reflex_rate": round(buckets["bucket_1_keyword_reflex_errors"] / max(1, buckets["bucket_1_keyword_reflex_p2_p5a_total"]), 4),
        "target_mismatch_miss_rate": round(buckets["bucket_2_target_mismatch_errors"] / max(1, buckets["bucket_2_target_mismatch_p3_p5b_total"]), 4),
        "real_cf_fn_allow_drift_rate": round(buckets["bucket_3_real_cf_allow_drift_fn"] / max(1, buckets["bucket_3_real_cf_block_total"]), 4),
        "order_swap_flip_rate": round(buckets["bucket_4_order_flips"] / max(1, buckets["bucket_4_order_flip_total"]), 4),
    }

    report = {
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "dev_rows_diagnosed": len(dev_rows),
        "observe_rows_diagnosed": len(obs_rows),
        "error_buckets_counts": buckets,
        "error_buckets_rates": bucket_rates,
        "logit_lens_24_layers": logit_lens_rows,
        "plain_vs_oracle_24_layers": plain_vs_oracle_rows,
        "plots": [
            "experiments/plots/track0_logit_lens.png",
            "experiments/plots/track0_plain_vs_oracle.png",
        ],
    }

    json_path = EXPERIMENTS_DIR / "track0_diagnostics.json"
    md_path = EXPERIMENTS_DIR / "track0_diagnostics.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md_lines = [
        "# Track 0 Mechanistic Circuit Diagnostics",
        "",
        f"- `select_dev` rows diagnosed: `{len(dev_rows)}` | `observe` paired rows: `{len(obs_rows)}` | Elapsed: `{report['elapsed_sec']}s`",
        "",
        "## 4-Bucket Error Breakdown (`select_dev`, Plain Zero-Shot)",
        f"- **Bucket 1 — Keyword Reflex** (P2/P5a wrongly Blocked): `{buckets['bucket_1_keyword_reflex_errors']}/{buckets['bucket_1_keyword_reflex_p2_p5a_total']}` (`{bucket_rates['keyword_reflex_rate']}`)",
        f"- **Bucket 2 — Target Mismatch Blindness** (P3/P5b wrongly Allowed): `{buckets['bucket_2_target_mismatch_errors']}/{buckets['bucket_2_target_mismatch_p3_p5b_total']}` (`{bucket_rates['target_mismatch_miss_rate']}`)",
        f"- **Bucket 3 — Real-Log Allow Drift (FN)** (`real_cf` Block wrongly Allowed): `{buckets['bucket_3_real_cf_allow_drift_fn']}/{buckets['bucket_3_real_cf_block_total']}` (`{bucket_rates['real_cf_fn_allow_drift_rate']}`)",
        f"- **Bucket 4 — Order-Swap Flip**: `{buckets['bucket_4_order_flips']}/{buckets['bucket_4_order_flip_total']}` (`{bucket_rates['order_swap_flip_rate']}`)",
        "",
        "## 24-Layer Summary (Logit Lens & Plain-vs-Oracle Shift)",
        "",
        "| L | Type | Lens Sep (`real_cf`) | Lens Sep (`P3-P2`) | Attn Sep (`real_cf`) | MLP Sep (`real_cf`) | `cos(Plain,Oracle)` | Oracle Net Sep Shift | Probe AUROC (`real_cf` P/O) | Probe AUROC (`P23` P/O) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for ll, po in zip(logit_lens_rows, plain_vs_oracle_rows):
        lt = "**full**" if ll["layer_type"] == "full_attention" else "linear"
        md_lines.append(
            f"| {ll['layer']} | {lt} | {ll['real_cf_lens_sep']:+.3f} | {ll['p3_minus_p2_lens_sep']:+.3f} | "
            f"{ll['real_cf_attn_contrib_sep']:+.3f} | {ll['real_cf_mlp_contrib_sep']:+.3f} | "
            f"{po['mean_cosine']:.4f} | {po['oracle_net_sep_shift']:+.3f} | "
            f"{po['plain_rc_probe_auroc']:.3f} / {po['oracle_rc_probe_auroc']:.3f} | "
            f"{po['plain_p23_probe_auroc']:.3f} / {po['oracle_p23_probe_auroc']:.3f} |"
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(md_path.read_text(encoding="utf-8"))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Track 0 24-layer diagnostics.")
    ap.add_argument("--observe-limit", type=int, default=148)
    ap.add_argument("--dev-limit", type=int, default=128)
    args = ap.parse_args()
    run_diagnostics(observe_limit=args.observe_limit, dev_limit=args.dev_limit)


if __name__ == "__main__":
    main()
