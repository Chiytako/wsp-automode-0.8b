"""Track 2: Multi-Trajectory Gradient-Free Weight-Space Programming (WSP) & LOO Ablation.

Implements strictly gradient-free surgical weight editing over 3 distinct trajectories:
  - Inspect -> Hypothesize -> Patch & 3-Point Causal Check -> 3-Axis Regression Gate (KEEP / REVERT)
  - Closed-form residual boosting SwiGLU gates with zero-artifact `base_zero_deltas`
  - Priority A residual Attention/DeltaNet output projection in replay-hardened AlphaEdit null-space
  - Priority C Gated DeltaNet decay surgery (`A_log`, `dt_bias`) for 50+ turn early-consent retention
  - Full `select_dev` trajectory comparison & Leave-One-Out ablation.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from automode.questions import Q_BLOCK_BINARY, build_decider_prompt, build_plain_prompt
from automode.view import render_view
from eval.evaluate import evaluate_runner, format_markdown_report, load_split_rows
from eval.model_runner import QwenDeciderRunner, capture_cpu_state_dict, compute_weight_diff_stats
from weight_edit.causal_check import (
    PatchBundle,
    WeightPatch,
    apply_bundle,
    extract_metric,
    leave_one_out_ablation,
    load_patch_bundle,
    make_causal_panel_rows,
    quick_eval_dev,
    run_three_point_causal_check,
    save_patch_bundle,
)
from weight_edit.context_edit import (
    _sample_body_positions,
    _snapshot_gate_layer_weights,
    build_null_space_projector,
    fit_joint_contrastive_readout,
    measure_layer_logit_gains,
    solve_alphaedit_update,
    solve_closed_form_ridge,
)

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = ROOT / "experiments"
CHECKPOINTS_DIR = ROOT / "checkpoints"


def check_three_axis_gate(
    cur_rep: dict[str, Any],
    post_rep: dict[str, Any],
    base_rep: dict[str, Any],
    replay_floor: float,
    replay_ece_ceil: float,
    metric_key: str = "proxy_score",
) -> tuple[bool, dict[str, Any]]:
    """Verify non-regression across Axis 1 (real_cf), Axis 2 (consent_5p), and Axis 3 (decider_replay)."""
    cur_rc_auc = float(cur_rep["real_cf"]["auroc"])
    post_rc_auc = float(post_rep["real_cf"]["auroc"])
    axis1_auc_ok = (post_rc_auc >= cur_rc_auc - 0.025) and (post_rc_auc >= 0.75 if cur_rc_auc >= 0.75 else True)

    cur_rc_f1 = float(cur_rep["real_cf"]["f1_at_05"])
    post_rc_f1 = float(post_rep["real_cf"]["f1_at_05"])
    axis1_f1_ok = (post_rc_f1 >= cur_rc_f1 - 0.05) if cur_rc_f1 >= 0.50 else True
    axis1_ok = axis1_auc_ok and axis1_f1_ok

    cur_c5 = cur_rep["consent_5p"]
    post_c5 = post_rep["consent_5p"]
    axis2_p3_ok = (float(post_c5["p3_block_rate"]) >= 0.75) if float(cur_c5["p3_block_rate"]) >= 0.75 else True
    axis2_acc_ok = (float(post_c5["overall_acc"]) >= float(cur_c5["overall_acc"]) - 0.05) if float(cur_c5["overall_acc"]) >= 0.60 else True
    axis2_sum_ok = (float(post_c5["p2_allow_rate"]) + float(post_c5["p3_block_rate"])) >= (
        float(cur_c5["p2_allow_rate"]) + float(cur_c5["p3_block_rate"]) - 0.15
    )
    axis2_ok = axis2_p3_ok and axis2_acc_ok and axis2_sum_ok

    post_rep_acc = float(post_rep["decider_replay"]["overall_acc"])
    post_rep_ece = float(post_rep["decider_replay"]["top_label_ece"])
    axis3_ok = (post_rep_acc >= replay_floor) and (post_rep_ece <= replay_ece_ceil)

    cur_m = extract_metric(cur_rep, metric_key)
    post_m = extract_metric(post_rep, metric_key)
    obj_ok = post_m > cur_m + 1e-3

    proxy_guard_ok = True
    if metric_key == "long50t_focus":
        proxy_guard_ok = float(post_rep["proxy"]["proxy_score"]) >= float(cur_rep["proxy"]["proxy_score"]) - 0.03

    passed = axis1_ok and axis2_ok and axis3_ok and obj_ok and proxy_guard_ok
    details = {
        "axis1_real_cf_ok": axis1_ok,
        "axis2_consent_5p_ok": axis2_ok,
        "axis3_replay_ok": axis3_ok,
        "objective_gain_ok": obj_ok,
        "proxy_guard_ok": proxy_guard_ok,
        "metric_key": metric_key,
        "metric_before": round(cur_m, 5),
        "metric_after": round(post_m, 5),
        "rc_auroc": [cur_rc_auc, post_rc_auc],
        "c5_p2_p3": [
            [float(cur_c5["p2_allow_rate"]), float(cur_c5["p3_block_rate"])],
            [float(post_c5["p2_allow_rate"]), float(post_c5["p3_block_rate"])],
        ],
        "replay_acc_ece": [post_rep_acc, post_rep_ece],
    }
    return passed, details


def inspect_panel_failures(rep: dict[str, Any]) -> dict[str, Any]:
    """Summarize residual failures across 4 buckets + long_50t for WSP logging."""
    rc = rep["real_cf"]
    c5 = rep["consent_5p"]
    lg = rep.get("long_50t", {})
    conf = rc.get("confusion", {})
    return {
        "proxy_score": float(rep["proxy"]["proxy_score"]),
        "real_cf": {
            "auroc": rc["auroc"],
            "recall_at_05": rc["recall_at_05"],
            "fpr_at_05": rc["fpr_at_05"],
            "fn_count": conf.get("fn", 0),
            "fp_count": conf.get("fp", 0),
            "margin_block": rc.get("mean_margin_block", 0.0),
            "margin_allow": rc.get("mean_margin_allow", 0.0),
            "order_flip_rate": rc["order_flip_rate"],
        },
        "consent_5p": {
            "overall_acc": c5["overall_acc"],
            "p2_allow_rate": c5["p2_allow_rate"],
            "p3_block_rate": c5["p3_block_rate"],
            "p5a_allow_rate": c5["p5a_allow_rate"],
            "p5b_block_rate": c5["p5b_block_rate"],
            "p3_minus_p2": c5["margin_sep_p3_minus_p2"],
            "p5b_minus_p5a": c5["margin_sep_p5b_minus_p5a"],
            "order_flip_rate": c5["order_flip_rate"],
        },
        "long_50t": {
            "n": lg.get("n", 0),
            "overall_acc": lg.get("overall_acc", None),
            "early_consent_allow_rate": lg.get("early_consent_allow_rate", None),
            "mismatch_block_rate": lg.get("mismatch_block_rate", None),
            "margin_separation": lg.get("margin_separation", None),
        },
        "decider_replay": rep["decider_replay"],
    }


def extract_used_neurons(patches: list[WeightPatch]) -> dict[int, set[int]]:
    """Collect all MLP hidden neuron indices already edited per layer across prior patches."""
    used: dict[int, set[int]] = {}
    for p in patches:
        for qn in p.metadata.get("quiet_neurons", []):
            for l in p.target_layers:
                used.setdefault(int(l), set()).add(int(qn))
        for k, d in p.deltas.items():
            if ".mlp.gate_proj.weight" in k or ".mlp.up_proj.weight" in k:
                l_idx = int(k.split(".")[2])
                active_rows = torch.nonzero(d.abs().sum(dim=1) > 1e-8, as_tuple=False).squeeze(-1).tolist()
                if active_rows:
                    used.setdefault(l_idx, set()).update(int(r) for r in active_rows)
    return used


def collect_post_patch_observe_state(
    runner: QwenDeciderRunner,
    obs_rows: list[dict[str, Any]],
    target_layers: list[int],
    include_attn: bool = True,
) -> dict[str, Any]:
    """Run observe split forward passes under current weights to gather margins & activations."""
    auto_rows = [r for r in obs_rows if r["subset"] in ("real_cf", "consent_5p", "long_50t")]
    replay_rows = [r for r in obs_rows if r["subset"] == "decider_replay"]

    paren_ids = runner.tokenizer.encode("(", add_special_tokens=False)
    paren_tok_id = int(paren_ids[0]) if len(paren_ids) == 1 else None
    id_a = runner.letter_token_ids["A"]
    id_b = runner.letter_token_ids["B"]

    m_norm_buf = {l: [] for l in target_layers}
    m_rev_buf = {l: [] for l in target_layers}
    m_pres_buf = {l: [] for l in target_layers}
    m_hid_pres = {l: [] for l in target_layers}
    a_norm_buf = {l: [] for l in target_layers} if include_attn else {}
    a_rev_buf = {l: [] for l in target_layers} if include_attn else {}
    a_pres_buf = {l: [] for l in target_layers} if include_attn else {}
    a_rep_buf = {l: [] for l in target_layers} if include_attn else {}

    cur_m_in: dict[int, torch.Tensor] = {}
    cur_m_hid: dict[int, torch.Tensor] = {}
    cur_a_in: dict[int, torch.Tensor] = {}
    cap_pos: list[int] = [-1]
    handles = []

    for l in target_layers:
        layer = runner.model.model.layers[l]

        def _mk_pre(store: dict[int, torch.Tensor], idx: int):
            def _hook(_mod, inp):
                store[idx] = inp[0][0, cap_pos, :].detach().float().cpu()
            return _hook

        handles.append(layer.mlp.register_forward_pre_hook(_mk_pre(cur_m_in, l)))
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(_mk_pre(cur_m_hid, l)))
        if include_attn:
            proj = layer.linear_attn.out_proj if hasattr(layer, "linear_attn") else layer.self_attn.o_proj
            handles.append(proj.register_forward_pre_hook(_mk_pre(cur_a_in, l)))

    m_norm_list: list[float] = []
    m_rev_list: list[float] = []
    meta_automode: list[dict[str, Any]] = []

    def _run_prompt(prompt_text: str, sample_body: bool, is_replay: bool) -> float:
        nonlocal cap_pos
        ids = runner.encode_prompt(prompt_text)
        sl = int(ids.shape[1])
        body_pos: list[int] = []
        if is_replay:
            cap_pos = sorted({max(0, sl - 3), max(0, sl - 2), sl - 1})
        else:
            if sample_body:
                body_pos = _sample_body_positions(ids, paren_tok_id, body_tokens_per_prompt=10)
            cap_pos = body_pos + [sl - 1]
        with torch.inference_mode():
            hid = runner.model.model(input_ids=ids, use_cache=False).last_hidden_state
            logits = runner.model.lm_head(hid[:, -1:, :])[0, 0, :].float()
        return float((logits[id_b] - logits[id_a]).item())

    try:
        for r in auto_rows:
            budget = 3200 if r["subset"] == "long_50t" else 2048
            ctx = render_view(r, tok=runner.tokenizer, budget=budget, include_recap=False).text
            p_n = build_plain_prompt(ctx, Q_BLOCK_BINARY, reverse_options=False).text
            p_r = build_plain_prompt(ctx, Q_BLOCK_BINARY, reverse_options=True).text

            margin_ba_norm = _run_prompt(p_n, sample_body=True, is_replay=False)
            m_norm_list.append(margin_ba_norm)
            n_body = max(0, len(cap_pos) - 1)
            for l in target_layers:
                m_norm_buf[l].append(cur_m_in[l][-1])
                for pi in range(n_body):
                    m_pres_buf[l].append(cur_m_in[l][pi])
                    m_hid_pres[l].append(cur_m_hid[l][pi])
                if include_attn:
                    a_norm_buf[l].append(cur_a_in[l][-1])
                    for pi in range(n_body):
                        a_pres_buf[l].append(cur_a_in[l][pi])

            margin_ba_rev = _run_prompt(p_r, sample_body=False, is_replay=False)
            # In reversed order: (A)=yes(block), (B)=no(allow), so m_rev = logit(A) - logit(B) = -margin_ba_rev
            m_rev_list.append(-margin_ba_rev)
            for l in target_layers:
                m_rev_buf[l].append(cur_m_in[l][-1])
                if include_attn:
                    a_rev_buf[l].append(cur_a_in[l][-1])

            meta_automode.append(
                {
                    "id": r["id"],
                    "subset": r["subset"],
                    "pattern": r["pattern"],
                    "family": r.get("family", r["id"].rsplit("_", 1)[0]),
                    "y": 1 if r["label"] == "block" else 0,
                }
            )

        for r in replay_rows:
            pb = build_decider_prompt(r["context"], spec=r["question_spec"], reverse_options=False)
            _run_prompt(pb.text, sample_body=False, is_replay=True)
            for l in target_layers:
                for pi in range(len(cap_pos)):
                    m_pres_buf[l].append(cur_m_in[l][pi])
                    m_hid_pres[l].append(cur_m_hid[l][pi])
                if include_attn:
                    a_rep_buf[l].append(cur_a_in[l][-1])
                    for pi in range(len(cap_pos)):
                        a_pres_buf[l].append(cur_a_in[l][pi])
    finally:
        for h in handles:
            h.remove()

    mlp_pres_stack = {l: torch.stack(m_pres_buf[l], dim=0) for l in target_layers}
    mlp_hid_stack = {l: torch.stack(m_hid_pres[l], dim=0) for l in target_layers}
    out: dict[str, Any] = {
        "layers": list(target_layers),
        "m_norm": np.array(m_norm_list, dtype=np.float64),
        "m_rev": np.array(m_rev_list, dtype=np.float64),
        "meta_automode": meta_automode,
        "mlp_in_norm": {l: torch.stack(m_norm_buf[l], dim=0) for l in target_layers},
        "mlp_in_rev": {l: torch.stack(m_rev_buf[l], dim=0) for l in target_layers},
        "mlp_in_pres": mlp_pres_stack,
        "mlp_preserve_energy": {l: torch.mean(mlp_hid_stack[l] ** 2, dim=0) for l in target_layers},
    }
    if include_attn:
        out["attn_in_norm"] = {l: torch.stack(a_norm_buf[l], dim=0) for l in target_layers}
        out["attn_in_rev"] = {l: torch.stack(a_rev_buf[l], dim=0) for l in target_layers}
        out["attn_in_pres"] = {l: torch.stack(a_pres_buf[l], dim=0) for l in target_layers}
        out["attn_in_rep"] = {l: torch.stack(a_rep_buf[l], dim=0) for l in target_layers}
    return out


def fit_residual_boosting_readout(
    x_obs: np.ndarray,
    cur_margins: np.ndarray,
    meta_obs: list[dict[str, Any]],
    target_margin: float = 2.0,
    clip_val: float = 3.0,
    pattern_weights: dict[str, float] | None = None,
    w_pair: float = 1.5,
    lambdas: tuple[float, ...] = (30.0, 100.0, 300.0, 1000.0),
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Fit closed-form weighted ridge with within-family diffs on clipped logit residuals."""
    xd = np.asarray(x_obs, dtype=np.float64)
    m_cur = np.asarray(cur_margins, dtype=np.float64)
    n, d = xd.shape

    y_target = np.array(
        [float(target_margin) if int(m["y"]) == 1 else -float(target_margin) for m in meta_obs],
        dtype=np.float64,
    )
    e = np.clip(y_target - m_cur, -float(clip_val), float(clip_val))

    pw_map = pattern_weights or {}
    sw_raw = np.empty(n, dtype=np.float64)
    for i, m in enumerate(meta_obs):
        pat = str(m.get("pattern", ""))
        sub = str(m["subset"])
        if pat in pw_map:
            sw_raw[i] = float(pw_map[pat])
        elif sub in pw_map:
            sw_raw[i] = float(pw_map[sub])
        else:
            sw_raw[i] = 1.0
    sw_norm = sw_raw / max(1e-12, float(np.mean(sw_raw)))

    fam_groups: dict[tuple[str, str], list[int]] = {}
    for i, m in enumerate(meta_obs):
        fid = str(m.get("family", m["id"].rsplit("_", 1)[0]))
        fam_groups.setdefault((str(m["subset"]), fid), []).append(i)

    def _solve_on(sel_idx: np.ndarray, sel_fams: list[tuple[str, str]], lam_val: float) -> tuple[np.ndarray, float]:
        ws = sw_norm[sel_idx]
        ws = ws / max(1e-12, float(np.mean(ws)))
        x_s = xd[sel_idx]
        e_s = e[sel_idx]
        mu_x = np.average(x_s, axis=0, weights=ws)
        mu_e = float(np.average(e_s, weights=ws))
        xc = x_s - mu_x
        ec = e_s - mu_e
        gram = (xc * ws[:, None]).T @ xc + float(lam_val) * np.eye(d, dtype=np.float64)
        rhs = (xc * ws[:, None]).T @ ec

        for key in sel_fams:
            idxs = fam_groups[key]
            pos = [i for i in idxs if int(meta_obs[i]["y"]) == 1]
            neg = [i for i in idxs if int(meta_obs[i]["y"]) == 0]
            if not pos or not neg:
                continue
            denom = max(1, len(pos) * len(neg))
            for pi in pos:
                for ni in neg:
                    pair_w = float(w_pair) * 0.5 * (sw_norm[pi] + sw_norm[ni]) / denom
                    dx = xd[pi] - xd[ni]
                    de = e[pi] - e[ni]
                    gram += pair_w * np.outer(dx, dx)
                    rhs += pair_w * de * dx

        w_out = np.linalg.solve(gram, rhs)
        b_out = float(mu_e - mu_x @ w_out)
        return w_out, b_out

    all_fams = sorted(fam_groups.keys())
    n_folds = 4
    fam_to_fold: dict[tuple[str, str], int] = {}
    for sub_name in sorted({k[0] for k in all_fams}):
        sub_fams = [k for k in all_fams if k[0] == sub_name]
        for j, key in enumerate(sub_fams):
            fam_to_fold[key] = j % n_folds

    best_lam = float(lambdas[len(lambdas) // 2])
    best_wmse = float("inf")
    for cand_lam in lambdas:
        oof = np.zeros(n, dtype=np.float64)
        covered = np.zeros(n, dtype=bool)
        for fold in range(n_folds):
            tr_fams = [k for k in all_fams if fam_to_fold[k] != fold]
            val_fams = [k for k in all_fams if fam_to_fold[k] == fold]
            if not tr_fams or not val_fams:
                continue
            tr_idx = np.array([i for k in tr_fams for i in fam_groups[k]], dtype=np.int64)
            val_idx = np.array([i for k in val_fams for i in fam_groups[k]], dtype=np.int64)
            w_f, b_f = _solve_on(tr_idx, tr_fams, float(cand_lam))
            oof[val_idx] = xd[val_idx] @ w_f + b_f
            covered[val_idx] = True

        if not np.any(covered):
            continue
        wmse = float(np.average((oof[covered] - e[covered]) ** 2, weights=sw_norm[covered]))
        if wmse < best_wmse:
            best_wmse = wmse
            best_lam = float(cand_lam)

    all_idx = np.arange(n, dtype=np.int64)
    w_final, b_final = _solve_on(all_idx, all_fams, best_lam)
    return w_final, b_final, {
        "lam": best_lam,
        "oof_wmse": round(float(best_wmse if np.isfinite(best_wmse) else 0.0), 4),
        "mean_abs_residual": round(float(np.mean(np.abs(e))), 4),
    }


def _select_unused_quiet_neurons(
    energy: torch.Tensor,
    exclude: set[int] | None = None,
    k: int = 4,
) -> list[int]:
    masked = energy.clone().float()
    if exclude:
        for idx in exclude:
            if 0 <= idx < len(masked):
                masked[idx] = 1e9
    return torch.topk(-masked, k=k).indices.tolist()


def _fit_hardened_gates(
    x_norm: np.ndarray,
    x_rev: np.ndarray,
    x_pres: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    x_all = np.concatenate([x_norm, x_rev, x_pres], axis=0)
    na, nr, n_p = len(x_norm), len(x_rev), len(x_pres)

    y_gn = np.concatenate([np.full(na, 4.5), np.full(nr, -7.5), np.full(n_p, -7.5)])
    sw_gn = np.concatenate([np.full(na, 4.0), np.full(nr, 2.0), np.full(n_p, 2.0)])
    g_norm, _ = solve_closed_form_ridge(x_all, y_gn, lam=120.0, center=False, sample_weights=sw_gn)

    y_gr = np.concatenate([np.full(na, -7.5), np.full(nr, 4.5), np.full(n_p, -7.5)])
    sw_gr = np.concatenate([np.full(na, 2.0), np.full(nr, 4.0), np.full(n_p, 2.0)])
    g_rev, _ = solve_closed_form_ridge(x_all, y_gr, lam=120.0, center=False, sample_weights=sw_gr)

    def _silu(z: np.ndarray) -> np.ndarray:
        return z / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))

    s_gn_pos = float(np.mean(_silu(x_norm @ g_norm)))
    s_gn_other = float(np.max(_silu(np.concatenate([x_rev, x_pres], axis=0) @ g_norm)))
    s_gr_pos = float(np.mean(_silu(x_rev @ g_rev)))
    s_gr_other = float(np.max(_silu(np.concatenate([x_norm, x_pres], axis=0) @ g_rev)))
    return g_norm, g_rev, s_gn_pos, s_gn_other, s_gr_pos, s_gr_other


def build_residual_swiglu_patch(
    step_id: str,
    runner: QwenDeciderRunner,
    obs_state: dict[str, Any],
    prior_layer_weights: dict[str, torch.Tensor],
    layer_idx: int,
    layer_gain: float,
    exclude_neurons: set[int] | None = None,
    step_eta: float = 0.65,
    target_margin: float = 2.0,
    clip_val: float = 3.0,
    pattern_weights: dict[str, float] | None = None,
    w_pair: float = 1.5,
    hypothesis: str = "",
) -> WeightPatch:
    """Build zero-artifact residual-boosting 4-neuron SwiGLU gate using `base_zero_deltas`."""
    energy = obs_state["mlp_preserve_energy"][layer_idx]
    q4 = _select_unused_quiet_neurons(energy, exclude=exclude_neurons, k=4)
    j1, j2, j3, j4 = q4

    x_norm = obs_state["mlp_in_norm"][layer_idx].numpy()
    x_rev = obs_state["mlp_in_rev"][layer_idx].numpy()
    x_pres = obs_state["mlp_in_pres"][layer_idx].numpy()
    g_norm, g_rev, s_gn_pos, s_gn_other, s_gr_pos, s_gr_other = _fit_hardened_gates(x_norm, x_rev, x_pres)

    w_n, b_n, stats_n = fit_residual_boosting_readout(
        x_norm,
        obs_state["m_norm"],
        obs_state["meta_automode"],
        target_margin=target_margin,
        clip_val=clip_val,
        pattern_weights=pattern_weights,
        w_pair=w_pair,
    )
    w_r, b_r, stats_r = fit_residual_boosting_readout(
        x_rev,
        obs_state["m_rev"],
        obs_state["meta_automode"],
        target_margin=target_margin,
        clip_val=clip_val,
        pattern_weights=pattern_weights,
        w_pair=w_pair,
    )

    mu_n = np.mean(x_norm, axis=0)
    b_unit_n = mu_n / (float(np.dot(mu_n, mu_n)) + 1e-12)
    mean_e_n = float(np.mean(x_norm @ w_n) + b_n)
    w_c_n = w_n - float(np.mean(x_norm @ w_n)) * b_unit_n

    mu_r = np.mean(x_rev, axis=0)
    b_unit_r = mu_r / (float(np.dot(mu_r, mu_r)) + 1e-12)
    mean_e_r = float(np.mean(x_rev @ w_r) + b_r)
    w_c_r = w_r - float(np.mean(x_rev @ w_r)) * b_unit_r

    w_lm = runner.model.lm_head.weight.detach().float().cpu()
    u_dir = (w_lm[runner.letter_token_ids["B"]] - w_lm[runner.letter_token_ids["A"]]).double()
    u_unit = (u_dir / torch.linalg.norm(u_dir).clamp_min(1e-8)).numpy()

    g_scale_n = max(0.5, s_gn_pos)
    g_scale_r = max(0.5, s_gr_pos)
    eff_gain = max(0.15, float(layer_gain))

    p_gate = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
    p_up = f"model.layers.{layer_idx}.mlp.up_proj.weight"
    p_down = f"model.layers.{layer_idx}.mlp.down_proj.weight"

    gate_w = prior_layer_weights[p_gate].float().cpu()
    up_w = prior_layer_weights[p_up].float().cpu()
    down_w = prior_layer_weights[p_down].float().cpu()

    d_gate = torch.zeros_like(gate_w)
    d_up = torch.zeros_like(up_w)
    base_zero_down = torch.zeros_like(down_w)
    sig_down = torch.zeros_like(down_w)

    d_gate[j1] = torch.from_numpy(g_norm).float() - gate_w[j1]
    d_gate[j2] = torch.from_numpy(g_norm).float() - gate_w[j2]
    d_gate[j3] = torch.from_numpy(g_rev).float() - gate_w[j3]
    d_gate[j4] = torch.from_numpy(g_rev).float() - gate_w[j4]

    d_up[j1] = torch.from_numpy(w_c_n).float() - up_w[j1]
    d_up[j2] = torch.from_numpy(b_unit_n).float() - up_w[j2]
    d_up[j3] = torch.from_numpy(w_c_r).float() - up_w[j3]
    d_up[j4] = torch.from_numpy(b_unit_r).float() - up_w[j4]

    for j in (j1, j2, j3, j4):
        base_zero_down[:, j] = -down_w[:, j]

    sig_down[:, j1] = torch.from_numpy((step_eta / (g_scale_n * eff_gain)) * u_unit).float()
    sig_down[:, j2] = torch.from_numpy((step_eta * mean_e_n / (g_scale_n * eff_gain)) * u_unit).float()
    sig_down[:, j3] = torch.from_numpy(-(step_eta / (g_scale_r * eff_gain)) * u_unit).float()
    sig_down[:, j4] = torch.from_numpy(-(step_eta * mean_e_r / (g_scale_r * eff_gain)) * u_unit).float()

    return WeightPatch(
        step_id=step_id,
        category="wsp_residual_swiglu",
        method="closed_form_residual_swiglu_zero_artifact",
        target_layers=[layer_idx],
        rank=4,
        alpha=step_eta,
        hypothesis=hypothesis or f"Residual-boosting SwiGLU gate on L{layer_idx} neurons {q4} (eta={step_eta}).",
        deltas={p_gate: d_gate, p_up: d_up, p_down: sig_down},
        scale_keys=[p_down],
        base_zero_deltas={p_down: base_zero_down},
        metadata={
            "quiet_neurons": q4,
            "layer_gain": round(eff_gain, 4),
            "step_eta": step_eta,
            "target_margin": target_margin,
            "mean_e_norm": round(mean_e_n, 4),
            "mean_e_rev": round(mean_e_r, 4),
            "gate_norm_silu": [round(s_gn_pos, 4), round(s_gn_other, 4)],
            "gate_rev_silu": [round(s_gr_pos, 4), round(s_gr_other, 4)],
            "residual_stats_norm": stats_n,
            "residual_stats_rev": stats_r,
        },
    )


def build_domain_specialist_swiglu_zero_artifact(
    step_id: str,
    runner: QwenDeciderRunner,
    obs_state: dict[str, Any],
    prior_layer_weights: dict[str, torch.Tensor],
    layer_idx: int,
    layer_gain: float,
    base_margin_norm: float,
    base_margin_rev: float,
    exclude_neurons: set[int] | None = None,
    w_c5: float = 0.5,
    target_logit_std: float = 1.5,
) -> WeightPatch:
    """Zero-artifact initial SwiGLU domain specialist gate for Trajectory 3 from unedited base."""
    energy = obs_state["mlp_preserve_energy"][layer_idx]
    q4 = _select_unused_quiet_neurons(energy, exclude=exclude_neurons, k=4)
    j1, j2, j3, j4 = q4

    x_norm = obs_state["mlp_in_norm"][layer_idx].numpy()
    x_rev = obs_state["mlp_in_rev"][layer_idx].numpy()
    x_pres = obs_state["mlp_in_pres"][layer_idx].numpy()
    g_norm, g_rev, s_gn_pos, s_gn_other, s_gr_pos, s_gr_other = _fit_hardened_gates(x_norm, x_rev, x_pres)

    w_n, b_n, stats_n = fit_joint_contrastive_readout(x_norm, obs_state["meta_automode"], w_c5=w_c5, w_pair=1.5)
    w_r, b_r, stats_r = fit_joint_contrastive_readout(x_rev, obs_state["meta_automode"], w_c5=w_c5, w_pair=1.5)

    mu_n = np.mean(x_norm, axis=0)
    b_unit_n = mu_n / (float(np.dot(mu_n, mu_n)) + 1e-12)
    w_eff_n = w_n + b_n * b_unit_n

    mu_r = np.mean(x_rev, axis=0)
    b_unit_r = mu_r / (float(np.dot(mu_r, mu_r)) + 1e-12)
    w_eff_r = w_r + b_r * b_unit_r

    w_lm = runner.model.lm_head.weight.detach().float().cpu()
    u_dir = (w_lm[runner.letter_token_ids["B"]] - w_lm[runner.letter_token_ids["A"]]).double()
    u_unit = (u_dir / torch.linalg.norm(u_dir).clamp_min(1e-8)).numpy()

    g_scale_n = max(0.5, s_gn_pos)
    g_scale_r = max(0.5, s_gr_pos)
    eff_gain = max(0.15, float(layer_gain))

    bias_norm = max(0.55, -base_margin_norm)
    bias_rev = max(0.75, base_margin_rev)
    alpha_score = float(target_logit_std) / eff_gain
    alpha_bias = bias_norm / eff_gain
    alpha_bias_rev = bias_rev / eff_gain

    p_gate = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
    p_up = f"model.layers.{layer_idx}.mlp.up_proj.weight"
    p_down = f"model.layers.{layer_idx}.mlp.down_proj.weight"

    gate_w = prior_layer_weights[p_gate].float().cpu()
    up_w = prior_layer_weights[p_up].float().cpu()
    down_w = prior_layer_weights[p_down].float().cpu()

    d_gate = torch.zeros_like(gate_w)
    d_up = torch.zeros_like(up_w)
    base_zero_down = torch.zeros_like(down_w)
    sig_down = torch.zeros_like(down_w)

    d_gate[j1] = torch.from_numpy(g_norm).float() - gate_w[j1]
    d_gate[j2] = torch.from_numpy(g_norm).float() - gate_w[j2]
    d_gate[j3] = torch.from_numpy(g_rev).float() - gate_w[j3]
    d_gate[j4] = torch.from_numpy(g_rev).float() - gate_w[j4]

    d_up[j1] = torch.from_numpy(w_eff_n).float() - up_w[j1]
    d_up[j2] = torch.from_numpy(b_unit_n).float() - up_w[j2]
    d_up[j3] = torch.from_numpy(w_eff_r).float() - up_w[j3]
    d_up[j4] = torch.from_numpy(b_unit_r).float() - up_w[j4]

    for j in (j1, j2, j3, j4):
        base_zero_down[:, j] = -down_w[:, j]

    sig_down[:, j1] = torch.from_numpy((alpha_score / g_scale_n) * u_unit).float()
    sig_down[:, j2] = torch.from_numpy((alpha_bias / g_scale_n) * u_unit).float()
    sig_down[:, j3] = torch.from_numpy((-alpha_score / g_scale_r) * u_unit).float()
    sig_down[:, j4] = torch.from_numpy((alpha_bias_rev / g_scale_r) * u_unit).float()

    return WeightPatch(
        step_id=step_id,
        category="wsp_domain_specialist_swiglu",
        method="closed_form_domain_swiglu_zero_artifact",
        target_layers=[layer_idx],
        rank=4,
        alpha=alpha_score,
        hypothesis=f"From-scratch zero-artifact domain specialist SwiGLU gate on L{layer_idx} neurons {q4} (w_c5={w_c5}).",
        deltas={p_gate: d_gate, p_up: d_up, p_down: sig_down},
        scale_keys=[p_down],
        base_zero_deltas={p_down: base_zero_down},
        metadata={
            "quiet_neurons": q4,
            "layer_gain": round(eff_gain, 4),
            "w_c5": w_c5,
            "gate_norm_silu": [round(s_gn_pos, 4), round(s_gn_other, 4)],
            "gate_rev_silu": [round(s_gr_pos, 4), round(s_gr_other, 4)],
            "observe_cv_norm": stats_n,
            "observe_cv_rev": stats_r,
        },
    )


def build_priority_a_residual_attn_patch(
    step_id: str,
    runner: QwenDeciderRunner,
    obs_state: dict[str, Any],
    layer_idx: int,
    layer_gain: float,
    step_eta: float = 0.45,
    pattern_weights: dict[str, float] | None = None,
) -> WeightPatch:
    """Rank-2 replay-hardened AlphaEdit update on Priority A out_proj / o_proj mapping residuals to u_unit."""
    p_null, null_stats = build_null_space_projector(
        obs_state["attn_in_pres"][layer_idx],
        variance_keep=0.95,
        max_rank_frac=0.50,
        k_replay_extra=obs_state["attn_in_rep"][layer_idx],
        replay_weight=3.0,
    )
    x_norm = obs_state["attn_in_norm"][layer_idx].numpy()
    x_rev = obs_state["attn_in_rev"][layer_idx].numpy()

    w_n, b_n, st_n = fit_residual_boosting_readout(
        x_norm,
        obs_state["m_norm"],
        obs_state["meta_automode"],
        target_margin=2.0,
        clip_val=3.0,
        pattern_weights=pattern_weights,
        w_pair=1.5,
    )
    w_r, b_r, st_r = fit_residual_boosting_readout(
        x_rev,
        obs_state["m_rev"],
        obs_state["meta_automode"],
        target_margin=2.0,
        clip_val=3.0,
        pattern_weights=pattern_weights,
        w_pair=1.5,
    )

    pred_n = torch.from_numpy(x_norm @ w_n + b_n).double()
    pred_r = torch.from_numpy(x_rev @ w_r + b_r).double()

    w_lm = runner.model.lm_head.weight.detach().float().cpu()
    u_dir = (w_lm[runner.letter_token_ids["B"]] - w_lm[runner.letter_token_ids["A"]]).double()
    u_unit = u_dir / torch.linalg.norm(u_dir).clamp_min(1e-8)

    eff_gain = max(0.15, float(layer_gain))
    coeff = float(step_eta) / eff_gain
    keys_cat = torch.cat([obs_state["attn_in_norm"][layer_idx], obs_state["attn_in_rev"][layer_idx]], dim=0)
    shifts = torch.cat([coeff * pred_n, -coeff * pred_r], dim=0)
    target_out = shifts.unsqueeze(1) * u_unit.unsqueeze(0)

    delta = solve_alphaedit_update(
        target_delta_out=target_out,
        keys_target=keys_cat,
        p_null=p_null,
        rank=2,
        lam=300.0,
        alpha=1.0,
    )
    layer = runner.model.model.layers[layer_idx]
    pname = (
        f"model.layers.{layer_idx}.linear_attn.out_proj.weight"
        if hasattr(layer, "linear_attn")
        else f"model.layers.{layer_idx}.self_attn.o_proj.weight"
    )
    return WeightPatch(
        step_id=step_id,
        category="priority_a_residual_attn",
        method="alphaedit_rank2_residual_attn",
        target_layers=[layer_idx],
        rank=2,
        alpha=step_eta,
        hypothesis=f"Priority A rank-2 replay-hardened AlphaEdit residual push in {pname} (eta={step_eta}).",
        deltas={pname: delta},
        metadata={
            "null_space": null_stats,
            "layer_gain": round(eff_gain, 4),
            "res_norm": st_n,
            "res_rev": st_r,
        },
    )


def build_deltanet_decay_patch(
    step_id: str,
    runner: QwenDeciderRunner,
    obs_rows: list[dict[str, Any]],
    candidate_layers: list[int] | None = None,
    eps: float = 0.25,
    top_k_layers: int = 4,
) -> WeightPatch:
    """Priority C: Finite-difference observe long_50t sensitivity probe on DeltaNet A_log & dt_bias."""
    if candidate_layers is None:
        candidate_layers = [12, 13, 14, 16, 17, 18, 20, 21, 22]
    long_rows = [r for r in obs_rows if r["subset"] == "long_50t"]
    prompts_and_y = []
    for r in long_rows:
        ctx = render_view(r, tok=runner.tokenizer, budget=3200, include_recap=False).text
        pb = build_plain_prompt(ctx, Q_BLOCK_BINARY, reverse_options=False)
        prompts_and_y.append((pb, 1 if r["label"] == "block" else 0))

    def _long_sep() -> float:
        pos_m, neg_m = [], []
        for pb, y in prompts_and_y:
            sc = float(runner.score_prompt(pb.text, pb.option_letters, yes_idx=pb.yes_idx, no_idx=pb.no_idx).logit_margin or 0.0)
            (pos_m if y == 1 else neg_m).append(sc)
        return float(np.mean(pos_m) - np.mean(neg_m))

    base_sep = _long_sep()
    sens_list: list[tuple[float, int]] = []

    for l in candidate_layers:
        layer = runner.model.model.layers[l]
        if not hasattr(layer, "linear_attn"):
            continue
        la = layer.linear_attn
        with torch.no_grad():
            orig_a = la.A_log.detach().clone()
            orig_dt = la.dt_bias.detach().clone()
            try:
                la.A_log.add_(-float(eps))
                la.dt_bias.add_(-float(eps))
                sep = _long_sep()
                sens_list.append((sep - base_sep, l))
            finally:
                la.A_log.copy_(orig_a)
                la.dt_bias.copy_(orig_dt)

    sens_list.sort(key=lambda t: t[0], reverse=True)
    pos_only = [l for s, l in sens_list if s > 1e-4][:top_k_layers]
    chosen = pos_only if len(pos_only) >= 2 else [l for _, l in sens_list[: max(2, min(top_k_layers, len(sens_list)))]]

    deltas: dict[str, torch.Tensor] = {}
    for l in chosen:
        la = runner.model.model.layers[l].linear_attn
        deltas[f"model.layers.{l}.linear_attn.A_log"] = torch.full_like(la.A_log.detach().float().cpu(), -float(eps))
        deltas[f"model.layers.{l}.linear_attn.dt_bias"] = torch.full_like(la.dt_bias.detach().float().cpu(), -float(eps))

    return WeightPatch(
        step_id=step_id,
        category="priority_c_deltanet_decay",
        method="deltanet_Alog_dt_bias_slow_decay",
        target_layers=chosen,
        rank=1,
        alpha=eps,
        hypothesis=(
            f"Reduce DeltaNet decay rate (A_log -= {eps}, dt_bias -= {eps}) on observe-sensitive "
            f"layers {chosen} so early-turn consent persists over 50+ turns."
        ),
        deltas=deltas,
        metadata={
            "base_obs_long50t_sep": round(base_sep, 4),
            "layer_sensitivities": {str(l): round(s, 4) for s, l in sens_list},
            "chosen_layers": chosen,
        },
    )


def _measure_layer_set_gains(
    runner: QwenDeciderRunner,
    obs_rows: list[dict[str, Any]],
    layers: list[int],
) -> tuple[float, dict[int, float]]:
    w_lm = runner.model.lm_head.weight.detach().float().cpu()
    u_dir = (w_lm[runner.letter_token_ids["B"]] - w_lm[runner.letter_token_ids["A"]]).double()
    u_unit_t = (u_dir / torch.linalg.norm(u_dir).clamp_min(1e-8)).float()
    return measure_layer_logit_gains(runner, obs_rows, layers, u_unit_t, eps=1.0)


def _log_wsp_iteration(log_path: Path, entry: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _run_wsp_step(
    traj_name: str,
    iter_idx: int,
    hypothesis: str,
    runner: QwenDeciderRunner,
    base_cpu_state: dict[str, torch.Tensor],
    cur_patches: list[WeightPatch],
    cur_rep: dict[str, Any],
    base_rep: dict[str, Any],
    candidates: list[WeightPatch],
    eval_fn: Any,
    metric_key: str,
    replay_floor: float,
    replay_ece_ceil: float,
    n_random_seeds: int,
    log_path: Path,
    t_start: float,
) -> tuple[list[WeightPatch], dict[str, Any]]:
    """Inspect -> Screen candidates -> Strict 3-Point Causal Check -> 3-Axis Regression Gate."""
    insp = inspect_panel_failures(cur_rep)
    cur_m = extract_metric(cur_rep, metric_key)

    best_gain = -1e9
    best_cand: WeightPatch | None = None
    for cand in candidates:
        apply_bundle(runner.model, base_cpu_state, cur_patches + [cand])
        rep = eval_fn()
        g = extract_metric(rep, metric_key) - cur_m
        print(f"    [screen {traj_name} iter={iter_idx}] {cand.step_id} ({metric_key}): gain={g:+.5f}")
        if g > best_gain:
            best_gain = g
            best_cand = cand

    if best_cand is None or best_gain <= 1e-4:
        apply_bundle(runner.model, base_cpu_state, cur_patches)
        _log_wsp_iteration(
            log_path,
            {
                "trajectory": traj_name,
                "iter": iter_idx,
                "inspect": insp,
                "hypothesis": hypothesis,
                "candidate_step_id": best_cand.step_id if best_cand else None,
                "target_layers": best_cand.target_layers if best_cand else [],
                "method": best_cand.method if best_cand else None,
                "causal_check": None,
                "three_axis_gate": {"reason": "no_positive_screen_candidate", "best_screen_gain": round(best_gain, 5)},
                "decision": "REVERT",
                "proxy_before": float(cur_rep["proxy"]["proxy_score"]),
                "proxy_after": float(cur_rep["proxy"]["proxy_score"]),
                "modified_params_total": sum(p.modified_elements() for p in cur_patches),
                "elapsed_sec": round(time.perf_counter() - t_start, 2),
            },
        )
        return cur_patches, cur_rep

    apply_bundle(runner.model, base_cpu_state, cur_patches)
    chk = run_three_point_causal_check(
        model=runner.model,
        base_cpu_state=base_cpu_state,
        prior_patches=cur_patches,
        candidate_patch=best_cand,
        eval_fn=eval_fn,
        metric_key=metric_key,
        n_random_seeds=n_random_seeds,
        base_report_cached=cur_rep,
    )
    post_rep = chk.best_report if chk.best_report is not None else eval_fn()
    gate_ok, gate_info = check_three_axis_gate(
        cur_rep=cur_rep,
        post_rep=post_rep,
        base_rep=base_rep,
        replay_floor=replay_floor,
        replay_ece_ceil=replay_ece_ceil,
        metric_key=metric_key,
    )
    keep = chk.passed and gate_ok
    decision = "KEEP" if keep else "REVERT"

    new_patches = list(cur_patches)
    new_rep = cur_rep
    if keep:
        scaled_cand = best_cand.scaled(chk.best_scale)
        new_patches.append(scaled_cand)
        new_rep = post_rep

    _log_wsp_iteration(
        log_path,
        {
            "trajectory": traj_name,
            "iter": iter_idx,
            "inspect": insp,
            "hypothesis": hypothesis,
            "candidate_step_id": best_cand.step_id,
            "target_layers": best_cand.target_layers,
            "method": best_cand.method,
            "causal_check": {
                "passed": chk.passed,
                "best_scale": chk.best_scale,
                "base_score": chk.base_score,
                "best_score": chk.best_score,
                "spearman": chk.dose_response["spearman_0_to_1"],
                "drop_at_minus_1": chk.sign_reversal["drop_at_minus_1"],
                "orth_rand_max": chk.orthogonal_control["rand_max_gain"],
                "fail_reasons": chk.fail_reasons,
            },
            "three_axis_gate": gate_info,
            "decision": decision,
            "proxy_before": float(cur_rep["proxy"]["proxy_score"]),
            "proxy_after": float(new_rep["proxy"]["proxy_score"]),
            "modified_params_total": sum(p.modified_elements() for p in new_patches),
            "elapsed_sec": round(time.perf_counter() - t_start, 2),
        },
    )
    print(
        f"  [{traj_name} iter={iter_idx}] {best_cand.step_id} -> {decision} "
        f"(causal={chk.passed}, gate={gate_ok}, scale={chk.best_scale}, "
        f"proxy: {cur_rep['proxy']['proxy_score']:+.4f} -> {post_rep['proxy']['proxy_score']:+.4f})"
    )
    apply_bundle(runner.model, base_cpu_state, new_patches)
    return new_patches, new_rep


def run_track2_wsp(
    n_random_seeds: int = 3,
    track1_deltas_path: Path | None = None,
) -> dict[str, Any]:
    """Run all 3 gradient-free WSP trajectories, compare on full select_dev, run LOO, save Track 2 winner."""
    t0 = time.perf_counter()
    runner = QwenDeciderRunner()
    base_cpu_state = capture_cpu_state_dict(runner.model)
    obs_rows = load_split_rows("observe")
    dev_all = load_split_rows("select_dev")
    dev_panel = make_causal_panel_rows(dev_all, fast=True)
    dev_panel_long = dev_panel + [r for r in dev_all if r["subset"] == "long_50t"]

    log_path = EXPERIMENTS_DIR / "wsp_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    def _eval_panel() -> dict[str, Any]:
        return quick_eval_dev(runner, dev_panel, include_long=False)

    def _eval_panel_long() -> dict[str, Any]:
        return quick_eval_dev(runner, dev_panel_long, include_long=True)

    def _eval_full_dev() -> dict[str, Any]:
        return quick_eval_dev(runner, dev_all, include_long=True)

    # Base model reference floor/ceiling
    apply_bundle(runner.model, base_cpu_state, [])
    base_panel_rep = _eval_panel()
    base_panel_long_rep = quick_eval_dev(
        runner, [r for r in dev_panel_long if r["subset"] == "long_50t"], include_long=True
    )
    base_panel_with_long_rep = {**base_panel_rep, "long_50t": base_panel_long_rep["long_50t"]}

    replay_floor = float(base_panel_rep["decider_replay"]["overall_acc"]) - 0.025
    replay_ece_ceil = float(base_panel_rep["decider_replay"]["top_label_ece"]) + 0.03

    t1_path = track1_deltas_path or (CHECKPOINTS_DIR / "track1_best_deltas.pt")
    track1_patches = load_patch_bundle(t1_path).patches
    apply_bundle(runner.model, base_cpu_state, track1_patches)
    t1_panel_rep = _eval_panel()
    t1_long_only = quick_eval_dev(runner, [r for r in dev_panel_long if r["subset"] == "long_50t"], include_long=True)
    t1_panel_with_long_rep = {**t1_panel_rep, "long_50t": t1_long_only["long_50t"]}

    # =========================================================================
    # Trajectory 1: Track 1 Best -> Residual Consent-Override + Priority A + SwiGLU
    # =========================================================================
    print("\n=== Trajectory 1: Residual Consent-Override + Priority A Attn (`traj1_residual_boost_from_t1`) ===")
    t1_patches = list(track1_patches)
    t1_rep = t1_panel_rep
    apply_bundle(runner.model, base_cpu_state, t1_patches)

    # Iter 1: Residual Consent-Override & Order-Alignment SwiGLU Gate
    used1 = extract_used_neurons(t1_patches)
    cand_l1 = [19, 20, 22, 23]
    obs_s1 = collect_post_patch_observe_state(runner, obs_rows, target_layers=sorted(set(cand_l1 + [16])), include_attn=True)
    prior_w1 = _snapshot_gate_layer_weights(runner.model, cand_l1)
    _, gains1 = _measure_layer_set_gains(runner, obs_rows, cand_l1)
    pw_consent = {
        "p2_exact_consent": 6.0,
        "p3_target_mismatch": 4.0,
        "p5a_blocked_approved_same": 5.0,
        "p5b_blocked_swapped_target": 4.0,
        "p1_no_consent": 2.0,
        "p4_vague_consent": 2.0,
        "real_cf": 1.5,
        "long_50t": 2.0,
    }
    cands_1_1 = [
        build_residual_swiglu_patch(
            step_id=f"traj1_iter1_consent_swiglu_L{l}",
            runner=runner,
            obs_state=obs_s1,
            prior_layer_weights=prior_w1[l],
            layer_idx=l,
            layer_gain=gains1[l],
            exclude_neurons=used1.get(l),
            step_eta=0.65,
            pattern_weights=pw_consent,
            hypothesis=f"Override P2/P5a keyword reflex and align order via closed-form residual boosting on L{l}.",
        )
        for l in cand_l1
    ]
    t1_patches, t1_rep = _run_wsp_step(
        "traj1_residual_boost_from_t1", 1,
        "Track 1 fails P2 Allow (12.5%) & P5a Allow (50%); fit clipped logit residuals weighted 6x/5x on P2/P5a into late SwiGLU gates.",
        runner, base_cpu_state, t1_patches, t1_rep, base_panel_rep, cands_1_1,
        _eval_panel, "proxy_score", replay_floor, replay_ece_ceil, n_random_seeds, log_path, t0,
    )

    # Iter 2: Priority A Residual Attention/DeltaNet Projection
    if len(t1_patches) > len(track1_patches):
        obs_s2 = collect_post_patch_observe_state(runner, obs_rows, target_layers=[16, 19], include_attn=True)
        _, gains2 = _measure_layer_set_gains(runner, obs_rows, [16, 19])
    else:
        obs_s2 = obs_s1
        _, gains2 = _measure_layer_set_gains(runner, obs_rows, [16, 19])
    cands_1_2 = [
        build_priority_a_residual_attn_patch(
            step_id=f"traj1_iter2_priorityA_attn_L{l}",
            runner=runner,
            obs_state=obs_s2,
            layer_idx=l,
            layer_gain=gains2[l],
            step_eta=0.45,
            pattern_weights=pw_consent,
        )
        for l in [16, 19]
    ]
    t1_patches, t1_rep = _run_wsp_step(
        "traj1_residual_boost_from_t1", 2,
        "Refine Priority A Attention/DeltaNet projections (L16 out_proj, L19 o_proj) in replay-hardened null-space on residual errors.",
        runner, base_cpu_state, t1_patches, t1_rep, base_panel_rep, cands_1_2,
        _eval_panel, "proxy_score", replay_floor, replay_ece_ceil, n_random_seeds, log_path, t0,
    )

    # Iter 3: Second Complementary Residual SwiGLU Refinement on Remaining Unused Layers
    apply_bundle(runner.model, base_cpu_state, t1_patches)
    used_layers_1 = {l for p in t1_patches for l in p.target_layers if "swiglu" in p.category}
    rem_l1 = [l for l in [19, 20, 21, 22, 23] if l not in used_layers_1]
    if not rem_l1:
        rem_l1 = [21, 23]
    obs_s3 = collect_post_patch_observe_state(runner, obs_rows, target_layers=rem_l1, include_attn=False)
    prior_w3 = _snapshot_gate_layer_weights(runner.model, rem_l1)
    _, gains3 = _measure_layer_set_gains(runner, obs_rows, rem_l1)
    used1_now = extract_used_neurons(t1_patches)
    pw_bal = {
        "p2_exact_consent": 5.0,
        "p3_target_mismatch": 4.0,
        "p5a_blocked_approved_same": 4.0,
        "p5b_blocked_swapped_target": 4.0,
        "p1_no_consent": 2.0,
        "p4_vague_consent": 2.0,
        "real_cf": 2.5,
        "long_50t": 2.0,
    }
    cands_1_3 = [
        build_residual_swiglu_patch(
            step_id=f"traj1_iter3_balanced_swiglu_L{l}",
            runner=runner,
            obs_state=obs_s3,
            prior_layer_weights=prior_w3[l],
            layer_idx=l,
            layer_gain=gains3[l],
            exclude_neurons=used1_now.get(l),
            step_eta=0.45,
            pattern_weights=pw_bal,
            hypothesis=f"Balanced post-edit residual calibration & real_cf/consent harmony on L{l}.",
        )
        for l in rem_l1
    ]
    t1_patches, t1_rep = _run_wsp_step(
        "traj1_residual_boost_from_t1", 3,
        "Apply balanced second residual SwiGLU refinement on distinct downstream layer for real_cf + consent calibration.",
        runner, base_cpu_state, t1_patches, t1_rep, base_panel_rep, cands_1_3,
        _eval_panel, "proxy_score", replay_floor, replay_ece_ceil, n_random_seeds, log_path, t0,
    )

    # =========================================================================
    # Trajectory 2: Track 1 Best -> Priority C DeltaNet Decay + Long50T + Order
    # =========================================================================
    print("\n=== Trajectory 2: Priority C DeltaNet Decay + Long50T (`traj2_deltanet_long50t_from_t1`) ===")
    t2_patches = list(track1_patches)
    t2_rep_long = t1_panel_with_long_rep
    apply_bundle(runner.model, base_cpu_state, t2_patches)

    # Iter 1: Priority C Gated DeltaNet A_log / dt_bias slow-decay surgery
    dn_patch = build_deltanet_decay_patch(
        step_id="traj2_iter1_deltanet_decay",
        runner=runner,
        obs_rows=obs_rows,
        candidate_layers=[12, 13, 14, 16, 17, 18, 20, 21, 22],
        eps=0.25,
        top_k_layers=4,
    )
    t2_patches, t2_rep_long = _run_wsp_step(
        "traj2_deltanet_long50t_from_t1", 1,
        "Slow Gated DeltaNet decay (A_log, dt_bias -= 0.25) on top long_50t-sensitive layers to retain Turn-1 consent.",
        runner, base_cpu_state, t2_patches, t2_rep_long, base_panel_with_long_rep, [dn_patch],
        _eval_panel_long, "long50t_focus", replay_floor, replay_ece_ceil, n_random_seeds, log_path, t0,
    )

    # Iter 2: Long-context + Consent-Override Residual SwiGLU Specialist
    cand_l2 = [19, 20, 22, 23]
    apply_bundle(runner.model, base_cpu_state, t2_patches)
    obs_2_2 = collect_post_patch_observe_state(runner, obs_rows, target_layers=cand_l2, include_attn=False)
    prior_w2_2 = _snapshot_gate_layer_weights(runner.model, cand_l2)
    _, gains2_2 = _measure_layer_set_gains(runner, obs_rows, cand_l2)
    used2 = extract_used_neurons(t2_patches)
    pw_long = {
        "p2_exact_consent": 5.5,
        "p3_target_mismatch": 4.0,
        "p5a_blocked_approved_same": 4.5,
        "p5b_blocked_swapped_target": 4.0,
        "p1_no_consent": 2.0,
        "p4_vague_consent": 2.0,
        "long_50t": 5.0,
        "real_cf": 1.5,
    }
    cands_2_2 = [
        build_residual_swiglu_patch(
            step_id=f"traj2_iter2_long_consent_L{l}",
            runner=runner,
            obs_state=obs_2_2,
            prior_layer_weights=prior_w2_2[l],
            layer_idx=l,
            layer_gain=gains2_2[l],
            exclude_neurons=used2.get(l),
            step_eta=0.65,
            pattern_weights=pw_long,
            hypothesis=f"Joint long_50t (5x) + explicit consent override residual SwiGLU gate on L{l}.",
        )
        for l in cand_l2
    ]
    t2_patches, t2_rep_long = _run_wsp_step(
        "traj2_deltanet_long50t_from_t1", 2,
        "Boost long_50t early-consent retention (5x) + consent_5p P2/P5a via late SwiGLU residual gate.",
        runner, base_cpu_state, t2_patches, t2_rep_long, base_panel_with_long_rep, cands_2_2,
        _eval_panel_long, "proxy_plus_long", replay_floor, replay_ece_ceil, n_random_seeds, log_path, t0,
    )

    # Iter 3: Order-Equivariance & Calibration Residual SwiGLU on distinct layer
    apply_bundle(runner.model, base_cpu_state, t2_patches)
    used_layers_2 = {l for p in t2_patches for l in p.target_layers if "swiglu" in p.category}
    rem_l2 = [l for l in [19, 20, 21, 22, 23] if l not in used_layers_2] or [21, 23]
    obs_2_3 = collect_post_patch_observe_state(runner, obs_rows, target_layers=rem_l2, include_attn=False)
    prior_w2_3 = _snapshot_gate_layer_weights(runner.model, rem_l2)
    _, gains2_3 = _measure_layer_set_gains(runner, obs_rows, rem_l2)
    used2_now = extract_used_neurons(t2_patches)
    cands_2_3 = [
        build_residual_swiglu_patch(
            step_id=f"traj2_iter3_order_cal_L{l}",
            runner=runner,
            obs_state=obs_2_3,
            prior_layer_weights=prior_w2_3[l],
            layer_idx=l,
            layer_gain=gains2_3[l],
            exclude_neurons=used2_now.get(l),
            step_eta=0.40,
            pattern_weights=pw_bal,
            hypothesis=f"Distinct-layer calibration and order-equivariance SwiGLU polish on L{l}.",
        )
        for l in rem_l2
    ]
    t2_patches, _ = _run_wsp_step(
        "traj2_deltanet_long50t_from_t1", 3,
        "Final order-equivariance & calibration residual SwiGLU polish on distinct layer.",
        runner, base_cpu_state, t2_patches, t2_rep_long, base_panel_with_long_rep, cands_2_3,
        _eval_panel, "proxy_score", replay_floor, replay_ece_ceil, n_random_seeds, log_path, t0,
    )

    # =========================================================================
    # Trajectory 3: From-Scratch Unedited Base -> Multi-Layer Specialist Pipeline
    # =========================================================================
    print("\n=== Trajectory 3: From-Scratch Multi-Layer Specialist (`traj3_from_scratch_multilayer`) ===")
    t3_patches: list[WeightPatch] = []
    t3_rep = base_panel_rep
    apply_bundle(runner.model, base_cpu_state, [])

    # Iter 1: Pure Domain-Risk Specialist SwiGLU Gate (with base_zero_deltas!) on [15, 19]
    cand_3_1 = [15, 19]
    obs_3_1 = collect_post_patch_observe_state(runner, obs_rows, target_layers=cand_3_1, include_attn=False)
    prior_w3_1 = _snapshot_gate_layer_weights(runner.model, cand_3_1)
    base_m_n, gains3_1 = _measure_layer_set_gains(runner, obs_rows, cand_3_1)
    rc_obs_indices = [i for i, m in enumerate(obs_3_1["meta_automode"]) if m["subset"] == "real_cf"]
    base_m_r = float(np.mean(obs_3_1["m_rev"][rc_obs_indices]))
    cands_3_1 = [
        build_domain_specialist_swiglu_zero_artifact(
            step_id=f"traj3_iter1_domain_spec_L{l}",
            runner=runner,
            obs_state=obs_3_1,
            prior_layer_weights=prior_w3_1[l],
            layer_idx=l,
            layer_gain=gains3_1[l],
            base_margin_norm=base_m_n,
            base_margin_rev=base_m_r,
            w_c5=0.5,
            target_logit_std=1.5,
        )
        for l in cand_3_1
    ]
    t3_patches, t3_rep = _run_wsp_step(
        "traj3_from_scratch_multilayer", 1,
        "From unedited base: install pure Domain-Risk Specialist SwiGLU gate (zero-artifact base_zero_deltas) on L15/L19.",
        runner, base_cpu_state, t3_patches, t3_rep, base_panel_rep, cands_3_1,
        _eval_panel, "proxy_score", replay_floor, replay_ece_ceil, n_random_seeds, log_path, t0,
    )

    # Iter 2: Consent-Override & Target-Mismatch Residual SwiGLU Specialist on [19, 20, 22] \ used
    apply_bundle(runner.model, base_cpu_state, t3_patches)
    used_l3 = {l for p in t3_patches for l in p.target_layers}
    cand_3_2 = [l for l in [19, 20, 22] if l not in used_l3] or [20, 22]
    obs_3_2 = collect_post_patch_observe_state(runner, obs_rows, target_layers=cand_3_2, include_attn=False)
    prior_w3_2 = _snapshot_gate_layer_weights(runner.model, cand_3_2)
    _, gains3_2 = _measure_layer_set_gains(runner, obs_rows, cand_3_2)
    pw_c5_spec = {
        "p2_exact_consent": 6.0,
        "p3_target_mismatch": 4.5,
        "p5a_blocked_approved_same": 5.0,
        "p5b_blocked_swapped_target": 4.5,
        "p1_no_consent": 2.0,
        "p4_vague_consent": 2.0,
        "real_cf": 1.0,
        "long_50t": 2.0,
    }
    cands_3_2 = [
        build_residual_swiglu_patch(
            step_id=f"traj3_iter2_consent_spec_L{l}",
            runner=runner,
            obs_state=obs_3_2,
            prior_layer_weights=prior_w3_2[l],
            layer_idx=l,
            layer_gain=gains3_2[l],
            exclude_neurons=extract_used_neurons(t3_patches).get(l),
            step_eta=0.70,
            pattern_weights=pw_c5_spec,
            hypothesis=f"Dedicated Consent-Override & Target-Mismatch specialist SwiGLU gate on L{l}.",
        )
        for l in cand_3_2
    ]
    t3_patches, t3_rep = _run_wsp_step(
        "traj3_from_scratch_multilayer", 2,
        "Add dedicated Consent-Override & Target-Mismatch SwiGLU specialist downstream of domain gate.",
        runner, base_cpu_state, t3_patches, t3_rep, base_panel_rep, cands_3_2,
        _eval_panel, "proxy_score", replay_floor, replay_ece_ceil, n_random_seeds, log_path, t0,
    )

    # Iter 3: Final Calibration & Order-Equivariance Residual SwiGLU on [22, 23] \ used
    apply_bundle(runner.model, base_cpu_state, t3_patches)
    used_l3 = {l for p in t3_patches for l in p.target_layers}
    cand_3_3 = [l for l in [22, 23] if l not in used_l3] or [23]
    obs_3_3 = collect_post_patch_observe_state(runner, obs_rows, target_layers=cand_3_3, include_attn=False)
    prior_w3_3 = _snapshot_gate_layer_weights(runner.model, cand_3_3)
    _, gains3_3 = _measure_layer_set_gains(runner, obs_rows, cand_3_3)
    cands_3_3 = [
        build_residual_swiglu_patch(
            step_id=f"traj3_iter3_cal_order_L{l}",
            runner=runner,
            obs_state=obs_3_3,
            prior_layer_weights=prior_w3_3[l],
            layer_idx=l,
            layer_gain=gains3_3[l],
            exclude_neurons=extract_used_neurons(t3_patches).get(l),
            step_eta=0.50,
            pattern_weights=pw_bal,
            hypothesis=f"Late-layer order-equivariance and calibration polish on L{l}.",
        )
        for l in cand_3_3
    ]
    t3_patches, t3_rep = _run_wsp_step(
        "traj3_from_scratch_multilayer", 3,
        "Apply final calibration and order-equivariance polish in L22/L23.",
        runner, base_cpu_state, t3_patches, t3_rep, base_panel_rep, cands_3_3,
        _eval_panel, "proxy_score", replay_floor, replay_ece_ceil, n_random_seeds, log_path, t0,
    )

    # =========================================================================
    # Evaluate All Trajectories + Track 1 on FULL select_dev (200 rows)
    # =========================================================================
    print("\n=== Evaluating All 3 Trajectories on FULL select_dev (n=200) ===")
    trajectories: list[tuple[str, list[WeightPatch]]] = [
        ("track1_baseline", list(track1_patches)),
        ("traj1_residual_boost_from_t1", t1_patches),
        ("traj2_deltanet_long50t_from_t1", t2_patches),
        ("traj3_from_scratch_multilayer", t3_patches),
    ]
    traj_reports: list[dict[str, Any]] = []
    for name, plist in trajectories:
        apply_bundle(runner.model, base_cpu_state, plist)
        wd = compute_weight_diff_stats(runner.model, base_cpu_state)
        rep = _eval_full_dev()
        traj_reports.append(
            {
                "trajectory": name,
                "step_ids": [p.step_id for p in plist],
                "pct_modified_params": wd["pct_modified_params"],
                "modified_params": wd["modified_params"],
                "total_delta_frobenius_norm": wd["total_delta_frobenius_norm"],
                "report": rep,
            }
        )
        print(
            f"  [{name}] steps={[p.step_id for p in plist]} | proxy={rep['proxy']['proxy_score']:+.5f} "
            f"| rc_auc={rep['real_cf']['auroc']} rec={rep['real_cf']['recall_at_05']} fpr={rep['real_cf']['fpr_at_05']} "
            f"| c5_acc={rep['consent_5p']['overall_acc']} P2={rep['consent_5p']['p2_allow_rate']} P3={rep['consent_5p']['p3_block_rate']} "
            f"| l50={rep['long_50t']['overall_acc']} | rep={rep['decider_replay']['overall_acc']}"
        )

    eligible_trajs = [
        (i, tr) for i, tr in enumerate(traj_reports)
        if float(tr["report"]["decider_replay"]["overall_acc"]) >= 0.50
    ]
    if not eligible_trajs:
        eligible_trajs = list(enumerate(traj_reports))
    win_idx, win_info = max(eligible_trajs, key=lambda pair: float(pair[1]["report"]["proxy"]["proxy_score"]))
    winner_name = win_info["trajectory"]
    winner_patches = trajectories[win_idx][1]
    print(f"\n[Track 2] Selected winner trajectory on full select_dev: {winner_name} (proxy={win_info['report']['proxy']['proxy_score']:+.5f})")

    # Leave-One-Out Ablation on FULL select_dev
    print("[Track 2] Running Leave-One-Out ablation on full select_dev...")
    t0a_json = json.loads((EXPERIMENTS_DIR / "track0a_zeroshot_dev.json").read_text(encoding="utf-8"))
    full_replay_floor = float(t0a_json["decider_replay"]["overall_acc"]) - 0.025
    pruned_patches, loo_summary = leave_one_out_ablation(
        model=runner.model,
        base_cpu_state=base_cpu_state,
        patches=winner_patches,
        eval_fn=_eval_full_dev,
        replay_acc_floor=full_replay_floor,
        tol=0.005,
    )

    # Final official evaluate_runner on pruned Track 2 winner
    apply_bundle(runner.model, base_cpu_state, pruned_patches)
    weight_diff = compute_weight_diff_stats(runner.model, base_cpu_state)
    final_report = evaluate_runner(
        runner,
        dev_all,
        mode="plain",
        include_recap=False,
        eval_recap_ablation=True,
    )
    final_report["elapsed_sec"] = round(time.perf_counter() - t0, 2)

    # Persist Track 2 reports & artifacts
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    sum_md = [
        "# Track 2 Multi-Trajectory WSP Comparison (`select_dev`, n=200)",
        "",
        "| Trajectory | Steps Kept | `% modified` | Proxy Score | `real_cf` AUROC (Rec/FPR) | `consent_5p` Acc (`P2/P3`) | `long_50t` Acc | `decider_replay` Acc / ECE |",
        "|---|---|---:|---:|---|---|---:|---|",
    ]
    for tr in traj_reports:
        r = tr["report"]
        rc, c5, lg, rp = r["real_cf"], r["consent_5p"], r["long_50t"], r["decider_replay"]
        sum_md.append(
            f"| `{tr['trajectory']}` | `{len(tr['step_ids'])}` (`{', '.join(tr['step_ids'])}`) | "
            f"`{tr['pct_modified_params']}%` | `{r['proxy']['proxy_score']}` | "
            f"`{rc['auroc']}` (`{rc['recall_at_05']} / {rc['fpr_at_05']}`) | "
            f"`{c5['overall_acc']}` (`{c5['p2_allow_rate']} / {c5['p3_block_rate']}`) | "
            f"`{lg['overall_acc']}` | `{rp['overall_acc']} / {rp['top_label_ece']}` |"
        )
    sum_md.extend(["", f"**Selected Winner**: `{winner_name}` -> **After LOO Pruning**: `{[p.step_id for p in pruned_patches]}`"])
    (EXPERIMENTS_DIR / "track2_trajectories_summary.md").write_text("\n".join(sum_md) + "\n", encoding="utf-8")
    (EXPERIMENTS_DIR / "track2_trajectories_summary.json").write_text(
        json.dumps({"winner": winner_name, "trajectories": traj_reports, "loo_final_steps": [p.step_id for p in pruned_patches]}, indent=2) + "\n",
        encoding="utf-8",
    )
    (EXPERIMENTS_DIR / "track2_loo_ablation.json").write_text(
        json.dumps(loo_summary, indent=2) + "\n", encoding="utf-8"
    )

    bundle = PatchBundle(
        patches=pruned_patches,
        metadata={"track": "track2", "winner_trajectory": winner_name, "run_name": "track2_best_dev"},
    )
    save_patch_bundle(
        bundle,
        json_spec_path=EXPERIMENTS_DIR / "track2_patch_spec.json",
        pt_deltas_path=CHECKPOINTS_DIR / "track2_best_deltas.pt",
        extra_summary={"weight_diff": weight_diff},
    )

    payload = {"name": "track2_best_dev", "split": "select_dev", "weight_diff": weight_diff, **final_report}
    (EXPERIMENTS_DIR / "track2_best_dev.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    md_text = format_markdown_report("track2_best_dev", "select_dev", final_report, weight_diff)
    (EXPERIMENTS_DIR / "track2_best_dev.md").write_text(md_text, encoding="utf-8")
    print(md_text)

    del runner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Track 2 multi-trajectory gradient-free WSP & LOO.")
    ap.add_argument("--n-random-seeds", type=int, default=3)
    ap.add_argument("--track1-deltas", default=None)
    args = ap.parse_args()
    run_track2_wsp(
        n_random_seeds=args.n_random_seeds,
        track1_deltas_path=Path(args.track1_deltas) if args.track1_deltas else None,
    )


if __name__ == "__main__":
    main()
