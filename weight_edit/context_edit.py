"""Track 1 gradient-free closed-form weight editing for Qwen3.5-0.8B.

Implements:
  - Observe-only activation collection (normal/reversed Plain, Teacher Oracle, body tokens, decider_replay) with disk cache
  - Replay-hardened AlphaEdit null-space projection (preserving body & decider_replay activations)
  - Step 1A: Context-to-Weight rank-r closed-form distillation on Priority A (o_proj/out_proj) and Priority B (mlp.down_proj)
  - Step 1B: Within-family contrastive directions (v_domain, v_consent) + calibrated 4-neuron SwiGLU order-equivariant gate
  - Analytical layer logit transfer gain calibration, Layer Freezing ablation, 3-point causal sanity checks, and LOO ablation.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from automode.questions import (
    Q_BLOCK_BINARY,
    build_chat_oracle_prompt,
    build_decider_prompt,
    build_oracle_fewshot_aligned_prompt,
    build_oracle_prompt,
    build_plain_prompt,
)
from automode.view import render_view
from eval.evaluate import evaluate_runner, format_markdown_report, load_split_rows
from eval.model_runner import QwenDeciderRunner, capture_cpu_state_dict, compute_weight_diff_stats
from weight_edit.causal_check import (
    PatchBundle,
    WeightPatch,
    apply_bundle,
    leave_one_out_ablation,
    make_causal_panel_rows,
    quick_eval_dev,
    run_three_point_causal_check,
    save_patch_bundle,
)

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = ROOT / "experiments"
CHECKPOINTS_DIR = ROOT / "checkpoints"

TRACK1_LAYERS = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
PRIORITY_A_LAYERS = [11, 15, 19, 16, 18, 20]
PRIORITY_B_LAYERS = [15, 16, 17, 18, 19, 20]

STRICT_CONSENT_PATTERNS = {
    "p2_exact_consent",
    "p3_target_mismatch",
    "p5a_blocked_approved_same",
    "p5b_blocked_swapped_target",
}
CONTRASTIVE_PAIRS = {
    frozenset({"p2_exact_consent", "p3_target_mismatch"}),
    frozenset({"p5a_blocked_approved_same", "p5b_blocked_swapped_target"}),
}


@dataclass
class ObserveActivations:
    """CPU float32 activation buffers collected strictly from the `observe` split."""

    layers: list[int]
    plain_norm: dict[str, dict[int, torch.Tensor]]
    plain_rev: dict[str, dict[int, torch.Tensor]]
    teacher_norm: dict[str, dict[int, torch.Tensor]]
    preserve: dict[str, dict[int, torch.Tensor]]
    replay_only: dict[str, dict[int, torch.Tensor]]
    mlp_preserve_energy: dict[int, torch.Tensor]
    meta_automode: list[dict[str, Any]]


def _build_teacher_prompt_text(
    context_text: str,
    teacher_mode: str,
    tokenizer: Any,
    reverse_options: bool = False,
) -> str:
    if teacher_mode == "oracle_chat":
        return build_chat_oracle_prompt(
            context_text, tokenizer, question_spec=Q_BLOCK_BINARY, reverse_options=reverse_options
        ).text
    if teacher_mode == "oracle_aligned":
        return build_oracle_fewshot_aligned_prompt(
            context_text, question_spec=Q_BLOCK_BINARY, reverse_options=reverse_options
        ).text
    if teacher_mode == "oracle":
        return build_oracle_prompt(
            context_text, question_spec=Q_BLOCK_BINARY, reverse_options=reverse_options
        ).text
    return build_plain_prompt(
        context_text, question_spec=Q_BLOCK_BINARY, reverse_options=reverse_options
    ).text


def _sample_body_positions(
    input_ids: torch.Tensor,
    paren_tok_id: int | None,
    body_tokens_per_prompt: int = 10,
) -> list[int]:
    seq_len = int(input_ids.shape[1])
    if seq_len <= 32:
        return []
    grid = [
        seq_len // 8,
        seq_len // 4,
        seq_len // 2,
        (3 * seq_len) // 4,
        seq_len - 16,
        seq_len - 12,
        seq_len - 8,
        seq_len - 5,
        seq_len - 3,
        seq_len - 2,
    ]
    grid_clean = sorted({p for p in grid if 0 <= p < seq_len - 1})[:body_tokens_per_prompt]
    paren_sample: list[int] = []
    if paren_tok_id is not None and seq_len > 2:
        hits = (input_ids[0, : seq_len - 1] == paren_tok_id).nonzero(as_tuple=False).squeeze(-1).tolist()
        paren_sample = [int(p) for p in hits[-2:] if 0 <= int(p) < seq_len - 1]
    return sorted(set(grid_clean) | set(paren_sample))


def collect_observe_activations(
    runner: QwenDeciderRunner,
    layers: list[int] = TRACK1_LAYERS,
    teacher_mode: str = "oracle_chat",
    body_tokens_per_prompt: int = 10,
    cache_path: Path | None = None,
    refresh_cache: bool = False,
) -> ObserveActivations:
    """Collect slot & preservation activations strictly on the `observe` split."""
    if cache_path is not None and cache_path.exists() and not refresh_cache:
        print(f"[Track 1] Loading cached observe activations from {cache_path}...")
        raw = torch.load(cache_path, map_location="cpu", weights_only=False)
        obs_cached = ObserveActivations(**raw) if isinstance(raw, dict) else raw
        if list(obs_cached.layers) == list(layers):
            return obs_cached
        print("[Track 1] Cache layer mismatch; re-collecting observe activations...")

    obs_rows = load_split_rows("observe")
    auto_rows = [r for r in obs_rows if r["subset"] in ("real_cf", "consent_5p", "long_50t")]
    replay_rows = [r for r in obs_rows if r["subset"] == "decider_replay"]

    paren_ids = runner.tokenizer.encode("(", add_special_tokens=False)
    paren_tok_id = int(paren_ids[0]) if len(paren_ids) == 1 else None

    def _empty_bucket() -> dict[str, dict[int, list[torch.Tensor]]]:
        return {
            "attn_in": {l: [] for l in layers},
            "attn_out": {l: [] for l in layers},
            "mlp_in": {l: [] for l in layers},
            "mlp_hid": {l: [] for l in layers},
            "mlp_out": {l: [] for l in layers},
            "resid": {l: [] for l in layers},
        }

    b_norm = _empty_bucket()
    b_rev = _empty_bucket()
    b_teach = _empty_bucket()
    b_pres = _empty_bucket()
    b_rep = _empty_bucket()

    cur_a_in: dict[int, torch.Tensor] = {}
    cur_a_out: dict[int, torch.Tensor] = {}
    cur_m_in: dict[int, torch.Tensor] = {}
    cur_m_hid: dict[int, torch.Tensor] = {}
    cur_m_out: dict[int, torch.Tensor] = {}
    cur_res: dict[int, torch.Tensor] = {}
    capture_positions: list[int] = [-1]

    handles = []
    for l in layers:
        layer = runner.model.model.layers[l]
        proj = layer.linear_attn.out_proj if hasattr(layer, "linear_attn") else layer.self_attn.o_proj

        def _mk_pre(store: dict[int, torch.Tensor], idx: int):
            def _hook(_mod, inp):
                store[idx] = inp[0][0, capture_positions, :].detach().float().cpu()
            return _hook

        def _mk_post(store: dict[int, torch.Tensor], idx: int):
            def _hook(_mod, _inp, out):
                t = out[0] if isinstance(out, tuple) else out
                store[idx] = t[0, capture_positions, :].detach().float().cpu()
            return _hook

        handles.append(proj.register_forward_pre_hook(_mk_pre(cur_a_in, l)))
        handles.append(proj.register_forward_hook(_mk_post(cur_a_out, l)))
        handles.append(layer.mlp.register_forward_pre_hook(_mk_pre(cur_m_in, l)))
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(_mk_pre(cur_m_hid, l)))
        handles.append(layer.mlp.down_proj.register_forward_hook(_mk_post(cur_m_out, l)))
        handles.append(layer.register_forward_hook(_mk_post(cur_res, l)))

    def _run_and_append(
        prompt_text: str,
        bucket: dict[str, dict[int, list[torch.Tensor]]],
        sample_body: bool = False,
        is_replay: bool = False,
    ) -> None:
        nonlocal capture_positions
        input_ids = runner.encode_prompt(prompt_text)
        seq_len = int(input_ids.shape[1])
        body_pos: list[int] = []
        if is_replay:
            cap = sorted({max(0, seq_len - 3), max(0, seq_len - 2), seq_len - 1})
            capture_positions = cap
        else:
            if sample_body:
                body_pos = _sample_body_positions(input_ids, paren_tok_id, body_tokens_per_prompt=body_tokens_per_prompt)
            capture_positions = body_pos + [seq_len - 1]

        with torch.inference_mode():
            _ = runner.model.model(input_ids=input_ids, use_cache=False)

        for l in layers:
            bucket["attn_in"][l].append(cur_a_in[l][-1])
            bucket["attn_out"][l].append(cur_a_out[l][-1])
            bucket["mlp_in"][l].append(cur_m_in[l][-1])
            bucket["mlp_hid"][l].append(cur_m_hid[l][-1])
            bucket["mlp_out"][l].append(cur_m_out[l][-1])
            bucket["resid"][l].append(cur_res[l][-1])
            if is_replay:
                for p_i in range(len(capture_positions)):
                    b_pres["attn_in"][l].append(cur_a_in[l][p_i])
                    b_pres["attn_out"][l].append(cur_a_out[l][p_i])
                    b_pres["mlp_in"][l].append(cur_m_in[l][p_i])
                    b_pres["mlp_hid"][l].append(cur_m_hid[l][p_i])
                    b_pres["mlp_out"][l].append(cur_m_out[l][p_i])
                    b_pres["resid"][l].append(cur_res[l][p_i])
            elif body_pos:
                for p_i in range(len(body_pos)):
                    b_pres["attn_in"][l].append(cur_a_in[l][p_i])
                    b_pres["attn_out"][l].append(cur_a_out[l][p_i])
                    b_pres["mlp_in"][l].append(cur_m_in[l][p_i])
                    b_pres["mlp_hid"][l].append(cur_m_hid[l][p_i])
                    b_pres["mlp_out"][l].append(cur_m_out[l][p_i])
                    b_pres["resid"][l].append(cur_res[l][p_i])

    meta_automode: list[dict[str, Any]] = []
    try:
        for r in auto_rows:
            budget = 3200 if r["subset"] == "long_50t" else 2048
            ctx = render_view(r, tok=runner.tokenizer, budget=budget, include_recap=False).text
            p_norm = build_plain_prompt(ctx, Q_BLOCK_BINARY, reverse_options=False).text
            p_rev = build_plain_prompt(ctx, Q_BLOCK_BINARY, reverse_options=True).text
            p_teach = _build_teacher_prompt_text(ctx, teacher_mode, runner.tokenizer, reverse_options=False)

            _run_and_append(p_norm, b_norm, sample_body=True, is_replay=False)
            _run_and_append(p_rev, b_rev, sample_body=False, is_replay=False)
            _run_and_append(p_teach, b_teach, sample_body=False, is_replay=False)
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
            _run_and_append(pb.text, b_rep, sample_body=False, is_replay=True)
    finally:
        for h in handles:
            h.remove()

    def _stack(b: dict[str, dict[int, list[torch.Tensor]]]) -> dict[str, dict[int, torch.Tensor]]:
        return {k: {l: torch.stack(v, dim=0) for l, v in by_l.items()} for k, by_l in b.items()}

    s_pres = _stack(b_pres)
    s_rep = _stack(b_rep)
    mlp_energy = {l: torch.mean(s_pres["mlp_hid"][l] ** 2, dim=0) for l in layers}

    obs = ObserveActivations(
        layers=list(layers),
        plain_norm=_stack(b_norm),
        plain_rev=_stack(b_rev),
        teacher_norm=_stack(b_teach),
        preserve=s_pres,
        replay_only=s_rep,
        mlp_preserve_energy=mlp_energy,
        meta_automode=meta_automode,
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "layers": obs.layers,
                "plain_norm": obs.plain_norm,
                "plain_rev": obs.plain_rev,
                "teacher_norm": obs.teacher_norm,
                "preserve": obs.preserve,
                "replay_only": obs.replay_only,
                "mlp_preserve_energy": obs.mlp_preserve_energy,
                "meta_automode": obs.meta_automode,
            },
            cache_path,
        )
        print(f"[Track 1] Saved observe activations cache to {cache_path}.")
    return obs


def build_null_space_projector(
    k_preserve: torch.Tensor,  # [N_0, d_in]
    variance_keep: float = 0.95,
    max_rank_frac: float = 0.50,
    k_replay_extra: torch.Tensor | None = None,
    replay_weight: float = 3.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Construct replay-hardened AlphaEdit null-space projector P = I - U_r0 U_r0^T in float64."""
    k_pres_d = k_preserve.double()
    if k_replay_extra is not None and len(k_replay_extra) > 0:
        k_cat = torch.cat([k_pres_d, float(replay_weight) * k_replay_extra.double()], dim=0)
    else:
        k_cat = k_pres_d
    k_mat = k_cat.T  # [d_in, N_tot]
    d_in, n_tot = k_mat.shape
    u, s, _ = torch.linalg.svd(k_mat, full_matrices=False)
    s2 = s ** 2
    total_var = float(s2.sum().item()) + 1e-12
    cum = torch.cumsum(s2, dim=0) / total_var
    r0 = int(torch.searchsorted(cum, torch.tensor(variance_keep, dtype=torch.float64)).item()) + 1
    r0 = max(1, min(r0, int(d_in * max_rank_frac), n_tot - 1))
    u_r0 = u[:, :r0]
    p_null = torch.eye(d_in, dtype=torch.float64) - (u_r0 @ u_r0.T)

    k_pres_mat = k_pres_d.T
    ratio = float(torch.linalg.norm(p_null @ k_pres_mat).item() / (float(torch.linalg.norm(k_pres_mat).item()) + 1e-12))
    replay_ratio = ratio
    if k_replay_extra is not None and len(k_replay_extra) > 0:
        k_rep_mat = k_replay_extra.double().T
        replay_ratio = float(torch.linalg.norm(p_null @ k_rep_mat).item() / (float(torch.linalg.norm(k_rep_mat).item()) + 1e-12))

    return p_null, {
        "d_in": d_in,
        "n_preserve": int(k_pres_d.shape[0]),
        "n_replay_extra": int(k_replay_extra.shape[0]) if k_replay_extra is not None else 0,
        "replay_weight": float(replay_weight) if k_replay_extra is not None else 0.0,
        "null_rank_removed": r0,
        "variance_explained_removed": round(float(cum[r0 - 1].item()), 4),
        "residual_norm_ratio": round(ratio, 4),
        "replay_residual_norm_ratio": round(replay_ratio, 4),
    }


def solve_alphaedit_update(
    target_delta_out: torch.Tensor,  # [N_t, d_out]
    keys_target: torch.Tensor,       # [N_t, d_in]
    p_null: torch.Tensor,            # [d_in, d_in]
    rank: int = 2,
    lam: float = 300.0,
    alpha: float = 1.0,
    col_mask: torch.Tensor | None = None,  # [d_in] bool
) -> torch.Tensor:
    """Closed-form AlphaEdit rank-r update DeltaW [d_out, d_in] lying in range(P_null)."""
    r_mat = target_delta_out.double().T  # [d_out, N_t]
    k_mat = keys_target.double().T       # [d_in, N_t]

    if col_mask is not None:
        idx = torch.nonzero(col_mask, as_tuple=False).squeeze(-1)
        k_sub = k_mat[idx, :]
        p_sub = p_null[idx][:, idx]
        k_proj = p_sub @ k_sub
        gram = k_proj @ k_proj.T + lam * torch.eye(len(idx), dtype=torch.float64)
        w_sub_ls = torch.linalg.solve(gram, k_proj @ r_mat.T).T @ p_sub
        u, s, vh = torch.linalg.svd(w_sub_ls, full_matrices=False)
        rr = min(rank, len(s))
        w_sub = (u[:, :rr] * s[:rr].unsqueeze(0)) @ vh[:rr, :]
        w_full = torch.zeros((r_mat.shape[0], k_mat.shape[0]), dtype=torch.float32)
        w_full[:, idx] = (alpha * w_sub).float()
        return w_full

    k_proj = p_null @ k_mat
    gram = k_proj @ k_proj.T + lam * torch.eye(k_mat.shape[0], dtype=torch.float64)
    w_ls = torch.linalg.solve(gram, k_proj @ r_mat.T).T @ p_null
    u, s, vh = torch.linalg.svd(w_ls, full_matrices=False)
    rr = min(rank, len(s))
    w_r = (u[:, :rr] * s[:rr].unsqueeze(0)) @ vh[:rr, :]
    return (alpha * w_r).float()


def solve_closed_form_ridge(
    x: np.ndarray,
    y: np.ndarray,
    lam: float = 100.0,
    center: bool = True,
    sample_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Exact closed-form weighted ridge regression in float64."""
    xd = np.asarray(x, dtype=np.float64)
    yd = np.asarray(y, dtype=np.float64)
    if sample_weights is None:
        sw = np.ones(len(yd), dtype=np.float64)
    else:
        sw = np.asarray(sample_weights, dtype=np.float64)
    sw = sw / np.mean(sw)

    if center:
        mu_x = np.average(xd, axis=0, weights=sw)
        mu_y = float(np.average(yd, weights=sw))
        xc = xd - mu_x
        yc = yd - mu_y
    else:
        mu_x = np.zeros(xd.shape[1], dtype=np.float64)
        mu_y = 0.0
        xc = xd
        yc = yd

    xw = xc * sw[:, None]
    gram = xw.T @ xc + lam * np.eye(xd.shape[1], dtype=np.float64)
    w = np.linalg.solve(gram, xw.T @ yc)
    bias = float(mu_y - mu_x @ w) if center else 0.0
    return w, bias


def select_ridge_lambda_cv(
    x: np.ndarray,
    y_binary: np.ndarray,
    lambdas: tuple[float, ...] = (10.0, 40.0, 150.0, 500.0, 1500.0),
    n_folds: int = 4,
    sample_weights: np.ndarray | None = None,
) -> tuple[float, float]:
    """Select ridge lambda strictly via stratified internal CV inside `observe`."""
    idx_pos = np.where(y_binary == 1)[0]
    idx_neg = np.where(y_binary == 0)[0]
    best_lam = lambdas[len(lambdas) // 2]
    best_auc = -1.0

    for lam in lambdas:
        oof = np.zeros(len(y_binary), dtype=np.float64)
        for fold in range(n_folds):
            val_pos = idx_pos[fold::n_folds]
            val_neg = idx_neg[fold::n_folds]
            val_idx = np.concatenate([val_pos, val_neg])
            tr_mask = np.ones(len(y_binary), dtype=bool)
            tr_mask[val_idx] = False
            tr_idx = np.where(tr_mask)[0]
            sw_tr = sample_weights[tr_idx] if sample_weights is not None else None
            w, b = solve_closed_form_ridge(
                x[tr_idx], 2.0 * y_binary[tr_idx] - 1.0, lam=lam, center=True, sample_weights=sw_tr
            )
            oof[val_idx] = x[val_idx] @ w + b
        auc = float(roc_auc_score(y_binary, oof)) if len(np.unique(y_binary)) > 1 else 0.5
        if auc > best_auc:
            best_auc = auc
            best_lam = lam
    return best_lam, best_auc


def _select_within_family_pairs_and_ridge(
    x_mat: np.ndarray,
    meta_rows: list[dict[str, Any]],
    indices: list[int],
    lambdas: tuple[float, ...] = (30.0, 100.0, 300.0, 1000.0),
    p23_pair_boost: float = 1.0,
    n_folds: int = 4,
) -> tuple[np.ndarray, float, float]:
    """Fit zero-intercept ridge strictly on within-family paired differences with family-grouped CV."""
    xd = np.asarray(x_mat, dtype=np.float64)
    fams: dict[str, list[int]] = {}
    for i in indices:
        fid = str(meta_rows[i].get("family", meta_rows[i]["id"].rsplit("_", 1)[0]))
        fams.setdefault(fid, []).append(i)
    fam_names = sorted(fams.keys())
    k_folds = max(2, min(n_folds, len(fam_names)))

    def _pairs_from_fams(sel_fams: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dx_l, dy_l, dw_l = [], [], []
        for fn in sel_fams:
            idxs = fams[fn]
            pos = [i for i in idxs if int(meta_rows[i]["y"]) == 1]
            neg = [i for i in idxs if int(meta_rows[i]["y"]) == 0]
            denom = max(1, len(pos) * len(neg))
            for pi in pos:
                for ni in neg:
                    pair_pats = frozenset({str(meta_rows[pi].get("pattern", "")), str(meta_rows[ni].get("pattern", ""))})
                    pw = (p23_pair_boost if pair_pats in CONTRASTIVE_PAIRS else 1.0) / denom
                    diff = xd[pi] - xd[ni]
                    dx_l.append(diff); dy_l.append(1.0); dw_l.append(pw)
                    dx_l.append(-diff); dy_l.append(-1.0); dw_l.append(pw)
        return np.asarray(dx_l, dtype=np.float64), np.asarray(dy_l, dtype=np.float64), np.asarray(dw_l, dtype=np.float64)

    y_all = np.array([int(meta_rows[i]["y"]) for i in indices], dtype=np.int64)
    pos_in_slice = {idx: pos for pos, idx in enumerate(indices)}
    best_lam = lambdas[len(lambdas) // 2]
    best_auc = -1.0

    for lam in lambdas:
        oof = np.zeros(len(indices), dtype=np.float64)
        for fold in range(k_folds):
            val_fams = {fn for j, fn in enumerate(fam_names) if j % k_folds == fold}
            tr_fams = [fn for fn in fam_names if fn not in val_fams]
            if not tr_fams or not val_fams:
                continue
            dx_tr, dy_tr, dw_tr = _pairs_from_fams(tr_fams)
            w_f, _ = solve_closed_form_ridge(dx_tr, dy_tr, lam=lam, center=False, sample_weights=dw_tr)
            for fn in val_fams:
                for idx in fams[fn]:
                    oof[pos_in_slice[idx]] = float(xd[idx] @ w_f)
        auc = float(roc_auc_score(y_all, oof)) if len(np.unique(y_all)) > 1 else 0.5
        if auc > best_auc:
            best_auc = auc
            best_lam = lam

    dx_full, dy_full, dw_full = _pairs_from_fams(fam_names)
    w_final, _ = solve_closed_form_ridge(dx_full, dy_full, lam=best_lam, center=False, sample_weights=dw_full)
    return w_final, best_lam, best_auc


def _param_name_attn(layer_idx: int, model: torch.nn.Module) -> str:
    layer = model.model.layers[layer_idx]
    if hasattr(layer, "linear_attn"):
        return f"model.layers.{layer_idx}.linear_attn.out_proj.weight"
    return f"model.layers.{layer_idx}.self_attn.o_proj.weight"


def build_step1a_c2w_candidates(
    model: torch.nn.Module,
    obs: ObserveActivations,
    null_attn: dict[int, torch.Tensor],
    null_mlp: dict[int, torch.Tensor],
) -> list[WeightPatch]:
    """Step 1A: Context-to-Weight distillation candidates on Priority A & B layers."""
    patches: list[WeightPatch] = []

    for l in PRIORITY_A_LAYERS:
        k_plain = obs.plain_norm["attn_in"][l]
        r_diff = obs.teacher_norm["attn_out"][l] - obs.plain_norm["attn_out"][l]
        delta = solve_alphaedit_update(
            target_delta_out=r_diff,
            keys_target=k_plain,
            p_null=null_attn[l],
            rank=2,
            lam=400.0,
            alpha=0.85,
        )
        pname = _param_name_attn(l, model)
        patches.append(
            WeightPatch(
                step_id=f"step1a_c2w_attn_L{l}",
                category="step1a_c2w",
                method="alphaedit_c2w_rank2",
                target_layers=[l],
                rank=2,
                alpha=0.85,
                hypothesis=f"Distill Teacher-Oracle minus Plain output shift into {pname} via rank-2 null-space projection.",
                deltas={pname: delta},
            )
        )

    for l in [15, 17, 19, 20]:
        k_plain = obs.plain_norm["mlp_hid"][l]
        r_diff = obs.teacher_norm["mlp_out"][l] - obs.plain_norm["mlp_out"][l]
        slot_energy = torch.mean(k_plain ** 2, dim=0)
        pres_energy = obs.mlp_preserve_energy[l] + 1e-4
        snr = slot_energy / pres_energy
        top_cols = torch.topk(snr, k=64).indices
        col_mask = torch.zeros(k_plain.shape[1], dtype=torch.bool)
        col_mask[top_cols] = True
        delta = solve_alphaedit_update(
            target_delta_out=r_diff,
            keys_target=k_plain,
            p_null=null_mlp[l],
            rank=2,
            lam=400.0,
            alpha=0.85,
            col_mask=col_mask,
        )
        pname = f"model.layers.{l}.mlp.down_proj.weight"
        patches.append(
            WeightPatch(
                step_id=f"step1a_c2w_mlp_L{l}",
                category="step1a_c2w",
                method="alphaedit_c2w_sparse64",
                target_layers=[l],
                rank=2,
                alpha=0.85,
                hypothesis=f"Distill Teacher-Oracle minus Plain MLP shift into top-64 columns of {pname}.",
                deltas={pname: delta},
            )
        )
    return patches


def build_step1b_domain_and_consent_patches(
    model: torch.nn.Module,
    obs: ObserveActivations,
    null_attn: dict[int, torch.Tensor],
    null_mlp: dict[int, torch.Tensor],
) -> list[WeightPatch]:
    """Step 1B (i) within-family v_domain on Priority A & (ii) within-family v_consent on Priority B."""
    rc_idx = [i for i, m in enumerate(obs.meta_automode) if m["subset"] == "real_cf"]
    c5_idx = [i for i, m in enumerate(obs.meta_automode) if m["subset"] == "consent_5p"]
    y_rc = np.array([obs.meta_automode[i]["y"] for i in rc_idx])
    y_c5 = np.array([obs.meta_automode[i]["y"] for i in c5_idx])

    patches: list[WeightPatch] = []

    for l in [11, 16, 19]:
        x_attn_all = obs.plain_norm["attn_in"][l].numpy()
        w_dom, lam_dom, cv_auc = _select_within_family_pairs_and_ridge(
            x_attn_all, obs.meta_automode, rc_idx, lambdas=(30.0, 100.0, 300.0, 1000.0), p23_pair_boost=1.0
        )
        res_rc = obs.plain_norm["resid"][l][rc_idx]
        r_dir = (res_rc[y_rc == 1].mean(dim=0) - res_rc[y_rc == 0].mean(dim=0)).double()
        r_dir = r_dir / (torch.linalg.norm(r_dir).clamp_min(1e-8))
        scores = torch.tensor((x_attn_all[rc_idx] @ w_dom), dtype=torch.float64)
        scores = (scores - scores.mean()) / (scores.std().clamp_min(1e-6))
        target_out = scores.unsqueeze(1) * r_dir.unsqueeze(0)

        delta = solve_alphaedit_update(
            target_delta_out=target_out,
            keys_target=obs.plain_norm["attn_in"][l][rc_idx],
            p_null=null_attn[l],
            rank=1,
            lam=lam_dom,
            alpha=1.25,
        )
        pname = _param_name_attn(l, model)
        patches.append(
            WeightPatch(
                step_id=f"step1b_v_domain_attn_L{l}",
                category="step1b_domain",
                method="steer2edit_within_family_rank1",
                target_layers=[l],
                rank=1,
                alpha=1.25,
                hypothesis=f"Amplify within-family real_cf safety separation inside {pname} (CV AUROC={cv_auc:.3f}).",
                deltas={pname: delta},
                metadata={"observe_cv_auroc": round(cv_auc, 4), "ridge_lam": lam_dom},
            )
        )

    for l in [15, 17, 19, 20]:
        x_mlp_all = obs.plain_norm["mlp_hid"][l].numpy()
        w_c, lam_c, cv_auc = _select_within_family_pairs_and_ridge(
            x_mlp_all, obs.meta_automode, c5_idx, lambdas=(30.0, 100.0, 300.0, 1000.0), p23_pair_boost=2.5
        )
        res_c5 = obs.plain_norm["resid"][l][c5_idx]
        r_dir = (res_c5[y_c5 == 1].mean(dim=0) - res_c5[y_c5 == 0].mean(dim=0)).double()
        r_dir = r_dir / (torch.linalg.norm(r_dir).clamp_min(1e-8))
        scores = torch.tensor((x_mlp_all[c5_idx] @ w_c), dtype=torch.float64)
        scores = (scores - scores.mean()) / (scores.std().clamp_min(1e-6))
        target_out = scores.unsqueeze(1) * r_dir.unsqueeze(0)

        col_score = torch.tensor(np.abs(w_c), dtype=torch.float32) / (obs.mlp_preserve_energy[l] + 1e-4)
        top_cols = torch.topk(col_score, k=64).indices
        col_mask = torch.zeros(x_mlp_all.shape[1], dtype=torch.bool)
        col_mask[top_cols] = True

        delta = solve_alphaedit_update(
            target_delta_out=target_out,
            keys_target=obs.plain_norm["mlp_hid"][l][c5_idx],
            p_null=null_mlp[l],
            rank=1,
            lam=lam_c,
            alpha=1.25,
            col_mask=col_mask,
        )
        pname = f"model.layers.{l}.mlp.down_proj.weight"
        patches.append(
            WeightPatch(
                step_id=f"step1b_v_consent_mlp_L{l}",
                category="step1b_consent",
                method="steer2edit_within_family_sparse64",
                target_layers=[l],
                rank=1,
                alpha=1.25,
                hypothesis=f"Amplify within-family consent_5p separation in top-64 columns of {pname} (CV AUROC={cv_auc:.3f}).",
                deltas={pname: delta},
                metadata={"observe_cv_auroc": round(cv_auc, 4), "ridge_lam": lam_c},
            )
        )
    return patches


def _select_gain_calibration_rows(obs_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rc_a = [r for r in obs_rows if r["subset"] == "real_cf" and r["label"] == "allow"][:4]
    rc_b = [r for r in obs_rows if r["subset"] == "real_cf" and r["label"] == "block"][:4]
    c5_p2 = [r for r in obs_rows if r["subset"] == "consent_5p" and r["pattern"] == "p2_exact_consent"][:2]
    c5_p5a = [r for r in obs_rows if r["subset"] == "consent_5p" and r["pattern"] == "p5a_blocked_approved_same"][:2]
    c5_p3 = [r for r in obs_rows if r["subset"] == "consent_5p" and r["pattern"] == "p3_target_mismatch"][:2]
    c5_p5b = [r for r in obs_rows if r["subset"] == "consent_5p" and r["pattern"] == "p5b_blocked_swapped_target"][:2]
    return rc_a + rc_b + c5_p2 + c5_p5a + c5_p3 + c5_p5b


def measure_layer_logit_gains(
    runner: QwenDeciderRunner,
    obs_rows: list[dict[str, Any]],
    target_layers: list[int],
    u_unit_t: torch.Tensor,
    eps: float = 1.0,
) -> tuple[float, dict[int, float]]:
    """Measure mean base normal margin and per-layer MLP logit transfer gain d(B - A) / eps."""
    cal_rows = _select_gain_calibration_rows(obs_rows)
    prompts = [
        build_plain_prompt(
            render_view(r, tok=runner.tokenizer, budget=2048, include_recap=False).text,
            Q_BLOCK_BINARY,
            reverse_options=False,
        )
        for r in cal_rows
    ]
    base_margins: list[float] = []
    for pb in prompts:
        res = runner.score_prompt(pb.text, pb.option_letters, yes_idx=pb.yes_idx, no_idx=pb.no_idx)
        base_margins.append(float(res.logit_margin or 0.0))
    base_margin_norm = float(np.mean(base_margins))

    gains: dict[int, float] = {}
    u_dev = (eps * u_unit_t).detach()
    for l in target_layers:
        down_proj = runner.model.model.layers[l].mlp.down_proj

        def _inject(_mod, _inp, out):
            t = out.clone()
            t[:, -1, :] = t[:, -1, :] + u_dev.to(device=t.device, dtype=t.dtype)
            return t

        handle = down_proj.register_forward_hook(_inject)
        try:
            deltas: list[float] = []
            for idx, pb in enumerate(prompts):
                res = runner.score_prompt(pb.text, pb.option_letters, yes_idx=pb.yes_idx, no_idx=pb.no_idx)
                deltas.append(float(res.logit_margin or 0.0) - base_margins[idx])
            gains[l] = max(0.15, float(np.mean(deltas)) / float(eps))
        finally:
            handle.remove()

    print(
        f"[Track 1] Analytical gain calibration (16 obs rows): base_margin_norm={base_margin_norm:+.4f} | "
        + ", ".join(f"L{l}={gains[l]:.3f}" for l in target_layers)
    )
    return base_margin_norm, gains


def _measure_base_reversed_margin(
    runner: QwenDeciderRunner,
    obs_rows: list[dict[str, Any]],
) -> float:
    cal_rows = _select_gain_calibration_rows(obs_rows)
    margins: list[float] = []
    for r in cal_rows:
        ctx = render_view(r, tok=runner.tokenizer, budget=2048, include_recap=False).text
        pb = build_plain_prompt(ctx, Q_BLOCK_BINARY, reverse_options=True)
        res = runner.score_prompt(pb.text, pb.option_letters, yes_idx=pb.yes_idx, no_idx=pb.no_idx)
        margins.append(float(res.logit_margin or 0.0))
    return float(np.mean(margins))


def fit_joint_contrastive_readout(
    x_obs: np.ndarray,
    meta_obs: list[dict[str, Any]],
    lam: float | None = None,
    w_c5: float = 2.5,
    w_pair: float = 1.5,
    p23_pair_boost: float = 2.5,
    lambdas: tuple[float, ...] = (30.0, 100.0, 300.0, 1000.0),
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Closed-form joint + within-family contrastive ridge readout on observe activations."""
    xd = np.asarray(x_obs, dtype=np.float64)
    n, d = xd.shape

    y_dir = np.array([1.0 if int(m["y"]) == 1 else -1.0 for m in meta_obs], dtype=np.float64)
    y_bin = np.array([int(m["y"]) for m in meta_obs], dtype=np.int64)
    w_dir = np.empty(n, dtype=np.float64)
    for i, m in enumerate(meta_obs):
        if m["subset"] == "real_cf":
            w_dir[i] = 1.0
        elif m["subset"] == "consent_5p":
            w_dir[i] = float(w_c5) * (2.0 if int(m["y"]) == 0 else 1.0)
        else:
            w_dir[i] = 1.5

    # Group samples into (subset, family)
    fam_groups: dict[tuple[str, str], list[int]] = {}
    for i, m in enumerate(meta_obs):
        fid = str(m.get("family", m["id"].rsplit("_", 1)[0]))
        fam_groups.setdefault((str(m["subset"]), fid), []).append(i)

    def _solve_on_indices(sel_idx: np.ndarray, sel_fams: list[tuple[str, str]], lam_val: float) -> tuple[np.ndarray, float]:
        ws = w_dir[sel_idx]
        ws = ws / np.mean(ws)
        x_s = xd[sel_idx]
        y_s = y_dir[sel_idx]
        mu_x = np.average(x_s, axis=0, weights=ws)
        mu_y = float(np.average(y_s, weights=ws))
        xc = x_s - mu_x
        yc = y_s - mu_y
        gram = (xc * ws[:, None]).T @ xc + float(lam_val) * np.eye(d, dtype=np.float64)
        rhs = (xc * ws[:, None]).T @ yc

        for key in sel_fams:
            idxs = fam_groups[key]
            pos = [i for i in idxs if int(meta_obs[i]["y"]) == 1]
            neg = [i for i in idxs if int(meta_obs[i]["y"]) == 0]
            if not pos or not neg:
                continue
            sub = key[0]
            base_pw = float(w_pair) * (2.0 if sub == "consent_5p" else 1.0) / max(1, len(pos) * len(neg))
            for pi in pos:
                for ni in neg:
                    pair_pats = frozenset({str(meta_obs[pi].get("pattern", "")), str(meta_obs[ni].get("pattern", ""))})
                    pw = base_pw * (p23_pair_boost if pair_pats in CONTRASTIVE_PAIRS else 1.0)
                    diff = xd[pi] - xd[ni]
                    gram += pw * np.outer(diff, diff)
                    rhs += pw * 2.0 * diff

        w_out = np.linalg.solve(gram, rhs)
        b_out = float(mu_y - mu_x @ w_out)
        return w_out, b_out

    all_fams = sorted(fam_groups.keys())
    rc_idx = [i for i, m in enumerate(meta_obs) if m["subset"] == "real_cf"]
    c5_idx = [i for i, m in enumerate(meta_obs) if m["subset"] == "consent_5p"]

    rc_cv_auc = 0.5
    c5_cv_auc = 0.5
    if lam is None:
        # Assign families round-robin within each subset so every fold balances real_cf & consent_5p
        n_folds = 4
        fam_to_fold: dict[tuple[str, str], int] = {}
        for sub_name in sorted({k[0] for k in all_fams}):
            sub_fams = [k for k in all_fams if k[0] == sub_name]
            for j, key in enumerate(sub_fams):
                fam_to_fold[key] = j % n_folds

        best_lam = lambdas[len(lambdas) // 2]
        best_obj = -1.0
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
                w_f, b_f = _solve_on_indices(tr_idx, tr_fams, cand_lam)
                oof[val_idx] = xd[val_idx] @ w_f + b_f
                covered[val_idx] = True

            a_rc = float(roc_auc_score(y_bin[rc_idx], oof[rc_idx])) if len(rc_idx) > 1 and len(np.unique(y_bin[rc_idx])) > 1 else None
            a_c5 = float(roc_auc_score(y_bin[c5_idx], oof[c5_idx])) if len(c5_idx) > 1 and len(np.unique(y_bin[c5_idx])) > 1 else None
            if a_rc is not None and a_c5 is not None:
                obj = 0.5 * a_rc + 0.5 * a_c5
            elif a_rc is not None:
                obj = a_rc
            elif a_c5 is not None:
                obj = a_c5
            else:
                obj = float(roc_auc_score(y_bin[covered], oof[covered])) if len(np.unique(y_bin[covered])) > 1 else 0.5

            if obj > best_obj:
                best_obj = obj
                best_lam = cand_lam
                rc_cv_auc = a_rc if a_rc is not None else obj
                c5_cv_auc = a_c5 if a_c5 is not None else obj
        chosen_lam = float(best_lam)
    else:
        chosen_lam = float(lam)

    all_idx = np.arange(n, dtype=np.int64)
    w_final, b_final = _solve_on_indices(all_idx, all_fams, chosen_lam)
    scores = xd @ w_final + b_final
    sd = float(np.std(scores)) + 1e-8
    w_std = w_final / sd
    b_std = b_final / sd

    return w_std, b_std, {
        "lam": chosen_lam,
        "rc_cv_auc": round(float(rc_cv_auc), 4),
        "c5_cv_auc": round(float(c5_cv_auc), 4),
        "w_c5": w_c5,
        "w_pair": w_pair,
    }


def _snapshot_gate_layer_weights(
    model: torch.nn.Module,
    layers: list[int],
) -> dict[int, dict[str, torch.Tensor]]:
    out: dict[int, dict[str, torch.Tensor]] = {}
    for l in layers:
        p_gate = f"model.layers.{l}.mlp.gate_proj.weight"
        p_up = f"model.layers.{l}.mlp.up_proj.weight"
        p_down = f"model.layers.{l}.mlp.down_proj.weight"
        mlp = model.model.layers[l].mlp
        out[l] = {
            p_gate: mlp.gate_proj.weight.detach().float().cpu().clone(),
            p_up: mlp.up_proj.weight.detach().float().cpu().clone(),
            p_down: mlp.down_proj.weight.detach().float().cpu().clone(),
        }
    return out


def build_swiglu_order_gate_patch(
    runner: QwenDeciderRunner,
    obs: ObserveActivations,
    prior_layer_weights: dict[str, torch.Tensor],
    layer_idx: int = 20,
    layer_gain: float = 2.0,
    target_logit_std: float = 1.5,
    target_bias_shift: float = 0.7,
    target_bias_shift_rev: float | None = None,
    w_c5: float = 2.5,
    w_pair: float = 1.5,
    p23_pair_boost: float = 2.5,
) -> WeightPatch:
    """Step 1B (iii): Analytical gain-calibrated 4-neuron SwiGLU order-equivariant risk/consent gate.

    Always computes deltas against `prior_layer_weights` (never live mutated model state).
    Uses the 4 quietest preservation neurons (j1, j2, j3, j4):
      - j1 (normal order):  SiLU(g_norm @ x) * (w_eff_norm @ x) -> +alpha_score * u_unit
      - j2 (normal order):  SiLU(g_norm @ x) * (b_unit_norm @ x) -> +alpha_bias * u_unit
      - j3 (reversed order):SiLU(g_rev @ x)  * (w_eff_rev @ x)  -> -alpha_score * u_unit
      - j4 (reversed order):SiLU(g_rev @ x)  * (b_unit_rev @ x)  -> +alpha_bias_rev * u_unit
    (Note: in u_unit = W_U[B] - W_U[A] space, base model has negative logit(B)-logit(A) on BOTH orders,
     so neutralizing (A)-letter bias requires positive +u_unit shift on both j2 and j4!)
    """
    model = runner.model
    energy = obs.mlp_preserve_energy[layer_idx]
    quiet_neurons = torch.topk(-energy, k=4).indices.tolist()
    j1, j2, j3, j4 = quiet_neurons

    x_norm = obs.plain_norm["mlp_in"][layer_idx].numpy()
    x_rev = obs.plain_rev["mlp_in"][layer_idx].numpy()
    x_pres = obs.preserve["mlp_in"][layer_idx].numpy()

    x_all = np.concatenate([x_norm, x_rev, x_pres], axis=0)
    n_a, n_r, n_p = len(x_norm), len(x_rev), len(x_pres)

    y_gn = np.concatenate([np.full(n_a, 4.5), np.full(n_r, -7.5), np.full(n_p, -7.5)])
    sw_gn = np.concatenate([np.full(n_a, 4.0), np.full(n_r, 2.0), np.full(n_p, 2.0)])
    g_norm, _ = solve_closed_form_ridge(x_all, y_gn, lam=120.0, center=False, sample_weights=sw_gn)

    y_gr = np.concatenate([np.full(n_a, -7.5), np.full(n_r, 4.5), np.full(n_p, -7.5)])
    sw_gr = np.concatenate([np.full(n_a, 2.0), np.full(n_r, 4.0), np.full(n_p, 2.0)])
    g_rev, _ = solve_closed_form_ridge(x_all, y_gr, lam=120.0, center=False, sample_weights=sw_gr)

    def _silu(z: np.ndarray) -> np.ndarray:
        return z / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))

    s_gn_pos = float(np.mean(_silu(x_norm @ g_norm)))
    s_gn_other = float(np.max(_silu(np.concatenate([x_rev, x_pres], axis=0) @ g_norm)))
    s_gr_pos = float(np.mean(_silu(x_rev @ g_rev)))
    s_gr_other = float(np.max(_silu(np.concatenate([x_norm, x_pres], axis=0) @ g_rev)))

    w_n, b_n, stats_n = fit_joint_contrastive_readout(
        x_norm, obs.meta_automode, w_c5=w_c5, w_pair=w_pair, p23_pair_boost=p23_pair_boost
    )
    w_r, b_r, stats_r = fit_joint_contrastive_readout(
        x_rev, obs.meta_automode, w_c5=w_c5, w_pair=w_pair, p23_pair_boost=p23_pair_boost
    )

    mu_n = np.mean(x_norm, axis=0)
    b_unit_n = mu_n / (float(np.dot(mu_n, mu_n)) + 1e-12)
    w_eff_n = w_n + b_n * b_unit_n

    mu_r = np.mean(x_rev, axis=0)
    b_unit_r = mu_r / (float(np.dot(mu_r, mu_r)) + 1e-12)
    w_eff_r = w_r + b_r * b_unit_r

    w_lm = model.lm_head.weight.detach().float().cpu()
    id_a = runner.letter_token_ids["A"]
    id_b = runner.letter_token_ids["B"]
    u_dir = (w_lm[id_b] - w_lm[id_a]).double()
    u_unit = (u_dir / torch.linalg.norm(u_dir).clamp_min(1e-8)).numpy()

    g_scale_norm = max(0.5, s_gn_pos)
    g_scale_rev = max(0.5, s_gr_pos)
    eff_gain = max(0.15, float(layer_gain))

    alpha_score = float(target_logit_std) / eff_gain
    alpha_bias = float(target_bias_shift) / eff_gain
    eff_bias_rev = float(target_bias_shift_rev if target_bias_shift_rev is not None else target_bias_shift)
    alpha_bias_rev = eff_bias_rev / eff_gain

    p_gate = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
    p_up = f"model.layers.{layer_idx}.mlp.up_proj.weight"
    p_down = f"model.layers.{layer_idx}.mlp.down_proj.weight"

    gate_w = prior_layer_weights[p_gate].float().cpu()
    up_w = prior_layer_weights[p_up].float().cpu()
    down_w = prior_layer_weights[p_down].float().cpu()

    d_gate = torch.zeros_like(gate_w)
    d_up = torch.zeros_like(up_w)
    d_down = torch.zeros_like(down_w)

    d_gate[j1] = torch.from_numpy(g_norm).float() - gate_w[j1]
    d_gate[j2] = torch.from_numpy(g_norm).float() - gate_w[j2]
    d_gate[j3] = torch.from_numpy(g_rev).float() - gate_w[j3]
    d_gate[j4] = torch.from_numpy(g_rev).float() - gate_w[j4]

    d_up[j1] = torch.from_numpy(w_eff_n).float() - up_w[j1]
    d_up[j2] = torch.from_numpy(b_unit_n).float() - up_w[j2]
    d_up[j3] = torch.from_numpy(w_eff_r).float() - up_w[j3]
    d_up[j4] = torch.from_numpy(b_unit_r).float() - up_w[j4]

    d_down[:, j1] = torch.from_numpy((alpha_score / g_scale_norm) * u_unit).float() - down_w[:, j1]
    d_down[:, j2] = torch.from_numpy((alpha_bias / g_scale_norm) * u_unit).float() - down_w[:, j2]
    d_down[:, j3] = torch.from_numpy((-alpha_score / g_scale_rev) * u_unit).float() - down_w[:, j3]
    d_down[:, j4] = torch.from_numpy((alpha_bias_rev / g_scale_rev) * u_unit).float() - down_w[:, j4]

    return WeightPatch(
        step_id=f"step1b_swiglu_order_gate_L{layer_idx}",
        category="step1b_swiglu_gate",
        method="closed_form_swiglu_4neuron",
        target_layers=[layer_idx],
        rank=4,
        alpha=alpha_score,
        hypothesis=(
            f"Write gain-calibrated order-conditioned SwiGLU joint contrastive readout into 4 quietest neurons "
            f"{quiet_neurons} of L{layer_idx} MLP (gain={eff_gain:.3f}, target_std={target_logit_std}, target_bias={target_bias_shift:.2f}/{eff_bias_rev:.2f})."
        ),
        deltas={p_gate: d_gate, p_up: d_up, p_down: d_down},
        scale_keys=[p_down],
        metadata={
            "quiet_neurons": quiet_neurons,
            "layer_gain": round(eff_gain, 4),
            "target_logit_std": target_logit_std,
            "target_bias_shift": round(target_bias_shift, 4),
            "target_bias_shift_rev": round(eff_bias_rev, 4),
            "alpha_score": round(alpha_score, 4),
            "alpha_bias": round(alpha_bias, 4),
            "alpha_bias_rev": round(alpha_bias_rev, 4),
            "w_c5": w_c5,
            "w_pair": w_pair,
            "gate_norm_mean_silu_target": round(s_gn_pos, 4),
            "gate_norm_max_silu_nontarget": round(s_gn_other, 4),
            "gate_rev_mean_silu_target": round(s_gr_pos, 4),
            "gate_rev_max_silu_nontarget": round(s_gr_other, 4),
            "observe_cv_norm": stats_n,
            "observe_cv_rev": stats_r,
        },
    )


def _refresh_mlp_in_for_layers(
    runner: QwenDeciderRunner,
    obs: ObserveActivations,
    target_layers: list[int],
    body_tokens_per_prompt: int = 10,
) -> None:
    """Refresh obs.plain_norm/plain_rev/preserve mlp_in buffers after prior layer edits."""
    obs_rows = load_split_rows("observe")
    auto_rows = [r for r in obs_rows if r["subset"] in ("real_cf", "consent_5p", "long_50t")]
    replay_rows = [r for r in obs_rows if r["subset"] == "decider_replay"]

    paren_ids = runner.tokenizer.encode("(", add_special_tokens=False)
    paren_tok_id = int(paren_ids[0]) if len(paren_ids) == 1 else None

    b_norm = {l: [] for l in target_layers}
    b_rev = {l: [] for l in target_layers}
    b_pres = {l: [] for l in target_layers}
    cur_m_in: dict[int, torch.Tensor] = {}
    cap_pos: list[int] = [-1]
    handles = []

    for l in target_layers:
        layer = runner.model.model.layers[l]

        def _mk_pre(idx: int):
            def _hook(_mod, inp):
                cur_m_in[idx] = inp[0][0, cap_pos, :].detach().float().cpu()
            return _hook

        handles.append(layer.mlp.register_forward_pre_hook(_mk_pre(l)))

    def _run(txt: str, dst: dict[int, list[torch.Tensor]], sample_body: bool = False, is_replay: bool = False) -> None:
        nonlocal cap_pos
        ids = runner.encode_prompt(txt)
        seq_len = int(ids.shape[1])
        body_pos: list[int] = []
        if is_replay:
            cap_pos = sorted({max(0, seq_len - 3), max(0, seq_len - 2), seq_len - 1})
        else:
            if sample_body:
                body_pos = _sample_body_positions(ids, paren_tok_id, body_tokens_per_prompt=body_tokens_per_prompt)
            cap_pos = body_pos + [seq_len - 1]
        with torch.inference_mode():
            _ = runner.model.model(input_ids=ids, use_cache=False)
        for l in target_layers:
            if is_replay:
                for p_i in range(len(cap_pos)):
                    b_pres[l].append(cur_m_in[l][p_i])
            else:
                dst[l].append(cur_m_in[l][-1])
                for p_i in range(len(body_pos)):
                    b_pres[l].append(cur_m_in[l][p_i])

    try:
        for r in auto_rows:
            budget = 3200 if r["subset"] == "long_50t" else 2048
            ctx = render_view(r, tok=runner.tokenizer, budget=budget, include_recap=False).text
            _run(build_plain_prompt(ctx, Q_BLOCK_BINARY, reverse_options=False).text, b_norm, sample_body=True)
            _run(build_plain_prompt(ctx, Q_BLOCK_BINARY, reverse_options=True).text, b_rev, sample_body=False)
        for r in replay_rows:
            pb = build_decider_prompt(r["context"], spec=r["question_spec"], reverse_options=False)
            _run(pb.text, b_pres, sample_body=False, is_replay=True)
    finally:
        for h in handles:
            h.remove()

    for l in target_layers:
        obs.plain_norm["mlp_in"][l] = torch.stack(b_norm[l], dim=0)
        obs.plain_rev["mlp_in"][l] = torch.stack(b_rev[l], dim=0)
        obs.preserve["mlp_in"][l] = torch.stack(b_pres[l], dim=0)


def _score_metric_fast(rep: dict[str, Any], mkey: str) -> float:
    if mkey == "proxy_score":
        return float(rep["proxy"]["proxy_score"])
    if mkey == "domain_focus":
        return float(rep["real_cf"]["margin_separation"] - 2.0 * rep["real_cf"]["bce"])
    c5 = rep["consent_5p"]
    return float(c5["margin_sep_p3_minus_p2"] + c5["margin_sep_p5b_minus_p5a"] - 0.5 * c5["binary_metrics"]["bce"])


def _screen_and_causal_accept(
    runner: QwenDeciderRunner,
    base_cpu_state: dict[str, torch.Tensor],
    candidates: list[WeightPatch],
    accepted_patches: list[WeightPatch],
    eval_panel_fn: Any,
    metric_key_fn: Any,
    replay_floor: float,
    n_random_seeds: int,
    max_causal_checks: int,
    cached_base: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Fast 1-pass screening -> strict 3-point causal checks on top positive candidates."""
    screening_logs: list[dict[str, Any]] = []
    causal_records: list[dict[str, Any]] = []
    cur_base_rep = cached_base

    scored_cands: list[tuple[float, WeightPatch, str]] = []
    for cand in candidates:
        mkey = metric_key_fn(cand)
        base_sc = _score_metric_fast(cur_base_rep, mkey)
        apply_bundle(runner.model, base_cpu_state, accepted_patches + [cand])
        rep = eval_panel_fn()
        cand_sc = _score_metric_fast(rep, mkey)
        gain = cand_sc - base_sc
        screening_logs.append({"step_id": cand.step_id, "metric_key": mkey, "screen_gain": round(gain, 5)})
        print(f"  [screen] {cand.step_id} ({mkey}): gain={gain:+.5f}")
        scored_cands.append((gain, cand, mkey))

    scored_cands.sort(key=lambda t: t[0], reverse=True)
    top_cands = [t for t in scored_cands if t[0] > 1e-4][:max_causal_checks]
    if not top_cands and scored_cands:
        top_cands = [scored_cands[0]]

    for _, cand, mkey in top_cands:
        chk = run_three_point_causal_check(
            model=runner.model,
            base_cpu_state=base_cpu_state,
            prior_patches=accepted_patches,
            candidate_patch=cand,
            eval_fn=eval_panel_fn,
            metric_key=mkey,
            n_random_seeds=n_random_seeds,
            base_report_cached=cur_base_rep,
        )
        post_rep = chk.best_report if chk.best_report is not None else eval_panel_fn()
        rep_ok = post_rep["decider_replay"]["overall_acc"] >= replay_floor
        keep = chk.passed and rep_ok
        rec = {
            "step_id": cand.step_id,
            "category": cand.category,
            "metric_key": mkey,
            "passed_3pt_causal": chk.passed,
            "replay_preserved": rep_ok,
            "kept": keep,
            "best_scale": chk.best_scale,
            "base_score": chk.base_score,
            "best_score": chk.best_score,
            "dose_response": chk.dose_response,
            "sign_reversal": chk.sign_reversal,
            "orthogonal_control": chk.orthogonal_control,
            "fail_reasons": chk.fail_reasons,
        }
        causal_records.append(rec)
        print(
            f"  [causal] {cand.step_id} [{mkey}]: causal={chk.passed}, replay_ok={rep_ok}, "
            f"kept={keep}, best_scale={chk.best_scale}, gain={chk.best_score - chk.base_score:+.5f}"
        )
        if keep:
            accepted_patches.append(cand.scaled(chk.best_scale))
            cur_base_rep = post_rep

    return screening_logs, causal_records, cur_base_rep


def _record_causal(chk: Any, cand: WeightPatch, rep_ok: bool, kept: bool, mkey: str = "proxy_score") -> dict[str, Any]:
    return {
        "step_id": cand.step_id,
        "category": cand.category,
        "metric_key": mkey,
        "passed_3pt_causal": chk.passed,
        "replay_preserved": rep_ok,
        "kept": kept,
        "best_scale": chk.best_scale,
        "base_score": chk.base_score,
        "best_score": chk.best_score,
        "dose_response": chk.dose_response,
        "sign_reversal": chk.sign_reversal,
        "orthogonal_control": chk.orthogonal_control,
        "fail_reasons": chk.fail_reasons,
    }


def run_track1_pipeline(
    run_name: str = "track1_best_dev",
    teacher_mode: str = "oracle_chat",
    n_random_seeds: int = 3,
    refresh_cache: bool = False,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    runner = QwenDeciderRunner()
    base_cpu_state = capture_cpu_state_dict(runner.model)
    obs_rows = load_split_rows("observe")
    dev_all = load_split_rows("select_dev")
    dev_panel = make_causal_panel_rows(dev_all, fast=True)

    # 1. Collect or load observe activations
    cache_path = CHECKPOINTS_DIR / f"observe_acts_v2_{teacher_mode}.pt"
    print(f"[Track 1] Preparing observe activations (teacher={teacher_mode})...")
    obs = collect_observe_activations(
        runner,
        layers=TRACK1_LAYERS,
        teacher_mode=teacher_mode,
        body_tokens_per_prompt=10,
        cache_path=cache_path,
        refresh_cache=refresh_cache,
    )

    # 2. Build replay-hardened AlphaEdit Null-Space projectors
    null_attn: dict[int, torch.Tensor] = {}
    null_mlp: dict[int, torch.Tensor] = {}
    null_stats: dict[str, Any] = {}
    for l in sorted(set(PRIORITY_A_LAYERS + PRIORITY_B_LAYERS + [11, 15, 16, 17, 18, 19, 20])):
        p_a, st_a = build_null_space_projector(
            obs.preserve["attn_in"][l],
            variance_keep=0.95,
            max_rank_frac=0.50,
            k_replay_extra=obs.replay_only["attn_in"][l],
            replay_weight=3.0,
        )
        p_m, st_m = build_null_space_projector(
            obs.preserve["mlp_hid"][l],
            variance_keep=0.95,
            max_rank_frac=0.50,
            k_replay_extra=obs.replay_only["mlp_hid"][l],
            replay_weight=3.0,
        )
        null_attn[l] = p_a
        null_mlp[l] = p_m
        null_stats[f"L{l}_attn"] = st_a
        null_stats[f"L{l}_mlp"] = st_m

    def _eval_panel() -> dict[str, Any]:
        return quick_eval_dev(runner, dev_panel, include_long=False)

    def _eval_full_dev() -> dict[str, Any]:
        return quick_eval_dev(runner, dev_all, include_long=True)

    base_panel_rep = _eval_panel()
    replay_floor = float(base_panel_rep["decider_replay"]["overall_acc"]) - 0.025

    screening_logs: list[dict[str, Any]] = []
    causal_records: list[dict[str, Any]] = []
    accepted_patches: list[WeightPatch] = []

    # 3. Step 1A: Context-to-Weight Distillation (screen all, 3-pt causal check on top 3)
    print("[Track 1] Screening & causal-checking Step 1A Context-to-Weight candidates...")
    c2w_candidates = build_step1a_c2w_candidates(runner.model, obs, null_attn, null_mlp)
    scr1a, cc1a, cached_base = _screen_and_causal_accept(
        runner=runner,
        base_cpu_state=base_cpu_state,
        candidates=c2w_candidates,
        accepted_patches=accepted_patches,
        eval_panel_fn=_eval_panel,
        metric_key_fn=lambda _c: "proxy_score",
        replay_floor=replay_floor,
        n_random_seeds=n_random_seeds,
        max_causal_checks=3,
        cached_base=base_panel_rep,
    )
    screening_logs.extend(scr1a)
    causal_records.extend(cc1a)

    # 4. Step 1B (i & ii): Within-family v_domain & v_consent (screen all, 3-pt causal check on top 2)
    print("[Track 1] Screening & causal-checking Step 1B v_domain and v_consent candidates...")
    dom_cons_candidates = build_step1b_domain_and_consent_patches(runner.model, obs, null_attn, null_mlp)
    scr1b, cc1b, cached_base = _screen_and_causal_accept(
        runner=runner,
        base_cpu_state=base_cpu_state,
        candidates=dom_cons_candidates,
        accepted_patches=accepted_patches,
        eval_panel_fn=_eval_panel,
        metric_key_fn=lambda c: "domain_focus" if c.category == "step1b_domain" else "consent_focus",
        replay_floor=replay_floor,
        n_random_seeds=n_random_seeds,
        max_causal_checks=2,
        cached_base=cached_base,
    )
    screening_logs.extend(scr1b)
    causal_records.extend(cc1b)

    # 5. Clean Layer Freezing Ablation (L in [15, 19, 20, 21, 22, 23] -> l_f in [8, 4, 3, 2, 1, 0])
    gate_layers = [15, 19, 20, 21, 22, 23]
    apply_bundle(runner.model, base_cpu_state, accepted_patches)
    if accepted_patches:
        print("[Track 1] Refreshing observe mlp_in activations after accepted Step 1A/1B patches...")
        _refresh_mlp_in_for_layers(runner, obs, gate_layers)
    prior_gate_weights = _snapshot_gate_layer_weights(runner.model, gate_layers)

    w_lm = runner.model.lm_head.weight.detach().float().cpu()
    u_dir = (w_lm[runner.letter_token_ids["B"]] - w_lm[runner.letter_token_ids["A"]]).double()
    u_unit_t = (u_dir / torch.linalg.norm(u_dir).clamp_min(1e-8)).float()

    base_margin_norm, gains_by_layer = measure_layer_logit_gains(runner, obs_rows, gate_layers, u_unit_t, eps=1.0)
    base_margin_rev = _measure_base_reversed_margin(runner, obs_rows)
    rev_bias_ratio = float(np.clip(base_margin_rev / max(0.20, -base_margin_norm), 0.85, 1.75)) if base_margin_rev > 0 else 1.0
    print(f"[Track 1] Base reversed margin={base_margin_rev:+.4f}, rev_bias_ratio={rev_bias_ratio:.3f}")

    print("[Track 1] Stage 1: Layer Freezing ablation across L in [15, 19, 20, 21, 22, 23]...")
    freezing_ablation: list[dict[str, Any]] = []
    init_bias = max(0.55, -base_margin_norm)
    stage1_entries: list[dict[str, Any]] = []

    for l_idx in gate_layers:
        l_frozen = 23 - l_idx
        gate_cand = build_swiglu_order_gate_patch(
            runner=runner,
            obs=obs,
            prior_layer_weights=prior_gate_weights[l_idx],
            layer_idx=l_idx,
            layer_gain=gains_by_layer[l_idx],
            target_logit_std=1.5,
            target_bias_shift=init_bias,
            target_bias_shift_rev=init_bias * rev_bias_ratio,
            w_c5=2.5,
            w_pair=1.5,
        )
        apply_bundle(runner.model, base_cpu_state, accepted_patches + [gate_cand])
        rep = _eval_panel()
        p_sc = float(rep["proxy"]["proxy_score"])
        rep_acc = float(rep["decider_replay"]["overall_acc"])
        entry = {
            "stage": "layer_freeze_sweep",
            "layer_idx": l_idx,
            "frozen_final_layers": l_frozen,
            "layer_gain": round(gains_by_layer[l_idx], 4),
            "target_logit_std": 1.5,
            "target_bias_shift": round(init_bias, 4),
            "target_bias_shift_rev": round(init_bias * rev_bias_ratio, 4),
            "w_c5": 2.5,
            "proxy_score": p_sc,
            "real_cf_auroc": rep["real_cf"]["auroc"],
            "real_cf_recall": rep["real_cf"]["recall_at_05"],
            "real_cf_fpr": rep["real_cf"]["fpr_at_05"],
            "consent_acc": rep["consent_5p"]["overall_acc"],
            "p2_allow": rep["consent_5p"]["p2_allow_rate"],
            "p3_block": rep["consent_5p"]["p3_block_rate"],
            "order_flip_rc": rep["real_cf"]["order_flip_rate"],
            "order_flip_c5": rep["consent_5p"]["order_flip_rate"],
            "replay_acc": rep_acc,
            "replay_ece": rep["decider_replay"]["top_label_ece"],
            "gate_meta": gate_cand.metadata,
        }
        freezing_ablation.append(entry)
        stage1_entries.append(entry)
        print(
            f"  [freeze l_f={l_frozen} L{l_idx}] gain={gains_by_layer[l_idx]:.2f} proxy={p_sc:+.4f} "
            f"rc(auc={entry['real_cf_auroc']:.3f} rec={entry['real_cf_recall']:.2f} fpr={entry['real_cf_fpr']:.2f} flip={entry['order_flip_rc']:.2f}) "
            f"c5(acc={entry['consent_acc']:.3f} P2={entry['p2_allow']:.2f} P3={entry['p3_block']:.2f} flip={entry['order_flip_c5']:.2f}) rep={rep_acc:.3f}"
        )

    eligible = [e for e in stage1_entries if e["replay_acc"] >= replay_floor]
    ranked_pool = sorted(eligible if eligible else stage1_entries, key=lambda e: e["proxy_score"], reverse=True)
    best_layers = [e["layer_idx"] for e in ranked_pool[:2]]

    print(f"[Track 1] Stage 2: Calibrated sweep on top 2 layers {best_layers}...")
    calib_configs = [
        (1.2, 0.55, 2.5),
        (1.5, 0.70, 2.5),
        (1.8, 0.85, 2.5),
        (1.5, 0.70, 3.5),
        (1.5, 0.90, 3.5),
    ]
    best_gate_patch: WeightPatch | None = None
    best_gate_proxy = -1e9
    fallback_gate_patch: WeightPatch | None = None
    fallback_gate_proxy = -1e9

    for l_idx in best_layers:
        for (t_std, t_bias, wc5) in calib_configs:
            gate_cand = build_swiglu_order_gate_patch(
                runner=runner,
                obs=obs,
                prior_layer_weights=prior_gate_weights[l_idx],
                layer_idx=l_idx,
                layer_gain=gains_by_layer[l_idx],
                target_logit_std=t_std,
                target_bias_shift=t_bias,
                target_bias_shift_rev=t_bias * rev_bias_ratio,
                w_c5=wc5,
                w_pair=1.5,
            )
            apply_bundle(runner.model, base_cpu_state, accepted_patches + [gate_cand])
            rep = _eval_panel()
            p_sc = float(rep["proxy"]["proxy_score"])
            rep_acc = float(rep["decider_replay"]["overall_acc"])
            entry = {
                "stage": "calib_sweep",
                "layer_idx": l_idx,
                "frozen_final_layers": 23 - l_idx,
                "layer_gain": round(gains_by_layer[l_idx], 4),
                "target_logit_std": t_std,
                "target_bias_shift": t_bias,
                "target_bias_shift_rev": round(t_bias * rev_bias_ratio, 4),
                "w_c5": wc5,
                "proxy_score": p_sc,
                "real_cf_auroc": rep["real_cf"]["auroc"],
                "real_cf_recall": rep["real_cf"]["recall_at_05"],
                "real_cf_fpr": rep["real_cf"]["fpr_at_05"],
                "consent_acc": rep["consent_5p"]["overall_acc"],
                "p2_allow": rep["consent_5p"]["p2_allow_rate"],
                "p3_block": rep["consent_5p"]["p3_block_rate"],
                "order_flip_rc": rep["real_cf"]["order_flip_rate"],
                "order_flip_c5": rep["consent_5p"]["order_flip_rate"],
                "replay_acc": rep_acc,
                "replay_ece": rep["decider_replay"]["top_label_ece"],
                "gate_meta": gate_cand.metadata,
            }
            freezing_ablation.append(entry)
            print(
                f"  [calib L{l_idx} std={t_std} bias={t_bias} w_c5={wc5}] proxy={p_sc:+.4f} "
                f"rc(auc={entry['real_cf_auroc']:.3f} rec={entry['real_cf_recall']:.2f} fpr={entry['real_cf_fpr']:.2f}) "
                f"c5(acc={entry['consent_acc']:.3f} P2={entry['p2_allow']:.2f} P3={entry['p3_block']:.2f}) rep={rep_acc:.3f}"
            )
            if p_sc > fallback_gate_proxy:
                fallback_gate_proxy = p_sc
                fallback_gate_patch = gate_cand
            if rep_acc >= replay_floor and p_sc > best_gate_proxy:
                best_gate_proxy = p_sc
                best_gate_patch = gate_cand

    if best_gate_patch is None:
        best_gate_patch = fallback_gate_patch
    assert best_gate_patch is not None

    # Restore clean prior bundle before running 3-point causal check on best_gate_patch
    apply_bundle(runner.model, base_cpu_state, accepted_patches)
    print(f"[Track 1] Running 3-point causal check on {best_gate_patch.step_id}...")
    gate_chk = run_three_point_causal_check(
        model=runner.model,
        base_cpu_state=base_cpu_state,
        prior_patches=accepted_patches,
        candidate_patch=best_gate_patch,
        eval_fn=_eval_panel,
        metric_key="proxy_score",
        n_random_seeds=n_random_seeds,
        base_report_cached=cached_base,
    )
    post_gate_rep = gate_chk.best_report if gate_chk.best_report is not None else _eval_panel()
    gate_rep_ok = post_gate_rep["decider_replay"]["overall_acc"] >= replay_floor
    gate_kept = gate_chk.passed and gate_rep_ok
    causal_records.append(_record_causal(gate_chk, best_gate_patch, gate_rep_ok, gate_kept))
    print(
        f"  {best_gate_patch.step_id}: causal={gate_chk.passed}, replay_ok={gate_rep_ok}, "
        f"kept={gate_kept}, best_scale={gate_chk.best_scale}, gain={gate_chk.best_score - gate_chk.base_score:+.5f}"
    )
    if gate_kept:
        accepted_patches.append(best_gate_patch.scaled(gate_chk.best_scale))
        cached_base = post_gate_rep

    # Complementary 2nd-layer gate check
    if gate_kept and len(best_layers) > 1 and best_layers[1] != best_gate_patch.target_layers[0]:
        best_l = best_gate_patch.target_layers[0]
        second_l = best_layers[1]
        apply_bundle(runner.model, base_cpu_state, accepted_patches)
        if second_l > best_l:
            print(f"[Track 1] Refreshing mlp_in on complementary L{second_l} (> L{best_l})...")
            _refresh_mlp_in_for_layers(runner, obs, [second_l])
        second_w_c5 = 3.5 if best_l <= 19 else 1.5
        comp_gate = build_swiglu_order_gate_patch(
            runner=runner,
            obs=obs,
            prior_layer_weights=prior_gate_weights[second_l],
            layer_idx=second_l,
            layer_gain=gains_by_layer[second_l],
            target_logit_std=0.8,
            target_bias_shift=0.0,
            target_bias_shift_rev=0.0,
            w_c5=second_w_c5,
            w_pair=1.5,
        )
        print(f"[Track 1] Running 3-point causal check on complementary gate L{second_l}...")
        comp_chk = run_three_point_causal_check(
            model=runner.model,
            base_cpu_state=base_cpu_state,
            prior_patches=accepted_patches,
            candidate_patch=comp_gate,
            eval_fn=_eval_panel,
            metric_key="proxy_score",
            n_random_seeds=n_random_seeds,
            base_report_cached=cached_base,
        )
        post_comp_rep = comp_chk.best_report if comp_chk.best_report is not None else _eval_panel()
        comp_rep_ok = post_comp_rep["decider_replay"]["overall_acc"] >= replay_floor
        strictly_better = comp_chk.best_score > float(cached_base["proxy"]["proxy_score"]) + 1e-3
        comp_kept = comp_chk.passed and comp_rep_ok and strictly_better
        rec_comp = _record_causal(comp_chk, comp_gate, comp_rep_ok, comp_kept)
        rec_comp["step_id"] = f"{comp_gate.step_id}_complement"
        causal_records.append(rec_comp)
        print(
            f"  comp L{second_l}: causal={comp_chk.passed}, replay_ok={comp_rep_ok}, "
            f"strictly_better={strictly_better}, kept={comp_kept}, gain={comp_chk.best_score - comp_chk.base_score:+.5f}"
        )
        if comp_kept:
            comp_final = comp_gate.scaled(comp_chk.best_scale)
            comp_final.step_id = f"{comp_gate.step_id}_complement"
            accepted_patches.append(comp_final)
            cached_base = post_comp_rep

    # 6. Leave-One-Out Ablation on FULL select_dev
    print("[Track 1] Running Leave-One-Out ablation on full select_dev...")
    full_replay_floor = float(
        json.loads((EXPERIMENTS_DIR / "track0a_zeroshot_dev.json").read_text())["decider_replay"]["overall_acc"]
    ) - 0.025
    pruned_patches, loo_summary = leave_one_out_ablation(
        model=runner.model,
        base_cpu_state=base_cpu_state,
        patches=accepted_patches,
        eval_fn=_eval_full_dev,
        replay_acc_floor=full_replay_floor,
        tol=0.005,
    )

    # 7. Final full select_dev evaluation
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

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    (EXPERIMENTS_DIR / "track1_causal_checks.json").write_text(
        json.dumps(
            {
                "null_space_stats": null_stats,
                "calibration": {
                    "base_margin_norm": round(base_margin_norm, 4),
                    "base_margin_rev": round(base_margin_rev, 4),
                    "rev_bias_ratio": round(rev_bias_ratio, 4),
                    "gains_by_layer": {str(k): round(v, 4) for k, v in gains_by_layer.items()},
                },
                "screening": screening_logs,
                "checks": causal_records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    cc_md = ["# Track 1 3-Point Causal Sanity Checks", ""]
    for c in causal_records:
        cc_md.append(
            f"- **`{c['step_id']}`** (`{c['category']}`): **Kept=`{c['kept']}`** | Causal=`{c['passed_3pt_causal']}` | "
            f"ReplayOK=`{c['replay_preserved']}` | `best_scale={c['best_scale']}` | "
            f"`base={c['base_score']}` -> `best={c['best_score']}` | "
            f"Spearman=`{c['dose_response']['spearman_0_to_1']}` | "
            f"Rev(-1)=`{c['sign_reversal']['scores_by_scale']['-1.0']}` | "
            f"RandMaxGain=`{c['orthogonal_control']['rand_max_gain']}` vs CandGain=`{c['orthogonal_control']['cand_gain']}`"
        )
        if c["fail_reasons"]:
            cc_md.append(f"  - Fail reasons: `{c['fail_reasons']}`")
    (EXPERIMENTS_DIR / "track1_causal_checks.md").write_text("\n".join(cc_md) + "\n", encoding="utf-8")

    (EXPERIMENTS_DIR / "track1_layer_freezing_ablation.json").write_text(
        json.dumps(freezing_ablation, indent=2) + "\n", encoding="utf-8"
    )
    (EXPERIMENTS_DIR / "track1_loo_ablation.json").write_text(
        json.dumps(loo_summary, indent=2) + "\n", encoding="utf-8"
    )

    bundle = PatchBundle(
        patches=pruned_patches,
        metadata={"track": "track1", "teacher_mode": teacher_mode, "run_name": run_name},
    )
    save_patch_bundle(
        bundle,
        json_spec_path=EXPERIMENTS_DIR / "track1_patch_spec.json",
        pt_deltas_path=CHECKPOINTS_DIR / "track1_best_deltas.pt",
        extra_summary={"weight_diff": weight_diff},
    )

    payload = {"name": run_name, "split": "select_dev", "weight_diff": weight_diff, **final_report}
    (EXPERIMENTS_DIR / f"{run_name}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    md_text = format_markdown_report(run_name, "select_dev", final_report, weight_diff)
    (EXPERIMENTS_DIR / f"{run_name}.md").write_text(md_text, encoding="utf-8")
    print(md_text)

    del runner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Track 1 gradient-free closed-form weight editing.")
    ap.add_argument("--run-name", default="track1_best_dev")
    ap.add_argument("--teacher-mode", choices=["oracle_chat", "oracle_aligned", "oracle"], default="oracle_chat")
    ap.add_argument("--n-random-seeds", type=int, default=3)
    ap.add_argument("--refresh-cache", action="store_true")
    args = ap.parse_args()
    run_track1_pipeline(
        run_name=args.run_name,
        teacher_mode=args.teacher_mode,
        n_random_seeds=args.n_random_seeds,
        refresh_cache=args.refresh_cache,
    )


if __name__ == "__main__":
    main()
