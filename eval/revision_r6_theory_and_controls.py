"""Revision Suite R6: Quantitative Jacobian Theory, Q6_K/Q5_K_M Investigation,
Bit-Budget-Matched Dissociation, Shared-Subset Warm-Start Gradients, and Multi-Edit Generality.

Merges five diagnostic sections incrementally into `experiments/paper_revision_controls.json`:
  - `r6a_kquant_q6_q5_investigation`: Complete GGUF tensor-type inventory across Q8_0/Q6_K/Q5_K_M/Q4_K_M,
    direct `gguf.dequantize` probing of Q6_K & Q5_K_M (including BF16-edit vs BF16-backbone hybrids),
    and surgical `llama-server` GGUF evaluations (`t2_Q6_K_{6x_bf16,L0_14_q8_0}`, `t2_Q5_K_M_{6x_bf16,L0_14_q8_0}`).
  - `r6b_jacobian_and_margin_decomp`: Exact first-order SwiGLU Jacobian `||grad_h g_j||_2`,
    linearized vs. measured `Delta g_j` correlation/MAE, scale-free relative `|Delta g|/|g|`,
    and exact causal neuron-substitution margin attribution across Q8_0..Q4_K_M.
  - `r6c_bit_matched_controls`: Exact byte-budget-matched downstream (`L15-23+embd`, `L9-23`) vs.
    upstream (`L0-14 only`) controls on Track 2, plus identical `L0-14` protection on Base and Track 3.
  - `r6d_warmstart_fair_grad_and_backprop`: Replays Steps 0..20 across all 5 R4 conditions logging
    common `shared_L15_23_grad_norm` alongside total/upstream/MLP gradient norms, plus backward-hook
    measurement of `||dL/dg_15||_2` and `||dL/dh_15_via_mlp||_2`.
  - `r6e_multi_edit_generality`: Extends Jacobian + `gguf.dequantize` 4-bit causal analysis to two
    additional independent sparse edits: Track 1 (single-layer L15 SwiGLU) and MIT WSP (L15/L19 down_proj only),
    including MIT WSP next-token KL/logit parity on GENERAL_CORPUS (2,026 tokens).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_GGUF_PY = str(Path(__file__).resolve().parent.parent / "third_party" / "llama.cpp" / "gguf-py")
if _GGUF_PY not in sys.path:
    sys.path.insert(0, _GGUF_PY)

import gguf
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download

from automode.questions import Q_BLOCK_BINARY, build_plain_prompt
from automode.view import render_view
from eval.evaluate import load_split_rows
from eval.mechanism_analysis import (
    BASE_ID,
    GENERAL_CORPUS,
    MIT_WSP_ID,
    QUANT_BIN,
    QUANT_TYPES,
    T2_L15_NEURONS,
    T2_L22_NEURONS,
)
from eval.model_runner import QwenDeciderRunner, capture_cpu_state_dict, restore_cpu_state_dict
from eval.revision_quant_mitigations import (
    CONTROLS_JSON,
    CONVERT_HF_GGUF,
    TMP_GGUF_DIR,
    TRACK2_GGUF_TENSORS,
    _load_controls_json,
    _save_controls_json,
)
from eval.revision_r5_amplification_and_cis import (
    BOOTSTRAP_B,
    BOOTSTRAP_SEED,
    _apply_dequant_dict,
    _bootstrap_distribution,
    _dequantize_gguf_2d_weights,
    _paired_diff_summary,
    _probe_model_internal,
    collect_pt_select_dev_ref,
    eval_gguf_select_dev_vectors,
)
from eval.revision_warmstart_controls import MAX_GROUPS, PROBE_OPT_STEPS, _prepare_probe_prompts
from runtime.export_hf import export_patched_hf
from train.lora_model import inject_lora, lora_param_groups
from train.train_track3 import (
    DEFAULT_TRAIN_FILE,
    _cosine_lr_multiplier,
    prepare_training_groups,
    train_group_step,
)
from weight_edit.causal_check import apply_bundle, load_patch_bundle

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = ROOT / "checkpoints"

T1_L15_NEURONS = [291, 917, 1066, 3447]
DIAG_DEV_COUNT = 16


def _blk_num(tname: str) -> int | None:
    if tname.startswith("blk."):
        try:
            return int(tname.split(".")[1])
        except (IndexError, ValueError):
            return None
    return None


def _build_surgical_gguf(
    src_bf16: Path,
    out_path: Path,
    base_q: str,
    tmap: dict[str, str],
    extra_flags: list[str] | None = None,
) -> Path:
    if not out_path.exists():
        tt_file = TMP_GGUF_DIR / f"{out_path.stem}_types.txt"
        tt_file.write_text("\n".join(f"{k}={v}" for k, v in sorted(tmap.items())) + "\n", encoding="utf-8")
        cmd = [str(QUANT_BIN), *(extra_flags or []), "--tensor-type-file", str(tt_file), str(src_bf16), str(out_path), base_q]
        subprocess.run(cmd, check=True)
        tt_file.unlink(missing_ok=True)
    return out_path


def _inventory_gguf(gguf_path: Path) -> dict[str, Any]:
    reader = gguf.GGUFReader(str(gguf_path))
    counts = Counter(str(t.tensor_type.name) for t in reader.tensors)
    by_bytes = Counter()
    for t in reader.tensors:
        by_bytes[str(t.tensor_type.name)] += int(t.n_bytes)
    t2_types = {t.name: str(t.tensor_type.name) for t in reader.tensors if t.name in TRACK2_GGUF_TENSORS}
    q6_tensors = [t.name for t in reader.tensors if str(t.tensor_type.name) == "Q6_K"]
    return {
        "gguf_mib": round(gguf_path.stat().st_size / (1024.0 ** 2), 2),
        "tensor_type_counts": dict(sorted(counts.items())),
        "bytes_mib_by_type": {k: round(v / (1024.0 ** 2), 2) for k, v in sorted(by_bytes.items())},
        "track2_edited_tensor_types": {k: t2_types[k] for k in TRACK2_GGUF_TENSORS if k in t2_types},
        "token_embd_type": next((str(t.tensor_type.name) for t in reader.tensors if t.name == "token_embd.weight"), ""),
        "q6_k_tensor_names_if_mixed": q6_tensors if counts.get("Q6_K", 0) < 50 else [],
    }


def _select_dev_prompt_ids(runner: QwenDeciderRunner, dev_rows: list[dict[str, Any]]) -> list[torch.Tensor]:
    bin_rows = [r for sub in ("real_cf", "consent_5p", "long_50t") for r in dev_rows if r["subset"] == sub]
    prompt_ids: list[torch.Tensor] = []
    for r in bin_rows:
        budget = 3200 if r["subset"] == "long_50t" else 2048
        ctx = render_view(r, tok=runner.tokenizer, budget=budget, include_recap=False).text
        pr = build_plain_prompt(ctx, Q_BLOCK_BINARY, False).text
        ids = runner.tokenizer.encode(pr, add_special_tokens=False)
        prompt_ids.append(torch.tensor([ids], dtype=torch.long))
    return prompt_ids


def _compare_with_rel_and_signed(
    cur: dict[str, Any],
    ref: dict[str, Any],
    l15_unt: torch.Tensor,
    l22_unt: torch.Tensor,
    llama_q_m: np.ndarray | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for layer_tag, idx, unt_mask in (
        ("l15", T2_L15_NEURONS, l15_unt),
        ("l22", T2_L22_NEURONS, l22_unt),
    ):
        hk, gk, mk = f"h{layer_tag[1:]}", f"g{layer_tag[1:]}", f"m{layer_tag[1:]}"
        hc, hr = cur[hk], ref[hk]
        cos = F.cosine_similarity(hc, hr, dim=-1)
        rel_h = torch.linalg.norm(hc - hr, dim=-1) / torch.linalg.norm(hr, dim=-1).clamp_min(1e-8)

        gc_e, gr_e = cur[gk][:, idx], ref[gk][:, idx]
        dg_e = gc_e - gr_e
        dg_e_abs = dg_e.abs()
        gc_u, gr_u = cur[gk][:, unt_mask], ref[gk][:, unt_mask]
        dg_u_abs = (gc_u - gr_u).abs()

        mc, mr = cur[mk], ref[mk]
        rel_m = torch.linalg.norm(mc - mr, dim=-1) / torch.linalg.norm(mr, dim=-1).clamp_min(1e-8)

        ed_mean = float(dg_e_abs.mean().item())
        ed_signed = float(dg_e.mean().item())
        ed_bf16_act = float(gr_e.abs().mean().item())
        un_mean = float(dg_u_abs.mean().item())
        un_bf16_act = float(gr_u.abs().mean().item())

        out[layer_tag] = {
            "h_cos_vs_bf16": round(float(cos.mean().item()), 6),
            "h_rel_l2_drift": round(float(rel_h.mean().item()), 5),
            "edited4_bf16_mean_abs_act": round(ed_bf16_act, 4),
            "edited4_q_mean_abs_act": round(float(gc_e.abs().mean().item()), 4),
            "edited4_signed_mean_delta": round(ed_signed, 5),
            "edited4_mean_abs_delta": round(ed_mean, 5),
            "edited4_rel_shift_dg_over_g": round(ed_mean / max(1e-8, ed_bf16_act), 4),
            "untouched_bf16_mean_abs_act": round(un_bf16_act, 5),
            "untouched_mean_abs_delta": round(un_mean, 5),
            "untouched_rel_shift_dg_over_g": round(un_mean / max(1e-8, un_bf16_act), 4),
            "amplification_vs_untouched_mean": round(ed_mean / max(1e-8, un_mean), 2),
            "mlp_out_rel_l2_drift": round(float(rel_m.mean().item()), 5),
        }

    cm, rm = cur["margins"], ref["margins"]
    cp = 1.0 / (1.0 + np.exp(-np.clip(cm, -30.0, 30.0)))
    rp = 1.0 / (1.0 + np.exp(-np.clip(rm, -30.0, 30.0)))
    logits = {
        "pt_deq_bf16_mean_abs_margin": round(float(np.mean(np.abs(rm))), 4),
        "margin_mae": round(float(np.mean(np.abs(cm - rm))), 4),
        "signed_margin_shift": round(float(np.mean(cm - rm)), 4),
        "pearson_r_margin": round(float(np.corrcoef(rm, cm)[0, 1]), 4),
        "pearson_r_p_block": round(float(np.corrcoef(rp, cp)[0, 1]), 4),
        "p_block_mae": round(float(np.mean(np.abs(cp - rp))), 5),
    }
    if llama_q_m is not None:
        lq_p = 1.0 / (1.0 + np.exp(-np.clip(llama_q_m, -30.0, 30.0)))
        logits["pearson_r_p_vs_llama_cpp"] = round(float(np.corrcoef(lq_p, cp)[0, 1]), 4)
    out["logits"] = logits
    return out


def _compute_swiglu_jacobian_stats(
    runner: QwenDeciderRunner,
    prompt_ids: list[torch.Tensor],
    layer_idx: int,
    neuron_indices: list[int],
    untouched_mask: torch.Tensor,
) -> dict[str, Any]:
    """Compute exact SwiGLU gate Jacobian grad_h g_j(h) at MLP input across all prompts."""
    runner.model.eval()
    h_buf: list[torch.Tensor] = []
    handle = runner.model.model.layers[layer_idx].mlp.register_forward_pre_hook(
        lambda _m, inp: h_buf.append(inp[0][0, -1, :].detach().float().cpu())
    )
    try:
        with torch.inference_mode():
            for ids in prompt_ids:
                runner.model.model(input_ids=ids.to(runner.device), use_cache=False)
    finally:
        handle.remove()

    H = torch.stack(h_buf, dim=0)  # (N, 1024) float32 CPU
    mlp = runner.model.model.layers[layer_idx].mlp
    Wg = mlp.gate_proj.weight.detach().float().cpu()  # (3584, 1024)
    Wu = mlp.up_proj.weight.detach().float().cpu()    # (3584, 1024)

    # Compute exact Jacobian vectors ONLY for edited neurons (N, K, 1024) and norm stats for all 3584 neurons
    Wg_norm2 = (Wg * Wg).sum(dim=1)  # (3584,)
    Wu_norm2 = (Wu * Wu).sum(dim=1)  # (3584,)
    Wg_dot_Wu = (Wg * Wu).sum(dim=1)  # (3584,)

    a = H @ Wg.T  # (N, 3584)
    u = H @ Wu.T  # (N, 3584)
    sig = torch.sigmoid(a)
    silu = a * sig
    silu_p = sig * (1.0 + a * (1.0 - sig))

    alpha = silu_p * u    # coeff on w_g
    beta = silu           # coeff on w_u
    grad_norm = torch.sqrt(
        (alpha * alpha) * Wg_norm2.unsqueeze(0)
        + 2.0 * alpha * beta * Wg_dot_Wu.unsqueeze(0)
        + (beta * beta) * Wu_norm2.unsqueeze(0)
        + 1e-16
    )

    ed_norms = grad_norm[:, neuron_indices]
    un_norms = grad_norm[:, untouched_mask]

    # Exact Jacobian tensor for edited neurons: (N, K, 1024)
    Wg_e = Wg[neuron_indices]  # (K, 1024)
    Wu_e = Wu[neuron_indices]  # (K, 1024)
    J_ed = alpha[:, neuron_indices].unsqueeze(-1) * Wg_e.unsqueeze(0) + beta[:, neuron_indices].unsqueeze(-1) * Wu_e.unsqueeze(0)

    return {
        "H": H,
        "J_ed": J_ed,
        "edited_wg_row_norm_mean": round(float(torch.sqrt(Wg_norm2[neuron_indices]).mean().item()), 4),
        "edited_wu_row_norm_mean": round(float(torch.sqrt(Wu_norm2[neuron_indices]).mean().item()), 4),
        "untouched_wg_row_norm_mean": round(float(torch.sqrt(Wg_norm2[untouched_mask]).mean().item()), 4),
        "untouched_wu_row_norm_mean": round(float(torch.sqrt(Wu_norm2[untouched_mask]).mean().item()), 4),
        "edited_jac_norm_mean": round(float(ed_norms.mean().item()), 4),
        "edited_jac_norm_p95": round(float(torch.quantile(ed_norms.reshape(-1), 0.95).item()), 4),
        "untouched_jac_norm_mean": round(float(un_norms.mean().item()), 4),
        "jac_ratio_vs_untouched": round(float(ed_norms.mean().item() / max(1e-8, un_norms.mean().item())), 2),
    }


@torch.inference_mode()
def _capture_edited_seq_activations(
    runner: QwenDeciderRunner,
    prompt_ids: list[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor], np.ndarray]:
    """Capture full-sequence (T, 4) L15/L22 edited SwiGLU activations and BF16/Q margins."""
    runner.model.eval()
    g15_seq: list[torch.Tensor] = []
    g22_seq: list[torch.Tensor] = []
    margins: list[float] = []
    ab_ids = [runner.letter_token_ids["A"], runner.letter_token_ids["B"]]
    lm_w = runner.model.lm_head.weight[ab_ids]
    lm_b = runner.model.lm_head.bias[ab_ids] if getattr(runner.model.lm_head, "bias", None) is not None else None

    handles = [
        runner.model.model.layers[15].mlp.down_proj.register_forward_pre_hook(
            lambda _m, inp: g15_seq.append(inp[0][0, :, T2_L15_NEURONS].detach().clone().cpu())
        ),
        runner.model.model.layers[22].mlp.down_proj.register_forward_pre_hook(
            lambda _m, inp: g22_seq.append(inp[0][0, :, T2_L22_NEURONS].detach().clone().cpu())
        ),
    ]
    try:
        for ids in prompt_ids:
            hid = runner.model.model(input_ids=ids.to(runner.device), use_cache=False).last_hidden_state[:, -1:, :]
            lg = F.linear(hid.to(dtype=lm_w.dtype), lm_w, lm_b)[0, 0, :].float()
            margins.append(float((lg[1] - lg[0]).item()))
    finally:
        for h in handles:
            h.remove()
    return g15_seq, g22_seq, np.array(margins, dtype=np.float64)


@torch.inference_mode()
def _run_with_neuron_substitution(
    runner: QwenDeciderRunner,
    prompt_ids: list[torch.Tensor],
    src_g15: list[torch.Tensor] | None,
    src_g22: list[torch.Tensor] | None,
) -> np.ndarray:
    """Run forward passes replacing L15/L22 edited-4 activations with supplied full-sequence tensors."""
    runner.model.eval()
    ab_ids = [runner.letter_token_ids["A"], runner.letter_token_ids["B"]]
    lm_w = runner.model.lm_head.weight[ab_ids]
    lm_b = runner.model.lm_head.bias[ab_ids] if getattr(runner.model.lm_head, "bias", None) is not None else None
    cur_idx = [0]
    handles = []

    if src_g15 is not None:
        def _hook15(_m, inp):
            x = inp[0].clone()
            x[0, :, T2_L15_NEURONS] = src_g15[cur_idx[0]].to(device=x.device, dtype=x.dtype)
            return (x,) + inp[1:]
        handles.append(runner.model.model.layers[15].mlp.down_proj.register_forward_pre_hook(_hook15))

    if src_g22 is not None:
        def _hook22(_m, inp):
            x = inp[0].clone()
            x[0, :, T2_L22_NEURONS] = src_g22[cur_idx[0]].to(device=x.device, dtype=x.dtype)
            return (x,) + inp[1:]
        handles.append(runner.model.model.layers[22].mlp.down_proj.register_forward_pre_hook(_hook22))

    margins: list[float] = []
    try:
        for i, ids in enumerate(prompt_ids):
            cur_idx[0] = i
            hid = runner.model.model(input_ids=ids.to(runner.device), use_cache=False).last_hidden_state[:, -1:, :]
            lg = F.linear(hid.to(dtype=lm_w.dtype), lm_w, lm_b)[0, 0, :].float()
            margins.append(float((lg[1] - lg[0]).item()))
    finally:
        for h in handles:
            h.remove()
    return np.array(margins, dtype=np.float64)


def run_r6a_b_c(
    runner: QwenDeciderRunner,
    base_cpu: dict[str, torch.Tensor],
    dev_rows: list[dict[str, Any]],
    refs: dict[str, dict[str, Any]],
    port: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Execute R6a (Q6_K/Q5_K_M investigation), R6b (Jacobian + causal margin decomp), and R6c (bit-matched controls)."""
    TMP_GGUF_DIR.mkdir(parents=True, exist_ok=True)
    ab_ids = [runner.letter_token_ids["A"], runner.letter_token_ids["B"]]
    prompt_ids = _select_dev_prompt_ids(runner, dev_rows)

    # --- 1. Prepare Base BF16/Q8/Q6/Q5/Q4 and Track 2/3 GGUF paths ---
    base_snap = Path(snapshot_download(BASE_ID, local_files_only=True))
    base_bf16 = TMP_GGUF_DIR / "base-bf16.gguf"
    if not base_bf16.exists():
        subprocess.run(
            [sys.executable, str(CONVERT_HF_GGUF), str(base_snap), "--outfile", str(base_bf16), "--outtype", "bf16"],
            check=True,
        )
    base_q_paths = {"bf16": base_bf16}
    for qt in QUANT_TYPES:
        dst = TMP_GGUF_DIR / f"base-{qt}.gguf"
        if not dst.exists():
            subprocess.run([str(QUANT_BIN), str(base_bf16), str(dst), qt], check=True)
        base_q_paths[qt] = dst

    t2_bf16 = CHECKPOINTS_DIR / "wsp-automode-0.8b-bf16.gguf"
    t3_bf16 = CHECKPOINTS_DIR / "automode-0.8b-track3-bf16.gguf"
    t2_q_paths = {qt: CHECKPOINTS_DIR / f"wsp-automode-0.8b-{qt}.gguf" for qt in QUANT_TYPES}
    t3_q_paths = {qt: CHECKPOINTS_DIR / f"automode-0.8b-track3-{qt}.gguf" for qt in QUANT_TYPES}

    # Inventory across all 4 quant formats
    inventory = {f"track2_{qt}": _inventory_gguf(t2_q_paths[qt]) for qt in QUANT_TYPES}
    print("[R6a] Tensor inventory verified across Q8_0..Q4_K_M.", flush=True)

    # Build L0-14 / L15-23 / L9-23 maps
    t2_reader = gguf.GGUFReader(str(t2_bf16))
    bf16_names = [t.name for t in t2_reader.tensors if str(t.tensor_type.name) in ("BF16", "F16")]
    t2_set = set(TRACK2_GGUF_TENSORS)

    map_t2_6x_bf16 = {t: "bf16" for t in TRACK2_GGUF_TENSORS}
    map_L0_14_with_embd_bf16: dict[str, str] = {}
    map_L0_14_only: dict[str, str] = {}
    map_L15_23_embd: dict[str, str] = {}
    map_L9_23_only: dict[str, str] = {}

    for tname in bf16_names:
        bn = _blk_num(tname)
        if tname in t2_set:
            map_L0_14_with_embd_bf16[tname] = "bf16"
        elif tname in ("token_embd.weight", "output.weight"):
            map_L0_14_with_embd_bf16[tname] = "q8_0"
        elif bn is not None and bn <= 14:
            map_L0_14_with_embd_bf16[tname] = "q8_0"

        if bn is not None and bn <= 14:
            map_L0_14_only[tname] = "q8_0"
        if (bn is not None and bn >= 15) or tname in ("token_embd.weight", "output.weight"):
            map_L15_23_embd[tname] = "q8_0"
        if bn is not None and bn >= 9:
            map_L9_23_only[tname] = "q8_0"

    # Build surgical R6a + R6c GGUFs
    embd_flags = ["--token-embedding-type", "q8_0", "--output-tensor-type", "q8_0"]
    r6_ggufs = {
        "t2_Q6_K_6x_bf16": ("t2", _build_surgical_gguf(t2_bf16, TMP_GGUF_DIR / "r6a_t2_Q6_K_6x_bf16.gguf", "Q6_K", map_t2_6x_bf16)),
        "t2_Q6_K_L0_14_q8_0": ("t2", _build_surgical_gguf(t2_bf16, TMP_GGUF_DIR / "r6a_t2_Q6_K_L0_14_q8_0.gguf", "Q6_K", map_L0_14_with_embd_bf16, embd_flags)),
        "t2_Q5_K_M_6x_bf16": ("t2", _build_surgical_gguf(t2_bf16, TMP_GGUF_DIR / "r6a_t2_Q5_K_M_6x_bf16.gguf", "Q5_K_M", map_t2_6x_bf16)),
        "t2_Q5_K_M_L0_14_q8_0": ("t2", _build_surgical_gguf(t2_bf16, TMP_GGUF_DIR / "r6a_t2_Q5_K_M_L0_14_q8_0.gguf", "Q5_K_M", map_L0_14_with_embd_bf16, embd_flags)),
        "t2_Q4_K_M_L0_14_only_q8_0": ("t2", _build_surgical_gguf(t2_bf16, TMP_GGUF_DIR / "r6c_t2_Q4_K_M_L0_14_only_q8_0.gguf", "Q4_K_M", map_L0_14_only)),
        "t2_Q4_K_M_L15_23_embd_q8_0": ("t2", _build_surgical_gguf(t2_bf16, TMP_GGUF_DIR / "r6c_t2_Q4_K_M_L15_23_embd_q8_0.gguf", "Q4_K_M", map_L15_23_embd, embd_flags)),
        "t2_Q4_K_M_L9_23_only_q8_0": ("t2", _build_surgical_gguf(t2_bf16, TMP_GGUF_DIR / "r6c_t2_Q4_K_M_L9_23_only_q8_0.gguf", "Q4_K_M", map_L9_23_only)),
        "base_Q4_K_M_L0_14_only_q8_0": ("base", _build_surgical_gguf(base_bf16, TMP_GGUF_DIR / "r6c_base_Q4_K_M_L0_14_only_q8_0.gguf", "Q4_K_M", map_L0_14_only)),
        "track3_Q4_K_M_L0_14_only_q8_0": ("t3", _build_surgical_gguf(t3_bf16, TMP_GGUF_DIR / "r6c_track3_Q4_K_M_L0_14_only_q8_0.gguf", "Q4_K_M", map_L0_14_only)),
    }

    # --- 2. Evaluate all new surgical GGUFs via llama-server + vanilla refs needed for paired bootstrap diffs ---
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n_bin = len(refs["t2"]["binary"])
    n_all = n_bin + len(refs["t2"]["replay"])
    idx_bin = rng.integers(0, n_bin, size=(BOOTSTRAP_B, n_bin), dtype=np.int32)
    idx_all = rng.integers(0, n_all, size=(BOOTSTRAP_B, n_all), dtype=np.int32)

    all_targets: list[tuple[str, str, Path]] = [
        ("base_Q4_K_M", "base", base_q_paths["Q4_K_M"]),
        ("track2_Q6_K", "t2", t2_q_paths["Q6_K"]),
        ("track2_Q5_K_M", "t2", t2_q_paths["Q5_K_M"]),
        ("track2_Q4_K_M", "t2", t2_q_paths["Q4_K_M"]),
        ("track3_Q4_K_M", "t3", t3_q_paths["Q4_K_M"]),
        *[(k, v[0], v[1]) for k, v in r6_ggufs.items()],
    ]
    srv_summaries: dict[str, Any] = {}
    srv_stats: dict[str, dict[str, Any]] = {}
    srv_dists: dict[str, dict[str, np.ndarray]] = {}
    srv_vecs: dict[str, dict[str, np.ndarray]] = {}

    for key, ref_k, gpath in all_targets:
        print(f"[R6 Server] Evaluating {key} ({gpath.name})...", flush=True)
        summary, vecs = eval_gguf_select_dev_vectors(gpath, refs[ref_k], port)
        b_stats, b_dists = _bootstrap_distribution(vecs, idx_bin, idx_all)
        inv_extra = {"token_embd_type": next((str(t.tensor_type.name) for t in gguf.GGUFReader(str(gpath)).tensors if t.name == "token_embd.weight"), "")}
        srv_summaries[key] = {**summary, **inv_extra, "bootstrap_ci95": b_stats}
        srv_stats[key] = b_stats
        srv_dists[key] = b_dists
        srv_vecs[key] = vecs

    # --- 3. Load Track 3 non-2D params (norms + DeltaNet gates) for exact GGUF dequantize probing ---
    t3_bundle = load_patch_bundle(CHECKPOINTS_DIR / "track3_final_best_deltas.pt")
    apply_bundle(runner.model, base_cpu, t3_bundle.patches)
    t3_full_cpu = capture_cpu_state_dict(runner.model)
    restore_cpu_state_dict(runner.model, base_cpu)

    w_base = {qt: _dequantize_gguf_2d_weights(base_q_paths[qt]) for qt in ("bf16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M")}
    w_t2 = {qt: _dequantize_gguf_2d_weights(t2_bf16 if qt == "bf16" else t2_q_paths[qt]) for qt in ("bf16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M")}
    w_t3 = {qt: _dequantize_gguf_2d_weights(t3_bf16 if qt == "bf16" else t3_q_paths[qt]) for qt in ("bf16", "Q6_K", "Q5_K_M", "Q4_K_M")}

    d_ff = runner.model.model.layers[15].mlp.gate_proj.weight.shape[0]
    l15_unt = torch.ones(d_ff, dtype=torch.bool)
    l15_unt[T2_L15_NEURONS] = False
    l22_unt = torch.ones(d_ff, dtype=torch.bool)
    l22_unt[T2_L22_NEURONS] = False

    # Base, Track 2, Track 3 BF16 refs & Jacobians (R6b)
    _apply_dequant_dict(runner.model, base_cpu, w_base["bf16"])
    ref_base = _probe_model_internal(runner, prompt_ids, ab_ids)
    jac_base_l15 = _compute_swiglu_jacobian_stats(runner, prompt_ids, 15, T2_L15_NEURONS, l15_unt)
    jac_base_l22 = _compute_swiglu_jacobian_stats(runner, prompt_ids, 22, T2_L22_NEURONS, l22_unt)

    _apply_dequant_dict(runner.model, base_cpu, w_t2["bf16"])
    ref_t2 = _probe_model_internal(runner, prompt_ids, ab_ids)
    jac_t2_l15 = _compute_swiglu_jacobian_stats(runner, prompt_ids, 15, T2_L15_NEURONS, l15_unt)
    jac_t2_l22 = _compute_swiglu_jacobian_stats(runner, prompt_ids, 22, T2_L22_NEURONS, l22_unt)
    bf16_g15_seq, bf16_g22_seq, bf16_t2_margins = _capture_edited_seq_activations(runner, prompt_ids)

    _apply_dequant_dict(runner.model, t3_full_cpu, w_t3["bf16"])
    ref_t3 = _probe_model_internal(runner, prompt_ids, ab_ids)
    jac_t3_l15 = _compute_swiglu_jacobian_stats(runner, prompt_ids, 15, T2_L15_NEURONS, l15_unt)
    jac_t3_l22 = _compute_swiglu_jacobian_stats(runner, prompt_ids, 22, T2_L22_NEURONS, l22_unt)

    # Probe all Q8_0 / Q6_K / Q5_K_M / Q4_K_M + R6a Q6/Q5 hybrids on dequantized PyTorch
    t2_edit_set = set(TRACK2_GGUF_TENSORS)
    w_t2_bf16_ed = _dequantize_gguf_2d_weights(t2_bf16, filter_names=t2_edit_set)
    w_t2_q_ed = {qt: _dequantize_gguf_2d_weights(t2_q_paths[qt], filter_names=t2_edit_set) for qt in ("Q6_K", "Q5_K_M", "Q4_K_M")}

    probe_specs = [
        ("base_Q8_0", base_cpu, w_base["Q8_0"], ref_base, None),
        ("base_Q6_K", base_cpu, w_base["Q6_K"], ref_base, None),
        ("base_Q5_K_M", base_cpu, w_base["Q5_K_M"], ref_base, None),
        ("base_Q4_K_M", base_cpu, w_base["Q4_K_M"], ref_base, srv_vecs["base_Q4_K_M"]["q_m"]),
        ("track2_Q8_0", base_cpu, w_t2["Q8_0"], ref_t2, None),
        ("track2_Q6_K", base_cpu, w_t2["Q6_K"], ref_t2, srv_vecs["track2_Q6_K"]["q_m"]),
        ("track2_Q5_K_M", base_cpu, w_t2["Q5_K_M"], ref_t2, srv_vecs["track2_Q5_K_M"]["q_m"]),
        ("track2_Q4_K_M", base_cpu, w_t2["Q4_K_M"], ref_t2, srv_vecs["track2_Q4_K_M"]["q_m"]),
        ("track2_Q6_bb_bf16_edits", base_cpu, {**w_t2["Q6_K"], **w_t2_bf16_ed}, ref_t2, srv_vecs["t2_Q6_K_6x_bf16"]["q_m"]),
        ("track2_bf16_bb_Q6_edits", base_cpu, {**w_t2["bf16"], **w_t2_q_ed["Q6_K"]}, ref_t2, None),
        ("track2_Q5_bb_bf16_edits", base_cpu, {**w_t2["Q5_K_M"], **w_t2_bf16_ed}, ref_t2, srv_vecs["t2_Q5_K_M_6x_bf16"]["q_m"]),
        ("track2_bf16_bb_Q5_edits", base_cpu, {**w_t2["bf16"], **w_t2_q_ed["Q5_K_M"]}, ref_t2, None),
        ("track2_Q4_bb_bf16_edits", base_cpu, {**w_t2["Q4_K_M"], **w_t2_bf16_ed}, ref_t2, None),
        ("track2_bf16_bb_Q4_edits", base_cpu, {**w_t2["bf16"], **w_t2_q_ed["Q4_K_M"]}, ref_t2, None),
        ("track3_Q6_K", t3_full_cpu, w_t3["Q6_K"], ref_t3, None),
        ("track3_Q5_K_M", t3_full_cpu, w_t3["Q5_K_M"], ref_t3, None),
        ("track3_Q4_K_M", t3_full_cpu, w_t3["Q4_K_M"], ref_t3, srv_vecs["track3_Q4_K_M"]["q_m"]),
    ]

    dequant_probes: dict[str, Any] = {}
    t2_raw_probes: dict[str, dict[str, Any]] = {}
    for pname, b_st, wd, r_ref, lq_m in probe_specs:
        print(f"[R6a/b Dequant] Probing {pname}...", flush=True)
        _apply_dequant_dict(runner.model, b_st, wd)
        cur_p = _probe_model_internal(runner, prompt_ids, ab_ids)
        dequant_probes[pname] = _compare_with_rel_and_signed(cur_p, r_ref, l15_unt, l22_unt, llama_q_m=lq_m)
        if pname.startswith("track2_"):
            t2_raw_probes[pname] = cur_p

    # Linearized Jacobian prediction check + causal 8-neuron margin substitution across Q8_0..Q4_K_M (R6b)
    lin_validation: dict[str, Any] = {}
    causal_decomp: dict[str, Any] = {}

    for qt in ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"):
        # (a) First-order prediction accuracy on Q_full and (where available) Q-bb + BF16-edits
        cur_variants = [f"track2_{qt}"]
        bb_only_k = f"track2_{qt.split('_')[0]}_bb_bf16_edits"
        if bb_only_k in t2_raw_probes:
            cur_variants.append(bb_only_k)
        for vk in cur_variants:
            cp = t2_raw_probes[vk]
            lv: dict[str, Any] = {}
            for lt, r_h, jac_d, idx in (
                ("l15", ref_t2["h15"], jac_t2_l15, T2_L15_NEURONS),
                ("l22", ref_t2["h22"], jac_t2_l22, T2_L22_NEURONS),
            ):
                dh = cp[f"h{lt[1:]}"] - r_h  # (N, 1024)
                dg_pred = (jac_d["J_ed"] * dh.unsqueeze(1)).sum(dim=-1)  # (N, 4)
                dg_true = cp[f"g{lt[1:]}"] [:, idx] - ref_t2[f"g{lt[1:]}"][:, idx]
                p_flat = dg_pred.reshape(-1).numpy()
                t_flat = dg_true.reshape(-1).numpy()
                lv[lt] = {
                    "pearson_r_lin_vs_true": round(float(np.corrcoef(p_flat, t_flat)[0, 1]), 4),
                    "mae_lin_vs_true": round(float(np.mean(np.abs(p_flat - t_flat))), 4),
                    "mean_abs_dg_pred": round(float(np.mean(np.abs(p_flat))), 4),
                    "mean_abs_dg_true": round(float(np.mean(np.abs(t_flat))), 4),
                }
            lin_validation[vk] = lv

        # (b) Exact causal 8-neuron margin attribution for Track 2 @ qt
        print(f"[R6b Attribution] Running 8-neuron causal swaps for {qt}...", flush=True)
        _apply_dequant_dict(runner.model, base_cpu, w_t2[qt])
        q_g15_seq, q_g22_seq, q_full_m = _capture_edited_seq_activations(runner, prompt_ids)
        m_rescue = _run_with_neuron_substitution(runner, prompt_ids, bf16_g15_seq, bf16_g22_seq)

        _apply_dequant_dict(runner.model, base_cpu, w_t2["bf16"])
        m_inj_l15 = _run_with_neuron_substitution(runner, prompt_ids, q_g15_seq, None)
        m_inj_l22 = _run_with_neuron_substitution(runner, prompt_ids, None, q_g22_seq)
        m_inj_both = _run_with_neuron_substitution(runner, prompt_ids, q_g15_seq, q_g22_seq)

        full_mae = float(np.mean(np.abs(q_full_m - bf16_t2_margins)))
        inj_both_mae = float(np.mean(np.abs(m_inj_both - bf16_t2_margins)))
        rescue_mae = float(np.mean(np.abs(m_rescue - bf16_t2_margins)))
        causal_decomp[qt] = {
            "full_q_margin_mae": round(full_mae, 4),
            "full_q_signed_bias": round(float(np.mean(q_full_m - bf16_t2_margins)), 4),
            "full_q_pearson_r_margin": round(float(np.corrcoef(bf16_t2_margins, q_full_m)[0, 1]), 4),
            "bf16_inject_q_L15_ed4_mae": round(float(np.mean(np.abs(m_inj_l15 - bf16_t2_margins))), 4),
            "bf16_inject_q_L15_ed4_bias": round(float(np.mean(m_inj_l15 - bf16_t2_margins)), 4),
            "bf16_inject_q_L22_ed4_mae": round(float(np.mean(np.abs(m_inj_l22 - bf16_t2_margins))), 4),
            "bf16_inject_q_L22_ed4_bias": round(float(np.mean(m_inj_l22 - bf16_t2_margins)), 4),
            "bf16_inject_q_L15_L22_ed8_mae": round(inj_both_mae, 4),
            "bf16_inject_q_L15_L22_ed8_bias": round(float(np.mean(m_inj_both - bf16_t2_margins)), 4),
            "bf16_inject_q_ed8_r_vs_full_q": round(float(np.corrcoef(q_full_m, m_inj_both)[0, 1]), 4),
            "bf16_inject_q_ed8_r_vs_bf16": round(float(np.corrcoef(bf16_t2_margins, m_inj_both)[0, 1]), 4),
            "q_rescue_bf16_ed8_mae": round(rescue_mae, 4),
            "q_rescue_bf16_ed8_bias": round(float(np.mean(m_rescue - bf16_t2_margins)), 4),
            "q_rescue_bf16_ed8_r_vs_bf16": round(float(np.corrcoef(bf16_t2_margins, m_rescue)[0, 1]), 4),
            "pct_total_abs_m_drift_removed_by_8neu_rescue": round(100.0 * (full_mae - rescue_mae) / max(1e-8, full_mae), 1),
        }

    # --- 4. Paired bootstrap differences for R6a + R6c ---
    r6a_paired = [
        _paired_diff_summary("t2_Q6_K_L0_14_q8_0", "t2_Q6_K_6x_bf16", srv_stats["t2_Q6_K_L0_14_q8_0"], srv_stats["t2_Q6_K_6x_bf16"], srv_dists["t2_Q6_K_L0_14_q8_0"], srv_dists["t2_Q6_K_6x_bf16"], "margin_mae"),
        _paired_diff_summary("t2_Q6_K_L0_14_q8_0", "t2_Q6_K_6x_bf16", srv_stats["t2_Q6_K_L0_14_q8_0"], srv_stats["t2_Q6_K_6x_bf16"], srv_dists["t2_Q6_K_L0_14_q8_0"], srv_dists["t2_Q6_K_6x_bf16"], "pearson_r"),
        _paired_diff_summary("t2_Q5_K_M_L0_14_q8_0", "t2_Q5_K_M_6x_bf16", srv_stats["t2_Q5_K_M_L0_14_q8_0"], srv_stats["t2_Q5_K_M_6x_bf16"], srv_dists["t2_Q5_K_M_L0_14_q8_0"], srv_dists["t2_Q5_K_M_6x_bf16"], "margin_mae"),
        _paired_diff_summary("t2_Q5_K_M_L0_14_q8_0", "t2_Q5_K_M_6x_bf16", srv_stats["t2_Q5_K_M_L0_14_q8_0"], srv_stats["t2_Q5_K_M_6x_bf16"], srv_dists["t2_Q5_K_M_L0_14_q8_0"], srv_dists["t2_Q5_K_M_6x_bf16"], "pearson_r"),
    ]
    r6c_paired = [
        _paired_diff_summary("t2_Q4_K_M_L0_14_only_q8_0", "track2_Q4_K_M", srv_stats["t2_Q4_K_M_L0_14_only_q8_0"], srv_stats["track2_Q4_K_M"], srv_dists["t2_Q4_K_M_L0_14_only_q8_0"], srv_dists["track2_Q4_K_M"], "pearson_r"),
        _paired_diff_summary("t2_Q4_K_M_L0_14_only_q8_0", "t2_Q4_K_M_L15_23_embd_q8_0", srv_stats["t2_Q4_K_M_L0_14_only_q8_0"], srv_stats["t2_Q4_K_M_L15_23_embd_q8_0"], srv_dists["t2_Q4_K_M_L0_14_only_q8_0"], srv_dists["t2_Q4_K_M_L15_23_embd_q8_0"], "pearson_r"),
        _paired_diff_summary("t2_Q4_K_M_L0_14_only_q8_0", "t2_Q4_K_M_L9_23_only_q8_0", srv_stats["t2_Q4_K_M_L0_14_only_q8_0"], srv_stats["t2_Q4_K_M_L9_23_only_q8_0"], srv_dists["t2_Q4_K_M_L0_14_only_q8_0"], srv_dists["t2_Q4_K_M_L9_23_only_q8_0"], "pearson_r"),
        _paired_diff_summary("t2_Q4_K_M_L0_14_only_q8_0", "t2_Q4_K_M_L15_23_embd_q8_0", srv_stats["t2_Q4_K_M_L0_14_only_q8_0"], srv_stats["t2_Q4_K_M_L15_23_embd_q8_0"], srv_dists["t2_Q4_K_M_L0_14_only_q8_0"], srv_dists["t2_Q4_K_M_L15_23_embd_q8_0"], "margin_mae"),
        _paired_diff_summary("t2_Q4_K_M_L0_14_only_q8_0", "t2_Q4_K_M_L9_23_only_q8_0", srv_stats["t2_Q4_K_M_L0_14_only_q8_0"], srv_stats["t2_Q4_K_M_L9_23_only_q8_0"], srv_dists["t2_Q4_K_M_L0_14_only_q8_0"], srv_dists["t2_Q4_K_M_L9_23_only_q8_0"], "margin_mae"),
        _paired_diff_summary("base_Q4_K_M_L0_14_only_q8_0", "base_Q4_K_M", srv_stats["base_Q4_K_M_L0_14_only_q8_0"], srv_stats["base_Q4_K_M"], srv_dists["base_Q4_K_M_L0_14_only_q8_0"], srv_dists["base_Q4_K_M"], "margin_mae"),
        _paired_diff_summary("track3_Q4_K_M_L0_14_only_q8_0", "track3_Q4_K_M", srv_stats["track3_Q4_K_M_L0_14_only_q8_0"], srv_stats["track3_Q4_K_M"], srv_dists["track3_Q4_K_M_L0_14_only_q8_0"], srv_dists["track3_Q4_K_M"], "margin_mae"),
    ]

    def _strip_tensors(d: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in d.items() if k not in ("H", "J_ed")}

    r6a = {
        "gguf_tensor_inventory": inventory,
        "server_configs": {k: srv_summaries[k] for k in ("track2_Q6_K", "t2_Q6_K_6x_bf16", "t2_Q6_K_L0_14_q8_0", "track2_Q5_K_M", "t2_Q5_K_M_6x_bf16", "t2_Q5_K_M_L0_14_q8_0")},
        "paired_differences": r6a_paired,
        "dequant_probes": dequant_probes,
    }
    r6b = {
        "jacobian_stats": {
            "base": {"l15": _strip_tensors(jac_base_l15), "l22": _strip_tensors(jac_base_l22)},
            "track2": {"l15": _strip_tensors(jac_t2_l15), "l22": _strip_tensors(jac_t2_l22)},
            "track3": {"l15": _strip_tensors(jac_t3_l15), "l22": _strip_tensors(jac_t3_l22)},
        },
        "linearized_prediction_validation": lin_validation,
        "causal_8neuron_margin_attribution": causal_decomp,
    }
    r6c = {
        "configs": {
            k: srv_summaries[k]
            for k in (
                "track2_Q4_K_M",
                "t2_Q4_K_M_L0_14_only_q8_0",
                "t2_Q4_K_M_L15_23_embd_q8_0",
                "t2_Q4_K_M_L9_23_only_q8_0",
                "base_Q4_K_M",
                "base_Q4_K_M_L0_14_only_q8_0",
                "track3_Q4_K_M",
                "track3_Q4_K_M_L0_14_only_q8_0",
            )
        },
        "paired_differences": r6c_paired,
    }
    restore_cpu_state_dict(runner.model, base_cpu)
    del w_base, w_t2, w_t3, t3_full_cpu, bf16_g15_seq, bf16_g22_seq
    gc.collect()
    return r6a, r6b, r6c


def _measure_warmstart_backward_probe(
    runner: QwenDeciderRunner,
    diag_items: list[tuple[torch.Tensor, float]],
    ab_ids: list[int],
) -> dict[str, Any]:
    """Run checkpointed diagnostic forward+backward pass on fixed 16-item subset to capture ||dL/dg_15|| and ||dL/dh_15||."""
    d_ff = int(runner.model.model.layers[15].mlp.down_proj.in_features)
    l15_unt_mask = torch.ones(d_ff, dtype=torch.bool)
    l15_unt_mask[T2_L15_NEURONS] = False

    loss_sum = 0.0
    dm_vals: list[float] = []
    dg_ed_sq = 0.0
    dg_un_sq = 0.0
    dh_mlp_sq = 0.0

    def _emb_req_grad(_m, _inp, out):
        return out.detach().requires_grad_(True) if not out.requires_grad else out

    def _bwd_mlp(_m, grad_in, _grad_out):
        nonlocal dh_mlp_sq
        g = grad_in[0]
        if g is not None:
            hv = g[0, -1, :].detach().float().cpu()
            dh_mlp_sq += float((hv ** 2).sum().item())

    def _bwd_down(_m, grad_in, _grad_out):
        nonlocal dg_ed_sq, dg_un_sq
        g = grad_in[0]
        if g is not None:
            v = g[0, -1, :].detach().float().cpu()
            dg_ed_sq += float((v[T2_L15_NEURONS] ** 2).sum().item())
            dg_un_sq += float((v[l15_unt_mask] ** 2).sum().item())

    # Keep train() active so gradient checkpointing stays active within 4 GiB VRAM (LoRA dropout=0)
    runner.model.train()
    h_emb = runner.model.get_input_embeddings().register_forward_hook(_emb_req_grad)
    h_mlp = runner.model.model.layers[15].mlp.register_full_backward_hook(_bwd_mlp)
    h_down = runner.model.model.layers[15].mlp.down_proj.register_full_backward_hook(_bwd_down)

    lm_w = runner.model.lm_head.weight[ab_ids]
    lm_b = runner.model.lm_head.bias[ab_ids] if getattr(runner.model.lm_head, "bias", None) is not None else None
    try:
        for ids, y_val in diag_items:
            runner.model.zero_grad(set_to_none=True)
            hid = runner.model.model(input_ids=ids.to(runner.device), use_cache=False).last_hidden_state[:, -1:, :]
            lg = F.linear(hid.to(dtype=lm_w.dtype), lm_w, lm_b)[0, 0, :].float()
            margin = lg[1] - lg[0]
            tgt = torch.tensor(y_val, dtype=torch.float32, device=margin.device)
            loss = F.binary_cross_entropy_with_logits(margin, tgt) / float(len(diag_items))
            loss_sum += float(loss.item())
            dm_vals.append(float((torch.sigmoid(margin.detach()) - tgt).abs().item() / len(diag_items)))
            loss.backward()
    finally:
        h_emb.remove()
        h_mlp.remove()
        h_down.remove()
        runner.model.zero_grad(set_to_none=True)

    n = float(len(diag_items))
    return {
        "diag_bce_loss": round(loss_sum, 4),
        "diag_mean_abs_dL_dm": round(float(sum(dm_vals)), 5),
        "rms_dL_dg15_ed4": round(math.sqrt(dg_ed_sq / max(1.0, 4.0 * n)), 5),
        "rms_dL_dg15_unt": round(math.sqrt(dg_un_sq / max(1.0, float(l15_unt_mask.sum().item()) * n)), 6),
        "rms_dL_dh15_via_mlp": round(math.sqrt(dh_mlp_sq / n), 4),
    }


def _group_grad_norm(params: list[torch.nn.Parameter]) -> float:
    sq = sum(float(p.grad.detach().float().pow(2).sum().item()) for p in params if p.grad is not None)
    return math.sqrt(sq)


def _run_r6d_condition(
    cond_name: str,
    lr: float,
    gate_lr: float,
    freeze_mode: str,
    init_from_track2: bool,
    dev_rows: list[dict[str, Any]],
    diag_items: list[tuple[torch.Tensor, float]],
) -> dict[str, Any]:
    seed = 20261124
    grad_accum = 8
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    runner = QwenDeciderRunner()
    base_cpu = capture_cpu_state_dict(runner.model)
    ab_ids = [runner.letter_token_ids["A"], runner.letter_token_ids["B"]]
    groups = prepare_training_groups(runner, Path(DEFAULT_TRAIN_FILE))

    if init_from_track2:
        t2_bundle = load_patch_bundle(CHECKPOINTS_DIR / "track2_best_deltas.pt")
        apply_bundle(runner.model, base_cpu, t2_bundle.patches)

    inject_lora(runner.model, r=16, alpha=32.0, train_deltanet_gates=True)

    # Identify exact parameter subsets before freezing so `shared_L15_23` is 100% identical across ALL 5 conditions
    shared_L15_23_params: list[torch.nn.Parameter] = []
    l0_14_params: list[torch.nn.Parameter] = []
    t2_mlp_params: list[torch.nn.Parameter] = []
    other_params: list[torch.nn.Parameter] = []

    lora_params_all, gate_params_all = lora_param_groups(runner.model)
    trainable_set = set(lora_params_all + gate_params_all)
    for n, p in runner.model.named_parameters():
        if p not in trainable_set:
            continue
        if n.startswith("model.layers."):
            l_idx = int(n.split(".")[2])
            if l_idx < 15:
                l0_14_params.append(p)
            elif ("model.layers.15.mlp." in n) or ("model.layers.22.mlp." in n):
                t2_mlp_params.append(p)
            else:
                shared_L15_23_params.append(p)
        else:
            other_params.append(p)

    if freeze_mode == "freeze_L0_14":
        for p in l0_14_params:
            p.requires_grad_(False)
        need_input_grad = False
    elif freeze_mode == "freeze_t2gates":
        for p in t2_mlp_params:
            p.requires_grad_(False)
        need_input_grad = True
    elif freeze_mode == "none":
        need_input_grad = True
    else:
        raise ValueError(freeze_mode)

    lora_params, gate_params = lora_param_groups(runner.model)
    all_trainable = lora_params + gate_params
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": lr, "weight_decay": 0.02},
            {"params": gate_params, "lr": gate_lr, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.95),
    )
    steps_per_epoch = math.ceil(len(groups) / grad_accum)
    total_opt_steps = 2 * steps_per_epoch
    warmup_steps = max(1, int(0.08 * total_opt_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: _cosine_lr_multiplier(s, total_opt_steps, warmup_steps, min_ratio=0.1)
    )

    runner.model.train()
    if need_input_grad:
        runner.model.enable_input_require_grads()
    runner.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    step0_bwd = _measure_warmstart_backward_probe(runner, diag_items, ab_ids)
    snapshots: list[dict[str, Any]] = [{"opt_step": 0, **step0_bwd}]

    rng = random.Random(seed)
    order = list(range(len(groups)))
    rng.shuffle(order)
    optimizer.zero_grad(set_to_none=True)

    step_grad_logs: list[dict[str, Any]] = []
    opt_step = 0
    for step_idx, g_i in enumerate(order[:MAX_GROUPS]):
        stats = train_group_step(runner.model, runner.device, ab_ids, groups[g_i], grad_accum)
        is_bnd = ((step_idx + 1) % grad_accum == 0) or (step_idx == MAX_GROUPS - 1)
        if is_bnd:
            gn_total = _group_grad_norm(all_trainable)
            gn_shared = _group_grad_norm(shared_L15_23_params)
            gn_l0_14 = _group_grad_norm(l0_14_params)
            gn_t2_mlp = _group_grad_norm(t2_mlp_params)

            torch.nn.utils.clip_grad_norm_(all_trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            opt_step += 1

            step_grad_logs.append(
                {
                    "opt_step": opt_step,
                    "total_grad_norm": round(gn_total, 4),
                    "shared_L15_23_grad_norm": round(gn_shared, 4),
                    "L0_14_grad_norm": round(gn_l0_14, 4),
                    "L15_22_mlp_grad_norm": round(gn_t2_mlp, 4),
                }
            )
            if opt_step in PROBE_OPT_STEPS:
                bwd_st = _measure_warmstart_backward_probe(runner, diag_items, ab_ids)
                snapshots.append({"opt_step": opt_step, "shared_L15_23_grad_norm": round(gn_shared, 4), "total_grad_norm": round(gn_total, 4), **bwd_st})

    res = {
        "n_trainable_total": sum(p.numel() for p in all_trainable),
        "n_shared_L15_23_params": sum(p.numel() for p in shared_L15_23_params),
        "max_total_grad_norm": round(max(x["total_grad_norm"] for x in step_grad_logs), 2),
        "max_shared_L15_23_grad_norm": round(max(x["shared_L15_23_grad_norm"] for x in step_grad_logs), 2),
        "max_L15_22_mlp_grad_norm": round(max(x["L15_22_mlp_grad_norm"] for x in step_grad_logs), 2),
        "max_rms_dL_dh15_via_mlp": round(max(s["rms_dL_dh15_via_mlp"] for s in snapshots), 4),
        "step_grad_logs": step_grad_logs,
        "backward_snapshots": snapshots,
    }
    print(
        f"[R6d {cond_name}] max_total={res['max_total_grad_norm']} | max_shared_L15_23={res['max_shared_L15_23_grad_norm']} | max_dL_dh15={res['max_rms_dL_dh15_via_mlp']}",
        flush=True,
    )
    del runner, base_cpu, optimizer, groups
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


def run_r6d(dev_rows: list[dict[str, Any]]) -> dict[str, Any]:
    tmp_r = QwenDeciderRunner()
    rc_b = [r for r in dev_rows if r["subset"] == "real_cf" and r["label"] == "block"][:8]
    rc_a = [r for r in dev_rows if r["subset"] == "real_cf" and r["label"] == "allow"][:8]
    diag_items: list[tuple[torch.Tensor, float]] = []
    for r in rc_b + rc_a:
        pr = build_plain_prompt(render_view(r, tok=tmp_r.tokenizer, budget=2048, include_recap=False).text, Q_BLOCK_BINARY, False).text
        ids = tmp_r.tokenizer.encode(pr, add_special_tokens=False)
        diag_items.append((torch.tensor([ids], dtype=torch.long), 1.0 if r["label"] == "block" else 0.0))
    del tmp_r
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    specs = [
        ("t3a_insitu_base", 1.5e-4, 5.0e-4, "none", False),
        ("t3b_insitu_full", 1.5e-4, 5.0e-4, "none", True),
        ("t3b_freeze_t2gates", 1.5e-4, 5.0e-4, "freeze_t2gates", True),
        ("t3b_lowlr_5x", 3.0e-5, 1.0e-4, "none", True),
        ("t3b_freeze_L0_14", 1.5e-4, 5.0e-4, "freeze_L0_14", True),
    ]
    out: dict[str, Any] = {"n_diag_items": len(diag_items), "conditions": {}}
    for cname, lr, glr, fmode, init_t2 in specs:
        out["conditions"][cname] = _run_r6d_condition(cname, lr, glr, fmode, init_t2, dev_rows, diag_items)
    return out


@torch.inference_mode()
def _eval_general_corpus_quant_kl(
    runner: QwenDeciderRunner,
    corpus_texts: list[str],
    ref_lps: list[torch.Tensor] | None = None,
) -> tuple[dict[str, Any], list[torch.Tensor]]:
    """Measure full-vocab perplexity and next-token KL/logit MAE across GENERAL_CORPUS (2,026 tokens)."""
    runner.model.eval()
    stored: list[torch.Tensor] = []
    total_nll, total_kl, l1_sum, toks = 0.0, 0.0, 0.0, 0
    for idx, txt in enumerate(corpus_texts):
        ids = torch.tensor([runner.tokenizer.encode(txt, add_special_tokens=False)], dtype=torch.long)
        hid = runner.model.model(input_ids=ids.to(runner.device), use_cache=False).last_hidden_state[:, :-1, :]
        logits = F.linear(hid.to(dtype=runner.model.lm_head.weight.dtype), runner.model.lm_head.weight).float()[0].cpu()
        tgt = ids[0, 1:]
        lp = F.log_softmax(logits, dim=-1)
        total_nll += float(F.nll_loss(lp, tgt, reduction="sum").item())
        toks += int(tgt.numel())
        if ref_lps is None:
            stored.append(lp)
        else:
            rp = ref_lps[idx]
            total_kl += float(torch.sum(rp.exp() * (rp - lp)).item())
            l1_sum += float(torch.abs(rp - lp).mean().item()) * int(tgt.numel())
    ppl = math.exp(total_nll / max(1, toks))
    stats = {
        "tokens": toks,
        "perplexity": round(ppl, 4),
        "mean_kl_vs_bf16": 0.0 if ref_lps is None else round(total_kl / max(1, toks), 6),
        "mean_logprob_l1_vs_bf16": 0.0 if ref_lps is None else round(l1_sum / max(1, toks), 5),
    }
    return stats, stored


def run_r6e(dev_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Run R6e: Multi-edit generality across Track 1 (single-layer L15 SwiGLU) and MIT WSP (down_proj-only)."""
    TMP_GGUF_DIR.mkdir(parents=True, exist_ok=True)
    mit_snap = Path(snapshot_download(MIT_WSP_ID, local_files_only=True))
    base_snap = Path(snapshot_download(BASE_ID, local_files_only=True))

    # Export Track 1 HF dir & convert Track 1 + MIT WSP -> BF16 GGUF -> Q8_0/Q6_K/Q5_K_M/Q4_K_M
    t1_hf = TMP_GGUF_DIR / "track1-hf"
    if not t1_hf.exists():
        export_patched_hf(CHECKPOINTS_DIR / "track1_best_deltas.pt", t1_hf, base_snapshot_dir=base_snap)
    t1_bf16 = TMP_GGUF_DIR / "t1-bf16.gguf"
    if not t1_bf16.exists():
        subprocess.run([sys.executable, str(CONVERT_HF_GGUF), str(t1_hf), "--outfile", str(t1_bf16), "--outtype", "bf16"], check=True)
    mit_bf16 = TMP_GGUF_DIR / "mit-bf16.gguf"
    if not mit_bf16.exists():
        subprocess.run([sys.executable, str(CONVERT_HF_GGUF), str(mit_snap), "--outfile", str(mit_bf16), "--outtype", "bf16"], check=True)

    t1_q = {"bf16": t1_bf16}
    mit_q = {"bf16": mit_bf16}
    for qt in ("Q8_0", "Q6_K", "Q4_K_M"):
        p1 = TMP_GGUF_DIR / f"t1-{qt}.gguf"
        if not p1.exists():
            subprocess.run([str(QUANT_BIN), str(t1_bf16), str(p1), qt], check=True)
        t1_q[qt] = p1
        pm = TMP_GGUF_DIR / f"mit-{qt}.gguf"
        if not pm.exists():
            subprocess.run([str(QUANT_BIN), str(mit_bf16), str(pm), qt], check=True)
        mit_q[qt] = pm

    base_bf16 = TMP_GGUF_DIR / "base-bf16.gguf"
    base_q4 = TMP_GGUF_DIR / "base-Q4_K_M.gguf"
    if not base_bf16.exists():
        subprocess.run([sys.executable, str(CONVERT_HF_GGUF), str(base_snap), "--outfile", str(base_bf16), "--outtype", "bf16"], check=True)
    if not base_q4.exists():
        subprocess.run([str(QUANT_BIN), str(base_bf16), str(base_q4), "Q4_K_M"], check=True)

    runner = QwenDeciderRunner()
    base_cpu = capture_cpu_state_dict(runner.model)
    ab_ids = [runner.letter_token_ids["A"], runner.letter_token_ids["B"]]
    prompt_ids = _select_dev_prompt_ids(runner, dev_rows)
    d_ff = runner.model.model.layers[15].mlp.gate_proj.weight.shape[0]
    t1_unt = torch.ones(d_ff, dtype=torch.bool)
    t1_unt[T1_L15_NEURONS] = False
    t2_l22_unt = torch.ones(d_ff, dtype=torch.bool)
    t2_l22_unt[T2_L22_NEURONS] = False

    # 1. Track 1 Jacobian & Q8/Q6/Q4 + surgical L15 hybrids
    w_t1_bf16 = _dequantize_gguf_2d_weights(t1_bf16)
    _apply_dequant_dict(runner.model, base_cpu, w_t1_bf16)
    ref_t1 = _probe_model_internal(runner, prompt_ids, ab_ids)
    jac_t1_l15 = _compute_swiglu_jacobian_stats(runner, prompt_ids, 15, T1_L15_NEURONS, t1_unt)
    t1_ed_set = {"blk.15.ffn_gate.weight", "blk.15.ffn_up.weight", "blk.15.ffn_down.weight"}
    w_t1_bf16_ed = _dequantize_gguf_2d_weights(t1_bf16, filter_names=t1_ed_set)
    w_t1_q4 = _dequantize_gguf_2d_weights(t1_q["Q4_K_M"])
    w_t1_q4_ed = _dequantize_gguf_2d_weights(t1_q["Q4_K_M"], filter_names=t1_ed_set)

    t1_probes = {}
    for k, wd in [
        ("t1_Q8_0", _dequantize_gguf_2d_weights(t1_q["Q8_0"])),
        ("t1_Q6_K", _dequantize_gguf_2d_weights(t1_q["Q6_K"])),
        ("t1_Q4_K_M", w_t1_q4),
        ("t1_Q4_bb_bf16_L15", {**w_t1_q4, **w_t1_bf16_ed}),
        ("t1_bf16_bb_Q4_L15", {**w_t1_bf16, **w_t1_q4_ed}),
    ]:
        _apply_dequant_dict(runner.model, base_cpu, wd)
        cp = _probe_model_internal(runner, prompt_ids, ab_ids)
        t1_probes[k] = _compare_with_rel_and_signed(cp, ref_t1, t1_unt, t2_l22_unt)

    # 2. MIT WSP (down_proj-only at L15 & L19) Jacobian, select_dev parity, and GENERAL_CORPUS Q4 KL
    w_mit_bf16 = _dequantize_gguf_2d_weights(mit_bf16)
    w_mit_q4 = _dequantize_gguf_2d_weights(mit_q["Q4_K_M"])
    _apply_dequant_dict(runner.model, base_cpu, w_mit_bf16)
    ref_mit = _probe_model_internal(runner, prompt_ids, ab_ids)

    w_base_1519 = _dequantize_gguf_2d_weights(base_bf16, filter_names={"blk.15.ffn_down.weight", "blk.19.ffn_down.weight"})
    mit_diff_l15 = (w_mit_bf16["model.layers.15.mlp.down_proj.weight"].float() - w_base_1519["model.layers.15.mlp.down_proj.weight"].float()).abs()
    mit_cols_15 = (mit_diff_l15.sum(dim=0) > 1e-6).nonzero(as_tuple=True)[0].tolist()
    mit_unt_15 = torch.ones(d_ff, dtype=torch.bool)
    mit_unt_15[mit_cols_15] = False
    jac_mit_l15 = _compute_swiglu_jacobian_stats(runner, prompt_ids, 15, mit_cols_15, mit_unt_15)

    mit_diff_l19 = (w_mit_bf16["model.layers.19.mlp.down_proj.weight"].float() - w_base_1519["model.layers.19.mlp.down_proj.weight"].float()).abs()
    mit_cols_19 = (mit_diff_l19.sum(dim=0) > 1e-6).nonzero(as_tuple=True)[0].tolist()
    mit_unt_19 = torch.ones(d_ff, dtype=torch.bool)
    mit_unt_19[mit_cols_19] = False
    jac_mit_l19 = _compute_swiglu_jacobian_stats(runner, prompt_ids, 19, mit_cols_19, mit_unt_19)

    _apply_dequant_dict(runner.model, base_cpu, w_mit_q4)
    cur_mit_q4 = _probe_model_internal(runner, prompt_ids, ab_ids)
    mit_q4_cmp = _compare_with_rel_and_signed(cur_mit_q4, ref_mit, mit_unt_15, t2_l22_unt)
    dg_mit15 = (cur_mit_q4["g15"][:, mit_cols_15] - ref_mit["g15"][:, mit_cols_15]).abs()
    act_mit15 = ref_mit["g15"][:, mit_cols_15].abs()
    # Note: MIT WSP also modifies L19 down_proj; gate_proj/up_proj at L19 are Base-identical
    _apply_dequant_dict(runner.model, base_cpu, w_mit_bf16)
    g19_ref_buf: list[torch.Tensor] = []
    h_19_ref = runner.model.model.layers[19].mlp.down_proj.register_forward_pre_hook(lambda _m, inp: g19_ref_buf.append(inp[0][0, -1, mit_cols_19].detach().float().cpu()))
    with torch.inference_mode():
        for ids in prompt_ids:
            runner.model.model(input_ids=ids.to(runner.device), use_cache=False)
    h_19_ref.remove()
    _apply_dequant_dict(runner.model, base_cpu, w_mit_q4)
    g19_q_buf: list[torch.Tensor] = []
    h_19_q = runner.model.model.layers[19].mlp.down_proj.register_forward_pre_hook(lambda _m, inp: g19_q_buf.append(inp[0][0, -1, mit_cols_19].detach().float().cpu()))
    with torch.inference_mode():
        for ids in prompt_ids:
            runner.model.model(input_ids=ids.to(runner.device), use_cache=False)
    h_19_q.remove()
    g19_ref_t = torch.stack(g19_ref_buf, dim=0)
    g19_q_t = torch.stack(g19_q_buf, dim=0)
    dg_mit19 = (g19_q_t - g19_ref_t).abs()
    act_mit19 = g19_ref_t.abs()

    corpus_texts = [t for _, t in GENERAL_CORPUS]
    _apply_dequant_dict(runner.model, base_cpu, _dequantize_gguf_2d_weights(base_bf16))
    base_lm_bf16, base_lps = _eval_general_corpus_quant_kl(runner, corpus_texts, None)
    _apply_dequant_dict(runner.model, base_cpu, _dequantize_gguf_2d_weights(base_q4))
    base_lm_q4, _ = _eval_general_corpus_quant_kl(runner, corpus_texts, base_lps)

    _apply_dequant_dict(runner.model, base_cpu, w_mit_bf16)
    mit_lm_bf16, mit_lps = _eval_general_corpus_quant_kl(runner, corpus_texts, None)
    _apply_dequant_dict(runner.model, base_cpu, w_mit_q4)
    mit_lm_q4, _ = _eval_general_corpus_quant_kl(runner, corpus_texts, mit_lps)

    def _strip(d: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in d.items() if k not in ("H", "J_ed")}

    del runner, base_cpu, base_lps, mit_lps
    gc.collect()
    return {
        "track1_L15_swiglu": {
            "edited_neurons": T1_L15_NEURONS,
            "jacobian_l15": _strip(jac_t1_l15),
            "dequant_probes": t1_probes,
        },
        "mit_wsp_down_proj_only": {
            "n_modified_l15_cols": len(mit_cols_15),
            "n_modified_l19_cols": len(mit_cols_19),
            "jacobian_l15_edited64": _strip(jac_mit_l15),
            "jacobian_l19_edited64": _strip(jac_mit_l19),
            "l15_edited64_mean_abs_dg_q4": round(float(dg_mit15.mean().item()), 5),
            "l15_edited64_rel_shift_dg_over_g_q4": round(float(dg_mit15.mean().item() / max(1e-8, float(act_mit15.mean().item()))), 4),
            "l19_edited64_mean_abs_dg_q4": round(float(dg_mit19.mean().item()), 5),
            "l19_edited64_rel_shift_dg_over_g_q4": round(float(dg_mit19.mean().item() / max(1e-8, float(act_mit19.mean().item()))), 4),
            "select_dev_q4": mit_q4_cmp,
            "general_corpus_quant_parity": {
                "base_bf16": base_lm_bf16,
                "base_Q4_K_M_vs_bf16": base_lm_q4,
                "mit_wsp_bf16": mit_lm_bf16,
                "mit_wsp_Q4_K_M_vs_bf16": mit_lm_q4,
            },
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Revision Suite R6 (Theory, Q6_K, Bit-Matched, Shared-Grad, Generality).")
    ap.add_argument("--port", type=int, default=8096)
    ap.add_argument("--only", type=str, default="all", help="Comma-separated stages: r6abc,r6d,r6e or 'all'")
    args = ap.parse_args()

    stages = {"r6abc", "r6d", "r6e"} if args.only == "all" else {s.strip() for s in args.only.split(",") if s.strip()}
    t0 = time.perf_counter()
    dev_rows = load_split_rows("select_dev")
    try:
        if "r6abc" in stages:
            refs = {
                "base": collect_pt_select_dev_ref(None, dev_rows),
                "t2": collect_pt_select_dev_ref(CHECKPOINTS_DIR / "track2_best_deltas.pt", dev_rows),
                "t3": collect_pt_select_dev_ref(CHECKPOINTS_DIR / "track3_final_best_deltas.pt", dev_rows),
            }
            runner = QwenDeciderRunner()
            base_cpu = capture_cpu_state_dict(runner.model)

            r6a, r6b, r6c = run_r6a_b_c(runner, base_cpu, dev_rows, refs, args.port)
            del runner, base_cpu
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            payload = _load_controls_json()
            payload["r6a_kquant_q6_q5_investigation"] = r6a
            payload["r6b_jacobian_and_margin_decomp"] = r6b
            payload["r6c_bit_matched_controls"] = r6c
            _save_controls_json(payload)
            print(f"[R6a/b/c] Saved -> {CONTROLS_JSON}", flush=True)

        if "r6d" in stages:
            r6d = run_r6d(dev_rows)
            payload = _load_controls_json()
            payload["r6d_warmstart_fair_grad_and_backprop"] = r6d
            _save_controls_json(payload)
            print(f"[R6d] Saved -> {CONTROLS_JSON}", flush=True)

        if "r6e" in stages:
            r6e = run_r6e(dev_rows)
            payload = _load_controls_json()
            payload["r6e_multi_edit_generality"] = r6e
            _save_controls_json(payload)
            print(f"[R6e] Saved -> {CONTROLS_JSON}", flush=True)

        print(f"[R6] Stages {sorted(stages)} complete in {time.perf_counter() - t0:.1f}s -> {CONTROLS_JSON}", flush=True)
    finally:
        if TMP_GGUF_DIR.exists():
            shutil.rmtree(TMP_GGUF_DIR, ignore_errors=True)
            print(f"[R6] Cleaned temp GGUF directory {TMP_GGUF_DIR}", flush=True)


if __name__ == "__main__":
    main()
