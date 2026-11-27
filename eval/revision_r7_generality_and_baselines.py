"""Revision Suite R7: P2 Controlled Same-Neuron Ablation, Dose-Response, Real HAWQ/AWQ Baselines,
Leakage-Free Calibration, Asymmetric Risk, Non-Pooled Taylor Controls, 3-Seed + L9-23/Clip Warm-Start,
and Cross-Architecture Replication on Pure-Attention SmolLM2-360M (LlamaForCausalLM).

Merges incrementally into `experiments/paper_revision_controls.json` under:
  - `r7a_p2_ablation_doseresponse_and_cancellation`
  - `r7b_hawq_awq_sensitivity_baselines`
  - `r7c_calibration_asymmetric_risk_and_taylor_controls`
  - `r7d_warmstart_controls_3seeds_and_receptive_field`
  - `r7e_second_arch_smollm2_360m`
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import shutil
import subprocess
import sys
import time
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
from safetensors.torch import load_file, save_file

from automode.questions import Q_BLOCK_BINARY, build_plain_prompt
from automode.view import render_view
from eval.evaluate import compute_binary_metrics, evaluate_runner, load_split_rows
from eval.mechanism_analysis import (
    BASE_ID,
    GENERAL_CORPUS,
    MIT_WSP_ID,
    QUANT_BIN,
    QUANT_TYPES,
    T2_L15_NEURONS,
    T2_L22_NEURONS,
    _wait_server,
)
from eval.model_runner import QwenDeciderRunner, capture_cpu_state_dict, restore_cpu_state_dict
from eval.revision_quant_mitigations import (
    CONVERT_HF_GGUF,
    LAUNCH_SCRIPT,
    TMP_GGUF_DIR,
    TRACK2_GGUF_TENSORS,
    _load_controls_json,
    _save_controls_json,
)
from eval.revision_r5_amplification_and_cis import (
    BOOTSTRAP_B,
    BOOTSTRAP_SEED,
    FULL_ATTN_LAYERS,
    _apply_dequant_dict,
    _bootstrap_distribution,
    _dequantize_gguf_2d_weights,
    _paired_diff_summary,
    _probe_model_internal,
    collect_pt_select_dev_ref,
    eval_gguf_select_dev_vectors,
)
from eval.revision_r6_theory_and_controls import (
    DIAG_DEV_COUNT,
    _blk_num,
    _build_surgical_gguf,
    _compare_with_rel_and_signed,
    _compute_swiglu_jacobian_stats,
    _eval_general_corpus_quant_kl,
    _group_grad_norm,
    _select_dev_prompt_ids,
)
from eval.revision_warmstart_controls import MAX_GROUPS, _prepare_probe_prompts
from runtime.client import LlamaCompletionClient, _extract_token_logprobs, score_options_from_logprobs, sigmoid
from runtime.export_hf import export_patched_hf
from scripts.quantize_upstream_q8 import plan_upstream_overrides
from train.lora_model import inject_lora, lora_param_groups
from train.train_track3 import (
    DEFAULT_TRAIN_FILE,
    _cosine_lr_multiplier,
    prepare_training_groups,
    train_group_step,
)
from weight_edit.causal_check import load_patch_bundle
from weight_edit.context_edit import measure_layer_logit_gains
from weight_edit.wsp_runner import (
    _select_unused_quiet_neurons,
    build_residual_swiglu_patch,
    collect_post_patch_observe_state,
)

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = ROOT / "checkpoints"

SMOLLM2_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
SMOLLM2_LAYERS = (20, 28)  # 62.5% and 87.5% depth of 32-layer LlamaForCausalLM
LLAMA32_ID = "unsloth/Llama-3.2-1B-Instruct"
LLAMA32_LAYERS = (10, 14)  # 62.5% and 87.5% depth of 16-layer LlamaForCausalLM (all dims 256-aligned)


def _load_train_rows() -> list[dict[str, Any]]:
    with open(DEFAULT_TRAIN_FILE, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _clean_tmp_ggufs() -> None:
    if TMP_GGUF_DIR.exists():
        shutil.rmtree(TMP_GGUF_DIR)


def _export_custom_qwen_deltas_to_q4(
    net_deltas: dict[str, torch.Tensor],
    base_snap: Path,
    stem: str,
    build_q4: bool = True,
) -> tuple[Path, Path | None]:
    """Export arbitrary causal-named net deltas over base Qwen3.5-0.8B -> BF16 GGUF (and optional Q4_K_M GGUF)."""
    TMP_GGUF_DIR.mkdir(parents=True, exist_ok=True)
    pt_path = TMP_GGUF_DIR / f"{stem}_deltas.pt"
    hf_dir = TMP_GGUF_DIR / f"{stem}_hf"
    bf16_gguf = TMP_GGUF_DIR / f"{stem}-bf16.gguf"
    q4_gguf = TMP_GGUF_DIR / f"{stem}-Q4_K_M.gguf"

    payload_patch = {
        "step_id": stem,
        "category": "r7_custom",
        "method": "r7_custom_delta",
        "target_layers": [15, 22],
        "rank": 4,
        "alpha": 1.0,
        "hypothesis": stem,
        "metadata": {},
        "deltas": {k: v.detach().float().cpu() for k, v in net_deltas.items()},
    }
    torch.save({"metadata": {}, "patches": [payload_patch]}, pt_path)
    export_patched_hf(pt_path, hf_dir, base_snapshot_dir=base_snap)
    pt_path.unlink(missing_ok=True)

    subprocess.run(
        [sys.executable, str(CONVERT_HF_GGUF), str(hf_dir), "--outfile", str(bf16_gguf), "--outtype", "bf16"],
        check=True,
    )
    shutil.rmtree(hf_dir, ignore_errors=True)
    if build_q4:
        subprocess.run([str(QUANT_BIN), str(bf16_gguf), str(q4_gguf), "Q4_K_M"], check=True)
        return bf16_gguf, q4_gguf
    return bf16_gguf, None


def _compute_common_edit_agnostic_jacobians(
    runner: QwenDeciderRunner,
    prompt_ids: list[torch.Tensor],
    layers: tuple[int, ...],
    margin_layer: int,
) -> dict[str, float]:
    """Compute exact full MLP input Jacobian norms ||dMLP/dh||_{F,2} and end-to-end ||dm/dh||_2."""
    runner.model.eval()
    sub_ids = prompt_ids[:32]
    h_mlp: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    handles = [
        runner.model.model.layers[l].mlp.register_forward_pre_hook(
            lambda _m, inp, li=l: h_mlp[li].append(inp[0][0, -1, :].detach().float().cpu())
        )
        for l in layers
    ]
    try:
        with torch.inference_mode():
            for ids in sub_ids:
                runner.model.model(input_ids=ids.to(runner.device), use_cache=False)
    finally:
        for h in handles:
            h.remove()

    out: dict[str, float] = {}
    for l in layers:
        H = torch.stack(h_mlp[l], dim=0).to(runner.device)  # (N, D)
        mlp = runner.model.model.layers[l].mlp
        Wg = mlp.gate_proj.weight.detach().float()
        Wu = mlp.up_proj.weight.detach().float()
        Wd = mlp.down_proj.weight.detach().float()
        fro_vals: list[float] = []
        spec_vals: list[float] = []
        with torch.inference_mode():
            for i in range(H.shape[0]):
                h = H[i]
                a = Wg @ h
                b = Wu @ h
                sig = torch.sigmoid(a)
                silu = a * sig
                silu_p = sig * (1.0 + a * (1.0 - sig))
                inner = (silu_p * b).unsqueeze(1) * Wg + silu.unsqueeze(1) * Wu
                J = Wd @ inner
                sv = torch.linalg.svdvals(J)
                fro_vals.append(float(torch.sqrt((sv * sv).sum()).item()))
                spec_vals.append(float(sv[0].item()))
                del inner, J, sv
        out[f"l{l}_mlp_jac_fro_mean"] = round(float(np.mean(fro_vals)), 4)
        out[f"l{l}_mlp_jac_spec_mean"] = round(float(np.mean(spec_vals)), 4)
        del H, Wg, Wu, Wd
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # End-to-end margin Jacobian ||dm/dh_{margin_layer}||_2 by detaching hidden at block input.
    # Freeze all parameters + activate gradient checkpointing (train() mode, dropout=0.0) so VRAM < 1.8 GiB.
    prev_req = [p.requires_grad for p in runner.model.parameters()]
    for p in runner.model.parameters():
        p.requires_grad_(False)
    runner.model.train()
    runner.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    ab_ids = [runner.letter_token_ids["A"], runner.letter_token_ids["B"]]
    lm_w = runner.model.lm_head.weight[ab_ids]
    lm_b = runner.model.lm_head.bias[ab_ids] if getattr(runner.model.lm_head, "bias", None) is not None else None
    margin_j_norms: list[float] = []
    leaf_box: list[torch.Tensor] = []

    def _blk_pre_hook(_m, inp):
        h_leaf = inp[0].detach().requires_grad_(True)
        leaf_box.clear()
        leaf_box.append(h_leaf)
        return (h_leaf,) + inp[1:]

    pre_h = runner.model.model.layers[margin_layer].register_forward_pre_hook(_blk_pre_hook)
    try:
        for ids in sub_ids:
            runner.model.zero_grad(set_to_none=True)
            hid = runner.model.model(input_ids=ids.to(runner.device), use_cache=False).last_hidden_state[:, -1:, :]
            lg = F.linear(hid.to(dtype=lm_w.dtype), lm_w, lm_b)[0, 0, :].float()
            margin = lg[1] - lg[0]
            g_h = torch.autograd.grad(margin, leaf_box[0], retain_graph=False)[0]
            margin_j_norms.append(float(torch.linalg.norm(g_h[0, -1, :].detach().float()).item()))
            leaf_box.clear()
            del hid, lg, margin, g_h
    finally:
        pre_h.remove()
        runner.model.zero_grad(set_to_none=True)
        runner.model.gradient_checkpointing_disable()
        runner.model.eval()
        for p, rg in zip(runner.model.parameters(), prev_req):
            p.requires_grad_(rg)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out[f"margin_jac_h{margin_layer}_l2_mean"] = round(float(np.mean(margin_j_norms)), 4)
    return out


def _eval_binary_accuracy_140(margins: np.ndarray, dev_rows: list[dict[str, Any]]) -> dict[str, float]:
    bin_rows = [r for sub in ("real_cf", "consent_5p", "long_50t") for r in dev_rows if r["subset"] == sub]
    ys = np.array([1 if r["label"] == "block" else 0 for r in bin_rows], dtype=np.int32)
    probs = 1.0 / (1.0 + np.exp(-np.clip(margins, -30.0, 30.0)))
    rc_idx = [i for i, r in enumerate(bin_rows) if r["subset"] == "real_cf"]
    c5_idx = [i for i, r in enumerate(bin_rows) if r["subset"] == "consent_5p"]
    b_rc = compute_binary_metrics(probs[rc_idx], ys[rc_idx], margins[rc_idx])
    b_c5 = compute_binary_metrics(probs[c5_idx], ys[c5_idx], margins[c5_idx])
    return {
        "mean_abs_margin": round(float(np.mean(np.abs(margins))), 4),
        "rc_auroc": round(float(b_rc["auroc"]), 4),
        "rc_rec": round(float(b_rc["recall_at_05"]), 4),
        "rc_fpr": round(float(b_rc["fpr_at_05"]), 4),
        "c5p_acc": round(float(b_c5["acc_at_05"]), 4),
    }


def _weight_delta_stats(deltas: dict[str, torch.Tensor]) -> dict[str, Any]:
    n_mod = sum(int((d.abs() > 1e-7).sum().item()) for d in deltas.values())
    sq = sum(float((d.float() ** 2).sum().item()) for d in deltas.values())
    frob = math.sqrt(sq)
    rms = frob / math.sqrt(max(1, n_mod))
    return {"modified_elements": n_mod, "frobenius_norm": round(frob, 4), "rms": round(rms, 6)}


def _probe_subset(
    runner: QwenDeciderRunner,
    prompt_ids: list[torch.Tensor],
    ab_ids: list[int],
    layer_indices: tuple[int, int] = (15, 22),
) -> dict[str, Any]:
    """Generic internal activation & margin probe supporting arbitrary layer indices (e.g. Qwen 15/22 or SmolLM2 20/28)."""
    l1, l2 = layer_indices
    runner.model.eval()
    h1_l, g1_l, m1_l = [], [], []
    h2_l, g2_l, m2_l = [], [], []
    margins: list[float] = []
    handles = [
        runner.model.model.layers[l1].mlp.register_forward_pre_hook(lambda _m, inp: h1_l.append(inp[0][0, -1, :].detach().float().cpu())),
        runner.model.model.layers[l1].mlp.down_proj.register_forward_pre_hook(lambda _m, inp: g1_l.append(inp[0][0, -1, :].detach().float().cpu())),
        runner.model.model.layers[l1].mlp.register_forward_hook(lambda _m, _inp, out: m1_l.append(out[0, -1, :].detach().float().cpu())),
        runner.model.model.layers[l2].mlp.register_forward_pre_hook(lambda _m, inp: h2_l.append(inp[0][0, -1, :].detach().float().cpu())),
        runner.model.model.layers[l2].mlp.down_proj.register_forward_pre_hook(lambda _m, inp: g2_l.append(inp[0][0, -1, :].detach().float().cpu())),
        runner.model.model.layers[l2].mlp.register_forward_hook(lambda _m, _inp, out: m2_l.append(out[0, -1, :].detach().float().cpu())),
    ]
    lm_w = runner.model.lm_head.weight[ab_ids]
    lm_b = runner.model.lm_head.bias[ab_ids] if getattr(runner.model.lm_head, "bias", None) is not None else None
    try:
        with torch.inference_mode():
            for ids in prompt_ids:
                hid = runner.model.model(input_ids=ids.to(runner.device), use_cache=False).last_hidden_state[:, -1:, :]
                lg = F.linear(hid.to(dtype=lm_w.dtype), lm_w, lm_b)[0, 0, :].float()
                margins.append(float((lg[1] - lg[0]).item()))
    finally:
        for h in handles:
            h.remove()
    return {
        "h1": torch.stack(h1_l, dim=0),
        "g1": torch.stack(g1_l, dim=0),
        "m1": torch.stack(m1_l, dim=0),
        "h2": torch.stack(h2_l, dim=0),
        "g2": torch.stack(g2_l, dim=0),
        "m2": torch.stack(m2_l, dim=0),
        "margins": np.array(margins, dtype=np.float64),
    }


def _summarize_q_vs_ref(
    q_probe: dict[str, Any],
    ref_probe: dict[str, Any],
    ed_l1: list[int],
    ed_l2: list[int],
) -> dict[str, Any]:
    rm, cm = ref_probe["margins"], q_probe["margins"]
    rp = 1.0 / (1.0 + np.exp(-np.clip(rm, -30.0, 30.0)))
    cp = 1.0 / (1.0 + np.exp(-np.clip(cm, -30.0, 30.0)))
    agree_140 = float(np.mean((cm >= 0.0) == (rm >= 0.0)))
    r_p = float(np.corrcoef(rp, cp)[0, 1]) if np.std(rp) > 1e-12 and np.std(cp) > 1e-12 else 0.0
    r_m = float(np.corrcoef(rm, cm)[0, 1]) if np.std(rm) > 1e-12 and np.std(cm) > 1e-12 else 0.0

    rg1, cg1 = ref_probe["g1"][:, ed_l1], q_probe["g1"][:, ed_l1]
    rg2, cg2 = ref_probe["g2"][:, ed_l2], q_probe["g2"][:, ed_l2]
    dg1 = float((cg1 - rg1).abs().mean().item())
    dg2 = float((cg2 - rg2).abs().mean().item())
    g1_base = float(rg1.abs().mean().item())
    g2_base = float(rg2.abs().mean().item())

    return {
        "bf16_mean_abs_m": round(float(np.mean(np.abs(rm))), 4),
        "raw_margin_mae": round(float(np.mean(np.abs(cm - rm))), 4),
        "signed_margin_bias": round(float(np.mean(cm - rm)), 4),
        "pearson_r_p": round(r_p, 4),
        "pearson_r_m": round(r_m, 4),
        "agree_140": round(agree_140, 4),
        "l1_ed4_mean_abs_dg": round(dg1, 4),
        "l1_ed4_rel_dg": round(dg1 / max(1e-8, g1_base), 4),
        "l2_ed4_mean_abs_dg": round(dg2, 4),
        "l2_ed4_rel_dg": round(dg2 / max(1e-8, g2_base), 4),
    }


def run_r7a(dev_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Same-neuron P2 component ablation + dose-response + opposite-sign cancellation forensics."""
    base_snap = Path(snapshot_download(BASE_ID, local_files_only=True))
    mit_snap = Path(snapshot_download(MIT_WSP_ID, local_files_only=True))

    runner = QwenDeciderRunner()
    for p in runner.model.parameters():
        p.requires_grad_(False)
    base_cpu = capture_cpu_state_dict(runner.model)
    ab_ids = [runner.letter_token_ids["A"], runner.letter_token_ids["B"]]
    prompt_ids = _select_dev_prompt_ids(runner, dev_rows)

    d_ff = int(runner.model.model.layers[15].mlp.gate_proj.weight.shape[0])
    unt_15 = torch.ones(d_ff, dtype=torch.bool)
    unt_15[T2_L15_NEURONS] = False
    unt_22 = torch.ones(d_ff, dtype=torch.bool)
    unt_22[T2_L22_NEURONS] = False

    # Reconstruct net Track 2 deltas (summed across steps)
    t2_bundle = load_patch_bundle(CHECKPOINTS_DIR / "track2_best_deltas.pt")
    full_t2_deltas: dict[str, torch.Tensor] = {}
    for p in t2_bundle.patches:
        for k, d_t in p.net_deltas().items():
            full_t2_deltas[k] = full_t2_deltas[k] + d_t.float().cpu() if k in full_t2_deltas else d_t.float().cpu().clone()

    gate_up_deltas = {k: v.clone() for k, v in full_t2_deltas.items() if ("gate_proj" in k or "up_proj" in k)}
    down_deltas = {k: v.clone() for k, v in full_t2_deltas.items() if "down_proj" in k}

    full_stats = _weight_delta_stats(full_t2_deltas)
    down_stats = _weight_delta_stats(down_deltas)
    scale_rms = full_stats["rms"] / max(1e-12, (down_stats["frobenius_norm"] / math.sqrt(down_stats["modified_elements"])))
    scale_frob = full_stats["frobenius_norm"] / max(1e-12, down_stats["frobenius_norm"])
    down_rms_deltas = {k: v * scale_rms for k, v in down_deltas.items()}
    down_frob_deltas = {k: v * scale_frob for k, v in down_deltas.items()}

    # 1. Same-8-neuron component ablation
    variants: list[tuple[str, dict[str, torch.Tensor] | None, Path | None]] = [
        ("base", None, None),
        ("mit_wsp", None, mit_snap),
        ("full_t2", full_t2_deltas, None),
        ("gate_up_only", gate_up_deltas, None),
        ("down_only_unscaled", down_deltas, None),
        ("down_only_rms_matched", down_rms_deltas, None),
        ("down_only_frob_matched", down_frob_deltas, None),
    ]

    TMP_GGUF_DIR.mkdir(parents=True, exist_ok=True)
    ablation_out: dict[str, Any] = {}
    for vname, vdeltas, snap_override in variants:
        print(f"[R7a Ablation] Building & probing {vname}...", flush=True)
        if vname == "base":
            bf16_path = TMP_GGUF_DIR / "r7a_base-bf16.gguf"
            q4_path = TMP_GGUF_DIR / "r7a_base-Q4_K_M.gguf"
            if not bf16_path.exists():
                subprocess.run([sys.executable, str(CONVERT_HF_GGUF), str(base_snap), "--outfile", str(bf16_path), "--outtype", "bf16"], check=True)
            if not q4_path.exists():
                subprocess.run([str(QUANT_BIN), str(bf16_path), str(q4_path), "Q4_K_M"], check=True)
            w_stats = {"modified_elements": 0, "frobenius_norm": 0.0, "rms": 0.0}
        elif vname == "mit_wsp":
            bf16_path = TMP_GGUF_DIR / "r7a_mit-bf16.gguf"
            q4_path = TMP_GGUF_DIR / "r7a_mit-Q4_K_M.gguf"
            if not bf16_path.exists():
                subprocess.run([sys.executable, str(CONVERT_HF_GGUF), str(snap_override), "--outfile", str(bf16_path), "--outtype", "bf16"], check=True)
            if not q4_path.exists():
                subprocess.run([str(QUANT_BIN), str(bf16_path), str(q4_path), "Q4_K_M"], check=True)
            w_stats = {"modified_elements": 130188, "frobenius_norm": 2.24, "rms": 0.006207}
        else:
            bf16_path, q4_path = _export_custom_qwen_deltas_to_q4(vdeltas, base_snap, f"r7a_{vname}", build_q4=True)
            w_stats = _weight_delta_stats(vdeltas)

        w_bf16 = _dequantize_gguf_2d_weights(bf16_path)
        w_q4 = _dequantize_gguf_2d_weights(q4_path)

        corpus_texts = [t for _, t in GENERAL_CORPUS]
        _apply_dequant_dict(runner.model, base_cpu, w_bf16)
        ref_probe = _probe_subset(runner, prompt_ids, ab_ids, (15, 22))
        j15 = _compute_swiglu_jacobian_stats(runner, prompt_ids, 15, T2_L15_NEURONS, unt_15)
        j22 = _compute_swiglu_jacobian_stats(runner, prompt_ids, 22, T2_L22_NEURONS, unt_22)
        common_jac = _compute_common_edit_agnostic_jacobians(runner, prompt_ids, (15, 22), 15)
        corp_bf16, ref_lps = _eval_general_corpus_quant_kl(runner, corpus_texts, None)
        bf_perf = _eval_binary_accuracy_140(ref_probe["margins"], dev_rows)

        _apply_dequant_dict(runner.model, base_cpu, w_q4)
        q4_probe = _probe_subset(runner, prompt_ids, ab_ids, (15, 22))
        q4_sum = _summarize_q_vs_ref(q4_probe, ref_probe, T2_L15_NEURONS, T2_L22_NEURONS)
        corp_q4, _ = _eval_general_corpus_quant_kl(runner, corpus_texts, ref_lps)

        ablation_out[vname] = {
            **w_stats,
            "swiglu_jac_l15_mean": j15["edited_jac_norm_mean"],
            "swiglu_jac_l22_mean": j22["edited_jac_norm_mean"],
            **common_jac,
            "bf16_perf": bf_perf,
            "q4_k_m_parity": q4_sum,
            "general_corpus_bf16_ppl": corp_bf16["perplexity"],
            "general_corpus_q4_vs_bf16": corp_q4,
        }
        del w_bf16, w_q4, ref_lps
        if vname not in ("base", "mit_wsp"):
            bf16_path.unlink(missing_ok=True)
            q4_path.unlink(missing_ok=True)

    # 2. Edit-norm dose-response curve (alpha in [0.0..1.25])
    dose_out: list[dict[str, Any]] = []
    for alpha in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25):
        print(f"[R7a DoseResponse] alpha={alpha}...", flush=True)
        if alpha == 0.0:
            row = {
                "alpha": 0.0,
                **{k: ablation_out["base"][k] for k in ("modified_elements", "frobenius_norm", "rms", "swiglu_jac_l15_mean", "swiglu_jac_l22_mean", "l15_mlp_jac_fro_mean", "l15_mlp_jac_spec_mean", "margin_jac_h15_l2_mean", "bf16_perf", "q4_k_m_parity")},
            }
            dose_out.append(row)
            continue
        if abs(alpha - 1.0) < 1e-8:
            row = {
                "alpha": 1.0,
                **{k: ablation_out["full_t2"][k] for k in ("modified_elements", "frobenius_norm", "rms", "swiglu_jac_l15_mean", "swiglu_jac_l22_mean", "l15_mlp_jac_fro_mean", "l15_mlp_jac_spec_mean", "margin_jac_h15_l2_mean", "bf16_perf", "q4_k_m_parity")},
            }
            dose_out.append(row)
            continue

        scaled_d = {k: v * alpha for k, v in full_t2_deltas.items()}
        bf16_p, q4_p = _export_custom_qwen_deltas_to_q4(scaled_d, base_snap, f"r7a_alpha_{alpha:.2f}".replace(".", "_"), build_q4=True)
        w_bf = _dequantize_gguf_2d_weights(bf16_p)
        w_q4 = _dequantize_gguf_2d_weights(q4_p)
        _apply_dequant_dict(runner.model, base_cpu, w_bf)
        rp = _probe_subset(runner, prompt_ids, ab_ids, (15, 22))
        j15 = _compute_swiglu_jacobian_stats(runner, prompt_ids, 15, T2_L15_NEURONS, unt_15)
        j22 = _compute_swiglu_jacobian_stats(runner, prompt_ids, 22, T2_L22_NEURONS, unt_22)
        cjac = _compute_common_edit_agnostic_jacobians(runner, prompt_ids, (15, 22), 15)
        bf_perf = _eval_binary_accuracy_140(rp["margins"], dev_rows)

        _apply_dequant_dict(runner.model, base_cpu, w_q4)
        qp = _probe_subset(runner, prompt_ids, ab_ids, (15, 22))
        q4s = _summarize_q_vs_ref(qp, rp, T2_L15_NEURONS, T2_L22_NEURONS)
        dose_out.append(
            {
                "alpha": alpha,
                **_weight_delta_stats(scaled_d),
                "swiglu_jac_l15_mean": j15["edited_jac_norm_mean"],
                "swiglu_jac_l22_mean": j22["edited_jac_norm_mean"],
                **cjac,
                "bf16_perf": bf_perf,
                "q4_k_m_parity": q4s,
            }
        )
        bf16_p.unlink(missing_ok=True)
        q4_p.unlink(missing_ok=True)

    # 3. Opposite-sign cancellation forensics (why Q4_bb + BF16_edits < vanilla Q4_K_M slightly in r)
    t2_bf16 = CHECKPOINTS_DIR / "wsp-automode-0.8b-bf16.gguf"
    t2_q4 = CHECKPOINTS_DIR / "wsp-automode-0.8b-Q4_K_M.gguf"
    t2_set = set(TRACK2_GGUF_TENSORS)
    w_t2_bf16 = _dequantize_gguf_2d_weights(t2_bf16)
    w_t2_bf16_ed = _dequantize_gguf_2d_weights(t2_bf16, filter_names=t2_set)
    w_t2_q4 = _dequantize_gguf_2d_weights(t2_q4)
    w_t2_q4_ed = _dequantize_gguf_2d_weights(t2_q4, filter_names=t2_set)

    _apply_dequant_dict(runner.model, base_cpu, w_t2_bf16)
    p_bf = _probe_subset(runner, prompt_ids, ab_ids, (15, 22))
    _apply_dequant_dict(runner.model, base_cpu, {**w_t2_bf16, **w_t2_q4_ed})
    p_wt = _probe_subset(runner, prompt_ids, ab_ids, (15, 22))
    _apply_dequant_dict(runner.model, base_cpu, {**w_t2_q4, **w_t2_bf16_ed})
    p_up = _probe_subset(runner, prompt_ids, ab_ids, (15, 22))
    _apply_dequant_dict(runner.model, base_cpu, w_t2_q4)
    p_bo = _probe_subset(runner, prompt_ids, ab_ids, (15, 22))

    dm_wt = p_wt["margins"] - p_bf["margins"]
    dm_up = p_up["margins"] - p_bf["margins"]
    dm_bo = p_bo["margins"] - p_bf["margins"]

    dg15_wt = (p_wt["g1"][:, T2_L15_NEURONS] - p_bf["g1"][:, T2_L15_NEURONS]).reshape(-1).numpy()
    dg15_up = (p_up["g1"][:, T2_L15_NEURONS] - p_bf["g1"][:, T2_L15_NEURONS]).reshape(-1).numpy()
    dg22_wt = (p_wt["g2"][:, T2_L22_NEURONS] - p_bf["g2"][:, T2_L22_NEURONS]).reshape(-1).numpy()
    dg22_up = (p_up["g2"][:, T2_L22_NEURONS] - p_bf["g2"][:, T2_L22_NEURONS]).reshape(-1).numpy()

    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        return round(float(np.dot(a, b) / max(1e-12, np.linalg.norm(a) * np.linalg.norm(b))), 4)

    cancel_out = {
        "n_prompts": len(dm_bo),
        "mean_dm_weight_only": round(float(np.mean(dm_wt)), 4),
        "mean_abs_dm_weight_only": round(float(np.mean(np.abs(dm_wt))), 4),
        "mean_dm_upstream_only": round(float(np.mean(dm_up)), 4),
        "mean_abs_dm_upstream_only": round(float(np.mean(np.abs(dm_up))), 4),
        "mean_dm_both": round(float(np.mean(dm_bo)), 4),
        "mean_abs_dm_both": round(float(np.mean(np.abs(dm_bo))), 4),
        "mean_abs_linear_sum_up_plus_wt": round(float(np.mean(np.abs(dm_up + dm_wt))), 4),
        "frac_prompts_opposite_sign_dm": round(float(np.mean((dm_wt * dm_up) < 0.0)), 4),
        "cos_dm_wt_vs_up": _cos(dm_wt, dm_up),
        "pearson_r_dm_wt_vs_up": round(float(np.corrcoef(dm_wt, dm_up)[0, 1]), 4),
        "cos_dg15_wt_vs_up": _cos(dg15_wt, dg15_up),
        "cos_dg22_wt_vs_up": _cos(dg22_wt, dg22_up),
    }

    restore_cpu_state_dict(runner.model, base_cpu)
    del runner, base_cpu
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "component_ablation": ablation_out,
        "dose_response": dose_out,
        "sign_cancellation_forensics": cancel_out,
    }


def _hf_key_to_gguf_qwen35(hf_name: str, bid: int) -> tuple[str, int | None, int | None]:
    """Map HF param name in layer `bid` -> (gguf_tensor_name, slice_start, slice_end)."""
    if hf_name == f"model.layers.{bid}.mlp.gate_proj.weight":
        return f"blk.{bid}.ffn_gate.weight", None, None
    if hf_name == f"model.layers.{bid}.mlp.up_proj.weight":
        return f"blk.{bid}.ffn_up.weight", None, None
    if hf_name == f"model.layers.{bid}.mlp.down_proj.weight":
        return f"blk.{bid}.ffn_down.weight", None, None
    if bid in FULL_ATTN_LAYERS:
        if hf_name == f"model.layers.{bid}.self_attn.q_proj.weight":
            return f"blk.{bid}.attn_q.weight", None, None
        if hf_name == f"model.layers.{bid}.self_attn.k_proj.weight":
            return f"blk.{bid}.attn_k.weight", None, None
        if hf_name == f"model.layers.{bid}.self_attn.v_proj.weight":
            return f"blk.{bid}.attn_v.weight", None, None
        if hf_name == f"model.layers.{bid}.self_attn.o_proj.weight":
            return f"blk.{bid}.attn_output.weight", None, None
    else:
        if hf_name == f"model.layers.{bid}.linear_attn.in_proj_qkv.weight":
            return f"blk.{bid}.attn_qkv_gate.weight", 0, 6144
        if hf_name == f"model.layers.{bid}.linear_attn.in_proj_z.weight":
            return f"blk.{bid}.attn_qkv_gate.weight", 6144, 8192
        if hf_name == f"model.layers.{bid}.linear_attn.in_proj_b.weight":
            return f"blk.{bid}.ssm_beta_alpha.weight", 0, 16
        if hf_name == f"model.layers.{bid}.linear_attn.in_proj_a.weight":
            return f"blk.{bid}.ssm_beta_alpha.weight", 16, 32
        if hf_name == f"model.layers.{bid}.linear_attn.out_proj.weight":
            return f"blk.{bid}.ssm_out.weight", None, None
    raise KeyError(f"Unmapped HF layer param: {hf_name}")


def _compute_qwen35_hawq_awq_scores(cal_rows: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, float]]:
    """Compute real HAWQ-V2 (Fisher diag * err^2) and AWQ (salient col norm * err^2) per GGUF 2D matrix."""
    runner = QwenDeciderRunner()
    for p in runner.model.parameters():
        p.requires_grad_(False)
    base_cpu = capture_cpu_state_dict(runner.model)
    t2_bundle = load_patch_bundle(CHECKPOINTS_DIR / "track2_best_deltas.pt")
    t2_bf16 = CHECKPOINTS_DIR / "wsp-automode-0.8b-bf16.gguf"
    t2_q4 = CHECKPOINTS_DIR / "wsp-automode-0.8b-Q4_K_M.gguf"

    # Load BF16 & Q4_K_M GGUF weights mapped to HF for exact quantization error (W_bf16 - W_Q4)
    # Apply Track 2 bundle first for BF16 forward/backward sensitivity
    from weight_edit.causal_check import apply_bundle
    apply_bundle(runner.model, base_cpu, t2_bundle.patches)
    ab_ids = [runner.letter_token_ids["A"], runner.letter_token_ids["B"]]

    # Build raw GGUF quantization error squared dict on CPU keyed by GGUF tensor name
    r_bf = gguf.GGUFReader(str(t2_bf16))
    r_q4 = gguf.GGUFReader(str(t2_q4))
    w_bf_raw = {t.name: gguf.dequantize(t.data, t.tensor_type) for t in r_bf.tensors if len(t.shape) == 2 and str(t.tensor_type.name) != "F32"}
    w_q4_raw = {t.name: gguf.dequantize(t.data, t.tensor_type) for t in r_q4.tensors if len(t.shape) == 2 and str(t.tensor_type.name) != "F32"}
    err_sq_gguf: dict[str, torch.Tensor] = {}
    for gname in w_bf_raw:
        bn = _blk_num(gname)
        if gname == "token_embd.weight" or (bn is not None and 0 <= bn < 24):
            diff = torch.from_numpy(np.array(w_bf_raw[gname], copy=True)).float() - torch.from_numpy(np.array(w_q4_raw[gname], copy=True)).float()
            err_sq_gguf[gname] = diff * diff
    del w_bf_raw, w_q4_raw

    # Calibration prompts strictly from phase5_train (disjoint from select_dev)
    cal_items: list[tuple[torch.Tensor, float]] = []
    for r in cal_rows:
        pr = build_plain_prompt(render_view(r, tok=runner.tokenizer, budget=2048, include_recap=False).text, Q_BLOCK_BINARY, False).text
        ids = torch.tensor([runner.tokenizer.encode(pr, add_special_tokens=False)], dtype=torch.long)
        cal_items.append((ids, 1.0 if r["label"] == "block" else 0.0))

    # Enable grad on all active 2D matrices + embed_tokens, register post_accumulate_grad_hook to offload CPU immediately (VRAM < 2.2 GiB)
    for p in runner.model.parameters():
        p.requires_grad_(False)

    param_to_gguf: dict[int, tuple[str, int | None, int | None]] = {}
    fisher_gguf = {gname: torch.zeros_like(t, dtype=torch.float32) for gname, t in err_sq_gguf.items()}
    emb_p = runner.model.get_input_embeddings().weight
    emb_p.requires_grad_(True)
    param_to_gguf[id(emb_p)] = ("token_embd.weight", None, None)

    for b in range(24):
        blk = runner.model.model.layers[b]
        mod_names = (
            [("mlp.gate_proj", blk.mlp.gate_proj), ("mlp.up_proj", blk.mlp.up_proj), ("mlp.down_proj", blk.mlp.down_proj)]
            + (
                [("self_attn.q_proj", blk.self_attn.q_proj), ("self_attn.k_proj", blk.self_attn.k_proj), ("self_attn.v_proj", blk.self_attn.v_proj), ("self_attn.o_proj", blk.self_attn.o_proj)]
                if b in FULL_ATTN_LAYERS
                else [
                    ("linear_attn.in_proj_qkv", blk.linear_attn.in_proj_qkv),
                    ("linear_attn.in_proj_z", blk.linear_attn.in_proj_z),
                    ("linear_attn.in_proj_b", blk.linear_attn.in_proj_b),
                    ("linear_attn.in_proj_a", blk.linear_attn.in_proj_a),
                    ("linear_attn.out_proj", blk.linear_attn.out_proj),
                ]
            )
        )
        for suffix, mod in mod_names:
            hf_n = f"model.layers.{b}.{suffix}.weight"
            mod.weight.requires_grad_(True)
            param_to_gguf[id(mod.weight)] = _hf_key_to_gguf_qwen35(hf_n, b)

    # Register gradient offload hooks + forward AWQ activation norm hooks
    hook_handles = []
    act_sq_sum: dict[str, torch.Tensor] = {}
    act_tok_cnt: dict[str, int] = {}

    for _pname, p in runner.model.named_parameters():
        if id(p) in param_to_gguf:
            gname, s0, s1 = param_to_gguf[id(p)]

            def _make_grad_hook(gn: str, st: int | None, en: int | None):
                def _gh(param: torch.Tensor) -> None:
                    if param.grad is None:
                        return
                    g_sq = param.grad.detach().cpu().float().pow_(2)
                    if st is None:
                        fisher_gguf[gn] += g_sq
                    else:
                        fisher_gguf[gn][st:en, :] += g_sq
                    param.grad = None
                return _gh

            hook_handles.append(p.register_post_accumulate_grad_hook(_make_grad_hook(gname, s0, s1)))

    # AWQ activation collection hooks (forward_pre_hook on block linears; lm_head uses F.linear below)
    def _awq_pre_hook(gn: str):
        def _fh(_m, inp):
            x = inp[0][0].detach().cpu().float()  # (T, in_d)
            v = (x * x).sum(dim=0)
            if gn in act_sq_sum:
                act_sq_sum[gn] += v
                act_tok_cnt[gn] += int(x.shape[0])
            else:
                act_sq_sum[gn] = v
                act_tok_cnt[gn] = int(x.shape[0])
        return _fh

    for b in range(24):
        blk = runner.model.model.layers[b]
        hook_handles.append(blk.mlp.gate_proj.register_forward_pre_hook(_awq_pre_hook(f"blk.{b}.ffn_gate.weight")))
        hook_handles.append(blk.mlp.up_proj.register_forward_pre_hook(_awq_pre_hook(f"blk.{b}.ffn_up.weight")))
        hook_handles.append(blk.mlp.down_proj.register_forward_pre_hook(_awq_pre_hook(f"blk.{b}.ffn_down.weight")))
        if b in FULL_ATTN_LAYERS:
            hook_handles.append(blk.self_attn.q_proj.register_forward_pre_hook(_awq_pre_hook(f"blk.{b}.attn_q.weight")))
            hook_handles.append(blk.self_attn.k_proj.register_forward_pre_hook(_awq_pre_hook(f"blk.{b}.attn_k.weight")))
            hook_handles.append(blk.self_attn.v_proj.register_forward_pre_hook(_awq_pre_hook(f"blk.{b}.attn_v.weight")))
            hook_handles.append(blk.self_attn.o_proj.register_forward_pre_hook(_awq_pre_hook(f"blk.{b}.attn_output.weight")))
        else:
            hook_handles.append(blk.linear_attn.in_proj_qkv.register_forward_pre_hook(_awq_pre_hook(f"blk.{b}.attn_qkv_gate.weight")))
            hook_handles.append(blk.linear_attn.in_proj_b.register_forward_pre_hook(_awq_pre_hook(f"blk.{b}.ssm_beta_alpha.weight")))
            hook_handles.append(blk.linear_attn.out_proj.register_forward_pre_hook(_awq_pre_hook(f"blk.{b}.ssm_out.weight")))

    runner.model.train()
    runner.model.enable_input_require_grads()
    runner.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    for ids, y_val in cal_items:
        runner.model.zero_grad(set_to_none=True)
        hid = runner.model.model(input_ids=ids.to(runner.device), use_cache=False).last_hidden_state[:, -1:, :]
        x_lm = hid[0].detach().float()
        v_lm = (x_lm * x_lm).sum(dim=0).cpu()
        if "token_embd.weight" in act_sq_sum:
            act_sq_sum["token_embd.weight"] += v_lm
            act_tok_cnt["token_embd.weight"] += int(x_lm.shape[0])
        else:
            act_sq_sum["token_embd.weight"] = v_lm
            act_tok_cnt["token_embd.weight"] = int(x_lm.shape[0])
        lm_w = runner.model.lm_head.weight[ab_ids]
        lm_b = runner.model.lm_head.bias[ab_ids] if getattr(runner.model.lm_head, "bias", None) is not None else None
        lg = F.linear(hid.to(dtype=lm_w.dtype), lm_w, lm_b)[0, 0, :].float()
        margin = lg[1] - lg[0]
        tgt = torch.tensor(y_val, dtype=torch.float32, device=margin.device)
        loss = F.binary_cross_entropy_with_logits(margin, tgt)
        loss.backward()
        del loss, margin, lg, hid, tgt
        runner.model.zero_grad(set_to_none=True)

    peak_vram = torch.cuda.max_memory_allocated() / (1024.0 ** 3) if torch.cuda.is_available() else 0.0
    print(f"[R7b Sensitivity] 32-prompt Fisher+AWQ scan done (peak VRAM={peak_vram:.2f} GiB).", flush=True)
    for h in hook_handles:
        h.remove()

    n_cal = float(len(cal_items))
    hawq_scores: dict[str, float] = {}
    awq_scores: dict[str, float] = {}
    for gname, esq in err_sq_gguf.items():
        hawq_scores[gname] = float(((fisher_gguf[gname] / n_cal) * esq).sum().item())
        x_mean_sq = act_sq_sum[gname] / max(1, act_tok_cnt[gname])
        awq_scores[gname] = float((esq.sum(dim=0) * x_mean_sq).sum().item())

    del runner, base_cpu, fisher_gguf, err_sq_gguf
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return hawq_scores, awq_scores


def run_r7b_and_r7c(dev_rows: list[dict[str, Any]], port: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run R7b (HAWQ-V2 / AWQ budget-matched baselines) + R7c (OOF & external calibration, asymmetric risk, error decomp, non-pooled Taylor)."""
    TMP_GGUF_DIR.mkdir(parents=True, exist_ok=True)
    train_rows = _load_train_rows()
    cal_rows = [r for r in train_rows if r["subset"] == "real_cf"][:16] + [r for r in train_rows if r["subset"] == "consent_5p"][:16]
    hawq_scores, awq_scores = _compute_qwen35_hawq_awq_scores(cal_rows)

    t2_bf16 = CHECKPOINTS_DIR / "wsp-automode-0.8b-bf16.gguf"
    t3_bf16 = CHECKPOINTS_DIR / "automode-0.8b-track3-bf16.gguf"
    base_snap = Path(snapshot_download(BASE_ID, local_files_only=True))
    base_bf16 = TMP_GGUF_DIR / "r7_base-bf16.gguf"
    if not base_bf16.exists():
        subprocess.run([sys.executable, str(CONVERT_HF_GGUF), str(base_snap), "--outfile", str(base_bf16), "--outtype", "bf16"], check=True)

    # Exact byte costs for Q8_0 vs Q4_K_M per GGUF tensor
    r_q8 = gguf.GGUFReader(str(CHECKPOINTS_DIR / "wsp-automode-0.8b-Q8_0.gguf"))
    r_q4 = gguf.GGUFReader(str(CHECKPOINTS_DIR / "wsp-automode-0.8b-Q4_K_M.gguf"))
    bytes_q8 = {t.name: int(t.n_bytes) for t in r_q8.tensors}
    bytes_q4 = {t.name: int(t.n_bytes) for t in r_q4.tensors}
    base_q4_size = (CHECKPOINTS_DIR / "wsp-automode-0.8b-Q4_K_M.gguf").stat().st_size

    # Build L0-14 only Q8_0 GGUF first to get exact target byte budget
    t2_reader = gguf.GGUFReader(str(t2_bf16))
    bf16_names = [t.name for t in t2_reader.tensors if str(t.tensor_type.name) in ("BF16", "F16")]
    map_l0_14 = plan_upstream_overrides([(t.name, str(t.tensor_type.name)) for t in t2_reader.tensors], edit_layer=15, upstream_quant="Q8_0", include_embd=False)
    g_l0_14 = _build_surgical_gguf(t2_bf16, TMP_GGUF_DIR / "r7_t2_L0_14_only.gguf", "Q4_K_M", map_l0_14)
    target_size_bytes = g_l0_14.stat().st_size
    budget_delta = target_size_bytes - base_q4_size

    def _greedy_allocate(scores: dict[str, float], granularity: str) -> tuple[dict[str, str], list[str], list[tuple[str, float, float]]]:
        """Greedily upgrade highest-scoring blocks or tensors to Q8_0 within exact L0-14 byte budget."""
        if granularity == "block":
            units: dict[str, list[str]] = {}
            unit_score: dict[str, float] = {}
            for tname, sc in scores.items():
                bn = _blk_num(tname)
                uk = f"blk.{bn}" if bn is not None else tname
                units.setdefault(uk, []).append(tname)
                unit_score[uk] = unit_score.get(uk, 0.0) + sc
            unit_cost = {uk: sum(max(0, bytes_q8[t] - bytes_q4[t]) for t in tlist) for uk, tlist in units.items()}
        else:
            units = {t: [t] for t in scores}
            unit_score = dict(scores)
            unit_cost = {t: max(0, bytes_q8[t] - bytes_q4[t]) for t in scores}

        ranked = sorted(unit_score.keys(), key=lambda u: unit_score[u], reverse=True)
        chosen: list[str] = []
        spent = 0
        tmap: dict[str, str] = {}
        for u in ranked:
            c = unit_cost[u]
            if spent + c <= budget_delta + 512 * 1024:  # within 0.5 MiB tolerance, strictly <= if possible
                if spent + c <= budget_delta or not chosen:
                    chosen.append(u)
                    spent += c
                    for tn in units[u]:
                        tmap[tn] = "q8_0"
        top_table = [(u, round(unit_score[u], 4), round(unit_cost[u] / (1024.0 ** 2), 2)) for u in ranked[:15]]
        return tmap, chosen, top_table

    hawq_b_map, hawq_b_chosen, hawq_b_top = _greedy_allocate(hawq_scores, "block")
    hawq_t_map, hawq_t_chosen, hawq_t_top = _greedy_allocate(hawq_scores, "tensor")
    awq_b_map, awq_b_chosen, awq_b_top = _greedy_allocate(awq_scores, "block")
    awq_t_map, awq_t_chosen, awq_t_top = _greedy_allocate(awq_scores, "tensor")

    def _emb_flags(tmap: dict[str, str]) -> list[str]:
        return ["--token-embedding-type", "q8_0", "--output-tensor-type", "q8_0"] if "token_embd.weight" in tmap else []

    g_hawq_b = _build_surgical_gguf(t2_bf16, TMP_GGUF_DIR / "r7b_hawq_block.gguf", "Q4_K_M", hawq_b_map, _emb_flags(hawq_b_map))
    g_hawq_t = _build_surgical_gguf(t2_bf16, TMP_GGUF_DIR / "r7b_hawq_tensor.gguf", "Q4_K_M", hawq_t_map, _emb_flags(hawq_t_map))
    g_awq_b = _build_surgical_gguf(t2_bf16, TMP_GGUF_DIR / "r7b_awq_block.gguf", "Q4_K_M", awq_b_map, _emb_flags(awq_b_map))
    g_awq_t = _build_surgical_gguf(t2_bf16, TMP_GGUF_DIR / "r7b_awq_tensor.gguf", "Q4_K_M", awq_t_map, _emb_flags(awq_t_map))

    # Build remaining configs needed for complete R7c calibration + Pareto table
    map_6x_bf16 = {t: "bf16" for t in TRACK2_GGUF_TENSORS}
    map_l15_23_embd = {t: "q8_0" for t in bf16_names if (t in ("token_embd.weight", "output.weight") or ((_blk_num(t) or -1) >= 15))}
    map_l9_23 = {t: "q8_0" for t in bf16_names if (_blk_num(t) or -1) >= 9}
    map_l0_14_bf16 = {t: ("bf16" if t in TRACK2_GGUF_TENSORS else "q8_0") for t in bf16_names if (t in TRACK2_GGUF_TENSORS or t in ("token_embd.weight", "output.weight") or ((_blk_num(t) if _blk_num(t) is not None else 99) <= 14))}
    map_l0_22_bf16 = {t: ("bf16" if t in TRACK2_GGUF_TENSORS else "q8_0") for t in bf16_names if (t in TRACK2_GGUF_TENSORS or t in ("token_embd.weight", "output.weight") or ((_blk_num(t) if _blk_num(t) is not None else 99) <= 22))}
    map_q8_bb_q4 = {t: "q4_K" for t in TRACK2_GGUF_TENSORS}
    emb_q8 = ["--token-embedding-type", "q8_0", "--output-tensor-type", "q8_0"]

    g_specs: list[tuple[str, str, Path]] = [
        ("base_Q4_K_M", "base", _build_surgical_gguf(base_bf16, TMP_GGUF_DIR / "r7_base_Q4_K_M.gguf", "Q4_K_M", {})),
        ("base_L0_14_only_q8_0", "base", _build_surgical_gguf(base_bf16, TMP_GGUF_DIR / "r7_base_L0_14.gguf", "Q4_K_M", map_l0_14)),
        ("track3_Q4_K_M", "t3", CHECKPOINTS_DIR / "automode-0.8b-track3-Q4_K_M.gguf"),
        ("track3_L0_14_only_q8_0", "t3", _build_surgical_gguf(t3_bf16, TMP_GGUF_DIR / "r7_t3_L0_14.gguf", "Q4_K_M", map_l0_14)),
        ("track2_Q8_0", "t2", CHECKPOINTS_DIR / "wsp-automode-0.8b-Q8_0.gguf"),
        ("track2_Q6_K", "t2", CHECKPOINTS_DIR / "wsp-automode-0.8b-Q6_K.gguf"),
        ("track2_Q5_K_M", "t2", CHECKPOINTS_DIR / "wsp-automode-0.8b-Q5_K_M.gguf"),
        ("track2_Q4_K_M", "t2", CHECKPOINTS_DIR / "wsp-automode-0.8b-Q4_K_M.gguf"),
        ("t2_Q4_6x_bf16", "t2", _build_surgical_gguf(t2_bf16, TMP_GGUF_DIR / "r7_t2_6x_bf16.gguf", "Q4_K_M", map_6x_bf16)),
        ("t2_Q4_hawq_block_q8_0", "t2", g_hawq_b),
        ("t2_Q4_hawq_tensor_q8_0", "t2", g_hawq_t),
        ("t2_Q4_awq_block_q8_0", "t2", g_awq_b),
        ("t2_Q4_awq_tensor_q8_0", "t2", g_awq_t),
        ("t2_Q4_L15_23_embd_q8_0", "t2", _build_surgical_gguf(t2_bf16, TMP_GGUF_DIR / "r7_t2_L15_23.gguf", "Q4_K_M", map_l15_23_embd, emb_q8)),
        ("t2_Q4_L9_23_only_q8_0", "t2", _build_surgical_gguf(t2_bf16, TMP_GGUF_DIR / "r7_t2_L9_23.gguf", "Q4_K_M", map_l9_23)),
        ("t2_Q4_L0_14_only_q8_0", "t2", g_l0_14),
        ("t2_Q4_L0_14_6x_bf16", "t2", _build_surgical_gguf(t2_bf16, TMP_GGUF_DIR / "r7_t2_L0_14_bf16.gguf", "Q4_K_M", map_l0_14_bf16, emb_q8)),
        ("t2_Q8_bb_6x_Q4_K", "t2", _build_surgical_gguf(t2_bf16, TMP_GGUF_DIR / "r7_t2_Q8_bb.gguf", "Q8_0", map_q8_bb_q4)),
        ("t2_Q4_L0_22_6x_bf16", "t2", _build_surgical_gguf(t2_bf16, TMP_GGUF_DIR / "r7_t2_L0_22.gguf", "Q4_K_M", map_l0_22_bf16, emb_q8)),
    ]

    refs = {
        "base": collect_pt_select_dev_ref(None, dev_rows),
        "t2": collect_pt_select_dev_ref(CHECKPOINTS_DIR / "track2_best_deltas.pt", dev_rows),
        "t3": collect_pt_select_dev_ref(CHECKPOINTS_DIR / "track3_final_best_deltas.pt", dev_rows),
    }

    # Also build external observe calibration rows (40 binary prompts disjoint from select_dev)
    obs_rows = load_split_rows("observe")
    obs_bin = [r for r in obs_rows if r["subset"] == "real_cf"][:20] + [r for r in obs_rows if r["subset"] == "consent_5p"][:20]
    runner_obs = QwenDeciderRunner()
    base_obs_cpu = capture_cpu_state_dict(runner_obs.model)
    from weight_edit.causal_check import apply_bundle
    obs_pt_margins: dict[str, np.ndarray] = {}
    obs_ctxs = [render_view(r, tok=runner_obs.tokenizer, budget=2048, include_recap=False).text for r in obs_bin]
    for rkey, dpath in [("base", None), ("t2", CHECKPOINTS_DIR / "track2_best_deltas.pt"), ("t3", CHECKPOINTS_DIR / "track3_final_best_deltas.pt")]:
        if dpath is None:
            restore_cpu_state_dict(runner_obs.model, base_obs_cpu)
        else:
            apply_bundle(runner_obs.model, base_obs_cpu, load_patch_bundle(dpath).patches)
        obs_pt_margins[rkey] = np.array(
            [float(runner_obs.score_binary_paired(c, mode="plain", question_spec=Q_BLOCK_BINARY)["margin_normal"]) for c in obs_ctxs],
            dtype=np.float64,
        )
    del runner_obs, base_obs_cpu
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n_bin = len(refs["t2"]["binary"])
    n_all = n_bin + len(refs["t2"]["replay"])
    idx_bin = rng.integers(0, n_bin, size=(BOOTSTRAP_B, n_bin), dtype=np.int32)
    idx_all = rng.integers(0, n_all, size=(BOOTSTRAP_B, n_all), dtype=np.int32)
    # 5 stratified folds across 140 binary prompts for strictly leakage-free OOF bias calibration
    fold_ids = np.arange(n_bin) % 5

    all_summaries: dict[str, Any] = {}
    all_stats: dict[str, Any] = {}
    all_dists: dict[str, Any] = {}

    for cname, rkey, gpath in g_specs:
        print(f"[R7b/c Server] {cname} ({gpath.name})...", flush=True)
        summary, vecs = eval_gguf_select_dev_vectors(gpath, refs[rkey], port)
        b_stats, b_dists = _bootstrap_distribution(vecs, idx_bin, idx_all)

        # Evaluate 40 external observe items on the same GGUF? Or launch quickly for observe subset too!
        # Using exact same launch script cleanly takes ~4s per GGUF
        obs_url = f"http://127.0.0.1:{port}"
        log_f = open(CHECKPOINTS_DIR / "llama_revision_controls.log", "w", encoding="utf-8")
        proc = subprocess.Popen([str(LAUNCH_SCRIPT), str(gpath), str(port)], stdout=log_f, stderr=subprocess.STDOUT, text=True)
        try:
            _wait_server(obs_url, proc)
            client = LlamaCompletionClient(base_url=obs_url, second_reading_band=(2.0, 2.0))
            obs_q_m = np.array([float(client.score_binary_paired(c, force_paired=False)["margin_normal"]) for c in obs_ctxs], dtype=np.float64)
        finally:
            proc.terminate()
            proc.wait(timeout=10)
            log_f.close()

        pt_m = vecs["pt_m"]
        q_m = vecs["q_m"]
        ag_200 = vecs["agree_200"]
        ag_rep = ag_200[n_bin:]

        # 5-fold stratified OOF bias shift calibration on select_dev (binary prompts)
        q_m_cal_oof = np.empty_like(q_m)
        for fid in range(5):
            tr_mask = fold_ids != fid
            te_mask = fold_ids == fid
            delta_f = float(np.mean(q_m[tr_mask] - pt_m[tr_mask]))
            q_m_cal_oof[te_mask] = q_m[te_mask] - delta_f

        # External observe calibration (zero select_dev usage)
        delta_obs = float(np.mean(obs_q_m - obs_pt_margins[rkey]))
        q_m_cal_obs = q_m - delta_obs

        cal_ag_bin_oof = ((q_m_cal_oof >= 0.0) == (pt_m >= 0.0)).astype(np.float64)
        cal_ag_200_oof = np.concatenate([cal_ag_bin_oof, ag_rep])
        cal_ag_bin_obs = ((q_m_cal_obs >= 0.0) == (pt_m >= 0.0)).astype(np.float64)
        cal_ag_200_obs = np.concatenate([cal_ag_bin_obs, ag_rep])

        lo_c, hi_c = np.percentile(cal_ag_200_oof[idx_all].mean(axis=1), [2.5, 97.5])
        pt_blk = pt_m >= 0.0
        q_blk = q_m >= 0.0
        q_cal_blk = q_m_cal_oof >= 0.0

        # Asymmetric risk wrt BF16 reference (underblock FN = BF16 blocks, Q allows; overblock FP = BF16 allows, Q blocks)
        raw_fn_rate = float(np.mean((~q_blk[pt_blk]))) if np.any(pt_blk) else 0.0
        raw_fp_rate = float(np.mean(q_blk[~pt_blk])) if np.any(~pt_blk) else 0.0
        cal_fn_rate = float(np.mean((~q_cal_blk[pt_blk]))) if np.any(pt_blk) else 0.0
        cal_fp_rate = float(np.mean(q_cal_blk[~pt_blk])) if np.any(~pt_blk) else 0.0

        # real_cf ground-truth Rec / FPR after OOF calibration
        rc_mask = np.array([x["subset"] == "real_cf" for x in refs[rkey]["binary"]], dtype=bool)
        rc_y = np.array([x["y"] == 1 for x in refs[rkey]["binary"]], dtype=bool)
        rc_rec_raw = float(np.mean(q_blk[rc_mask & rc_y]))
        rc_fpr_raw = float(np.mean(q_blk[rc_mask & (~rc_y)]))
        rc_rec_cal = float(np.mean(q_cal_blk[rc_mask & rc_y]))
        rc_fpr_cal = float(np.mean(q_cal_blk[rc_mask & (~rc_y)]))

        all_summaries[cname] = {
            **summary,
            "bootstrap_ci95": b_stats,
            "cal_oof_margin_mae": round(float(np.mean(np.abs(q_m_cal_oof - pt_m))), 4),
            "cal_oof_agree_140": round(float(np.mean(cal_ag_bin_oof)), 4),
            "cal_oof_agree_200": {
                "point": round(float(np.mean(cal_ag_200_oof)), 4),
                "ci95_lo": round(float(lo_c), 4),
                "ci95_hi": round(float(hi_c), 4),
            },
            "ext_observe_delta_bias": round(delta_obs, 4),
            "cal_ext_agree_140": round(float(np.mean(cal_ag_bin_obs)), 4),
            "cal_ext_agree_200": round(float(np.mean(cal_ag_200_obs)), 4),
            "raw_underblock_fn_vs_bf16": round(raw_fn_rate, 4),
            "raw_overblock_fp_vs_bf16": round(raw_fp_rate, 4),
            "cal_underblock_fn_vs_bf16": round(cal_fn_rate, 4),
            "cal_overblock_fp_vs_bf16": round(cal_fp_rate, 4),
            "rc_rec_fpr_raw": [round(rc_rec_raw, 4), round(rc_fpr_raw, 4)],
            "rc_rec_fpr_cal": [round(rc_rec_cal, 4), round(rc_fpr_cal, 4)],
        }
        all_stats[cname] = b_stats
        all_dists[cname] = b_dists

    # Paired diffs: L0-14 Only vs HAWQ & AWQ baselines
    r7b_paired = [
        _paired_diff_summary("t2_Q4_L0_14_only_q8_0", b_k, all_stats["t2_Q4_L0_14_only_q8_0"], all_stats[b_k], all_dists["t2_Q4_L0_14_only_q8_0"], all_dists[b_k], "pearson_r")
        for b_k in ("t2_Q4_hawq_block_q8_0", "t2_Q4_hawq_tensor_q8_0", "t2_Q4_awq_block_q8_0", "t2_Q4_awq_tensor_q8_0")
    ]

    # Non-pooled & direction-controlled Taylor validation across Q8_0..Q4_K_M on Track 2
    runner_t = QwenDeciderRunner()
    base_cpu_t = capture_cpu_state_dict(runner_t.model)
    ab_ids_t = [runner_t.letter_token_ids["A"], runner_t.letter_token_ids["B"]]
    prompt_ids_t = _select_dev_prompt_ids(runner_t, dev_rows)
    d_ff = int(runner_t.model.model.layers[15].mlp.gate_proj.weight.shape[0])
    unt15 = torch.ones(d_ff, dtype=torch.bool)
    unt15[T2_L15_NEURONS] = False
    unt22 = torch.ones(d_ff, dtype=torch.bool)
    unt22[T2_L22_NEURONS] = False

    w_t2_bf = _dequantize_gguf_2d_weights(t2_bf16)
    _apply_dequant_dict(runner_t.model, base_cpu_t, w_t2_bf)
    ref_t2_int = _probe_subset(runner_t, prompt_ids_t, ab_ids_t, (15, 22))
    j_t2_15 = _compute_swiglu_jacobian_stats(runner_t, prompt_ids_t, 15, T2_L15_NEURONS, unt15)
    j_t2_22 = _compute_swiglu_jacobian_stats(runner_t, prompt_ids_t, 22, T2_L22_NEURONS, unt22)

    taylor_controls: dict[str, Any] = {}
    rng_rot = np.random.default_rng(12345)
    # Build 20 fixed random orthogonal 1024x1024 projections for L15 & L22? Even faster: random normalized directions!
    # Wait: isotropic rotation of vector dh in R^d: R @ dh is distributed identically to ||dh|| * u where u ~ Uniform(S^{d-1})!
    # Generating orthogonal matrices once cleanly via QR on (1024, 1024) takes < 0.3s total!
    rot_mats = []
    for _ in range(20):
        q_mat, _ = np.linalg.qr(rng_rot.standard_normal((1024, 1024)).astype(np.float32))
        rot_mats.append(torch.from_numpy(q_mat))

    for qt in QUANT_TYPES:
        w_q = _dequantize_gguf_2d_weights(CHECKPOINTS_DIR / f"wsp-automode-0.8b-{qt}.gguf")
        _apply_dequant_dict(runner_t.model, base_cpu_t, w_q)
        qp = _probe_subset(runner_t, prompt_ids_t, ab_ids_t, (15, 22))

        qt_dict: dict[str, Any] = {}
        for lkey, hk, gk, ed_idx, j_obj in [("l15", "h1", "g1", T2_L15_NEURONS, j_t2_15), ("l22", "h2", "g2", T2_L22_NEURONS, j_t2_22)]:
            dh = qp[hk] - ref_t2_int[hk]  # (N, 1024)
            dg_true = (qp[gk][:, ed_idx] - ref_t2_int[gk][:, ed_idx]).numpy()  # (N, 4)
            J = j_obj["J_ed"]  # (N, 4, 1024)

            pred_dir = torch.einsum("nkd,nd->nk", J, dh).numpy()  # signed (N, 4)
            abs_pred_dir = np.abs(pred_dir)
            abs_true = np.abs(dg_true)
            dh_norm = torch.linalg.norm(dh, dim=1, keepdim=True).numpy()  # (N, 1)
            dh_norm_b = np.repeat(dh_norm, len(ed_idx), axis=1)
            j_norm = torch.linalg.norm(J, dim=2).numpy()  # (N, 4)
            iso_mag = j_norm * dh_norm_b

            # Random orthogonal rotations (20 trials)
            rot_r_pool, rot_r_within = [], []
            for R in rot_mats:
                dh_r = dh @ R.T
                pr_r = np.abs(torch.einsum("nkd,nd->nk", J, dh_r).numpy())
                rot_r_pool.append(float(np.corrcoef(pr_r.reshape(-1), abs_true.reshape(-1))[0, 1]))
                rot_r_within.append(float(np.mean([np.corrcoef(pr_r[:, k], abs_true[:, k])[0, 1] for k in range(len(ed_idx))])))

            within_dir_abs = [float(np.corrcoef(abs_pred_dir[:, k], abs_true[:, k])[0, 1]) for k in range(len(ed_idx))]
            within_dir_signed = [float(np.corrcoef(pred_dir[:, k], dg_true[:, k])[0, 1]) for k in range(len(ed_idx))]
            within_norm = [float(np.corrcoef(dh_norm_b[:, k], abs_true[:, k])[0, 1]) for k in range(len(ed_idx))]
            within_iso = [float(np.corrcoef(iso_mag[:, k], abs_true[:, k])[0, 1]) for k in range(len(ed_idx))]
            sample_rs = [
                float(np.corrcoef(abs_pred_dir[i], abs_true[i])[0, 1])
                for i in range(abs_true.shape[0])
                if np.std(abs_pred_dir[i]) > 1e-8 and np.std(abs_true[i]) > 1e-8
            ]

            qt_dict[lkey] = {
                "pooled_r_directional_abs": round(float(np.corrcoef(abs_pred_dir.reshape(-1), abs_true.reshape(-1))[0, 1]), 4),
                "pooled_r_directional_signed": round(float(np.corrcoef(pred_dir.reshape(-1), dg_true.reshape(-1))[0, 1]), 4),
                "within_neuron_r_dir_abs_mean": round(float(np.mean(within_dir_abs)), 4),
                "within_neuron_r_dir_abs_min": round(float(np.min(within_dir_abs)), 4),
                "within_neuron_r_dir_signed_mean": round(float(np.mean(within_dir_signed)), 4),
                "within_sample_r_dir_abs_mean": round(float(np.mean(sample_rs)), 4),
                "pooled_r_norm_only": round(float(np.corrcoef(dh_norm_b.reshape(-1), abs_true.reshape(-1))[0, 1]), 4),
                "within_neuron_r_norm_only_mean": round(float(np.mean(within_norm)), 4),
                "pooled_r_iso_gain_x_norm": round(float(np.corrcoef(iso_mag.reshape(-1), abs_true.reshape(-1))[0, 1]), 4),
                "within_neuron_r_iso_gain_x_norm_mean": round(float(np.mean(within_iso)), 4),
                "pooled_r_random_rot_mean": round(float(np.mean(rot_r_pool)), 4),
                "within_neuron_r_random_rot_mean": round(float(np.mean(rot_r_within)), 4),
            }
        taylor_controls[qt] = qt_dict

    del runner_t, base_cpu_t
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    r7b = {
        "target_l0_14_mib": round(target_size_bytes / (1024.0 ** 2), 2),
        "hawq_block_chosen": hawq_b_chosen,
        "hawq_block_top15": hawq_b_top,
        "hawq_tensor_chosen": hawq_t_chosen,
        "hawq_tensor_top15": hawq_t_top,
        "awq_block_chosen": awq_b_chosen,
        "awq_block_top15": awq_b_top,
        "awq_tensor_chosen": awq_t_chosen,
        "awq_tensor_top15": awq_t_top,
        "configs": {k: all_summaries[k] for k in ("t2_Q4_L0_14_only_q8_0", "t2_Q4_hawq_block_q8_0", "t2_Q4_hawq_tensor_q8_0", "t2_Q4_awq_block_q8_0", "t2_Q4_awq_tensor_q8_0")},
        "paired_differences": r7b_paired,
    }
    r7c = {
        "calibrated_and_risk_configs": all_summaries,
        "error_decomposition_q4": {
            "base_generic_raw_mae": all_summaries["base_Q4_K_M"]["bootstrap_ci95"]["margin_mae"]["point"],
            "track2_total_raw_mae": all_summaries["track2_Q4_K_M"]["bootstrap_ci95"]["margin_mae"]["point"],
            "track2_excess_over_base": round(all_summaries["track2_Q4_K_M"]["bootstrap_ci95"]["margin_mae"]["point"] - all_summaries["base_Q4_K_M"]["bootstrap_ci95"]["margin_mae"]["point"], 4),
            "base_l0_14_raw_mae": all_summaries["base_L0_14_only_q8_0"]["bootstrap_ci95"]["margin_mae"]["point"],
            "track3_q4_raw_mae": all_summaries["track3_Q4_K_M"]["bootstrap_ci95"]["margin_mae"]["point"],
            "track3_l0_14_raw_mae": all_summaries["track3_L0_14_only_q8_0"]["bootstrap_ci95"]["margin_mae"]["point"],
        },
        "taylor_direction_and_nonpooled_controls": taylor_controls,
    }
    return r7b, r7c


def _run_r7d_single(
    cond_name: str,
    seed: int,
    lr: float,
    gate_lr: float,
    freeze_mode: str,
    init_from_track2: bool,
    clip_norm: float,
    warmup_frac: float,
    dev_rows: list[dict[str, Any]],
    diag_items: list[tuple[torch.Tensor, float]],
    groups: list[Any],
    probe_ids: list[torch.Tensor],
    prompt_ids: list[torch.Tensor],
    max_groups: int = 96,
) -> dict[str, Any]:
    """Run Steps 0..12 for one seed & condition, logging direct pre-activation a_j receptive-field distribution."""
    grad_accum = 8
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    runner = QwenDeciderRunner()
    for p in runner.model.parameters():
        p.requires_grad_(False)
    base_cpu = capture_cpu_state_dict(runner.model)
    ab_ids = [runner.letter_token_ids["A"], runner.letter_token_ids["B"]]

    if init_from_track2:
        from weight_edit.causal_check import apply_bundle
        apply_bundle(runner.model, base_cpu, load_patch_bundle(CHECKPOINTS_DIR / "track2_best_deltas.pt").patches)

    inject_lora(runner.model, r=16, alpha=32.0, train_deltanet_gates=True)

    lora_all, gate_all = lora_param_groups(runner.model)
    t_set = set(lora_all + gate_all)
    l0_8_params, l9_14_params, l15_23_shared, t2_mlp_params, l9_23_all = [], [], [], [], []
    for n, p in runner.model.named_parameters():
        if p not in t_set or not n.startswith("model.layers."):
            continue
        li = int(n.split(".")[2])
        if li < 9:
            l0_8_params.append(p)
        else:
            l9_23_all.append(p)
            if li < 15:
                l9_14_params.append(p)
            elif ("model.layers.15.mlp." in n) or ("model.layers.22.mlp." in n):
                t2_mlp_params.append(p)
            else:
                l15_23_shared.append(p)

    l0_14_params = l0_8_params + l9_14_params
    need_input_grad = True
    if freeze_mode == "freeze_L0_14":
        for p in l0_14_params:
            p.requires_grad_(False)
        need_input_grad = False
    elif freeze_mode == "freeze_L9_23":
        for p in l9_23_all:
            p.requires_grad_(False)
    elif freeze_mode == "freeze_t2gates":
        for p in t2_mlp_params:
            p.requires_grad_(False)
    elif freeze_mode != "none":
        raise ValueError(freeze_mode)

    lora_params, gate_params = lora_param_groups(runner.model)
    all_trainable = lora_params + gate_params
    optimizer = torch.optim.AdamW(
        [{"params": lora_params, "lr": lr, "weight_decay": 0.02}, {"params": gate_params, "lr": gate_lr, "weight_decay": 0.0}],
        betas=(0.9, 0.95),
    )
    steps_per_epoch = math.ceil(len(groups) / grad_accum)
    total_opt_steps = 2 * steps_per_epoch
    warmup_steps = max(1, int(warmup_frac * total_opt_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: _cosine_lr_multiplier(s, total_opt_steps, warmup_steps, min_ratio=0.1)
    )

    h15_ref_box: list[torch.Tensor] = []

    @torch.inference_mode()
    def _snap_state() -> dict[str, Any]:
        runner.model.eval()
        h15_l, a15_l, g15_l = [], [], []
        hnds = [
            runner.model.model.layers[15].mlp.register_forward_pre_hook(lambda _m, inp: h15_l.append(inp[0][0, -1, :].detach().float().cpu())),
            runner.model.model.layers[15].mlp.gate_proj.register_forward_hook(lambda _m, _inp, out: a15_l.append(out[0, -1, T2_L15_NEURONS].detach().float().cpu())),
            runner.model.model.layers[15].mlp.down_proj.register_forward_pre_hook(lambda _m, inp: g15_l.append(inp[0][0, -1, T2_L15_NEURONS].detach().float().cpu())),
        ]
        try:
            for ids in probe_ids:
                runner.model.model(input_ids=ids.to(runner.device), use_cache=False)
        finally:
            for h in hnds:
                h.remove()
        H15 = torch.stack(h15_l, dim=0)
        A15 = torch.stack(a15_l, dim=0).numpy()  # pre-activation w_gate^T h_15
        G15 = torch.stack(g15_l, dim=0).numpy()
        if not h15_ref_box:
            h15_ref_box.append(H15)
        cos = float(F.cosine_similarity(H15, h15_ref_box[0], dim=-1).mean().item())
        return {
            "h15_cos": round(cos, 6),
            "l15_live_swiglu_abs": round(float(np.mean(np.abs(G15))), 4),
            "l15_preact_a_mean": round(float(np.mean(A15)), 4),
            "l15_preact_a_p10": round(float(np.percentile(A15, 10)), 4),
            "l15_preact_a_p90": round(float(np.percentile(A15, 90)), 4),
            "l15_preact_frac_gt0": round(float(np.mean(A15 > 0.0)), 4),
        }

    from eval.revision_r6_theory_and_controls import _measure_warmstart_backward_probe
    runner.model.train()
    if need_input_grad:
        runner.model.enable_input_require_grads()
    runner.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    snap_steps = (0, 2, 5, 7, 10, 12)
    bwd_steps = (0, 5, 12)
    snapshots: list[dict[str, Any]] = []
    bwd0 = _measure_warmstart_backward_probe(runner, diag_items, ab_ids)
    st0 = _snap_state()
    snapshots.append({"opt_step": 0, "recent_loss": None, **st0, **bwd0})

    runner.model.train()
    rng = random.Random(seed)
    order = list(range(len(groups)))
    rng.shuffle(order)
    optimizer.zero_grad(set_to_none=True)

    recent_losses: list[float] = []
    step_grad_logs: list[dict[str, Any]] = []
    opt_step = 0
    for step_idx, g_i in enumerate(order[:max_groups]):
        st = train_group_step(runner.model, runner.device, ab_ids, groups[g_i], grad_accum)
        recent_losses.append(float(st["total"]))
        if len(recent_losses) > 20:
            recent_losses.pop(0)
        is_bnd = ((step_idx + 1) % grad_accum == 0) or (step_idx == max_groups - 1)
        if is_bnd:
            gn_tot = _group_grad_norm(all_trainable)
            gn_l0_8 = _group_grad_norm(l0_8_params)
            gn_sh = _group_grad_norm(l15_23_shared)
            torch.nn.utils.clip_grad_norm_(all_trainable, clip_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            opt_step += 1
            rec_mean = float(np.mean(recent_losses))
            step_grad_logs.append(
                {
                    "opt_step": opt_step,
                    "recent_loss": round(rec_mean, 4),
                    "total_grad_norm": round(gn_tot, 4),
                    "L0_8_grad_norm": round(gn_l0_8, 4),
                    "shared_L15_23_grad_norm": round(gn_sh, 4),
                }
            )
            if opt_step in snap_steps:
                s_row: dict[str, Any] = {"opt_step": opt_step, "recent_loss": round(rec_mean, 4), **_snap_state()}
                if opt_step in bwd_steps:
                    s_row.update(_measure_warmstart_backward_probe(runner, diag_items, ab_ids))
                snapshots.append(s_row)
                runner.model.train()

    # Step-12 fast dev evaluation on all 140 binary select_dev prompts
    runner.model.gradient_checkpointing_disable()
    runner.model.eval()
    p12 = _probe_subset(runner, prompt_ids, ab_ids, (15, 22))
    perf12 = _eval_binary_accuracy_140(p12["margins"], dev_rows)

    out = {
        "seed": seed,
        "n_trainable_total": sum(p.numel() for p in all_trainable),
        "peak_recent20_loss": round(max(x["recent_loss"] for x in step_grad_logs), 4),
        "max_total_grad_norm": round(max(x["total_grad_norm"] for x in step_grad_logs), 2),
        "max_L0_8_grad_norm": round(max(x["L0_8_grad_norm"] for x in step_grad_logs), 2),
        "max_shared_L15_23_grad_norm": round(max(x["shared_L15_23_grad_norm"] for x in step_grad_logs), 2),
        "step5_L0_8_grad_norm": step_grad_logs[4]["L0_8_grad_norm"],
        "step5_shared_L15_23_grad_norm": step_grad_logs[4]["shared_L15_23_grad_norm"],
        "max_rms_dL_dh15_via_mlp": round(max(s["rms_dL_dh15_via_mlp"] for s in snapshots if "rms_dL_dh15_via_mlp" in s), 4),
        "end_h15_cos": snapshots[-1]["h15_cos"],
        "start_l15_swiglu_abs": snapshots[0]["l15_live_swiglu_abs"],
        "end_l15_swiglu_abs": snapshots[-1]["l15_live_swiglu_abs"],
        "start_preact_a_mean": snapshots[0]["l15_preact_a_mean"],
        "end_preact_a_mean": snapshots[-1]["l15_preact_a_mean"],
        "start_preact_frac_gt0": snapshots[0]["l15_preact_frac_gt0"],
        "end_preact_frac_gt0": snapshots[-1]["l15_preact_frac_gt0"],
        "step12_rc_auroc": perf12["rc_auroc"],
        "step12_rc_rec": perf12["rc_rec"],
        "step12_rc_fpr": perf12["rc_fpr"],
        "step12_c5p_acc": perf12["c5p_acc"],
        "step12_mean_abs_margin": perf12["mean_abs_margin"],
        "step_grad_logs": step_grad_logs,
        "snapshots": snapshots,
    }
    print(f"[R7d {cond_name} s={seed}] pkLoss={out['peak_recent20_loss']:.2f} maxTot={out['max_total_grad_norm']} sh5={out['step5_shared_L15_23_grad_norm']:.1f} L0_8={out['max_L0_8_grad_norm']:.1f} frac>0: {out['start_preact_frac_gt0']}->{out['end_preact_frac_gt0']} auroc12={out['step12_rc_auroc']}", flush=True)
    del runner, base_cpu, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def run_r7d(dev_rows: list[dict[str, Any]]) -> dict[str, Any]:
    tmp_r = QwenDeciderRunner()
    groups = prepare_training_groups(tmp_r, Path(DEFAULT_TRAIN_FILE))
    probe_ids = _prepare_probe_prompts(tmp_r, dev_rows)
    prompt_ids = _select_dev_prompt_ids(tmp_r, dev_rows)
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

    seeds = (20261124, 20261125, 20261126)
    max_groups = 96
    specs = [
        ("t3a_insitu_base", 1.5e-4, 5.0e-4, "none", False, 1.0, 0.08),
        ("t3b_insitu_full", 1.5e-4, 5.0e-4, "none", True, 1.0, 0.08),
        ("t3b_freeze_t2gates", 1.5e-4, 5.0e-4, "freeze_t2gates", True, 1.0, 0.08),
        ("t3b_freeze_L0_14", 1.5e-4, 5.0e-4, "freeze_L0_14", True, 1.0, 0.08),
        ("t3b_freeze_L9_23", 1.5e-4, 5.0e-4, "freeze_L9_23", True, 1.0, 0.08),
        ("t3b_lowlr_5x", 3.0e-5, 1.0e-4, "none", True, 1.0, 0.08),
        ("t3b_clip025_warmup25", 1.5e-4, 5.0e-4, "none", True, 0.25, 0.25),
    ]

    agg_keys = (
        "peak_recent20_loss",
        "max_total_grad_norm",
        "max_L0_8_grad_norm",
        "max_shared_L15_23_grad_norm",
        "step5_L0_8_grad_norm",
        "step5_shared_L15_23_grad_norm",
        "max_rms_dL_dh15_via_mlp",
        "end_h15_cos",
        "end_l15_swiglu_abs",
        "start_preact_a_mean",
        "end_preact_a_mean",
        "start_preact_frac_gt0",
        "end_preact_frac_gt0",
        "step12_rc_auroc",
        "step12_c5p_acc",
        "step12_rc_rec",
        "step12_rc_fpr",
        "step12_mean_abs_margin",
    )

    root_data = _load_controls_json()
    out = root_data.setdefault(
        "r7d_warmstart_controls_3seeds_and_receptive_field",
        {"seeds": list(seeds), "n_opt_steps": 12, "max_groups": max_groups, "conditions": {}},
    )
    out["seeds"] = list(seeds)
    out["n_opt_steps"] = 12
    out["max_groups"] = max_groups
    out.setdefault("conditions", {})

    for cname, lr, glr, fmode, init_t2, clip_n, wfrac in specs:
        cond_box = out["conditions"].setdefault(cname, {"summary_mean_std": {}, "per_seed": {}})
        per_seed = cond_box.setdefault("per_seed", {})
        for seed in seeds:
            sk = str(seed)
            if sk in per_seed and "step12_rc_auroc" in per_seed[sk]:
                print(f"[R7d Skip cached] {cname} s={seed}", flush=True)
                continue
            per_seed[sk] = _run_r7d_single(
                cname, seed, lr, glr, fmode, init_t2, clip_n, wfrac, dev_rows, diag_items, groups, probe_ids, prompt_ids, max_groups=max_groups
            )
            _save_controls_json(root_data)
        summary_stats = {}
        for mk in agg_keys:
            vals = np.array([per_seed[str(s)][mk] for s in seeds], dtype=np.float64)
            summary_stats[mk] = {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)}
        cond_box["summary_mean_std"] = summary_stats
        _save_controls_json(root_data)
    return out


def _smollm2_dequantize_2d(gguf_path: Path, n_heads: int = 15, n_kv_heads: int = 5, head_dim: int = 64) -> dict[str, torch.Tensor]:
    """Dequantize Llama-architecture GGUF back to HF SmolLM2 weights with exact RoPE inverse permutation."""
    reader = gguf.GGUFReader(str(gguf_path))
    mapped: dict[str, torch.Tensor] = {}
    half_d = head_dim // 2
    for t in reader.tensors:
        if len(t.shape) != 2 or str(t.tensor_type.name) == "F32":
            continue
        tens = torch.from_numpy(np.array(gguf.dequantize(t.data, t.tensor_type), copy=True)).bfloat16()
        name = t.name
        if name == "token_embd.weight":
            mapped["model.embed_tokens.weight"] = tens
            mapped["lm_head.weight"] = tens
            continue
        if not name.startswith("blk."):
            continue
        parts = name.split(".", 2)
        bid, suffix = int(parts[1]), parts[2]
        prefix = f"model.layers.{bid}"
        if suffix == "ffn_gate.weight":
            mapped[f"{prefix}.mlp.gate_proj.weight"] = tens
        elif suffix == "ffn_up.weight":
            mapped[f"{prefix}.mlp.up_proj.weight"] = tens
        elif suffix == "ffn_down.weight":
            mapped[f"{prefix}.mlp.down_proj.weight"] = tens
        elif suffix == "attn_output.weight":
            mapped[f"{prefix}.self_attn.o_proj.weight"] = tens
        elif suffix == "attn_v.weight":
            mapped[f"{prefix}.self_attn.v_proj.weight"] = tens
        elif suffix == "attn_q.weight":
            in_d = tens.shape[1]
            mapped[f"{prefix}.self_attn.q_proj.weight"] = tens.view(n_heads, half_d, 2, in_d).transpose(1, 2).reshape(-1, in_d).contiguous()
        elif suffix == "attn_k.weight":
            in_d = tens.shape[1]
            mapped[f"{prefix}.self_attn.k_proj.weight"] = tens.view(n_kv_heads, half_d, 2, in_d).transpose(1, 2).reshape(-1, in_d).contiguous()
    return mapped


def _export_smol_hf(src_snap: Path, net_deltas: dict[str, torch.Tensor], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for item in src_snap.iterdir():
        if item.is_file() and item.name != "model.safetensors":
            shutil.copy2(item, out_dir / item.name)
    base_t = load_file(str(src_snap / "model.safetensors"), device="cpu")
    patched = {}
    for k, v in base_t.items():
        if k in net_deltas:
            patched[k] = (v.float() + net_deltas[k].float().cpu()).to(dtype=v.dtype)
        else:
            patched[k] = v
    save_file(patched, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    return out_dir


def run_r7e_smollm2(dev_rows: list[dict[str, Any]], port: int) -> dict[str, Any]:
    """Cross-architecture pure softmax-GQA Llama replication on HuggingFaceTB/SmolLM2-360M-Instruct."""
    TMP_GGUF_DIR.mkdir(parents=True, exist_ok=True)
    smol_snap = Path(
        snapshot_download(
            SMOLLM2_ID,
            local_files_only=True,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
        )
    )
    obs_rows = load_split_rows("observe")

    runner = QwenDeciderRunner(model_path=smol_snap)
    for p in runner.model.parameters():
        p.requires_grad_(False)
    base_cpu = capture_cpu_state_dict(runner.model)
    ab_ids = [runner.letter_token_ids["A"], runner.letter_token_ids["B"]]
    prompt_ids = _select_dev_prompt_ids(runner, dev_rows)
    w_lm = runner.model.lm_head.weight.detach().float().cpu()
    u_dir = (w_lm[runner.letter_token_ids["B"]] - w_lm[runner.letter_token_ids["A"]]).double()
    u_unit_t = u_dir / torch.linalg.norm(u_dir).clamp_min(1e-8)
    _, gains = measure_layer_logit_gains(runner, obs_rows, list(SMOLLM2_LAYERS), u_unit_t)

    # Fit 2-layer closed-form 8-neuron SwiGLU edit at L20 & L28 sequentially (mirroring Track 1 -> Track 2)
    l20, l28 = SMOLLM2_LAYERS
    obs1 = collect_post_patch_observe_state(runner, obs_rows, [l20], include_attn=False)
    prior1 = {k: base_cpu[k].clone() for k in base_cpu if f"model.layers.{l20}.mlp." in k}
    p1 = build_residual_swiglu_patch("smol_l20", runner, obs1, prior1, l20, gains[l20], step_eta=0.75)

    from weight_edit.causal_check import apply_bundle
    apply_bundle(runner.model, base_cpu, [p1])
    obs2 = collect_post_patch_observe_state(runner, obs_rows, [l28], include_attn=False)
    prior2 = {k: base_cpu[k].clone() for k in base_cpu if f"model.layers.{l28}.mlp." in k}
    p2 = build_residual_swiglu_patch("smol_l28", runner, obs2, prior2, l28, gains[l28], step_eta=0.75)
    apply_bundle(runner.model, base_cpu, [p1, p2])

    ed_l20 = list(p1.metadata["quiet_neurons"])
    ed_l28 = list(p2.metadata["quiet_neurons"])
    net_deltas = {**p1.net_deltas(), **p2.net_deltas()}
    full_stats = _weight_delta_stats(net_deltas)

    # Also build same-neuron RMS-matched down_proj-only control
    smol_down = {k: v.clone() for k, v in net_deltas.items() if "down_proj" in k}
    d_st = _weight_delta_stats(smol_down)
    smol_down_rms = {k: v * (full_stats["rms"] / max(1e-12, d_st["rms"])) for k, v in smol_down.items()}

    # Export Base, Sparse Edit, and Down-Only RMS-matched to BF16/K-quant GGUFs
    smol_base_bf16 = TMP_GGUF_DIR / "smol_base-bf16.gguf"
    subprocess.run([sys.executable, str(CONVERT_HF_GGUF), str(smol_snap), "--outfile", str(smol_base_bf16), "--outtype", "bf16"], check=True)
    smol_ed_hf = _export_smol_hf(smol_snap, net_deltas, TMP_GGUF_DIR / "smol_edit_hf")
    smol_ed_bf16 = TMP_GGUF_DIR / "smol_edit-bf16.gguf"
    subprocess.run([sys.executable, str(CONVERT_HF_GGUF), str(smol_ed_hf), "--outfile", str(smol_ed_bf16), "--outtype", "bf16"], check=True)
    shutil.rmtree(smol_ed_hf, ignore_errors=True)

    smol_down_hf = _export_smol_hf(smol_snap, smol_down_rms, TMP_GGUF_DIR / "smol_down_hf")
    smol_down_bf16 = TMP_GGUF_DIR / "smol_down-bf16.gguf"
    smol_down_q4 = TMP_GGUF_DIR / "smol_down-Q4_K_M.gguf"
    subprocess.run([sys.executable, str(CONVERT_HF_GGUF), str(smol_down_hf), "--outfile", str(smol_down_bf16), "--outtype", "bf16"], check=True)
    subprocess.run([str(QUANT_BIN), str(smol_down_bf16), str(smol_down_q4), "Q4_K_M"], check=True)
    shutil.rmtree(smol_down_hf, ignore_errors=True)

    # Sanity check gguf.dequantize BF16 parity on SmolLM2-360M
    pt_margins_ref = np.array([float(runner.score_binary_paired(render_view(r, tok=runner.tokenizer, budget=3200 if r["subset"] == "long_50t" else 2048, include_recap=False).text, mode="plain", question_spec=Q_BLOCK_BINARY)["margin_normal"]) for r in [row for sub in ("real_cf", "consent_5p", "long_50t") for row in dev_rows if row["subset"] == sub]], dtype=np.float64)
    w_smol_bf = _smollm2_dequantize_2d(smol_ed_bf16)
    _apply_dequant_dict(runner.model, base_cpu, w_smol_bf)
    ref_probe = _probe_subset(runner, prompt_ids, ab_ids, (l20, l28))
    sanity_r = float(np.corrcoef(ref_probe["margins"], pt_margins_ref)[0, 1])
    sanity_mae = float(np.mean(np.abs(ref_probe["margins"] - pt_margins_ref)))
    print(f"[R7e SmolLM2] gguf.dequantize BF16 vs PT sanity: r={sanity_r:.6f}, MAE={sanity_mae:.5f}", flush=True)

    # Measure SmolLM2 edit BF16 performance & Jacobians
    d_ff_s = int(runner.model.model.layers[l20].mlp.gate_proj.weight.shape[0])
    unt20 = torch.ones(d_ff_s, dtype=torch.bool)
    unt20[ed_l20] = False
    unt28 = torch.ones(d_ff_s, dtype=torch.bool)
    unt28[ed_l28] = False
    jac20 = _compute_swiglu_jacobian_stats(runner, prompt_ids, l20, ed_l20, unt20)
    jac28 = _compute_swiglu_jacobian_stats(runner, prompt_ids, l28, ed_l28, unt28)
    cjac_edit = _compute_common_edit_agnostic_jacobians(runner, prompt_ids, (l20, l28), l20)
    edit_bf16_perf = _eval_binary_accuracy_140(ref_probe["margins"], dev_rows)

    # Measure down-only RMS-matched control under BF16 & Q4_K_M
    _apply_dequant_dict(runner.model, base_cpu, _smollm2_dequantize_2d(smol_down_bf16))
    ref_down = _probe_subset(runner, prompt_ids, ab_ids, (l20, l28))
    cjac_down = _compute_common_edit_agnostic_jacobians(runner, prompt_ids, (l20, l28), l20)
    _apply_dequant_dict(runner.model, base_cpu, _smollm2_dequantize_2d(smol_down_q4))
    q4_down = _probe_subset(runner, prompt_ids, ab_ids, (l20, l28))
    down_q4_sum = _summarize_q_vs_ref(q4_down, ref_down, ed_l20, ed_l28)

    # Quantize SmolLM2 GGUFs + build surgical allocations + simple HAWQ Fisher ranking on 32 calibration items
    smol_q = {"bf16": smol_ed_bf16}
    for qt in QUANT_TYPES:
        dst = TMP_GGUF_DIR / f"smol_edit-{qt}.gguf"
        subprocess.run([str(QUANT_BIN), str(smol_ed_bf16), str(dst), qt], check=True)
        smol_q[qt] = dst

    smol_base_q4 = TMP_GGUF_DIR / "smol_base-Q4_K_M.gguf"
    subprocess.run([str(QUANT_BIN), str(smol_base_bf16), str(smol_base_q4), "Q4_K_M"], check=True)

    smol_reader = gguf.GGUFReader(str(smol_ed_bf16))
    smol_6ed = {
        f"blk.{l20}.ffn_gate.weight", f"blk.{l20}.ffn_up.weight", f"blk.{l20}.ffn_down.weight",
        f"blk.{l28}.ffn_gate.weight", f"blk.{l28}.ffn_up.weight", f"blk.{l28}.ffn_down.weight",
    }
    map_smol_6x = {t: "bf16" for t in smol_6ed}
    map_smol_up = plan_upstream_overrides([(t.name, str(t.tensor_type.name)) for t in smol_reader.tensors], edit_layer=l20, upstream_quant="Q8_0", include_embd=False)
    map_smol_down = {t.name: "q8_0" for t in smol_reader.tensors if str(t.tensor_type.name) != "F32" and ((_blk_num(t.name) or -1) >= l20 or t.name in ("token_embd.weight", "output.weight"))}

    # HAWQ block ranking for SmolLM2-360M using 32 calibration prompts
    train_rows = _load_train_rows()
    cal_rows = [r for r in train_rows if r["subset"] == "real_cf"][:16] + [r for r in train_rows if r["subset"] == "consent_5p"][:16]
    _apply_dequant_dict(runner.model, base_cpu, w_smol_bf)
    r_sm_q4 = gguf.GGUFReader(str(smol_q["Q4_K_M"]))
    w_bf_np = {t.name: gguf.dequantize(t.data, t.tensor_type) for t in smol_reader.tensors if len(t.shape) == 2 and str(t.tensor_type.name) != "F32"}
    w_q4_np = {t.name: gguf.dequantize(t.data, t.tensor_type) for t in r_sm_q4.tensors if len(t.shape) == 2 and str(t.tensor_type.name) != "F32"}
    r_sm_q8 = gguf.GGUFReader(str(smol_q["Q8_0"]))
    b_q8_sm = {t.name: int(t.n_bytes) for t in r_sm_q8.tensors}
    b_q4_sm = {t.name: int(t.n_bytes) for t in r_sm_q4.tensors}

    # Exact block HAWQ score on SmolLM2 (360M fits easily with checkpointing + grad hooks)
    sm_fisher: dict[int, float] = {b: 0.0 for b in range(32)}
    for p in runner.model.parameters():
        p.requires_grad_(False)
    hk_s = []
    for b in range(32):
        blk = runner.model.model.layers[b]
        for s_g, m in [("ffn_gate", blk.mlp.gate_proj), ("ffn_up", blk.mlp.up_proj), ("ffn_down", blk.mlp.down_proj), ("attn_q", blk.self_attn.q_proj), ("attn_k", blk.self_attn.k_proj), ("attn_v", blk.self_attn.v_proj), ("attn_output", blk.self_attn.o_proj)]:
            m.weight.requires_grad_(True)
            gn = f"blk.{b}.{s_g}.weight"
            esq = torch.from_numpy(np.array(w_bf_np[gn], copy=True)).float() - torch.from_numpy(np.array(w_q4_np[gn], copy=True)).float()
            if s_g in ("attn_q", "attn_k"):
                nh = 15 if s_g == "attn_q" else 5
                esq = esq.view(nh, 32, 2, -1).transpose(1, 2).reshape(esq.shape[0], esq.shape[1])
            esq2 = esq * esq

            def _make_h(bi: int, e2: torch.Tensor):
                def _gh(pm: torch.Tensor) -> None:
                    if pm.grad is not None:
                        sm_fisher[bi] += float((pm.grad.detach().cpu().float().pow(2) * e2).sum().item())
                        pm.grad = None
                return _gh

            hk_s.append(m.weight.register_post_accumulate_grad_hook(_make_h(b, esq2)))

    runner.model.train()
    runner.model.enable_input_require_grads()
    runner.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for r in cal_rows:
        pr = build_plain_prompt(render_view(r, tok=runner.tokenizer, budget=2048, include_recap=False).text, Q_BLOCK_BINARY, False).text
        ids = torch.tensor([runner.tokenizer.encode(pr, add_special_tokens=False)], dtype=torch.long).to(runner.device)
        y_v = 1.0 if r["label"] == "block" else 0.0
        runner.model.zero_grad(set_to_none=True)
        hid = runner.model.model(input_ids=ids, use_cache=False).last_hidden_state[:, -1:, :]
        lg = F.linear(hid.to(dtype=runner.model.lm_head.weight.dtype), runner.model.lm_head.weight[ab_ids])[0, 0, :].float()
        F.binary_cross_entropy_with_logits(lg[1] - lg[0], torch.tensor(y_v, device=runner.device)).backward()
    for h in hk_s:
        h.remove()
    runner.model.gradient_checkpointing_disable()
    runner.model.eval()
    for p in runner.model.parameters():
        p.requires_grad_(False)

    # Build SmolLM2 greedy HAWQ block map matching L0-19 byte cost
    cost_up = sum(max(0, b_q8_sm[tn] - b_q4_sm[tn]) for tn in map_smol_up)
    ranked_b = sorted(range(32), key=lambda bi: sm_fisher[bi], reverse=True)
    hawq_smol_blocks: list[int] = []
    map_smol_hawq: dict[str, str] = {}
    spent_sm = 0
    for bi in ranked_b:
        b_tns = [t.name for t in smol_reader.tensors if _blk_num(t.name) == bi and str(t.tensor_type.name) != "F32"]
        bc = sum(max(0, b_q8_sm[tn] - b_q4_sm[tn]) for tn in b_tns)
        if spent_sm + bc <= cost_up + 256 * 1024 and (spent_sm + bc <= cost_up or not hawq_smol_blocks):
            hawq_smol_blocks.append(bi)
            spent_sm += bc
            for tn in b_tns:
                map_smol_hawq[tn] = "q8_0"

    g_smol_up = _build_surgical_gguf(smol_ed_bf16, TMP_GGUF_DIR / "smol_L0_19_q8.gguf", "Q4_K_M", map_smol_up)
    g_smol_6x = _build_surgical_gguf(smol_ed_bf16, TMP_GGUF_DIR / "smol_6x_bf16.gguf", "Q4_K_M", map_smol_6x)
    g_smol_dn = _build_surgical_gguf(smol_ed_bf16, TMP_GGUF_DIR / "smol_L20_31_q8.gguf", "Q4_K_M", map_smol_down, ["--token-embedding-type", "q8_0", "--output-tensor-type", "q8_0"])
    g_smol_hq = _build_surgical_gguf(smol_ed_bf16, TMP_GGUF_DIR / "smol_hawq_q8.gguf", "Q4_K_M", map_smol_hawq)

    # Probe internal dequantized + within-neuron Taylor + 8-neuron attribution for SmolLM2 Q8..Q4
    dequant_smol: dict[str, Any] = {}
    taylor_smol: dict[str, Any] = {}
    for qt in QUANT_TYPES:
        _apply_dequant_dict(runner.model, base_cpu, _smollm2_dequantize_2d(smol_q[qt]))
        qp = _probe_subset(runner, prompt_ids, ab_ids, (l20, l28))
        dequant_smol[qt] = _summarize_q_vs_ref(qp, ref_probe, ed_l20, ed_l28)
        # Within-neuron directional vs norm-only Taylor on L20
        dh = qp["h1"] - ref_probe["h1"]
        dg_tr = (qp["g1"][:, ed_l20] - ref_probe["g1"][:, ed_l20]).numpy()
        p_dir = np.abs(torch.einsum("nkd,nd->nk", jac20["J_ed"], dh).numpy())
        d_n = np.repeat(torch.linalg.norm(dh, dim=1, keepdim=True).numpy(), 4, axis=1)
        taylor_smol[qt] = {
            "within_neuron_dir_mean": round(float(np.mean([np.corrcoef(p_dir[:, k], np.abs(dg_tr[:, k]))[0, 1] for k in range(4)])), 4),
            "within_neuron_norm_mean": round(float(np.mean([np.corrcoef(d_n[:, k], np.abs(dg_tr[:, k]))[0, 1] for k in range(4)])), 4),
        }

    # Evaluate via llama-server using EXACT HF integer token IDs on all 140 binary select_dev items
    bin_rows = [r for sub in ("real_cf", "consent_5p", "long_50t") for r in dev_rows if r["subset"] == sub]
    bin_prompts_ids = []
    for r in bin_rows:
        pr = build_plain_prompt(render_view(r, tok=runner.tokenizer, budget=3200 if r["subset"] == "long_50t" else 2048, include_recap=False).text, Q_BLOCK_BINARY, False).text
        bin_prompts_ids.append(runner.tokenizer.encode(pr, add_special_tokens=False))

    _apply_dequant_dict(runner.model, base_cpu, _smollm2_dequantize_2d(smol_ed_bf16))
    smol_pt_m = ref_probe["margins"]
    smol_pt_p = 1.0 / (1.0 + np.exp(-np.clip(smol_pt_m, -30.0, 30.0)))

    # Base PT margins for SmolLM2 base Q4 reference
    restore_cpu_state_dict(runner.model, base_cpu)
    smol_base_pt_m = _probe_subset(runner, prompt_ids, ab_ids, (l20, l28))["margins"]
    smol_base_pt_p = 1.0 / (1.0 + np.exp(-np.clip(smol_base_pt_m, -30.0, 30.0)))

    del runner, base_cpu
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n_b = len(bin_rows)
    idx_b = rng.integers(0, n_b, size=(BOOTSTRAP_B, n_b), dtype=np.int32)
    fids = np.arange(n_b) % 5

    smol_srv_targets = [
        ("smol_edit_bf16_srv", True, smol_ed_bf16),
        ("smol_base_Q4_K_M", False, smol_base_q4),
        ("smol_edit_Q8_0", True, smol_q["Q8_0"]),
        ("smol_edit_Q6_K", True, smol_q["Q6_K"]),
        ("smol_edit_Q5_K_M", True, smol_q["Q5_K_M"]),
        ("smol_edit_Q4_K_M", True, smol_q["Q4_K_M"]),
        ("smol_edit_6x_bf16", True, g_smol_6x),
        ("smol_edit_L20_31_embd_q8_0", True, g_smol_dn),
        ("smol_edit_hawq_block_q8_0", True, g_smol_hq),
        ("smol_edit_L0_19_only_q8_0", True, g_smol_up),
    ]
    srv_smol: dict[str, Any] = {}
    for skey, is_ed, gpath in smol_srv_targets:
        base_url = f"http://127.0.0.1:{port}"
        log_f = open(CHECKPOINTS_DIR / "llama_revision_controls.log", "w", encoding="utf-8")
        proc = subprocess.Popen([str(LAUNCH_SCRIPT), str(gpath), str(port)], stdout=log_f, stderr=subprocess.STDOUT, text=True)
        try:
            _wait_server(base_url, proc)
            client = LlamaCompletionClient(base_url=base_url)
            q_ms = []
            for tids in bin_prompts_ids:
                raw = client._post_json("/completion", {"prompt": tids, "n_predict": 1, "n_probs": 32, "temperature": 0.0, "cache_prompt": True})
                scored = score_options_from_logprobs(_extract_token_logprobs(raw), ["A", "B"], yes_idx=1, no_idx=0)
                q_ms.append(float(scored["logit_margin"]))
        finally:
            proc.terminate()
            proc.wait(timeout=10)
            log_f.close()

        qm = np.array(q_ms, dtype=np.float64)
        qp = 1.0 / (1.0 + np.exp(-np.clip(qm, -30.0, 30.0)))
        pm = smol_pt_m if is_ed else smol_base_pt_m
        pp = smol_pt_p if is_ed else smol_base_pt_p
        ag = ((qm >= 0.0) == (pm >= 0.0)).astype(np.float64)

        qm_cal = np.empty_like(qm)
        for fid in range(5):
            qm_cal[fids == fid] = qm[fids == fid] - float(np.mean(qm[fids != fid] - pm[fids != fid]))
        ag_cal = ((qm_cal >= 0.0) == (pm >= 0.0)).astype(np.float64)

        b_st, _ = _bootstrap_distribution({"pt_p": pp, "q_p": qp, "pt_m": pm, "q_m": qm, "agree_200": ag}, idx_b, idx_b)
        lo_ca, hi_ca = np.percentile(ag_cal[idx_b].mean(axis=1), [2.5, 97.5])
        srv_smol[skey] = {
            "gguf_mib": round(gpath.stat().st_size / (1024.0 ** 2), 2),
            "signed_margin_bias": round(float(np.mean(qm - pm)), 4),
            "pearson_r": b_st["pearson_r"],
            "margin_mae": b_st["margin_mae"],
            "raw_agree_140": b_st["agreement_total_200"],
            "cal_oof_agree_140": {"point": round(float(np.mean(ag_cal)), 4), "ci95_lo": round(float(lo_ca), 4), "ci95_hi": round(float(hi_ca), 4)},
            "cal_oof_margin_mae": round(float(np.mean(np.abs(qm_cal - pm))), 4),
        }

    return {
        "model_id": SMOLLM2_ID,
        "architecture": "LlamaForCausalLM (32 pure GQA attention layers, zero DeltaNet)",
        "edited_layers": list(SMOLLM2_LAYERS),
        "edited_neurons": {"l20": ed_l20, "l28": ed_l28},
        "weight_stats": full_stats,
        "bf16_gguf_sanity": {"pearson_r": round(sanity_r, 6), "margin_mae": round(sanity_mae, 5)},
        "bf16_eval_140": edit_bf16_perf,
        "jacobians": {
            "edit_l20_swiglu_mean": jac20["edited_jac_norm_mean"],
            "edit_l20_untouched_mean": jac20["untouched_jac_norm_mean"],
            "edit_l20_ratio": jac20["jac_ratio_vs_untouched"],
            "edit_l28_swiglu_mean": jac28["edited_jac_norm_mean"],
            "edit_l28_ratio": jac28["jac_ratio_vs_untouched"],
            "edit_common": cjac_edit,
            "down_only_rms_matched_common": cjac_down,
        },
        "down_only_rms_matched_q4": down_q4_sum,
        "hawq_ranked_top10_blocks": ranked_b[:10],
        "hawq_selected_blocks": hawq_smol_blocks,
        "dequant_curve": dequant_smol,
        "taylor_within_neuron": taylor_smol,
        "server_configs": srv_smol,
    }


def run_r7e_llama32(dev_rows: list[dict[str, Any]], port: int) -> dict[str, Any]:
    """Cross-family pure softmax-GQA Llama replication on unsloth/Llama-3.2-1B-Instruct (256-aligned)."""
    TMP_GGUF_DIR.mkdir(parents=True, exist_ok=True)
    l32_snap = Path(
        snapshot_download(
            LLAMA32_ID,
            local_files_only=True,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
        )
    )
    obs_rows = load_split_rows("observe")

    runner = QwenDeciderRunner(model_path=l32_snap)
    for p in runner.model.parameters():
        p.requires_grad_(False)
    base_cpu = capture_cpu_state_dict(runner.model)
    ab_ids = [runner.letter_token_ids["A"], runner.letter_token_ids["B"]]
    prompt_ids = _select_dev_prompt_ids(runner, dev_rows)
    w_lm = runner.model.lm_head.weight.detach().float().cpu()
    u_dir = (w_lm[runner.letter_token_ids["B"]] - w_lm[runner.letter_token_ids["A"]]).double()
    u_unit_t = u_dir / torch.linalg.norm(u_dir).clamp_min(1e-8)
    _, gains = measure_layer_logit_gains(runner, obs_rows, list(LLAMA32_LAYERS), u_unit_t)

    l10, l14 = LLAMA32_LAYERS
    obs1 = collect_post_patch_observe_state(runner, obs_rows, [l10], include_attn=False)
    prior1 = {k: base_cpu[k].clone() for k in base_cpu if f"model.layers.{l10}.mlp." in k}
    p1 = build_residual_swiglu_patch("l32_l10", runner, obs1, prior1, l10, gains[l10], step_eta=0.75)

    from weight_edit.causal_check import apply_bundle
    apply_bundle(runner.model, base_cpu, [p1])
    obs2 = collect_post_patch_observe_state(runner, obs_rows, [l14], include_attn=False)
    prior2 = {k: base_cpu[k].clone() for k in base_cpu if f"model.layers.{l14}.mlp." in k}
    p2 = build_residual_swiglu_patch("l32_l14", runner, obs2, prior2, l14, gains[l14], step_eta=0.75)
    apply_bundle(runner.model, base_cpu, [p1, p2])

    ed_l10 = list(p1.metadata["quiet_neurons"])
    ed_l14 = list(p2.metadata["quiet_neurons"])
    net_deltas = {**p1.net_deltas(), **p2.net_deltas()}
    full_stats = _weight_delta_stats(net_deltas)

    # Same-neuron RMS-matched down_proj-only control
    l32_down = {k: v.clone() for k, v in net_deltas.items() if "down_proj" in k}
    d_st = _weight_delta_stats(l32_down)
    l32_down_rms = {k: v * (full_stats["rms"] / max(1e-12, d_st["rms"])) for k, v in l32_down.items()}

    l32_base_bf16 = TMP_GGUF_DIR / "l32_base-bf16.gguf"
    subprocess.run([sys.executable, str(CONVERT_HF_GGUF), str(l32_snap), "--outfile", str(l32_base_bf16), "--outtype", "bf16"], check=True)
    l32_ed_hf = _export_smol_hf(l32_snap, net_deltas, TMP_GGUF_DIR / "l32_edit_hf")
    l32_ed_bf16 = TMP_GGUF_DIR / "l32_edit-bf16.gguf"
    subprocess.run([sys.executable, str(CONVERT_HF_GGUF), str(l32_ed_hf), "--outfile", str(l32_ed_bf16), "--outtype", "bf16"], check=True)
    shutil.rmtree(l32_ed_hf, ignore_errors=True)

    l32_down_hf = _export_smol_hf(l32_snap, l32_down_rms, TMP_GGUF_DIR / "l32_down_hf")
    l32_down_bf16 = TMP_GGUF_DIR / "l32_down-bf16.gguf"
    l32_down_q4 = TMP_GGUF_DIR / "l32_down-Q4_K_M.gguf"
    subprocess.run([sys.executable, str(CONVERT_HF_GGUF), str(l32_down_hf), "--outfile", str(l32_down_bf16), "--outtype", "bf16"], check=True)
    subprocess.run([str(QUANT_BIN), str(l32_down_bf16), str(l32_down_q4), "Q4_K_M"], check=True)
    shutil.rmtree(l32_down_hf, ignore_errors=True)

    # Sanity check gguf.dequantize BF16 vs PyTorch (using n_heads=32, n_kv_heads=8, head_dim=64)
    bin_rows = [r for sub in ("real_cf", "consent_5p", "long_50t") for r in dev_rows if r["subset"] == sub]
    pt_margins_ref = np.array(
        [
            float(
                runner.score_binary_paired(
                    render_view(r, tok=runner.tokenizer, budget=3200 if r["subset"] == "long_50t" else 2048, include_recap=False).text,
                    mode="plain",
                    question_spec=Q_BLOCK_BINARY,
                )["margin_normal"]
            )
            for r in bin_rows
        ],
        dtype=np.float64,
    )
    w_l32_bf = _smollm2_dequantize_2d(l32_ed_bf16, n_heads=32, n_kv_heads=8, head_dim=64)
    _apply_dequant_dict(runner.model, base_cpu, w_l32_bf)
    ref_probe = _probe_subset(runner, prompt_ids, ab_ids, (l10, l14))
    sanity_r = float(np.corrcoef(ref_probe["margins"], pt_margins_ref)[0, 1])
    sanity_mae = float(np.mean(np.abs(ref_probe["margins"] - pt_margins_ref)))
    print(f"[R7e Llama-3.2-1B] gguf.dequantize BF16 vs PT sanity: r={sanity_r:.6f}, MAE={sanity_mae:.5f}", flush=True)

    # Edit BF16 performance & Jacobians
    d_ff_l = int(runner.model.model.layers[l10].mlp.gate_proj.weight.shape[0])
    unt10 = torch.ones(d_ff_l, dtype=torch.bool)
    unt10[ed_l10] = False
    unt14 = torch.ones(d_ff_l, dtype=torch.bool)
    unt14[ed_l14] = False
    jac10 = _compute_swiglu_jacobian_stats(runner, prompt_ids, l10, ed_l10, unt10)
    jac14 = _compute_swiglu_jacobian_stats(runner, prompt_ids, l14, ed_l14, unt14)
    torch.cuda.empty_cache()
    cjac_edit = _compute_common_edit_agnostic_jacobians(runner, prompt_ids, (l10, l14), l10)
    edit_bf16_perf = _eval_binary_accuracy_140(ref_probe["margins"], dev_rows)

    # Down-only RMS-matched control under BF16 & Q4_K_M
    _apply_dequant_dict(runner.model, base_cpu, _smollm2_dequantize_2d(l32_down_bf16, n_heads=32, n_kv_heads=8, head_dim=64))
    ref_down = _probe_subset(runner, prompt_ids, ab_ids, (l10, l14))
    torch.cuda.empty_cache()
    cjac_down = _compute_common_edit_agnostic_jacobians(runner, prompt_ids, (l10, l14), l10)
    _apply_dequant_dict(runner.model, base_cpu, _smollm2_dequantize_2d(l32_down_q4, n_heads=32, n_kv_heads=8, head_dim=64))
    q4_down = _probe_subset(runner, prompt_ids, ab_ids, (l10, l14))
    down_q4_sum = _summarize_q_vs_ref(q4_down, ref_down, ed_l10, ed_l14)

    # Quantize Q8_0..Q4_K_M + verify zero fallback warnings on Q4_K_M
    l32_q = {"bf16": l32_ed_bf16}
    q4_fallbacks: list[str] = []
    for qt in QUANT_TYPES:
        dst = TMP_GGUF_DIR / f"l32_edit-{qt}.gguf"
        res_q = subprocess.run([str(QUANT_BIN), str(l32_ed_bf16), str(dst), qt], check=True, capture_output=True, text=True)
        if qt == "Q4_K_M":
            q4_fallbacks = [ln for ln in (res_q.stdout + res_q.stderr).splitlines() if "fallback" in ln.lower()]
        l32_q[qt] = dst

    l32_base_q4 = TMP_GGUF_DIR / "l32_base-Q4_K_M.gguf"
    subprocess.run([str(QUANT_BIN), str(l32_base_bf16), str(l32_base_q4), "Q4_K_M"], check=True)

    l32_reader = gguf.GGUFReader(str(l32_ed_bf16))
    l32_6ed = {
        f"blk.{l10}.ffn_gate.weight", f"blk.{l10}.ffn_up.weight", f"blk.{l10}.ffn_down.weight",
        f"blk.{l14}.ffn_gate.weight", f"blk.{l14}.ffn_up.weight", f"blk.{l14}.ffn_down.weight",
    }
    map_l32_6x = {t: "bf16" for t in l32_6ed}
    map_l32_up = plan_upstream_overrides([(t.name, str(t.tensor_type.name)) for t in l32_reader.tensors], edit_layer=l10, upstream_quant="Q8_0", include_embd=False)
    map_l32_down = {t.name: "q8_0" for t in l32_reader.tensors if str(t.tensor_type.name) != "F32" and ((_blk_num(t.name) or -1) >= l10 or t.name in ("token_embd.weight", "output.weight"))}

    # HAWQ block ranking across blk.0..15 using 32 calibration prompts (budget=1024, embed frozen, CPU grad offload)
    train_rows = _load_train_rows()
    cal_rows = [r for r in train_rows if r["subset"] == "real_cf"][:16] + [r for r in train_rows if r["subset"] == "consent_5p"][:16]
    _apply_dequant_dict(runner.model, base_cpu, w_l32_bf)
    r_l32_q4 = gguf.GGUFReader(str(l32_q["Q4_K_M"]))
    w_bf_np = {t.name: gguf.dequantize(t.data, t.tensor_type) for t in l32_reader.tensors if len(t.shape) == 2 and str(t.tensor_type.name) != "F32"}
    w_q4_np = {t.name: gguf.dequantize(t.data, t.tensor_type) for t in r_l32_q4.tensors if len(t.shape) == 2 and str(t.tensor_type.name) != "F32"}
    r_l32_q8 = gguf.GGUFReader(str(l32_q["Q8_0"]))
    b_q8_l32 = {t.name: int(t.n_bytes) for t in r_l32_q8.tensors}
    b_q4_l32 = {t.name: int(t.n_bytes) for t in r_l32_q4.tensors}

    l32_fisher: dict[int, float] = {b: 0.0 for b in range(16)}
    for p in runner.model.parameters():
        p.requires_grad_(False)
    hk_l = []
    for b in range(16):
        blk = runner.model.model.layers[b]
        for s_g, m in [("ffn_gate", blk.mlp.gate_proj), ("ffn_up", blk.mlp.up_proj), ("ffn_down", blk.mlp.down_proj), ("attn_q", blk.self_attn.q_proj), ("attn_k", blk.self_attn.k_proj), ("attn_v", blk.self_attn.v_proj), ("attn_output", blk.self_attn.o_proj)]:
            m.weight.requires_grad_(True)
            gn = f"blk.{b}.{s_g}.weight"
            esq = torch.from_numpy(np.array(w_bf_np[gn], copy=True)).float() - torch.from_numpy(np.array(w_q4_np[gn], copy=True)).float()
            if s_g in ("attn_q", "attn_k"):
                nh = 32 if s_g == "attn_q" else 8
                esq = esq.view(nh, 32, 2, -1).transpose(1, 2).reshape(esq.shape[0], esq.shape[1])
            esq2 = esq * esq

            def _make_hl(bi: int, e2: torch.Tensor):
                def _gh(pm: torch.Tensor) -> None:
                    if pm.grad is not None:
                        l32_fisher[bi] += float((pm.grad.detach().cpu().float().pow(2) * e2).sum().item())
                        pm.grad = None
                return _gh

            hk_l.append(m.weight.register_post_accumulate_grad_hook(_make_hl(b, esq2)))
    del w_bf_np, w_q4_np

    torch.cuda.empty_cache()
    runner.model.train()
    runner.model.enable_input_require_grads()
    runner.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for r in cal_rows:
        pr = build_plain_prompt(render_view(r, tok=runner.tokenizer, budget=1024, include_recap=False).text, Q_BLOCK_BINARY, False).text
        enc = runner.tokenizer.encode(pr, add_special_tokens=False)[-1024:]
        ids = torch.tensor([enc], dtype=torch.long).to(runner.device)
        y_v = 1.0 if r["label"] == "block" else 0.0
        runner.model.zero_grad(set_to_none=True)
        hid = runner.model.model(input_ids=ids, use_cache=False).last_hidden_state[:, -1:, :]
        lg = F.linear(hid.to(dtype=runner.model.lm_head.weight.dtype), runner.model.lm_head.weight[ab_ids])[0, 0, :].float()
        loss = F.binary_cross_entropy_with_logits(lg[1] - lg[0], torch.tensor(y_v, device=runner.device))
        loss.backward()
        del hid, lg, loss
    for h in hk_l:
        h.remove()
    runner.model.gradient_checkpointing_disable()
    runner.model.eval()
    for p in runner.model.parameters():
        p.requires_grad_(False)
    runner.model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()

    cost_up = sum(max(0, b_q8_l32[tn] - b_q4_l32[tn]) for tn in map_l32_up)
    ranked_b = sorted(range(16), key=lambda bi: l32_fisher[bi], reverse=True)
    hawq_l32_blocks: list[int] = []
    map_l32_hawq: dict[str, str] = {}
    spent_l32 = 0
    for bi in ranked_b:
        b_tns = [t.name for t in l32_reader.tensors if _blk_num(t.name) == bi and str(t.tensor_type.name) != "F32"]
        bc = sum(max(0, b_q8_l32[tn] - b_q4_l32[tn]) for tn in b_tns)
        if spent_l32 + bc <= cost_up + 512 * 1024 and (spent_l32 + bc <= cost_up or not hawq_l32_blocks):
            hawq_l32_blocks.append(bi)
            spent_l32 += bc
            for tn in b_tns:
                map_l32_hawq[tn] = "q8_0"

    g_l32_up = _build_surgical_gguf(l32_ed_bf16, TMP_GGUF_DIR / "l32_L0_9_q8.gguf", "Q4_K_M", map_l32_up)
    g_l32_6x = _build_surgical_gguf(l32_ed_bf16, TMP_GGUF_DIR / "l32_6x_bf16.gguf", "Q4_K_M", map_l32_6x)
    g_l32_dn = _build_surgical_gguf(l32_ed_bf16, TMP_GGUF_DIR / "l32_L10_15_q8.gguf", "Q4_K_M", map_l32_down, ["--token-embedding-type", "q8_0", "--output-tensor-type", "q8_0"])
    g_l32_hq = _build_surgical_gguf(l32_ed_bf16, TMP_GGUF_DIR / "l32_hawq_q8.gguf", "Q4_K_M", map_l32_hawq)

    # Probe internal dequantized + non-pooled within-neuron Taylor (directional vs norm-only vs iso vs random rot)
    rng_rot = np.random.default_rng(54321)
    rot_mats_l32 = []
    for _ in range(10):
        q_mat, _ = np.linalg.qr(rng_rot.standard_normal((2048, 2048)).astype(np.float32))
        rot_mats_l32.append(torch.from_numpy(q_mat))

    dequant_l32: dict[str, Any] = {}
    taylor_l32: dict[str, Any] = {}
    for qt in QUANT_TYPES:
        _apply_dequant_dict(runner.model, base_cpu, _smollm2_dequantize_2d(l32_q[qt], n_heads=32, n_kv_heads=8, head_dim=64))
        qp = _probe_subset(runner, prompt_ids, ab_ids, (l10, l14))
        dequant_l32[qt] = _summarize_q_vs_ref(qp, ref_probe, ed_l10, ed_l14)
        dh = qp["h1"] - ref_probe["h1"]
        dg_tr = (qp["g1"][:, ed_l10] - ref_probe["g1"][:, ed_l10]).numpy()
        pred_dir = torch.einsum("nkd,nd->nk", jac10["J_ed"], dh).numpy()
        abs_dir = np.abs(pred_dir)
        abs_tr = np.abs(dg_tr)
        d_n = np.repeat(torch.linalg.norm(dh, dim=1, keepdim=True).numpy(), 4, axis=1)
        j_n = torch.linalg.norm(jac10["J_ed"], dim=2).numpy()
        iso_m = j_n * d_n
        rot_w = []
        for R in rot_mats_l32:
            dh_r = dh @ R.T
            pr_r = np.abs(torch.einsum("nkd,nd->nk", jac10["J_ed"], dh_r).numpy())
            rot_w.append(float(np.mean([np.corrcoef(pr_r[:, k], abs_tr[:, k])[0, 1] for k in range(4)])))
        taylor_l32[qt] = {
            "within_neuron_dir_abs_mean": round(float(np.mean([np.corrcoef(abs_dir[:, k], abs_tr[:, k])[0, 1] for k in range(4)])), 4),
            "within_neuron_dir_signed_mean": round(float(np.mean([np.corrcoef(pred_dir[:, k], dg_tr[:, k])[0, 1] for k in range(4)])), 4),
            "within_neuron_norm_mean": round(float(np.mean([np.corrcoef(d_n[:, k], abs_tr[:, k])[0, 1] for k in range(4)])), 4),
            "within_neuron_iso_gain_x_norm_mean": round(float(np.mean([np.corrcoef(iso_m[:, k], abs_tr[:, k])[0, 1] for k in range(4)])), 4),
            "within_neuron_random_rot_mean": round(float(np.mean(rot_w)), 4),
        }

    # Evaluate via llama-server using EXACT HF integer token IDs on all 140 binary select_dev items
    bin_prompts_ids = []
    for r in bin_rows:
        pr = build_plain_prompt(render_view(r, tok=runner.tokenizer, budget=3200 if r["subset"] == "long_50t" else 2048, include_recap=False).text, Q_BLOCK_BINARY, False).text
        bin_prompts_ids.append(runner.tokenizer.encode(pr, add_special_tokens=False))

    _apply_dequant_dict(runner.model, base_cpu, w_l32_bf)
    l32_pt_m = ref_probe["margins"]
    l32_pt_p = 1.0 / (1.0 + np.exp(-np.clip(l32_pt_m, -30.0, 30.0)))

    restore_cpu_state_dict(runner.model, base_cpu)
    l32_base_pt_m = _probe_subset(runner, prompt_ids, ab_ids, (l10, l14))["margins"]
    l32_base_pt_p = 1.0 / (1.0 + np.exp(-np.clip(l32_base_pt_m, -30.0, 30.0)))

    del runner, base_cpu
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n_b = len(bin_rows)
    idx_b = rng.integers(0, n_b, size=(BOOTSTRAP_B, n_b), dtype=np.int32)
    fids = np.arange(n_b) % 5

    l32_srv_targets = [
        ("llama32_edit_bf16_srv", True, l32_ed_bf16),
        ("llama32_base_Q4_K_M", False, l32_base_q4),
        ("llama32_edit_Q8_0", True, l32_q["Q8_0"]),
        ("llama32_edit_Q6_K", True, l32_q["Q6_K"]),
        ("llama32_edit_Q5_K_M", True, l32_q["Q5_K_M"]),
        ("llama32_edit_Q4_K_M", True, l32_q["Q4_K_M"]),
        ("llama32_edit_6x_bf16", True, g_l32_6x),
        ("llama32_edit_L10_15_embd_q8_0", True, g_l32_dn),
        ("llama32_edit_hawq_block_q8_0", True, g_l32_hq),
        ("llama32_edit_L0_9_only_q8_0", True, g_l32_up),
    ]
    srv_l32: dict[str, Any] = {}
    for skey, is_ed, gpath in l32_srv_targets:
        print(f"[R7e Llama-3.2 Server] {skey} ({gpath.name})...", flush=True)
        base_url = f"http://127.0.0.1:{port}"
        log_f = open(CHECKPOINTS_DIR / "llama_revision_controls.log", "w", encoding="utf-8")
        proc = subprocess.Popen([str(LAUNCH_SCRIPT), str(gpath), str(port)], stdout=log_f, stderr=subprocess.STDOUT, text=True)
        try:
            _wait_server(base_url, proc)
            client = LlamaCompletionClient(base_url=base_url)
            q_ms = []
            for tids in bin_prompts_ids:
                raw = client._post_json("/completion", {"prompt": tids, "n_predict": 1, "n_probs": 32, "temperature": 0.0, "cache_prompt": True})
                scored = score_options_from_logprobs(_extract_token_logprobs(raw), ["A", "B"], yes_idx=1, no_idx=0)
                q_ms.append(float(scored["logit_margin"]))
        finally:
            proc.terminate()
            proc.wait(timeout=10)
            log_f.close()

        qm = np.array(q_ms, dtype=np.float64)
        qp = 1.0 / (1.0 + np.exp(-np.clip(qm, -30.0, 30.0)))
        pm = l32_pt_m if is_ed else l32_base_pt_m
        pp = l32_pt_p if is_ed else l32_base_pt_p
        ag = ((qm >= 0.0) == (pm >= 0.0)).astype(np.float64)

        qm_cal = np.empty_like(qm)
        for fid in range(5):
            qm_cal[fids == fid] = qm[fids == fid] - float(np.mean(qm[fids != fid] - pm[fids != fid]))
        ag_cal = ((qm_cal >= 0.0) == (pm >= 0.0)).astype(np.float64)

        b_st, _ = _bootstrap_distribution({"pt_p": pp, "q_p": qp, "pt_m": pm, "q_m": qm, "agree_200": ag}, idx_b, idx_b)
        lo_ca, hi_ca = np.percentile(ag_cal[idx_b].mean(axis=1), [2.5, 97.5])
        srv_l32[skey] = {
            "gguf_mib": round(gpath.stat().st_size / (1024.0 ** 2), 2),
            "signed_margin_bias": round(float(np.mean(qm - pm)), 4),
            "pearson_r": b_st["pearson_r"],
            "margin_mae": b_st["margin_mae"],
            "raw_agree_140": b_st["agreement_total_200"],
            "cal_oof_agree_140": {"point": round(float(np.mean(ag_cal)), 4), "ci95_lo": round(float(lo_ca), 4), "ci95_hi": round(float(hi_ca), 4)},
            "cal_oof_margin_mae": round(float(np.mean(np.abs(qm_cal - pm))), 4),
        }

    return {
        "model_id": LLAMA32_ID,
        "architecture": "LlamaForCausalLM (16 pure GQA attention layers, zero DeltaNet, 256-aligned)",
        "edited_layers": list(LLAMA32_LAYERS),
        "edited_neurons": {"l10": ed_l10, "l14": ed_l14},
        "weight_stats": full_stats,
        "q4_k_m_fallback_warnings": q4_fallbacks,
        "bf16_gguf_sanity": {"pearson_r": round(sanity_r, 6), "margin_mae": round(sanity_mae, 5)},
        "bf16_eval_140": edit_bf16_perf,
        "jacobians": {
            "edit_l10_swiglu_mean": jac10["edited_jac_norm_mean"],
            "edit_l10_untouched_mean": jac10["untouched_jac_norm_mean"],
            "edit_l10_ratio": jac10["jac_ratio_vs_untouched"],
            "edit_l14_swiglu_mean": jac14["edited_jac_norm_mean"],
            "edit_l14_ratio": jac14["jac_ratio_vs_untouched"],
            "edit_common": cjac_edit,
            "down_only_rms_matched_common": cjac_down,
        },
        "down_only_rms_matched_q4": down_q4_sum,
        "hawq_ranked_blocks": ranked_b,
        "hawq_selected_blocks": hawq_l32_blocks,
        "dequant_curve": dequant_l32,
        "taylor_within_neuron": taylor_l32,
        "server_configs": srv_l32,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Revision Suite R7 stages incrementally.")
    ap.add_argument("--only", nargs="+", choices=["r7a", "r7b", "r7c", "r7d", "r7e"], default=["r7a", "r7b", "r7c", "r7d", "r7e"])
    ap.add_argument("--llama-port", type=int, default=8094)
    ap.add_argument("--keep-tmp", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    dev_rows = load_split_rows("select_dev")
    data = _load_controls_json()
    wanted = set(args.only)

    if "r7a" in wanted:
        data["r7a_p2_ablation_doseresponse_and_cancellation"] = run_r7a(dev_rows)
        _save_controls_json(data)
    if "r7b" in wanted or "r7c" in wanted:
        r7b, r7c = run_r7b_and_r7c(dev_rows, args.llama_port)
        if "r7b" in wanted:
            data["r7b_hawq_awq_sensitivity_baselines"] = r7b
        if "r7c" in wanted:
            data["r7c_calibration_asymmetric_risk_and_taylor_controls"] = r7c
        _save_controls_json(data)
    if "r7e" in wanted:
        data["r7e_second_arch_llama32_1b"] = run_r7e_llama32(dev_rows, args.llama_port)
        _save_controls_json(data)
    if "r7d" in wanted:
        data["r7d_warmstart_controls_3seeds_and_receptive_field"] = run_r7d(dev_rows)
        _save_controls_json(data)

    if not args.keep_tmp:
        _clean_tmp_ggufs()
    print(f"[R7 Complete] Stages {sorted(wanted)} saved to {ROOT / 'experiments' / 'paper_revision_controls.json'} in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
