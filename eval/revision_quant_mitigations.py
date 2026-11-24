"""Revision Suite R2: K-Quant Diagnostics, Selective Mixed-Precision, and Norm-Matched WSP Dispersion.

Merges results cleanly into `experiments/paper_revision_controls.json`:
  - `kquant_diagnostics` (R2a): Exact GGML per-tensor types via GGUFReader + margin-collapse statistics
    explaining K-quant degradation and the Q4_K_M vs Q5_K_M non-monotonic MAE.
  - `mixed_precision_mitigation` (R2b): Selective preservation of Track 2's 6 (or 4) edited tensors
    at `bf16` or `q8_0` inside `Q4_K_M` / `Q5_K_M` / `Q6_K` GGUFs via `llama-quantize --tensor-type`.
  - `multi_neuron_dispersion` (R2c): Norm-matched dispersion of Track 2's closed-form SwiGLU edits
    across K=16 (m=4) and K=64 (m=16) quiet neurons per layer, evaluated in bf16 and Q8_0..Q4_K_M.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
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

import numpy as np
import torch
from gguf import GGUFReader
from huggingface_hub import snapshot_download

from automode.questions import Q_BLOCK_BINARY, build_decider_prompt
from automode.view import render_view
from eval.evaluate import compute_binary_metrics, compute_ece, compute_proxy_score, evaluate_runner, load_split_rows
from eval.eval_paper_suite import _analyze_weight_forensics_against_base, _load_hf_safetensors_dir
from eval.mechanism_analysis import (
    QUANT_BIN,
    QUANT_TYPES,
    _collect_pt_select_dev_ref,
    _wait_server,
)
from eval.model_runner import QwenDeciderRunner, capture_cpu_state_dict, restore_cpu_state_dict
from runtime.client import LlamaCompletionClient
from runtime.export_hf import export_patched_hf, map_causal_to_hf_key
from weight_edit.causal_check import PatchBundle, WeightPatch, apply_bundle, save_patch_bundle
from weight_edit.context_edit import _snapshot_gate_layer_weights, fit_joint_contrastive_readout, solve_closed_form_ridge
from weight_edit.wsp_runner import (
    _measure_layer_set_gains,
    _select_unused_quiet_neurons,
    collect_post_patch_observe_state,
    fit_residual_boosting_readout,
)

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = ROOT / "checkpoints"
EXPERIMENTS_DIR = ROOT / "experiments"
CONVERT_HF_GGUF = ROOT / "third_party" / "llama.cpp" / "convert_hf_to_gguf.py"
LAUNCH_SCRIPT = ROOT / "runtime" / "launch_llama_server.sh"
CONTROLS_JSON = EXPERIMENTS_DIR / "paper_revision_controls.json"
TMP_GGUF_DIR = Path("/tmp/revision_ggufs")

TRACK2_GGUF_TENSORS = (
    "blk.15.ffn_gate.weight",
    "blk.15.ffn_up.weight",
    "blk.15.ffn_down.weight",
    "blk.22.ffn_gate.weight",
    "blk.22.ffn_up.weight",
    "blk.22.ffn_down.weight",
)


def _load_controls_json() -> dict[str, Any]:
    if CONTROLS_JSON.exists():
        return json.loads(CONTROLS_JSON.read_text(encoding="utf-8"))
    return {}


def _save_controls_json(payload: dict[str, Any]) -> None:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    CONTROLS_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_gguf_tensor_types(gguf_path: Path, tensor_names: tuple[str, ...] = TRACK2_GGUF_TENSORS) -> dict[str, str]:
    reader = GGUFReader(str(gguf_path))
    tset = set(tensor_names)
    out: dict[str, str] = {}
    for t in reader.tensors:
        if t.name in tset:
            out[t.name] = str(t.tensor_type.name)
    return {k: out[k] for k in tensor_names if k in out}


def eval_gguf_select_dev_detailed(gguf_path: Path, ref: dict[str, Any], port: int) -> dict[str, Any]:
    """Evaluate a GGUF on select_dev (n=200) vs PyTorch bf16 reference, including margin collapse stats."""
    base_url = f"http://127.0.0.1:{port}"
    log_f = open(CHECKPOINTS_DIR / "llama_revision_controls.log", "w", encoding="utf-8")
    proc = subprocess.Popen(
        [str(LAUNCH_SCRIPT), str(gguf_path), str(port)],
        stdout=log_f,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_server(base_url, proc)
        client = LlamaCompletionClient(base_url=base_url, second_reading_band=(0.30, 0.70))

        lat_ms: list[float] = []
        q_bin: list[dict[str, Any]] = []
        for item in ref["binary"]:
            sc = client.score_binary_paired(item["ctx"], force_paired=True)
            lat_ms.append(float(sc["latencies_ms"][0]))
            pn = float(sc["p_block_normal"])
            pr = float(sc["p_block_reversed"])
            mn = float(sc["margin_normal"])
            mr = float(sc["margin_reversed"])
            q_bin.append(
                {
                    "subset": item["subset"],
                    "pattern": item["pattern"],
                    "y": item["y"],
                    "pt_p": item["pt_p_block"],
                    "pt_m": item["pt_margin"],
                    "pt_pred": item["pt_pred_norm"],
                    "q_p": pn,
                    "q_m": mn,
                    "q_mr": mr,
                    "q_pred": "block" if pn >= 0.5 else "allow",
                    "q_flipped": (pn >= 0.5) != (pr >= 0.5),
                    "q_abs_diff": abs(mn - mr),
                }
            )

        q_rep: list[dict[str, Any]] = []
        for item in ref["replay"]:
            sc = client.score_question(item["context"], item["question_spec"])
            lat_ms.append(float(sc["latency_ms"]))
            pred_i = int(sc["pred_index"])
            probs = list(sc["option_probs"])
            q_rep.append(
                {
                    "agree": int(pred_i == item["pt_pred_index"]),
                    "correct": int(pred_i == item["gold_index"]),
                    "top_prob": float(probs[pred_i]),
                }
            )

        by_sub: dict[str, list[dict[str, Any]]] = {"real_cf": [], "consent_5p": [], "long_50t": []}
        for x in q_bin:
            by_sub[x["subset"]].append(x)

        rc_l = by_sub["real_cf"]
        c5_l = by_sub["consent_5p"]
        lg_l = by_sub["long_50t"]

        b_rc = compute_binary_metrics([x["q_p"] for x in rc_l], [x["y"] for x in rc_l], [x["q_m"] for x in rc_l])
        b_rc["order_flip_rate"] = round(sum(1 for x in rc_l if x["q_flipped"]) / len(rc_l), 4)
        b_rc["mean_abs_order_diff"] = round(float(np.mean([x["q_abs_diff"] for x in rc_l])), 4)

        b_c5 = compute_binary_metrics([x["q_p"] for x in c5_l], [x["y"] for x in c5_l], [x["q_m"] for x in c5_l])
        p2s = [x for x in c5_l if x["pattern"] == "p2_exact_consent"]
        p3s = [x for x in c5_l if x["pattern"] == "p3_target_mismatch"]
        p5as = [x for x in c5_l if x["pattern"] == "p5a_blocked_approved_same"]
        p5bs = [x for x in c5_l if x["pattern"] == "p5b_blocked_swapped_target"]
        c5_sum = {
            "overall_acc": b_c5["acc_at_05"],
            "margin_sep_p3_minus_p2": round(float(np.mean([x["q_m"] for x in p3s]) - np.mean([x["q_m"] for x in p2s])), 4),
            "margin_sep_p5b_minus_p5a": round(float(np.mean([x["q_m"] for x in p5bs]) - np.mean([x["q_m"] for x in p5as])), 4),
            "mean_abs_order_diff": round(float(np.mean([x["q_abs_diff"] for x in c5_l])), 4),
            "binary_metrics": b_c5,
        }
        lg_acc = round(sum(1 for x in lg_l if (x["q_pred"] == "block") == (x["y"] == 1)) / len(lg_l), 4)
        rep_sum = {
            "overall_acc": round(sum(x["correct"] for x in q_rep) / len(q_rep), 4),
            "top_label_ece": round(compute_ece([x["top_prob"] for x in q_rep], [x["correct"] for x in q_rep]), 4),
        }
        proxy = compute_proxy_score({"real_cf": b_rc, "consent_5p": c5_sum, "decider_replay": rep_sum})

        pt_p = np.array([x["pt_p"] for x in q_bin], dtype=np.float64)
        q_p = np.array([x["q_p"] for x in q_bin], dtype=np.float64)
        pt_m = np.array([x["pt_m"] for x in q_bin], dtype=np.float64)
        q_m = np.array([x["q_m"] for x in q_bin], dtype=np.float64)
        ys = np.array([x["y"] for x in q_bin], dtype=np.int64)

        p_err = np.abs(pt_p - q_p)
        m_err = np.abs(pt_m - q_m)
        pearson = float(np.corrcoef(pt_p, q_p)[0, 1]) if np.std(q_p) > 1e-12 and np.std(pt_p) > 1e-12 else 0.0
        agree_bin = sum(1 for x in q_bin if x["pt_pred"] == x["q_pred"]) / len(q_bin)
        agree_rep = sum(x["agree"] for x in q_rep) / len(q_rep)
        agree_tot = (sum(1 for x in q_bin if x["pt_pred"] == x["q_pred"]) + sum(x["agree"] for x in q_rep)) / (len(q_bin) + len(q_rep))

        mae_by_sub = {
            sub: round(float(np.mean([abs(x["pt_p"] - x["q_p"]) for x in lst])), 5)
            for sub, lst in by_sub.items()
        }
        rc_allow_mae = round(float(np.mean([abs(x["pt_p"] - x["q_p"]) for x in rc_l if x["y"] == 0])), 5)
        rc_block_mae = round(float(np.mean([abs(x["pt_p"] - x["q_p"]) for x in rc_l if x["y"] == 1])), 5)

        return {
            "gguf_name": gguf_path.name,
            "gguf_mib": round(gguf_path.stat().st_size / (1024.0 ** 2), 2),
            "p_block_mae_normal": round(float(np.mean(p_err)), 5),
            "margin_mae_normal": round(float(np.mean(m_err)), 5),
            "pearson_r": round(pearson, 6),
            "agreement_binary_normal": round(agree_bin, 4),
            "agreement_replay": round(agree_rep, 4),
            "agreement_total_200": round(agree_tot, 4),
            "real_cf_auroc": b_rc["auroc"],
            "real_cf_rec": b_rc["recall_at_05"],
            "real_cf_fpr": b_rc["fpr_at_05"],
            "real_cf_flip": b_rc["order_flip_rate"],
            "consent_5p_acc": b_c5["acc_at_05"],
            "long_50t_acc": lg_acc,
            "decider_replay_acc": rep_sum["overall_acc"],
            "decider_replay_ece": rep_sum["top_label_ece"],
            "proxy_score": proxy["proxy_score"],
            "latency_1pass_p50_ms": round(float(np.percentile(lat_ms, 50)), 2),
            "margin_stats": {
                "pt_mean_abs_margin": round(float(np.mean(np.abs(pt_m))), 4),
                "pt_mean_margin_block": round(float(np.mean(pt_m[ys == 1])), 4),
                "pt_mean_margin_allow": round(float(np.mean(pt_m[ys == 0])), 4),
                "q_mean_margin": round(float(np.mean(q_m)), 4),
                "q_mean_abs_margin": round(float(np.mean(np.abs(q_m))), 4),
                "q_std_margin": round(float(np.std(q_m)), 4),
                "q_mean_margin_block": round(float(np.mean(q_m[ys == 1])), 4),
                "q_mean_margin_allow": round(float(np.mean(q_m[ys == 0])), 4),
                "mean_signed_margin_shift": round(float(np.mean(q_m - pt_m)), 4),
                "mean_signed_p_shift": round(float(np.mean(q_p - pt_p)), 5),
                "frac_q_abs_margin_lt_0_5": round(float(np.mean(np.abs(q_m) < 0.5)), 4),
                "mae_by_subset": mae_by_sub,
                "real_cf_allow_p_mae": rc_allow_mae,
                "real_cf_block_p_mae": rc_block_mae,
            },
        }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log_f.close()


def run_r2a_kquant_diagnostics(port: int) -> dict[str, Any]:
    print("[R2a] Running K-quant superblock tensor mapping & margin-collapse diagnostics...", flush=True)
    dev_rows = load_split_rows("select_dev")
    refs = {
        "track2_wsp": _collect_pt_select_dev_ref(CHECKPOINTS_DIR / "track2_best_deltas.pt", dev_rows),
        "track3_soup": _collect_pt_select_dev_ref(CHECKPOINTS_DIR / "track3_final_best_deltas.pt", dev_rows),
    }
    prefixes = {"track2_wsp": "wsp-automode-0.8b", "track3_soup": "automode-0.8b-track3"}
    out: dict[str, Any] = {"track2_wsp": {}, "track3_soup": {}}
    for mk in ("track2_wsp", "track3_soup"):
        for qtype in QUANT_TYPES:
            gp = CHECKPOINTS_DIR / f"{prefixes[mk]}-{qtype}.gguf"
            stats = eval_gguf_select_dev_detailed(gp, refs[mk], port)
            stats["track2_tensor_types"] = read_gguf_tensor_types(gp)
            out[mk][qtype] = stats
    return out


def run_r2b_mixed_precision_mitigation(port: int) -> dict[str, Any]:
    print("[R2b] Running selective mixed-precision `--tensor-type` mitigation on Track 2...", flush=True)
    TMP_GGUF_DIR.mkdir(parents=True, exist_ok=True)
    dev_rows = load_split_rows("select_dev")
    t2_ref = _collect_pt_select_dev_ref(CHECKPOINTS_DIR / "track2_best_deltas.pt", dev_rows)
    bf16_src = CHECKPOINTS_DIR / "wsp-automode-0.8b-bf16.gguf"

    base_sizes = {
        qt: round((CHECKPOINTS_DIR / f"wsp-automode-0.8b-{qt}.gguf").stat().st_size / (1024.0 ** 2), 2)
        for qt in ("Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0")
    }

    configs = [
        ("Q4_K_M_6x_bf16", "Q4_K_M", list(TRACK2_GGUF_TENSORS), "bf16"),
        (
            "Q4_K_M_4x_bf16_gateup",
            "Q4_K_M",
            [
                "blk.15.ffn_gate.weight",
                "blk.15.ffn_up.weight",
                "blk.22.ffn_gate.weight",
                "blk.22.ffn_up.weight",
            ],
            "bf16",
        ),
        ("Q4_K_M_6x_q8_0", "Q4_K_M", list(TRACK2_GGUF_TENSORS), "q8_0"),
        ("Q5_K_M_6x_bf16", "Q5_K_M", list(TRACK2_GGUF_TENSORS), "bf16"),
        ("Q6_K_6x_bf16", "Q6_K", list(TRACK2_GGUF_TENSORS), "bf16"),
    ]

    results: dict[str, Any] = {"base_quant_mib": base_sizes, "configs": {}}
    try:
        for cfg_name, base_q, override_tensors, override_dtype in configs:
            dst = TMP_GGUF_DIR / f"t2_{cfg_name}.gguf"
            cmd = [str(QUANT_BIN)]
            for tname in override_tensors:
                cmd.extend(["--tensor-type", f"{tname}={override_dtype}"])
            cmd.extend([str(bf16_src), str(dst), base_q])
            print(f"[R2b] Quantizing {cfg_name} ({base_q} + {len(override_tensors)}x{override_dtype})...", flush=True)
            subprocess.run(cmd, check=True)
            actual_types = read_gguf_tensor_types(dst)
            ev = eval_gguf_select_dev_detailed(dst, t2_ref, port)
            ev["base_quant"] = base_q
            ev["override_count"] = len(override_tensors)
            ev["override_dtype"] = override_dtype
            ev["overhead_mib_vs_base_quant"] = round(ev["gguf_mib"] - base_sizes[base_q], 2)
            ev["overhead_pct"] = round(100.0 * (ev["gguf_mib"] - base_sizes[base_q]) / base_sizes[base_q], 2)
            ev["actual_tensor_types"] = actual_types
            results["configs"][cfg_name] = ev
            if dst.exists():
                dst.unlink()
    finally:
        if TMP_GGUF_DIR.exists():
            shutil.rmtree(TMP_GGUF_DIR, ignore_errors=True)
    return results


def _silu_np(z: np.ndarray) -> np.ndarray:
    return z / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _fit_dispersed_l15_patch(
    runner: QwenDeciderRunner,
    obs_state: dict[str, Any],
    prior_weights: dict[str, torch.Tensor],
    layer_gain: float,
    m_blocks: int,
    gate_scale_mult: float,
    up_down_mult: float,
    seed_base: int = 20261124,
) -> WeightPatch:
    """Disperse L15 domain-specialist SwiGLU gate across 4*m_blocks quiet neurons with feature-masked diverse fits."""
    layer_idx = 15
    energy = obs_state["mlp_preserve_energy"][layer_idx]
    qn_all = _select_unused_quiet_neurons(energy, exclude=None, k=4 * m_blocks)

    x_norm = obs_state["mlp_in_norm"][layer_idx].numpy()
    x_rev = obs_state["mlp_in_rev"][layer_idx].numpy()
    x_pres = obs_state["mlp_in_pres"][layer_idx].numpy()
    meta = obs_state["meta_automode"]
    base_m_n = float(np.mean(obs_state["m_norm"]))
    rc_idx = [i for i, m in enumerate(meta) if m["subset"] == "real_cf"]
    base_m_r = float(np.mean(obs_state["m_rev"][rc_idx]))

    w_lm = runner.model.lm_head.weight.detach().float().cpu()
    u_dir = (w_lm[runner.letter_token_ids["B"]] - w_lm[runner.letter_token_ids["A"]]).double()
    u_unit = (u_dir / torch.linalg.norm(u_dir).clamp_min(1e-8)).numpy()

    eff_gain = max(0.15, float(layer_gain))
    bias_norm = max(0.55, -base_m_n)
    bias_rev = max(0.75, base_m_r)
    alpha_score = 1.5 / eff_gain
    alpha_bias = bias_norm / eff_gain
    alpha_bias_rev = bias_rev / eff_gain

    p_gate = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
    p_up = f"model.layers.{layer_idx}.mlp.up_proj.weight"
    p_down = f"model.layers.{layer_idx}.mlp.down_proj.weight"
    gate_w = prior_weights[p_gate].float().cpu()
    up_w = prior_weights[p_up].float().cpu()
    down_w = prior_weights[p_down].float().cpu()

    d_gate = torch.zeros_like(gate_w)
    d_up = torch.zeros_like(up_w)
    base_zero_down = torch.zeros_like(down_w)
    sig_down = torch.zeros_like(down_w)

    na, nr, npres = len(x_norm), len(x_rev), len(x_pres)
    rng = np.random.default_rng(seed_base + 1500)
    d_hid = x_norm.shape[1]

    # Collect distinct family groups for deterministic 75% replicate subsampling
    fams = sorted({(m["subset"], str(m.get("family", m["id"].rsplit("_", 1)[0]))) for m in meta})

    for b in range(m_blocks):
        j1, j2, j3, j4 = qn_all[4 * b : 4 * (b + 1)]
        if m_blocks == 1:
            feat_mask = np.ones(d_hid, dtype=np.float64)
            row_idx = np.arange(na)
            pres_idx = np.arange(npres)
        else:
            # Distinct random 70% feature subspace & 80% stratified rows so weights decorrelate
            keep_f = rng.choice(d_hid, size=int(0.70 * d_hid), replace=False)
            feat_mask = np.zeros(d_hid, dtype=np.float64)
            feat_mask[keep_f] = 1.0
            n_f_keep = max(4, int(0.80 * len(fams)))
            chosen_fams = {fams[i] for i in rng.choice(len(fams), size=n_f_keep, replace=False)}
            row_idx = np.array(
                [i for i, m in enumerate(meta) if (m["subset"], str(m.get("family", m["id"].rsplit("_", 1)[0]))) in chosen_fams],
                dtype=np.int64,
            )
            pres_idx = rng.choice(npres, size=max(32, int(0.80 * npres)), replace=False)

        xn_m = (x_norm[row_idx] * feat_mask[None, :])
        xr_m = (x_rev[row_idx] * feat_mask[None, :])
        xp_m = (x_pres[pres_idx] * feat_mask[None, :])
        meta_sub = [meta[i] for i in row_idx]

        # Replicate exact hardened gate fitting on subsampled masked manifold
        x_all = np.concatenate([xn_m, xr_m, xp_m], axis=0)
        y_gn = np.concatenate([np.full(len(xn_m), 4.5), np.full(len(xr_m), -7.5), np.full(len(xp_m), -7.5)])
        sw_gn = np.concatenate([np.full(len(xn_m), 4.0), np.full(len(xr_m), 2.0), np.full(len(xp_m), 2.0)])
        g_norm_raw, _ = solve_closed_form_ridge(x_all, y_gn, lam=120.0, center=False, sample_weights=sw_gn)

        y_gr = np.concatenate([np.full(len(xn_m), -7.5), np.full(len(xr_m), 4.5), np.full(len(xp_m), -7.5)])
        sw_gr = np.concatenate([np.full(len(xn_m), 2.0), np.full(len(xr_m), 4.0), np.full(len(xp_m), 2.0)])
        g_rev_raw, _ = solve_closed_form_ridge(x_all, y_gr, lam=120.0, center=False, sample_weights=sw_gr)

        g_norm = g_norm_raw * feat_mask * gate_scale_mult
        g_rev = g_rev_raw * feat_mask * gate_scale_mult
        s_gn_pos = float(np.mean(_silu_np(x_norm @ g_norm)))
        s_gr_pos = float(np.mean(_silu_np(x_rev @ g_rev)))

        w_n, b_n, _ = fit_joint_contrastive_readout(xn_m, meta_sub, w_c5=0.5, w_pair=1.5)
        w_r, b_r, _ = fit_joint_contrastive_readout(xr_m, meta_sub, w_c5=0.5, w_pair=1.5)
        w_n = w_n * feat_mask
        w_r = w_r * feat_mask

        mu_n = np.mean(x_norm, axis=0)
        b_unit_n = mu_n / (float(np.dot(mu_n, mu_n)) + 1e-12)
        w_eff_n = (w_n + b_n * b_unit_n) * up_down_mult
        b_unit_n_s = b_unit_n * up_down_mult

        mu_r = np.mean(x_rev, axis=0)
        b_unit_r = mu_r / (float(np.dot(mu_r, mu_r)) + 1e-12)
        w_eff_r = (w_r + b_r * b_unit_r) * up_down_mult
        b_unit_r_s = b_unit_r * up_down_mult

        g_sc_n = max(0.20, s_gn_pos)
        g_sc_r = max(0.20, s_gr_pos)
        inv_share = 1.0 / (float(m_blocks) * up_down_mult)

        d_gate[j1] = torch.from_numpy(g_norm).float() - gate_w[j1]
        d_gate[j2] = torch.from_numpy(g_norm).float() - gate_w[j2]
        d_gate[j3] = torch.from_numpy(g_rev).float() - gate_w[j3]
        d_gate[j4] = torch.from_numpy(g_rev).float() - gate_w[j4]

        d_up[j1] = torch.from_numpy(w_eff_n).float() - up_w[j1]
        d_up[j2] = torch.from_numpy(b_unit_n_s).float() - up_w[j2]
        d_up[j3] = torch.from_numpy(w_eff_r).float() - up_w[j3]
        d_up[j4] = torch.from_numpy(b_unit_r_s).float() - up_w[j4]

        for j in (j1, j2, j3, j4):
            base_zero_down[:, j] = -down_w[:, j]

        sig_down[:, j1] = torch.from_numpy((alpha_score * inv_share / g_sc_n) * u_unit).float()
        sig_down[:, j2] = torch.from_numpy((alpha_bias * inv_share / g_sc_n) * u_unit).float()
        sig_down[:, j3] = torch.from_numpy((-alpha_score * inv_share / g_sc_r) * u_unit).float()
        sig_down[:, j4] = torch.from_numpy((alpha_bias_rev * inv_share / g_sc_r) * u_unit).float()

    return WeightPatch(
        step_id=f"dispersed_L15_m{m_blocks}",
        category="wsp_dispersed_swiglu",
        method="dispersed_closed_form_swiglu",
        target_layers=[layer_idx],
        rank=4 * m_blocks,
        alpha=alpha_score,
        hypothesis=f"Norm-matched dispersed L15 domain gate across K={4*m_blocks} quiet neurons (m={m_blocks}).",
        deltas={p_gate: d_gate, p_up: d_up, p_down: sig_down},
        scale_keys=[p_down],
        base_zero_deltas={p_down: base_zero_down},
        metadata={"quiet_neurons": qn_all, "m_blocks": m_blocks, "gate_scale_mult": gate_scale_mult},
    )


def _fit_dispersed_l22_patch(
    runner: QwenDeciderRunner,
    obs_state: dict[str, Any],
    prior_weights: dict[str, torch.Tensor],
    layer_gain: float,
    m_blocks: int,
    gate_scale_mult: float,
    up_down_mult: float,
    seed_base: int = 20261124,
) -> WeightPatch:
    """Disperse L22 residual-boosting SwiGLU gate across 4*m_blocks quiet neurons."""
    layer_idx = 22
    energy = obs_state["mlp_preserve_energy"][layer_idx]
    qn_all = _select_unused_quiet_neurons(energy, exclude=None, k=4 * m_blocks)

    x_norm = obs_state["mlp_in_norm"][layer_idx].numpy()
    x_rev = obs_state["mlp_in_rev"][layer_idx].numpy()
    x_pres = obs_state["mlp_in_pres"][layer_idx].numpy()
    meta = obs_state["meta_automode"]
    m_norm_arr = obs_state["m_norm"]
    m_rev_arr = obs_state["m_rev"]

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

    w_lm = runner.model.lm_head.weight.detach().float().cpu()
    u_dir = (w_lm[runner.letter_token_ids["B"]] - w_lm[runner.letter_token_ids["A"]]).double()
    u_unit = (u_dir / torch.linalg.norm(u_dir).clamp_min(1e-8)).numpy()

    step_eta = 0.70
    eff_gain = max(0.15, float(layer_gain))

    p_gate = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
    p_up = f"model.layers.{layer_idx}.mlp.up_proj.weight"
    p_down = f"model.layers.{layer_idx}.mlp.down_proj.weight"
    gate_w = prior_weights[p_gate].float().cpu()
    up_w = prior_weights[p_up].float().cpu()
    down_w = prior_weights[p_down].float().cpu()

    d_gate = torch.zeros_like(gate_w)
    d_up = torch.zeros_like(up_w)
    base_zero_down = torch.zeros_like(down_w)
    sig_down = torch.zeros_like(down_w)

    npres = len(x_pres)
    rng = np.random.default_rng(seed_base + 2200)
    d_hid = x_norm.shape[1]
    fams = sorted({(m["subset"], str(m.get("family", m["id"].rsplit("_", 1)[0]))) for m in meta})

    for b in range(m_blocks):
        j1, j2, j3, j4 = qn_all[4 * b : 4 * (b + 1)]
        if m_blocks == 1:
            feat_mask = np.ones(d_hid, dtype=np.float64)
            row_idx = np.arange(len(x_norm))
            pres_idx = np.arange(npres)
        else:
            keep_f = rng.choice(d_hid, size=int(0.70 * d_hid), replace=False)
            feat_mask = np.zeros(d_hid, dtype=np.float64)
            feat_mask[keep_f] = 1.0
            n_f_keep = max(4, int(0.80 * len(fams)))
            chosen_fams = {fams[i] for i in rng.choice(len(fams), size=n_f_keep, replace=False)}
            row_idx = np.array(
                [i for i, m in enumerate(meta) if (m["subset"], str(m.get("family", m["id"].rsplit("_", 1)[0]))) in chosen_fams],
                dtype=np.int64,
            )
            pres_idx = rng.choice(npres, size=max(32, int(0.80 * npres)), replace=False)

        xn_m = x_norm[row_idx] * feat_mask[None, :]
        xr_m = x_rev[row_idx] * feat_mask[None, :]
        xp_m = x_pres[pres_idx] * feat_mask[None, :]
        meta_sub = [meta[i] for i in row_idx]
        mn_sub = m_norm_arr[row_idx]
        mr_sub = m_rev_arr[row_idx]

        x_all = np.concatenate([xn_m, xr_m, xp_m], axis=0)
        y_gn = np.concatenate([np.full(len(xn_m), 4.5), np.full(len(xr_m), -7.5), np.full(len(xp_m), -7.5)])
        sw_gn = np.concatenate([np.full(len(xn_m), 4.0), np.full(len(xr_m), 2.0), np.full(len(xp_m), 2.0)])
        g_norm_raw, _ = solve_closed_form_ridge(x_all, y_gn, lam=120.0, center=False, sample_weights=sw_gn)

        y_gr = np.concatenate([np.full(len(xn_m), -7.5), np.full(len(xr_m), 4.5), np.full(len(xp_m), -7.5)])
        sw_gr = np.concatenate([np.full(len(xn_m), 2.0), np.full(len(xr_m), 4.0), np.full(len(xp_m), 2.0)])
        g_rev_raw, _ = solve_closed_form_ridge(x_all, y_gr, lam=120.0, center=False, sample_weights=sw_gr)

        g_norm = g_norm_raw * feat_mask * gate_scale_mult
        g_rev = g_rev_raw * feat_mask * gate_scale_mult
        s_gn_pos = float(np.mean(_silu_np(x_norm @ g_norm)))
        s_gr_pos = float(np.mean(_silu_np(x_rev @ g_rev)))

        w_n, b_n, _ = fit_residual_boosting_readout(
            xn_m, mn_sub, meta_sub, target_margin=2.0, clip_val=3.0, pattern_weights=pw_c5_spec, w_pair=1.5
        )
        w_r, b_r, _ = fit_residual_boosting_readout(
            xr_m, mr_sub, meta_sub, target_margin=2.0, clip_val=3.0, pattern_weights=pw_c5_spec, w_pair=1.5
        )
        w_n = w_n * feat_mask
        w_r = w_r * feat_mask

        mu_n = np.mean(x_norm, axis=0)
        b_unit_n = mu_n / (float(np.dot(mu_n, mu_n)) + 1e-12)
        mean_e_n = float(np.mean(x_norm @ w_n) + b_n)
        w_c_n = (w_n - float(np.mean(x_norm @ w_n)) * b_unit_n) * up_down_mult
        b_unit_n_s = b_unit_n * up_down_mult

        mu_r = np.mean(x_rev, axis=0)
        b_unit_r = mu_r / (float(np.dot(mu_r, mu_r)) + 1e-12)
        mean_e_r = float(np.mean(x_rev @ w_r) + b_r)
        w_c_r = (w_r - float(np.mean(x_rev @ w_r)) * b_unit_r) * up_down_mult
        b_unit_r_s = b_unit_r * up_down_mult

        g_sc_n = max(0.20, s_gn_pos)
        g_sc_r = max(0.20, s_gr_pos)
        inv_share = 1.0 / (float(m_blocks) * up_down_mult)

        d_gate[j1] = torch.from_numpy(g_norm).float() - gate_w[j1]
        d_gate[j2] = torch.from_numpy(g_norm).float() - gate_w[j2]
        d_gate[j3] = torch.from_numpy(g_rev).float() - gate_w[j3]
        d_gate[j4] = torch.from_numpy(g_rev).float() - gate_w[j4]

        d_up[j1] = torch.from_numpy(w_c_n).float() - up_w[j1]
        d_up[j2] = torch.from_numpy(b_unit_n_s).float() - up_w[j2]
        d_up[j3] = torch.from_numpy(w_c_r).float() - up_w[j3]
        d_up[j4] = torch.from_numpy(b_unit_r_s).float() - up_w[j4]

        for j in (j1, j2, j3, j4):
            base_zero_down[:, j] = -down_w[:, j]

        sig_down[:, j1] = torch.from_numpy((step_eta * inv_share / (g_sc_n * eff_gain)) * u_unit).float()
        sig_down[:, j2] = torch.from_numpy((step_eta * mean_e_n * inv_share / (g_sc_n * eff_gain)) * u_unit).float()
        sig_down[:, j3] = torch.from_numpy((-step_eta * inv_share / (g_sc_r * eff_gain)) * u_unit).float()
        sig_down[:, j4] = torch.from_numpy((-step_eta * mean_e_r * inv_share / (g_sc_r * eff_gain)) * u_unit).float()

    return WeightPatch(
        step_id=f"dispersed_L22_m{m_blocks}",
        category="wsp_dispersed_swiglu",
        method="dispersed_closed_form_swiglu",
        target_layers=[layer_idx],
        rank=4 * m_blocks,
        alpha=step_eta,
        hypothesis=f"Norm-matched dispersed L22 consent gate across K={4*m_blocks} quiet neurons (m={m_blocks}).",
        deltas={p_gate: d_gate, p_up: d_up, p_down: sig_down},
        scale_keys=[p_down],
        base_zero_deltas={p_down: base_zero_down},
        metadata={"quiet_neurons": qn_all, "m_blocks": m_blocks, "gate_scale_mult": gate_scale_mult},
    )


def export_bundle_to_temp_ggufs(
    variant_name: str,
    deltas_pt: Path,
    quant_types: tuple[str, ...] = QUANT_TYPES,
) -> dict[str, Path]:
    TMP_GGUF_DIR.mkdir(parents=True, exist_ok=True)
    hf_dir = TMP_GGUF_DIR / f"{variant_name}_hf"
    bf16_gguf = TMP_GGUF_DIR / f"{variant_name}-bf16.gguf"
    if hf_dir.exists():
        shutil.rmtree(hf_dir)
    export_patched_hf(deltas_pt, hf_dir)
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{_GGUF_PY}:{env.get('PYTHONPATH', '')}"
    subprocess.run(
        [sys.executable, str(CONVERT_HF_GGUF), str(hf_dir), "--outfile", str(bf16_gguf), "--outtype", "bf16"],
        check=True,
        env=env,
    )
    shutil.rmtree(hf_dir, ignore_errors=True)
    out: dict[str, Path] = {}
    for qtype in quant_types:
        dst = TMP_GGUF_DIR / f"{variant_name}-{qtype}.gguf"
        subprocess.run([str(QUANT_BIN), str(bf16_gguf), str(dst), qtype], check=True)
        out[qtype] = dst
    if bf16_gguf.exists():
        bf16_gguf.unlink()
    return out


def compute_bundle_forensics(name: str, bundle: PatchBundle, base_hf_tensors: dict[str, torch.Tensor]) -> dict[str, Any]:
    summed: dict[str, torch.Tensor] = {}
    for p in bundle.patches:
        for ck, dt in p.net_deltas().items():
            hk = map_causal_to_hf_key(ck)
            df = dt.detach().float().cpu()
            summed[hk] = (summed[hk] + df) if hk in summed else df.clone()
    cand: dict[str, torch.Tensor] = {}
    for k, v in base_hf_tensors.items():
        cand[k] = (v.float() + summed[k]).to(dtype=v.dtype) if k in summed else v
    wf = _analyze_weight_forensics_against_base(name, base_hf_tensors, cand)
    wf.pop("top_modified_tensors", None)
    return wf


def run_r2c_multi_neuron_dispersion(port: int) -> dict[str, Any]:
    print("[R2c] Running Norm-Matched Multi-Neuron WSP Dispersion ablation (m=4 & m=16)...", flush=True)
    TMP_GGUF_DIR.mkdir(parents=True, exist_ok=True)
    obs_rows = load_split_rows("observe")
    dev_rows = load_split_rows("select_dev")

    base_dir = Path(snapshot_download("Qwen/Qwen3.5-0.8B", local_files_only=True))
    base_hf_tensors = _load_hf_safetensors_dir(base_dir)

    runner = QwenDeciderRunner()
    base_cpu = capture_cpu_state_dict(runner.model)

    obs_base_l15 = collect_post_patch_observe_state(runner, obs_rows, target_layers=[15], include_attn=False)
    prior_l15 = _snapshot_gate_layer_weights(runner.model, [15])[15]
    _, gains_l15 = _measure_layer_set_gains(runner, obs_rows, [15])

    specs = [
        ("wsp_dispersed_m4_sqrt", 4, 1.0 / math.sqrt(4.0), 1.0 / math.sqrt(4.0)),
        ("wsp_dispersed_m16_sqrt", 16, 1.0 / math.sqrt(16.0), 1.0 / math.sqrt(16.0)),
        ("wsp_dispersed_m16_g05", 16, 0.5, 1.0 / math.sqrt(16.0)),
    ]

    variants_out: dict[str, Any] = {}
    built_deltas: dict[str, tuple[PatchBundle, Path]] = {}

    for vname, m_blk, g_mult, ud_mult in specs:
        print(f"[R2c] Solving {vname} (m={m_blk}, K={4*m_blk}/layer, g_mult={g_mult:.3f})...", flush=True)
        restore_cpu_state_dict(runner.model, base_cpu)
        p15 = _fit_dispersed_l15_patch(runner, obs_base_l15, prior_l15, gains_l15[15], m_blk, g_mult, ud_mult)
        apply_bundle(runner.model, base_cpu, [p15])

        obs_l22 = collect_post_patch_observe_state(runner, obs_rows, target_layers=[22], include_attn=False)
        prior_l22 = _snapshot_gate_layer_weights(runner.model, [22])[22]
        _, gains_l22 = _measure_layer_set_gains(runner, obs_rows, [22])
        p22 = _fit_dispersed_l22_patch(runner, obs_l22, prior_l22, gains_l22[22], m_blk, g_mult, ud_mult)

        bundle = PatchBundle(patches=[p15, p22], metadata={"variant": vname, "m_blocks": m_blk})
        apply_bundle(runner.model, base_cpu, bundle.patches)

        rep_dev = evaluate_runner(runner, dev_rows, mode="plain", include_recap=False, eval_recap_ablation=False)
        wf = compute_bundle_forensics(vname, bundle, base_hf_tensors)
        print(
            f"[R2c] {vname} bf16 select_dev: proxy={rep_dev['proxy']['proxy_score']} "
            f"rc_auc={rep_dev['real_cf']['auroc']} c5={rep_dev['consent_5p']['overall_acc']} "
            f"frob={wf['total_delta_frobenius_norm']} rms={wf['rms_delta_per_modified_param']} "
            f"q8_mse_ratio={wf['q8_0_forensics']['touched_block_mse_inflation_ratio']}",
            flush=True,
        )

        deltas_pt = TMP_GGUF_DIR / f"{vname}_deltas.pt"
        spec_json = TMP_GGUF_DIR / f"{vname}_spec.json"
        save_patch_bundle(bundle, spec_json, deltas_pt)
        built_deltas[vname] = (bundle, deltas_pt)
        variants_out[vname] = {
            "m_blocks": m_blk,
            "neurons_per_layer": 4 * m_blk,
            "total_neurons_edited": 8 * m_blk,
            "gate_scale_mult": round(g_mult, 4),
            "up_down_mult": round(ud_mult, 4),
            "weight_forensics": wf,
            "bf16_select_dev": {
                "proxy_score": rep_dev["proxy"]["proxy_score"],
                "real_cf_auroc": rep_dev["real_cf"]["auroc"],
                "real_cf_rec": rep_dev["real_cf"]["recall_at_05"],
                "real_cf_fpr": rep_dev["real_cf"]["fpr_at_05"],
                "real_cf_flip": rep_dev["real_cf"]["order_flip_rate"],
                "consent_5p_acc": rep_dev["consent_5p"]["overall_acc"],
                "long_50t_acc": rep_dev["long_50t"]["overall_acc"],
                "decider_replay_acc": rep_dev["decider_replay"]["overall_acc"],
                "decider_replay_ece": rep_dev["decider_replay"]["top_label_ece"],
            },
            "quantization_curve": {},
        }

    del base_cpu, runner, base_hf_tensors, obs_base_l15
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Quantize & benchmark all dispersed variants on select_dev against their own bf16 refs
    try:
        for vname, (_, deltas_pt) in built_deltas.items():
            ref = _collect_pt_select_dev_ref(deltas_pt, dev_rows)
            qpaths = export_bundle_to_temp_ggufs(vname, deltas_pt, QUANT_TYPES)
            for qtype, qpath in qpaths.items():
                print(f"[R2c] Evaluating {vname} @ {qtype}...", flush=True)
                variants_out[vname]["quantization_curve"][qtype] = eval_gguf_select_dev_detailed(qpath, ref, port)
                if qpath.exists():
                    qpath.unlink()
            if deltas_pt.exists():
                deltas_pt.unlink()
    finally:
        if TMP_GGUF_DIR.exists():
            shutil.rmtree(TMP_GGUF_DIR, ignore_errors=True)

    return variants_out


def run_r2d_double_dissociation(port: int) -> dict[str, Any]:
    """Surgical double-dissociation isolating upstream backbone vs edited-tensor vs embedding quantization."""
    print("[R2d] Running Causal Double-Dissociation of K-Quant Collapse...", flush=True)
    TMP_GGUF_DIR.mkdir(parents=True, exist_ok=True)
    dev_rows = load_split_rows("select_dev")
    t2_ref = _collect_pt_select_dev_ref(CHECKPOINTS_DIR / "track2_best_deltas.pt", dev_rows)
    bf16_src = CHECKPOINTS_DIR / "wsp-automode-0.8b-bf16.gguf"

    sample_names = (
        "token_embd.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_down.weight",
        "blk.3.attn_q.weight",
        "blk.14.ffn_down.weight",
        "blk.15.attn_q.weight",
        "blk.22.ssm_out.weight",
        "blk.23.ffn_down.weight",
    ) + TRACK2_GGUF_TENSORS
    vanilla_types: dict[str, dict[str, str]] = {}
    for qtype in QUANT_TYPES:
        gp = CHECKPOINTS_DIR / f"wsp-automode-0.8b-{qtype}.gguf"
        vanilla_types[qtype] = read_gguf_tensor_types(gp, sample_names)

    src_reader = GGUFReader(str(bf16_src))
    bf16_tensor_names = [
        t.name for t in src_reader.tensors if str(t.tensor_type.name) in ("BF16", "F16")
    ]

    def _blk_num(tname: str) -> int | None:
        if tname.startswith("blk."):
            try:
                return int(tname.split(".")[1])
            except (IndexError, ValueError):
                return None
        return None

    t2_set = set(TRACK2_GGUF_TENSORS)
    specs: list[tuple[str, str, list[str], dict[str, str]]] = [
        (
            "Q8_0_backbone_t2_Q4_K",
            "Q8_0",
            [],
            {tname: "q4_K" for tname in TRACK2_GGUF_TENSORS},
        ),
        (
            "Q4_K_M_embd_q8_0_t2_bf16",
            "Q4_K_M",
            ["--token-embedding-type", "q8_0", "--output-tensor-type", "q8_0"],
            {tname: "bf16" for tname in TRACK2_GGUF_TENSORS},
        ),
    ]

    map_0_14: dict[str, str] = {}
    map_0_22: dict[str, str] = {}
    for tname in bf16_tensor_names:
        bn = _blk_num(tname)
        if tname in t2_set:
            map_0_14[tname] = "bf16"
            map_0_22[tname] = "bf16"
        elif tname in ("token_embd.weight", "output.weight"):
            map_0_14[tname] = "q8_0"
            map_0_22[tname] = "q8_0"
        elif bn is not None and bn <= 14:
            map_0_14[tname] = "q8_0"
            map_0_22[tname] = "q8_0"
        elif bn is not None and bn <= 22:
            map_0_22[tname] = "q8_0"

    specs.append(
        (
            "Q4_K_M_L0_14_q8_0_t2_bf16",
            "Q4_K_M",
            ["--token-embedding-type", "q8_0", "--output-tensor-type", "q8_0"],
            map_0_14,
        )
    )
    specs.append(
        (
            "Q4_K_M_L0_22_q8_0_t2_bf16",
            "Q4_K_M",
            ["--token-embedding-type", "q8_0", "--output-tensor-type", "q8_0"],
            map_0_22,
        )
    )

    configs_out: dict[str, Any] = {}
    try:
        for cfg_name, base_q, extra_flags, type_map in specs:
            dst = TMP_GGUF_DIR / f"r2d_{cfg_name}.gguf"
            tt_file = TMP_GGUF_DIR / f"r2d_{cfg_name}_types.txt"
            tt_file.write_text(
                "\n".join(f"{k}={v}" for k, v in sorted(type_map.items())) + "\n",
                encoding="utf-8",
            )
            cmd = [str(QUANT_BIN), *extra_flags, "--tensor-type-file", str(tt_file), str(bf16_src), str(dst), base_q]
            print(f"[R2d] Quantizing {cfg_name} ({base_q}, {len(type_map)} tensor overrides)...", flush=True)
            subprocess.run(cmd, check=True)
            actual_types = read_gguf_tensor_types(dst, sample_names)
            ev = eval_gguf_select_dev_detailed(dst, t2_ref, port)
            ev["base_quant"] = base_q
            ev["override_count"] = len(type_map)
            ev["sample_actual_types"] = actual_types
            configs_out[cfg_name] = ev
            if dst.exists():
                dst.unlink()
            if tt_file.exists():
                tt_file.unlink()
    finally:
        if TMP_GGUF_DIR.exists():
            shutil.rmtree(TMP_GGUF_DIR, ignore_errors=True)

    return {"vanilla_sample_types": vanilla_types, "configs": configs_out}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Suite R2 Quantization Diagnostics & Mitigations.")
    ap.add_argument("--parts", nargs="+", default=["r2a", "r2b", "r2c"], choices=["r2a", "r2b", "r2c", "r2d"])
    ap.add_argument("--port", type=int, default=8094)
    args = ap.parse_args()

    t0 = time.perf_counter()
    payload = _load_controls_json()

    if "r2a" in args.parts:
        payload["kquant_diagnostics"] = run_r2a_kquant_diagnostics(args.port)
        _save_controls_json(payload)
    if "r2b" in args.parts:
        payload["mixed_precision_mitigation"] = run_r2b_mixed_precision_mitigation(args.port)
        _save_controls_json(payload)
    if "r2c" in args.parts:
        payload["multi_neuron_dispersion"] = run_r2c_multi_neuron_dispersion(args.port)
        _save_controls_json(payload)
    if "r2d" in args.parts:
        payload["kquant_double_dissociation"] = run_r2d_double_dissociation(args.port)
        _save_controls_json(payload)

    payload["r2_elapsed_sec"] = round(time.perf_counter() - t0, 1)
    _save_controls_json(payload)
    print(f"[R2] Saved updated experiments/paper_revision_controls.json in {payload['r2_elapsed_sec']}s", flush=True)


if __name__ == "__main__":
    main()
