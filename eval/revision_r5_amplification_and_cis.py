"""Revision Suite R5: Unmodified Base GGUF Curve, Bootstrap CIs, and Direct 4-Bit Gate Amplification.

Adds two diagnostic sections into `experiments/paper_revision_controls.json`:
  - `r5_base_and_bootstrap_cis` (R5a): Evaluates unmodified `Qwen/Qwen3.5-0.8B` across
    `Q8_0..Q4_K_M` alongside Track 2, Track 3 Soup, and the 5 R2b/R2d surgical GGUFs on
    `select_dev` (n=200) via `llama-server` with exact HF token IDs, computing B=10,000
    bootstrap 95% CIs for Pearson `r`, `p_block` MAE, raw `|Δm|` MAE, normalized `|Δm|/|m|`,
    and decision agreement, plus paired bootstrap difference CIs.
  - `r5_activation_amplification` (R5b): Directly dequantizes real GGUF weights via
    `gguf.dequantize` back into PyTorch `Qwen3.5-0.8B` and captures live internal `h_15`/`h_22`
    drift, L15/L22 SwiGLU gate perturbation on Track 2's 4 edited neurons vs. the 3,580
    untouched neurons, and post-MLP output drift across Base, Track 2, Track 3, and
    backbone-vs-edited-tensor surgical hybrids.
"""

from __future__ import annotations

import argparse
import gc
import json
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

from automode.questions import Q_BLOCK_BINARY, build_decider_prompt, build_plain_prompt
from automode.view import render_view
from eval.evaluate import compute_binary_metrics, compute_ece, compute_proxy_score, load_split_rows
from eval.mechanism_analysis import (
    BASE_ID,
    QUANT_BIN,
    QUANT_TYPES,
    T2_L15_NEURONS,
    T2_L22_NEURONS,
    _wait_server,
)
from eval.model_runner import QwenDeciderRunner, capture_cpu_state_dict, restore_cpu_state_dict
from eval.revision_quant_mitigations import (
    CONTROLS_JSON,
    CONVERT_HF_GGUF,
    LAUNCH_SCRIPT,
    TMP_GGUF_DIR,
    TRACK2_GGUF_TENSORS,
    _load_controls_json,
    _save_controls_json,
)
from runtime.client import LlamaCompletionClient
from weight_edit.causal_check import apply_bundle, load_patch_bundle

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = ROOT / "checkpoints"
EXPERIMENTS_DIR = ROOT / "experiments"

FULL_ATTN_LAYERS = {3, 7, 11, 15, 19, 23}
BOOTSTRAP_B = 10000
BOOTSTRAP_SEED = 20261124


def collect_pt_select_dev_ref(deltas_path: Path | None, dev_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Collect PyTorch bf16 reference predictions on `select_dev` (`deltas_path=None` = unmodified Base)."""
    runner = QwenDeciderRunner()
    if deltas_path is not None:
        base_cpu = capture_cpu_state_dict(runner.model)
        bundle = load_patch_bundle(deltas_path)
        apply_bundle(runner.model, base_cpu, bundle.patches)

    by_sub: dict[str, list[dict[str, Any]]] = {}
    for r in dev_rows:
        by_sub.setdefault(r["subset"], []).append(r)

    bin_recs: list[dict[str, Any]] = []
    for sub in ("real_cf", "consent_5p", "long_50t"):
        budget = 3200 if sub == "long_50t" else 2048
        for r in by_sub.get(sub, []):
            ctx = render_view(r, tok=runner.tokenizer, budget=budget, include_recap=False).text
            sc = runner.score_binary_paired(ctx, mode="plain", question_spec=Q_BLOCK_BINARY)
            bin_recs.append(
                {
                    "id": r["id"],
                    "subset": sub,
                    "pattern": r.get("pattern", ""),
                    "y": 1 if r["label"] == "block" else 0,
                    "ctx": ctx,
                    "pt_p_block": float(sc["p_block"]),
                    "pt_p_sr": float(sc["p_block_second_reading"]),
                    "pt_margin": float(sc["margin_normal"]),
                    "pt_pred_norm": sc["pred_normal"],
                }
            )

    rep_recs: list[dict[str, Any]] = []
    for r in by_sub.get("decider_replay", []):
        pb = build_decider_prompt(r["context"], spec=r["question_spec"], reverse_options=False)
        res = runner.score_prompt(pb.text, pb.option_letters, yes_idx=pb.yes_idx, no_idx=pb.no_idx)
        rep_recs.append(
            {
                "id": r["id"],
                "q_type": r["q_type"],
                "context": r["context"],
                "question_spec": r["question_spec"],
                "gold_index": int(r["gold_index"]),
                "pt_pred_index": int(res.pred_index),
            }
        )

    del runner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"binary": bin_recs, "replay": rep_recs}


def eval_gguf_select_dev_vectors(
    gguf_path: Path,
    ref: dict[str, Any],
    port: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run GGUF on `select_dev` via `llama-server` and return both summary metrics and per-item arrays."""
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

        q_bin: list[dict[str, Any]] = []
        for item in ref["binary"]:
            sc = client.score_binary_paired(item["ctx"], force_paired=True)
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
                    "q_flipped": (pn >= 0.5) != (pr >= 0.5),
                    "q_abs_diff": abs(mn - mr),
                }
            )

        q_rep_agree: list[int] = []
        q_rep_correct: list[int] = []
        q_rep_top_p: list[float] = []
        for item in ref["replay"]:
            sc = client.score_question(item["context"], item["question_spec"])
            pred_i = int(sc["pred_index"])
            probs = list(sc["option_probs"])
            q_rep_agree.append(int(pred_i == item["pt_pred_index"]))
            q_rep_correct.append(int(pred_i == item["gold_index"]))
            q_rep_top_p.append(float(probs[pred_i]))

        rc_l = [x for x in q_bin if x["subset"] == "real_cf"]
        c5_l = [x for x in q_bin if x["subset"] == "consent_5p"]
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
        rep_sum = {
            "overall_acc": round(sum(q_rep_correct) / len(q_rep_correct), 4),
            "top_label_ece": round(compute_ece(q_rep_top_p, q_rep_correct), 4),
        }
        proxy = compute_proxy_score({"real_cf": b_rc, "consent_5p": c5_sum, "decider_replay": rep_sum})

        pt_p = np.array([x["pt_p"] for x in q_bin], dtype=np.float64)
        q_p = np.array([x["q_p"] for x in q_bin], dtype=np.float64)
        pt_m = np.array([x["pt_m"] for x in q_bin], dtype=np.float64)
        q_m = np.array([x["q_m"] for x in q_bin], dtype=np.float64)
        agree_bin = np.array([int(x["pt_pred"] == ("block" if x["q_p"] >= 0.5 else "allow")) for x in q_bin], dtype=np.float64)
        agree_200 = np.concatenate([agree_bin, np.array(q_rep_agree, dtype=np.float64)])

        summary = {
            "gguf_mib": round(gguf_path.stat().st_size / (1024.0 ** 2), 2),
            "real_cf_auroc": b_rc["auroc"],
            "real_cf_rec": b_rc["recall_at_05"],
            "real_cf_fpr": b_rc["fpr_at_05"],
            "real_cf_flip": b_rc["order_flip_rate"],
            "consent_5p_acc": b_c5["acc_at_05"],
            "decider_replay_acc": rep_sum["overall_acc"],
            "proxy_score": proxy["proxy_score"],
            "pt_mean_abs_margin": round(float(np.mean(np.abs(pt_m))), 4),
            "pt_std_margin": round(float(np.std(pt_m)), 4),
            "pt_std_p_block": round(float(np.std(pt_p)), 4),
            "q_std_margin": round(float(np.std(q_m)), 4),
            "q_std_p_block": round(float(np.std(q_p)), 4),
            "signed_margin_bias": round(float(np.mean(q_m - pt_m)), 4),
        }
        vecs = {"pt_p": pt_p, "q_p": q_p, "pt_m": pt_m, "q_m": q_m, "agree_200": agree_200}
        return summary, vecs
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log_f.close()


def _bootstrap_distribution(
    vecs: dict[str, np.ndarray],
    idx_bin: np.ndarray,
    idx_all: np.ndarray,
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    pt_p, q_p = vecs["pt_p"], vecs["q_p"]
    pt_m, q_m = vecs["pt_m"], vecs["q_m"]
    ag = vecs["agree_200"]

    pt_abs_m = float(np.mean(np.abs(pt_m)))
    pt_r = float(np.corrcoef(pt_p, q_p)[0, 1]) if np.std(q_p) > 1e-12 and np.std(pt_p) > 1e-12 else 0.0
    pt_p_mae = float(np.mean(np.abs(pt_p - q_p)))
    pt_m_mae = float(np.mean(np.abs(pt_m - q_m)))
    pt_m_norm = pt_m_mae / max(1e-8, pt_abs_m)
    pt_ag = float(np.mean(ag))

    xb = pt_p[idx_bin]
    yb = q_p[idx_bin]
    dx = xb - xb.mean(axis=1, keepdims=True)
    dy = yb - yb.mean(axis=1, keepdims=True)
    denom = np.sqrt((dx * dx).sum(axis=1) * (dy * dy).sum(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        r_dist = np.where(denom > 1e-12, (dx * dy).sum(axis=1) / denom, np.nan)

    p_mae_dist = np.abs(xb - yb).mean(axis=1)
    m_diff = np.abs(pt_m[idx_bin] - q_m[idx_bin])
    m_mae_dist = m_diff.mean(axis=1)
    m_norm_dist = m_mae_dist / np.clip(np.abs(pt_m[idx_bin]).mean(axis=1), 1e-8, None)
    ag_dist = ag[idx_all].mean(axis=1)

    def _ci_pack(point: float, arr: np.ndarray, digits: int = 4) -> dict[str, Any]:
        valid = arr[~np.isnan(arr)]
        lo, hi = np.percentile(valid, [2.5, 97.5])
        return {
            "point": round(point, digits),
            "ci95_lo": round(float(lo), digits),
            "ci95_hi": round(float(hi), digits),
        }

    stats = {
        "pearson_r": _ci_pack(pt_r, r_dist, 4),
        "p_block_mae": _ci_pack(pt_p_mae, p_mae_dist, 5),
        "margin_mae": _ci_pack(pt_m_mae, m_mae_dist, 4),
        "margin_mae_norm": _ci_pack(pt_m_norm, m_norm_dist, 4),
        "agreement_total_200": _ci_pack(pt_ag, ag_dist, 4),
    }
    dists = {
        "pearson_r": r_dist,
        "p_block_mae": p_mae_dist,
        "margin_mae": m_mae_dist,
        "margin_mae_norm": m_norm_dist,
        "agreement_total_200": ag_dist,
    }
    return stats, dists


def _paired_diff_summary(
    name_a: str,
    name_b: str,
    stats_a: dict[str, Any],
    stats_b: dict[str, Any],
    dists_a: dict[str, np.ndarray],
    dists_b: dict[str, np.ndarray],
    metric: str,
    digits: int = 4,
) -> dict[str, Any]:
    diff_pt = float(stats_a[metric]["point"] - stats_b[metric]["point"])
    diff_arr = dists_a[metric] - dists_b[metric]
    valid = diff_arr[~np.isnan(diff_arr)]
    lo, hi = np.percentile(valid, [2.5, 97.5])
    p_two = min(1.0, 2.0 * min(float(np.mean(valid <= 0.0)), float(np.mean(valid >= 0.0))))
    return {
        "a": name_a,
        "b": name_b,
        "metric": metric,
        "delta_a_minus_b": round(diff_pt, digits),
        "ci95_lo": round(float(lo), digits),
        "ci95_hi": round(float(hi), digits),
        "two_sided_p": float(f"{p_two:.4e}") if p_two < 1e-3 else round(p_two, 4),
    }


def _build_base_and_r2_ggufs() -> dict[str, Path]:
    """Create temp Base GGUFs + R2b/R2d surgical GGUFs in `TMP_GGUF_DIR`."""
    TMP_GGUF_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    base_snap = Path(snapshot_download(BASE_ID, local_files_only=True))
    base_bf16 = TMP_GGUF_DIR / "base-bf16.gguf"
    if not base_bf16.exists():
        print(f"[R5] Converting unmodified {BASE_ID} -> {base_bf16.name}...", flush=True)
        subprocess.run(
            [sys.executable, str(CONVERT_HF_GGUF), str(base_snap), "--outfile", str(base_bf16), "--outtype", "bf16"],
            check=True,
        )
    paths["base_bf16"] = base_bf16
    for qt in QUANT_TYPES:
        dst = TMP_GGUF_DIR / f"base-{qt}.gguf"
        if not dst.exists():
            subprocess.run([str(QUANT_BIN), str(base_bf16), str(dst), qt], check=True)
        paths[f"base_{qt}"] = dst

    t2_bf16 = CHECKPOINTS_DIR / "wsp-automode-0.8b-bf16.gguf"
    src_reader = gguf.GGUFReader(str(t2_bf16))
    bf16_names = [t.name for t in src_reader.tensors if str(t.tensor_type.name) in ("BF16", "F16")]
    t2_set = set(TRACK2_GGUF_TENSORS)

    def _blk_num(tname: str) -> int | None:
        if tname.startswith("blk."):
            try:
                return int(tname.split(".")[1])
            except (IndexError, ValueError):
                return None
        return None

    map_0_14: dict[str, str] = {}
    map_0_22: dict[str, str] = {}
    for tname in bf16_names:
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

    r2_specs = [
        ("r2b_Q4_K_M_6x_bf16", "Q4_K_M", [], {t: "bf16" for t in TRACK2_GGUF_TENSORS}),
        (
            "r2d_Q4_K_M_embd_q8_0_t2_bf16",
            "Q4_K_M",
            ["--token-embedding-type", "q8_0", "--output-tensor-type", "q8_0"],
            {t: "bf16" for t in TRACK2_GGUF_TENSORS},
        ),
        ("r2d_Q8_0_backbone_t2_Q4_K", "Q8_0", [], {t: "q4_K" for t in TRACK2_GGUF_TENSORS}),
        (
            "r2d_Q4_K_M_L0_14_q8_0_t2_bf16",
            "Q4_K_M",
            ["--token-embedding-type", "q8_0", "--output-tensor-type", "q8_0"],
            map_0_14,
        ),
        (
            "r2d_Q4_K_M_L0_22_q8_0_t2_bf16",
            "Q4_K_M",
            ["--token-embedding-type", "q8_0", "--output-tensor-type", "q8_0"],
            map_0_22,
        ),
    ]
    for cfg_name, base_q, extra_flags, tmap in r2_specs:
        dst = TMP_GGUF_DIR / f"{cfg_name}.gguf"
        if not dst.exists():
            tt_file = TMP_GGUF_DIR / f"{cfg_name}_types.txt"
            tt_file.write_text("\n".join(f"{k}={v}" for k, v in sorted(tmap.items())) + "\n", encoding="utf-8")
            cmd = [str(QUANT_BIN), *extra_flags, "--tensor-type-file", str(tt_file), str(t2_bf16), str(dst), base_q]
            subprocess.run(cmd, check=True)
            tt_file.unlink()
        paths[cfg_name] = dst

    return paths


def run_r5a(port: int, built_paths: dict[str, Path]) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    """Run R5a: unmodified Base GGUF curve + Track 2/3/R2b/R2d B=10,000 bootstrap CIs."""
    print("[R5a] Collecting PyTorch bf16 select_dev references (Base, Track 2, Track 3)...", flush=True)
    dev_rows = load_split_rows("select_dev")
    refs = {
        "base": collect_pt_select_dev_ref(None, dev_rows),
        "t2": collect_pt_select_dev_ref(CHECKPOINTS_DIR / "track2_best_deltas.pt", dev_rows),
        "t3": collect_pt_select_dev_ref(CHECKPOINTS_DIR / "track3_final_best_deltas.pt", dev_rows),
    }

    targets: list[tuple[str, str, Path]] = []
    for qt in QUANT_TYPES:
        targets.append((f"base_{qt}", "base", built_paths[f"base_{qt}"]))
    for qt in QUANT_TYPES:
        targets.append((f"track2_{qt}", "t2", CHECKPOINTS_DIR / f"wsp-automode-0.8b-{qt}.gguf"))
    for qt in QUANT_TYPES:
        targets.append((f"track3_{qt}", "t3", CHECKPOINTS_DIR / f"automode-0.8b-track3-{qt}.gguf"))
    for k in (
        "r2b_Q4_K_M_6x_bf16",
        "r2d_Q4_K_M_embd_q8_0_t2_bf16",
        "r2d_Q8_0_backbone_t2_Q4_K",
        "r2d_Q4_K_M_L0_14_q8_0_t2_bf16",
        "r2d_Q4_K_M_L0_22_q8_0_t2_bf16",
    ):
        targets.append((k, "t2", built_paths[k]))

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx_bin = rng.integers(0, len(refs["t2"]["binary"]), size=(BOOTSTRAP_B, len(refs["t2"]["binary"])), dtype=np.int32)
    n_all = len(refs["t2"]["binary"]) + len(refs["t2"]["replay"])
    idx_all = rng.integers(0, n_all, size=(BOOTSTRAP_B, n_all), dtype=np.int32)

    configs_out: dict[str, Any] = {}
    all_stats: dict[str, dict[str, Any]] = {}
    all_dists: dict[str, dict[str, np.ndarray]] = {}
    all_vecs: dict[str, dict[str, np.ndarray]] = {}

    for key, ref_k, gpath in targets:
        print(f"[R5a] Evaluating {key} ({gpath.name})...", flush=True)
        summary, vecs = eval_gguf_select_dev_vectors(gpath, refs[ref_k], port)
        b_stats, b_dists = _bootstrap_distribution(vecs, idx_bin, idx_all)
        configs_out[key] = {**summary, "bootstrap_ci95": b_stats}
        all_stats[key] = b_stats
        all_dists[key] = b_dists
        all_vecs[key] = vecs

    paired_specs = [
        ("r2d_Q8_0_backbone_t2_Q4_K", "r2b_Q4_K_M_6x_bf16", "pearson_r"),
        ("r2d_Q4_K_M_L0_14_q8_0_t2_bf16", "r2b_Q4_K_M_6x_bf16", "pearson_r"),
        ("r2d_Q4_K_M_L0_22_q8_0_t2_bf16", "r2b_Q4_K_M_6x_bf16", "pearson_r"),
        ("r2d_Q4_K_M_L0_22_q8_0_t2_bf16", "r2b_Q4_K_M_6x_bf16", "margin_mae"),
        ("r2d_Q8_0_backbone_t2_Q4_K", "r2b_Q4_K_M_6x_bf16", "margin_mae"),
        ("track2_Q4_K_M", "base_Q4_K_M", "margin_mae"),
        ("track2_Q4_K_M", "track3_Q4_K_M", "margin_mae"),
        ("track2_Q4_K_M", "base_Q4_K_M", "margin_mae_norm"),
        ("track2_Q4_K_M", "track3_Q4_K_M", "margin_mae_norm"),
        ("r2d_Q4_K_M_L0_22_q8_0_t2_bf16", "track2_Q8_0", "pearson_r"),
        ("r2d_Q4_K_M_L0_22_q8_0_t2_bf16", "track2_Q4_K_M", "pearson_r"),
    ]
    paired_out = [
        _paired_diff_summary(a, b, all_stats[a], all_stats[b], all_dists[a], all_dists[b], m)
        for a, b, m in paired_specs
    ]

    r5a = {
        "n_binary": len(refs["t2"]["binary"]),
        "n_total": n_all,
        "bootstrap_B": BOOTSTRAP_B,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "configs": configs_out,
        "paired_differences": paired_out,
    }
    return r5a, all_vecs


def _dequantize_gguf_2d_weights(gguf_path: Path, filter_names: set[str] | None = None) -> dict[str, torch.Tensor]:
    """Dequantize 2D linear / embedding matrices from GGUF back into HF causal weight dict."""
    reader = gguf.GGUFReader(str(gguf_path))
    mapped: dict[str, torch.Tensor] = {}

    for t in reader.tensors:
        name = t.name
        if filter_names is not None and name not in filter_names:
            continue
        if len(t.shape) != 2:
            continue

        arr = gguf.dequantize(t.data, t.tensor_type)
        # Note: convert_hf_to_gguf stores 2D weights as (out, in) inside dequantize result for Qwen3.5
        tens = torch.from_numpy(np.array(arr, copy=True)).bfloat16()

        if name == "token_embd.weight":
            mapped["model.embed_tokens.weight"] = tens
            mapped["lm_head.weight"] = tens
            continue
        if name == "output.weight":
            mapped["lm_head.weight"] = tens
            continue
        if not name.startswith("blk."):
            continue

        parts = name.split(".", 2)
        bid = int(parts[1])
        suffix = parts[2]
        prefix = f"model.layers.{bid}"

        if suffix == "ffn_gate.weight":
            mapped[f"{prefix}.mlp.gate_proj.weight"] = tens
        elif suffix == "ffn_up.weight":
            mapped[f"{prefix}.mlp.up_proj.weight"] = tens
        elif suffix == "ffn_down.weight":
            mapped[f"{prefix}.mlp.down_proj.weight"] = tens
        elif bid in FULL_ATTN_LAYERS:
            if suffix == "attn_q.weight":
                mapped[f"{prefix}.self_attn.q_proj.weight"] = tens
            elif suffix == "attn_k.weight":
                mapped[f"{prefix}.self_attn.k_proj.weight"] = tens
            elif suffix == "attn_v.weight":
                mapped[f"{prefix}.self_attn.v_proj.weight"] = tens
            elif suffix == "attn_output.weight":
                mapped[f"{prefix}.self_attn.o_proj.weight"] = tens
        else:
            if suffix == "attn_qkv_gate.weight":
                # Verified exact slice: [:6144] = in_proj_qkv, [6144:] = in_proj_z
                mapped[f"{prefix}.linear_attn.in_proj_qkv.weight"] = tens[:6144, :].contiguous()
                mapped[f"{prefix}.linear_attn.in_proj_z.weight"] = tens[6144:, :].contiguous()
            elif suffix == "ssm_beta_alpha.weight":
                # Verified exact slice: [:16] = in_proj_b, [16:] = in_proj_a
                mapped[f"{prefix}.linear_attn.in_proj_b.weight"] = tens[:16, :].contiguous()
                mapped[f"{prefix}.linear_attn.in_proj_a.weight"] = tens[16:, :].contiguous()
            elif suffix == "ssm_out.weight":
                mapped[f"{prefix}.linear_attn.out_proj.weight"] = tens

    return mapped


@torch.inference_mode()
def _probe_model_internal(
    runner: QwenDeciderRunner,
    prompt_ids: list[torch.Tensor],
    ab_ids: list[int],
) -> dict[str, Any]:
    runner.model.eval()
    h15_l, g15_l, m15_l = [], [], []
    h22_l, g22_l, m22_l = [], [], []
    margins: list[float] = []

    handles = [
        runner.model.model.layers[15].mlp.register_forward_pre_hook(
            lambda _m, inp: h15_l.append(inp[0][0, -1, :].detach().float().cpu())
        ),
        runner.model.model.layers[15].mlp.down_proj.register_forward_pre_hook(
            lambda _m, inp: g15_l.append(inp[0][0, -1, :].detach().float().cpu())
        ),
        runner.model.model.layers[15].mlp.register_forward_hook(
            lambda _m, _inp, out: m15_l.append(out[0, -1, :].detach().float().cpu())
        ),
        runner.model.model.layers[22].mlp.register_forward_pre_hook(
            lambda _m, inp: h22_l.append(inp[0][0, -1, :].detach().float().cpu())
        ),
        runner.model.model.layers[22].mlp.down_proj.register_forward_pre_hook(
            lambda _m, inp: g22_l.append(inp[0][0, -1, :].detach().float().cpu())
        ),
        runner.model.model.layers[22].mlp.register_forward_hook(
            lambda _m, _inp, out: m22_l.append(out[0, -1, :].detach().float().cpu())
        ),
    ]
    lm_w = runner.model.lm_head.weight[ab_ids]
    lm_b = runner.model.lm_head.bias[ab_ids] if getattr(runner.model.lm_head, "bias", None) is not None else None
    try:
        for ids in prompt_ids:
            hid = runner.model.model(input_ids=ids.to(runner.device), use_cache=False).last_hidden_state[:, -1:, :]
            lg = F.linear(hid.to(dtype=lm_w.dtype), lm_w, lm_b)[0, 0, :].float()
            margins.append(float((lg[1] - lg[0]).item()))
    finally:
        for h in handles:
            h.remove()

    return {
        "h15": torch.stack(h15_l, dim=0),
        "g15": torch.stack(g15_l, dim=0),
        "m15": torch.stack(m15_l, dim=0),
        "h22": torch.stack(h22_l, dim=0),
        "g22": torch.stack(g22_l, dim=0),
        "m22": torch.stack(m22_l, dim=0),
        "margins": np.array(margins, dtype=np.float64),
    }


def _apply_dequant_dict(model: torch.nn.Module, base_cpu: dict[str, torch.Tensor], deq: dict[str, torch.Tensor]) -> None:
    restore_cpu_state_dict(model, base_cpu)
    named = dict(model.named_parameters())
    for k, val in deq.items():
        if k in named:
            named[k].data.copy_(val.to(device=named[k].device, dtype=named[k].dtype))


def _compare_probes(
    cur: dict[str, Any],
    ref: dict[str, Any],
    l15_mask_untouched: torch.Tensor,
    l22_mask_untouched: torch.Tensor,
    llama_q_m: np.ndarray | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for layer_tag, idx, unt_mask in (
        ("l15", T2_L15_NEURONS, l15_mask_untouched),
        ("l22", T2_L22_NEURONS, l22_mask_untouched),
    ):
        hk = f"h{layer_tag[1:]}"
        gk = f"g{layer_tag[1:]}"
        mk = f"m{layer_tag[1:]}"
        hc = cur[hk]
        hr = ref[hk]
        cos = F.cosine_similarity(hc, hr, dim=-1)
        rel_h = torch.linalg.norm(hc - hr, dim=-1) / torch.linalg.norm(hr, dim=-1).clamp_min(1e-8)
        abs_h = torch.linalg.norm(hc - hr, dim=-1)

        gc_e = cur[gk][:, idx]
        gr_e = ref[gk][:, idx]
        dg_e = (gc_e - gr_e).abs()
        gc_u = cur[gk][:, unt_mask]
        gr_u = ref[gk][:, unt_mask]
        dg_u = (gc_u - gr_u).abs()

        mc = cur[mk]
        mr = ref[mk]
        rel_m = torch.linalg.norm(mc - mr, dim=-1) / torch.linalg.norm(mr, dim=-1).clamp_min(1e-8)
        abs_m = torch.linalg.norm(mc - mr, dim=-1)

        ed_mean = float(dg_e.mean().item())
        ed_rms = float(torch.sqrt((dg_e * dg_e).mean()).item())
        un_mean = float(dg_u.mean().item())
        un_rms = float(torch.sqrt((dg_u * dg_u).mean()).item())

        out[layer_tag] = {
            "h_cos_vs_bf16": round(float(cos.mean().item()), 6),
            "h_rel_l2_drift": round(float(rel_h.mean().item()), 5),
            "h_abs_l2_drift": round(float(abs_h.mean().item()), 4),
            "edited4_bf16_mean_abs_act": round(float(gr_e.abs().mean().item()), 4),
            "edited4_q_mean_abs_act": round(float(gc_e.abs().mean().item()), 4),
            "edited4_mean_abs_delta": round(ed_mean, 5),
            "edited4_rms_delta": round(ed_rms, 5),
            "edited4_max_abs_delta": round(float(dg_e.max().item()), 4),
            "untouched_bf16_mean_abs_act": round(float(gr_u.abs().mean().item()), 5),
            "untouched_mean_abs_delta": round(un_mean, 5),
            "untouched_rms_delta": round(un_rms, 5),
            "amplification_vs_untouched_mean": round(ed_mean / max(1e-8, un_mean), 2),
            "amplification_vs_untouched_rms": round(ed_rms / max(1e-8, un_rms), 2),
            "mlp_out_rel_l2_drift": round(float(rel_m.mean().item()), 5),
            "mlp_out_abs_l2_drift": round(float(abs_m.mean().item()), 4),
        }

    cm = cur["margins"]
    rm = ref["margins"]
    cp = 1.0 / (1.0 + np.exp(-np.clip(cm, -30.0, 30.0)))
    rp = 1.0 / (1.0 + np.exp(-np.clip(rm, -30.0, 30.0)))
    out["logits"] = {
        "pt_deq_bf16_mean_abs_margin": round(float(np.mean(np.abs(rm))), 4),
        "margin_mae": round(float(np.mean(np.abs(cm - rm))), 4),
        "margin_rms_delta": round(float(np.sqrt(np.mean((cm - rm) ** 2))), 4),
        "signed_margin_shift": round(float(np.mean(cm - rm)), 4),
        "pearson_r_margin": round(float(np.corrcoef(rm, cm)[0, 1]), 4),
        "pearson_r_p_block": round(float(np.corrcoef(rp, cp)[0, 1]), 4),
        "p_block_mae": round(float(np.mean(np.abs(cp - rp))), 5),
    }
    if llama_q_m is not None:
        lq_p = 1.0 / (1.0 + np.exp(-np.clip(llama_q_m, -30.0, 30.0)))
        out["logits"]["pearson_r_margin_vs_llama_cpp"] = round(float(np.corrcoef(llama_q_m, cm)[0, 1]), 4)
        out["logits"]["pearson_r_p_vs_llama_cpp"] = round(float(np.corrcoef(lq_p, cp)[0, 1]), 4)
    return out


def run_r5b(built_paths: dict[str, Path], r5a_vecs: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    """Run R5b: Direct PyTorch probing of dequantized 4-bit weights vs BF16 refs."""
    print("[R5b] Probing direct 4-bit internal h_15/h_22 drift & SwiGLU gate amplification...", flush=True)
    dev_rows = load_split_rows("select_dev")
    bin_rows = [r for sub in ("real_cf", "consent_5p", "long_50t") for r in dev_rows if r["subset"] == sub]

    runner = QwenDeciderRunner()
    base_cpu = capture_cpu_state_dict(runner.model)
    ab_ids = [runner.letter_token_ids["A"], runner.letter_token_ids["B"]]
    prompt_ids: list[torch.Tensor] = []
    native_t2_margins: list[float] = []

    for r in bin_rows:
        budget = 3200 if r["subset"] == "long_50t" else 2048
        ctx = render_view(r, tok=runner.tokenizer, budget=budget, include_recap=False).text
        pr = build_plain_prompt(ctx, Q_BLOCK_BINARY, False).text
        ids = runner.tokenizer.encode(pr, add_special_tokens=False)
        prompt_ids.append(torch.tensor([ids], dtype=torch.long))

    # Measure native Track 2 margins before touching weights for BF16-GGUF sanity check
    t2_bundle = load_patch_bundle(CHECKPOINTS_DIR / "track2_best_deltas.pt")
    apply_bundle(runner.model, base_cpu, t2_bundle.patches)
    native_t2_probe = _probe_model_internal(runner, prompt_ids, ab_ids)
    native_t2_margins = native_t2_probe["margins"].copy()
    restore_cpu_state_dict(runner.model, base_cpu)

    t2_edit_set = set(TRACK2_GGUF_TENSORS)
    t2_bf16_gguf = CHECKPOINTS_DIR / "wsp-automode-0.8b-bf16.gguf"
    t2_q8_gguf = CHECKPOINTS_DIR / "wsp-automode-0.8b-Q8_0.gguf"
    t2_q4_gguf = CHECKPOINTS_DIR / "wsp-automode-0.8b-Q4_K_M.gguf"
    t3_bf16_gguf = CHECKPOINTS_DIR / "automode-0.8b-track3-bf16.gguf"
    t3_q4_gguf = CHECKPOINTS_DIR / "automode-0.8b-track3-Q4_K_M.gguf"

    print("[R5b] Dequantizing BF16/Q8_0/Q4_K_M GGUF weights for direct internal hooks...", flush=True)
    w_base_bf16 = _dequantize_gguf_2d_weights(built_paths["base_bf16"])
    w_base_q8 = _dequantize_gguf_2d_weights(built_paths["base_Q8_0"])
    w_base_q4 = _dequantize_gguf_2d_weights(built_paths["base_Q4_K_M"])

    w_t2_bf16 = _dequantize_gguf_2d_weights(t2_bf16_gguf)
    w_t2_q8 = _dequantize_gguf_2d_weights(t2_q8_gguf)
    w_t2_q4 = _dequantize_gguf_2d_weights(t2_q4_gguf)
    w_t2_bf16_edits = _dequantize_gguf_2d_weights(t2_bf16_gguf, filter_names=t2_edit_set)
    w_t2_q4_edits = _dequantize_gguf_2d_weights(t2_q4_gguf, filter_names=t2_edit_set)

    # For Track 3 Soup, apply Track 3's lossless non-2D params (norms + DeltaNet gates) then GGUF 2D weights
    t3_bundle = load_patch_bundle(CHECKPOINTS_DIR / "track3_final_best_deltas.pt")
    apply_bundle(runner.model, base_cpu, t3_bundle.patches)
    t3_full_cpu = capture_cpu_state_dict(runner.model)
    restore_cpu_state_dict(runner.model, base_cpu)
    w_t3_bf16 = _dequantize_gguf_2d_weights(t3_bf16_gguf)
    w_t3_q4 = _dequantize_gguf_2d_weights(t3_q4_gguf)

    # Probe references
    _apply_dequant_dict(runner.model, base_cpu, w_base_bf16)
    ref_base = _probe_model_internal(runner, prompt_ids, ab_ids)

    _apply_dequant_dict(runner.model, base_cpu, w_t2_bf16)
    ref_t2 = _probe_model_internal(runner, prompt_ids, ab_ids)

    sanity_max_diff = float(np.max(np.abs(ref_t2["margins"] - native_t2_margins)))
    sanity_mean_diff = float(np.mean(np.abs(ref_t2["margins"] - native_t2_margins)))
    sanity_r = float(np.corrcoef(ref_t2["margins"], native_t2_margins)[0, 1])
    print(
        f"[R5b] Sanity check Track 2 GGUF-BF16 vs Native PT: r={sanity_r:.6f}, mean|Δm|={sanity_mean_diff:.4f}, max|Δm|={sanity_max_diff:.4f}",
        flush=True,
    )
    if sanity_r < 0.998 or sanity_max_diff > 0.5:
        raise RuntimeError(f"GGUF dequantize sanity check failed: r={sanity_r}, max|Δm|={sanity_max_diff}")

    # Also log Track 3 GGUF-BF16 vs Native Track 3 sanity check
    apply_bundle(runner.model, base_cpu, t3_bundle.patches)
    native_t3_margins = _probe_model_internal(runner, prompt_ids, ab_ids)["margins"].copy()
    _apply_dequant_dict(runner.model, t3_full_cpu, w_t3_bf16)
    ref_t3 = _probe_model_internal(runner, prompt_ids, ab_ids)
    t3_r = float(np.corrcoef(ref_t3["margins"], native_t3_margins)[0, 1])
    t3_max = float(np.max(np.abs(ref_t3["margins"] - native_t3_margins)))
    print(f"[R5b] Sanity check Track 3 GGUF-BF16 vs Native PT: r={t3_r:.6f}, max|Δm|={t3_max:.4f}", flush=True)

    d_ff = int(ref_t2["g15"].shape[1])
    l15_untouched = torch.ones(d_ff, dtype=torch.bool)
    l15_untouched[T2_L15_NEURONS] = False
    l22_untouched = torch.ones(d_ff, dtype=torch.bool)
    l22_untouched[T2_L22_NEURONS] = False

    conds: list[tuple[str, dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any], np.ndarray | None]] = [
        ("base_Q8_0", base_cpu, w_base_q8, ref_base, r5a_vecs["base_Q8_0"]["q_m"]),
        ("base_Q4_K_M", base_cpu, w_base_q4, ref_base, r5a_vecs["base_Q4_K_M"]["q_m"]),
        ("track2_Q8_0", base_cpu, w_t2_q8, ref_t2, r5a_vecs["track2_Q8_0"]["q_m"]),
        ("track2_Q4_K_M", base_cpu, w_t2_q4, ref_t2, r5a_vecs["track2_Q4_K_M"]["q_m"]),
        (
            "track2_Q4_backbone_bf16_edits",
            base_cpu,
            {**w_t2_q4, **w_t2_bf16_edits},
            ref_t2,
            r5a_vecs["r2b_Q4_K_M_6x_bf16"]["q_m"],
        ),
        (
            "track2_bf16_backbone_Q4_edits",
            base_cpu,
            {**w_t2_bf16, **w_t2_q4_edits},
            ref_t2,
            None,
        ),
        ("track3_Q4_K_M", t3_full_cpu, w_t3_q4, ref_t3, r5a_vecs["track3_Q4_K_M"]["q_m"]),
    ]

    cond_results: dict[str, Any] = {}
    for cname, b_state, w_dict, ref_p, llama_qm in conds:
        print(f"[R5b] Probing {cname}...", flush=True)
        _apply_dequant_dict(runner.model, b_state, w_dict)
        cur_p = _probe_model_internal(runner, prompt_ids, ab_ids)
        cond_results[cname] = _compare_probes(cur_p, ref_p, l15_untouched, l22_untouched, llama_q_m=llama_qm)

    # Cross-model amplification ratios vs Base Q4_K_M on the exact same 4 L15/L22 neurons
    for cname in ("track2_Q4_K_M", "track2_Q4_backbone_bf16_edits", "track2_bf16_backbone_Q4_edits"):
        for lt in ("l15", "l22"):
            base_ed = cond_results["base_Q4_K_M"][lt]["edited4_mean_abs_delta"]
            cur_ed = cond_results[cname][lt]["edited4_mean_abs_delta"]
            cond_results[cname][lt]["amplification_vs_base_Q4_same4"] = round(cur_ed / max(1e-8, base_ed), 2)

    del runner, base_cpu, t3_full_cpu
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "n_prompts": len(prompt_ids),
        "d_ff": d_ff,
        "sanity_t2_gguf_bf16_vs_native_pt": {
            "pearson_r_margin": round(sanity_r, 6),
            "mean_abs_margin_diff": round(sanity_mean_diff, 5),
            "max_abs_margin_diff": round(sanity_max_diff, 5),
        },
        "sanity_t3_gguf_bf16_vs_native_pt": {
            "pearson_r_margin": round(t3_r, 6),
            "max_abs_margin_diff": round(t3_max, 5),
        },
        "conditions": cond_results,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Revision Suite R5 (Base Curve, Bootstrap CIs, Direct 4-Bit Amplification).")
    ap.add_argument("--port", type=int, default=8095)
    args = ap.parse_args()

    t0 = time.perf_counter()
    try:
        built_paths = _build_base_and_r2_ggufs()
        r5a, r5a_vecs = run_r5a(args.port, built_paths)
        payload = _load_controls_json()
        payload["r5_base_and_bootstrap_cis"] = r5a
        _save_controls_json(payload)
        print(f"[R5a] Saved -> {CONTROLS_JSON}", flush=True)

        r5b = run_r5b(built_paths, r5a_vecs)
        payload = _load_controls_json()
        payload["r5_base_and_bootstrap_cis"] = r5a
        payload["r5_activation_amplification"] = r5b
        _save_controls_json(payload)
        print(f"[R5] Complete in {time.perf_counter() - t0:.1f}s -> {CONTROLS_JSON}", flush=True)
    finally:
        if TMP_GGUF_DIR.exists():
            shutil.rmtree(TMP_GGUF_DIR, ignore_errors=True)
            print(f"[R5] Cleaned temp GGUF directory {TMP_GGUF_DIR}", flush=True)


if __name__ == "__main__":
    main()
