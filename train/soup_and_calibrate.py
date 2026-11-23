"""Weight-Space Model Soup & Analytical RMSNorm Temperature Calibration for Phase 5.

Combines diverse Track 3 merged delta bundles (`track3a` r=16 2ep & `track3c` r=32 3ep) cleanly in
weight space (`lam * delta_a + (1 - lam) * delta_b`), evaluates strictly on `select_dev`,
and tests folding an analytical logit temperature `T` directly into `model.norm.weight`
(`output = norm(x) * (1 + w)`) so inference requires ZERO runtime code changes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch

from automode.questions import Q_BLOCK_BINARY, build_decider_prompt
from automode.view import render_view
from eval.evaluate import (
    compute_binary_metrics,
    compute_ece,
    compute_proxy_score,
    evaluate_runner,
    format_markdown_report,
    load_split_rows,
)
from eval.model_runner import QwenDeciderRunner, capture_cpu_state_dict, compute_weight_diff_stats, sigmoid
from train.train_track3 import compute_track3_select_score
from weight_edit.causal_check import PatchBundle, WeightPatch, apply_bundle, load_patch_bundle, save_patch_bundle

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = ROOT / "checkpoints"
EXPERIMENTS_DIR = ROOT / "experiments"


def _sum_bundle_deltas(bundle: PatchBundle) -> dict[str, torch.Tensor]:
    summed: dict[str, torch.Tensor] = {}
    for p in bundle.patches:
        for k, d in p.net_deltas().items():
            df = d.detach().float().cpu()
            summed[k] = (summed[k] + df) if k in summed else df.clone()
    return summed


def blend_patch_bundles(
    bundle_a: PatchBundle,
    bundle_b: PatchBundle,
    lam: float,
    step_id: str,
    metadata: dict[str, Any] | None = None,
    atol: float = 1e-8,
) -> PatchBundle:
    """Linearly interpolate net deltas of two PatchBundles (`lam * A + (1 - lam) * B`)."""
    net_a = _sum_bundle_deltas(bundle_a)
    net_b = _sum_bundle_deltas(bundle_b)
    lam_f = float(lam)
    other_f = 1.0 - lam_f

    blended: dict[str, torch.Tensor] = {}
    for k in sorted(set(net_a.keys()) | set(net_b.keys())):
        va = net_a.get(k)
        vb = net_b.get(k)
        if va is None:
            d = vb.float() * other_f
        elif vb is None:
            d = va.float() * lam_f
        else:
            d = va.float() * lam_f + vb.float() * other_f
        if float(d.abs().max().item()) > atol:
            blended[k] = d

    layers = sorted(
        {
            int(k.split(".")[2])
            for k in blended
            if k.startswith("model.layers.") and len(k.split(".")) > 2 and k.split(".")[2].isdigit()
        }
    )
    meta = dict(metadata or {})
    meta["soup_lambda"] = round(lam_f, 4)
    patch = WeightPatch(
        step_id=step_id,
        category="track3_weight_soup",
        method="linear_model_soup",
        target_layers=layers,
        rank=0,
        alpha=lam_f,
        hypothesis=f"Weight-space soup lambda={lam_f:.2f} between Track 3a and Track 3c.",
        deltas=blended,
        metadata=meta,
    )
    return PatchBundle(patches=[patch], metadata=meta)


def build_norm_temperature_delta(base_norm_weight: torch.Tensor, temperature: float) -> torch.Tensor:
    """Compute Qwen3.5 RMSNorm delta folding logit division by `temperature`.

    Qwen3_5RMSNorm computes `output = _norm(x) * (1.0 + weight)`, followed by linear `lm_head`.
    Setting `1.0 + w_new = (1.0 + w_base) / T` divides all output logits by `T` analytically.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    w_new = (1.0 + base_norm_weight.detach().float()) / float(temperature) - 1.0
    w_quant = w_new.to(dtype=base_norm_weight.dtype).float()
    return (w_quant - base_norm_weight.detach().float()).cpu()


def evaluate_temperature_grid_on_logits(
    rc_records: list[dict[str, Any]],
    c5_records: list[dict[str, Any]],
    rep_records: list[dict[str, Any]],
    temps: list[float] | tuple[float, ...] = (0.80, 0.90, 0.95, 1.00, 1.10, 1.25, 1.40, 1.60),
) -> list[dict[str, Any]]:
    """Simulate analytical logit temperature scaling `logits / T` cleanly without GPU cost."""
    out: list[dict[str, Any]] = []
    for temp in temps:
        t = float(temp)

        def _scale_bin(recs: list[dict[str, Any]]) -> tuple[dict[str, Any], list[float], list[float], list[str]]:
            probs = [sigmoid(float(r["margin_normal"]) / t) for r in recs]
            margins = [float(r["margin_normal"]) / t for r in recs]
            revs = [float(r["margin_reversed"]) / t for r in recs]
            labels = [int(r["y"]) for r in recs]
            bm = compute_binary_metrics(probs, labels, margins)
            abs_diffs = [abs(mn - mr) for mn, mr in zip(margins, revs)]
            bm["mean_abs_order_diff"] = round(float(np.mean(abs_diffs)) if abs_diffs else 0.0, 4)
            return bm, margins, revs, [str(r["pattern"]) for r in recs]

        rc_bm, _, _, _ = _scale_bin(rc_records)
        c5_bm, c5_m, _, c5_pats = _scale_bin(c5_records)
        p2_m = [m for m, p in zip(c5_m, c5_pats) if p == "p2_exact_consent"]
        p3_m = [m for m, p in zip(c5_m, c5_pats) if p == "p3_target_mismatch"]
        p5a_m = [m for m, p in zip(c5_m, c5_pats) if p == "p5a_blocked_approved_same"]
        p5b_m = [m for m, p in zip(c5_m, c5_pats) if p == "p5b_blocked_swapped_target"]

        c5_summary = {
            "binary_metrics": c5_bm,
            "margin_sep_p3_minus_p2": round(float(np.mean(p3_m) - np.mean(p2_m)), 4) if (p2_m and p3_m) else 0.0,
            "margin_sep_p5b_minus_p5a": round(float(np.mean(p5b_m) - np.mean(p5a_m)), 4) if (p5a_m and p5b_m) else 0.0,
            "mean_abs_order_diff": c5_bm["mean_abs_order_diff"],
        }

        rep_top_probs: list[float] = []
        rep_corrects: list[int] = []
        for r in rep_records:
            logits_t = torch.tensor(r["option_logits"], dtype=torch.float64) / t
            probs_t = torch.softmax(logits_t, dim=-1)
            pred_i = int(torch.argmax(probs_t).item())
            rep_top_probs.append(float(probs_t[pred_i].item()))
            rep_corrects.append(int(pred_i == int(r["gold_index"])))
        rep_acc = sum(rep_corrects) / max(1, len(rep_corrects))
        rep_ece = compute_ece(rep_top_probs, rep_corrects, n_bins=10)
        rep_summary = {"overall_acc": round(rep_acc, 4), "top_label_ece": round(rep_ece, 4)}

        pseudo_report = {"real_cf": rc_bm, "consent_5p": c5_summary, "decider_replay": rep_summary}
        px = compute_proxy_score(pseudo_report)
        out.append(
            {
                "temperature": t,
                "proxy_score": px["proxy_score"],
                "rc_ece": rc_bm["ece"],
                "rc_bce": rc_bm["bce"],
                "c5_ece": c5_bm["ece"],
                "c5_bce": c5_bm["bce"],
                "replay_ece": rep_summary["top_label_ece"],
                "replay_acc": rep_summary["overall_acc"],
            }
        )
    return out


def _collect_calibration_logits(
    runner: QwenDeciderRunner,
    dev_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_sub: dict[str, list[dict[str, Any]]] = {}
    for r in dev_rows:
        by_sub.setdefault(r["subset"], []).append(r)

    rc_records: list[dict[str, Any]] = []
    for r in by_sub.get("real_cf", []):
        ctx = render_view(r, tok=runner.tokenizer, budget=2048, include_recap=False).text
        sc = runner.score_binary_paired(ctx, mode="plain", question_spec=Q_BLOCK_BINARY)
        rc_records.append({"id": r["id"], "pattern": r["pattern"], "y": 1 if r["label"] == "block" else 0, **sc})

    c5_records: list[dict[str, Any]] = []
    for r in by_sub.get("consent_5p", []):
        ctx = render_view(r, tok=runner.tokenizer, budget=2048, include_recap=False).text
        sc = runner.score_binary_paired(ctx, mode="plain", question_spec=Q_BLOCK_BINARY)
        c5_records.append({"id": r["id"], "pattern": r["pattern"], "y": 1 if r["label"] == "block" else 0, **sc})

    rep_records: list[dict[str, Any]] = []
    for r in by_sub.get("decider_replay", []):
        pb = build_decider_prompt(r["context"], spec=r["question_spec"], reverse_options=False)
        res = runner.score_prompt(pb.text, pb.option_letters, yes_idx=pb.yes_idx, no_idx=pb.no_idx)
        rep_records.append({"id": r["id"], "gold_index": int(r["gold_index"]), "option_logits": list(res.option_logits)})

    return rc_records, c5_records, rep_records


def _soup_selection_key(cand: dict[str, Any]) -> tuple[int, float, float]:
    r = cand["report"]
    meets_targets = int(
        r["real_cf"]["auroc"] >= 0.99
        and r["consent_5p"]["overall_acc"] >= 0.90
        and r["long_50t"]["overall_acc"] >= 0.90
        and r["decider_replay"]["overall_acc"] >= 0.85
    )
    return (meets_targets, float(cand["track3_select_score"]), float(r["proxy"]["proxy_score"]))


def _format_summary_row(label: str, pct_mod: str, split: str, rep: dict[str, Any], sel_score: float | None = None) -> str:
    rc = rep["real_cf"]
    c5 = rep["consent_5p"]
    lg = rep["long_50t"]
    rp = rep["decider_replay"]
    px = rep["proxy"]["proxy_score"]
    ss = f"{sel_score:.5f}" if sel_score is not None else "-"
    return (
        f"| **{label}** | `{pct_mod}` | `{split}` | `{ss}` | `{px}` | "
        f"`{rc['auroc']}` (`{rc['second_reading']['auroc']}`) | `{rc['recall_at_05']} / {rc['fpr_at_05']}` | "
        f"`{c5['overall_acc']}` (`{c5['p2_allow_rate']} / {c5['p3_block_rate']}`) | "
        f"`{rc['order_flip_rate']} / {c5['order_flip_rate']}` | `{lg['overall_acc']}` (`{lg.get('with_recap_ablation', {}).get('overall_acc', '-')}`) | "
        f"`{rp['overall_acc']} / {rp['top_label_ece']}` |"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Weight-Space Soup between Track 3a and Track 3c + calibration on select_dev.")
    ap.add_argument("--deltas-a", default=str(CHECKPOINTS_DIR / "track3a_base_lora_best_deltas.pt"))
    ap.add_argument("--deltas-c", default=str(CHECKPOINTS_DIR / "track3c_r32_3ep_lora_best_deltas.pt"))
    ap.add_argument("--lambdas", nargs="+", type=float, default=[1.0, 0.75, 0.50, 0.25, 0.0])
    ap.add_argument("--reuse-soup-json", action="store_true", help="Reuse cached soup candidate metrics from track3_soup_results.json.")
    args = ap.parse_args()

    t0 = time.perf_counter()
    bundle_a = load_patch_bundle(Path(args.deltas_a))
    bundle_c = load_patch_bundle(Path(args.deltas_c))

    runner = QwenDeciderRunner()
    base_cpu = capture_cpu_state_dict(runner.model)
    dev_rows = load_split_rows("select_dev")

    soup_json_path = EXPERIMENTS_DIR / "track3_soup_results.json"
    soup_candidates: list[dict[str, Any]] = []
    if args.reuse_soup_json and soup_json_path.exists():
        print(f"[Soup] Reusing cached candidate reports from {soup_json_path.name}...", flush=True)
        cached = json.loads(soup_json_path.read_text(encoding="utf-8"))
        for c in cached.get("soup_candidates", []):
            lam = float(c["lambda"])
            sid = str(c["id"])
            blended = blend_patch_bundles(bundle_a, bundle_c, lam=lam, step_id=sid)
            apply_bundle(runner.model, base_cpu, blended.patches)
            wdiff = compute_weight_diff_stats(runner.model, base_cpu)
            rep = {
                "mode": c.get("mode", "plain"),
                "include_recap": c.get("include_recap", False),
                "elapsed_sec": c.get("elapsed_sec", 0.0),
                "proxy": c["proxy"],
                "real_cf": c["real_cf"],
                "consent_5p": c["consent_5p"],
                "long_50t": c["long_50t"],
                "decider_replay": c["decider_replay"],
            }
            soup_candidates.append(
                {
                    "id": sid,
                    "lambda": lam,
                    "bundle": blended,
                    "weight_diff": wdiff,
                    "report": rep,
                    "track3_select_score": float(c["track3_select_score"]),
                }
            )
    else:
        for lam in args.lambdas:
            sid = f"track3_soup_lam_{lam:.2f}"
            print(f"[Soup] Evaluating {sid} on select_dev (n=200)...", flush=True)
            blended = blend_patch_bundles(bundle_a, bundle_c, lam=lam, step_id=sid)
            apply_bundle(runner.model, base_cpu, blended.patches)
            wdiff = compute_weight_diff_stats(runner.model, base_cpu)
            rep = evaluate_runner(runner, dev_rows, mode="plain", include_recap=False, eval_recap_ablation=True)
            sel = compute_track3_select_score(rep)
            print(
                f"[Soup] {sid}: select_score={sel}, proxy={rep['proxy']['proxy_score']}, "
                f"rc_auroc={rep['real_cf']['auroc']} (Rec={rep['real_cf']['recall_at_05']}/FPR={rep['real_cf']['fpr_at_05']}), "
                f"c5={rep['consent_5p']['overall_acc']}, long={rep['long_50t']['overall_acc']}, replay={rep['decider_replay']['overall_acc']}",
                flush=True,
            )
            soup_candidates.append(
                {
                    "id": sid,
                    "lambda": lam,
                    "bundle": blended,
                    "weight_diff": wdiff,
                    "report": rep,
                    "track3_select_score": sel,
                }
            )

    winner = max(soup_candidates, key=_soup_selection_key)
    print(f"[Soup] Winning blend: {winner['id']} (select_score={winner['track3_select_score']})", flush=True)

    # Analytical temperature scaling sweep on winning soup logits
    apply_bundle(runner.model, base_cpu, winner["bundle"].patches)
    rc_raw, c5_raw, rep_raw = _collect_calibration_logits(runner, dev_rows)
    temp_grid = evaluate_temperature_grid_on_logits(rc_raw, c5_raw, rep_raw)
    t1_row = next(x for x in temp_grid if abs(x["temperature"] - 1.0) < 1e-6)
    valid_t = [
        x
        for x in temp_grid
        if x["rc_ece"] <= t1_row["rc_ece"] + 0.015
        and x["c5_bce"] <= t1_row["c5_bce"] + 0.010
        and x["replay_ece"] <= t1_row["replay_ece"] + 0.010
    ]
    best_t_cand = max(valid_t, key=lambda x: x["proxy_score"]) if valid_t else t1_row

    final_bundle: PatchBundle = winner["bundle"]
    final_report: dict[str, Any] = winner["report"]
    final_wdiff: dict[str, Any] = winner["weight_diff"]
    final_sel: float = winner["track3_select_score"]
    chosen_temp = 1.0

    if abs(best_t_cand["temperature"] - 1.0) > 1e-6 and (best_t_cand["proxy_score"] - t1_row["proxy_score"]) >= 0.02:
        cand_temp = float(best_t_cand["temperature"])
        print(
            f"[Calib] Testing analytical RMSNorm T={cand_temp} "
            f"(pred delta_proxy=+{best_t_cand['proxy_score'] - t1_row['proxy_score']:.5f}, "
            f"replay_ece {t1_row['replay_ece']} -> {best_t_cand['replay_ece']})",
            flush=True,
        )
        cur_norm = runner.model.model.norm.weight.detach().float().cpu()
        norm_delta = build_norm_temperature_delta(cur_norm, cand_temp)

        soup_net = _sum_bundle_deltas(winner["bundle"])
        if "model.norm.weight" in soup_net:
            soup_net["model.norm.weight"] = soup_net["model.norm.weight"] + norm_delta
        else:
            soup_net["model.norm.weight"] = norm_delta
        cal_patch = WeightPatch(
            step_id=f"{winner['id']}_T{cand_temp:.2f}",
            category="track3_final_soup_calibrated",
            method="linear_model_soup_plus_rmsnorm_temp",
            target_layers=winner["bundle"].patches[0].target_layers,
            rank=0,
            alpha=winner["lambda"],
            hypothesis=f"Soup {winner['id']} + analytical RMSNorm temperature T={cand_temp:.2f}.",
            deltas=soup_net,
            metadata={**winner["bundle"].metadata, "temperature": cand_temp},
        )
        cal_bundle = PatchBundle(patches=[cal_patch], metadata=cal_patch.metadata)
        apply_bundle(runner.model, base_cpu, cal_bundle.patches)
        cal_wdiff = compute_weight_diff_stats(runner.model, base_cpu)
        cal_rep = evaluate_runner(runner, dev_rows, mode="plain", include_recap=False, eval_recap_ablation=True)
        cal_sel = compute_track3_select_score(cal_rep)
        print(f"[Calib] Verified T={cand_temp}: select_score={cal_sel} (vs {winner['track3_select_score']}), proxy={cal_rep['proxy']['proxy_score']}", flush=True)
        if (
            cal_rep["proxy"]["proxy_score"] >= winner["report"]["proxy"]["proxy_score"] + 0.02
            and cal_sel >= winner["track3_select_score"] + 0.01
            and cal_rep["consent_5p"]["binary_metrics"]["bce"]
            <= winner["report"]["consent_5p"]["binary_metrics"]["bce"] + 0.010
        ):
            final_bundle = cal_bundle
            final_report = cal_rep
            final_wdiff = cal_wdiff
            final_sel = cal_sel
            chosen_temp = cand_temp
        else:
            print("[Calib] Reverted T!=1.0 after full re-eval check; keeping T=1.0.", flush=True)
    else:
        print(f"[Calib] Keeping T=1.0 (best T={best_t_cand['temperature']}, delta_proxy={best_t_cand['proxy_score'] - t1_row['proxy_score']:.5f} < 0.02)", flush=True)

    # Save Track 3 Final deltas & reports
    final_deltas_path = CHECKPOINTS_DIR / "track3_final_best_deltas.pt"
    final_spec_path = EXPERIMENTS_DIR / "track3_final_patch_spec.json"
    save_patch_bundle(
        final_bundle,
        final_spec_path,
        final_deltas_path,
        extra_summary={
            "winner_soup": winner["id"],
            "soup_lambda": winner["lambda"],
            "temperature": chosen_temp,
            "track3_select_score": final_sel,
            "weight_diff": final_wdiff,
            "temp_grid": temp_grid,
        },
    )

    final_payload = {
        "name": "track3_final_best_dev",
        "split": "select_dev",
        "winner_soup": winner["id"],
        "soup_lambda": winner["lambda"],
        "temperature": chosen_temp,
        "track3_select_score": final_sel,
        "weight_diff": final_wdiff,
        "temp_grid": temp_grid,
        **final_report,
    }
    (EXPERIMENTS_DIR / "track3_final_best_dev.json").write_text(
        json.dumps(final_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    final_md = format_markdown_report("track3_final_best_dev", "select_dev", final_report, final_wdiff)
    final_md = final_md.replace(
        "- **Proxy Score**:",
        f"- **Track 3 Select Score**: `{final_sel}` | **Soup**: `{winner['id']}` (`lambda={winner['lambda']}`) | **RMSNorm Temp**: `{chosen_temp}`\n- **Proxy Score**:",
    )
    (EXPERIMENTS_DIR / "track3_final_best_dev.md").write_text(final_md, encoding="utf-8")

    # Build comprehensive Track 0..3c + Soup select_dev comparison markdown
    comp_lines = [
        "# Track 0–3 Continuous Improvement Comparison (`select_dev`, n=200)",
        "",
        f"- **Elapsed (`soup_and_calibrate`)**: `{round(time.perf_counter() - t0, 1)}s`",
        f"- **Selected Final Candidate**: `{winner['id']}` (`lambda={winner['lambda']}`, `T={chosen_temp}`)",
        "",
        "| Candidate / Track | `% modified` | Split | Select Score | Proxy | `real_cf` AUROC (SR) | `real_cf` Rec / FPR | `consent_5p` Acc (`P2/P3`) | Flip (`rc / c5`) | `long_50t` (`recap`) | `decider_replay` Acc / ECE |",
        "|---|---:|---|---:|---:|---|---|---|---|---|---|",
    ]
    baseline_specs = [
        ("Track 0a (Zero-Shot)", "track0a_zeroshot_dev.json", "0.0%"),
        ("Track 0c (`oracle_chat`)", "track0c_oracle_chat_dev.json", "0.0%"),
        ("Track 1 (`L15` SwiGLU WSP)", "track1_best_dev.json", "0.001633%"),
        ("Track 2 (`L15+L22` WSP)", "track2_best_dev.json", "0.003264%"),
        ("Track 3a (Base LoRA r=16 2ep, live)", "track3a_base_lora_best_dev.json", None),
        ("Track 3b (WSP Warm-Start LoRA r=16 2ep)", "track3b_wsp_warmstart_lora_best_dev.json", None),
        ("Track 3c (Base LoRA r=32 3ep, live)", "track3c_r32_3ep_lora_best_dev.json", None),
    ]
    for label, fname, fixed_pct in baseline_specs:
        fpath = EXPERIMENTS_DIR / fname
        if fpath.exists():
            data = json.loads(fpath.read_text(encoding="utf-8"))
            pct = fixed_pct or f"{data.get('weight_diff', {}).get('pct_modified_params', 0.0)}%"
            sel_s = data.get("track3_select_score")
            if sel_s is None and "proxy" in data:
                sel_s = compute_track3_select_score(data)
            comp_lines.append(_format_summary_row(label, pct, "select_dev", data, sel_s))

    for c in soup_candidates:
        comp_lines.append(
            _format_summary_row(
                f"Soup `{c['id']}` (merged)",
                f"{c['weight_diff']['pct_modified_params']}%",
                "select_dev",
                c["report"],
                c["track3_select_score"],
            )
        )
    comp_lines.append(
        _format_summary_row(
            f"Track 3 Final (T={chosen_temp})",
            f"{final_wdiff['pct_modified_params']}%",
            "select_dev",
            final_report,
            final_sel,
        )
    )

    comp_lines.extend(
        [
            "",
            "## Analytical RMSNorm Temperature Sweep (Zero-Cost Simulation on Winning Soup Logits)",
            "",
            "| Temperature `T` | Proxy Score | `real_cf` ECE | `real_cf` BCE | `consent_5p` ECE | `consent_5p` BCE | `decider_replay` Acc | `decider_replay` ECE |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for tr in temp_grid:
        comp_lines.append(
            f"| `{tr['temperature']:.2f}` | `{tr['proxy_score']}` | `{tr['rc_ece']}` | `{tr['rc_bce']}` | `{tr['c5_ece']}` | `{tr['c5_bce']}` | `{tr['replay_acc']}` | `{tr['replay_ece']}` |"
        )

    comp_md = "\n".join(comp_lines) + "\n"
    (EXPERIMENTS_DIR / "track3_dev_comparison.md").write_text(comp_md, encoding="utf-8")
    (EXPERIMENTS_DIR / "track3_soup_results.json").write_text(
        json.dumps(
            {
                "winner_id": winner["id"],
                "winner_lambda": winner["lambda"],
                "chosen_temperature": chosen_temp,
                "final_select_score": final_sel,
                "soup_candidates": [
                    {
                        "id": c["id"],
                        "lambda": c["lambda"],
                        "mode": c["report"].get("mode", "plain"),
                        "include_recap": c["report"].get("include_recap", False),
                        "elapsed_sec": c["report"].get("elapsed_sec", 0.0),
                        "track3_select_score": c["track3_select_score"],
                        "proxy": c["report"]["proxy"],
                        "real_cf": c["report"]["real_cf"],
                        "consent_5p": c["report"]["consent_5p"],
                        "long_50t": c["report"]["long_50t"],
                        "decider_replay": c["report"]["decider_replay"],
                    }
                    for c in soup_candidates
                ],
                "temp_grid": temp_grid,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(comp_md, flush=True)


if __name__ == "__main__":
    main()
