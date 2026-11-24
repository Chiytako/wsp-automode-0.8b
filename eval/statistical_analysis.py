"""Statistical significance testing, 10,000-resample stratified bootstrap CIs, and error analysis (Step 4).

Reads `experiments/paper_external_frozen.json`, `experiments/paper_adversarial_results.json`, and
`data/adversarial_redteam.jsonl` (CPU-only, deterministic seed=20261124). Computes:
  1. 10,000-resample stratified bootstrap 95% CIs across all subsets/categories for all 5+1 models.
  2. Exact two-sided McNemar binomial tests & paired bootstrap difference tests with Holm-Bonferroni
     FWER correction across pairwise model comparisons and Blind vs. Leaky ablations.
  3. Sanitized fine-grained error analysis (counts by pattern/category on `frozen_test`, plus concrete
     synthetic failure cases on `adversarial_redteam`).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

from data.build_adversarial_bench import ADV_PATH

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = ROOT / "experiments"
SEED = 20261124
N_BOOT = 10000


def _exact_mcnemar(c1: list[int], c2: list[int]) -> dict[str, Any]:
    """Two-sided exact McNemar test using scipy.stats.binomtest on discordant pairs."""
    a = np.asarray(c1, dtype=np.int64)
    b = np.asarray(c2, dtype=np.int64)
    if a.shape != b.shape or len(a) == 0:
        raise ValueError(f"Mismatched arrays for McNemar: {a.shape} vs {b.shape}")
    n01 = int(np.sum((a == 0) & (b == 1)))  # model 2 right, model 1 wrong
    n10 = int(np.sum((a == 1) & (b == 0)))  # model 1 right, model 2 wrong
    n_disc = n01 + n10
    p_val = 1.0 if n_disc == 0 else float(binomtest(min(n01, n10), n=n_disc, p=0.5, alternative="two-sided").pvalue)
    return {
        "n": int(len(a)),
        "acc_a": round(float(np.mean(a)), 4),
        "acc_b": round(float(np.mean(b)), 4),
        "delta_acc_a_minus_b": round(float(np.mean(a) - np.mean(b)), 4),
        "n_a_only": n10,
        "n_b_only": n01,
        "p_value": p_val,
    }


def _bootstrap_proportion_ci(
    hits: np.ndarray,
    strata: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
) -> dict[str, float]:
    n = len(hits)
    uniq_s = np.unique(strata)
    idx_groups = [np.where(strata == s)[0] for s in uniq_s]

    boot_vals = np.empty(n_boot, dtype=np.float64)
    # Draw all stratified sample indices vectorized per stratum
    sum_hits = np.zeros(n_boot, dtype=np.float64)
    for g in idx_groups:
        m = len(g)
        picks = g[rng.integers(0, m, size=(n_boot, m))]
        sum_hits += hits[picks].sum(axis=1)
    boot_vals = sum_hits / n
    lo, hi = np.percentile(boot_vals, [2.5, 97.5])
    return {
        "point": round(float(np.mean(hits)), 4),
        "ci95_lo": round(float(lo), 4),
        "ci95_hi": round(float(hi), 4),
    }


def _bootstrap_auroc(
    probs: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
) -> tuple[dict[str, float], np.ndarray]:
    pos_p = probs[labels == 1]
    neg_p = probs[labels == 0]
    n_pos = len(pos_p)
    n_neg = len(neg_p)
    # Exact pairwise Mann-Whitney comparison matrix [P, N]
    mat = (pos_p[:, None] > neg_p[None, :]).astype(np.float64) + 0.5 * (
        pos_p[:, None] == neg_p[None, :]
    ).astype(np.float64)
    point = float(mat.mean())
    sk_point = float(roc_auc_score(labels, probs))
    if abs(point - sk_point) > 1e-6:
        raise AssertionError(f"AUROC matrix mismatch: {point} vs {sk_point}")

    boot = np.empty(n_boot, dtype=np.float64)
    chunk = 500
    for s in range(0, n_boot, chunk):
        e = min(n_boot, s + chunk)
        k = e - s
        ip = rng.integers(0, n_pos, size=(k, n_pos))
        ineg = rng.integers(0, n_neg, size=(k, n_neg))
        sub_m = mat[ip[:, :, None], ineg[:, None, :]]
        boot[s:e] = sub_m.mean(axis=(1, 2))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "point": round(point, 4),
        "ci95_lo": round(float(lo), 4),
        "ci95_hi": round(float(hi), 4),
    }, boot


def _paired_bootstrap_auroc_diff(
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
) -> dict[str, Any]:
    pos_a, neg_a = probs_a[labels == 1], probs_a[labels == 0]
    pos_b, neg_b = probs_b[labels == 1], probs_b[labels == 0]
    n_pos, n_neg = len(pos_a), len(neg_a)
    mat_a = (pos_a[:, None] > neg_a[None, :]).astype(np.float64) + 0.5 * (pos_a[:, None] == neg_a[None, :]).astype(np.float64)
    mat_b = (pos_b[:, None] > neg_b[None, :]).astype(np.float64) + 0.5 * (pos_b[:, None] == neg_b[None, :]).astype(np.float64)
    diff_point = float(mat_a.mean() - mat_b.mean())

    diffs = np.empty(n_boot, dtype=np.float64)
    chunk = 500
    for s in range(0, n_boot, chunk):
        e = min(n_boot, s + chunk)
        k = e - s
        ip = rng.integers(0, n_pos, size=(k, n_pos))
        ineg = rng.integers(0, n_neg, size=(k, n_neg))
        auc_a = mat_a[ip[:, :, None], ineg[:, None, :]].mean(axis=(1, 2))
        auc_b = mat_b[ip[:, :, None], ineg[:, None, :]].mean(axis=(1, 2))
        diffs[s:e] = auc_a - auc_b

    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # Two-sided paired bootstrap p-value against H0: diff == 0
    if np.all(diffs == 0.0):
        p_val = 1.0
    else:
        p_le = float((np.sum(diffs <= 0.0) + 1) / (n_boot + 1))
        p_ge = float((np.sum(diffs >= 0.0) + 1) / (n_boot + 1))
        p_val = min(1.0, 2.0 * min(p_le, p_ge))
    return {
        "auroc_a": round(float(mat_a.mean()), 4),
        "auroc_b": round(float(mat_b.mean()), 4),
        "delta_auroc_a_minus_b": round(diff_point, 4),
        "ci95_lo": round(float(lo), 4),
        "ci95_hi": round(float(hi), 4),
        "p_value": round(p_val, 6),
    }


def _holm_adjust(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [1.0] * m
    prev_max = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, pvals[idx] * (m - rank))
        val = max(prev_max, val)
        prev_max = val
        adj[idx] = float(val)
    return adj


def run_statistical_analysis() -> dict[str, Any]:
    ext = json.loads((EXPERIMENTS_DIR / "paper_external_frozen.json").read_text(encoding="utf-8"))
    adv = json.loads((EXPERIMENTS_DIR / "paper_adversarial_results.json").read_text(encoding="utf-8"))
    adv_raw = {
        r["id"]: r
        for r in (json.loads(ln) for ln in ADV_PATH.read_text(encoding="utf-8").splitlines() if ln.strip())
    }

    rng = np.random.default_rng(SEED)
    model_keys = list(ext["models"].keys())

    # 1. Stratified Bootstrap CIs per model on frozen_test & adversarial_redteam
    frozen_cis: dict[str, Any] = {}
    adv_cis: dict[str, Any] = {}

    frozen_item_arrays: dict[str, dict[str, Any]] = {}
    adv_item_arrays: dict[str, dict[str, Any]] = {}

    for mk in model_keys:
        fm = ext["models"][mk]
        bin_recs = fm["binary_records"]
        rep_recs = fm["replay_records"]

        rc = [x for x in bin_recs if x["subset"] == "real_cf"]
        c5 = [x for x in bin_recs if x["subset"] == "consent_5p"]
        lg = [x for x in bin_recs if x["subset"] == "long_50t"]

        rc_y = np.array([x["y"] for x in rc], dtype=np.int64)
        rc_p = np.array([x["p_block"] for x in rc], dtype=np.float64)
        rc_corr = np.array([int((x["pred_norm"] == "block") == (x["y"] == 1)) for x in rc], dtype=np.int64)
        rc_pos = rc_corr[rc_y == 1]
        rc_neg_fp = (1 - rc_corr[rc_y == 0])
        rc_flip = np.array([int(x["flipped"]) for x in rc], dtype=np.int64)

        c5_pats = np.array([x["pattern"] for x in c5])
        c5_corr = np.array([int((x["pred_norm"] == "block") == (x["y"] == 1)) for x in c5], dtype=np.int64)

        lg_y = np.array([x["y"] for x in lg], dtype=np.int64)
        lg_corr = np.array([int((x["pred_norm"] == "block") == (x["y"] == 1)) for x in lg], dtype=np.int64)

        rp_qt = np.array([x["q_type"] for x in rep_recs])
        rp_corr = np.array([int(x["correct"]) for x in rep_recs], dtype=np.int64)

        rc_auc_ci, _ = _bootstrap_auroc(rc_p, rc_y, rng)
        frozen_cis[mk] = {
            "real_cf_auroc": rc_auc_ci,
            "real_cf_acc": _bootstrap_proportion_ci(rc_corr, rc_y, rng),
            "real_cf_recall": _bootstrap_proportion_ci(rc_pos, np.ones_like(rc_pos), rng),
            "real_cf_fpr": _bootstrap_proportion_ci(rc_neg_fp, np.zeros_like(rc_neg_fp), rng),
            "real_cf_flip": _bootstrap_proportion_ci(rc_flip, rc_y, rng),
            "consent_5p_acc": _bootstrap_proportion_ci(c5_corr, c5_pats, rng),
            "long_50t_acc": _bootstrap_proportion_ci(lg_corr, lg_y, rng),
            "decider_replay_acc": _bootstrap_proportion_ci(rp_corr, rp_qt, rng),
        }
        frozen_item_arrays[mk] = {
            "rc_y": rc_y,
            "rc_p": rc_p,
            "rc_corr": rc_corr,
            "c5_corr": c5_corr,
            "lg_corr": lg_corr,
            "rp_corr": rp_corr,
            "all_bin_corr": np.concatenate([rc_corr, c5_corr, lg_corr]),
            "all_200_corr": np.concatenate([rc_corr, c5_corr, lg_corr, rp_corr]),
        }

        am = adv["models"][mk]
        abl = am["blind_records"]
        alk = am["leaky_records"]
        ab_y = np.array([x["y"] for x in abl], dtype=np.int64)
        ab_p = np.array([x["p_block"] for x in abl], dtype=np.float64)
        ab_cats = np.array([x["category"] for x in abl])
        ab_corr = np.array([int((x["pred_norm"] == "block") == (x["y"] == 1)) for x in abl], dtype=np.int64)
        ab_atk = (1 - ab_corr[ab_y == 1])  # ASR indicator
        ab_ben = (1 - ab_corr[ab_y == 0])  # Benign FPR indicator

        ab_auc_ci, _ = _bootstrap_auroc(ab_p, ab_y, rng)
        cat_cis = {}
        for cat in sorted(set(ab_cats.tolist())):
            mask = ab_cats == cat
            cat_cis[cat] = _bootstrap_proportion_ci(ab_corr[mask], ab_y[mask], rng)

        rb_blind_arr = np.array(
            [int(x["pred_norm"] != "block") for x in abl if x["category"] == "reasoning_blind_ablation"], dtype=np.int64
        )
        rb_leaky_arr = np.array([int(x["pred_norm"] != "block") for x in alk], dtype=np.int64)
        rb_delta = rb_leaky_arr - rb_blind_arr

        adv_cis[mk] = {
            "auroc": ab_auc_ci,
            "overall_acc": _bootstrap_proportion_ci(ab_corr, ab_cats, rng),
            "overall_asr": _bootstrap_proportion_ci(ab_atk, ab_cats[ab_y == 1], rng),
            "benign_fpr": _bootstrap_proportion_ci(ab_ben, ab_cats[ab_y == 0], rng),
            "by_category_acc": cat_cis,
            "rb_blind_asr": _bootstrap_proportion_ci(rb_blind_arr, np.ones_like(rb_blind_arr), rng),
            "rb_leaky_asr": _bootstrap_proportion_ci(rb_leaky_arr, np.ones_like(rb_leaky_arr), rng),
            "rb_delta_asr_ci": _bootstrap_proportion_ci(rb_delta.astype(np.float64), np.ones_like(rb_delta), rng),
        }
        adv_item_arrays[mk] = {
            "ab_y": ab_y,
            "ab_p": ab_p,
            "ab_corr": ab_corr,
            "ab_atk_rec": ab_corr[ab_y == 1],
            "ab_ben_acc": ab_corr[ab_y == 0],
            "rb_blind_rec": (1 - rb_blind_arr),
            "rb_leaky_rec": (1 - rb_leaky_arr),
        }

    # 2. Pairwise Hypothesis Testing + Holm-Bonferroni
    comparison_pairs = [
        ("ours_track2_wsp", "base_qwen35_0.8b"),
        ("ours_track2_wsp", "mit_wsp_0.8b"),
        ("ours_track2_wsp", "mapika_decider_0.8b"),
        ("ours_track3_soup_bf16", "ours_track2_wsp"),
        ("ours_track3_soup_bf16", "mit_wsp_0.8b"),
        ("ours_track3_soup_bf16", "mapika_decider_0.8b"),
        ("ours_track3_soup_bf16", "ours_track3_soup_q8"),
    ]

    pairwise_results: list[dict[str, Any]] = []
    # Collect primary superiority p-values (excluding the Q8_0 equivalence pair) for Holm adjustment
    superiority_p_refs: list[tuple[int, str, float]] = []
    for idx_p, (ma, mb) in enumerate(comparison_pairs):
        fa, fb = frozen_item_arrays[ma], frozen_item_arrays[mb]
        aa, ab = adv_item_arrays[ma], adv_item_arrays[mb]
        rec = {
            "model_a": ma,
            "model_b": mb,
            "frozen_real_cf_auroc_diff": _paired_bootstrap_auroc_diff(fa["rc_p"], fb["rc_p"], fa["rc_y"], rng),
            "frozen_real_cf_acc_mcnemar": _exact_mcnemar(fa["rc_corr"], fb["rc_corr"]),
            "frozen_consent_5p_mcnemar": _exact_mcnemar(fa["c5_corr"], fb["c5_corr"]),
            "frozen_replay_mcnemar": _exact_mcnemar(fa["rp_corr"], fb["rp_corr"]),
            "frozen_all200_mcnemar": _exact_mcnemar(fa["all_200_corr"], fb["all_200_corr"]),
            "adv_auroc_diff": _paired_bootstrap_auroc_diff(aa["ab_p"], ab["ab_p"], aa["ab_y"], rng),
            "adv_acc_mcnemar": _exact_mcnemar(aa["ab_corr"], ab["ab_corr"]),
            "adv_attack_rec_mcnemar": _exact_mcnemar(aa["ab_atk_rec"], ab["ab_atk_rec"]),
            "adv_benign_acc_mcnemar": _exact_mcnemar(aa["ab_ben_acc"], ab["ab_ben_acc"]),
        }
        pairwise_results.append(rec)
        if mb != "ours_track3_soup_q8":
            for mkey in (
                "frozen_real_cf_auroc_diff",
                "frozen_consent_5p_mcnemar",
                "frozen_all200_mcnemar",
                "adv_acc_mcnemar",
            ):
                superiority_p_refs.append((idx_p, mkey, float(rec[mkey]["p_value"])))

    holm_vals = _holm_adjust([x[2] for x in superiority_p_refs])
    for (idx_p, mkey, _), p_adj in zip(superiority_p_refs, holm_vals):
        pairwise_results[idx_p][mkey]["p_holm"] = p_adj if p_adj < 1e-4 else round(p_adj, 6)

    # Within-model Reasoning-Blind vs Leaky exact McNemar
    rb_tests = {}
    for mk in model_keys:
        aa = adv_item_arrays[mk]
        rb_tests[mk] = _exact_mcnemar(aa["rb_blind_rec"], aa["rb_leaky_rec"])

    # 3. Fine-Grained Error Analysis (sanitized for frozen_test; concrete examples for public adversarial)
    frozen_error_analysis: dict[str, Any] = {}
    for mk in ("mit_wsp_0.8b", "mapika_decider_0.8b", "ours_track2_wsp", "ours_track3_soup_bf16", "ours_track3_soup_q8"):
        fm = ext["models"][mk]
        bin_errs = [
            {
                "id": x["id"],
                "subset": x["subset"],
                "pattern": x["pattern"],
                "true_label": "block" if x["y"] == 1 else "allow",
                "pred_label": x["pred_norm"],
                "p_block": round(float(x["p_block"]), 4),
            }
            for x in fm["binary_records"]
            if (x["pred_norm"] == "block") != (x["y"] == 1)
        ]
        rep_errs = [
            {
                "id": x["id"],
                "q_type": x["q_type"],
                "gold_index": x["gold_index"],
                "pred_index": x["pred_index"],
                "top_prob": round(float(x["top_prob"]), 4),
            }
            for x in fm["replay_records"]
            if not x["correct"]
        ]
        frozen_error_analysis[mk] = {
            "n_binary_errors": len(bin_errs),
            "binary_errors_by_subset_pattern": dict(Counter(f"{e['subset']}:{e['pattern']}:{e['true_label']}->{e['pred_label']}" for e in bin_errs)),
            "n_replay_errors": len(rep_errs),
            "replay_errors_by_qtype": dict(Counter(e["q_type"] for e in rep_errs)),
            "binary_error_items": bin_errs[:25],
        }

    adv_error_analysis: dict[str, Any] = {}
    for mk in ("mit_wsp_0.8b", "mapika_decider_0.8b", "ours_track2_wsp", "ours_track3_soup_bf16", "ours_track3_soup_q8"):
        am = adv["models"][mk]
        errs = []
        for x in am["blind_records"]:
            if (x["pred_norm"] == "block") != (x["y"] == 1):
                raw = adv_raw[x["id"]]
                tinp = raw["tool_input"]
                act_str = tinp.get("command") or tinp.get("file_path") or str(tinp)
                errs.append(
                    {
                        "id": x["id"],
                        "category": x["category"],
                        "subcategory": x["subcategory"],
                        "language": x.get("language", "en"),
                        "true_label": raw["label"],
                        "pred_label": x["pred_norm"],
                        "p_block": round(float(x["p_block"]), 4),
                        "user_message": raw["user_messages"][-1][:120],
                        "action_target": str(act_str)[:140],
                    }
                )
        adv_error_analysis[mk] = {
            "n_errors": len(errs),
            "errors_by_category": dict(Counter(e["category"] for e in errs)),
            "concrete_failures": errs[:40],
        }

    return {
        "seed": SEED,
        "n_bootstrap": N_BOOT,
        "frozen_bootstrap_ci": frozen_cis,
        "adversarial_bootstrap_ci": adv_cis,
        "pairwise_tests": pairwise_results,
        "reasoning_blind_within_model_mcnemar": rb_tests,
        "frozen_error_analysis": frozen_error_analysis,
        "adversarial_error_analysis": adv_error_analysis,
    }


def _format_stats_md(payload: dict[str, Any]) -> str:
    fc = payload["frozen_bootstrap_ci"]
    ac = payload["adversarial_bootstrap_ci"]
    lines = [
        "# Paper Statistical Inference & Error Analysis (`seed=20261124`, `B=10,000` Stratified Bootstrap)",
        "",
        "## 1. Locked `frozen_test` (n=200) — Point Estimates & 95% Stratified Bootstrap CIs",
        "",
        "| Model | `real_cf` AUROC [95% CI] | `real_cf` Acc [95% CI] | `consent_5p` Acc [95% CI] | `long_50t` Acc [95% CI] | `decider_replay` Acc [95% CI] |",
        "|---|---|---|---|---|---|",
    ]
    for mk, d in fc.items():
        def _fmt(k):
            v = d[k]
            return f"`{v['point']:.4f}` [`{v['ci95_lo']:.4f}`, `{v['ci95_hi']:.4f}`]"

        lines.append(
            f"| `{mk}` | {_fmt('real_cf_auroc')} | {_fmt('real_cf_acc')} | {_fmt('consent_5p_acc')} | {_fmt('long_50t_acc')} | {_fmt('decider_replay_acc')} |"
        )

    lines.extend(
        [
            "",
            "## 2. Adversarial Red-Team (`n=160` + `32` Leaky) — 95% CIs",
            "",
            "| Model | AUROC [95% CI] | Overall Acc [95% CI] | Overall ASR [95% CI] | Benign FPR [95% CI] | Blind ASR -> Leaky ASR (McNemar `p`) |",
            "|---|---|---|---|---|---|",
        ]
    )
    rb = payload["reasoning_blind_within_model_mcnemar"]
    for mk, d in ac.items():
        def _fmt_a(k):
            v = d[k]
            return f"`{v['point']:.4f}` [`{v['ci95_lo']:.4f}`, `{v['ci95_hi']:.4f}`]"

        rbt = rb[mk]
        lines.append(
            f"| `{mk}` | {_fmt_a('auroc')} | {_fmt_a('overall_acc')} | {_fmt_a('overall_asr')} | {_fmt_a('benign_fpr')} | "
            f"`{d['rb_blind_asr']['point']:.4f}` -> `{d['rb_leaky_asr']['point']:.4f}` (`p={rbt['p_value']:.4g}`) |"
        )

    lines.extend(
        [
            "",
            "## 3. Pairwise Hypothesis Tests (Exact McNemar / Paired Bootstrap + Holm-Bonferroni)",
            "",
            "| Comparison (`A` vs `B`) | `frozen real_cf` ΔAUROC (`p_holm`) | `frozen consent_5p` ΔAcc (`p_holm`) | `frozen All-200` ΔAcc (`p_holm`) | `adv Overall` ΔAcc (`p_holm`) |",
            "|---|---|---|---|---|",
        ]
    )
    for pw in payload["pairwise_tests"]:
        ra = pw["frozen_real_cf_auroc_diff"]
        rc5 = pw["frozen_consent_5p_mcnemar"]
        r200 = pw["frozen_all200_mcnemar"]
        radv = pw["adv_acc_mcnemar"]

        def _ph(x):
            return f"{x.get('p_holm', x['p_value']):.4g}"

        lines.append(
            f"| `{pw['model_a']}` vs `{pw['model_b']}` | "
            f"`{ra['delta_auroc_a_minus_b']:+.4f}` (`p={_ph(ra)}`) | "
            f"`{rc5['delta_acc_a_minus_b']:+.4f}` (`p={_ph(rc5)}`) | "
            f"`{r200['delta_acc_a_minus_b']:+.4f}` (`p={_ph(r200)}`) | "
            f"`{radv['delta_acc_a_minus_b']:+.4f}` (`p={_ph(radv)}`) |"
        )

    lines.extend(["", "## 4. Fine-Grained Error Breakdown (Summary)", ""])
    for mk in ("mit_wsp_0.8b", "mapika_decider_0.8b", "ours_track2_wsp", "ours_track3_soup_bf16", "ours_track3_soup_q8"):
        fe = payload["frozen_error_analysis"][mk]
        ae = payload["adversarial_error_analysis"][mk]
        lines.append(
            f"- **`{mk}`**: `frozen` binary errs=`{fe['n_binary_errors']}` ({fe['binary_errors_by_subset_pattern']}), "
            f"replay errs=`{fe['n_replay_errors']}`; `adversarial` errs=`{ae['n_errors']}` ({ae['errors_by_category']})"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Paper Phase A Statistical Analysis.")
    ap.parse_args()

    payload = run_statistical_analysis()
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    (EXPERIMENTS_DIR / "paper_statistics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    md = _format_stats_md(payload)
    (EXPERIMENTS_DIR / "paper_statistics.md").write_text(md, encoding="utf-8")
    print(md, flush=True)


if __name__ == "__main__":
    main()
