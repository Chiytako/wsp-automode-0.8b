"""Reversible weight-patch primitives, 3-point causal sanity checks, and LOO ablation.

Every track-1 and track-2 edit is represented as an explicit, reversible delta
bundle in host CPU float32 memory. No gradients or optimizer steps are ever used.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from scipy.stats import spearmanr

from automode.questions import Q_BLOCK_BINARY, build_decider_prompt
from automode.view import render_view
from eval.evaluate import compute_binary_metrics, compute_ece, compute_proxy_score
from eval.model_runner import (
    QwenDeciderRunner,
    compute_weight_diff_stats,
    restore_cpu_state_dict,
)


@dataclass
class WeightPatch:
    """A single reversible gradient-free weight edit step."""

    step_id: str
    category: str
    method: str
    target_layers: list[int]
    rank: int
    alpha: float
    hypothesis: str
    deltas: dict[str, torch.Tensor]  # param_name -> CPU float32 delta tensor
    metadata: dict[str, Any] = field(default_factory=dict)
    scale_keys: list[str] | None = None  # if set, scale() & orth-control vary only these keys (at scale=0, all zero)
    base_zero_deltas: dict[str, torch.Tensor] | None = None  # unscaled baseline-zeroing offsets applied whenever scale != 0

    def net_deltas(self) -> dict[str, torch.Tensor]:
        keys = sorted(set(self.deltas.keys()) | (set(self.base_zero_deltas.keys()) if self.base_zero_deltas else set()))
        out: dict[str, torch.Tensor] = {}
        for k in keys:
            val: torch.Tensor | None = None
            if self.base_zero_deltas and k in self.base_zero_deltas:
                val = self.base_zero_deltas[k].detach().float().cpu().clone()
            if k in self.deltas:
                d = self.deltas[k].detach().float().cpu()
                val = d.clone() if val is None else (val + d)
            if val is not None:
                out[k] = val
        return out

    def frobenius_norm(self) -> float:
        sq = sum(float(torch.sum(d.float() ** 2).item()) for d in self.net_deltas().values())
        return math.sqrt(sq)

    def modified_elements(self, atol: float = 1e-8) -> int:
        return sum(int((d.abs() > atol).sum().item()) for d in self.net_deltas().values())

    def scaled(self, factor: float) -> "WeightPatch":
        if factor == 0.0:
            new_deltas = {k: torch.zeros_like(v.float().cpu()) for k, v in self.deltas.items()}
            new_base_zero = None
        else:
            if not self.scale_keys:
                new_deltas = {k: v.float().cpu() * float(factor) for k, v in self.deltas.items()}
            else:
                s_set = set(self.scale_keys)
                new_deltas = {
                    k: (v.float().cpu() * float(factor) if k in s_set else v.float().cpu().clone())
                    for k, v in self.deltas.items()
                }
            new_base_zero = (
                {k: v.float().cpu().clone() for k, v in self.base_zero_deltas.items()}
                if self.base_zero_deltas is not None
                else None
            )
        return WeightPatch(
            step_id=self.step_id,
            category=self.category,
            method=self.method,
            target_layers=list(self.target_layers),
            rank=self.rank,
            alpha=self.alpha * factor,
            hypothesis=self.hypothesis,
            deltas=new_deltas,
            metadata=dict(self.metadata),
            scale_keys=list(self.scale_keys) if self.scale_keys is not None else None,
            base_zero_deltas=new_base_zero,
        )


@dataclass
class PatchBundle:
    """Ordered sequence of WeightPatch steps."""

    patches: list[WeightPatch] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalCheckResult:
    """Outcome of the 3-point causal sanity test (Dose-Response, Sign-Reversal, Orthogonal Control)."""

    step_id: str
    metric_key: str
    passed: bool
    base_score: float
    best_scale: float
    best_score: float
    dose_response: dict[str, Any]
    sign_reversal: dict[str, Any]
    orthogonal_control: dict[str, Any]
    fail_reasons: list[str] = field(default_factory=list)
    best_report: dict[str, Any] | None = None


def apply_patch(model: torch.nn.Module, patch: WeightPatch, scale: float = 1.0) -> None:
    """Add `base_zero_deltas + scale * patch.deltas` to `model` parameters under torch.no_grad()."""
    if scale == 0.0:
        return
    param_map = dict(model.named_parameters())
    with torch.no_grad():
        if patch.base_zero_deltas:
            for name, bdelta in patch.base_zero_deltas.items():
                if name not in param_map:
                    raise KeyError(f"Parameter {name!r} not found on model")
                p = param_map[name]
                p.add_(bdelta.to(device=p.device, dtype=p.dtype), alpha=1.0)
        for name, delta in patch.deltas.items():
            if name not in param_map:
                raise KeyError(f"Parameter {name!r} not found on model")
            p = param_map[name]
            p.add_(delta.to(device=p.device, dtype=p.dtype), alpha=float(scale))


def apply_bundle(
    model: torch.nn.Module,
    base_cpu_state: dict[str, torch.Tensor],
    patches: list[WeightPatch],
) -> None:
    """Restore base weights from CPU snapshot and apply `patches` in exact order."""
    restore_cpu_state_dict(model, base_cpu_state)
    for p in patches:
        apply_patch(model, p, scale=1.0)


def patch_diff_summary(
    model: torch.nn.Module,
    base_cpu_state: dict[str, torch.Tensor],
) -> dict[str, Any]:
    return compute_weight_diff_stats(model, base_cpu_state)


def save_patch_bundle(
    bundle: PatchBundle,
    json_spec_path: Path,
    pt_deltas_path: Path,
    extra_summary: dict[str, Any] | None = None,
) -> None:
    """Persist human-readable spec JSON (committed) and net tensor deltas .pt (git-ignored)."""
    json_spec_path.parent.mkdir(parents=True, exist_ok=True)
    pt_deltas_path.parent.mkdir(parents=True, exist_ok=True)

    steps_spec = []
    patches_payload = []
    for p in bundle.patches:
        net = p.net_deltas()
        steps_spec.append(
            {
                "step_id": p.step_id,
                "category": p.category,
                "method": p.method,
                "target_layers": p.target_layers,
                "rank": p.rank,
                "alpha": round(float(p.alpha), 6),
                "hypothesis": p.hypothesis,
                "frobenius_norm": round(p.frobenius_norm(), 6),
                "modified_elements": p.modified_elements(),
                "tensors": {k: list(v.shape) for k, v in net.items()},
                "metadata": p.metadata,
            }
        )
        patches_payload.append(
            {
                "step_id": p.step_id,
                "category": p.category,
                "method": p.method,
                "target_layers": p.target_layers,
                "rank": p.rank,
                "alpha": p.alpha,
                "hypothesis": p.hypothesis,
                "metadata": p.metadata,
                "deltas": {k: v.detach().float().cpu() for k, v in net.items()},
            }
        )

    spec = {
        "metadata": bundle.metadata,
        "extra_summary": extra_summary or {},
        "steps": steps_spec,
    }
    json_spec_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    torch.save(
        {"metadata": bundle.metadata, "patches": patches_payload},
        pt_deltas_path,
    )


def load_patch_bundle(pt_deltas_path: Path) -> PatchBundle:
    raw = torch.load(pt_deltas_path, map_location="cpu", weights_only=True)
    patches = [
        WeightPatch(
            step_id=item["step_id"],
            category=item["category"],
            method=item["method"],
            target_layers=list(item["target_layers"]),
            rank=int(item["rank"]),
            alpha=float(item["alpha"]),
            hypothesis=item["hypothesis"],
            deltas={k: v.float().cpu() for k, v in item["deltas"].items()},
            metadata=dict(item.get("metadata", {})),
        )
        for item in raw["patches"]
    ]
    return PatchBundle(patches=patches, metadata=dict(raw.get("metadata", {})))


def make_causal_panel_rows(all_dev_rows: list[dict[str, Any]], fast: bool = True) -> list[dict[str, Any]]:
    """Select either full select_dev or a balanced deterministic panel for fast causal checks."""
    if not fast:
        return list(all_dev_rows)
    rc = [r for r in all_dev_rows if r["subset"] == "real_cf"][:20]       # 10 pairs (10 allow, 10 block)
    c5 = [r for r in all_dev_rows if r["subset"] == "consent_5p"][:24]    # 4 full families x 6 patterns = 24 rows
    rep_noul = [r for r in all_dev_rows if r["subset"] == "decider_replay" and r["q_type"] == "noul"][:6]
    rep_choice = [r for r in all_dev_rows if r["subset"] == "decider_replay" and r["q_type"] == "choice"][:6]
    rep_score = [r for r in all_dev_rows if r["subset"] == "decider_replay" and r["q_type"] == "score"][:6]
    return rc + c5 + rep_noul + rep_choice + rep_score


def quick_eval_dev(
    runner: QwenDeciderRunner,
    rows: list[dict[str, Any]],
    include_long: bool = True,
) -> dict[str, Any]:
    """Evaluate current model weights in plain mode across any subset combination safely."""
    by_subset: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_subset.setdefault(r["subset"], []).append(r)

    rc_rows = by_subset.get("real_cf", [])
    rc_recs = []
    for r in rc_rows:
        ctx = render_view(r, tok=runner.tokenizer, budget=2048, include_recap=False).text
        sc = runner.score_binary_paired(ctx, mode="plain", question_spec=Q_BLOCK_BINARY)
        rc_recs.append({"y": 1 if r["label"] == "block" else 0, **sc})

    rc_metrics = compute_binary_metrics(
        [x["p_block"] for x in rc_recs],
        [x["y"] for x in rc_recs],
        [x["margin_normal"] for x in rc_recs],
    )
    rc_metrics["order_flip_rate"] = round(
        sum(1 for x in rc_recs if x["flipped"]) / max(1, len(rc_recs)), 4
    )
    rc_metrics["mean_abs_order_diff"] = round(
        float(np.mean([x["order_abs_margin_diff"] for x in rc_recs])) if rc_recs else 0.0, 4
    )
    rc_metrics["second_reading"] = compute_binary_metrics(
        [x["p_block_second_reading"] for x in rc_recs],
        [x["y"] for x in rc_recs],
        [x["margin_avg"] for x in rc_recs],
    )

    c5_rows = by_subset.get("consent_5p", [])
    c5_recs = []
    for r in c5_rows:
        ctx = render_view(r, tok=runner.tokenizer, budget=2048, include_recap=False).text
        sc = runner.score_binary_paired(ctx, mode="plain", question_spec=Q_BLOCK_BINARY)
        c5_recs.append({"y": 1 if r["label"] == "block" else 0, "pattern": r["pattern"], **sc})

    c5_bin = compute_binary_metrics(
        [x["p_block"] for x in c5_recs],
        [x["y"] for x in c5_recs],
        [x["margin_normal"] for x in c5_recs],
    )
    p2_sub = [x for x in c5_recs if x["pattern"] == "p2_exact_consent"]
    p3_sub = [x for x in c5_recs if x["pattern"] == "p3_target_mismatch"]
    p5a_sub = [x for x in c5_recs if x["pattern"] == "p5a_blocked_approved_same"]
    p5b_sub = [x for x in c5_recs if x["pattern"] == "p5b_blocked_swapped_target"]

    per_pat_acc = {}
    for pat in (
        "p1_no_consent",
        "p2_exact_consent",
        "p3_target_mismatch",
        "p4_vague_consent",
        "p5a_blocked_approved_same",
        "p5b_blocked_swapped_target",
    ):
        sub = [x for x in c5_recs if x["pattern"] == pat]
        if sub:
            per_pat_acc[pat] = round(
                sum(1 for x in sub if (x["pred_normal"] == "block") == (x["y"] == 1)) / len(sub), 4
            )

    margin_sep_p3_p2 = (
        float(np.mean([x["margin_normal"] for x in p3_sub])) - float(np.mean([x["margin_normal"] for x in p2_sub]))
        if (p2_sub and p3_sub)
        else 0.0
    )
    margin_sep_p5b_p5a = (
        float(np.mean([x["margin_normal"] for x in p5b_sub])) - float(np.mean([x["margin_normal"] for x in p5a_sub]))
        if (p5a_sub and p5b_sub)
        else 0.0
    )
    c5_summary = {
        "n": len(c5_recs),
        "overall_acc": c5_bin["acc_at_05"],
        "per_pattern_acc": per_pat_acc,
        "p2_allow_rate": per_pat_acc.get("p2_exact_consent", 0.0),
        "p3_block_rate": per_pat_acc.get("p3_target_mismatch", 0.0),
        "p5a_allow_rate": per_pat_acc.get("p5a_blocked_approved_same", 0.0),
        "p5b_block_rate": per_pat_acc.get("p5b_blocked_swapped_target", 0.0),
        "margin_sep_p3_minus_p2": round(margin_sep_p3_p2, 4),
        "margin_sep_p5b_minus_p5a": round(margin_sep_p5b_p5a, 4),
        "order_flip_rate": round(sum(1 for x in c5_recs if x["flipped"]) / max(1, len(c5_recs)), 4),
        "mean_abs_order_diff": round(
            float(np.mean([x["order_abs_margin_diff"] for x in c5_recs])) if c5_recs else 0.0, 4
        ),
        "binary_metrics": c5_bin,
    }

    long_summary: dict[str, Any] = {"n": 0}
    l_rows = by_subset.get("long_50t", []) if include_long else []
    if l_rows:
        l_recs = []
        for r in l_rows:
            ctx = render_view(r, tok=runner.tokenizer, budget=3200, include_recap=False).text
            sc = runner.score_binary_paired(ctx, mode="plain", question_spec=Q_BLOCK_BINARY)
            l_recs.append({"y": 1 if r["label"] == "block" else 0, **sc})
        l_bin = compute_binary_metrics(
            [x["p_block"] for x in l_recs],
            [x["y"] for x in l_recs],
            [x["margin_normal"] for x in l_recs],
        )
        allow_sub = [x for x in l_recs if x["y"] == 0]
        blk_sub = [x for x in l_recs if x["y"] == 1]
        long_summary = {
            "n": len(l_recs),
            "overall_acc": l_bin["acc_at_05"],
            "early_consent_allow_rate": round(
                sum(1 for x in allow_sub if x["pred_normal"] == "allow") / max(1, len(allow_sub)), 4
            ),
            "mismatch_block_rate": round(
                sum(1 for x in blk_sub if x["pred_normal"] == "block") / max(1, len(blk_sub)), 4
            ),
            "margin_separation": l_bin["margin_separation"],
            "bce": l_bin["bce"],
            "order_flip_rate": round(sum(1 for x in l_recs if x["flipped"]) / max(1, len(l_recs)), 4),
            "mean_abs_order_diff": round(
                float(np.mean([x["order_abs_margin_diff"] for x in l_recs])), 4
            ),
        }

    rep_rows = by_subset.get("decider_replay", [])
    rep_recs = []
    for r in rep_rows:
        pb = build_decider_prompt(r["context"], spec=r["question_spec"], reverse_options=False)
        res = runner.score_prompt(pb.text, pb.option_letters, yes_idx=pb.yes_idx, no_idx=pb.no_idx)
        gold_idx = int(r["gold_index"])
        brier_multi = sum(
            (p - (1.0 if j == gold_idx else 0.0)) ** 2 for j, p in enumerate(res.option_probs)
        ) / len(res.option_probs)
        rep_recs.append(
            {
                "correct": int(res.pred_index == gold_idx),
                "top_prob": float(res.option_probs[res.pred_index]),
                "brier": brier_multi,
            }
        )
    rep_summary = {
        "n": len(rep_recs),
        "overall_acc": round(sum(x["correct"] for x in rep_recs) / max(1, len(rep_recs)), 4),
        "top_label_ece": round(
            compute_ece([x["top_prob"] for x in rep_recs], [x["correct"] for x in rep_recs], n_bins=10), 4
        ),
        "brier": round(float(np.mean([x["brier"] for x in rep_recs])) if rep_recs else 0.0, 4),
    }

    report = {
        "real_cf": rc_metrics,
        "consent_5p": c5_summary,
        "long_50t": long_summary,
        "decider_replay": rep_summary,
    }
    report["proxy"] = compute_proxy_score(report)
    return report


def extract_metric(report: dict[str, Any], metric_key: str) -> float:
    """Extract scalar objective from an evaluation report dict."""
    if metric_key == "proxy_score":
        return float(report["proxy"]["proxy_score"])
    if metric_key == "order_consistency":
        return -float(report["real_cf"]["mean_abs_order_diff"] + report["consent_5p"]["mean_abs_order_diff"])
    if metric_key == "consent_focus":
        c5 = report["consent_5p"]
        return float(c5["margin_sep_p3_minus_p2"] + c5["margin_sep_p5b_minus_p5a"] - 0.5 * c5["binary_metrics"]["bce"])
    if metric_key == "domain_focus":
        rc = report["real_cf"]
        return float(rc["margin_separation"] - 2.0 * rc["bce"])
    if metric_key == "long50t_focus":
        lg = report["long_50t"]
        return float(lg["margin_separation"] - 1.5 * lg["bce"] - 0.15 * lg["mean_abs_order_diff"])
    if metric_key == "proxy_plus_long":
        lg = report["long_50t"]
        return float(report["proxy"]["proxy_score"] + 0.5 * math.tanh(lg["margin_separation"] / 4.0) - 0.25 * lg["bce"])
    cur: Any = report
    for part in metric_key.split("."):
        cur = cur[part]
    return float(cur)


def make_orthogonal_random_patch(
    patch: WeightPatch,
    seed: int,
    restrict_to_support: bool = True,
) -> WeightPatch:
    """Generate a random delta on the exact same tensors/support with matched Frobenius norm and 0 inner product."""
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    keys = sorted(patch.scale_keys) if patch.scale_keys else sorted(patch.deltas.keys())
    orig_flats = []
    rand_flats = []
    masks = []
    for k in keys:
        d = patch.deltas[k].float().cpu()
        flat = d.reshape(-1)
        mask = (flat.abs() > 1e-8) if restrict_to_support else torch.ones_like(flat, dtype=torch.bool)
        if int(mask.sum().item()) < 2:
            mask = torch.ones_like(flat, dtype=torch.bool)
        r = torch.zeros_like(flat)
        r[mask] = torch.randn(int(mask.sum().item()), generator=rng, dtype=torch.float32)
        orig_flats.append(flat)
        rand_flats.append(r)
        masks.append(mask)

    v = torch.cat(orig_flats)
    r_vec = torch.cat(rand_flats)
    v_norm_sq = float(torch.dot(v, v).item())
    if v_norm_sq <= 1e-18:
        raise ValueError(f"Patch {patch.step_id} has zero Frobenius norm")

    # Project onto orthogonal complement of v within active support
    support = torch.cat(masks)
    v_s = v[support]
    r_s = r_vec[support]
    proj = torch.dot(r_s, v_s) / torch.dot(v_s, v_s)
    r_s = r_s - proj * v_s
    r_norm = float(torch.linalg.norm(r_s).item())
    if r_norm <= 1e-12:
        raise RuntimeError("Degenerate orthogonal random control vector")
    r_s = r_s * (math.sqrt(v_norm_sq) / r_norm)

    r_full = torch.zeros_like(v)
    r_full[support] = r_s

    rand_deltas: dict[str, torch.Tensor] = {}
    if patch.scale_keys:
        s_set = set(patch.scale_keys)
        for k, orig in patch.deltas.items():
            if k not in s_set:
                rand_deltas[k] = orig.float().cpu().clone()
    offset = 0
    for k in keys:
        n = patch.deltas[k].numel()
        rand_deltas[k] = r_full[offset : offset + n].reshape(patch.deltas[k].shape).clone()
        offset += n

    base_zero_copy = (
        {k: v.float().cpu().clone() for k, v in patch.base_zero_deltas.items()}
        if patch.base_zero_deltas is not None
        else None
    )

    return WeightPatch(
        step_id=f"{patch.step_id}_orth_rand_{seed}",
        category="orthogonal_random_control",
        method="gram_schmidt_support",
        target_layers=list(patch.target_layers),
        rank=patch.rank,
        alpha=patch.alpha,
        hypothesis="Null control",
        deltas=rand_deltas,
        scale_keys=list(patch.scale_keys) if patch.scale_keys is not None else None,
        base_zero_deltas=base_zero_copy,
    )


def run_three_point_causal_check(
    model: torch.nn.Module,
    base_cpu_state: dict[str, torch.Tensor],
    prior_patches: list[WeightPatch],
    candidate_patch: WeightPatch,
    eval_fn: Callable[[], dict[str, Any]],
    metric_key: str = "proxy_score",
    n_random_seeds: int = 3,
    min_gain: float = 1e-3,
    min_spearman: float = 0.75,
    base_report_cached: dict[str, Any] | None = None,
) -> CausalCheckResult:
    """Execute mandatory 3-point causal sanity test:
    1. Dose-Response across scales [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
    2. Sign-Reversal at scales [-0.5, -1.0]
    3. Orthogonal Random Control matched in Frobenius norm and support (3+ seeds)
    """
    fail_reasons: list[str] = []

    if base_report_cached is None:
        apply_bundle(model, base_cpu_state, prior_patches)
        base_report = eval_fn()
    else:
        base_report = base_report_cached
    base_score = extract_metric(base_report, metric_key)

    # 1. Dose-Response
    pos_scales = [0.25, 0.5, 0.75, 1.0, 1.25]
    dose_scores: dict[float, float] = {0.0: base_score}
    dose_reports: dict[float, dict[str, Any]] = {}
    for s in pos_scales:
        apply_bundle(model, base_cpu_state, prior_patches + [candidate_patch.scaled(s)])
        rep = eval_fn()
        dose_scores[s] = extract_metric(rep, metric_key)
        dose_reports[s] = rep

    mono_scales = [0.0, 0.25, 0.5, 0.75, 1.0]
    mono_vals = [dose_scores[s] for s in mono_scales]
    if len(set(round(v, 9) for v in mono_vals)) <= 1:
        rho = 0.0
    else:
        rho_raw = spearmanr(mono_scales, mono_vals).statistic
        rho = float(0.0 if np.isnan(rho_raw) else rho_raw)
    best_scale = max(pos_scales, key=lambda s: dose_scores[s])
    best_score = dose_scores[best_scale]
    cand_gain = best_score - base_score
    dose_passed = (cand_gain > min_gain) and (rho >= min_spearman)
    if not dose_passed:
        fail_reasons.append(f"Dose-Response failed (gain={cand_gain:.5f}, spearman={rho:.3f})")

    # 2. Sign-Reversal
    neg_scales = [-0.5, -1.0]
    neg_scores: dict[float, float] = {}
    for s in neg_scales:
        apply_bundle(model, base_cpu_state, prior_patches + [candidate_patch.scaled(s)])
        rep = eval_fn()
        neg_scores[s] = extract_metric(rep, metric_key)

    rev_drop = base_score - neg_scores[-1.0]
    rev_passed = (neg_scores[-1.0] < base_score - min_gain) and (min(neg_scores.values()) < base_score - min_gain)
    if not rev_passed:
        fail_reasons.append(f"Sign-Reversal failed (score(-1.0)={neg_scores[-1.0]:.5f} vs base={base_score:.5f})")

    # 3. Orthogonal Random Control (evaluated at same norm as best_scale)
    scaled_cand = candidate_patch.scaled(best_scale)
    rand_scores: list[float] = []
    rand_gains: list[float] = []
    for seed_idx in range(n_random_seeds):
        r_patch = make_orthogonal_random_patch(scaled_cand, seed=1000 + seed_idx)
        apply_bundle(model, base_cpu_state, prior_patches + [r_patch])
        rep = eval_fn()
        sc = extract_metric(rep, metric_key)
        rand_scores.append(sc)
        rand_gains.append(sc - base_score)

    r_mean = float(np.mean(rand_gains))
    r_std = float(np.std(rand_gains, ddof=0))
    r_max = float(np.max(rand_gains))
    orth_passed = (cand_gain > r_max + 1e-4) and (cand_gain > r_mean + 2.0 * r_std)
    if not orth_passed:
        fail_reasons.append(
            f"Orthogonal Control failed (cand_gain={cand_gain:.5f}, rand_max={r_max:.5f}, rand_mean={r_mean:.5f}, rand_std={r_std:.5f})"
        )

    # Restore clean prior state
    apply_bundle(model, base_cpu_state, prior_patches)

    return CausalCheckResult(
        step_id=candidate_patch.step_id,
        metric_key=metric_key,
        passed=dose_passed and rev_passed and orth_passed,
        base_score=round(base_score, 5),
        best_scale=best_scale,
        best_score=round(best_score, 5),
        dose_response={
            "scores_by_scale": {str(k): round(v, 5) for k, v in sorted(dose_scores.items())},
            "spearman_0_to_1": round(rho, 4),
            "gain_at_best_scale": round(cand_gain, 5),
            "passed": dose_passed,
        },
        sign_reversal={
            "scores_by_scale": {str(k): round(v, 5) for k, v in sorted(neg_scores.items())},
            "drop_at_minus_1": round(rev_drop, 5),
            "passed": rev_passed,
        },
        orthogonal_control={
            "n_seeds": n_random_seeds,
            "rand_gains": [round(g, 5) for g in rand_gains],
            "rand_mean_gain": round(r_mean, 5),
            "rand_std_gain": round(r_std, 5),
            "rand_max_gain": round(r_max, 5),
            "cand_gain": round(cand_gain, 5),
            "passed": orth_passed,
        },
        fail_reasons=fail_reasons,
        best_report=dose_reports[best_scale],
    )


def leave_one_out_ablation(
    model: torch.nn.Module,
    base_cpu_state: dict[str, torch.Tensor],
    patches: list[WeightPatch],
    eval_fn: Callable[[], dict[str, Any]],
    replay_acc_floor: float,
    tol: float = 0.005,
) -> tuple[list[WeightPatch], dict[str, Any]]:
    """Iteratively prune steps whose removal does not hurt proxy_score or violate replay floor."""
    kept = list(patches)
    history: list[dict[str, Any]] = []

    while len(kept) > 1:
        apply_bundle(model, base_cpu_state, kept)
        full_rep = eval_fn()
        full_proxy = float(full_rep["proxy"]["proxy_score"])

        best_drop_idx: int | None = None
        best_drop_proxy = -1e9
        trials = []
        for idx, step in enumerate(kept):
            sub = [p for j, p in enumerate(kept) if j != idx]
            apply_bundle(model, base_cpu_state, sub)
            rep = eval_fn()
            p_sc = float(rep["proxy"]["proxy_score"])
            rep_acc = float(rep["decider_replay"]["overall_acc"])
            trials.append(
                {
                    "dropped_step": step.step_id,
                    "proxy_without": round(p_sc, 5),
                    "delta_proxy": round(p_sc - full_proxy, 5),
                    "replay_acc_without": rep_acc,
                }
            )
            if rep_acc >= replay_acc_floor and p_sc >= full_proxy - tol and p_sc > best_drop_proxy:
                best_drop_proxy = p_sc
                best_drop_idx = idx

        history.append({"n_steps": len(kept), "full_proxy": round(full_proxy, 5), "trials": trials})
        if best_drop_idx is not None:
            kept.pop(best_drop_idx)
        else:
            break

    apply_bundle(model, base_cpu_state, kept)
    final_rep = eval_fn()
    return kept, {"final_step_ids": [p.step_id for p in kept], "final_proxy": final_rep["proxy"]["proxy_score"], "rounds": history}
