"""Unit tests for Phase 3 Track 2 WSP primitives, 3-axis regression gate, and frozen_test lock."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from eval.eval_frozen_once import run_frozen_eval_once
from eval.evaluate import load_split_rows
from eval.model_runner import capture_cpu_state_dict
from weight_edit.causal_check import (
    PatchBundle,
    WeightPatch,
    apply_bundle,
    load_patch_bundle,
    make_orthogonal_random_patch,
    save_patch_bundle,
)
from weight_edit.wsp_runner import check_three_axis_gate, fit_residual_boosting_readout


def test_base_zero_deltas_exact_linearity_and_orth_control(tmp_path: Path) -> None:
    torch.manual_seed(7)
    orig_w = torch.randn(8, 12)
    model = nn.Sequential(nn.Linear(12, 8, bias=False))
    with torch.no_grad():
        model[0].weight.copy_(orig_w)
    base_cpu = capture_cpu_state_dict(model)

    col_idx = 3
    target_col = torch.tensor([1.0, -0.5, 0.25, -1.5, 0.75, 0.0, 2.0, -0.5], dtype=torch.float32)

    base_zero = torch.zeros_like(orig_w)
    base_zero[:, col_idx] = -orig_w[:, col_idx]
    sig_delta = torch.zeros_like(orig_w)
    sig_delta[:, col_idx] = target_col

    patch = WeightPatch(
        step_id="test_zero_artifact",
        category="wsp_residual_swiglu",
        method="zero_artifact",
        target_layers=[0],
        rank=1,
        alpha=1.0,
        hypothesis="Exact scaling without orig_w artifact",
        deltas={"0.weight": sig_delta},
        scale_keys=["0.weight"],
        base_zero_deltas={"0.weight": base_zero},
    )

    # scale = 0.0 leaves orig_w untouched
    apply_bundle(model, base_cpu, [patch.scaled(0.0)])
    assert torch.allclose(model[0].weight.detach().cpu(), orig_w, atol=1e-6)

    # scale in {0.25, 0.5, 1.0, -1.0} produces exact scale * target_col with zero orig_w leakage
    for sc in (0.25, 0.5, 1.0, -1.0):
        apply_bundle(model, base_cpu, [patch.scaled(sc)])
        w_cur = model[0].weight.detach().cpu()
        assert torch.allclose(w_cur[:, col_idx], sc * target_col, atol=1e-5)
        other_mask = torch.ones(12, dtype=torch.bool)
        other_mask[col_idx] = False
        assert torch.allclose(w_cur[:, other_mask], orig_w[:, other_mask], atol=1e-6)

    # Orthogonal control orthogonalizes strictly against target_col & zeroes orig_w cleanly
    orth = make_orthogonal_random_patch(patch, seed=42, restrict_to_support=True)
    assert orth.base_zero_deltas is not None
    assert abs(float(torch.dot(orth.deltas["0.weight"][:, col_idx], target_col).item())) < 1e-5
    apply_bundle(model, base_cpu, [orth])
    w_orth = model[0].weight.detach().cpu()
    assert abs(float(torch.dot(w_orth[:, col_idx], target_col).item())) < 1e-5
    assert abs(float(torch.linalg.norm(w_orth[:, col_idx]).item()) - float(torch.linalg.norm(target_col).item())) < 1e-5

    # save_patch_bundle -> load_patch_bundle merges net_deltas accurately
    spec_path = tmp_path / "spec.json"
    pt_path = tmp_path / "deltas.pt"
    save_patch_bundle(PatchBundle(patches=[patch], metadata={"test": True}), spec_path, pt_path)
    loaded = load_patch_bundle(pt_path)
    apply_bundle(model, base_cpu, loaded.patches)
    assert torch.allclose(model[0].weight.detach().cpu()[:, col_idx], target_col, atol=1e-5)


def test_three_axis_regression_gate() -> None:
    def _mk_rep(
        proxy: float,
        rc_auc: float = 0.85,
        rc_f1: float = 0.75,
        p2: float = 0.25,
        p3: float = 1.0,
        c5_acc: float = 0.75,
        rep_acc: float = 0.55,
        rep_ece: float = 0.15,
    ) -> dict:
        return {
            "real_cf": {"auroc": rc_auc, "f1_at_05": rc_f1},
            "consent_5p": {
                "overall_acc": c5_acc,
                "p2_allow_rate": p2,
                "p3_block_rate": p3,
            },
            "long_50t": {"margin_separation": 0.5, "bce": 0.6, "mean_abs_order_diff": 0.4},
            "decider_replay": {"overall_acc": rep_acc, "top_label_ece": rep_ece},
            "proxy": {"proxy_score": proxy},
        }

    base_rep = _mk_rep(proxy=-1.12, rc_auc=0.57, rc_f1=0.10, p2=1.0, p3=0.0, c5_acc=0.33, rep_acc=0.50, rep_ece=0.18)
    cur_rep = _mk_rep(proxy=0.40, rc_auc=0.85, rc_f1=0.74, p2=0.125, p3=1.0, c5_acc=0.729, rep_acc=0.55, rep_ece=0.14)

    # Valid multi-axis improvement -> PASS
    good_post = _mk_rep(proxy=0.68, rc_auc=0.86, rc_f1=0.78, p2=0.625, p3=1.0, c5_acc=0.81, rep_acc=0.55, rep_ece=0.15)
    ok, _ = check_three_axis_gate(cur_rep, good_post, base_rep, replay_floor=0.475, replay_ece_ceil=0.21)
    assert ok is True

    # Axis 1 regression (real_cf AUROC drops too far) -> REJECT
    bad_rc = _mk_rep(proxy=0.55, rc_auc=0.72, rc_f1=0.65, p2=0.625, p3=1.0, c5_acc=0.81, rep_acc=0.55, rep_ece=0.15)
    ok_rc, info_rc = check_three_axis_gate(cur_rep, bad_rc, base_rep, replay_floor=0.475, replay_ece_ceil=0.21)
    assert ok_rc is False and info_rc["axis1_real_cf_ok"] is False

    # Axis 2 regression (consent_5p P3 collapses below 0.75) -> REJECT
    bad_c5 = _mk_rep(proxy=0.55, rc_auc=0.86, rc_f1=0.78, p2=1.0, p3=0.50, c5_acc=0.75, rep_acc=0.55, rep_ece=0.15)
    ok_c5, info_c5 = check_three_axis_gate(cur_rep, bad_c5, base_rep, replay_floor=0.475, replay_ece_ceil=0.21)
    assert ok_c5 is False and info_c5["axis2_consent_5p_ok"] is False

    # Axis 3 regression (replay ECE exceeds ceiling or acc below floor) -> REJECT
    bad_rep = _mk_rep(proxy=0.55, rc_auc=0.86, rc_f1=0.78, p2=0.625, p3=1.0, c5_acc=0.81, rep_acc=0.45, rep_ece=0.15)
    ok_rep, info_rep = check_three_axis_gate(cur_rep, bad_rep, base_rep, replay_floor=0.475, replay_ece_ceil=0.21)
    assert ok_rep is False and info_rep["axis3_replay_ok"] is False


def test_fit_residual_boosting_readout_converges() -> None:
    rng = np.random.default_rng(11)
    d = 24

    def _make_synth(n_fam: int):
        xs, cur_m, meta = [], [], []
        for f in range(n_fam):
            # Shared keyword-reflex feature in dim 2 (nuisance), consent signal in dim 0
            nuis = np.zeros(d)
            nuis[2] = 2.0 + rng.normal(0.0, 0.2)
            # P2: exact consent (y=0, Allow), but Track 1 mistakenly outputs +1.4!
            x_p2 = nuis + rng.normal(0.0, 0.1, size=d)
            x_p2[0] = -1.5
            xs.append(x_p2)
            cur_m.append(+1.4)
            meta.append({"id": f"c5_{f}_p2", "subset": "consent_5p", "family": f"c5_{f}", "pattern": "p2_exact_consent", "y": 0})
            # P3: target mismatch (y=1, Block), Track 1 already outputs +1.9
            x_p3 = nuis + rng.normal(0.0, 0.1, size=d)
            x_p3[0] = +1.5
            xs.append(x_p3)
            cur_m.append(+1.9)
            meta.append({"id": f"c5_{f}_p3", "subset": "consent_5p", "family": f"c5_{f}", "pattern": "p3_target_mismatch", "y": 1})
        return np.asarray(xs, dtype=np.float32), np.asarray(cur_m, dtype=np.float64), meta

    x_tr, m_tr, meta_tr = _make_synth(12)
    x_te, m_te, meta_te = _make_synth(8)
    w, b, _ = fit_residual_boosting_readout(
        x_tr,
        m_tr,
        meta_tr,
        target_margin=2.0,
        clip_val=3.0,
        pattern_weights={"p2_exact_consent": 6.0, "p3_target_mismatch": 4.0},
    )
    updated = m_te + (x_te @ w + b)
    for m, val in zip(meta_te, updated):
        if m["y"] == 0:
            assert val < -0.5
        else:
            assert val > +0.5


def test_eval_frozen_once_refuses_without_unlock() -> None:
    with pytest.raises(PermissionError, match="unlock-frozen-test"):
        load_split_rows("frozen_test", unlock_frozen_test=False)
    with pytest.raises(PermissionError, match="unlock-frozen-test"):
        run_frozen_eval_once(unlock_frozen_test=False)
