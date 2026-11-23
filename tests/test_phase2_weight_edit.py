"""Unit tests for Phase 2 gradient-free weight editing primitives (CPU-only)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from eval.model_runner import capture_cpu_state_dict, compute_weight_diff_stats
from weight_edit.causal_check import (
    PatchBundle,
    WeightPatch,
    apply_bundle,
    leave_one_out_ablation,
    make_orthogonal_random_patch,
    run_three_point_causal_check,
)
from weight_edit.context_edit import (
    ObserveActivations,
    build_null_space_projector,
    build_swiglu_order_gate_patch,
    fit_joint_contrastive_readout,
    solve_alphaedit_update,
)


def test_alphaedit_null_space_invariance() -> None:
    torch.manual_seed(42)
    # 60 preservation vectors living mostly in a 10-D subspace of R^32
    basis = torch.randn(10, 32)
    coeffs = torch.randn(60, 10)
    k_preserve = coeffs @ basis  # [60, 32]
    p_null, stats = build_null_space_projector(k_preserve, variance_keep=0.99, max_rank_frac=0.5)
    assert stats["residual_norm_ratio"] < 0.05

    k_target = torch.randn(25, 32)
    r_target = torch.randn(25, 16)
    dw = solve_alphaedit_update(r_target, k_target, p_null, rank=2, lam=10.0, alpha=1.0)
    # Preserved keys must be mapped to ~0 by dw
    leaked = torch.linalg.norm(dw @ k_preserve.T).item()
    assert leaked < 1e-4


def test_apply_rollback_and_diff_stats() -> None:
    model = nn.Sequential(nn.Linear(16, 16, bias=False), nn.Linear(16, 8, bias=False))
    base_cpu = capture_cpu_state_dict(model)
    p1 = WeightPatch(
        step_id="s1",
        category="test",
        method="rank1",
        target_layers=[0],
        rank=1,
        alpha=1.0,
        hypothesis="test",
        deltas={"0.weight": torch.ones(16, 16) * 0.25},
    )
    apply_bundle(model, base_cpu, [p1])
    stats = compute_weight_diff_stats(model, base_cpu)
    assert stats["modified_params"] == 256
    assert stats["modified_tensor_count"] == 1
    assert abs(stats["total_delta_frobenius_norm"] - 4.0) < 1e-4

    # Rollback to empty bundle must be bit-identical
    apply_bundle(model, base_cpu, [])
    for k, p in model.named_parameters():
        assert torch.equal(p.detach().cpu(), base_cpu[k])


def test_three_point_causal_check_and_loo() -> None:
    model = nn.Sequential(nn.Linear(16, 4, bias=False))
    with torch.no_grad():
        model[0].weight.zero_()
    base_cpu = capture_cpu_state_dict(model)

    x_eval = torch.zeros(1, 16)
    x_eval[0, 0] = 1.0
    target_dir = torch.tensor([[1.0, -1.0, 0.5, -0.5]]).T  # [4, 1]

    def _mock_eval() -> dict:
        # Score depends strictly on alignment with target_dir along column 0
        out = model[0].weight @ x_eval.T  # [4, 1]
        score = float((out[:, 0] * target_dir[:, 0]).sum().item())
        return {
            "real_cf": {"mean_abs_order_diff": 0.0, "margin_separation": score, "bce": 0.5},
            "consent_5p": {"mean_abs_order_diff": 0.0, "margin_sep_p3_minus_p2": 0.0, "margin_sep_p5b_minus_p5a": 0.0, "binary_metrics": {"bce": 0.5}},
            "decider_replay": {"overall_acc": 0.8},
            "proxy": {"proxy_score": score},
        }

    # True causal patch aligned with target_dir on column 0..15 (only col 0 matters)
    true_delta = torch.zeros(4, 16)
    true_delta[:, 0] = target_dir[:, 0]
    p_true = WeightPatch(
        step_id="true_patch",
        category="test",
        method="synthetic",
        target_layers=[0],
        rank=1,
        alpha=1.0,
        hypothesis="aligned",
        deltas={"0.weight": true_delta},
    )
    res_true = run_three_point_causal_check(
        model=model,
        base_cpu_state=base_cpu,
        prior_patches=[],
        candidate_patch=p_true,
        eval_fn=_mock_eval,
        metric_key="proxy_score",
        n_random_seeds=3,
    )
    assert res_true.passed is True

    # Orthogonal control must have 0 inner product and matched Frobenius norm
    orth = make_orthogonal_random_patch(p_true, seed=7, restrict_to_support=True)
    dot = float(torch.sum(orth.deltas["0.weight"] * p_true.deltas["0.weight"]).item())
    assert abs(dot) < 1e-5
    assert abs(orth.frobenius_norm() - p_true.frobenius_norm()) < 1e-5

    # Redundant no-op patch (col 5 has 0 weight in x_eval) fails causal check & gets pruned by LOO
    noop_delta = torch.zeros(4, 16)
    noop_delta[:, 5] = 1.0
    p_noop = WeightPatch(
        step_id="noop_patch",
        category="test",
        method="synthetic",
        target_layers=[0],
        rank=1,
        alpha=1.0,
        hypothesis="noop",
        deltas={"0.weight": noop_delta},
    )
    res_noop = run_three_point_causal_check(
        model=model,
        base_cpu_state=base_cpu,
        prior_patches=[],
        candidate_patch=p_noop,
        eval_fn=_mock_eval,
        metric_key="proxy_score",
        n_random_seeds=3,
    )
    assert res_noop.passed is False

    kept, loo_info = leave_one_out_ablation(
        model=model,
        base_cpu_state=base_cpu,
        patches=[p_true, p_noop],
        eval_fn=_mock_eval,
        replay_acc_floor=0.7,
        tol=0.005,
    )
    assert [p.step_id for p in kept] == ["true_patch"]
    assert loo_info["final_step_ids"] == ["true_patch"]


def test_replay_weighted_null_space() -> None:
    torch.manual_seed(123)
    # Preservation body tokens live in a 6-D subspace, while replay keys add a 4-D orthogonal subspace
    b_pres = torch.randn(6, 32)
    b_rep = torch.randn(4, 32)
    b_rep = b_rep - (b_rep @ b_pres.T @ torch.linalg.pinv(b_pres.T))
    k_preserve = torch.randn(50, 6) @ b_pres
    k_replay = torch.randn(16, 4) @ b_rep

    p_null, stats = build_null_space_projector(
        k_preserve,
        variance_keep=0.95,
        max_rank_frac=0.5,
        k_replay_extra=k_replay,
        replay_weight=3.0,
    )
    assert stats["replay_residual_norm_ratio"] < 0.05

    k_target = torch.randn(20, 32)
    r_target = torch.randn(20, 12)
    dw = solve_alphaedit_update(r_target, k_target, p_null, rank=2, lam=10.0, alpha=1.0)
    replay_leak = torch.linalg.norm(dw @ k_replay.T).item()
    assert replay_leak < 1e-4


def test_fit_joint_contrastive_readout_synthetic() -> None:
    rng = np.random.default_rng(7)
    d = 32

    def _make_dataset(n_rc_fam: int, n_c5_fam: int):
        xs, meta = [], []
        # real_cf pairs: signal in dim 0, large family nuisance offset in dims 2..5
        for f in range(n_rc_fam):
            nuisance = np.zeros(d)
            nuisance[2:6] = rng.normal(0.0, 2.5, size=4)
            x_a = nuisance + rng.normal(0.0, 0.15, size=d)
            x_a[0] = -1.5
            x_b = nuisance + rng.normal(0.0, 0.15, size=d)
            x_b[0] = +1.5
            xs.extend([x_a, x_b])
            meta.append({"id": f"rc_{f}_allow", "subset": "real_cf", "family": f"rc_{f}", "pattern": "real_benign", "y": 0})
            meta.append({"id": f"rc_{f}_block", "subset": "real_cf", "family": f"rc_{f}", "pattern": "cf_risk", "y": 1})
        # consent_5p families: signal in dim 1 + family nuisance offset in dims 2..5
        for f in range(n_c5_fam):
            nuisance = np.zeros(d)
            nuisance[2:6] = rng.normal(0.0, 2.5, size=4)
            for pat, y in [
                ("p1_no_consent", 1),
                ("p2_exact_consent", 0),
                ("p3_target_mismatch", 1),
                ("p5a_blocked_approved_same", 0),
                ("p5b_blocked_swapped_target", 1),
            ]:
                x_row = nuisance + rng.normal(0.0, 0.15, size=d)
                x_row[1] = 1.6 if y == 1 else -1.6
                xs.append(x_row)
                meta.append({"id": f"c5_{f}_{pat}", "subset": "consent_5p", "family": f"c5_{f}", "pattern": pat, "y": y})
        return np.asarray(xs, dtype=np.float32), meta

    x_tr, m_tr = _make_dataset(16, 8)
    x_te, m_te = _make_dataset(12, 6)
    w, b, _ = fit_joint_contrastive_readout(x_tr, m_tr, lam=None)
    scores = x_te @ w + b
    y_te = np.array([m["y"] for m in m_te])
    auc = float(roc_auc_score(y_te, scores))
    assert auc == 1.0


def test_swiglu_patch_prior_weights_invariance() -> None:
    torch.manual_seed(99)
    d_model, d_ff = 16, 32

    class _DummyMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
            self.up_proj = nn.Linear(d_model, d_ff, bias=False)
            self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    class _DummyLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = _DummyMLP()

    class _DummyBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([_DummyLayer()])

    class _DummyCausalLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = _DummyBackbone()
            self.lm_head = nn.Linear(d_model, 4, bias=False)

    class _DummyRunner:
        def __init__(self) -> None:
            self.model = _DummyCausalLM()
            self.letter_token_ids = {"A": 0, "B": 1}

    runner = _DummyRunner()
    prior_weights = {
        "model.layers.0.mlp.gate_proj.weight": runner.model.model.layers[0].mlp.gate_proj.weight.detach().float().cpu().clone(),
        "model.layers.0.mlp.up_proj.weight": runner.model.model.layers[0].mlp.up_proj.weight.detach().float().cpu().clone(),
        "model.layers.0.mlp.down_proj.weight": runner.model.model.layers[0].mlp.down_proj.weight.detach().float().cpu().clone(),
    }

    meta = []
    for i in range(16):
        sub = "real_cf" if i < 8 else "consent_5p"
        pat = "real_benign" if i % 2 == 0 else "p3_target_mismatch"
        meta.append({"id": f"ex_{i}", "subset": sub, "family": f"fam_{i // 2}", "pattern": pat, "y": i % 2})

    x_norm = torch.randn(16, d_model)
    x_rev = torch.randn(16, d_model)
    x_pres = torch.randn(24, d_model)
    obs = ObserveActivations(
        layers=[0],
        plain_norm={"mlp_in": {0: x_norm}},
        plain_rev={"mlp_in": {0: x_rev}},
        teacher_norm={},
        preserve={"mlp_in": {0: x_pres}},
        replay_only={},
        mlp_preserve_energy={0: torch.rand(d_ff) + 0.1},
        meta_automode=meta,
    )

    p1 = build_swiglu_order_gate_patch(
        runner=runner,
        obs=obs,
        prior_layer_weights=prior_weights,
        layer_idx=0,
        layer_gain=1.5,
        target_logit_std=1.2,
        target_bias_shift=0.6,
    )
    # Mutate live weights on runner.model in-place (simulating an active sweep)
    with torch.no_grad():
        runner.model.model.layers[0].mlp.gate_proj.weight.add_(5.0)
        runner.model.model.layers[0].mlp.up_proj.weight.add_(-3.0)
        runner.model.model.layers[0].mlp.down_proj.weight.add_(2.5)

    p2 = build_swiglu_order_gate_patch(
        runner=runner,
        obs=obs,
        prior_layer_weights=prior_weights,
        layer_idx=0,
        layer_gain=1.5,
        target_logit_std=1.2,
        target_bias_shift=0.6,
    )
    for k in p1.deltas:
        assert torch.equal(p1.deltas[k], p2.deltas[k])
