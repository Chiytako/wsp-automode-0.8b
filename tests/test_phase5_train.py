"""Unit tests for Phase 5 Track 3+ dataset isolation, LoRA equivalence, losses, and decoupled gradients."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from data.build_datasets import sha256_file
from data.build_phase5_train import PHASE5_MANIFEST_PATH, PHASE5_TRAIN_PATH, verify_zero_leakage
from eval.model_runner import capture_cpu_state_dict, restore_cpu_state_dict
from train.lora_model import (
    LoRALinear,
    extract_merged_patch_bundle,
    inject_lora,
    load_lora_adapter,
    save_lora_adapter,
)
from train.losses import compute_binary_family_loss, compute_replay_loss
from train.soup_and_calibrate import blend_patch_bundles, build_norm_temperature_delta
from runtime.upload_hf import ONE_GIB, assert_upload_hygiene, build_model_cards
from weight_edit.causal_check import PatchBundle, WeightPatch, apply_bundle

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "data" / "manifest.json"


def test_phase5_train_zero_leakage_and_base_hashes_intact() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for split_name in ("observe", "select_dev", "frozen_test"):
        info = manifest["splits"][split_name]
        assert sha256_file(ROOT / info["file"]) == info["sha256"]

    p5_manifest = json.loads(PHASE5_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert p5_manifest["audit"]["zero_leakage_vs_select_dev_and_frozen_test"] is True
    assert p5_manifest["audit"]["base_splits_sha256_intact"] is True
    assert sha256_file(PHASE5_TRAIN_PATH) == p5_manifest["sha256"]

    rows = [json.loads(ln) for ln in PHASE5_TRAIN_PATH.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(rows) == p5_manifest["total_rows"]
    audit = verify_zero_leakage(rows)
    assert audit["zero_leakage_vs_select_dev_and_frozen_test"] is True


def test_lora_linear_merge_equivalence_and_bundle_roundtrip(tmp_path: Path) -> None:
    torch.manual_seed(42)

    class TinyBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = nn.Module()
            self.self_attn.q_proj = nn.Linear(16, 16, bias=False)
            self.mlp = nn.Module()
            self.mlp.gate_proj = nn.Linear(16, 24, bias=False)
            self.mlp.down_proj = nn.Linear(24, 16, bias=False)
            self.linear_attn = nn.Module()
            self.linear_attn.A_log = nn.Parameter(torch.randn(4))
            self.linear_attn.dt_bias = nn.Parameter(torch.randn(4))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            q = self.self_attn.q_proj(x)
            h = self.mlp.down_proj(torch.relu(self.mlp.gate_proj(q)))
            return h * (self.linear_attn.A_log.sum() + self.linear_attn.dt_bias.sum())

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([TinyBlock()])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model.layers[0](x)

    model = TinyModel()
    base_cpu = capture_cpu_state_dict(model)
    info = inject_lora(model, r=4, alpha=8.0, train_deltanet_gates=True)
    assert info["wrapped_modules"] == 3
    assert info["gate_params"] == 8

    # Simulate trained non-zero LoRA weights
    with torch.no_grad():
        for mod in model.modules():
            if isinstance(mod, LoRALinear):
                mod.lora_A.normal_(std=0.2)
                mod.lora_B.normal_(std=0.2)
        model.model.layers[0].linear_attn.A_log.add_(0.15)
        model.model.layers[0].linear_attn.dt_bias.sub_(0.10)

    x = torch.randn(3, 16)
    out_lora = model(x).detach()

    # Verify individual LoRALinear exact merge equivalence
    q_mod: LoRALinear = model.model.layers[0].self_attn.q_proj  # type: ignore[assignment]
    w_merged = q_mod.merged_weight()
    out_lin_lora = q_mod(x)
    out_lin_merged = torch.nn.functional.linear(x, w_merged, None)
    assert torch.allclose(out_lin_lora, out_lin_merged, atol=1e-5, rtol=1e-5)

    # Extract merged PatchBundle, apply to a fresh stock model, verify exact output match
    bundle = extract_merged_patch_bundle(model, base_cpu, step_id="test_merge", metadata={"r": 4, "alpha": 8.0})
    adapter_path = tmp_path / "adapter.pt"
    save_lora_adapter(model, adapter_path, metadata={"ok": True})

    fresh_merged = TinyModel()
    apply_bundle(fresh_merged, base_cpu, bundle.patches)
    out_bundle = fresh_merged(x).detach()
    assert torch.allclose(out_lora, out_bundle, atol=1e-5, rtol=1e-5)

    # Verify adapter load roundtrip matches too
    fresh_lora = TinyModel()
    restore_cpu_state_dict(fresh_lora, base_cpu)
    inject_lora(fresh_lora, r=4, alpha=8.0, train_deltanet_gates=True)
    meta = load_lora_adapter(fresh_lora, adapter_path)
    assert meta["ok"] is True
    assert torch.allclose(out_lora, fresh_lora(x).detach(), atol=1e-5, rtol=1e-5)


def test_decoupled_scalar_chain_rule_gradient_parity() -> None:
    torch.manual_seed(123)

    def make_net() -> nn.Sequential:
        net = nn.Sequential(nn.Linear(8, 12), nn.SiLU(), nn.Linear(12, 4))
        return net

    net_joint = make_net()
    net_dec = make_net()
    net_dec.load_state_dict(net_joint.state_dict())

    xs_n = [torch.randn(8) for _ in range(4)]
    xs_r = [torch.randn(8) for _ in range(4)]
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    pairs = [(0, 1), (2, 3)]

    # 1) Joint binary loss backward
    net_joint.zero_grad()
    m_n_j = torch.stack([net_joint(x)[1] - net_joint(x)[0] for x in xs_n])
    m_r_j = torch.stack([net_joint(x)[0] - net_joint(x)[1] for x in xs_r])
    loss_j, _ = compute_binary_family_loss(m_n_j, m_r_j, labels, pairs, is_synthetic=True)
    loss_j.backward()

    # 2) Decoupled scalar-chain-rule backward (no joint activation graph held across examples)
    net_dec.zero_grad()
    with torch.no_grad():
        m_n_vals = [float((net_dec(x)[1] - net_dec(x)[0]).item()) for x in xs_n]
        m_r_vals = [float((net_dec(x)[0] - net_dec(x)[1]).item()) for x in xs_r]
    t_mn = torch.tensor(m_n_vals, requires_grad=True)
    t_mr = torch.tensor(m_r_vals, requires_grad=True)
    loss_d, _ = compute_binary_family_loss(t_mn, t_mr, labels, pairs, is_synthetic=True)
    loss_d.backward()
    assert t_mn.grad is not None and t_mr.grad is not None
    for k in range(4):
        mk = net_dec(xs_n[k])[1] - net_dec(xs_n[k])[0]
        (mk * t_mn.grad[k]).backward()
        mrk = net_dec(xs_r[k])[0] - net_dec(xs_r[k])[1]
        (mrk * t_mr.grad[k]).backward()

    for (n1, p1), (n2, p2) in zip(net_joint.named_parameters(), net_dec.named_parameters()):
        assert n1 == n2
        assert p1.grad is not None and p2.grad is not None
        assert torch.allclose(p1.grad, p2.grad, atol=1e-5, rtol=1e-5)

    # 3) Decoupled replay loss parity (4-option choice)
    net_joint.zero_grad()
    net_dec.zero_grad()
    bp_n = torch.softmax(torch.randn(4), dim=-1)
    bp_r = torch.softmax(torch.randn(4), dim=-1)
    ln_j = net_joint(xs_n[0])
    lr_j = net_joint(xs_r[0])
    rl_j, _ = compute_replay_loss(ln_j, lr_j, gold_idx=1, base_probs_norm=bp_n, base_probs_rev=bp_r)
    rl_j.backward()

    with torch.no_grad():
        ln_val = net_dec(xs_n[0])
        lr_val = net_dec(xs_r[0])
    t_ln = ln_val.clone().detach().requires_grad_(True)
    t_lr = lr_val.clone().detach().requires_grad_(True)
    rl_d, _ = compute_replay_loss(t_ln, t_lr, gold_idx=1, base_probs_norm=bp_n, base_probs_rev=bp_r)
    rl_d.backward()
    assert t_ln.grad is not None and t_lr.grad is not None
    torch.sum(net_dec(xs_n[0]) * t_ln.grad).backward()
    torch.sum(net_dec(xs_r[0]) * t_lr.grad).backward()

    for (_, p1), (_, p2) in zip(net_joint.named_parameters(), net_dec.named_parameters()):
        assert p1.grad is not None and p2.grad is not None
        assert torch.allclose(p1.grad, p2.grad, atol=1e-5, rtol=1e-5)


def test_losses_symmetry_and_contrastive_direction() -> None:
    # Monotonic push: when P3 margin < P2 margin (wrong ordering), grad pushes P3 UP and P2 DOWN
    mn = torch.tensor([0.5, 1.0, 0.2, 0.8], requires_grad=True)  # [P1, P2, P3, P4]
    mr = torch.tensor([0.5, 1.0, 0.2, 0.8], requires_grad=True)
    labels = torch.tensor([1.0, 0.0, 1.0, 1.0])
    loss, stats = compute_binary_family_loss(mn, mr, labels, pair_indices=[(0, 1), (2, 1), (3, 1)], is_synthetic=True)
    assert torch.isfinite(loss)
    assert stats["ord"] == 0.0
    loss.backward()
    assert mn.grad is not None and mr.grad is not None
    # Symmetry: equal m_norm and m_rev receive identical gradients
    assert torch.allclose(mn.grad, mr.grad, atol=1e-6)
    # P2 (label=0, negative of all pairs) has positive gradient -> gradient descent reduces P2 margin
    assert float(mn.grad[1].item()) > 0.0
    # P3 (label=1, block) has negative gradient -> gradient descent increases P3 margin
    assert float(mn.grad[2].item()) < 0.0

    # Option-reversal symmetry in compute_replay_loss
    ln = torch.tensor([2.0, 0.5, -1.0], requires_grad=True)
    lr = torch.flip(ln.detach(), dims=[0]).clone().requires_grad_(True)
    bp_n = torch.tensor([0.7, 0.2, 0.1])
    bp_r = torch.flip(bp_n, dims=[0])
    rl, rstats = compute_replay_loss(ln, lr, gold_idx=0, base_probs_norm=bp_n, base_probs_rev=bp_r)
    assert torch.isfinite(rl)
    assert rstats["ord"] < 1e-6
    rl.backward()
    assert ln.grad is not None and lr.grad is not None
    assert torch.allclose(ln.grad, torch.flip(lr.grad, dims=[0]), atol=1e-6)


def test_blend_patch_bundles_linearity_and_disjoint_keys() -> None:
    d_shared_a = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
    d_shared_b = torch.tensor([[5.0, 2.0], [-1.0, 0.0]])
    d_only_a = torch.tensor([2.0, -4.0])
    d_only_b = torch.tensor([8.0, 4.0])

    ba = PatchBundle(
        patches=[
            WeightPatch(
                step_id="a",
                category="c",
                method="m",
                target_layers=[0],
                rank=4,
                alpha=1.0,
                hypothesis="ha",
                deltas={"model.layers.0.mlp.gate_proj.weight": d_shared_a, "model.layers.0.linear_attn.A_log": d_only_a},
            )
        ]
    )
    bb = PatchBundle(
        patches=[
            WeightPatch(
                step_id="b",
                category="c",
                method="m",
                target_layers=[0, 1],
                rank=8,
                alpha=1.0,
                hypothesis="hb",
                deltas={"model.layers.0.mlp.gate_proj.weight": d_shared_b, "model.layers.1.linear_attn.dt_bias": d_only_b},
            )
        ]
    )

    b1 = blend_patch_bundles(ba, bb, lam=1.0, step_id="s1")
    net1 = b1.patches[0].net_deltas()
    assert torch.allclose(net1["model.layers.0.mlp.gate_proj.weight"], d_shared_a)
    assert torch.allclose(net1["model.layers.0.linear_attn.A_log"], d_only_a)
    assert "model.layers.1.linear_attn.dt_bias" not in net1

    b0 = blend_patch_bundles(ba, bb, lam=0.0, step_id="s0")
    net0 = b0.patches[0].net_deltas()
    assert torch.allclose(net0["model.layers.0.mlp.gate_proj.weight"], d_shared_b)
    assert torch.allclose(net0["model.layers.1.linear_attn.dt_bias"], d_only_b)
    assert "model.layers.0.linear_attn.A_log" not in net0

    b_half = blend_patch_bundles(ba, bb, lam=0.5, step_id="sh")
    neth = b_half.patches[0].net_deltas()
    assert torch.allclose(neth["model.layers.0.mlp.gate_proj.weight"], 0.5 * (d_shared_a + d_shared_b))
    assert torch.allclose(neth["model.layers.0.linear_attn.A_log"], 0.5 * d_only_a)
    assert torch.allclose(neth["model.layers.1.linear_attn.dt_bias"], 0.5 * d_only_b)


def test_rmsnorm_temperature_folding_exact_logit_scaling() -> None:
    torch.manual_seed(7)
    x = torch.randn(4, 16)
    w_norm = torch.randn(16) * 0.25  # Qwen3.5 RMSNorm computes normed(x) * (1 + w)
    w_head = torch.randn(5, 16)  # bias-free lm_head

    def _rmsnorm_logits(h: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        inv_rms = torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        normed = h * inv_rms * (1.0 + weight)
        return torch.nn.functional.linear(normed, w_head)

    logits_orig = _rmsnorm_logits(x, w_norm)
    for temp in (0.85, 1.25, 1.60):
        delta = build_norm_temperature_delta(w_norm, temp)
        logits_scaled = _rmsnorm_logits(x, w_norm + delta)
        assert torch.allclose(logits_scaled, logits_orig / temp, atol=1e-5, rtol=1e-5)


def test_hf_model_cards_and_upload_hygiene(tmp_path: Path) -> None:
    def _mock_metric(auroc: float = 0.95, pct: float = 0.5, frob: float = 1.5) -> dict:
        return {
            "weight_diff": {"pct_modified_params": pct, "total_delta_frobenius_norm": frob, "modified_params": 100, "total_params": 20000, "modified_tensor_count": 6},
            "proxy": {"proxy_score": 3.5},
            "real_cf": {
                "auroc": auroc,
                "recall_at_05": 0.95,
                "fpr_at_05": 0.05,
                "order_flip_rate": 0.02,
                "second_reading": {"auroc": auroc},
            },
            "consent_5p": {
                "overall_acc": 0.9,
                "p2_allow_rate": 0.9,
                "p3_block_rate": 0.9,
                "p5a_allow_rate": 0.9,
                "p5b_block_rate": 0.9,
                "order_flip_rate": 0.01,
            },
            "long_50t": {
                "overall_acc": 0.9,
                "with_recap_ablation": {"overall_acc": 0.85},
            },
            "decider_replay": {"overall_acc": 0.88, "top_label_ece": 0.08},
        }

    def _mock_parity(name: str) -> dict:
        return {
            "parity": {
                "gguf_file": name,
                "gguf_name": name,
                "gguf_bytes": 800_000_000,
                "gguf_mib": 762.94,
                "under_1gb": True,
                "tokenizer_exact_match": True,
                "p_block_mae_normal": 0.01,
                "p_block_max_ae_normal": 0.05,
                "p_block_pearson_r": 0.999,
                "agreement_binary_normal": 0.99,
                "agreement_binary_second_reading": 0.99,
                "agreement_replay": 0.98,
                "agreement_total_200": 0.99,
                "latency_1pass_p50_ms": 70.0,
                "latency_1pass_p95_ms": 300.0,
                "latency_e2e_p50_ms": 72.0,
                "latency_e2e_p95_ms": 310.0,
            },
            "pt_split_metrics": _mock_metric(0.99),
            "q8_split_metrics": _mock_metric(0.99),
            "q8_select_dev_metrics": _mock_metric(0.84),
        }

    metrics = {
        "t0a_dev": _mock_metric(0.56, 0.0, 0.0),
        "t0c_dev": _mock_metric(0.65, 0.0, 0.0),
        "t1_dev": _mock_metric(0.85, 0.0016, 8.0),
        "t2_dev": _mock_metric(0.87, 0.0033, 11.9),
        "t0a_frozen": _mock_metric(0.66, 0.0, 0.0),
        "t0c_frozen": _mock_metric(0.64, 0.0, 0.0),
        "t1_frozen": _mock_metric(0.82, 0.0016, 8.0),
        "t2_frozen": _mock_metric(0.83, 0.0033, 11.9),
        "p4_parity": _mock_parity("wsp-automode-0.8b-Q8_0.gguf"),
        "t2_spec": {},
        "t3a_dev": _mock_metric(1.0, 52.28, 2.80),
        "t3b_dev": _mock_metric(0.99, 51.53, 12.16),
        "t3c_dev": _mock_metric(1.0, 54.81, 3.48),
        "t3_dev": _mock_metric(1.0, 52.79, 2.77),
        "t3_p_dev": _mock_parity("automode-0.8b-track3-Q8_0.gguf"),
        "t3_p_frozen": _mock_parity("automode-0.8b-track3-Q8_0.gguf"),
        "t3_spec": {},
    }

    cards = build_model_cards("MockUser", metrics)
    assert set(cards.keys()) == {"wsp-automode-0.8b", "automode-0.8b"}

    ok_gguf = tmp_path / "ok.gguf"
    ok_gguf.write_bytes(b"GGUF")
    ok_spec = tmp_path / "spec.json"
    ok_spec.write_text('{"steps": []}\n', encoding="utf-8")

    assert_upload_hygiene(cards, spec_paths=[ok_spec], gguf_paths=[ok_gguf])

    for name, md in cards.items():
        assert md.startswith("---\n")
        assert "base_model: Qwen/Qwen3.5-0.8B" in md
        low = md.lower()
        assert "/home/sit" not in low, name
        assert "decider-auto" not in low, name
        assert "decider-4b" not in low, name
        assert "https://huggingface.co/MockUser/" in md

    # Hygiene guards must reject leaks & oversized GGUFs
    bad_cards_path = dict(cards, **{"automode-0.8b": cards["automode-0.8b"] + "\n/home/sit/leak\n"})
    try:
        assert_upload_hygiene(bad_cards_path)
        raise AssertionError("Expected local path check to fail")
    except ValueError:
        pass

    bad_cards_model = dict(cards, **{"wsp-automode-0.8b": cards["wsp-automode-0.8b"] + "\ndecider-4b\n"})
    try:
        assert_upload_hygiene(bad_cards_model)
        raise AssertionError("Expected forbidden model check to fail")
    except ValueError:
        pass

    big_gguf = tmp_path / "too_big.gguf"
    with open(big_gguf, "wb") as f:
        f.truncate(ONE_GIB)
    try:
        assert_upload_hygiene(cards, gguf_paths=[big_gguf])
        raise AssertionError("Expected >= 1 GiB GGUF check to fail")
    except ValueError:
        pass


