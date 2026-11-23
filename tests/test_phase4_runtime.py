"""Unit tests for Phase 4 HF export, llama.cpp client parsing, second_reading, and turn guard."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from runtime.client import (
    AutoModeTurnGuard,
    LlamaCompletionClient,
    _extract_token_logprobs,
    score_options_from_logprobs,
)
from runtime.export_hf import export_patched_hf, map_causal_to_hf_key
from weight_edit.causal_check import PatchBundle, WeightPatch, save_patch_bundle


def test_map_causal_to_hf_key_and_export_roundtrip(tmp_path: Path) -> None:
    assert map_causal_to_hf_key("model.layers.15.mlp.down_proj.weight") == "model.language_model.layers.15.mlp.down_proj.weight"
    assert map_causal_to_hf_key("model.language_model.layers.0.mlp.up_proj.weight") == "model.language_model.layers.0.mlp.up_proj.weight"

    src_dir = tmp_path / "base_hf"
    out_dir = tmp_path / "exported_hf"
    src_dir.mkdir(parents=True)
    (src_dir / "config.json").write_text('{"model_type": "qwen3_5"}\n', encoding="utf-8")

    torch.manual_seed(5)
    base_tensors = {
        "model.language_model.layers.15.mlp.down_proj.weight": torch.randn(8, 16, dtype=torch.bfloat16),
        "model.language_model.layers.15.mlp.gate_proj.weight": torch.randn(16, 8, dtype=torch.bfloat16),
        "model.visual.patch_embed.weight": torch.randn(4, 4, dtype=torch.bfloat16),
    }
    save_file(base_tensors, str(src_dir / "model.safetensors-00001-of-00001.safetensors"), metadata={"format": "pt"})

    d_down = torch.zeros(8, 16, dtype=torch.float32)
    d_down[:, 3] = 0.5
    patch = WeightPatch(
        step_id="mock_p",
        category="wsp",
        method="test",
        target_layers=[15],
        rank=1,
        alpha=1.0,
        hypothesis="test",
        deltas={"model.layers.15.mlp.down_proj.weight": d_down},
    )
    spec_path = tmp_path / "spec.json"
    pt_path = tmp_path / "deltas.pt"
    save_patch_bundle(PatchBundle(patches=[patch]), spec_path, pt_path)

    report = export_patched_hf(pt_path, out_dir, base_snapshot_dir=src_dir)
    assert report["modified_tensor_count"] == 1
    assert report["modified_params"] == 8
    expected_frob = float(torch.linalg.norm(d_down.bfloat16().float()).item())
    assert abs(report["total_delta_frobenius_norm"] - expected_frob) < 0.05

    reloaded = load_file(str(out_dir / "model.safetensors-00001-of-00001.safetensors"), device="cpu")
    assert set(reloaded.keys()) == set(base_tensors.keys())
    for k, base_t in base_tensors.items():
        assert reloaded[k].dtype == torch.bfloat16
        if k.endswith("down_proj.weight"):
            diff = reloaded[k].float() - base_t.float()
            assert torch.allclose(diff[:, 3], torch.full((8,), 0.5), atol=0.05)
        else:
            assert torch.equal(reloaded[k], base_t)


def test_client_completion_parsing_and_second_reading_band() -> None:
    # Support both `top_probs` and `top_logprobs` formats
    resp_probs = {
        "completion_probabilities": [
            {"top_probs": [{"token": "A", "prob": 0.25}, {"token": "B", "prob": 0.75}]}
        ]
    }
    lp1 = _extract_token_logprobs(resp_probs)
    sc1 = score_options_from_logprobs(lp1, ["A", "B"], yes_idx=1, no_idx=0)
    assert abs(sc1["p_yes"] - 0.75) < 1e-6

    resp_logprobs = {
        "completion_probabilities": [
            {"top_logprobs": [{"token": "A", "logprob": math.log(0.6)}, {"token": "B", "logprob": math.log(0.4)}]}
        ]
    }
    lp2 = _extract_token_logprobs(resp_logprobs)
    sc2 = score_options_from_logprobs(lp2, ["A", "B"], yes_idx=1, no_idx=0)
    assert abs(sc2["p_yes"] - 0.4) < 1e-6

    # Verify boundary band [0.30, 0.70]: outside skips second call, inside triggers 2nd call
    calls: list[str] = []

    class _MockClient(LlamaCompletionClient):
        def __init__(self, p_b_norm: float, p_b_rev: float) -> None:
            super().__init__(base_url="http://mock", second_reading_band=(0.30, 0.70))
            self.p_b_norm = p_b_norm
            self.p_b_rev = p_b_rev

        def _post_json(self, path: str, payload: dict) -> dict:
            calls.append(payload["prompt"])
            # Detect normal vs reversed order by `(A) no:` vs `(A) yes:`
            is_rev = "(A) yes:" in payload["prompt"]
            if not is_rev:
                p_b = self.p_b_norm
                p_a = 1.0 - p_b
            else:
                # Reversed order: A=yes (block), B=no (allow)
                p_a = self.p_b_rev
                p_b = 1.0 - p_a
            return {
                "completion_probabilities": [
                    {"top_probs": [{"token": "A", "prob": p_a}, {"token": "B", "prob": p_b}]}
                ]
            }

    # Confident Allow (0.08) -> outside band -> 1 call only
    calls.clear()
    res_out = _MockClient(p_b_norm=0.08, p_b_rev=0.90).score_binary_paired("Context text")
    assert res_out["second_reading_triggered"] is False
    assert res_out["decision"] == "Allow"
    assert len(calls) == 1

    # Uncertain (0.45 normal, 0.85 reversed -> flips avg > 0.5 -> Block) -> 2 calls
    calls.clear()
    res_in = _MockClient(p_b_norm=0.45, p_b_rev=0.85).score_binary_paired("Context text")
    assert res_in["second_reading_triggered"] is True
    assert len(calls) == 2
    assert res_in["decision"] == "Block"


def test_turn_guard_feedback_and_consecutive_block_cap() -> None:
    calls: list[str] = []

    class _GuardMockClient(LlamaCompletionClient):
        def __init__(self) -> None:
            super().__init__(base_url="http://mock", second_reading_band=(0.30, 0.70))
            self.next_allow = False

        def _post_json(self, path: str, payload: dict) -> dict:
            calls.append(payload["prompt"])
            if "Which primary risk category applies" in payload["prompt"]:
                # Answer (B) credentials_or_exfil
                return {
                    "completion_probabilities": [
                        {"top_probs": [{"token": "B", "prob": 0.9}, {"token": "A", "prob": 0.1}]}
                    ]
                }
            # Binary question: high confidence so 1-pass only
            p_b = 0.05 if self.next_allow else 0.95
            return {
                "completion_probabilities": [
                    {"top_probs": [{"token": "A", "prob": 1.0 - p_b}, {"token": "B", "prob": p_b}]}
                ]
            }

    client = _GuardMockClient()
    guard = AutoModeTurnGuard(client=client, max_consecutive_blocks=3)
    state = {"env": "local_dev", "cwd": "/workspace", "user_messages": ["Fix tests"], "prior_actions": []}

    # 1. Block #1 -> Q_REASON queried (2 calls total) & trusted Blocked action 1 recorded
    calls.clear()
    d1 = guard.check_action(state, "Bash", {"command": "cat ~/.ssh/id_rsa"})
    assert d1.decision == "Block"
    assert d1.reason_category == "credentials_or_exfil"
    assert d1.trusted_blocked_log_entry is not None and d1.trusted_blocked_log_entry.startswith("Blocked action 1:")
    assert d1.turn_limit_reached is False
    assert len(calls) == 2

    # 2. Allow -> 1 call only, Q_REASON NOT queried, consecutive counter resets
    calls.clear()
    client.next_allow = True
    d2 = guard.check_action(state, "Bash", {"command": "pytest -q"})
    assert d2.decision == "Allow"
    assert d2.reason_category is None
    assert d2.consecutive_blocks_in_turn == 0
    assert len(calls) == 1

    # 3. Consecutive blocks up to cap=3 -> never bypasses, flags turn_limit_reached=True at #3
    client.next_allow = False
    for idx in range(3):
        dec = guard.check_action(state, "Bash", {"command": f"rm -rf /tmp/x_{idx}"})
        assert dec.decision == "Block"
        assert dec.turn_limit_reached is (idx == 2)
