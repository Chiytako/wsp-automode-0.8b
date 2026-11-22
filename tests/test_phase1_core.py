"""Unit tests for Phase 1 core modules (CPU-only, zero frozen_test exposure)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from automode.questions import (
    Q_BLOCK_BINARY,
    build_decider_prompt,
    build_oracle_prompt,
    build_plain_prompt,
)
from automode.view import render_view
from eval.evaluate import load_split_rows

ROOT = Path(__file__).resolve().parent.parent


def test_view_reasoning_blind_and_blocked_action() -> None:
    row = {
        "env": "local_dev",
        "cwd": "/work/app",
        "user_messages": ["First rule: test locally", "Middle ping", "Yes, run it now"],
        "prior_actions": ["Bash: pytest -q", "Edit: src/main.py"],
        "blocked_actions": [{"index": 1, "tool": "Bash", "summary": "rm -rf /tmp/x", "reason": "recursive delete"}],
        "tool_name": "Bash",
        "tool_input": {"command": "rm -rf /tmp/x"},
        "assistant_reasoning": "SECRET_PROSE_MUST_NOT_APPEAR",
        "tool_result": "SECRET_RESULT_MUST_NOT_APPEAR",
    }
    v_off = render_view(row, include_recap=False)
    v_on = render_view(row, include_recap=True)
    assert "SECRET_PROSE_MUST_NOT_APPEAR" not in v_off.text
    assert "SECRET_RESULT_MUST_NOT_APPEAR" not in v_off.text
    assert "Blocked action 1: Bash(rm -rf /tmp/x) [reason: recursive delete]" in v_off.text
    assert "User instructions recap:" not in v_off.text
    assert "User instructions recap:" in v_on.text
    assert v_off.prefix.endswith("\n")
    assert v_off.suffix.startswith("Proposed action:")
    assert not v_off.prefix.endswith("\n\n")


def test_view_budget_preserves_blocked_action() -> None:
    class _DummyTok:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return list(range(max(1, len(text) // 3)))

    long_priors = [f"Bash: echo step_{i}_" + ("x" * 120) for i in range(40)]
    long_priors.append("Blocked action 1: Bash(git push --force origin feat-1) [reason: force push]")
    row = {
        "env": "default",
        "cwd": "/work/repo",
        "user_messages": ["Push feat-1", "Confirm"],
        "prior_actions": long_priors,
        "tool_name": "Bash",
        "tool_input": {"command": "git push --force origin feat-1"},
    }
    v = render_view(row, tok=_DummyTok(), budget=400)
    assert "Blocked action 1: Bash(git push --force origin feat-1)" in v.text
    assert v.dropped_prior > 0


def test_questions_reversal_and_oracle_trailing_slot() -> None:
    ctx = "Environment:\n- test\nProposed action:\nTool: Bash\ncommand: ls"
    p_norm = build_plain_prompt(ctx, Q_BLOCK_BINARY, reverse_options=False)
    p_rev = build_plain_prompt(ctx, Q_BLOCK_BINARY, reverse_options=True)
    p_ora = build_oracle_prompt(ctx, Q_BLOCK_BINARY, reverse_options=False)

    assert p_norm.option_letters == ["A", "B"]
    assert p_norm.option_labels == ["no", "yes"]
    assert p_norm.no_idx == 0 and p_norm.yes_idx == 1

    assert p_rev.option_labels == ["yes", "no"]
    assert p_rev.yes_idx == 0 and p_rev.no_idx == 1

    assert p_norm.text.endswith("Answer: (")
    tail_norm = p_norm.text.split("\n\nQuestion:", 1)[1]
    tail_ora = p_ora.text.split("\n\nQuestion:", 1)[1]
    assert tail_norm == tail_ora


def test_decider_replay_question_types() -> None:
    p_noul = build_decider_prompt("ctx", {"type": "noul", "instructions": "Is it valid?"})
    assert p_noul.option_letters == ["A", "B"]
    assert "(A) no" in p_noul.text and "(B) yes" in p_noul.text

    p_choice = build_decider_prompt("ctx", {"type": "choice", "instructions": "Pick one", "criteria": ["x", "y", "z"]})
    assert p_choice.option_letters == ["A", "B", "C"]

    p_score = build_decider_prompt("ctx", {"type": "score", "instructions": "Rate", "criteria": ["low", "mid", "high"]})
    assert "(0: low)" not in p_score.text
    assert "(A) 0: low" in p_score.text


def test_manifest_and_frozen_test_lock() -> None:
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["frozen_test_locked"] is True
    assert manifest["zero_leakage_verified"] is True

    splits = manifest["splits"]
    assert set(splits.keys()) == {"observe", "select_dev", "frozen_test"}
    pg_obs = set(splits["observe"]["project_groups"])
    pg_dev = set(splits["select_dev"]["project_groups"])
    pg_tst = set(splits["frozen_test"]["project_groups"])
    assert pg_obs.isdisjoint(pg_dev)
    assert pg_obs.isdisjoint(pg_tst)
    assert pg_dev.isdisjoint(pg_tst)

    with pytest.raises(PermissionError):
        load_split_rows("frozen_test", unlock_frozen_test=False)
