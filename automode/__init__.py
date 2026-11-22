"""Reasoning-blind state views and plain-layout typed questions for wsp-automode-0.8b."""

from automode.questions import (
    LETTERS,
    ORACLE_POLICY_AND_FEWSHOT,
    Q_BLOCK_BINARY,
    Q_REASON,
    PromptBuild,
    build_decider_prompt,
    build_oracle_prompt,
    build_plain_prompt,
)
from automode.view import ENV_PRESETS, Scenario, View, normalize_scenario, render_view

__all__ = [
    "ENV_PRESETS",
    "LETTERS",
    "ORACLE_POLICY_AND_FEWSHOT",
    "PromptBuild",
    "Q_BLOCK_BINARY",
    "Q_REASON",
    "Scenario",
    "View",
    "build_decider_prompt",
    "build_oracle_prompt",
    "build_plain_prompt",
    "normalize_scenario",
    "render_view",
]
