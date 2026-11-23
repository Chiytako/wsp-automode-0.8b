"""Standard llama.cpp `/completion` client, boundary second-reading, and turn guard loop.

Enforces all deployment & safety non-negotiables:
  - Plain Mapika/decider wire layout (no BOS, no chat template, `n_predict=1`, `n_probs=16`)
  - Strictly binary output (`Allow` / `Block`) with no human-popup gray band
  - Boundary-only `second_reading` (`p_block in [0.30, 0.70]`)
  - `Block`-only secondary `Q_REASON` slot query for structured agent feedback
  - Reasoning-Blind state management trusting only harness-issued `Blocked action k:` logs
  - Per-turn consecutive block limit without any harness approval bypass.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from automode.questions import LETTERS, Q_BLOCK_BINARY, Q_REASON, build_plain_prompt
from automode.view import render_view


def sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _extract_token_logprobs(resp: dict[str, Any]) -> dict[str, float]:
    """Parse `/completion` token probabilities across `top_probs`, `top_logprobs`, or `probs` layouts."""
    comp_probs = resp.get("completion_probabilities") or resp.get("probs") or []
    if not comp_probs:
        raise ValueError("Missing completion_probabilities in llama-server /completion response")

    first = comp_probs[0]
    if isinstance(first, dict) and ("top_probs" in first or "top_logprobs" in first or "probs" in first):
        entries = first.get("top_logprobs") or first.get("top_probs") or first.get("probs") or []
    elif isinstance(first, dict) and ("token" in first or "tok_str" in first):
        entries = comp_probs
    else:
        raise ValueError(f"Unrecognized completion_probabilities layout: {first!r}")

    out: dict[str, float] = {}
    for item in entries:
        tok = item.get("token") if "token" in item else item.get("tok_str")
        if tok is None:
            continue
        if "logprob" in item and item["logprob"] is not None:
            lp = float(item["logprob"])
        elif "prob" in item and item["prob"] is not None:
            lp = math.log(max(float(item["prob"]), 1e-12))
        else:
            continue
        if tok not in out:
            out[str(tok)] = lp
    return out


def score_options_from_logprobs(
    tok_logprobs: dict[str, float],
    option_letters: list[str],
    yes_idx: int | None = None,
    no_idx: int | None = None,
) -> dict[str, Any]:
    """Compute softmax probabilities and binary logit margin across `option_letters`."""
    min_seen = min(tok_logprobs.values()) if tok_logprobs else -20.0
    fallback = min(min_seen - 4.0, -20.0)

    opt_logits: list[float] = []
    for letter in option_letters:
        opt_logits.append(float(tok_logprobs.get(letter, fallback)))

    max_l = max(opt_logits)
    exps = [math.exp(v - max_l) for v in opt_logits]
    z_sum = sum(exps)
    opt_probs = [e / z_sum for e in exps]
    pred_idx = max(range(len(opt_probs)), key=opt_probs.__getitem__)

    margin: float | None = None
    p_yes: float | None = None
    p_no: float | None = None
    if yes_idx is not None and no_idx is not None:
        margin = opt_logits[yes_idx] - opt_logits[no_idx]
        p_yes = sigmoid(margin)
        p_no = 1.0 - p_yes

    return {
        "option_letters": list(option_letters),
        "option_logits": opt_logits,
        "option_probs": opt_probs,
        "pred_index": pred_idx,
        "pred_letter": option_letters[pred_idx],
        "p_yes": p_yes,
        "p_no": p_no,
        "logit_margin": margin,
    }


def _summarize_proposed_action(tool_name: str, tool_input: Any) -> str:
    if isinstance(tool_input, dict):
        if "command" in tool_input:
            raw = str(tool_input["command"]).strip()
        elif "file_path" in tool_input or "path" in tool_input:
            raw = str(tool_input.get("file_path") or tool_input.get("path")).strip()
        else:
            raw = json.dumps(tool_input, ensure_ascii=False)
    else:
        raw = str(tool_input).strip()
    if len(raw) > 120:
        raw = raw[:117] + "..."
    return f"{tool_name}({raw})"


REASON_LABELS = list(Q_REASON["criteria"].keys())


@dataclass
class AutoModeDecision:
    """Strict binary decision returned by the Qwen3.5-0.8B Auto-Mode Decider."""

    decision: str  # "Allow" or "Block"
    p_block: float
    p_block_normal: float
    margin_normal: float
    second_reading_triggered: bool
    margin_reversed: float | None = None
    reason_category: str | None = None
    feedback_message: str | None = None
    trusted_blocked_log_entry: str | None = None
    turn_limit_reached: bool = False
    consecutive_blocks_in_turn: int = 0
    latencies_ms: list[float] = field(default_factory=list)


class LlamaCompletionClient:
    """Minimal HTTP client calling standard `llama-server` `/completion` with `n_predict=1`."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8091",
        timeout_sec: float = 60.0,
        n_probs: int = 16,
        second_reading_band: tuple[float, float] = (0.30, 0.70),
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = float(timeout_sec)
        self.n_probs = int(n_probs)
        self.second_reading_band = second_reading_band

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"llama-server HTTP {e.code} on {path}: {detail}") from e

    def tokenize(self, text: str, add_special: bool = False) -> list[int]:
        res = self._post_json("/tokenize", {"content": text, "add_special": add_special})
        return [int(t) for t in res.get("tokens", [])]

    def complete_slot(
        self,
        prompt_text: str,
        option_letters: list[str],
        yes_idx: int | None = None,
        no_idx: int | None = None,
    ) -> dict[str, Any]:
        payload = {
            "prompt": prompt_text,
            "n_predict": 1,
            "n_probs": max(self.n_probs, len(option_letters)),
            "temperature": 0.0,
            "top_k": 40,
            "top_p": 1.0,
            "min_p": 0.0,
            "cache_prompt": True,
        }
        t0 = time.perf_counter()
        raw = self._post_json("/completion", payload)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        tok_lp = _extract_token_logprobs(raw)
        scored = score_options_from_logprobs(tok_lp, option_letters, yes_idx=yes_idx, no_idx=no_idx)
        scored["latency_ms"] = elapsed_ms
        scored["tokens_evaluated"] = raw.get("tokens_evaluated")
        return scored

    def score_binary_paired(self, context_text: str, force_paired: bool = False) -> dict[str, Any]:
        p_norm = build_plain_prompt(context_text, question_spec=Q_BLOCK_BINARY, reverse_options=False)
        res_norm = self.complete_slot(p_norm.text, p_norm.option_letters, yes_idx=p_norm.yes_idx, no_idx=p_norm.no_idx)
        p_norm_val = float(res_norm["p_yes"])
        m_norm = float(res_norm["logit_margin"])

        lo, hi = self.second_reading_band
        triggered = force_paired or (lo <= p_norm_val <= hi)
        latencies = [float(res_norm["latency_ms"])]

        m_rev: float | None = None
        p_rev_val: float | None = None
        if triggered:
            p_rev = build_plain_prompt(context_text, question_spec=Q_BLOCK_BINARY, reverse_options=True)
            res_rev = self.complete_slot(p_rev.text, p_rev.option_letters, yes_idx=p_rev.yes_idx, no_idx=p_rev.no_idx)
            m_rev = float(res_rev["logit_margin"])
            p_rev_val = float(res_rev["p_yes"])
            latencies.append(float(res_rev["latency_ms"]))
            m_final = 0.5 * (m_norm + m_rev)
            p_final = sigmoid(m_final)
        else:
            m_final = m_norm
            p_final = p_norm_val

        return {
            "p_block_normal": p_norm_val,
            "p_block_reversed": p_rev_val,
            "p_block_final": p_final,
            "margin_normal": m_norm,
            "margin_reversed": m_rev,
            "margin_final": m_final,
            "second_reading_triggered": lo <= p_norm_val <= hi,
            "decision": "Block" if p_final >= 0.5 else "Allow",
            "latencies_ms": latencies,
        }

    def query_block_reason(self, context_text: str) -> tuple[str, float]:
        """Run secondary single-pass Q_REASON query (invoked strictly when decision == 'Block')."""
        p_reason = build_plain_prompt(context_text, question_spec=Q_REASON, reverse_options=False)
        res = self.complete_slot(p_reason.text, p_reason.option_letters)
        # Pick highest-probability non-safe risk category (or top overall if all else low)
        probs = list(res["option_probs"])
        best_idx = int(res["pred_index"])
        if REASON_LABELS[best_idx] == "none_safe" and len(probs) > 1:
            non_safe = [i for i, lbl in enumerate(REASON_LABELS) if lbl != "none_safe"]
            best_idx = max(non_safe, key=probs.__getitem__)
        return REASON_LABELS[best_idx], float(res["latency_ms"])

    def score_question(self, context_text: str, question_spec: dict[str, Any]) -> dict[str, Any]:
        """Evaluate general Mapika/decider noul/choice/score question via /completion."""
        pb = build_plain_prompt(context_text, question_spec=question_spec, reverse_options=False)
        return self.complete_slot(pb.text, pb.option_letters, yes_idx=pb.yes_idx, no_idx=pb.no_idx)


class AutoModeTurnGuard:
    """Stateful turn-level guard implementing binary Allow/Block + structured feedback loop.

    Maintains strict Reasoning-Blind history: only harness-issued `Blocked action k:` logs
    are retained in `blocked_actions` (no agent free-text or tool outputs; no consent bypass).
    """

    def __init__(
        self,
        client: LlamaCompletionClient,
        max_consecutive_blocks: int = 3,
        budget_tokens: int = 2048,
    ) -> None:
        self.client = client
        self.max_consecutive_blocks = int(max_consecutive_blocks)
        self.budget_tokens = int(budget_tokens)
        self.consecutive_blocks = 0
        self.blocked_actions: list[str] = []

    def reset_turn(self) -> None:
        self.consecutive_blocks = 0

    def check_action(
        self,
        state: dict[str, Any],
        tool_name: str,
        tool_input: Any,
    ) -> AutoModeDecision:
        """Decide `Allow` or `Block` solely via the Qwen3.5-0.8B model."""
        eval_state = {
            "env": state.get("env", "local_dev"),
            "cwd": state.get("cwd", "/workspace"),
            "user_messages": list(state.get("user_messages", [])),
            "prior_actions": list(state.get("prior_actions", [])),
            "blocked_actions": list(self.blocked_actions),
            "tool_name": tool_name,
            "tool_input": tool_input,
        }
        ctx = render_view(eval_state, tok=None, budget=self.budget_tokens, include_recap=False).text
        scored = self.client.score_binary_paired(ctx, force_paired=False)
        decision = str(scored["decision"])
        latencies = list(scored["latencies_ms"])

        if decision == "Allow":
            self.consecutive_blocks = 0
            return AutoModeDecision(
                decision="Allow",
                p_block=float(scored["p_block_final"]),
                p_block_normal=float(scored["p_block_normal"]),
                margin_normal=float(scored["margin_normal"]),
                second_reading_triggered=bool(scored["second_reading_triggered"]),
                margin_reversed=scored["margin_reversed"],
                consecutive_blocks_in_turn=0,
                latencies_ms=latencies,
            )

        # Decision == "Block": query Q_REASON once and issue structured harness feedback
        reason_cat, r_lat = self.client.query_block_reason(ctx)
        latencies.append(r_lat)

        self.consecutive_blocks += 1
        k_idx = len(self.blocked_actions) + 1
        summary = _summarize_proposed_action(tool_name, tool_input)
        log_entry = f"Blocked action {k_idx}: {summary} [reason: {reason_cat}]"
        self.blocked_actions.append(log_entry)

        limit_hit = self.consecutive_blocks >= self.max_consecutive_blocks
        if limit_hit:
            feedback = (
                f"[Auto-Mode Block #{k_idx} | reason={reason_cat}] Proposed action `{summary}` was blocked. "
                f"Consecutive block limit ({self.max_consecutive_blocks}) reached for this turn: "
                "halt execution and ask the user explicitly for instruction before retrying."
            )
        else:
            feedback = (
                f"[Auto-Mode Block #{k_idx} | reason={reason_cat}] Proposed action `{summary}` was blocked. "
                "Either choose a safer reversible alternative inside the workspace or ask the user for explicit confirmation "
                "naming the exact operation and exact target."
            )

        return AutoModeDecision(
            decision="Block",
            p_block=float(scored["p_block_final"]),
            p_block_normal=float(scored["p_block_normal"]),
            margin_normal=float(scored["margin_normal"]),
            second_reading_triggered=bool(scored["second_reading_triggered"]),
            margin_reversed=scored["margin_reversed"],
            reason_category=reason_cat,
            feedback_message=feedback,
            trusted_blocked_log_entry=log_entry,
            turn_limit_reached=limit_hit,
            consecutive_blocks_in_turn=self.consecutive_blocks,
            latencies_ms=latencies,
        )
