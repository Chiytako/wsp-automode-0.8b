"""Audit local Claude Code, Pi, and OMP session logs (read-only, privacy-safe).

Produces experiments/phase0_sessions_summary.json with anonymized per-session
counts, tool breakdowns, reasoning-blind view length statistics, and a
strict zero-leakage 3-split proposal (observe / select_dev / frozen_test).
No raw user text, secrets, or tool payloads are written to disk.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "experiments" / "phase0_sessions_summary.json"

READONLY_TOOLS = {
    "Read",
    "Glob",
    "Grep",
    "ToolSearch",
    "WebSearch",
    "WebFetch",
    "read",
    "fffind",
    "ffgrep",
    "web_search",
    "fetch_content",
    "source_check",
    "wait",
}

META_OR_INTERACTIVE_TOOLS = {
    "AskUserQuestion",
    "ask_user_question",
    "TaskStop",
    "mcp__t3-code__t3_background_wait",
}

ACTIONABLE_TOOLS = {
    "Bash",
    "Write",
    "Edit",
    "MultiEdit",
    "NotebookEdit",
    "Agent",
    "bash",
    "write",
    "edit",
    "eval",
}


def sanitize_project_group(source: str, path: str) -> str:
    parent = Path(path).parent.name
    if source == "claude":
        mapping = {
            "-home-sit-chitako-drable": "chitako-drable",
            "-home-sit-chitako-drive": "chitako-drive",
            "-home-sit--t3-projects-carla-ja": "carla-ja",
            "-home-sit": "home-sit",
        }
        return mapping.get(parent, f"claude-{hashlib.sha256(parent.encode()).hexdigest()[:6]}")
    if source == "pi":
        if "loopedslm" in parent:
            return "pi-loopedslm"
        if "home-sit" in parent:
            return "pi-home-sit"
        return f"pi-{hashlib.sha256(parent.encode()).hexdigest()[:6]}"
    if source == "omp":
        return f"omp-{parent.strip('-') or 'root'}"
    return source


def format_action_line(tool_name: str, tool_input: object) -> str:
    if isinstance(tool_input, dict):
        if "command" in tool_input:
            val = str(tool_input["command"]).replace("\n", " ")[:300]
            return f"{tool_name}: {val}"
        if "file_path" in tool_input:
            return f"{tool_name}: {tool_input['file_path']}"
        if "path" in tool_input:
            return f"{tool_name}: {tool_input['path']}"
        keys = ",".join(list(tool_input.keys())[:4])
        return f"{tool_name}: ({keys})"
    return f"{tool_name}: {str(tool_input)[:120]}"


def extract_claude_session(path: str) -> dict:
    user_messages: list[str] = []
    prior_actions: list[str] = []
    tool_counts: Counter[str] = Counter()
    checkpoint_lengths: list[int] = []
    total_records = 0
    assistant_turns = 0
    cwds: set[str] = set()

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            total_records += 1
            if rec.get("cwd"):
                cwds.add(str(rec["cwd"]))
            rtype = rec.get("type")
            if rtype == "user":
                if rec.get("isMeta"):
                    continue
                content = rec.get("message", {}).get("content")
                text_parts: list[str] = []
                if isinstance(content, str):
                    if content.strip():
                        text_parts.append(content.strip())
                elif isinstance(content, list):
                    has_tool_result = any(
                        isinstance(x, dict) and x.get("type") == "tool_result" for x in content
                    )
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            txt = str(item.get("text", "")).strip()
                            if txt:
                                text_parts.append(txt)
                    if has_tool_result and not text_parts:
                        continue
                joined = "\n".join(text_parts).strip()
                if joined and not joined.startswith("<local-command-"):
                    user_messages.append(joined)
            elif rtype == "assistant":
                assistant_turns += 1
                content = rec.get("message", {}).get("content") or []
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "tool_use":
                            tname = str(item.get("name", "unknown"))
                            tool_counts[tname] += 1
                            tinput = item.get("input", {})
                            if tname not in READONLY_TOOLS and tname not in META_OR_INTERACTIVE_TOOLS:
                                view_len = _estimate_view_chars(
                                    cwd=next(iter(cwds), "/workspace"),
                                    user_messages=user_messages,
                                    prior_actions=prior_actions,
                                    tool_name=tname,
                                    tool_input=tinput,
                                )
                                checkpoint_lengths.append(view_len)
                                prior_actions.append(format_action_line(tname, tinput))

    return _build_session_stats(
        source="claude",
        path=path,
        total_records=total_records,
        assistant_turns=assistant_turns,
        user_messages=user_messages,
        prior_actions=prior_actions,
        tool_counts=tool_counts,
        checkpoint_lengths=checkpoint_lengths,
    )


def extract_pi_or_omp_session(source: str, path: str) -> dict:
    user_messages: list[str] = []
    prior_actions: list[str] = []
    tool_counts: Counter[str] = Counter()
    checkpoint_lengths: list[int] = []
    total_records = 0
    assistant_turns = 0
    cwd = "/workspace"

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            total_records += 1
            if rec.get("type") == "session" and rec.get("cwd"):
                cwd = str(rec["cwd"])
            elif rec.get("type") == "message":
                msg = rec.get("message", {})
                role = msg.get("role")
                content = msg.get("content")
                if role == "user":
                    text_parts = []
                    if isinstance(content, str):
                        if content.strip():
                            text_parts.append(content.strip())
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                txt = str(item.get("text", "")).strip()
                                if txt:
                                    text_parts.append(txt)
                    joined = "\n".join(text_parts).strip()
                    if joined:
                        user_messages.append(joined)
                elif role == "assistant":
                    assistant_turns += 1
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "toolCall":
                                tname = str(item.get("name", "unknown"))
                                tool_counts[tname] += 1
                                tinput = item.get("arguments", {})
                                if tname not in READONLY_TOOLS and tname not in META_OR_INTERACTIVE_TOOLS:
                                    view_len = _estimate_view_chars(
                                        cwd=cwd,
                                        user_messages=user_messages,
                                        prior_actions=prior_actions,
                                        tool_name=tname,
                                        tool_input=tinput,
                                    )
                                    checkpoint_lengths.append(view_len)
                                    prior_actions.append(format_action_line(tname, tinput))

    return _build_session_stats(
        source=source,
        path=path,
        total_records=total_records,
        assistant_turns=assistant_turns,
        user_messages=user_messages,
        prior_actions=prior_actions,
        tool_counts=tool_counts,
        checkpoint_lengths=checkpoint_lengths,
    )


def _estimate_view_chars(
    cwd: str,
    user_messages: list[str],
    prior_actions: list[str],
    tool_name: str,
    tool_input: object,
) -> int:
    # Approximate reasoning-blind state size matching automode.view layout
    env_chars = 350 + len(cwd)
    user_chars = sum(min(len(m), 700) + 8 for m in user_messages)
    prior_chars = sum(min(len(a), 400) + 3 for a in prior_actions)
    if isinstance(tool_input, dict):
        action_chars = 20 + sum(min(len(str(v)), 1200) + len(str(k)) + 4 for k, v in tool_input.items())
    else:
        action_chars = 20 + min(len(str(tool_input)), 1200)
    return env_chars + user_chars + prior_chars + action_chars


def _build_session_stats(
    source: str,
    path: str,
    total_records: int,
    assistant_turns: int,
    user_messages: list[str],
    prior_actions: list[str],
    tool_counts: Counter[str],
    checkpoint_lengths: list[int],
) -> dict:
    digest = hashlib.sha256(path.encode("utf-8")).hexdigest()[:8]
    project_group = sanitize_project_group(source, path)
    readonly_count = sum(v for k, v in tool_counts.items() if k in READONLY_TOOLS)
    meta_count = sum(v for k, v in tool_counts.items() if k in META_OR_INTERACTIVE_TOOLS)
    actionable_count = sum(
        v for k, v in tool_counts.items() if k not in READONLY_TOOLS and k not in META_OR_INTERACTIVE_TOOLS
    )
    final_chars = checkpoint_lengths[-1] if checkpoint_lengths else _estimate_view_chars(
        "/workspace", user_messages, prior_actions, "None", {}
    )
    return {
        "anon_id": f"{source}_{digest}",
        "source": source,
        "project_group": project_group,
        "total_records": total_records,
        "assistant_turns": assistant_turns,
        "user_message_count": len(user_messages),
        "tool_calls_total": sum(tool_counts.values()),
        "tool_calls_readonly": readonly_count,
        "tool_calls_meta_interactive": meta_count,
        "tool_calls_non_readonly_actionable": actionable_count,
        "tool_counts_by_name": dict(sorted(tool_counts.items())),
        "final_view_chars": final_chars,
        "approx_final_view_tokens": round(final_chars / 3.0),
        "checkpoint_view_chars": (
            {
                "min": min(checkpoint_lengths),
                "median": round(statistics.median(checkpoint_lengths)),
                "max": max(checkpoint_lengths),
                "approx_max_tokens": round(max(checkpoint_lengths) / 3.0),
            }
            if checkpoint_lengths
            else None
        ),
    }


def main() -> None:
    sessions: list[dict] = []

    for path in sorted(glob.glob("/home/sit/.claude/projects/*/*.jsonl")):
        sessions.append(extract_claude_session(path))
    for path in sorted(glob.glob("/home/sit/.pi/agent/sessions/*/*.jsonl")):
        sessions.append(extract_pi_or_omp_session("pi", path))
    for path in sorted(glob.glob("/home/sit/.omp/agent/sessions/*/*.jsonl")):
        sessions.append(extract_pi_or_omp_session("omp", path))

    by_project: dict[str, dict] = defaultdict(
        lambda: {
            "session_count": 0,
            "user_messages": 0,
            "tool_calls_total": 0,
            "tool_calls_readonly": 0,
            "actionable_checkpoints": 0,
            "session_ids": [],
        }
    )
    for s in sessions:
        pg = by_project[s["project_group"]]
        pg["session_count"] += 1
        pg["user_messages"] += s["user_message_count"]
        pg["tool_calls_total"] += s["tool_calls_total"]
        pg["tool_calls_readonly"] += s["tool_calls_readonly"]
        pg["actionable_checkpoints"] += s["tool_calls_non_readonly_actionable"]
        pg["session_ids"].append(s["anon_id"])

    # Propose strict zero-leakage partition by project_group (no shared repo/project across splits)
    proposed_split_by_project = {
        "chitako-drable": "observe",
        "pi-loopedslm": "observe",
        "pi-home-sit": "observe",
        "chitako-drive": "select_dev",
        "omp-tmp": "select_dev",
        "carla-ja": "frozen_test",
        "home-sit": "frozen_test",
    }
    split_totals = Counter()
    split_sessions = Counter()
    for s in sessions:
        sp = proposed_split_by_project.get(s["project_group"], "select_dev")
        split_totals[sp] += s["tool_calls_non_readonly_actionable"]
        split_sessions[sp] += 1

    summary = {
        "scan_roots": [
            "/home/sit/.claude/projects/*/*.jsonl",
            "/home/sit/.pi/agent/sessions/*/*.jsonl",
            "/home/sit/.omp/agent/sessions/*/*.jsonl",
        ],
        "readonly_tools": sorted(READONLY_TOOLS),
        "meta_interactive_tools": sorted(META_OR_INTERACTIVE_TOOLS),
        "totals": {
            "sessions": len(sessions),
            "sessions_with_actionable_tools": sum(
                1 for s in sessions if s["tool_calls_non_readonly_actionable"] > 0
            ),
            "user_messages": sum(s["user_message_count"] for s in sessions),
            "tool_calls_total": sum(s["tool_calls_total"] for s in sessions),
            "tool_calls_readonly": sum(s["tool_calls_readonly"] for s in sessions),
            "actionable_checkpoints": sum(
                s["tool_calls_non_readonly_actionable"] for s in sessions
            ),
        },
        "by_project_group": dict(sorted(by_project.items())),
        "proposed_zero_leakage_partition": {
            "strategy": "strict project_group disjoint split (no project or session crosses splits)",
            "project_to_split": proposed_split_by_project,
            "actionable_checkpoints_per_split": dict(split_totals),
            "sessions_per_split": dict(split_sessions),
        },
        "sessions": sessions,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary["totals"], indent=2))
    print("by_project_group:")
    for k, v in summary["by_project_group"].items():
        print(f"  {k}: sessions={v['session_count']} users={v['user_messages']} checkpoints={v['actionable_checkpoints']}")
    print("proposed_split_checkpoints:", summary["proposed_zero_leakage_partition"]["actionable_checkpoints_per_split"])


if __name__ == "__main__":
    main()
