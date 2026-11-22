"""Reasoning-blind state serialization for single-model auto-mode permission decisions.

Compatible with Chiytako/decider-auto's plain append-only state view and extended with:
- Harness-trusted `Blocked action k: <tool>(<summary>) [reason: <reason>]` history entries
- Toggleable `User instructions recap` (`include_recap`) for independent Gated DeltaNet testing
- Clean single-newline prefix/suffix split so Qwen3.5 tokenization never merges across the boundary
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

ENV_PRESETS: dict[str, list[str]] = {
    "default": [
        "Organization: none configured",
        "Trusted repo: the repository the session started in (and its configured remotes)",
        "Repository visibility: assume private unless shown otherwise",
        "Trusted internal domains / buckets / services: none configured",
        "Internal sharing: none - public paste/gist services are outside the trust boundary",
        "Sensitive remote targets: any namespace/host/container named prod or production",
        "Protected IaC scopes: IAM, RBAC, networking, quota, node pools; anything tagged prod",
    ],
    "local_dev": [
        "Organization: personal research workstation",
        "Trusted repo: the local workspace git repository and origin remote",
        "Repository visibility: private local project unless explicitly published",
        "Sensitive paths: ~/.ssh, ~/.aws, ~/.config/gh, .env, credential files",
        "Protected branches: main, release/*",
        "Production targets: any remote host, cloud account, or systemd/system-wide service",
    ],
    "acme_saas": [
        "Organization: Acme (B2B SaaS)",
        "Cloud: AWS (accounts acme-dev, acme-staging, acme-prod)",
        "Source control: github.com/acme-corp/* repos are trusted",
        "Trusted internal domains: *.acme.internal, api.acme.dev",
        "Trusted buckets: s3://acme-dev-artifacts, s3://acme-ci-cache",
        "Internal package registry: https://npm.acme.internal, https://pypi.acme.internal",
        "CI/CD deploy targets: ArgoCD apps *-prod are production; *-staging is shared",
        "Protected branches: main, release/*",
    ],
    "fintech_k8s": [
        "Organization: Northwind Payments",
        "Cloud: GCP projects nw-sandbox (personal), nw-shared, nw-prod",
        "Kubernetes: cluster gke-shared (shared by all teams), gke-prod (production)",
        "Source control: gitlab.northwind.io/payments/* trusted",
        "Trusted domains: *.northwind.io",
        "Sensitive data: BigQuery datasets pii_*, customers table, any file under /data/exports",
        "Default branch: main (protected, requires 2 approvals)",
    ],
    "oss_solo": [
        "Organization: none (individual open-source maintainer)",
        "Trusted repo: github.com/jdoe/fastgrid (PUBLIC repository)",
        "Package publishing: PyPI project fastgrid owned by the user",
        "Trusted domains: none configured",
        "Primary use: software development on a personal laptop",
    ],
    "research_hpc": [
        "Organization: national lab HPC center",
        "Cluster: Slurm cluster 'aurora-shared' with shared /scratch and /projects filesystems",
        "Trusted repo: the session's repo; internal GitLab code.lab.gov",
        "Trusted domains: *.lab.gov",
        "Sensitive data: /projects/medical_* (patient data, IRB restricted)",
    ],
}

DEFAULT_BUDGET = 2048
LIMITS = {
    "command": 1000,
    "field": 500,
    "user": 650,
    "user_min": 260,
    "prior": 360,
    "action_min": 260,
}
LONG_FIELDS = (
    "command",
    "content",
    "prompt",
    "new_string",
    "old_string",
    "code",
    "script",
    "query",
    "body",
    "text",
)
KEEP_RECENT_PRIOR = 6


@dataclass
class Scenario:
    """Normalized reasoning-blind scenario representation."""

    env_lines: list[str]
    cwd: str | None
    git: dict[str, Any] | None
    user_messages: list[str]
    user_index0: int
    prior_actions: list[str]
    prior_hidden: int
    blocked_actions: list[str]
    tool: str
    tool_input: Any
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class View:
    """Rendered context text split into prefix and suffix on a clean token boundary."""

    prefix: str
    suffix: str
    n_tokens: int | None = None
    dropped_prior: int = 0
    dropped_users: int = 0
    over_budget: bool = False

    @property
    def text(self) -> str:
        return self.prefix + self.suffix


def _one_line(s: Any) -> str:
    return " ".join(str(s).split())


def _cut(s: Any, limit: int | None, head_frac: float = 2.0 / 3.0) -> str:
    txt = str(s)
    if limit is None or len(txt) <= limit:
        return txt
    h = max(1, int(limit * head_frac))
    t = max(1, limit - h)
    return f"{txt[:h]} …[{len(txt) - h - t} chars cut]… {txt[-t:]}"


def _format_blocked_entry(entry: Any, default_idx: int) -> str:
    if isinstance(entry, str):
        s = _one_line(entry)
        if s.startswith("Blocked action"):
            return s
        return f"Blocked action {default_idx}: {s}"
    if isinstance(entry, dict):
        idx = entry.get("index", default_idx)
        tool = entry.get("tool", entry.get("tool_name", "Action"))
        summary = _one_line(entry.get("summary", entry.get("command", entry.get("target", ""))))
        reason = _one_line(entry.get("reason", "requires explicit approval"))
        return f"Blocked action {idx}: {tool}({summary}) [reason: {reason}]"
    return f"Blocked action {default_idx}: {_one_line(entry)}"


def normalize_scenario(row: dict[str, Any] | Scenario, env: str | list[str] | None = None) -> Scenario:
    """Normalize dict row shapes (real or synthetic) into Scenario."""
    if isinstance(row, Scenario):
        if env is None:
            return row
        env_lines = list(env) if isinstance(env, (list, tuple)) else ENV_PRESETS.get(env, ENV_PRESETS["default"])
        return Scenario(
            env_lines=env_lines,
            cwd=row.cwd,
            git=row.git,
            user_messages=list(row.user_messages),
            user_index0=row.user_index0,
            prior_actions=list(row.prior_actions),
            prior_hidden=row.prior_hidden,
            blocked_actions=list(row.blocked_actions),
            tool=row.tool,
            tool_input=row.tool_input,
            extra=dict(row.extra),
        )

    if "action" in row and isinstance(row["action"], dict):
        tool = row["action"].get("tool", "Unknown")
        inp = row["action"].get("input", {})
    else:
        tool = row.get("tool_name") or row.get("tool") or "Unknown"
        inp = row.get("tool_input", row.get("input", {}))

    env_val = env if env is not None else row.get("env_lines") or row.get("env") or "default"
    if isinstance(env_val, (list, tuple)):
        env_lines = [str(x) for x in env_val]
    else:
        env_lines = list(ENV_PRESETS.get(str(env_val), ENV_PRESETS["default"]))

    um = [str(m) for m in (row.get("user_messages") or [])]
    n_total = int(row.get("n_user_messages_total") or len(um))
    idx0 = n_total - len(um) + 2 if len(um) >= 2 else 2

    pa = [str(a) for a in (row.get("prior_actions") or [])]
    hidden = max(0, int(row.get("n_prior_actions_total") or len(pa)) - len(pa))

    raw_blocked = row.get("blocked_actions") or []
    blocked = [_format_blocked_entry(b, i + 1) for i, b in enumerate(raw_blocked)]

    return Scenario(
        env_lines=env_lines,
        cwd=row.get("cwd"),
        git=row.get("git_context") or row.get("git") or None,
        user_messages=um,
        user_index0=idx0,
        prior_actions=pa,
        prior_hidden=hidden,
        blocked_actions=blocked,
        tool=str(tool),
        tool_input=inp,
        extra=dict(row.get("extra") or {}),
    )


def _value_lines(key: str, val: Any, limit: int) -> list[str]:
    if not isinstance(val, str):
        val = json.dumps(val, ensure_ascii=False, separators=(",", ":"))
    val = _cut(val.rstrip(), limit)
    if "\n" not in val:
        return [f"{key}: {val}"]
    return [f"{key}: |"] + ["  " + ln.rstrip() for ln in val.split("\n")]


def _render_action(tool: str, inp: Any, long_limit: int, field_limit: int) -> list[str]:
    lines = [f"Tool: {tool}"]
    if isinstance(inp, dict):
        for k, v in inp.items():
            if v is None or v == "":
                continue
            lines += _value_lines(str(k), v, long_limit if k in LONG_FIELDS else field_limit)
    elif inp not in (None, ""):
        lines += _value_lines("input", inp, long_limit)
    return lines


def _git_stable(g: dict[str, Any] | None) -> str | None:
    if not g:
        return None
    parts: list[str] = []
    if g.get("error") == "not_a_repo" or not g.get("root"):
        parts.append("not inside a git repository")
    else:
        b = g.get("branch") or "?"
        d = f" (default {g['default_branch']})" if g.get("default_branch") else ""
        parts.append(f"repo {g['root']}, branch {b}{d}")
    rem = g.get("remotes") or []
    if rem:
        parts.append("remotes: " + ", ".join(f"{r.get('remote')} ({r.get('visibility') or 'unknown'})" for r in rem))
    return "Git: " + "; ".join(parts)


def _git_volatile(g: dict[str, Any] | None, cwd: str | None) -> list[str]:
    out: list[str] = []
    if not g:
        return out
    if g.get("live_cwd") and cwd and g["live_cwd"] != cwd:
        out.append(f"Current directory now: {g['live_cwd']}")
    st = g.get("status")
    if isinstance(st, dict) and g.get("root"):
        if st.get("clean"):
            out.append("Git status now: clean")
        else:
            c = st.get("counts") or {}
            bits = [f"{c[k]} {k}" for k in ("staged", "modified") if c.get(k)]
            u = c.get("untracked_normal") or c.get("untracked")
            if u:
                bits.append(f"{u} untracked")
            out.append("Git status now: " + (", ".join(bits) or "dirty"))
    return out


def _count_tokens(tok: Any, text: str) -> int:
    if tok is None:
        return max(1, len(text) // 3)
    return len(tok.encode(text, add_special_tokens=False))


def _is_blocked_entry(entry: str) -> bool:
    return entry.lstrip().startswith("Blocked action")


def render_view(
    row: dict[str, Any] | Scenario,
    tok: Any = None,
    budget: int = DEFAULT_BUDGET,
    include_recap: bool = False,
    include_git: bool = True,
    env: str | list[str] | None = None,
    limits: dict[str, int] | None = None,
) -> View:
    """Render a reasoning-blind Scenario into View(prefix, suffix)."""
    cfg_lim = dict(LIMITS, **(limits or {}))
    sc = normalize_scenario(row, env=env)

    head = ["Environment:"] + [f"- {_one_line(x)}" for x in sc.env_lines]
    if sc.cwd:
        head.append(f"Working directory: {sc.cwd}")
    if include_git:
        g_line = _git_stable(sc.git)
        if g_line:
            head.append(g_line)
    head_txt = "\n".join(head)

    users: list[tuple[int, str]] = []
    for i, m in enumerate(sc.user_messages):
        num = 1 if i == 0 else sc.user_index0 + i - 1
        users.append((num, m.strip()))

    combined_prior = list(sc.prior_actions) + list(sc.blocked_actions)
    prior = [_cut(_one_line(a), cfg_lim["prior"]) for a in combined_prior]

    user_lim = cfg_lim["user"]
    long_lim = cfg_lim["command"]
    field_lim = cfg_lim["field"]
    vol = _git_volatile(sc.git, sc.cwd) if include_git else []

    def _user_line(n: int, msg: str, lim: int) -> str:
        msg_c = _cut(msg, lim)
        return f"[{n}] " + msg_c.replace("\n", "\n    ")

    def _format_prior_line(item: str) -> str:
        if _is_blocked_entry(item):
            return item
        return f"- {item}"

    def _build(
        users_kept: list[tuple[int, str]],
        prior_kept: list[str],
        prior_dropped: int,
        ulim: int,
        alim: int,
        flim: int,
    ) -> tuple[str, str]:
        u_lines = ["User messages (oldest first):"]
        if not users_kept:
            u_lines.append("(none)")
        for j, (n, m) in enumerate(users_kept):
            if j > 0:
                gap = n - users_kept[j - 1][0] - 1
                if gap > 0:
                    u_lines.append(f"[... {gap} earlier message{'s' if gap > 1 else ''} omitted ...]")
            first_or_last = j == 0 or j == len(users_kept) - 1
            u_lines.append(_user_line(n, m, ulim if first_or_last else min(ulim, cfg_lim["user"])))

        p_lines = ["Agent actions so far (oldest first; results are not shown):"]
        hidden = sc.prior_hidden + prior_dropped
        if hidden:
            p_lines.append(f"({hidden} earlier action{'s' if hidden > 1 else ''} not shown)")
        if prior_kept:
            p_lines.extend(_format_prior_line(a) for a in prior_kept)
        elif not hidden:
            p_lines.append("(none)")

        prefix = head_txt + "\n\n" + "\n".join(u_lines) + "\n\n" + "\n".join(p_lines) + "\n"

        suffix_parts: list[str] = []
        if include_recap and users:
            recap_lines = ["User instructions recap:", _user_line(users[0][0], users[0][1], cfg_lim["user_min"])]
            if len(users) > 1:
                recap_lines.append(_user_line(users[-1][0], users[-1][1], cfg_lim["user_min"]))
            suffix_parts.extend(recap_lines)
        suffix_parts.append("Proposed action:")
        suffix_parts.extend(vol)
        suffix_parts.extend(_render_action(sc.tool, sc.tool_input, alim, flim))
        suffix = "\n".join(suffix_parts)
        return prefix, suffix

    prefix, suffix = _build(users, prior, 0, user_lim, long_lim, field_lim)
    if tok is None:
        return View(prefix=prefix, suffix=suffix, n_tokens=None)

    total = _count_tokens(tok, prefix + suffix)
    users_kept = list(users)
    prior_kept = list(prior)
    dropped_p = 0
    dropped_u = 0
    over_budget = False

    def _pop_oldest_non_blocked() -> bool:
        for idx, item in enumerate(prior_kept):
            if not _is_blocked_entry(item):
                prior_kept.pop(idx)
                return True
        return False

    while total > budget:
        non_blocked_count = sum(1 for item in prior_kept if not _is_blocked_entry(item))
        if non_blocked_count > KEEP_RECENT_PRIOR and _pop_oldest_non_blocked():
            dropped_p += 1
        elif len(users_kept) > 2:
            users_kept.pop(1)
            dropped_u += 1
        elif non_blocked_count > 0 and _pop_oldest_non_blocked():
            dropped_p += 1
        elif user_lim > cfg_lim["user_min"]:
            user_lim = cfg_lim["user_min"]
        elif long_lim > cfg_lim["action_min"] or field_lim > cfg_lim["action_min"]:
            long_lim = max(cfg_lim["action_min"], long_lim // 2)
            field_lim = max(cfg_lim["action_min"], field_lim // 2)
        elif prior_kept:
            prior_kept.pop(0)
            dropped_p += 1
        else:
            over_budget = True
            break
        prefix, suffix = _build(users_kept, prior_kept, dropped_p, user_lim, long_lim, field_lim)
        total = _count_tokens(tok, prefix + suffix)

    return View(
        prefix=prefix,
        suffix=suffix,
        n_tokens=total,
        dropped_prior=dropped_p,
        dropped_users=dropped_u,
        over_budget=over_budget,
    )
