"""Generate `experiments/phase5_frozen_comparison.md` programmatically from frozen JSON results.

Preserves Phase 3's frozen reports (`experiments/frozen_test_comparison.md`, `experiments/frozen_track*.md`)
completely untouched while appending Track 3 Final (PyTorch `bf16` and `< 1 GiB` `Q8_0` GGUF) locked
`frozen_test` results side by side.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = ROOT / "experiments"


def _row_line(
    name: str,
    pct_mod: str,
    frob: str,
    metrics: dict[str, Any],
) -> str:
    px = metrics["proxy"]["proxy_score"]
    rc = metrics["real_cf"]
    sr = rc["second_reading"]
    c5 = metrics["consent_5p"]
    lg = metrics["long_50t"]
    lg_r = lg.get("with_recap_ablation", {})
    rp = metrics["decider_replay"]
    return (
        f"| `{name}` | `{pct_mod}` (`{frob}`) | `{px}` | "
        f"`{rc['auroc']}` (`{sr['auroc']}`) | `{rc['recall_at_05']} / {rc['fpr_at_05']}` | "
        f"`{rc['order_flip_rate']}` | `{c5['overall_acc']}` (`{c5['p2_allow_rate']} / {c5['p3_block_rate']}`) | "
        f"`{c5['p5a_allow_rate']} / {c5['p5b_block_rate']}` (`{c5['order_flip_rate']}`) | "
        f"`{lg['overall_acc']}` (`{lg_r.get('overall_acc', 'N/A')}`) | "
        f"`{rp['overall_acc']} / {rp['top_label_ece']}` |"
    )


def _detail_line(name: str, metrics: dict[str, Any]) -> str:
    rc = metrics["real_cf"]
    sr = rc["second_reading"]
    c5 = metrics["consent_5p"]
    c5_sr_acc = c5.get("second_reading", {}).get("overall_acc", "N/A")
    lg = metrics["long_50t"]
    rp = metrics["decider_replay"]
    return (
        f"- **`{name}`**: `real_cf` ECE=`{rc['ece']}`, MarginSep=`{rc['margin_separation']}`, "
        f"SR Recall/FPR=`{sr['recall_at_05']}/{sr['fpr_at_05']}`; "
        f"`consent_5p` P3-P2=`{c5['margin_sep_p3_minus_p2']}`, P5b-P5a=`{c5['margin_sep_p5b_minus_p5a']}`, "
        f"SR Acc=`{c5_sr_acc}`; "
        f"`long_50t` EarlyAllow=`{lg['early_consent_allow_rate']}`, MismatchBlock=`{lg['mismatch_block_rate']}`; "
        f"`decider_replay` by type=`{rp.get('acc_by_type', {})}`"
    )


def build_phase5_frozen_markdown(experiments_dir: Path = EXPERIMENTS_DIR) -> str:
    t0a = json.loads((experiments_dir / "frozen_track0a_zeroshot.json").read_text(encoding="utf-8"))
    t0c = json.loads((experiments_dir / "frozen_track0c_oracle_chat.json").read_text(encoding="utf-8"))
    t1 = json.loads((experiments_dir / "frozen_track1_best.json").read_text(encoding="utf-8"))
    t2 = json.loads((experiments_dir / "frozen_track2_best.json").read_text(encoding="utf-8"))
    t3_frozen = json.loads((experiments_dir / "phase5_track3_parity_frozen.json").read_text(encoding="utf-8"))
    t3_spec = json.loads((experiments_dir / "track3_final_patch_spec.json").read_text(encoding="utf-8"))

    t3_wd = t3_spec["extra_summary"]["weight_diff"]
    t3_pct = f"{t3_wd['pct_modified_params']}%"
    t3_frob = f"{t3_wd['total_delta_frobenius_norm']}"

    t3_pt = t3_frozen["pt_split_metrics"]
    t3_q8 = t3_frozen["q8_split_metrics"]
    parity = t3_frozen["parity"]

    entries: list[tuple[str, str, str, dict[str, Any]]] = [
        (
            t0a["name"],
            f"{t0a['weight_diff']['pct_modified_params']}%",
            f"{t0a['weight_diff']['total_delta_frobenius_norm']}",
            t0a,
        ),
        (
            t0c["name"],
            f"{t0c['weight_diff']['pct_modified_params']}%",
            f"{t0c['weight_diff']['total_delta_frobenius_norm']}",
            t0c,
        ),
        (
            t1["name"],
            f"{t1['weight_diff']['pct_modified_params']}%",
            f"{t1['weight_diff']['total_delta_frobenius_norm']}",
            t1,
        ),
        (
            t2["name"],
            f"{t2['weight_diff']['pct_modified_params']}%",
            f"{t2['weight_diff']['total_delta_frobenius_norm']}",
            t2,
        ),
        ("frozen_track3_final_bf16", t3_pct, t3_frob, t3_pt),
        ("frozen_track3_final_Q8_0", t3_pct, f"{parity['gguf_mib']} MiB", t3_q8),
    ]

    lines = [
        "# Phase 5 Locked `frozen_test` Comparison (`frozen_test`, n=200)",
        "",
        "> Executed ONCE ONLY under SHA-256 verification after Track 3 Final (`Soup lam=0.25`) weights and `< 1 GiB` `Q8_0` GGUF were completely finalized. Phase 3 baselines (`experiments/frozen_test_comparison.md`) remain unmodified.",
        "",
        "| System | `% modified` (`||ΔW||_F` / Size) | Proxy | `real_cf` AUROC (SR) | `real_cf` Recall / FPR | `real_cf` Flip | `consent_5p` Acc (`P2 / P3`) | `P5a / P5b` (`Flip`) | `long_50t` Acc (`recap ON`) | `decider_replay` Acc / ECE |",
        "|---|---:|---:|---|---|---|---|---|---|---|",
    ]
    for name, pct, frob, m in entries:
        lines.append(_row_line(name, pct, frob, m))

    lines.extend(["", "## Detailed `decider_replay` & `second_reading` Breakdown", ""])
    for name, _, _, m in entries:
        lines.append(_detail_line(name, m))

    lines.extend(
        [
            "",
            "## Phase 5 `frozen_test` Parity & Runtime Latency (`PyTorch bf16` vs `llama-server Q8_0`)",
            "",
            f"- **GGUF Artifact**: `{parity['gguf_name']}` (`{parity['gguf_mib']} MiB` / `{parity['gguf_bytes']}` bytes, `< 1 GiB = {parity['under_1gb']}`)",
            f"- **Tokenizer Parity (BOS-free)**: `exact_match = {parity['tokenizer_exact_match']}`",
            f"- **PyTorch `bf16` vs `Q8_0` Agreement (n=200)**: Total `{parity['agreement_total_200']}` | Binary Normal `{parity['agreement_binary_normal']}` | Second-Reading `{parity['agreement_binary_second_reading']}` | Replay `{parity['agreement_replay']}`",
            f"- **`p_block` Parity Error**: Normal MAE `{parity['p_block_mae_normal']}` (Max AE `{parity['p_block_max_ae_normal']}`, Pearson `r = {parity['p_block_pearson_r']}`) | SR MAE `{parity['p_block_mae_second_reading']}` | Margin MAE `{parity['margin_mae_normal']}`",
            f"- **Latency (`CUDA sm_86`, `-ngl all`)**: 1-Pass P50 `{parity['latency_1pass_p50_ms']} ms` (P95 `{parity['latency_1pass_p95_ms']} ms`) | E2E P50 `{parity['latency_e2e_p50_ms']} ms` (P95 `{parity['latency_e2e_p95_ms']} ms`)",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Phase 5 locked frozen_test markdown comparison report.")
    ap.add_argument("--experiments-dir", default=str(EXPERIMENTS_DIR))
    ap.add_argument("--output", default=str(EXPERIMENTS_DIR / "phase5_frozen_comparison.md"))
    args = ap.parse_args()

    exp_dir = Path(args.experiments_dir)
    md = build_phase5_frozen_markdown(exp_dir)
    out_path = Path(args.output)
    out_path.write_text(md, encoding="utf-8")
    print(f"[build_phase5_frozen_report] Wrote {out_path}")


if __name__ == "__main__":
    main()
