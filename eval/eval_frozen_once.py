"""Single locked evaluation of Tracks 0a, 0c, 1, and 2 on `frozen_test`.

Enforces TWO strict protections:
  1. Requires explicit `--unlock-frozen-test` CLI flag AND verifies SHA-256 lock via `load_split_rows`.
  2. Hard single-run guard: refuses to run if `experiments/frozen_test_comparison.md` already exists.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch

from eval.evaluate import evaluate_runner, format_markdown_report, load_split_rows
from eval.model_runner import (
    QwenDeciderRunner,
    capture_cpu_state_dict,
    compute_weight_diff_stats,
    restore_cpu_state_dict,
)
from weight_edit.causal_check import apply_bundle, load_patch_bundle

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = ROOT / "experiments"
CHECKPOINTS_DIR = ROOT / "checkpoints"


def format_comparison_markdown(results: list[dict[str, Any]]) -> str:
    """Render unified frozen_test markdown comparison across Tracks 0a, 0c, 1, and 2."""
    lines = [
        "# Frozen Test Single Evaluation Comparison (`frozen_test`, n=200)",
        "",
        "> Executed once under SHA-256 verification. Weights were finalized prior to unlocking and never re-tuned afterward.",
        "",
        "| System | `% modified` (`||ΔW||_F`) | Proxy | `real_cf` AUROC (SR) | `real_cf` Recall / FPR | `real_cf` Flip | `consent_5p` Acc (`P2 / P3`) | `P5a / P5b` (`Flip`) | `long_50t` Acc (`recap ON`) | `decider_replay` Acc / ECE |",
        "|---|---:|---:|---|---|---|---|---|---|---|",
    ]
    for res in results:
        wd = res["weight_diff"]
        px = res["proxy"]["proxy_score"]
        rc = res["real_cf"]
        sr = rc["second_reading"]
        c5 = res["consent_5p"]
        lg = res["long_50t"]
        lg_r = lg.get("with_recap_ablation", {})
        rp = res["decider_replay"]
        lines.append(
            f"| `{res['name']}` | `{wd['pct_modified_params']}%` (`{wd['total_delta_frobenius_norm']}`) | "
            f"`{px}` | `{rc['auroc']}` (`{sr['auroc']}`) | `{rc['recall_at_05']} / {rc['fpr_at_05']}` | "
            f"`{rc['order_flip_rate']}` | `{c5['overall_acc']}` (`{c5['p2_allow_rate']} / {c5['p3_block_rate']}`) | "
            f"`{c5['p5a_allow_rate']} / {c5['p5b_block_rate']}` (`{c5['order_flip_rate']}`) | "
            f"`{lg['overall_acc']}` (`{lg_r.get('overall_acc', 'N/A')}`) | "
            f"`{rp['overall_acc']} / {rp['top_label_ece']}` |"
        )

    lines.extend(["", "## Detailed `decider_replay` & `second_reading` Breakdown", ""])
    for res in results:
        rc = res["real_cf"]
        sr = rc["second_reading"]
        c5 = res["consent_5p"]
        lg = res["long_50t"]
        rp = res["decider_replay"]
        lines.append(
            f"- **`{res['name']}`**: `real_cf` ECE=`{rc['ece']}`, MarginSep=`{rc['margin_separation']}`, "
            f"SR Recall/FPR=`{sr['recall_at_05']}/{sr['fpr_at_05']}`; "
            f"`consent_5p` P3-P2=`{c5['margin_sep_p3_minus_p2']}`, P5b-P5a=`{c5['margin_sep_p5b_minus_p5a']}`, "
            f"SR Acc=`{c5['second_reading']['overall_acc']}`; "
            f"`long_50t` EarlyAllow=`{lg['early_consent_allow_rate']}`, MismatchBlock=`{lg['mismatch_block_rate']}`; "
            f"`decider_replay` by type=`{rp['acc_by_type']}`"
        )
    return "\n".join(lines) + "\n"


def run_frozen_eval_once(
    unlock_frozen_test: bool = False,
    track1_deltas_path: Path | None = None,
    track2_deltas_path: Path | None = None,
    experiments_dir: Path = EXPERIMENTS_DIR,
) -> dict[str, Any]:
    """Evaluate Track 0a, Track 0c, Track 1, and Track 2 on `frozen_test` exactly once."""
    rows = load_split_rows("frozen_test", unlock_frozen_test=unlock_frozen_test)
    comparison_md = experiments_dir / "frozen_test_comparison.md"
    comparison_json = experiments_dir / "frozen_test_comparison.json"
    if comparison_md.exists():
        raise FileExistsError("frozen_test has already been evaluated once; refusing to re-run.")

    t1_path = track1_deltas_path or (CHECKPOINTS_DIR / "track1_best_deltas.pt")
    t2_path = track2_deltas_path or (CHECKPOINTS_DIR / "track2_best_deltas.pt")
    if not t1_path.exists():
        raise FileNotFoundError(f"Track 1 checkpoint not found: {t1_path}")
    if not t2_path.exists():
        raise FileNotFoundError(f"Track 2 checkpoint not found: {t2_path}")

    runner = QwenDeciderRunner()
    base_cpu = capture_cpu_state_dict(runner.model)

    specs: list[tuple[str, str, Path | None]] = [
        ("track0a_zeroshot", "plain", None),
        ("track0c_oracle_chat", "oracle_chat", None),
        ("track1_best", "plain", t1_path),
        ("track2_best", "plain", t2_path),
    ]

    results: list[dict[str, Any]] = []
    experiments_dir.mkdir(parents=True, exist_ok=True)

    try:
        for sys_name, mode, delta_path in specs:
            if delta_path is None:
                restore_cpu_state_dict(runner.model, base_cpu)
                weight_diff = {
                    "total_params": sum(p.numel() for p in runner.model.parameters()),
                    "modified_params": 0,
                    "pct_modified_params": 0.0,
                    "modified_tensor_count": 0,
                    "total_delta_frobenius_norm": 0.0,
                    "modified_tensors": [],
                }
            else:
                bundle = load_patch_bundle(delta_path)
                apply_bundle(runner.model, base_cpu, bundle.patches)
                weight_diff = compute_weight_diff_stats(runner.model, base_cpu)

            report = evaluate_runner(
                runner,
                rows,
                mode=mode,
                include_recap=False,
                eval_recap_ablation=True,
            )
            run_id = f"frozen_{sys_name}"
            payload = {"name": run_id, "split": "frozen_test", "weight_diff": weight_diff, **report}
            (experiments_dir / f"{run_id}.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            (experiments_dir / f"{run_id}.md").write_text(
                format_markdown_report(run_id, "frozen_test", report, weight_diff), encoding="utf-8"
            )
            results.append(payload)
    finally:
        del runner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    comp_payload = {"split": "frozen_test", "n_rows": len(rows), "systems": results}
    comp_md_text = format_comparison_markdown(results)
    comparison_json.write_text(json.dumps(comp_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    comparison_md.write_text(comp_md_text, encoding="utf-8")
    print(comp_md_text)
    return comp_payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate finalized models ONCE on locked frozen_test.")
    ap.add_argument("--unlock-frozen-test", action="store_true")
    ap.add_argument("--track1-deltas", default=None)
    ap.add_argument("--track2-deltas", default=None)
    args = ap.parse_args()
    run_frozen_eval_once(
        unlock_frozen_test=args.unlock_frozen_test,
        track1_deltas_path=Path(args.track1_deltas) if args.track1_deltas else None,
        track2_deltas_path=Path(args.track2_deltas) if args.track2_deltas else None,
    )


if __name__ == "__main__":
    main()
