"""Train Track 3+ models on Qwen3.5-0.8B via exact scalar-chain-rule family LoRA.

Fits comfortably inside 3.77 GB VRAM by:
  1. Using activation checkpointing (`use_reentrant=False`) across all 24 layers.
  2. Computing lm_head logits ONLY on the tiny option-letter slice (`opt_token_ids`).
  3. Decoupling family losses (`compute_binary_family_loss` / `compute_replay_loss`) from
     per-sequence activations via exact scalar-chain-rule gradients (zero approximation, 1 seq in VRAM).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from automode.questions import Q_BLOCK_BINARY, build_decider_prompt, build_plain_prompt
from automode.view import render_view
from eval.evaluate import evaluate_runner, format_markdown_report, load_split_rows
from eval.model_runner import QwenDeciderRunner, capture_cpu_state_dict
from train.lora_model import (
    compute_live_weight_diff_stats,
    extract_merged_patch_bundle,
    inject_lora,
    lora_param_groups,
    save_lora_adapter,
)
from train.losses import compute_binary_family_loss, compute_replay_loss
from weight_edit.causal_check import apply_bundle, load_patch_bundle, save_patch_bundle

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN_FILE = ROOT / "data" / "local_private" / "phase5_train.jsonl"
CHECKPOINTS_DIR = ROOT / "checkpoints"
EXPERIMENTS_DIR = ROOT / "experiments"

PAIR_PATTERNS_C5 = [
    ("p1_no_consent", "p2_exact_consent"),
    ("p3_target_mismatch", "p2_exact_consent"),
    ("p4_vague_consent", "p2_exact_consent"),
    ("p5b_blocked_swapped_target", "p5a_blocked_approved_same"),
]


@dataclass
class BinaryItemEncoded:
    row_id: str
    pattern: str
    label: int
    ids_norm: torch.Tensor
    ids_rev: torch.Tensor
    yes_idx_norm: int
    no_idx_norm: int
    yes_idx_rev: int
    no_idx_rev: int


@dataclass
class BinaryFamilyGroup:
    group_id: str
    subset: str
    weight: float
    is_synthetic: bool
    items: list[BinaryItemEncoded]
    pair_indices: list[tuple[int, int]]


@dataclass
class ReplayItemGroup:
    group_id: str
    subset: str
    weight: float
    ids_norm: torch.Tensor
    ids_rev: torch.Tensor
    opt_ids_norm: list[int]
    opt_ids_rev: list[int]
    gold_idx: int
    base_probs_norm: torch.Tensor | None
    base_probs_rev: torch.Tensor | None
    lambda_ord: float
    lambda_kl: float


def _forward_slot_opt_logits(model: torch.nn.Module, input_ids: torch.Tensor, opt_token_ids: list[int]) -> torch.Tensor:
    """Run backbone + slot-sliced lm_head for exact option-letter logits with minimal VRAM."""
    hidden = model.model(input_ids=input_ids, use_cache=False).last_hidden_state
    slot_h = hidden[:, -1:, :]
    lm_w = model.lm_head.weight[opt_token_ids]
    lm_b = model.lm_head.bias[opt_token_ids] if getattr(model.lm_head, "bias", None) is not None else None
    logits = F.linear(slot_h.to(dtype=lm_w.dtype), lm_w, lm_b)[0, 0, :].float()
    return logits


def _forward_binary_margin(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    letter_ids: list[int],
    yes_idx: int,
    no_idx: int,
) -> torch.Tensor:
    opt_logits = _forward_slot_opt_logits(model, input_ids, letter_ids)
    return opt_logits[yes_idx] - opt_logits[no_idx]


def prepare_training_groups(
    runner: QwenDeciderRunner,
    train_path: Path,
) -> list[BinaryFamilyGroup | ReplayItemGroup]:
    """Load, render, tokenize, and cache stock replay posteriors BEFORE modifying any weights."""
    rows = [json.loads(ln) for ln in train_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    groups: list[BinaryFamilyGroup | ReplayItemGroup] = []

    binary_buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    replay_rows: list[dict[str, Any]] = []
    for r in rows:
        sub = r["subset"]
        if sub in ("real_cf", "real_cf_hard", "consent_5p", "long_50t"):
            fam = str(r["family"])
            binary_buckets.setdefault((sub, fam), []).append(r)
        elif sub in ("decider_replay", "reason_7class"):
            replay_rows.append(r)
        else:
            raise ValueError(f"Unknown subset {sub!r} in {train_path}")

    tok = runner.tokenizer
    ab_ids = [runner.letter_token_ids["A"], runner.letter_token_ids["B"]]

    for (sub, fam), fam_rows in binary_buckets.items():
        budget = 2400 if sub == "long_50t" else 1152
        encoded_items: list[BinaryItemEncoded] = []
        pat_to_idx: dict[str, int] = {}
        for r in fam_rows:
            ctx = render_view(r, tok=tok, budget=budget, include_recap=False).text
            pb_n = build_plain_prompt(ctx, question_spec=Q_BLOCK_BINARY, reverse_options=False)
            pb_r = build_plain_prompt(ctx, question_spec=Q_BLOCK_BINARY, reverse_options=True)
            ids_n = torch.tensor([tok.encode(pb_n.text, add_special_tokens=False)], dtype=torch.long)
            ids_r = torch.tensor([tok.encode(pb_r.text, add_special_tokens=False)], dtype=torch.long)
            if ids_n.shape[1] > 2600 or ids_r.shape[1] > 2600:
                ids_n = ids_n[:, -2600:]
                ids_r = ids_r[:, -2600:]
            idx_in_fam = len(encoded_items)
            pat_to_idx[r["pattern"]] = idx_in_fam
            encoded_items.append(
                BinaryItemEncoded(
                    row_id=r["id"],
                    pattern=r["pattern"],
                    label=1 if r["label"] == "block" else 0,
                    ids_norm=ids_n,
                    ids_rev=ids_r,
                    yes_idx_norm=int(pb_n.yes_idx),
                    no_idx_norm=int(pb_n.no_idx),
                    yes_idx_rev=int(pb_r.yes_idx),
                    no_idx_rev=int(pb_r.no_idx),
                )
            )

        pair_indices: list[tuple[int, int]] = []
        if sub == "consent_5p":
            for pos_p, neg_p in PAIR_PATTERNS_C5:
                if pos_p in pat_to_idx and neg_p in pat_to_idx:
                    pair_indices.append((pat_to_idx[pos_p], pat_to_idx[neg_p]))
            weight = 2.0
            is_synth = True
        else:
            blk_idxs = [i for i, it in enumerate(encoded_items) if it.label == 1]
            alw_idxs = [i for i, it in enumerate(encoded_items) if it.label == 0]
            for bi in blk_idxs:
                for ai in alw_idxs:
                    pair_indices.append((bi, ai))
            weight = 1.5 if sub == "long_50t" else (1.2 if sub == "real_cf_hard" else 1.0)
            is_synth = sub != "real_cf"

        groups.append(
            BinaryFamilyGroup(
                group_id=f"{sub}:{fam}",
                subset=sub,
                weight=weight,
                is_synthetic=is_synth,
                items=encoded_items,
                pair_indices=pair_indices,
            )
        )

    runner.model.eval()
    with torch.inference_mode():
        for r in replay_rows:
            sub = r["subset"]
            spec = r["question_spec"]
            pb_n = build_decider_prompt(r["context"], spec=spec, reverse_options=False)
            pb_r = build_decider_prompt(r["context"], spec=spec, reverse_options=True)
            ids_n = torch.tensor([tok.encode(pb_n.text, add_special_tokens=False)], dtype=torch.long)
            ids_r = torch.tensor([tok.encode(pb_r.text, add_special_tokens=False)], dtype=torch.long)
            opt_n = [runner.letter_token_ids[l] for l in pb_n.option_letters]
            opt_r = [runner.letter_token_ids[l] for l in pb_r.option_letters]

            if sub == "decider_replay":
                ln = _forward_slot_opt_logits(runner.model, ids_n.to(runner.device), opt_n)
                lr = _forward_slot_opt_logits(runner.model, ids_r.to(runner.device), opt_r)
                bp_n = torch.softmax(ln, dim=-1).cpu()
                bp_r = torch.softmax(lr, dim=-1).cpu()
                w = 0.65
                lam_kl = 0.10
            else:
                bp_n = None
                bp_r = None
                w = 0.50
                lam_kl = 0.0

            groups.append(
                ReplayItemGroup(
                    group_id=f"{sub}:{r['id']}",
                    subset=sub,
                    weight=w,
                    ids_norm=ids_n,
                    ids_rev=ids_r,
                    opt_ids_norm=opt_n,
                    opt_ids_rev=opt_r,
                    gold_idx=int(r["gold_index"]),
                    base_probs_norm=bp_n,
                    base_probs_rev=bp_r,
                    lambda_ord=0.15,
                    lambda_kl=lam_kl,
                )
            )
    return groups


def train_group_step(
    model: torch.nn.Module,
    device: str,
    letter_ids_ab: list[int],
    group: BinaryFamilyGroup | ReplayItemGroup,
    grad_accum: int,
) -> dict[str, float]:
    """Execute one family/item gradient accumulation step via exact scalar-chain-rule decoupling."""
    scale_factor = float(group.weight) / float(max(1, grad_accum))

    if isinstance(group, BinaryFamilyGroup):
        m_norm_vals: list[float] = []
        m_rev_vals: list[float] = []
        with torch.no_grad():
            for it in group.items:
                mn = _forward_binary_margin(model, it.ids_norm.to(device), letter_ids_ab, it.yes_idx_norm, it.no_idx_norm)
                mr = _forward_binary_margin(model, it.ids_rev.to(device), letter_ids_ab, it.yes_idx_rev, it.no_idx_rev)
                m_norm_vals.append(float(mn.item()))
                m_rev_vals.append(float(mr.item()))

        t_mn = torch.tensor(m_norm_vals, dtype=torch.float32, requires_grad=True)
        t_mr = torch.tensor(m_rev_vals, dtype=torch.float32, requires_grad=True)
        t_lbl = torch.tensor([it.label for it in group.items], dtype=torch.float32)

        loss, stats = compute_binary_family_loss(
            t_mn,
            t_mr,
            t_lbl,
            pair_indices=group.pair_indices,
            is_synthetic=group.is_synthetic,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss in {group.group_id}: {loss.item()}")
        (loss * scale_factor).backward()
        assert t_mn.grad is not None and t_mr.grad is not None
        g_norm = [float(x) for x in t_mn.grad.tolist()]
        g_rev = [float(x) for x in t_mr.grad.tolist()]

        for idx, it in enumerate(group.items):
            gn = g_norm[idx]
            if gn != 0.0:
                m_k = _forward_binary_margin(model, it.ids_norm.to(device), letter_ids_ab, it.yes_idx_norm, it.no_idx_norm)
                (m_k * gn).backward()
            gr = g_rev[idx]
            if gr != 0.0:
                m_r = _forward_binary_margin(model, it.ids_rev.to(device), letter_ids_ab, it.yes_idx_rev, it.no_idx_rev)
                (m_r * gr).backward()
        return stats

    # ReplayItemGroup
    with torch.no_grad():
        ln_val = _forward_slot_opt_logits(model, group.ids_norm.to(device), group.opt_ids_norm).cpu()
        lr_val = _forward_slot_opt_logits(model, group.ids_rev.to(device), group.opt_ids_rev).cpu()

    t_ln = ln_val.clone().detach().requires_grad_(True)
    t_lr = lr_val.clone().detach().requires_grad_(True)
    loss, stats = compute_replay_loss(
        t_ln,
        t_lr,
        gold_idx=group.gold_idx,
        base_probs_norm=group.base_probs_norm,
        base_probs_rev=group.base_probs_rev,
        lambda_ord=group.lambda_ord,
        lambda_kl=group.lambda_kl,
    )
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss in {group.group_id}: {loss.item()}")
    (loss * scale_factor).backward()
    assert t_ln.grad is not None and t_lr.grad is not None
    gn_vec = t_ln.grad.to(device=device)
    gr_vec = t_lr.grad.to(device=device)

    ln_g = _forward_slot_opt_logits(model, group.ids_norm.to(device), group.opt_ids_norm)
    torch.sum(ln_g * gn_vec).backward()
    lr_g = _forward_slot_opt_logits(model, group.ids_rev.to(device), group.opt_ids_rev)
    torch.sum(lr_g * gr_vec).backward()
    return stats


def compute_track3_select_score(report: dict[str, Any]) -> float:
    return round(
        float(report["proxy"]["proxy_score"])
        + 0.5 * float(report["long_50t"]["overall_acc"])
        + 0.5 * float(report["consent_5p"]["overall_acc"])
        + 0.5 * float(report["real_cf"]["auroc"])
        + 0.5 * float(report["decider_replay"]["overall_acc"]),
        5,
    )


def _cosine_lr_multiplier(step: int, total_steps: int, warmup_steps: int, min_ratio: float = 0.1) -> float:
    if total_steps <= 1:
        return 1.0
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    prog = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    prog = min(1.0, max(0.0, prog))
    return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * prog))


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    t_start = time.perf_counter()
    print(f"[Track 3] Loading runner for {args.run_name} ...")
    runner = QwenDeciderRunner()
    base_cpu_state = capture_cpu_state_dict(runner.model)

    groups = prepare_training_groups(runner, Path(args.train_file))
    if args.max_groups is not None:
        groups = groups[: args.max_groups]
    print(f"[Track 3] Prepared {len(groups)} training groups from {args.train_file}")

    if args.init_deltas:
        print(f"[Track 3] Applying warm-start deltas: {args.init_deltas}")
        init_bundle = load_patch_bundle(Path(args.init_deltas))
        apply_bundle(runner.model, base_cpu_state, init_bundle.patches)

    lora_summary = inject_lora(
        runner.model,
        r=args.r,
        alpha=args.alpha,
        train_deltanet_gates=True,
    )
    print(f"[Track 3] Injected LoRA + DeltaNet gates: {json.dumps(lora_summary, ensure_ascii=False)}")

    lora_params, gate_params = lora_param_groups(runner.model)
    all_trainable = lora_params + gate_params
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": gate_params, "lr": args.gate_lr, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.95),
    )

    steps_per_epoch = math.ceil(len(groups) / max(1, args.grad_accum))
    total_opt_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = max(1, int(0.08 * total_opt_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: _cosine_lr_multiplier(s, total_opt_steps, warmup_steps, min_ratio=0.1),
    )

    dev_rows = load_split_rows("select_dev") if not args.skip_eval else []
    ab_ids = [runner.letter_token_ids["A"], runner.letter_token_ids["B"]]

    train_log_path = EXPERIMENTS_DIR / f"{args.run_name}_train_log.jsonl"
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    if train_log_path.exists():
        train_log_path.unlink()

    best_select_score = -1e9
    best_epoch = -1
    epoch_summaries: list[dict[str, Any]] = []
    rng = random.Random(args.seed)
    opt_step = 0

    for ep in range(1, args.epochs + 1):
        runner.model.train()
        runner.model.enable_input_require_grads()
        runner.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        order = list(range(len(groups)))
        rng.shuffle(order)
        optimizer.zero_grad(set_to_none=True)

        running_total: list[float] = []
        by_subset_loss: dict[str, list[float]] = {}
        ep_t0 = time.perf_counter()

        for step_idx, g_i in enumerate(order):
            grp = groups[g_i]
            stats = train_group_step(runner.model, runner.device, ab_ids, grp, args.grad_accum)
            running_total.append(stats["total"])
            by_subset_loss.setdefault(grp.subset, []).append(stats["total"])

            is_step_boundary = ((step_idx + 1) % args.grad_accum == 0) or (step_idx == len(order) - 1)
            if is_step_boundary:
                grad_norm = torch.nn.utils.clip_grad_norm_(all_trainable, 1.0)
                if not torch.isfinite(grad_norm):
                    raise RuntimeError(f"Non-finite gradient norm at epoch {ep} step {step_idx + 1}: {grad_norm}")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1

            if (step_idx + 1) % 20 == 0 or step_idx == len(order) - 1:
                peak_mb = round(torch.cuda.max_memory_allocated() / 1048576.0, 1) if torch.cuda.is_available() else 0.0
                mean_loss = round(sum(running_total[-20:]) / len(running_total[-20:]), 4)
                sub_means = {k: round(sum(v[-20:]) / len(v[-20:]), 4) for k, v in sorted(by_subset_loss.items())}
                log_rec = {
                    "epoch": ep,
                    "group_progress": f"{step_idx + 1}/{len(order)}",
                    "opt_step": opt_step,
                    "lr_lora": round(optimizer.param_groups[0]["lr"], 8),
                    "lr_gate": round(optimizer.param_groups[1]["lr"], 8),
                    "recent_loss": mean_loss,
                    "subset_recent_loss": sub_means,
                    "peak_vram_mb": peak_mb,
                    "elapsed_sec": round(time.perf_counter() - t_start, 1),
                }
                with open(train_log_path, "a", encoding="utf-8") as lf:
                    lf.write(json.dumps(log_rec, ensure_ascii=False) + "\n")
                print(f"[Track 3 ep={ep}] {step_idx + 1}/{len(order)} loss={mean_loss} vram={peak_mb}MB lr={log_rec['lr_lora']}", flush=True)

        ep_train_sec = round(time.perf_counter() - ep_t0, 1)
        peak_vram = round(torch.cuda.max_memory_allocated() / 1048576.0, 1) if torch.cuda.is_available() else 0.0

        if args.skip_eval:
            ep_stat = {
                "epoch": ep,
                "train_sec": ep_train_sec,
                "peak_vram_mb": peak_vram,
                "mean_epoch_loss": round(sum(running_total) / max(1, len(running_total)), 4),
            }
            epoch_summaries.append(ep_stat)
            continue

        # End-of-epoch clean evaluation on select_dev
        runner.model.eval()
        runner.model.gradient_checkpointing_disable()
        torch.cuda.empty_cache()
        report = evaluate_runner(runner, dev_rows, mode="plain", include_recap=False, eval_recap_ablation=True)
        weight_diff = compute_live_weight_diff_stats(runner.model, base_cpu_state)
        select_score = compute_track3_select_score(report)

        ep_name = f"{args.run_name}_ep{ep}_dev"
        ep_payload = {
            "name": ep_name,
            "split": "select_dev",
            "epoch": ep,
            "train_sec": ep_train_sec,
            "peak_vram_mb": peak_vram,
            "track3_select_score": select_score,
            "weight_diff": weight_diff,
            **report,
        }
        (EXPERIMENTS_DIR / f"{ep_name}.json").write_text(json.dumps(ep_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        ep_md = format_markdown_report(ep_name, "select_dev", report, weight_diff)
        ep_md = ep_md.replace("- **Proxy Score**:", f"- **Track 3 Select Score**: `{select_score}` | **Peak VRAM**: `{peak_vram} MB` | **Train Sec**: `{ep_train_sec}s`\n- **Proxy Score**:")
        (EXPERIMENTS_DIR / f"{ep_name}.md").write_text(ep_md, encoding="utf-8")
        print(ep_md, flush=True)

        epoch_summaries.append(
            {
                "epoch": ep,
                "train_sec": ep_train_sec,
                "peak_vram_mb": peak_vram,
                "track3_select_score": select_score,
                "proxy_score": report["proxy"]["proxy_score"],
                "real_cf_auroc": report["real_cf"]["auroc"],
                "consent_5p_acc": report["consent_5p"]["overall_acc"],
                "long_50t_acc": report["long_50t"]["overall_acc"],
                "decider_replay_acc": report["decider_replay"]["overall_acc"],
            }
        )

        if select_score > best_select_score:
            best_select_score = select_score
            best_epoch = ep
            adapter_path = CHECKPOINTS_DIR / f"{args.run_name}_best_adapter.pt"
            deltas_path = CHECKPOINTS_DIR / f"{args.run_name}_best_deltas.pt"
            spec_path = EXPERIMENTS_DIR / f"{args.run_name}_patch_spec.json"

            save_lora_adapter(
                runner.model,
                adapter_path,
                metadata={
                    "run_name": args.run_name,
                    "epoch": ep,
                    "init_deltas": args.init_deltas,
                    "r": args.r,
                    "alpha": args.alpha,
                    "track3_select_score": select_score,
                },
            )
            bundle = extract_merged_patch_bundle(
                runner.model,
                base_cpu_state,
                step_id=f"{args.run_name}_ep{ep}",
                metadata={
                    "run_name": args.run_name,
                    "epoch": ep,
                    "init_deltas": args.init_deltas,
                    "r": args.r,
                    "alpha": args.alpha,
                    "hypothesis": f"Order-equivariant contrastive LoRA + DeltaNet gates ({args.run_name}, ep{ep}).",
                },
            )
            save_patch_bundle(
                bundle,
                spec_path,
                deltas_path,
                extra_summary={"weight_diff": weight_diff, "track3_select_score": select_score},
            )

            best_payload = {**ep_payload, "name": f"{args.run_name}_best_dev", "best_epoch": ep}
            (EXPERIMENTS_DIR / f"{args.run_name}_best_dev.json").write_text(
                json.dumps(best_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            (EXPERIMENTS_DIR / f"{args.run_name}_best_dev.md").write_text(
                ep_md.replace(f"`{ep_name}`", f"`{args.run_name}_best_dev`"), encoding="utf-8"
            )
            print(f"[Track 3] New best epoch={ep} saved -> {deltas_path} (select_score={best_select_score})", flush=True)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.skip_eval:
        save_lora_adapter(runner.model, CHECKPOINTS_DIR / f"{args.run_name}_smoke_adapter.pt", metadata={"smoke": True})
        bundle = extract_merged_patch_bundle(runner.model, base_cpu_state, step_id=f"{args.run_name}_smoke")
        save_patch_bundle(bundle, EXPERIMENTS_DIR / f"{args.run_name}_patch_spec.json", CHECKPOINTS_DIR / f"{args.run_name}_smoke_deltas.pt")

    summary = {
        "run_name": args.run_name,
        "best_epoch": best_epoch,
        "best_select_score": best_select_score,
        "total_elapsed_sec": round(time.perf_counter() - t_start, 1),
        "epochs": epoch_summaries,
    }
    print("[Track 3] Done:", json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Phase 5 Track 3+ Order-Equivariant Contrastive LoRA model.")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--init-deltas", default=None)
    ap.add_argument("--train-file", default=str(DEFAULT_TRAIN_FILE))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--gate-lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.02)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261124)
    ap.add_argument("--max-groups", type=int, default=None)
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
