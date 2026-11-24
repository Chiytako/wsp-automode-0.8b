"""Revision Suite R4: Direct In-Situ Warm-Start Causality & Intervention Controls.

Measures live `h_15` cosine drift, live in-situ L15 & L22 SwiGLU gate activations (captured directly
at `mlp.down_proj` pre-hook inside the training model — no proxy), live probe margins, and
step-by-step training loss across the first 160 groups (opt_step 0..20) of Track 3b Warm-Start:
  - R4a (`t3b_insitu_full`): Track 3b full LR (`1.5e-4` / `5e-4`), all 24 layers + DeltaNet gates.
  - R4b (`t3b_lowlr_5x`): Warm-start with 5x lower LR (`3e-5` / `1e-4`), all 24 layers.
  - R4c (`t3b_freeze_L0_14`): Warm-start at FULL LR (`1.5e-4` / `5e-4`) with Layers 0..14 frozen
    (`h_15` locked to base/Track-2 manifold; tests if upstream rotation is necessary for the spike).
  - R4d (`t3b_freeze_t2gates`): Warm-start at FULL LR (`1.5e-4` / `5e-4`) with ONLY Track 2's
    L15 & L22 MLP (`gate/up/down`) frozen while L0..14 & rest train (preventing gate self-attenuation).

Merges results cleanly into `experiments/paper_revision_controls.json` under `"warmstart_causality"`.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn.functional as F

from automode.questions import Q_BLOCK_BINARY, build_plain_prompt
from automode.view import render_view
from eval.evaluate import evaluate_runner, load_split_rows
from eval.mechanism_analysis import T2_L15_NEURONS, T2_L22_NEURONS
from eval.model_runner import QwenDeciderRunner, capture_cpu_state_dict
from eval.revision_quant_mitigations import CONTROLS_JSON
from train.lora_model import inject_lora, lora_param_groups
from train.train_track3 import (
    DEFAULT_TRAIN_FILE,
    _cosine_lr_multiplier,
    prepare_training_groups,
    train_group_step,
)
from weight_edit.causal_check import apply_bundle, load_patch_bundle

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = ROOT / "checkpoints"
EXPERIMENTS_DIR = ROOT / "experiments"

PROBE_OPT_STEPS = (0, 2, 5, 7, 10, 12, 15, 20)
MAX_GROUPS = 160  # opt_step 0..20 exact spike window


def _prepare_probe_prompts(runner: QwenDeciderRunner, dev_rows: list[dict[str, Any]]) -> list[torch.Tensor]:
    rc_rows = [r for r in dev_rows if r["subset"] == "real_cf"][:40]
    c5_rows = [r for r in dev_rows if r["subset"] == "consent_5p"][:24]
    probe_rows = rc_rows + c5_rows
    tok = runner.tokenizer
    out: list[torch.Tensor] = []
    for r in probe_rows:
        pr = build_plain_prompt(render_view(r, tok=tok, budget=2048, include_recap=False).text, Q_BLOCK_BINARY, False).text
        ids = tok.encode(pr, add_special_tokens=False)
        out.append(torch.tensor([ids], dtype=torch.long))
    return out


@torch.inference_mode()
def _probe_live_state(
    runner: QwenDeciderRunner,
    probe_ids: list[torch.Tensor],
    ab_ids: list[int],
    h15_base: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    runner.model.eval()
    h15_buf: list[torch.Tensor] = []
    l15_act_buf: list[torch.Tensor] = []
    l22_act_buf: list[torch.Tensor] = []
    margins: list[float] = []

    handles = [
        runner.model.model.layers[15].mlp.register_forward_pre_hook(
            lambda _m, inp: h15_buf.append(inp[0][0, -1, :].detach().float().cpu())
        ),
        runner.model.model.layers[15].mlp.down_proj.register_forward_pre_hook(
            lambda _m, inp: l15_act_buf.append(inp[0][0, -1, T2_L15_NEURONS].detach().float().cpu())
        ),
        runner.model.model.layers[22].mlp.down_proj.register_forward_pre_hook(
            lambda _m, inp: l22_act_buf.append(inp[0][0, -1, T2_L22_NEURONS].detach().float().cpu())
        ),
    ]
    lm_w = runner.model.lm_head.weight[ab_ids]
    lm_b = runner.model.lm_head.bias[ab_ids] if getattr(runner.model.lm_head, "bias", None) is not None else None
    try:
        for ids in probe_ids:
            hid = runner.model.model(input_ids=ids.to(runner.device), use_cache=False).last_hidden_state[:, -1:, :]
            lg = F.linear(hid.to(dtype=lm_w.dtype), lm_w, lm_b)[0, 0, :].float()
            margins.append(float((lg[1] - lg[0]).item()))
    finally:
        for h in handles:
            h.remove()

    h15_mat = torch.stack(h15_buf, dim=0)
    a15_mat = torch.stack(l15_act_buf, dim=0).abs()
    a22_mat = torch.stack(l22_act_buf, dim=0).abs()
    m_arr = torch.tensor(margins, dtype=torch.float32)

    stats: dict[str, Any] = {
        "l15_live_swiglu_mean_abs": round(float(a15_mat.mean().item()), 4),
        "l15_live_swiglu_p95_abs": round(float(torch.quantile(a15_mat.reshape(-1), 0.95).item()), 4),
        "l15_live_swiglu_max_abs": round(float(a15_mat.max().item()), 4),
        "l22_live_swiglu_mean_abs": round(float(a22_mat.mean().item()), 4),
        "l22_live_swiglu_max_abs": round(float(a22_mat.max().item()), 4),
        "probe_mean_abs_margin": round(float(m_arr.abs().mean().item()), 4),
        "probe_max_abs_margin": round(float(m_arr.abs().max().item()), 4),
    }
    if h15_base is not None:
        cos = F.cosine_similarity(h15_mat, h15_base, dim=-1)
        rel_l2 = torch.linalg.norm(h15_mat - h15_base, dim=-1) / torch.linalg.norm(h15_base, dim=-1).clamp_min(1e-8)
        stats["h15_cos_vs_base_mean"] = round(float(cos.mean().item()), 6)
        stats["h15_cos_vs_base_min"] = round(float(cos.min().item()), 6)
        stats["h15_rel_l2_drift_vs_base"] = round(float(rel_l2.mean().item()), 6)
    return h15_mat, stats


def run_warmstart_condition(
    cond_name: str,
    lr: float,
    gate_lr: float,
    freeze_mode: str,
    init_from_track2: bool = True,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    seed = 20261124
    grad_accum = 8
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(
        f"\n[R4] === Starting {cond_name} (lr={lr}, gate_lr={gate_lr}, freeze={freeze_mode}, init_t2={init_from_track2}) ===",
        flush=True,
    )
    runner = QwenDeciderRunner()
    base_cpu_state = capture_cpu_state_dict(runner.model)
    dev_rows = load_split_rows("select_dev")
    ab_ids = [runner.letter_token_ids["A"], runner.letter_token_ids["B"]]

    probe_ids = _prepare_probe_prompts(runner, dev_rows)
    h15_base, _ = _probe_live_state(runner, probe_ids, ab_ids, h15_base=None)

    groups = prepare_training_groups(runner, Path(DEFAULT_TRAIN_FILE))

    # Apply Track 2 WSP warm-start deltas unless running Clean Base control
    if init_from_track2:
        init_bundle = load_patch_bundle(CHECKPOINTS_DIR / "track2_best_deltas.pt")
        apply_bundle(runner.model, base_cpu_state, init_bundle.patches)

    inject_lora(runner.model, r=16, alpha=32.0, train_deltanet_gates=True)

    if freeze_mode == "freeze_L0_14":
        for n, p in runner.model.named_parameters():
            if n.startswith("model.layers."):
                try:
                    l_idx = int(n.split(".")[2])
                except (IndexError, ValueError):
                    l_idx = 999
                if l_idx < 15:
                    p.requires_grad_(False)
        need_input_grad = False
    elif freeze_mode == "freeze_t2gates":
        for n, p in runner.model.named_parameters():
            if ("model.layers.15.mlp." in n or "model.layers.22.mlp." in n) and (".lora_A" in n or ".lora_B" in n):
                p.requires_grad_(False)
        need_input_grad = True
    elif freeze_mode == "none":
        need_input_grad = True
    else:
        raise ValueError(f"Unknown freeze_mode={freeze_mode!r}")

    lora_params, gate_params = lora_param_groups(runner.model)
    all_trainable = lora_params + gate_params
    n_trainable = sum(p.numel() for p in all_trainable)
    print(f"[R4] {cond_name}: trainable_params={n_trainable}, need_input_grad={need_input_grad}", flush=True)

    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": lr, "weight_decay": 0.02},
            {"params": gate_params, "lr": gate_lr, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.95),
    )
    steps_per_epoch = math.ceil(len(groups) / grad_accum)
    total_opt_steps = 2 * steps_per_epoch  # exact 94-step cosine budget from Track 3b
    warmup_steps = max(1, int(0.08 * total_opt_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: _cosine_lr_multiplier(s, total_opt_steps, warmup_steps, min_ratio=0.1),
    )

    # Probe step 0 (right after Track 2 warm-start + untouched zero LoRA init)
    _, st0 = _probe_live_state(runner, probe_ids, ab_ids, h15_base=h15_base)
    step_snapshots: list[dict[str, Any]] = [
        {"opt_step": 0, "group_idx": 0, "recent_loss": None, "last_grad_norm": None, "lr_lora": lr, **st0}
    ]
    print(
        f"[R4 {cond_name}] opt_step=0 h15_cos={st0['h15_cos_vs_base_mean']} l15_act={st0['l15_live_swiglu_mean_abs']} "
        f"max_m={st0['probe_max_abs_margin']}",
        flush=True,
    )

    runner.model.train()
    if need_input_grad:
        runner.model.enable_input_require_grads()
    runner.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    rng = random.Random(seed)
    order = list(range(len(groups)))
    rng.shuffle(order)
    optimizer.zero_grad(set_to_none=True)

    running_losses: list[float] = []
    opt_step = 0
    probed_steps = {0}
    last_gnorm = 0.0

    for step_idx, g_i in enumerate(order[:MAX_GROUPS]):
        grp = groups[g_i]
        stats = train_group_step(runner.model, runner.device, ab_ids, grp, grad_accum)
        running_losses.append(stats["total"])

        is_bnd = ((step_idx + 1) % grad_accum == 0) or (step_idx == MAX_GROUPS - 1)
        if is_bnd:
            gnorm = torch.nn.utils.clip_grad_norm_(all_trainable, 1.0)
            if not torch.isfinite(gnorm):
                raise RuntimeError(f"[{cond_name}] Non-finite grad_norm={gnorm} at opt_step={opt_step}")
            last_gnorm = float(gnorm.item()) if isinstance(gnorm, torch.Tensor) else float(gnorm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            opt_step += 1

            if opt_step in PROBE_OPT_STEPS and opt_step not in probed_steps:
                probed_steps.add(opt_step)
                recent_20 = round(sum(running_losses[-20:]) / len(running_losses[-20:]), 4)
                runner.model.gradient_checkpointing_disable()
                _, st_cur = _probe_live_state(runner, probe_ids, ab_ids, h15_base=h15_base)
                runner.model.train()
                if need_input_grad:
                    runner.model.enable_input_require_grads()
                runner.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                snap = {
                    "opt_step": opt_step,
                    "group_idx": step_idx + 1,
                    "recent_loss": recent_20,
                    "peak_loss_so_far": round(max(running_losses), 4),
                    "last_grad_norm": round(last_gnorm, 4),
                    "lr_lora": round(optimizer.param_groups[0]["lr"], 8),
                    **st_cur,
                }
                step_snapshots.append(snap)
                print(
                    f"[R4 {cond_name}] opt_step={opt_step} grp={step_idx+1} loss20={recent_20} "
                    f"gn={last_gnorm:.2f} h15_cos={st_cur['h15_cos_vs_base_mean']} "
                    f"l15_act={st_cur['l15_live_swiglu_mean_abs']} max_m={st_cur['probe_max_abs_margin']}",
                    flush=True,
                )

    runner.model.eval()
    runner.model.gradient_checkpointing_disable()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dev_rep = evaluate_runner(runner, dev_rows, mode="plain", include_recap=False, eval_recap_ablation=False)
    res = {
        "cond_name": cond_name,
        "lr": lr,
        "gate_lr": gate_lr,
        "freeze_mode": freeze_mode,
        "trainable_params": n_trainable,
        "peak_recent20_loss": round(
            max(s["recent_loss"] for s in step_snapshots if s["recent_loss"] is not None), 4
        ),
        "peak_group_loss": round(max(running_losses), 4),
        "step20_recent_loss": step_snapshots[-1]["recent_loss"],
        "step_snapshots": step_snapshots,
        "step20_dev_summary": {
            "proxy_score": dev_rep["proxy"]["proxy_score"],
            "real_cf_auroc": dev_rep["real_cf"]["auroc"],
            "real_cf_rec": dev_rep["real_cf"]["recall_at_05"],
            "real_cf_fpr": dev_rep["real_cf"]["fpr_at_05"],
            "consent_5p_acc": dev_rep["consent_5p"]["overall_acc"],
            "long_50t_acc": dev_rep["long_50t"]["overall_acc"],
            "decider_replay_acc": dev_rep["decider_replay"]["overall_acc"],
        },
        "elapsed_sec": round(time.perf_counter() - t0, 1),
    }

    del runner, base_cpu_state, groups, all_trainable, h15_base, probe_ids
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Suite R4 In-Situ Warm-Start Causality & Intervention Controls.")
    ap.add_argument(
        "--conditions",
        nargs="+",
        default=["t3a_insitu_base", "t3b_insitu_full", "t3b_lowlr_5x", "t3b_freeze_L0_14", "t3b_freeze_t2gates"],
    )
    args = ap.parse_args()

    cond_cfg = {
        "t3a_insitu_base": {"lr": 1.5e-4, "gate_lr": 5.0e-4, "freeze_mode": "none", "init_t2": False},
        "t3b_insitu_full": {"lr": 1.5e-4, "gate_lr": 5.0e-4, "freeze_mode": "none", "init_t2": True},
        "t3b_lowlr_5x": {"lr": 3.0e-5, "gate_lr": 1.0e-4, "freeze_mode": "none", "init_t2": True},
        "t3b_freeze_L0_14": {"lr": 1.5e-4, "gate_lr": 5.0e-4, "freeze_mode": "freeze_L0_14", "init_t2": True},
        "t3b_freeze_t2gates": {"lr": 1.5e-4, "gate_lr": 5.0e-4, "freeze_mode": "freeze_t2gates", "init_t2": True},
    }

    payload = json.loads(CONTROLS_JSON.read_text(encoding="utf-8")) if CONTROLS_JSON.exists() else {}
    ws = payload.get("warmstart_causality", {})

    for cname in args.conditions:
        cfg = cond_cfg[cname]
        ws[cname] = run_warmstart_condition(
            cname, cfg["lr"], cfg["gate_lr"], cfg["freeze_mode"], init_from_track2=cfg["init_t2"]
        )
        payload["warmstart_causality"] = ws
        EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        CONTROLS_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[R4] Saved {cname} -> experiments/paper_revision_controls.json", flush=True)


if __name__ == "__main__":
    main()
