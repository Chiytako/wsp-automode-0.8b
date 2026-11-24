"""Revision Suite R3: Deconfounding Controls (`Sparse-Grad` vs `Dense LoRA r=1` vs `Track 2 WSP` vs `Track 3 Soup`).

Trains 3 controlled gradient baselines on exact `data/local_private/phase5_train.jsonl` (683 rows,
375 families), exact Track 3 loss, and `epochs=2`, selecting best epoch strictly on `select_dev` (n=200):
  - Control 2A (`ctrl_sparse_8neuron_grad`): Trainable ONLY on Track 2's exact 8 neurons (24,576 params).
  - Control 2B (`ctrl_lora_r1_same6`): Rank-1 dense LoRA on Track 2's exact 6 MLP tensors (27,648 trainable, 22.02M modified).
  - Control 2C (`ctrl_lora_r1_depth8`): Rank-1 dense LoRA on `out_proj`/`o_proj` across 8 layers [2,5,8,11,14,17,20,23]
    (exact 24,576 trainable = 1.00x capacity match, 16.78M modified across depth).

Evaluates each best checkpoint once on locked `frozen_test` (n=200), `adversarial_redteam` (n=160 + 32 leaky),
CPU safetensors `Q8_0` block forensics, and GGUF quantization curve (`Q8_0, Q6_K, Q5_K_M, Q4_K_M`) on `select_dev`,
merging results into `experiments/paper_revision_controls.json` under `"deconfound_controls"`.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download

from data.build_adversarial_bench import ADV_PATH
from eval.evaluate import evaluate_runner, load_split_rows
from eval.eval_paper_suite import (
    _analyze_weight_forensics_against_base,
    _eval_pytorch_model,
    _load_hf_safetensors_dir,
)
from eval.mechanism_analysis import QUANT_TYPES, T2_L15_NEURONS, T2_L22_NEURONS, _collect_pt_select_dev_ref
from eval.model_runner import QwenDeciderRunner, capture_cpu_state_dict
from eval.revision_quant_mitigations import (
    CONTROLS_JSON,
    TMP_GGUF_DIR,
    eval_gguf_select_dev_detailed,
    export_bundle_to_temp_ggufs,
)
from runtime.export_hf import map_causal_to_hf_key
from train.lora_model import LoRALinear
from train.train_track3 import (
    DEFAULT_TRAIN_FILE,
    _cosine_lr_multiplier,
    compute_track3_select_score,
    prepare_training_groups,
    train_group_step,
)
from weight_edit.causal_check import PatchBundle, WeightPatch, save_patch_bundle

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = ROOT / "experiments"

CTRL_L15_IDX = T2_L15_NEURONS  # [3447, 291, 1066, 917]
CTRL_L22_IDX = T2_L22_NEURONS  # [3394, 3581, 3037, 2593]
DEPTH8_LAYERS = [2, 5, 8, 11, 14, 17, 20, 23]


class SparseNeuronLinear(nn.Module):
    """Train ONLY specific output rows (gate/up) or input columns (down) of a frozen nn.Linear."""

    def __init__(self, base: nn.Linear, indices: list[int], mode: str) -> None:
        super().__init__()
        if base.bias is not None:
            base.bias.requires_grad_(False)
        base.weight.requires_grad_(False)
        self.base = base
        self.mode = mode
        self.indices = [int(i) for i in indices]
        dev = base.weight.device
        self.register_buffer("idx_t", torch.tensor(self.indices, device=dev, dtype=torch.long), persistent=False)
        if mode == "rows":
            self.delta = nn.Parameter(torch.zeros(len(self.indices), int(base.in_features), device=dev, dtype=torch.float32))
        elif mode == "cols":
            self.delta = nn.Parameter(torch.zeros(int(base.out_features), len(self.indices), device=dev, dtype=torch.float32))
        else:
            raise ValueError(f"Unsupported SparseNeuronLinear mode: {mode!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        if self.mode == "rows":
            sub = F.linear(x.float(), self.delta).to(dtype=base_out.dtype)
            return base_out.index_add(-1, self.idx_t, sub)
        sub_x = torch.index_select(x, -1, self.idx_t).float()
        upd = F.linear(sub_x, self.delta).to(dtype=base_out.dtype)
        return base_out + upd

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        w = self.base.weight.detach().float().clone()
        d = self.delta.detach().float().to(device=w.device)
        if self.mode == "rows":
            w[self.idx_t.to(device=w.device)] += d
        else:
            w[:, self.idx_t.to(device=w.device)] += d
        return w.to(dtype=self.base.weight.dtype)


def _inject_control_modules(model: nn.Module, ctrl_name: str) -> dict[str, Any]:
    for p in model.parameters():
        p.requires_grad_(False)

    layers = model.model.layers
    wrapped: list[str] = []

    if ctrl_name == "ctrl_sparse_8neuron_grad":
        for l_idx, idxs in ((15, CTRL_L15_IDX), (22, CTRL_L22_IDX)):
            mlp = layers[l_idx].mlp
            for leaf, mode in (("gate_proj", "rows"), ("up_proj", "rows"), ("down_proj", "cols")):
                orig = getattr(mlp, leaf)
                setattr(mlp, leaf, SparseNeuronLinear(orig, indices=idxs, mode=mode))
                wrapped.append(f"model.layers.{l_idx}.mlp.{leaf}")
        need_input_grad = False
    elif ctrl_name == "ctrl_lora_r1_same6":
        for l_idx in (15, 22):
            mlp = layers[l_idx].mlp
            for leaf in ("gate_proj", "up_proj", "down_proj"):
                orig = getattr(mlp, leaf)
                setattr(mlp, leaf, LoRALinear(orig, r=1, alpha=2.0))
                wrapped.append(f"model.layers.{l_idx}.mlp.{leaf}")
        need_input_grad = False
    elif ctrl_name == "ctrl_lora_r1_depth8":
        for l_idx in DEPTH8_LAYERS:
            layer = layers[l_idx]
            if hasattr(layer, "linear_attn"):
                orig = layer.linear_attn.out_proj
                layer.linear_attn.out_proj = LoRALinear(orig, r=1, alpha=2.0)
                wrapped.append(f"model.layers.{l_idx}.linear_attn.out_proj")
            elif hasattr(layer, "self_attn"):
                orig = layer.self_attn.o_proj
                layer.self_attn.o_proj = LoRALinear(orig, r=1, alpha=2.0)
                wrapped.append(f"model.layers.{l_idx}.self_attn.o_proj")
        need_input_grad = True
    else:
        raise ValueError(f"Unknown ctrl_name={ctrl_name!r}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "ctrl_name": ctrl_name,
        "wrapped": wrapped,
        "trainable_params": int(trainable),
        "need_input_grad": need_input_grad,
    }


@torch.no_grad()
def _extract_deltas_bundle(
    model: nn.Module,
    base_cpu_state: dict[str, torch.Tensor],
    ctrl_name: str,
    epoch: int,
    atol: float = 1e-8,
) -> tuple[PatchBundle, dict[str, torch.Tensor]]:
    deltas: dict[str, torch.Tensor] = {}
    for mod_name, mod in model.named_modules():
        if isinstance(mod, (SparseNeuronLinear, LoRALinear)):
            base_key = f"{mod_name}.weight"
            base_ref = base_cpu_state[base_key]
            w_merged = mod.merged_weight().cpu().float()
            diff = w_merged - base_ref.float()
            if int((diff.abs() > atol).sum().item()) > 0:
                deltas[base_key] = diff
    patch = WeightPatch(
        step_id=f"{ctrl_name}_ep{epoch}",
        category=ctrl_name,
        method="controlled_gradient_baseline",
        target_layers=[],
        rank=1,
        alpha=1.0,
        hypothesis=f"Deconfounding control {ctrl_name} best dev epoch {epoch}.",
        deltas=deltas,
        metadata={"epoch": epoch, "ctrl_name": ctrl_name},
    )
    return PatchBundle(patches=[patch], metadata={"ctrl_name": ctrl_name, "best_epoch": epoch}), deltas


def _compute_forensics_from_deltas(
    ctrl_name: str,
    deltas: dict[str, torch.Tensor],
    base_hf_tensors: dict[str, torch.Tensor],
) -> dict[str, Any]:
    summed = {map_causal_to_hf_key(k): v.detach().float().cpu() for k, v in deltas.items()}
    cand = {
        k: ((v.float() + summed[k]).to(dtype=v.dtype) if k in summed else v)
        for k, v in base_hf_tensors.items()
    }
    wf = _analyze_weight_forensics_against_base(ctrl_name, base_hf_tensors, cand)
    wf.pop("top_modified_tensors", None)
    return wf


def train_and_evaluate_control(
    ctrl_name: str,
    lr: float,
    epochs: int,
    unlock_frozen_test: bool,
    port: int,
    base_hf_tensors: dict[str, torch.Tensor],
    frozen_rows: list[dict[str, Any]],
    adv_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    t0 = time.perf_counter()
    seed = 20261124
    grad_accum = 8
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"\n[R3] === Starting Control: {ctrl_name} (lr={lr}, epochs={epochs}) ===", flush=True)
    runner = QwenDeciderRunner()
    base_cpu = capture_cpu_state_dict(runner.model)
    groups = prepare_training_groups(runner, Path(DEFAULT_TRAIN_FILE))
    dev_rows = load_split_rows("select_dev")
    ab_ids = [runner.letter_token_ids["A"], runner.letter_token_ids["B"]]

    inj = _inject_control_modules(runner.model, ctrl_name)
    print(f"[R3] Injected {ctrl_name}: trainable_params={inj['trainable_params']}", flush=True)
    trainable_params = [p for p in runner.model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        [{"params": trainable_params, "lr": lr, "weight_decay": 0.01 if "sparse" in ctrl_name else 0.02}],
        betas=(0.9, 0.95),
    )
    steps_per_epoch = math.ceil(len(groups) / grad_accum)
    total_opt_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = max(1, int(0.08 * total_opt_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: _cosine_lr_multiplier(s, total_opt_steps, warmup_steps, min_ratio=0.1),
    )

    best_score = -1e9
    best_epoch = -1
    best_deltas_bundle: PatchBundle | None = None
    best_deltas_dict: dict[str, torch.Tensor] | None = None
    best_dev_summary: dict[str, Any] = {}
    epoch_logs: list[dict[str, Any]] = []
    rng = random.Random(seed)
    opt_step = 0

    for ep in range(1, epochs + 1):
        runner.model.train()
        if inj["need_input_grad"]:
            runner.model.enable_input_require_grads()
        runner.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        order = list(range(len(groups)))
        rng.shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        ep_t0 = time.perf_counter()

        for step_idx, g_i in enumerate(order):
            grp = groups[g_i]
            stats = train_group_step(runner.model, runner.device, ab_ids, grp, grad_accum)
            losses.append(stats["total"])

            is_bnd = ((step_idx + 1) % grad_accum == 0) or (step_idx == len(order) - 1)
            if is_bnd:
                gnorm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                if not torch.isfinite(gnorm):
                    raise RuntimeError(f"[{ctrl_name}] Non-finite grad_norm={gnorm} at ep={ep} step={step_idx+1}")
                if opt_step == 0 and float(gnorm) <= 0.0:
                    raise RuntimeError(f"[{ctrl_name}] Zero grad_norm on step 0 — autograd graph detached!")
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1

            if (step_idx + 1) % 75 == 0 or step_idx == len(order) - 1:
                recent = round(sum(losses[-25:]) / len(losses[-25:]), 4)
                print(
                    f"[R3 {ctrl_name} ep={ep}] {step_idx+1}/{len(order)} opt={opt_step} loss={recent} lr={optimizer.param_groups[0]['lr']:.2e}",
                    flush=True,
                )

        runner.model.eval()
        runner.model.gradient_checkpointing_disable()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        rep_dev = evaluate_runner(runner, dev_rows, mode="plain", include_recap=False, eval_recap_ablation=False)
        sel_score = compute_track3_select_score(rep_dev)
        ep_sec = round(time.perf_counter() - ep_t0, 1)
        epoch_logs.append(
            {
                "epoch": ep,
                "train_sec": ep_sec,
                "mean_epoch_loss": round(sum(losses) / len(losses), 4),
                "track3_select_score": sel_score,
                "proxy_score": rep_dev["proxy"]["proxy_score"],
                "real_cf_auroc": rep_dev["real_cf"]["auroc"],
                "consent_5p_acc": rep_dev["consent_5p"]["overall_acc"],
                "long_50t_acc": rep_dev["long_50t"]["overall_acc"],
                "decider_replay_acc": rep_dev["decider_replay"]["overall_acc"],
            }
        )
        print(
            f"[R3 {ctrl_name}] Ep {ep} dev ({ep_sec}s): select={sel_score} proxy={rep_dev['proxy']['proxy_score']} "
            f"rc_auc={rep_dev['real_cf']['auroc']} c5={rep_dev['consent_5p']['overall_acc']} "
            f"l50={rep_dev['long_50t']['overall_acc']} rep={rep_dev['decider_replay']['overall_acc']}",
            flush=True,
        )

        if sel_score > best_score:
            best_score = sel_score
            best_epoch = ep
            best_deltas_bundle, best_deltas_dict = _extract_deltas_bundle(runner.model, base_cpu, ctrl_name, ep)
            best_dev_summary = {
                "proxy_score": rep_dev["proxy"]["proxy_score"],
                "real_cf_auroc": rep_dev["real_cf"]["auroc"],
                "real_cf_rec": rep_dev["real_cf"]["recall_at_05"],
                "real_cf_fpr": rep_dev["real_cf"]["fpr_at_05"],
                "consent_5p_acc": rep_dev["consent_5p"]["overall_acc"],
                "long_50t_acc": rep_dev["long_50t"]["overall_acc"],
                "decider_replay_acc": rep_dev["decider_replay"]["overall_acc"],
                "decider_replay_ece": rep_dev["decider_replay"]["top_label_ece"],
            }

    del runner, base_cpu, groups, trainable_params
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    assert best_deltas_bundle is not None and best_deltas_dict is not None
    TMP_GGUF_DIR.mkdir(parents=True, exist_ok=True)
    tmp_deltas_pt = TMP_GGUF_DIR / f"{ctrl_name}_best_deltas.pt"
    tmp_spec_json = TMP_GGUF_DIR / f"{ctrl_name}_spec.json"
    save_patch_bundle(best_deltas_bundle, tmp_spec_json, tmp_deltas_pt)

    # 1. Forensics vs Base
    wf = _compute_forensics_from_deltas(ctrl_name, best_deltas_dict, base_hf_tensors)

    # 2. Locked frozen_test & adversarial_redteam evaluation (single evaluation pass)
    fr_res, ad_res = _eval_pytorch_model(ctrl_name, None, tmp_deltas_pt, frozen_rows, adv_rows)
    fr_sum = fr_res["summary"]
    ad_sum = ad_res["summary"]
    all200_acc = round(
        (
            sum(1 for x in fr_res["binary_records"] if (x["pred_norm"] == "block") == (x["y"] == 1))
            + sum(int(x["correct"]) for x in fr_res["replay_records"])
        )
        / 200.0,
        4,
    )

    # 3. Multi-bit quantization curve on select_dev (n=200) vs own bf16 ref
    ref_dev = _collect_pt_select_dev_ref(tmp_deltas_pt, dev_rows)
    qpaths = export_bundle_to_temp_ggufs(ctrl_name, tmp_deltas_pt, QUANT_TYPES)
    q_curve: dict[str, Any] = {}
    try:
        for qtype, qpath in qpaths.items():
            print(f"[R3 {ctrl_name}] Evaluating {qtype}...", flush=True)
            q_curve[qtype] = eval_gguf_select_dev_detailed(qpath, ref_dev, port)
            if qpath.exists():
                qpath.unlink()
    finally:
        if TMP_GGUF_DIR.exists():
            shutil.rmtree(TMP_GGUF_DIR, ignore_errors=True)

    return {
        "ctrl_name": ctrl_name,
        "lr": lr,
        "epochs": epochs,
        "trainable_params": inj["trainable_params"],
        "best_epoch": best_epoch,
        "best_select_score": best_score,
        "best_dev_summary": best_dev_summary,
        "epoch_logs": epoch_logs,
        "weight_forensics": wf,
        "frozen_test_summary": {
            "proxy_score": fr_sum["proxy"]["proxy_score"],
            "all200_acc": all200_acc,
            "real_cf_auroc": fr_sum["real_cf"]["auroc"],
            "real_cf_rec": fr_sum["real_cf"]["recall_at_05"],
            "real_cf_fpr": fr_sum["real_cf"]["fpr_at_05"],
            "real_cf_ece": fr_sum["real_cf"]["ece"],
            "real_cf_flip": fr_sum["real_cf"]["order_flip_rate"],
            "consent_5p_acc": fr_sum["consent_5p"]["overall_acc"],
            "consent_5p_p2": fr_sum["consent_5p"]["p2_allow_rate"],
            "consent_5p_p3": fr_sum["consent_5p"]["p3_block_rate"],
            "long_50t_acc": fr_sum["long_50t"]["overall_acc"],
            "decider_replay_acc": fr_sum["decider_replay"]["overall_acc"],
            "decider_replay_ece": fr_sum["decider_replay"]["top_label_ece"],
        },
        "adversarial_summary": {
            "overall_acc": ad_sum["overall_acc"],
            "balanced_acc": ad_sum["balanced_acc"],
            "auroc": ad_sum["auroc"],
            "overall_asr": ad_sum["overall_asr"],
            "benign_fpr": ad_sum["benign_fpr"],
            "tier2_file_edit_asr": ad_sum["tier2_file_edit_asr"],
            "multilingual_acc": ad_sum["by_category"]["multilingual_paraphrase"]["acc"],
            "rb_blind_asr": ad_sum["reasoning_blind_ablation"]["blind_asr"],
            "rb_leaky_asr": ad_sum["reasoning_blind_ablation"]["leaky_asr"],
            "order_flip_rate": ad_sum["order_flip_rate"],
        },
        "quantization_curve": q_curve,
        "elapsed_sec": round(time.perf_counter() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Suite R3 Deconfounding Controls.")
    ap.add_argument("--unlock-frozen-test", action="store_true", required=True)
    ap.add_argument(
        "--controls",
        nargs="+",
        default=["ctrl_sparse_8neuron_grad", "ctrl_lora_r1_same6", "ctrl_lora_r1_depth8"],
    )
    ap.add_argument("--port", type=int, default=8095)
    args = ap.parse_args()

    frozen_rows = load_split_rows("frozen_test", unlock_frozen_test=args.unlock_frozen_test)
    adv_rows = [json.loads(ln) for ln in ADV_PATH.read_text(encoding="utf-8").splitlines() if ln.strip()]
    base_dir = Path(snapshot_download("Qwen/Qwen3.5-0.8B", local_files_only=True))
    base_hf_tensors = _load_hf_safetensors_dir(base_dir)

    ctrl_specs = {
        "ctrl_sparse_8neuron_grad": {"lr": 1.5e-3, "epochs": 2},
        "ctrl_lora_r1_same6": {"lr": 5.0e-4, "epochs": 2},
        "ctrl_lora_r1_depth8": {"lr": 5.0e-4, "epochs": 2},
    }

    payload = json.loads(CONTROLS_JSON.read_text(encoding="utf-8")) if CONTROLS_JSON.exists() else {}
    deconf = payload.get("deconfound_controls", {})

    for cname in args.controls:
        sp = ctrl_specs[cname]
        res = train_and_evaluate_control(
            cname,
            lr=sp["lr"],
            epochs=sp["epochs"],
            unlock_frozen_test=args.unlock_frozen_test,
            port=args.port,
            base_hf_tensors=base_hf_tensors,
            frozen_rows=frozen_rows,
            adv_rows=adv_rows,
        )
        deconf[cname] = res
        payload["deconfound_controls"] = deconf
        EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        CONTROLS_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[R3] Saved {cname} -> experiments/paper_revision_controls.json", flush=True)


if __name__ == "__main__":
    main()
