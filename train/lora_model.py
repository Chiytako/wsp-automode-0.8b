"""Pure-PyTorch deterministic LoRA wrapper & Gated DeltaNet gate trainer for Qwen3.5-0.8B.

Zero external `peft` dependency:
- Wraps target nn.Linear layers inside `model.model.layers[0..23]` with float32 `lora_A`/`lora_B` (`dropout=0.0`).
- Optionally unlocks 1D `linear_attn.A_log` & `linear_attn.dt_bias` (promoted to float32 during training).
- Exports exact net delta PatchBundle compatible with `eval/evaluate.py --patch-deltas` and `runtime/export_hf.py`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from weight_edit.causal_check import PatchBundle, WeightPatch

TARGET_LINEAR_NAMES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


class LoRALinear(nn.Module):
    """Deterministic Low-Rank Adaptation wrapper (`dropout=0.0`) around a frozen `nn.Linear`."""

    def __init__(self, base: nn.Linear, r: int = 16, alpha: float = 32.0) -> None:
        super().__init__()
        if base.bias is not None:
            base.bias.requires_grad_(False)
        base.weight.requires_grad_(False)
        self.base = base
        self.in_features = int(base.in_features)
        self.out_features = int(base.out_features)
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(r)

        device = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(self.r, self.in_features, device=device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.r, device=device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = F.linear(F.linear(x.float(), self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out.to(dtype=base_out.dtype)

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        w = self.base.weight.detach().float() + self.scaling * (self.lora_B.detach().float() @ self.lora_A.detach().float())
        return w.to(dtype=self.base.weight.dtype)


def inject_lora(
    model: nn.Module,
    r: int = 16,
    alpha: float = 32.0,
    train_deltanet_gates: bool = True,
    target_names: tuple[str, ...] = TARGET_LINEAR_NAMES,
) -> dict[str, Any]:
    """Freeze base parameters and inject LoRALinear into transformer layers (0..N-1)."""
    for p in model.parameters():
        p.requires_grad_(False)

    layers = model.model.layers if hasattr(model, "model") and hasattr(model.model, "layers") else None
    if layers is None:
        raise AttributeError("Expected model.model.layers on Qwen3.5 CausalLM")

    target_set = set(target_names)
    wrapped_names: list[str] = []
    gate_param_names: list[str] = []

    for idx, layer in enumerate(layers):
        # Replace matching nn.Linear modules cleanly using parent navigation
        replacements: list[tuple[nn.Module, str, str, LoRALinear]] = []
        for full_rel, mod in layer.named_modules():
            if not full_rel or isinstance(mod, LoRALinear):
                continue
            leaf = full_rel.split(".")[-1]
            if isinstance(mod, nn.Linear) and leaf in target_set:
                parent = layer
                parts = full_rel.split(".")
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                lora_mod = LoRALinear(mod, r=r, alpha=alpha)
                replacements.append((parent, leaf, f"model.layers.{idx}.{full_rel}", lora_mod))
        for parent, leaf, abs_name, lora_mod in replacements:
            setattr(parent, leaf, lora_mod)
            wrapped_names.append(abs_name)

        if train_deltanet_gates and hasattr(layer, "linear_attn"):
            lin_attn = layer.linear_attn
            for gname in ("A_log", "dt_bias"):
                if hasattr(lin_attn, gname):
                    gp = getattr(lin_attn, gname)
                    if isinstance(gp, torch.Tensor):
                        gp.data = gp.data.float()
                        gp.requires_grad_(True)
                        gate_param_names.append(f"model.layers.{idx}.linear_attn.{gname}")

    lora_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and (".lora_A" in n or ".lora_B" in n))
    gate_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and ("A_log" in n or "dt_bias" in n))
    return {
        "r": int(r),
        "alpha": float(alpha),
        "wrapped_modules": len(wrapped_names),
        "lora_params": int(lora_params),
        "gate_params": int(gate_params),
        "total_trainable": int(lora_params + gate_params),
        "gate_param_names": gate_param_names,
    }


def lora_param_groups(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Return `(lora_params, gate_params)` of currently trainable parameters."""
    lora_ps: list[nn.Parameter] = []
    gate_ps: list[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ".lora_A" in name or ".lora_B" in name:
            lora_ps.append(p)
        else:
            gate_ps.append(p)
    return lora_ps, gate_ps


@torch.no_grad()
def _collect_effective_weight_deltas(
    model: nn.Module,
    base_cpu_state: dict[str, torch.Tensor],
    atol: float = 1e-8,
) -> tuple[int, dict[str, torch.Tensor], dict[str, dict[str, Any]]]:
    """Collect effective merged weights vs `base_cpu_state` without mutating `LoRALinear` modules."""
    total_params = 0
    deltas: dict[str, torch.Tensor] = {}
    tensor_stats: dict[str, dict[str, Any]] = {}

    # Track handled weight keys under LoRALinear.base.weight
    handled_base_keys: set[str] = set()
    for mod_name, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            base_param_name = f"{mod_name}.weight"
            handled_base_keys.add(f"{mod_name}.base.weight")
            if base_param_name not in base_cpu_state:
                raise KeyError(f"Expected {base_param_name!r} in base_cpu_state")
            base_ref = base_cpu_state[base_param_name]
            n = base_ref.numel()
            total_params += n

            w_merged = mod.merged_weight().cpu().to(dtype=base_ref.dtype)
            diff = w_merged.float() - base_ref.float()
            abs_diff = diff.abs()
            n_changed = int((abs_diff > atol).sum().item())
            if n_changed > 0:
                frob = float(torch.linalg.norm(diff).item())
                deltas[base_param_name] = diff
                tensor_stats[base_param_name] = {
                    "name": base_param_name,
                    "shape": list(base_ref.shape),
                    "numel": n,
                    "modified_elements": n_changed,
                    "pct_elements": round(100.0 * n_changed / n, 4),
                    "delta_frobenius_norm": round(frob, 6),
                    "max_abs_delta": round(float(abs_diff.max().item()), 6),
                }

    # Include non-LoRA trainable/edited parameters (A_log, dt_bias, etc.) and untouched params
    for pname, param in model.named_parameters():
        if pname in handled_base_keys or ".lora_A" in pname or ".lora_B" in pname:
            continue
        if pname.endswith(".base.bias"):
            orig_name = pname[: -len(".base.bias")] + ".bias"
        else:
            orig_name = pname
        base_ref = base_cpu_state[orig_name]
        n = base_ref.numel()
        total_params += n
        cur = param.detach().cpu().to(dtype=base_ref.dtype)
        diff = cur.float() - base_ref.float()
        abs_diff = diff.abs()
        n_changed = int((abs_diff > atol).sum().item())
        if n_changed > 0:
            frob = float(torch.linalg.norm(diff).item())
            deltas[orig_name] = diff
            tensor_stats[orig_name] = {
                "name": orig_name,
                "shape": list(base_ref.shape),
                "numel": n,
                "modified_elements": n_changed,
                "pct_elements": round(100.0 * n_changed / n, 4),
                "delta_frobenius_norm": round(frob, 6),
                "max_abs_delta": round(float(abs_diff.max().item()), 6),
            }
    return total_params, deltas, tensor_stats


@torch.no_grad()
def compute_live_weight_diff_stats(
    model: nn.Module,
    base_cpu_state: dict[str, torch.Tensor],
    atol: float = 1e-7,
) -> dict[str, Any]:
    """Compute effective `% modified params` & `||ΔW||_F` matching `compute_weight_diff_stats` schema."""
    total_params, _deltas, tensor_stats = _collect_effective_weight_deltas(model, base_cpu_state, atol=atol)
    modified_params = sum(s["modified_elements"] for s in tensor_stats.values())
    sq_frob_sum = sum(s["delta_frobenius_norm"] ** 2 for s in tensor_stats.values())
    return {
        "total_params": int(total_params),
        "modified_params": int(modified_params),
        "pct_modified_params": round(100.0 * modified_params / max(1, total_params), 6),
        "modified_tensor_count": len(tensor_stats),
        "total_delta_frobenius_norm": round(math.sqrt(sq_frob_sum), 6),
        "modified_tensors": list(tensor_stats.values()),
    }


@torch.no_grad()
def extract_merged_patch_bundle(
    model: nn.Module,
    base_cpu_state: dict[str, torch.Tensor],
    step_id: str,
    metadata: dict[str, Any] | None = None,
    atol: float = 1e-8,
) -> PatchBundle:
    """Materialize the exact merged weight delta vs `base_cpu_state` as a `PatchBundle`."""
    _total_params, deltas, _stats = _collect_effective_weight_deltas(model, base_cpu_state, atol=atol)
    layers_touched = sorted(
        {
            int(k.split(".")[2])
            for k in deltas
            if k.startswith("model.layers.") and len(k.split(".")) > 2 and k.split(".")[2].isdigit()
        }
    )
    meta = dict(metadata or {})
    patch = WeightPatch(
        step_id=step_id,
        category="track3_lora_merged",
        method="order_equivariant_contrastive_lora",
        target_layers=layers_touched,
        rank=int(meta.get("r", 16)),
        alpha=float(meta.get("alpha", 32.0)),
        hypothesis=str(meta.get("hypothesis", "Order-equivariant contrastive LoRA + DeltaNet gate fine-tuning")),
        deltas=deltas,
        metadata=meta,
    )
    return PatchBundle(patches=[patch], metadata=meta)


def save_lora_adapter(
    model: nn.Module,
    path: Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save compact trainable LoRA + gate weights (~43 MB)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        name: param.detach().float().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    torch.save({"metadata": dict(metadata or {}), "state": state}, path)


def load_lora_adapter(model: nn.Module, path: Path) -> dict[str, Any]:
    """Load trainable LoRA + gate weights in-place onto an already `inject_lora`-wrapped model."""
    raw = torch.load(Path(path), map_location="cpu", weights_only=True)
    state: dict[str, torch.Tensor] = raw["state"]
    param_map = dict(model.named_parameters())
    missing = [k for k in state if k not in param_map]
    if missing:
        raise KeyError(f"Adapter keys missing on model: {missing[:5]}")
    with torch.no_grad():
        for k, v in state.items():
            p = param_map[k]
            p.copy_(v.to(device=p.device, dtype=p.dtype))
    return dict(raw.get("metadata", {}))
