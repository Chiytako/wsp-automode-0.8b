"""Phase 5 Track 3+ training utilities (Pure-PyTorch LoRA + order-equivariant contrastive losses)."""

from train.lora_model import (
    TARGET_LINEAR_NAMES,
    LoRALinear,
    compute_live_weight_diff_stats,
    extract_merged_patch_bundle,
    inject_lora,
    load_lora_adapter,
    lora_param_groups,
    save_lora_adapter,
)
from train.losses import compute_binary_family_loss, compute_replay_loss

__all__ = [
    "LoRALinear",
    "TARGET_LINEAR_NAMES",
    "inject_lora",
    "lora_param_groups",
    "compute_live_weight_diff_stats",
    "extract_merged_patch_bundle",
    "save_lora_adapter",
    "load_lora_adapter",
    "compute_binary_family_loss",
    "compute_replay_loss",
]
