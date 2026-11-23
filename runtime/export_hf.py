"""Export gradient-free patched weights into a standard Qwen3.5 HF safetensors directory.

Maps AutoModelForCausalLM parameter names (`model.*`) back to the on-disk HF multimodal
layout (`model.language_model.*`), preserves untouched `model.visual.*` tensors, keeps
original `bfloat16` dtypes, and verifies structural + Frobenius norm parity after saving.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

from weight_edit.causal_check import load_patch_bundle

MODEL_ID = "Qwen/Qwen3.5-0.8B"

COPY_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "LICENSE",
    "README.md",
]


def map_causal_to_hf_key(name: str) -> str:
    """Remap `Qwen3_5ForCausalLM` key names (`model.*`) to disk HF key names (`model.language_model.*`)."""
    if name.startswith("model.") and not name.startswith("model.language_model."):
        return "model.language_model." + name[len("model."):]
    return name


def export_patched_hf(
    deltas_path: Path,
    output_dir: Path,
    base_snapshot_dir: Path | None = None,
) -> dict[str, Any]:
    """Apply net delta bundle onto base HF snapshot into `output_dir` without modifying cache."""
    src_dir = (
        Path(base_snapshot_dir)
        if base_snapshot_dir is not None
        else Path(snapshot_download(MODEL_ID, local_files_only=True))
    )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    copied_files: list[str] = []
    for fname in COPY_FILES:
        src_file = src_dir / fname
        if src_file.exists():
            shutil.copy2(src_file, out_dir / fname)
            copied_files.append(fname)

    # Identify safetensors shard filename
    shard_name = "model.safetensors-00001-of-00001.safetensors"
    if not (src_dir / shard_name).exists():
        alt = src_dir / "model.safetensors"
        if alt.exists():
            shard_name = "model.safetensors"
        else:
            raise FileNotFoundError(f"No safetensors shard found in {src_dir}")

    # Load and sum net deltas across all steps in patch bundle
    bundle = load_patch_bundle(Path(deltas_path))
    summed_hf_deltas: dict[str, torch.Tensor] = {}
    for p in bundle.patches:
        for causal_key, d_t in p.net_deltas().items():
            hf_key = map_causal_to_hf_key(causal_key)
            if hf_key in summed_hf_deltas:
                summed_hf_deltas[hf_key] = summed_hf_deltas[hf_key] + d_t.detach().float().cpu()
            else:
                summed_hf_deltas[hf_key] = d_t.detach().float().cpu().clone()

    base_tensors = load_file(str(src_dir / shard_name), device="cpu")
    missing = [k for k in summed_hf_deltas if k not in base_tensors]
    if missing:
        raise KeyError(f"Mapped delta keys not found in base safetensors: {missing}")

    patched_tensors: dict[str, torch.Tensor] = {}
    for k, base_t in base_tensors.items():
        if k in summed_hf_deltas:
            delta_t = summed_hf_deltas[k]
            if list(delta_t.shape) != list(base_t.shape):
                raise ValueError(f"Shape mismatch on {k}: delta={list(delta_t.shape)} vs base={list(base_t.shape)}")
            patched_tensors[k] = (base_t.float() + delta_t.float()).to(dtype=base_t.dtype)
        else:
            patched_tensors[k] = base_t

    dst_sf_path = out_dir / shard_name
    save_file(patched_tensors, str(dst_sf_path), metadata={"format": "pt"})

    # Reload from disk to verify structural & Frobenius parity
    reloaded = load_file(str(dst_sf_path), device="cpu")
    assert set(reloaded.keys()) == set(base_tensors.keys()), "Exported key set mismatch!"

    total_params = 0
    modified_params = 0
    sq_frob_sum = 0.0
    modified_tensors: list[dict[str, Any]] = []
    for k, base_t in base_tensors.items():
        rel_t = reloaded[k]
        assert list(rel_t.shape) == list(base_t.shape), f"Shape mismatch on {k}"
        assert rel_t.dtype == base_t.dtype, f"Dtype mismatch on {k}"
        n = base_t.numel()
        total_params += n
        if k in summed_hf_deltas:
            diff = rel_t.float() - base_t.float()
            n_changed = int((diff.abs() > 1e-7).sum().item())
            frob = float(torch.linalg.norm(diff).item())
            sq_frob_sum += frob * frob
            modified_params += n_changed
            modified_tensors.append(
                {
                    "name": k,
                    "shape": list(base_t.shape),
                    "dtype": str(base_t.dtype),
                    "modified_elements": n_changed,
                    "delta_frobenius_norm": round(frob, 6),
                }
            )
        else:
            assert torch.equal(rel_t, base_t), f"Unexpected modification on untouched tensor {k}"

    report = {
        "source_snapshot_dir": str(src_dir),
        "deltas_path": str(deltas_path),
        "output_dir": str(out_dir),
        "shard_file": shard_name,
        "copied_files": copied_files,
        "total_tensors": len(reloaded),
        "total_params": total_params,
        "modified_tensor_count": len(modified_tensors),
        "modified_params": modified_params,
        "pct_modified_params": round(100.0 * modified_params / max(1, total_params), 6),
        "total_delta_frobenius_norm": round(math.sqrt(sq_frob_sum), 6),
        "modified_tensors": modified_tensors,
    }
    (out_dir / "export_verification.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Export patched weights into standard HF safetensors directory.")
    ap.add_argument("--deltas", required=True, help="Path to track2_best_deltas.pt")
    ap.add_argument("--output-dir", required=True, help="Destination directory for HF export")
    ap.add_argument("--base-snapshot-dir", default=None, help="Optional override for base HF snapshot directory")
    args = ap.parse_args()

    res = export_patched_hf(
        deltas_path=Path(args.deltas),
        output_dir=Path(args.output_dir),
        base_snapshot_dir=Path(args.base_snapshot_dir) if args.base_snapshot_dir else None,
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
