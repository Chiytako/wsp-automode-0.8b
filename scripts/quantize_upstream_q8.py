#!/usr/bin/env python3
"""Protect upstream transformer blocks in Q8_0 (leaving edited & downstream layers in K-quant).

Sparse closed-form SwiGLU edits induce high input sensitivity ``||grad_h g_j||_2``, causing
upstream quantization drift ``Delta h`` from ``blk.0 .. blk.(edit_layer-1)`` to be strongly
amplified even when the edited tensors themselves stay in BF16. Upgrading upstream blocks
to Q8_0 (``llama-quantize --tensor-type-file``) stabilizes inputs across hybrid DeltaNet,
Qwen, and Llama architectures while preserving compact K-quant storage on downstream weights.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

_ROOT = Path(__file__).resolve().parent.parent
_GGUF_PY = _ROOT / "third_party" / "llama.cpp" / "gguf-py"
if _GGUF_PY.exists() and str(_GGUF_PY) not in sys.path:
    sys.path.insert(0, str(_GGUF_PY))

import gguf  # noqa: E402


DEFAULT_QUANTIZE_BIN = _ROOT / "third_party" / "llama.cpp" / "build" / "bin" / "llama-quantize"


def parse_block_index(tensor_name: str) -> int | None:
    """Extract integer layer index from GGUF tensor names like ``blk.14.ffn_gate.weight``."""
    if not tensor_name.startswith("blk."):
        return None
    parts = tensor_name.split(".")
    if len(parts) < 3:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def plan_upstream_overrides(
    tensor_names_and_types: Iterable[tuple[str, str]],
    edit_layer: int,
    upstream_quant: str = "Q8_0",
    include_embd: bool = False,
    include_edit_layer: bool = False,
) -> dict[str, str]:
    """Return GGUF ``--tensor-type-file`` overrides protecting upstream blocks (and optionally ``edit_layer``)."""
    if edit_layer <= 0:
        raise ValueError(f"edit_layer must be positive, got {edit_layer}")
    max_blk = edit_layer if include_edit_layer else edit_layer - 1
    target_q = upstream_quant.strip().lower()
    overrides: dict[str, str] = {}
    for name, ttype in tensor_names_and_types:
        if str(ttype).upper() == "F32":
            continue
        blk = parse_block_index(name)
        if blk is not None and 0 <= blk <= max_blk:
            overrides[name] = target_q
        elif include_embd and name in ("token_embd.weight", "output.weight"):
            overrides[name] = target_q
    return overrides


def estimate_override_overhead_mib(
    reader: gguf.GGUFReader,
    overrides: dict[str, str],
    base_bits_per_weight: float = 4.85,
    upstream_bits_per_weight: float = 8.50,
) -> float:
    """Roughly estimate added MiB over uniform base quantization for logging/planning."""
    delta_bits = max(0.0, upstream_bits_per_weight - base_bits_per_weight)
    total_elems = sum(int(t.n_elements) for t in reader.tensors if t.name in overrides)
    return (total_elems * delta_bits / 8.0) / (1024.0 ** 2)


def quantize_with_upstream_protection(
    input_bf16_gguf: Path,
    output_gguf: Path,
    edit_layer: int,
    base_quant: str = "Q4_K_M",
    upstream_quant: str = "Q8_0",
    quantize_bin: Path = DEFAULT_QUANTIZE_BIN,
    include_embd: bool = False,
    include_edit_layer: bool = False,
    dry_run: bool = False,
) -> dict[str, object]:
    """Inspect ``input_bf16_gguf`` and invoke ``llama-quantize`` with upstream overrides."""
    reader = gguf.GGUFReader(str(input_bf16_gguf))
    pairs = [(t.name, str(t.tensor_type.name)) for t in reader.tensors]
    overrides = plan_upstream_overrides(
        pairs,
        edit_layer=edit_layer,
        upstream_quant=upstream_quant,
        include_embd=include_embd,
        include_edit_layer=include_edit_layer,
    )
    est_mib = estimate_override_overhead_mib(reader, overrides)
    extra_flags: list[str] = []
    if include_embd:
        uq = upstream_quant.strip().lower()
        extra_flags = ["--token-embedding-type", uq, "--output-tensor-type", uq]

    max_blk = edit_layer if include_edit_layer else edit_layer - 1
    info = {
        "input_bf16_gguf": str(input_bf16_gguf),
        "output_gguf": str(output_gguf),
        "edit_layer": edit_layer,
        "protected_blocks": f"blk.0..{max_blk}",
        "protected_tensor_count": len(overrides),
        "base_quant": base_quant,
        "upstream_quant": upstream_quant,
        "include_embd": include_embd,
        "include_edit_layer": include_edit_layer,
        "estimated_added_mib": round(est_mib, 2),
        "dry_run": dry_run,
    }
    print(
        f"[upstream-q8] protecting {len(overrides)} tensors in blk.0..{max_blk} "
        f"at {upstream_quant} (base={base_quant}, embd={'Q8' if include_embd else 'base'}, "
        f"est. +{est_mib:.2f} MiB)",
        flush=True,
    )
    if dry_run:
        return info

    output_gguf.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix="_upstream_types.txt", delete=False, encoding="utf-8") as tf:
        tf.write("\n".join(f"{k}={v}" for k, v in sorted(overrides.items())) + "\n")
        tt_path = Path(tf.name)
    try:
        cmd = [
            str(quantize_bin),
            *extra_flags,
            "--tensor-type-file",
            str(tt_path),
            str(input_bf16_gguf),
            str(output_gguf),
            base_quant,
        ]
        subprocess.run(cmd, check=True)
    finally:
        tt_path.unlink(missing_ok=True)
    info["output_mib"] = round(output_gguf.stat().st_size / (1024.0 ** 2), 2)
    print(f"[upstream-q8] wrote {output_gguf} ({info['output_mib']:.2f} MiB)", flush=True)
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description="Quantize a BF16/F16 GGUF with upstream blocks protected in Q8_0.")
    ap.add_argument("--input-bf16-gguf", type=Path, required=True)
    ap.add_argument("--output-gguf", type=Path, required=True)
    ap.add_argument("--edit-layer", type=int, required=True, help="First sparse-edited layer index (protects blk.0..edit_layer-1).")
    ap.add_argument("--base-quant", type=str, default="Q4_K_M")
    ap.add_argument("--upstream-quant", type=str, default="Q8_0")
    ap.add_argument("--quantize-bin", type=Path, default=DEFAULT_QUANTIZE_BIN)
    ap.add_argument("--include-embd", action="store_true", help="Also promote token_embd/output.weight to upstream precision.")
    ap.add_argument(
        "--include-edit-layer",
        action="store_true",
        help="Also protect blk.<edit_layer> alongside blk.0..edit_layer-1 (cancels edited-layer weight bias).",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    quantize_with_upstream_protection(
        input_bf16_gguf=args.input_bf16_gguf,
        output_gguf=args.output_gguf,
        edit_layer=args.edit_layer,
        base_quant=args.base_quant,
        upstream_quant=args.upstream_quant,
        quantize_bin=args.quantize_bin,
        include_embd=args.include_embd,
        include_edit_layer=args.include_edit_layer,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
