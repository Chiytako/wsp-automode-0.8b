"""Single-pass Qwen3.5-0.8B slot-logit runner and weight-diff accounting utilities.

Enforces 4GB VRAM safety:
- Loads AutoModelForCausalLM (Qwen3_5ForCausalLM) in bfloat16 (1459 MB VRAM)
- Runs backbone model.model(..., use_cache=False) under torch.inference_mode()
- Projects ONLY the final answer slot hidden vector through model.lm_head
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from automode.questions import LETTERS, Q_BLOCK_BINARY, build_oracle_prompt, build_plain_prompt

MODEL_ID = "Qwen/Qwen3.5-0.8B"


@dataclass
class ScoreResult:
    """Single-pass option-letter readout result."""

    prompt_tokens: int
    option_letters: list[str]
    option_logits: list[float]
    option_probs: list[float]
    pred_index: int
    pred_letter: str
    p_yes: float | None = None
    p_no: float | None = None
    logit_margin: float | None = None


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class QwenDeciderRunner:
    """Wrapper for Qwen3.5-0.8B single-pass slot readout on CUDA or CPU."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if model_path is None:
            resolved = Path(snapshot_download(MODEL_ID, local_files_only=True))
        else:
            resolved = Path(model_path)
        self.model_dir = resolved
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.dtype = dtype if device.startswith("cuda") else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(resolved)
        self.model = AutoModelForCausalLM.from_pretrained(resolved, dtype=self.dtype)
        self.model = self.model.to(self.device).eval()

        self.letter_token_ids: dict[str, int] = {}
        for letter in LETTERS:
            ids = self.tokenizer.encode(letter, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(f"Expected single token for option letter {letter!r}, got {ids}")
            self.letter_token_ids[letter] = int(ids[0])

    def encode_prompt(self, prompt_text: str) -> torch.Tensor:
        ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    @torch.inference_mode()
    def score_prompt(
        self,
        prompt_text: str,
        option_letters: list[str],
        yes_idx: int | None = None,
        no_idx: int | None = None,
    ) -> ScoreResult:
        input_ids = self.encode_prompt(prompt_text)
        n_tok = int(input_ids.shape[1])

        hidden = self.model.model(input_ids=input_ids, use_cache=False).last_hidden_state
        slot_hidden = hidden[:, -1:, :]
        vocab_logits = self.model.lm_head(slot_hidden)[0, 0, :].float()

        opt_ids = [self.letter_token_ids[l] for l in option_letters]
        opt_logits_t = vocab_logits[opt_ids]
        opt_probs_t = torch.softmax(opt_logits_t, dim=-1)

        opt_logits = [float(x) for x in opt_logits_t.tolist()]
        opt_probs = [float(x) for x in opt_probs_t.tolist()]
        pred_idx = max(range(len(opt_probs)), key=opt_probs.__getitem__)

        p_yes: float | None = None
        p_no: float | None = None
        margin: float | None = None
        if yes_idx is not None and no_idx is not None:
            p_yes = opt_probs[yes_idx]
            p_no = opt_probs[no_idx]
            margin = opt_logits[yes_idx] - opt_logits[no_idx]

        return ScoreResult(
            prompt_tokens=n_tok,
            option_letters=list(option_letters),
            option_logits=opt_logits,
            option_probs=opt_probs,
            pred_index=pred_idx,
            pred_letter=option_letters[pred_idx],
            p_yes=p_yes,
            p_no=p_no,
            logit_margin=margin,
        )

    def score_binary_paired(
        self,
        context_text: str,
        mode: str = "plain",
        question_spec: dict[str, Any] = Q_BLOCK_BINARY,
        second_reading_band: tuple[float, float] = (0.30, 0.70),
    ) -> dict[str, Any]:
        """Score a binary block/allow context in both normal and reversed option orders."""
        builder = build_oracle_prompt if mode == "oracle" else build_plain_prompt
        p_norm = builder(context_text, question_spec=question_spec, reverse_options=False)
        p_rev = builder(context_text, question_spec=question_spec, reverse_options=True)

        res_norm = self.score_prompt(
            p_norm.text,
            p_norm.option_letters,
            yes_idx=p_norm.yes_idx,
            no_idx=p_norm.no_idx,
        )
        res_rev = self.score_prompt(
            p_rev.text,
            p_rev.option_letters,
            yes_idx=p_rev.yes_idx,
            no_idx=p_rev.no_idx,
        )

        p_block_norm = float(res_norm.p_yes or 0.0)
        p_block_rev = float(res_rev.p_yes or 0.0)
        m_norm = float(res_norm.logit_margin or 0.0)
        m_rev = float(res_rev.logit_margin or 0.0)

        m_avg = 0.5 * (m_norm + m_rev)
        p_avg = sigmoid(m_avg)

        lo, hi = second_reading_band
        triggered_sr = lo <= p_block_norm <= hi
        m_sr = m_avg if triggered_sr else m_norm
        p_sr = sigmoid(m_sr)

        pred_norm = "block" if p_block_norm >= 0.5 else "allow"
        pred_rev = "block" if p_block_rev >= 0.5 else "allow"

        return {
            "prompt_tokens_normal": res_norm.prompt_tokens,
            "prompt_tokens_reversed": res_rev.prompt_tokens,
            "p_block": p_block_norm,
            "p_block_reversed": p_block_rev,
            "p_block_avg": p_avg,
            "p_block_second_reading": p_sr,
            "second_reading_triggered": triggered_sr,
            "margin_normal": m_norm,
            "margin_reversed": m_rev,
            "margin_avg": m_avg,
            "pred_normal": pred_norm,
            "pred_reversed": pred_rev,
            "pred_avg": "block" if p_avg >= 0.5 else "allow",
            "pred_second_reading": "block" if p_sr >= 0.5 else "allow",
            "flipped": pred_norm != pred_rev,
            "order_abs_margin_diff": abs(m_norm - m_rev),
        }


def capture_cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Snapshot all model parameters in host RAM (CPU) for rollback and diff stats."""
    return {k: v.detach().cpu().clone() for k, v in model.named_parameters()}


def restore_cpu_state_dict(model: torch.nn.Module, cpu_state: dict[str, torch.Tensor]) -> None:
    """Restore model parameters from a host RAM snapshot in-place."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(cpu_state[name].to(device=param.device, dtype=param.dtype))


def compute_weight_diff_stats(
    model: torch.nn.Module,
    base_cpu_state: dict[str, torch.Tensor],
    atol: float = 1e-7,
) -> dict[str, Any]:
    """Measure % modified params, modified tensor names, and Frobenius norm delta."""
    total_params = 0
    modified_params = 0
    modified_tensors: list[dict[str, Any]] = []
    sq_frob_sum = 0.0

    with torch.no_grad():
        for name, param in model.named_parameters():
            n = param.numel()
            total_params += n
            cur = param.detach().cpu().float()
            base = base_cpu_state[name].float()
            diff = cur - base
            abs_diff = diff.abs()
            changed_mask = abs_diff > atol
            n_changed = int(changed_mask.sum().item())
            if n_changed > 0:
                frob = float(torch.linalg.norm(diff).item())
                sq_frob_sum += frob * frob
                modified_params += n_changed
                modified_tensors.append(
                    {
                        "name": name,
                        "shape": list(param.shape),
                        "numel": n,
                        "modified_elements": n_changed,
                        "pct_elements": round(100.0 * n_changed / n, 4),
                        "delta_frobenius_norm": round(frob, 6),
                        "max_abs_delta": round(float(abs_diff.max().item()), 6),
                    }
                )

    pct_total = 100.0 * modified_params / max(1, total_params)
    return {
        "total_params": total_params,
        "modified_params": modified_params,
        "pct_modified_params": round(pct_total, 6),
        "modified_tensor_count": len(modified_tensors),
        "total_delta_frobenius_norm": round(math.sqrt(sq_frob_sum), 6),
        "modified_tensors": modified_tensors,
    }
