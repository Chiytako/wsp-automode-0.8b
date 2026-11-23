"""Order-equivariant, contrastive margin, calibration, and replay losses for Phase 5."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_binary_family_loss(
    margins_norm: torch.Tensor,
    margins_rev: torch.Tensor,
    labels: torch.Tensor,
    pair_indices: list[tuple[int, int]],
    is_synthetic: bool = True,
    label_smooth: float | None = None,
    lambda_ord: float = 0.35,
    lambda_rank: float = 0.40,
    rank_margin: float = 4.0,
    logit_cap: float = 6.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute order-equivariant BCE + pairwise margin ranking + logit-cap loss on one family.

    Args:
        margins_norm: `(K,)` tensor of `logit_yes - logit_no` in normal option order `(A)=no,(B)=yes`.
        margins_rev: `(K,)` tensor of `logit_yes - logit_no` in reversed option order `(A)=yes,(B)=no`.
        labels: `(K,)` float/int tensor in `{0, 1}` where `1 == block`, `0 == allow`.
        pair_indices: list of `(pos_idx, neg_idx)` pairs within the family expecting `pos > neg`.
    """
    m_norm = margins_norm.float()
    m_rev = margins_rev.float()
    y = labels.float().to(device=m_norm.device)

    eps = float(label_smooth) if label_smooth is not None else (0.02 if is_synthetic else 0.01)
    y_smooth = y * (1.0 - eps) + 0.5 * eps

    loss_bce = 0.5 * (
        F.binary_cross_entropy_with_logits(m_norm, y_smooth)
        + F.binary_cross_entropy_with_logits(m_rev, y_smooth)
    )
    loss_ord = float(lambda_ord) * torch.mean((m_norm - m_rev) ** 2)

    m_avg = 0.5 * (m_norm + m_rev)
    if pair_indices:
        diffs = torch.stack([m_avg[pos] - m_avg[neg] for pos, neg in pair_indices])
        loss_rank = float(lambda_rank) * torch.mean(F.softplus(float(rank_margin) - diffs))
    else:
        loss_rank = m_avg.new_zeros(())

    loss_cap = 0.01 * torch.mean(F.relu(m_avg.abs() - float(logit_cap)) ** 2)
    total = loss_bce + loss_ord + loss_rank + loss_cap
    stats = {
        "total": float(total.detach().item()),
        "bce": float(loss_bce.detach().item()),
        "ord": float(loss_ord.detach().item()),
        "rank": float(loss_rank.detach().item()),
        "cap": float(loss_cap.detach().item()),
    }
    return total, stats


def compute_replay_loss(
    logits_norm: torch.Tensor,
    logits_rev: torch.Tensor,
    gold_idx: int,
    base_probs_norm: torch.Tensor | None = None,
    base_probs_rev: torch.Tensor | None = None,
    lambda_ord: float = 0.15,
    lambda_kl: float = 0.10,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute multi-option CE + option-reversal symmetry + stock KL preservation loss.

    In reversed option order (`reverse_options=True`), option index `k` maps to `(K - 1) - k`
    relative to normal order, so `gold_rev = (K - 1) - gold_idx` and `torch.flip(logits_rev, [0])`
    aligns slot-for-slot with `logits_norm`.
    """
    ln = logits_norm.float()
    lr = logits_rev.float()
    num_opts = int(ln.shape[-1])
    if lr.shape[-1] != num_opts:
        raise ValueError(f"Option count mismatch: norm={num_opts} vs rev={lr.shape[-1]}")

    gold_norm_t = torch.tensor([int(gold_idx)], dtype=torch.long, device=ln.device)
    gold_rev_t = torch.tensor([(num_opts - 1) - int(gold_idx)], dtype=torch.long, device=ln.device)

    loss_ce = 0.5 * (
        F.cross_entropy(ln.unsqueeze(0), gold_norm_t)
        + F.cross_entropy(lr.unsqueeze(0), gold_rev_t)
    )

    logp_norm = F.log_softmax(ln, dim=-1)
    logp_rev = F.log_softmax(lr, dim=-1)
    logp_rev_aligned = torch.flip(logp_rev, dims=[0])
    loss_ord = float(lambda_ord) * F.mse_loss(logp_norm, logp_rev_aligned)

    if base_probs_norm is not None and base_probs_rev is not None and lambda_kl > 0.0:
        bp_n = base_probs_norm.float().to(device=ln.device).clamp_min(1e-8)
        bp_n = bp_n / bp_n.sum()
        bp_r = base_probs_rev.float().to(device=lr.device).clamp_min(1e-8)
        bp_r = bp_r / bp_r.sum()
        kl_n = F.kl_div(logp_norm, bp_n, reduction="sum")
        kl_r = F.kl_div(logp_rev, bp_r, reduction="sum")
        loss_kl = 0.5 * float(lambda_kl) * (kl_n + kl_r)
    else:
        loss_kl = ln.new_zeros(())

    total = loss_ce + loss_ord + loss_kl
    stats = {
        "total": float(total.detach().item()),
        "ce": float(loss_ce.detach().item()),
        "ord": float(loss_ord.detach().item()),
        "kl": float(loss_kl.detach().item()),
    }
    return total, stats
