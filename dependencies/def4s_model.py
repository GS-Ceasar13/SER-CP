"""Reachable four-state evidence routing for the frozen DEF predecessor.

The module operates on per-view evidence logits immediately before the
predecessor's Dempster--Shafer aggregation.  It never opens data or owns an
evaluation protocol; those boundaries belong to the formal runner.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


STATE_ORDER = ("fuse", "preserve", "repair", "suppress")


def _validate_probability(name: str, value: Tensor) -> None:
    if not torch.is_floating_point(value):
        raise ValueError(f"{name} must be floating point")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    if bool(((value < 0.0) | (value > 1.0)).any()):
        raise ValueError(f"{name} must lie in [0, 1]")


def factorized_state_weights(reliability: Tensor, support: Tensor) -> Dict[str, Tensor]:
    """Return the exact reliability-by-support Bernoulli partition."""

    if reliability.shape != support.shape:
        raise ValueError("reliability/support shape mismatch")
    _validate_probability("reliability", reliability)
    _validate_probability("support", support)
    return {
        "fuse": reliability * support,
        "preserve": reliability * (1.0 - support),
        "repair": (1.0 - reliability) * support,
        "suppress": (1.0 - reliability) * (1.0 - support),
    }


class FourStateEvidenceRouter(nn.Module):
    """Identity-initialized bounded evidence router.

    ``router='factorized'`` is DEF-4S. ``router='free'`` is its
    parameter-matched four-way softmax mechanism control.
    """

    def __init__(
        self,
        *,
        hidden: int,
        classes: int,
        router: str = "factorized",
        max_delta: float = 0.25,
    ) -> None:
        super().__init__()
        if type(hidden) is not int or hidden <= 0:
            raise ValueError("hidden must be a positive integer")
        if type(classes) is not int or classes < 2:
            raise ValueError("classes must be an integer >= 2")
        if router not in {"factorized", "free"}:
            raise ValueError("router must be factorized or free")
        if not math.isfinite(max_delta) or not 0.0 < max_delta <= 1.0:
            raise ValueError("max_delta must lie in (0, 1]")

        self.hidden = hidden
        self.classes = classes
        self.router = router
        self.max_delta = float(max_delta)

        route_width = max(8, hidden // 2)
        self.route_head = nn.Sequential(
            nn.Linear(2 * hidden + 2, route_width),
            nn.GELU(),
            nn.Linear(route_width, 4),
        )
        residual_width = max(8, 2 * classes)
        self.residual_head = nn.Sequential(
            nn.Linear(2 * classes + 6, residual_width),
            nn.GELU(),
            nn.Linear(residual_width, classes),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    @staticmethod
    def action_targets(own_evidence: Tensor, consensus_evidence: Tensor) -> Tensor:
        if own_evidence.shape != consensus_evidence.shape:
            raise ValueError("own/consensus evidence shape mismatch")
        fuse = 0.5 * (own_evidence + consensus_evidence)
        preserve = own_evidence
        repair = consensus_evidence
        suppress = torch.zeros_like(own_evidence)
        return torch.stack([fuse, preserve, repair, suppress], dim=-2)

    def _validate_inputs(self, logits: Tensor, encoded: Tensor, mask: Tensor) -> None:
        if logits.ndim != 3 or logits.shape[-1] != self.classes:
            raise ValueError("logits must have shape [batch, views, classes]")
        if encoded.ndim != 3 or encoded.shape[:2] != logits.shape[:2] or encoded.shape[-1] != self.hidden:
            raise ValueError("encoded must have shape [batch, views, hidden]")
        if not torch.is_floating_point(logits) or not torch.isfinite(logits).all():
            raise ValueError("logits must be finite floating point")
        if not torch.is_floating_point(encoded) or not torch.isfinite(encoded).all():
            raise ValueError("encoded must be finite floating point")
        if mask.shape != logits.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("mask must be boolean [batch, views]")
        if not bool(mask.any(dim=1).all()):
            raise ValueError("each sample needs at least one present view")
        if logits.device != encoded.device or logits.device != mask.device:
            raise ValueError("logits, encoded, and mask must share a device")

    def _leave_one_view_out(
        self, evidence: Tensor, encoded: Tensor, mask: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        alpha = evidence + 1.0
        strength = alpha.sum(dim=-1)
        reliability = (1.0 - self.classes / strength).clamp(0.0, 1.0)
        reliability = torch.where(mask, reliability, torch.zeros_like(reliability))

        views = evidence.shape[1]
        not_self = ~torch.eye(views, dtype=torch.bool, device=evidence.device)
        source_weight = reliability[:, None, :] * mask[:, None, :].to(evidence.dtype)
        source_weight = source_weight * not_self[None, :, :].to(evidence.dtype)
        denominator = source_weight.sum(dim=-1, keepdim=True)
        has_other = denominator.squeeze(-1) > 0.0

        consensus_evidence = torch.einsum("nvs,nsc->nvc", source_weight, evidence)
        consensus_encoded = torch.einsum("nvs,nsh->nvh", source_weight, encoded)
        consensus_evidence = consensus_evidence / denominator.clamp_min(torch.finfo(evidence.dtype).eps)
        consensus_encoded = consensus_encoded / denominator.clamp_min(torch.finfo(encoded.dtype).eps)
        consensus_evidence = torch.where(
            has_other[..., None], consensus_evidence, torch.zeros_like(consensus_evidence)
        )
        consensus_encoded = torch.where(
            has_other[..., None], consensus_encoded, torch.zeros_like(consensus_encoded)
        )
        return reliability, consensus_evidence, consensus_encoded, has_other

    def forward(
        self, logits: Tensor, encoded: Tensor, mask: Tensor
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        self._validate_inputs(logits, encoded, mask)
        evidence = F.softplus(logits)
        reliability, consensus_evidence, consensus_encoded, has_other = self._leave_one_view_out(
            evidence, encoded, mask
        )

        probability = (evidence + 1.0) / (evidence + 1.0).sum(dim=-1, keepdim=True)
        consensus_probability = (consensus_evidence + 1.0) / (
            consensus_evidence + 1.0
        ).sum(dim=-1, keepdim=True)
        support = 1.0 - 0.5 * (probability - consensus_probability).abs().sum(dim=-1)
        support = support.clamp(0.0, 1.0)
        support = torch.where(has_other & mask, support, torch.zeros_like(support))

        route_features = torch.cat(
            [encoded, consensus_encoded, reliability[..., None], support[..., None]], dim=-1
        )
        route_logits = self.route_head(route_features)
        if self.router == "factorized":
            state_map = factorized_state_weights(reliability, support)
            state_weights = torch.stack([state_map[name] for name in STATE_ORDER], dim=-1)
        else:
            state_weights = torch.softmax(route_logits, dim=-1)
        state_weights = torch.where(mask[..., None], state_weights, torch.zeros_like(state_weights))

        actions = self.action_targets(evidence, consensus_evidence)
        routed_target = (state_weights[..., None] * actions).sum(dim=-2)
        residual_features = torch.cat(
            [
                evidence,
                routed_target,
                reliability[..., None],
                support[..., None],
                route_logits,
            ],
            dim=-1,
        )
        delta_logits = self.max_delta * torch.tanh(self.residual_head(residual_features))
        delta_logits = torch.where(mask[..., None], delta_logits, torch.zeros_like(delta_logits))
        modified_logits = logits + delta_logits
        if not torch.isfinite(modified_logits).all():
            raise ValueError("non-finite routed evidence logits")

        diagnostics = {
            "reliability": reliability,
            "support": support,
            "state_weights": state_weights,
            "routed_target_evidence": routed_target,
            "delta_logits": delta_logits,
            "has_other_view": has_other,
            "route_logits": route_logits,
        }
        return modified_logits, diagnostics

