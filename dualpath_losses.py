"""Final posterior NLL, unweighted branch supervision, and soft patient damage.

Donor labels are never used. Only the training auxiliary mask can exclude an
isolated donor view; inference still sees the identical replaced inputs.
"""
import math
import torch


def compute_loss(output, labels, present, *, aux_present=None, class_weights=None,
                 anchor_probabilities=None, auxiliary_weight=.3, safety_weight=.5):
    probability = output['probabilities']
    if labels.ndim != 1 or labels.shape[0] != probability.shape[0] or labels.dtype != torch.long:
        raise ValueError('labels must be integer [batch]')
    if labels.device != probability.device or present.device != probability.device:
        raise ValueError('loss tensors must share device')
    if (labels < 0).any() or (labels >= probability.shape[-1]).any():
        raise ValueError('labels outside class range')
    if not torch.isfinite(probability).all() or (probability <= 0).any() or not torch.allclose(probability.sum(-1), torch.ones_like(probability[:, 0]), atol=1e-5):
        raise ValueError('invalid final probabilities')
    if present.shape != output['view_alpha'].shape[:2] or present.dtype != torch.bool:
        raise ValueError('present must match view_alpha')
    if aux_present is None:
        aux_present = present
    if aux_present.shape != present.shape or aux_present.dtype != torch.bool or aux_present.device != present.device or (aux_present & ~present).any():
        raise ValueError('aux_present must be subset of present')
    if any(not math.isfinite(v) or v < 0 for v in (auxiliary_weight, safety_weight)):
        raise ValueError('loss coefficients must be finite and nonnegative')
    index = torch.arange(labels.shape[0], device=labels.device)
    final_nll = -probability[index, labels].log()
    if output.get('weighted_final', False):
        if class_weights is None or class_weights.shape != (probability.shape[-1],) or class_weights.device != probability.device or not torch.isfinite(class_weights).all() or (class_weights <= 0).any():
            raise ValueError('weighted arm needs positive finite class_weights')
        final = (final_nll*class_weights[labels]).mean()
    else:
        final = final_nll.mean()
    alpha_v, alpha_j = output['view_alpha'], output['joint_alpha']
    if alpha_v.shape != (*present.shape, probability.shape[-1]) or alpha_j.shape != probability.shape:
        raise ValueError('invalid branch alpha shapes')
    for alpha in (alpha_v, alpha_j):
        if alpha.device != probability.device or not torch.isfinite(alpha).all() or (alpha < 1).any():
            raise ValueError('branch alpha must be finite evidence-plus-one on prediction device')
    pv, pj = alpha_v/alpha_v.sum(-1, keepdim=True), alpha_j/alpha_j.sum(-1, keepdim=True)
    view_nll = -pv.gather(-1, labels[:, None, None].expand(-1, pv.shape[1], 1)).squeeze(-1).log()
    per_patient_view = (view_nll*aux_present).sum(-1)/aux_present.sum(-1).clamp_min(1)
    auxiliary = per_patient_view.mean()-pj[index, labels].log().mean()
    safety = probability.new_zeros(())
    if anchor_probabilities is not None and safety_weight:
        anchor = anchor_probabilities.detach()
        if anchor.shape != probability.shape or anchor.device != probability.device or not torch.isfinite(anchor).all() or (anchor <= 0).any() or not torch.allclose(anchor.sum(-1), torch.ones_like(anchor[:, 0]), atol=1e-5):
            raise ValueError('invalid anchor probabilities')
        anchor_nll = -anchor[index, labels].log()
        safety = (final_nll-anchor_nll).clamp_min(0).mean()
    return final + auxiliary_weight*auxiliary + safety_weight*safety
