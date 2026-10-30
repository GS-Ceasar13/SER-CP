"""Isolated dual-path prediction; no patient I/O or deployment-only metadata.

The original factorized router and classical DS method are imported unchanged.
Posterior mixtures deliberately do not expose a fictitious final Dirichlet.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.util
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

ARMS = ('faithful_ds', 'shared_capacity_ds', 'joint_only', 'global_posterior',
        'patient_posterior_natural', 'patient_posterior_weighted',
        'patient_direction_natural', 'patient_direction_weighted')
SOURCE_PINS = {
    'def4s_model.py': '1d2d3a606a7564dcc5f2165c8665b8b95be521a221a1d3943a1c4810ae17e4f7',
    'train_legacy256_paired.py': '2b463259161dc47f63405ce938ca93f4899bc87f30c70f3dcba0abce2f80e10f',
}


@lru_cache(maxsize=4)
def _original_module(name):
    here = Path(__file__).resolve().parent
    local = here / 'dependencies' / name
    if local.is_file():
        path = local
    else:
        group = 'def_4s_evidence_router_v1' if name == 'def4s_model.py' else 'def_frozen_before_evitok_v1'
        path = here.parents[1] / 'automation_20260822' / group / 'source' / name
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != SOURCE_PINS[name]:
        raise ValueError(f'original source SHA mismatch: {name}')
    spec = importlib.util.spec_from_file_location('dualpath_original_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mix_paths(ds_alpha, joint_alpha, gate, operator='posterior'):
    """Mix full posterior means, or the matched direction-times-strength control."""
    if gate.shape != (ds_alpha.shape[0], 1) or not torch.isfinite(gate).all() or ((gate < 0) | (gate > 1)).any():
        raise ValueError('gate must be finite [batch, 1] within [0, 1]')
    pds = ds_alpha / ds_alpha.sum(-1, keepdim=True)
    pj = joint_alpha / joint_alpha.sum(-1, keepdim=True)
    if operator == 'posterior':
        return (1-gate)*pds + gate*pj
    if operator != 'direction':
        raise ValueError('unknown final operator')
    eds, ej = ds_alpha-1, joint_alpha-1
    sds, sj = eds.sum(-1, keepdim=True), ej.sum(-1, keepdim=True)
    eps = torch.finfo(ds_alpha.dtype).eps
    qds, qj = eds/sds.clamp_min(eps), ej/sj.clamp_min(eps)
    qds = torch.where(sds > 0, qds, torch.full_like(qds, 1/qds.shape[-1]))
    qj = torch.where(sj > 0, qj, torch.full_like(qj, 1/qj.shape[-1]))
    alpha = 1 + ((1-gate)*sds+gate*sj)*((1-gate)*qds+gate*qj)
    return alpha/alpha.sum(-1, keepdim=True)


class DualPath(nn.Module):
    def __init__(self, arm, input_dim, views, classes, hidden=128, *, dependencies=None):
        super().__init__()
        if arm not in ARMS:
            raise ValueError('unknown arm')
        if any(type(v) is not int or v <= 0 for v in (input_dim, views, classes, hidden)) or classes < 2 or hidden % 4:
            raise ValueError('invalid dimensions; hidden must be divisible by four for original DEF constructor')
        self.arm, self.input_dim, self.views, self.classes, self.C = arm, input_dim, views, classes, classes
        self.weighted_final = arm.endswith('_weighted')
        dependencies = dependencies or {}
        predecessor = dependencies.get('predecessor_class') or _original_module('train_legacy256_paired.py').DEF
        router = dependencies.get('router_class') or _original_module('def4s_model.py').FourStateEvidenceRouter
        # Retain only the reachable original topology, not unused experimental modules.
        original = predecessor(views, input_dim, classes=classes, hidden=hidden,
                               combine='gate', path='view', vagg='ds', fdiscount=True)
        self.enc, self.head_v, self.joint = original.enc, original.head_v, original.joint
        self._ds_two = predecessor.ds_two
        if arm == 'shared_capacity_ds':
            target = sum(p.numel() for p in self.enc.parameters()) + sum(p.numel() for p in self.head_v.parameters())
            fixed = hidden + hidden*classes + classes
            width = max(1, round((target-fixed)/(input_dim+1+hidden)))
            encoder = nn.Sequential(nn.Linear(input_dim, width), nn.ReLU(), nn.Dropout(.1),
                                    nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(.1))
            head = nn.Linear(hidden, classes)
            actual = sum(p.numel() for p in encoder.parameters())+sum(p.numel() for p in head.parameters())
            if abs(actual-target)/target > .05:
                raise ValueError('shared capacity cannot match view-head parameters within five percent')
            self.enc = nn.ModuleList([encoder]*views)
            self.head_v = nn.ModuleList([head]*views)
        self.router = router(hidden=hidden, classes=classes, router='factorized', max_delta=.25)
        if arm == 'global_posterior':
            self.global_logit = nn.Parameter(torch.zeros(()))
        elif arm.startswith('patient_'):
            # Same small scorer for all four cells; inputs are deployable only.
            self.gate_scorer = nn.Sequential(nn.Linear(2*classes+views+5, 16), nn.GELU(), nn.Linear(16, 1))
            nn.init.zeros_(self.gate_scorer[-1].weight)
            nn.init.zeros_(self.gate_scorer[-1].bias)

    def parameter_counts(self):
        count = lambda module: sum(p.numel() for p in module.parameters())
        # ModuleList.parameters deduplicates shared references within each list.
        return {'view_heads': count(self.enc)+count(self.head_v), 'joint': count(self.joint),
                'router': count(self.router), 'total': count(self)}

    def forward(self, features, present, anchor_probabilities=None, force_noop=False):
        if force_noop:
            if anchor_probabilities is None or anchor_probabilities.ndim != 2 or anchor_probabilities.shape != (features.shape[0], self.classes):
                raise ValueError('exact no-op needs external anchor probabilities')
            if not torch.isfinite(anchor_probabilities).all() or (anchor_probabilities < 0).any() or not torch.allclose(anchor_probabilities.sum(-1), torch.ones_like(anchor_probabilities[:, 0]), atol=1e-5):
                raise ValueError('invalid external anchor probabilities')
            return {'probabilities': anchor_probabilities, 'exact_noop': True, 'arm': self.arm, 'weighted_final': self.weighted_final}
        if features.ndim != 3 or features.shape[1:] != (self.views, self.input_dim):
            raise ValueError('features need [batch, views, input_dim]')
        if not torch.is_floating_point(features) or not torch.isfinite(features).all():
            raise ValueError('features must be finite floating point')
        if present.shape != features.shape[:2] or present.dtype != torch.bool or not present.any(-1).all():
            raise ValueError('present must be boolean and include a present view per patient')
        if present.device != features.device:
            raise ValueError('features and present must share device')
        hidden = torch.stack([self.enc[v](features[:, v]) for v in range(self.views)], 1)
        logits = torch.stack([self.head_v[v](hidden[:, v]) for v in range(self.views)], 1)
        routed, diagnostics = self.router(logits, hidden, present)
        view_alpha = torch.where(present[..., None], 1+F.softplus(routed), torch.ones_like(routed))
        ds_alpha = view_alpha[:, 0]
        for v in range(1, self.views):
            ds_alpha = self._ds_two(self, ds_alpha, view_alpha[:, v], discount=False)
        joint_alpha = 1 + F.softplus(self.joint((hidden*present[..., None]).flatten(1)))
        pds, pj = ds_alpha/ds_alpha.sum(-1, keepdim=True), joint_alpha/joint_alpha.sum(-1, keepdim=True)
        if self.arm in ('faithful_ds', 'shared_capacity_ds'):
            gate = features.new_zeros((features.shape[0], 1))
        elif self.arm == 'joint_only':
            gate = features.new_ones((features.shape[0], 1))
        elif self.arm == 'global_posterior':
            gate = self.global_logit.sigmoid().expand(features.shape[0], 1)
        else:
            denominator = present.sum(-1).clamp_min(1)
            reliability = (diagnostics['reliability']*present).sum(-1)/denominator
            support = (diagnostics['support']*present).sum(-1)/denominator
            uncertainty_ds = self.classes/ds_alpha.sum(-1)
            uncertainty_joint = self.classes/joint_alpha.sum(-1)
            disagreement = .5*(pds-pj).abs().sum(-1)
            summaries = torch.stack([disagreement, reliability, support, uncertainty_ds, uncertainty_joint], -1)
            gate = self.gate_scorer(torch.cat([pds, pj, present.to(features.dtype), summaries], -1)).sigmoid()
        probabilities = mix_paths(ds_alpha, joint_alpha, gate, 'direction' if 'direction' in self.arm else 'posterior')
        return {'probabilities': probabilities, 'ds_alpha': ds_alpha, 'joint_alpha': joint_alpha,
                'view_alpha': view_alpha, 'gate': gate, 'diagnostics': diagnostics,
                'arm': self.arm, 'weighted_final': self.weighted_final, 'exact_noop': False}


def build_model(arm, input_dim, views, classes, hidden=128, *, dependencies=None):
    return DualPath(arm, input_dim, views, classes, hidden, dependencies=dependencies)
