import importlib
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def api():
    assert importlib.util.find_spec('dualpath_losses') is not None, 'dual-path loss is not implemented'
    return importlib.import_module('dualpath_losses')


def outputs(weighted=False):
    return dict(probabilities=torch.tensor([[.8, .2], [.4, .6]], requires_grad=True),
                view_alpha=torch.tensor([[[4., 1.], [2., 3.]], [[1., 2.], [3., 1.]]], requires_grad=True),
                joint_alpha=torch.tensor([[3., 1.], [2., 5.]], requires_grad=True),
                weighted_final=weighted)


def test_final_natural_nll_exact_formula():
    out = outputs()
    loss = api().compute_loss(out, torch.tensor([0, 1]), torch.ones(2, 2, dtype=torch.bool), auxiliary_weight=0, safety_weight=0)
    torch.testing.assert_close(loss, (-torch.log(torch.tensor(.8))-torch.log(torch.tensor(.6)))/2)


def test_auxiliary_normalizes_per_patient_effective_views():
    out, y = outputs(), torch.tensor([0, 1])
    present = torch.ones(2, 2, dtype=torch.bool)
    auxiliary = torch.tensor([[1, 0], [1, 1]], dtype=torch.bool)
    loss = api().compute_loss(out, y, present, aux_present=auxiliary, safety_weight=0)
    pv = out['view_alpha']/out['view_alpha'].sum(-1, keepdim=True)
    pj = out['joint_alpha']/out['joint_alpha'].sum(-1, keepdim=True)
    final = -out['probabilities'][torch.arange(2), y].log().mean()
    view = (-pv[0, 0, 0].log()+(-pv[1, 0, 1].log()-pv[1, 1, 1].log())/2)/2
    expected = final + .3*(view-pj[torch.arange(2), y].log().mean())
    torch.testing.assert_close(loss, expected)


def test_donor_view_is_not_given_isolated_recipient_gradient():
    out = outputs()
    auxiliary = torch.tensor([[1, 0], [1, 1]], dtype=torch.bool)
    loss = api().compute_loss(out, torch.tensor([0, 1]), torch.ones(2, 2, dtype=torch.bool), aux_present=auxiliary, safety_weight=0)
    loss.backward()
    assert torch.equal(out['view_alpha'].grad[0, 1], torch.zeros(2))
    assert out['view_alpha'].grad[0, 0].abs().sum() > 0
    assert out['joint_alpha'].grad.abs().sum() > 0
    assert out['probabilities'].grad.abs().sum() > 0


def test_no_effective_auxiliary_views_is_finite_zero_view_term():
    out = outputs()
    loss = api().compute_loss(out, torch.tensor([0, 1]), torch.ones(2, 2, dtype=torch.bool), aux_present=torch.zeros(2, 2, dtype=torch.bool), safety_weight=0)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.equal(out['view_alpha'].grad, torch.zeros_like(out['view_alpha']))


def test_class_weights_change_only_final_term_for_weighted_arm():
    y, present, weights = torch.tensor([0, 1]), torch.ones(2, 2, dtype=torch.bool), torch.tensor([2., .5])
    natural, weighted = outputs(), outputs(True)
    normal = api().compute_loss(natural, y, present, class_weights=weights, safety_weight=0)
    base = api().compute_loss(natural, y, present, safety_weight=0)
    assert torch.equal(normal, base)
    result = api().compute_loss(weighted, y, present, class_weights=weights, safety_weight=0)
    delta = ((weights[y]-1)*(-weighted['probabilities'][torch.arange(2), y].log())).mean()
    torch.testing.assert_close(result-base, delta)
    with pytest.raises(ValueError, match='class_weights'):
        api().compute_loss(weighted, y, present, safety_weight=0)


def test_safety_positive_patient_nll_damage_uses_detached_anchor():
    out, y = outputs(), torch.tensor([0, 1])
    anchor = torch.tensor([[.9, .1], [.5, .5]], requires_grad=True)
    present = torch.ones(2, 2, dtype=torch.bool)
    base = api().compute_loss(out, y, present, safety_weight=0)
    result = api().compute_loss(out, y, present, anchor_probabilities=anchor)
    torch.testing.assert_close(result-base, .5*torch.tensor([torch.log(torch.tensor(.9/.8)), 0.]).mean())
    result.backward()
    assert anchor.grad is None


def test_auxiliary_mask_cannot_enable_missing_view():
    with pytest.raises(ValueError, match='aux_present'):
        api().compute_loss(outputs(), torch.tensor([0, 1]), torch.tensor([[1, 0], [1, 1]], dtype=torch.bool), aux_present=torch.ones(2, 2, dtype=torch.bool))


@pytest.mark.parametrize('key', ['view_alpha', 'joint_alpha'])
def test_loss_rejects_invalid_branch_alpha(key):
    out = outputs()
    out[key] = out[key].detach().clone()
    out[key].flatten()[0] = float('nan')
    with pytest.raises(ValueError, match='alpha'):
        api().compute_loss(out, torch.tensor([0, 1]), torch.ones(2, 2, dtype=torch.bool))
