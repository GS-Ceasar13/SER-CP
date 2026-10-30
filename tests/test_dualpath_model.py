import importlib
import inspect
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch.set_num_threads(2)


def api():
    assert importlib.util.find_spec('dualpath_model') is not None, 'dual-path model is not implemented'
    return importlib.import_module('dualpath_model')


def inputs():
    torch.manual_seed(11)
    return torch.randn(4, 3, 6), torch.tensor([[1, 1, 1], [1, 0, 1], [0, 1, 0], [1, 1, 0]], dtype=torch.bool)


ARMS = ['faithful_ds', 'shared_capacity_ds', 'joint_only', 'global_posterior',
        'patient_posterior_natural', 'patient_posterior_weighted',
        'patient_direction_natural', 'patient_direction_weighted']


@pytest.mark.parametrize('arm', ARMS)
def test_arm_has_finite_normalized_probability(arm):
    model = api().build_model(arm, 6, 3, 3, hidden=8).eval()
    x, present = inputs()
    out = model(x, present)
    assert out['probabilities'].shape == (4, 3)
    assert torch.isfinite(out['probabilities']).all()
    assert (out['probabilities'] > 0).all()
    torch.testing.assert_close(out['probabilities'].sum(-1), torch.ones(4))
    assert out['view_alpha'].shape == (4, 3, 3)
    assert torch.equal(out['view_alpha'][~present], torch.ones_like(out['view_alpha'][~present]))


def test_posterior_mix_endpoints_and_formula():
    fn = api().mix_paths
    ds, joint = torch.tensor([[8., 2., 1.]]), torch.tensor([[1., 3., 6.]])
    pds, pj = ds/ds.sum(-1, keepdim=True), joint/joint.sum(-1, keepdim=True)
    assert torch.equal(fn(ds, joint, torch.zeros(1, 1), 'posterior'), pds)
    assert torch.equal(fn(ds, joint, torch.ones(1, 1), 'posterior'), pj)
    torch.testing.assert_close(fn(ds, joint, torch.tensor([[.4]]), 'posterior'), .6*pds+.4*pj)


def test_direction_strength_has_original_m1_formula():
    ds, joint, g = torch.tensor([[9., 2., 1.]]), torch.tensor([[1., 4., 7.]]), torch.tensor([[.3]])
    eds, ej = ds-1, joint-1
    sds, sj = eds.sum(-1, keepdim=True), ej.sum(-1, keepdim=True)
    alpha = 1 + ((1-g)*sds+g*sj)*((1-g)*eds/sds+g*ej/sj)
    torch.testing.assert_close(api().mix_paths(ds, joint, g, 'direction'), alpha/alpha.sum(-1, keepdim=True))
    uniform = api().mix_paths(torch.ones(1, 3), torch.ones(1, 3), g, 'direction')
    torch.testing.assert_close(uniform, torch.full((1, 3), 1/3))


def test_independent_views_receive_independent_gradients():
    model = api().build_model('patient_posterior_natural', 6, 3, 3, hidden=8)
    x, present = inputs()
    out = model(x, present)
    loss = out['view_alpha'].sum() + out['joint_alpha'].sum() + out['probabilities'][:, 0].sum()
    loss.backward()
    assert len({id(p) for enc in model.enc for p in enc.parameters()}) == sum(len(list(enc.parameters())) for enc in model.enc)
    for enc, head in zip(model.enc, model.head_v):
        assert enc[0].weight.grad.abs().sum() > 0
        assert head.weight.grad.abs().sum() > 0
    assert model.joint[-1].weight.grad.abs().sum() > 0


def test_router_is_exact_original_factorized_class_and_ds_is_original_method():
    model = api().build_model('faithful_ds', 6, 3, 3, hidden=8).eval()
    assert model.router.__class__.__name__ == 'FourStateEvidenceRouter'
    assert model.router.router == 'factorized'
    assert model._ds_two.__name__ == 'ds_two'
    x, present = inputs()
    out = model(x, present)
    r, s = out['diagnostics']['reliability'], out['diagnostics']['support']
    expected = torch.stack([r*s, r*(1-s), (1-r)*s, (1-r)*(1-s)], -1)
    expected = expected*present[..., None]
    torch.testing.assert_close(out['diagnostics']['state_weights'], expected)
    fused = out['view_alpha'][:, 0]
    for v in range(1, 3):
        fused = model._ds_two(model, fused, out['view_alpha'][:, v], discount=False)
    assert torch.equal(fused, out['ds_alpha'])


def test_faithful_and_joint_only_select_named_path():
    x, present = inputs()
    for arm, key, gate in [('faithful_ds', 'ds_alpha', 0), ('joint_only', 'joint_alpha', 1)]:
        out = api().build_model(arm, 6, 3, 3, hidden=8).eval()(x, present)
        alpha = out[key]
        assert torch.equal(out['probabilities'], alpha/alpha.sum(-1, keepdim=True))
        assert torch.equal(out['gate'], torch.full((4, 1), float(gate)))


def test_global_gate_is_single_trainable_scalar():
    model = api().build_model('global_posterior', 6, 3, 3, hidden=8).eval()
    out = model(*inputs())
    assert model.global_logit.numel() == 1
    assert model.global_logit.requires_grad
    assert torch.equal(out['gate'], out['gate'][:1].expand_as(out['gate']))


def test_patient_gate_is_sample_local_and_has_no_deployment_labels():
    model = api().build_model('patient_posterior_natural', 6, 3, 3, hidden=8).eval()
    x, present = inputs()
    out = model(x, present)
    single = model(x[:1], present[:1])
    torch.testing.assert_close(out['probabilities'][:1], single['probabilities'])
    assert (out['gate'] > 0).all() and (out['gate'] < 1).all()
    params = inspect.signature(model.forward).parameters
    assert not any(k in params for k in ['labels', 'donor_labels', 'corruption', 'split', 'dataset'])


def test_masked_feature_changes_do_not_change_predictions():
    model = api().build_model('patient_posterior_natural', 6, 3, 3, hidden=8).eval()
    x, present = inputs()
    changed = x.clone()
    changed[~present] = 10000
    torch.testing.assert_close(model(x, present)['probabilities'], model(changed, present)['probabilities'])


@pytest.mark.parametrize('dimensions', [(6, 3, 3, 8), (512, 3, 2, 128), (12, 6, 20, 128)])
def test_shared_capacity_counts_only_active_trainable_head_parameters(dimensions):
    dim, views, classes, hidden = dimensions
    independent = api().build_model('faithful_ds', dim, views, classes, hidden)
    shared = api().build_model('shared_capacity_ds', dim, views, classes, hidden)
    a, b = independent.parameter_counts(), shared.parameter_counts()
    assert abs(a['view_heads']-b['view_heads'])/a['view_heads'] <= .05
    assert b['total'] == sum(p.numel() for p in shared.parameters())
    assert a['joint'] == b['joint'] and a['router'] == b['router']
    assert shared.enc[0] is shared.enc[1] and shared.head_v[0] is shared.head_v[1]


def test_exact_noop_returns_external_anchor_object_without_running_new_head():
    model = api().build_model('patient_posterior_natural', 6, 3, 3, hidden=8)
    anchor = torch.tensor([[.17, .23, .60]])
    out = model(torch.full((1, 3, 6), float('nan')), torch.zeros(1, 3, dtype=torch.bool), anchor, True)
    assert out['probabilities'] is anchor
    with pytest.raises(ValueError, match='anchor'):
        model(*inputs(), force_noop=True)


def test_invalid_inputs_and_arm_fail_closed():
    with pytest.raises(ValueError, match='arm'):
        api().build_model('arbitrary', 6, 3, 3, hidden=8)
    model = api().build_model('faithful_ds', 6, 3, 3, hidden=8)
    x, present = inputs()
    with pytest.raises(ValueError, match='present'):
        model(x, torch.zeros_like(present))
    with pytest.raises(ValueError, match='finite'):
        model(x*float('nan'), present)


def test_exact_noop_rejects_anchor_patient_count_mismatch():
    model = api().build_model('faithful_ds', 6, 3, 3, hidden=8)
    with pytest.raises(ValueError, match='anchor'):
        model(*inputs(), anchor_probabilities=torch.full((1, 3), 1/3), force_noop=True)


def test_mixer_rejects_out_of_range_gate():
    with pytest.raises(ValueError, match='gate'):
        api().mix_paths(torch.ones(2, 3), torch.ones(2, 3), torch.tensor([[1.2], [.2]]))


@pytest.mark.parametrize('arm', ARMS)
def test_real_matched_loss_reaches_each_trainable_path(arm):
    from dualpath_losses import compute_loss
    model = api().build_model(arm, 6, 3, 3, hidden=8)
    x, present = inputs()
    out = model(x, present)
    loss = compute_loss(out, torch.tensor([0, 1, 2, 1]), present,
                        class_weights=torch.ones(3), anchor_probabilities=torch.full((4, 3), 1/3))
    assert torch.isfinite(loss)
    loss.backward()
    for head in model.head_v:
        assert head.weight.grad is not None and head.weight.grad.abs().sum() > 0
    assert model.joint[-1].weight.grad.abs().sum() > 0
    assert model.router.residual_head[-1].weight.grad.abs().sum() > 0
