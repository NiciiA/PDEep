from dataclasses import replace
import pytest
import torch
from pdeep.a_checks import mixed_fixture,slice_tokens
from pdeep.a_training import objective as supervised_objective
from pdeep.architecture_a import build_architecture_a
from pdeep.architecture_c import build_architecture_c
from pdeep.constraints import (Quadrature,physical_losses,field_derivatives,prior_initial_scales,weak_integrals)
from pdeep.c_checks import architecture_c_checks
from pdeep.c_math_checks import mathematical_constraint_checks
from pdeep.c_training import objective,CTrainingConfig,make_training_batch
from pdeep.core import ValidationError
from pdeep.data import calibrate,CaseDataset
from pdeep.encoding import pack_graphs


@pytest.fixture(scope='module')
def norm():return calibrate()


def test_C1_C8_independent_mathematics():
    r=mathematical_constraint_checks()
    assert r['passed'],[c for c in r['checks'] if not c['passed']]


@pytest.mark.parametrize('device',['cpu','mps','cuda'])
def test_C_structure_gradients_strong_and_quadrature(norm,device):
    from pdeep.runtime import capabilities
    if not next(c.available for c in capabilities() if c.device==device):pytest.skip(device+' unavailable')
    r=architecture_c_checks(norm,device=device)
    assert r['passed'],[c for c in r['checks'] if not c['passed']]
    assert len(r['strong_diagnostics'])==3


def test_initial_scales_training_prior_exact_and_source_coupling(norm):
    c=build_architecture_c(norm);b=mixed_fixture(norm,queries=8)
    out=c(b.specs,b.observations,b.queries)
    for spec in b.specs:
        expected=tuple(sum(m*m+s*s for m,s in zip(block.prior.mean,block.prior.scale))**.5 for block in spec.parameter_blocks)
        assert prior_initial_scales(spec)==expected
    p=physical_losses(c,out.context,xi_hat=out.xi_hat)
    changed=out.xi_hat.clone();changed[:,0]+=.2;changed[2,9]+=.3
    alt=physical_losses(c,out.context,xi_hat=changed)
    assert not torch.allclose(p.weak_per_case[0],alt.weak_per_case[0])
    torch.testing.assert_close(p.weak_per_case[1:],alt.weak_per_case[1:],rtol=0,atol=0) # dynamic equations use neural state
    assert not torch.allclose(p.initial_per_case[1:],alt.initial_per_case[1:])
    # Ground truth changes have no effect on physical terms; only supervised xi loss changes.
    loss=objective(c,out,b,physical=p)
    altered=replace(b,unknowns=replace(b.unknowns,xi=b.unknowns.xi+2.))
    another=objective(c,out,altered,physical=p)
    torch.testing.assert_close(loss['weak'],another['weak'],rtol=0,atol=0)
    torch.testing.assert_close(loss['initial'],another['initial'],rtol=0,atol=0)
    assert not torch.equal(loss['xi'],another['xi'])
    for weak_weight in (0.,.1,1.):
        for initial_weight in (0.,.1,1.):
            l=objective(c,out,b,weak_weight=weak_weight,initial_weight=initial_weight,physical=p)
            torch.testing.assert_close(l['total'],l['field']+.1*l['xi']+weak_weight*l['weak']+initial_weight*l['initial'])


def test_chunk_integrals_and_gradients_not_chunk_squared(norm):
    c=build_architecture_c(norm);b=mixed_fixture(norm,queries=8).slice(0,1)
    # Poisson: the source has global test-space support, so chunk-squared error is detectable.
    def backward(chunk):
        c.zero_grad(set_to_none=True);out=c(b.specs,b.observations,b.queries)
        p=physical_losses(c,out.context,spatial_chunk=chunk)
        p.weak_per_case.mean().backward()
        return p.weak_per_case.detach(),{n:p.grad.clone() for n,p in c.named_parameters() if p.grad is not None}
    full,g=backward(None);parts,z=backward(71)
    torch.testing.assert_close(full,parts,rtol=2e-5,atol=2e-6)
    assert g.keys()==z.keys()
    for name in g:torch.testing.assert_close(g[name],z[name],rtol=4e-4,atol=4e-6)


def test_combined_microbatch_gradient_weighting(norm):
    c=build_architecture_c(norm);b=mixed_fixture(norm,queries=8).slice(0,2)
    out=c(b.specs,b.observations,b.queries);objective(c,out,b)['total'].backward()
    full={n:p.grad.clone() for n,p in c.named_parameters() if p.grad is not None}
    c.zero_grad(set_to_none=True)
    for i in range(2):
        part=b.slice(i,i+1);out=c(part.specs,part.observations,part.queries)
        (objective(c,out,part)['total']/2).backward()
    for name,p in c.named_parameters():
        if name in full:torch.testing.assert_close(p.grad,full[name],rtol=4e-4,atol=5e-6)


def test_independence_solver_exclusion_context_masks_no_memory(norm,monkeypatch):
    b=mixed_fixture(norm,queries=8)
    before=torch.random.get_rng_state().clone();c=build_architecture_c(norm)
    assert torch.equal(before,torch.random.get_rng_state())
    a=build_architecture_a(norm);same=build_architecture_c(norm)
    assert all(torch.equal(x,y) for x,y in zip(c.parameters(),same.parameters()))
    assert not ({p.data_ptr() for p in c.parameters()}&{p.data_ptr() for p in a.parameters()})
    assert not any('memory' in name or 'anchor' in name for name,_ in c.named_modules())
    def forbidden(*a,**kw):raise AssertionError('solver in C field path')
    import pdeep.reference as r
    import pdeep.bayes as d
    monkeypatch.setattr(r,'evaluate',forbidden);monkeypatch.setattr(r,'evaluate_batch',forbidden);monkeypatch.setattr(d,'sensor_matrix',forbidden)
    context=c.prepare_context(b.specs,b.observations)
    original=c.predict_queries(context,b.queries).predicted_value
    with torch.no_grad():c.coefficient_head[-1].bias.add_(100.)
    torch.testing.assert_close(original,c.predict_queries(context,b.queries).predicted_value,rtol=0,atol=0)
    with pytest.raises(ValidationError,match='another model'):same.predict_queries(context,b.queries)
    with pytest.raises(ValidationError,match='strictly positive'):c.prepare_context(b.specs,replace(b.observations,sigma=torch.zeros_like(b.observations.sigma)))
    with pytest.raises(ValidationError):c.double().prepare_context(b.specs,b.observations)


def test_no_physics_grad_tracking_suppression_and_config(norm):
    c=build_architecture_c(norm);b=mixed_fixture(norm,queries=3);ctx=c.prepare_context(b.specs,b.observations)
    q=replace(b.queries,position=b.queries.position.requires_grad_(),physical_time=b.queries.physical_time.requires_grad_())
    with torch.no_grad(),pytest.raises(ValidationError,match='gradient tracking'):field_derivatives(c,ctx,q)
    assert CTrainingConfig().spatial_order==16 and CTrainingConfig().time_order==8
    for kwargs in ({'spatial_order':8},{'time_order':4},{'updates':6},{'preset':'research'},{'tiny_overfit':True}):
        with pytest.raises(ValidationError):CTrainingConfig(**kwargs)
    batch=make_training_batch(CaseDataset(),norm,0)
    assert sorted(s.name for s in batch.specs)==['heat','poisson','wave']
    same=make_training_batch(CaseDataset(),norm,99,fixed=True)
    assert batch.case_ids==same.case_ids and torch.equal(batch.target,same.target)


def test_C_export_reload_identity_fresh_process_and_checksum(norm,tmp_path):
    from pdeep.c_artifacts import save_inference,load_inference
    import subprocess,sys
    c=build_architecture_c(norm);b=mixed_fixture(norm,queries=8)
    manifest=save_inference(c,tmp_path,source='fixture',config={})
    assert save_inference(c,tmp_path,source='fixture',config={})==manifest
    restored,meta=load_inference(manifest)
    assert set(meta['initial_RMS'])=={'heat','wave'}
    with torch.no_grad():torch.testing.assert_close(c(b.specs,b.observations,b.queries).predicted_value,restored(b.specs,b.observations,b.queries).predicted_value,rtol=0,atol=0)
    code='from pdeep.c_artifacts import load_inference; import sys; c,m=load_inference(sys.argv[1]); print(sum(c.parameter_inventory().values()))'
    r=subprocess.run([sys.executable,'-c',code,str(manifest)],capture_output=True,text=True,check=True)
    assert int(r.stdout.strip())==527796
    path=manifest.parent/'weights.pt';path.write_bytes(path.read_bytes()+b'bad')
    with pytest.raises(ValidationError,match='checksum'):load_inference(manifest)
