from dataclasses import replace
import json,subprocess,sys
import pytest
import torch
from pdeep.architecture_b import build_architecture_b,BAblation
from pdeep.b_checks import architecture_b_checks,inverse_math_checks
from pdeep.b_training import BTrainingConfig,coefficient_loss,make_training_batch
from pdeep.b_physics import build_system,anchor_projections
from pdeep.b_artifacts import save_checkpoint,load_checkpoint,tree_identity
from pdeep.a_checks import mixed_fixture
from pdeep.data import CaseDataset,calibrate
from pdeep.core import ValidationError
from pdeep.reference import basis,eigenvalues
from pdeep.models import select_architecture


@pytest.fixture(scope='module')
def norm():return calibrate()


def test_analytic_inverse_gradient_and_physical_observations(norm):
    r=inverse_math_checks(norm);assert r['passed'],r


@pytest.mark.parametrize('device',['cpu','mps','cuda'])
def test_B0_B1_device_structures_gradients_ablations(norm,device):
    from pdeep.runtime import capabilities
    if not next(c.available for c in capabilities() if c.device==device):pytest.skip(device+' unavailable')
    r=architecture_b_checks(norm,device=device)
    assert r['passed'],[x for x in r['checks'] if not x['passed']]


def test_spatial_projection_convention_recurrence_and_geometry(norm):
    b=mixed_fixture(norm,queries=8);m=build_architecture_b('B1',norm);ctx=m.prepare_context(b.specs,b.observations)
    r=m.infer_unknowns(ctx,return_diagnostics=True)
    s=m.spatial;positions=s.positions.double();phi=basis(positions);lam=eigenvalues()
    fields,grad,features=anchor_projections(ctx.system,r.trajectory[:,0],r.gradients[:,0],s.positions)
    torch.testing.assert_close(fields[0,:,0],phi@(lam*r.trajectory[0,0,:9]),rtol=1e-13,atol=1e-13)
    torch.testing.assert_close(grad[0,:,0],phi@(lam*r.gradients[0,0,:9]),rtol=1e-13,atol=1e-9)
    torch.testing.assert_close(fields[2,:,1],phi@r.trajectory[2,0,9:],rtol=1e-13,atol=1e-13)
    assert torch.bincount(s.levels).tolist()==[16,64] and ((s.positions>0)&(s.positions<1)).all()
    assert len(r.spatial_history)==3 and r.spatial_history[0]['tokens'].shape==(3,80,96)
    assert not torch.equal(r.spatial_history[0]['tokens'],r.spatial_history[1]['tokens'])
    # Previous-state projection must actually participate at later iterations.
    coefficient_loss(r,b.unknowns).backward();assert s.previous.weight.grad.abs().sum()>0
    again=m.infer_unknowns(ctx,return_diagnostics=True)
    torch.testing.assert_close(r.xi_hat,again.xi_hat,rtol=0,atol=0)
    assert build_architecture_b('B0',norm).spatial is None


def test_padding_masks_loss_and_queries_not_inverse_input(norm):
    b=mixed_fixture(norm,queries=8);m=build_architecture_b('B0',norm)
    r=m.infer_unknowns(m.prepare_context(b.specs,b.observations))
    truth=replace(b.unknowns,xi=torch.where(b.unknowns.active,b.unknowns.xi,torch.full_like(b.unknowns.xi,float('nan'))))
    torch.testing.assert_close(coefficient_loss(r,truth),coefficient_loss(r,b.unknowns),rtol=0,atol=0)
    x=r.xi_hat.clone();m.predict_queries(r,replace(b.queries,physical_time=torch.where(b.queries.time_present[...,None],b.queries.physical_time,torch.full_like(b.queries.physical_time,float('nan')))))
    assert torch.equal(x,r.xi_hat)
    with pytest.raises(ValidationError):m.predict_queries(r,replace(b.queries,quantity_id=torch.full_like(b.queries.quantity_id,99)))
    with pytest.raises(ValidationError,match='strictly positive'):m.prepare_context(b.specs,replace(b.observations,sigma=torch.zeros_like(b.observations.sigma)))


def test_all_neural_initializations_independent_and_rng_preserved(norm):
    from pdeep.architecture_a import build_architecture_a
    from pdeep.architecture_c import build_architecture_c
    before=torch.random.get_rng_state().clone();models=[build_architecture_b(x,norm) for x in ('B0','B1')]+[build_architecture_a(norm),build_architecture_c(norm)]
    assert torch.equal(before,torch.random.get_rng_state())
    pointers=[{p.data_ptr() for p in m.parameters()} for m in models]
    for i in range(4):
        for j in range(i):assert not pointers[i]&pointers[j]
    repeat=build_architecture_b('B0',norm);assert all(torch.equal(p,q) for p,q in zip(models[0].parameters(),repeat.parameters()))
    for approach in ('B0','B1'):assert select_architecture(approach,normalizers=norm).approach==approach
    with pytest.raises(ValidationError):select_architecture('B',normalizers=norm)


@pytest.mark.parametrize('approach',['B0','B1'])
def test_complete_checkpoint_variant_reload_checksum_fresh_process(norm,tmp_path,approach):
    m=build_architecture_b(approach,norm);config=BTrainingConfig(approach=approach)
    optimizer=torch.optim.AdamW(m.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    path=save_checkpoint(m,optimizer,config,tmp_path,source='fixture',dataset_id=CaseDataset().identity,next_step=0,counters={'successful':0})
    restored,opt,cfg,sampler,meta=load_checkpoint(path,expected_approach=approach)
    assert cfg==config and sampler['next_step']==0
    assert tree_identity(opt.state_dict())==tree_identity(optimizer.state_dict())
    assert save_checkpoint(m,optimizer,config,tmp_path,source='fixture',dataset_id=CaseDataset().identity,next_step=0,counters={'successful':0})==path
    b=mixed_fixture(norm,queries=8)
    with torch.no_grad():
        a=m(b.specs,b.observations,b.queries);z=restored(b.specs,b.observations,b.queries)
        torch.testing.assert_close(a.xi_hat,z.xi_hat,rtol=0,atol=0)
        torch.testing.assert_close(a.predicted_value,z.predicted_value,rtol=0,atol=0)
    other='B1' if approach=='B0' else 'B0'
    with pytest.raises(ValidationError,match='incompatible'):load_checkpoint(path,expected_approach=other)
    code='from pdeep.b_artifacts import load_checkpoint; import sys; m,o,c,s,a=load_checkpoint(sys.argv[1],expected_approach=sys.argv[2]); print(m.approach)'
    r=subprocess.run([sys.executable,'-c',code,str(path),approach],capture_output=True,text=True,check=True);assert r.stdout.strip()==approach
    payload=path.parent/'checkpoint.pt';payload.write_bytes(payload.read_bytes()+b'bad')
    with pytest.raises(ValidationError,match='checksum'):load_checkpoint(path,expected_approach=approach)


def test_matched_variant_sampler_and_microbatch_gradient(norm):
    dataset=CaseDataset();b=make_training_batch(dataset,norm,0,count=3)
    again=make_training_batch(dataset,norm,0,count=3)
    assert b.case_ids==again.case_ids
    torch.testing.assert_close(b.observations.value,again.observations.value,rtol=0,atol=0,equal_nan=True)
    assert sorted(s.name for s in b.specs)==['heat','poisson','wave']
    m=build_architecture_b('B0',norm)
    r=m.infer_unknowns(m.prepare_context(b.specs,b.observations));coefficient_loss(r,b.unknowns).backward()
    full={n:p.grad.clone() for n,p in m.named_parameters()};m.zero_grad(set_to_none=True)
    for lo,hi in ((0,2),(2,3)):
        part=b.slice(lo,hi);r=m.infer_unknowns(m.prepare_context(part.specs,part.observations));(coefficient_loss(r,part.unknowns)*((hi-lo)/3)).backward()
    for n,p in m.named_parameters():torch.testing.assert_close(p.grad,full[n],rtol=4e-4,atol=2e-6)
    for kwargs in ({'approach':'B'},{'updates':6},{'iterations':4},{'precision':'float64'}):
        with pytest.raises(ValidationError):BTrainingConfig(**kwargs)
