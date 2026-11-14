from dataclasses import replace
import numpy as np
import torch
import pytest
from pdeep.geometric.models import create
from pdeep.smooth.data import Case
from pdeep.unified.config import TASKS,MODELS
from pdeep.unified.data import Provider
from pdeep.unified.loss import UnifiedAdapter
from pdeep.unified.preflight import inventory
from .conftest import config,EVIDENCE


def test_actual_medium_parameter_counts_independent(arithmetic_dataset):
    configs=tuple(config(arithmetic_dataset,k) for k in MODELS)
    medium=tuple(replace(c,capacity='medium',preset='research',updates=30000) for c in configs)
    rows=inventory(medium,arithmetic_dataset.norms)
    assert [r['parameters'] for r in rows]==[6158789,6455429,6455429]
    a=create('C_GEO',arithmetic_dataset.norms,'verification',seed=42)
    b=create('C_GEO',arithmetic_dataset.norms,'verification',seed=42)
    assert all(torch.equal(x,y) and x.data_ptr()!=y.data_ptr() for x,y in zip(a.parameters(),b.parameters()))
    EVIDENCE['medium_parameter_counts']={r['model']:r['parameters'] for r in rows}

@pytest.mark.parametrize('arch',['A_GEO','C_GEO'])
def test_all24_model_forward_types_chunk_masks(arithmetic_dataset,arch):
    model=create(arch,arithmetic_dataset.norms,'verification',seed=42);model.eval();provider=Provider(arithmetic_dataset);maximum=0.
    for p,g in TASKS:
        c=Case(g,p,'validation',0);batch,_=provider.batch((c,),count=8)
        with torch.no_grad():
            ctx=model.prepare_context(batch.specs,batch.observations)
            a=model.predict(ctx,batch.queries,chunk=8);b=model.predict(ctx,batch.queries,chunk=3)
        assert a.shape==(1,8,1) and torch.isfinite(a).all()
        err=float((a-b).abs().max());maximum=max(maximum,err)
        torch.testing.assert_close(a,b,rtol=3e-5,atol=2e-6)
        if p=='poisson':assert not batch.queries.time_present.any()
        if p=='wave':assert set(batch.queries.quantity_id[0].tolist())=={0,1}
    EVIDENCE[arch+'_all24_max_chunk_difference']=maximum


def test_padding_permutation_observation_dependence_and_context_isolation(arithmetic_dataset,batch):
    c=config(arithmetic_dataset);model=create(c.architecture,arithmetic_dataset.norms,'verification',seed=42).eval();b=batch.slice(0,2)
    def output(obs):
        with torch.no_grad():return model.predict(model.prepare_context(b.specs,obs),b.queries,chunk=8)
    a=output(b.observations)
    from pdeep.geometric.data import Tokens
    permutation=torch.arange(b.observations.valid.shape[1]-1,-1,-1)
    perm=Tokens(**{k:None if v is None else v[:,permutation] for k,v in vars(b.observations).items()})
    error=float((a-output(perm)).abs().max());assert error<2e-6
    padded={}
    for k,v in vars(b.observations).items():
        if v is None:padded[k]=None;continue
        shape=list(v.shape);shape[1]=7;extra=torch.zeros(shape,dtype=v.dtype)
        if k in ('position','value','sigma','physical_time'):extra.fill_(123.)
        padded[k]=torch.cat((v,extra),dim=1)
    assert float((a-output(Tokens(**padded))).abs().max())<2e-6
    changed=Tokens(**{k:None if v is None else v.clone() for k,v in vars(b.observations).items()})
    changed.value[0]+=3*arithmetic_dataset.norms.scale('poisson','u')
    y=output(changed);assert not torch.allclose(a[0],y[0]);torch.testing.assert_close(a[1],y[1],atol=0,rtol=0)
    changed.position[0]=changed.position[0].roll(1,0)
    assert not torch.allclose(y[0],output(changed)[0])
    changed=Tokens(**{k:None if v is None else v.clone() for k,v in vars(b.observations).items()});changed.physical_time[0]=123.
    torch.testing.assert_close(a[0],output(changed)[0],atol=0,rtol=0)
    EVIDENCE['observation_permutation_max_difference']=error

@pytest.mark.parametrize('key',MODELS)
def test_exact_requested_terms_and_physical_gradients(arithmetic_dataset,batch,key):
    c=config(arithmetic_dataset,key);model=create(c.architecture,arithmetic_dataset.norms,'verification',seed=42);adapter=UnifiedAdapter(c)
    records=[]
    for i in range(3):
        b=batch.slice(i,i+1);r=adapter.loss(model,b);p=b.cases[0].pde
        active={'field','auxiliary'}
        if key=='A_GEO/KIN_001' and p=='wave':active.add('kinematic')
        if key=='C_GEO/ALL_001':active.update(('weak','boundary'));active.update(('initial',) if p!='poisson' else ())
        assert set(r.weighted)==active
        for term in active:
            weight=.1 if term=='auxiliary' else 1. if term=='field' else .001
            torch.testing.assert_close(r.weighted[term],weight*r.unweighted[term],rtol=1e-7,atol=1e-8)
        r.total.backward();assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
        norm=sum(float(v.grad.square().sum()) for v in model.parameters() if v.grad is not None)**.5
        assert norm>0;records.append({'pde':p,'terms':sorted(active),'gradient_norm':norm});model.zero_grad(set_to_none=True)
    EVIDENCE[key+'_loss_gradients']=records


def test_accumulation_equal_case_quantity_no_chunk_double_count(arithmetic_dataset,batch):
    c=config(arithmetic_dataset);model=create(c.architecture,arithmetic_dataset.norms,'verification',seed=42)
    a=UnifiedAdapter(c);r=a.loss(model,batch);r.total.backward();g=[p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in model.parameters()];value=float(r.total.detach());model.zero_grad(set_to_none=True)
    b=UnifiedAdapter(replace(c,query_chunk=3));v=0.
    for i in range(4):
        loss=b.loss(model,batch.slice(i,i+1));(loss.total/4).backward();v+=float(loss.total.detach())/4
    err=max(float((x-(p.grad if p.grad is not None else torch.zeros_like(p))).abs().max()) for x,p in zip(g,model.parameters()))
    relative=(sum(float((x-(p.grad if p.grad is not None else torch.zeros_like(p))).square().sum()) for x,p in zip(g,model.parameters()))/max(sum(float(x.square().sum()) for x in g),1e-30))**.5
    assert abs(value-v)<1e-5 and relative<3e-5
    EVIDENCE['effective4_accumulation_loss_difference']=abs(value-v);EVIDENCE['effective4_accumulation_max_gradient_difference']=err;EVIDENCE['effective4_accumulation_relative_gradient_difference']=relative
