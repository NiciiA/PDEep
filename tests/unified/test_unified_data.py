from collections import Counter
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import torch
import pytest
from pdeep.core import ValidationError
from pdeep.smooth.data import Case
from pdeep.smooth.definition import COEFFICIENTS
from pdeep.unified.config import DatasetConfig,TrainConfig,TASKS,MODELS,case_index
from pdeep.unified.dataset import inventory
from pdeep.unified.data import Provider,BalancedSampler
from pdeep.unified.workflow import preview
from pdeep.unified.training import train_models
from .conftest import config,EVIDENCE


def test_inventory_all24_stable_split_and_expansion():
    d=DatasetConfig(preset='smoke');rows=inventory(d)
    assert d.totals==dict(train=48,validation=48,test=48)
    assert len(rows)==144 and len({r['case_id'] for r in rows})==144
    assert len({r['latent_digest'] for r in rows})==len({r['field_digest'] for r in rows})==144
    more=inventory(replace(d,counts=(3,2,2)))
    lookup={tuple(r['key']):r for r in more}
    assert all(lookup[tuple(r['key'])]==r for r in rows)
    for p,g in TASKS:
        c=Case(g,p,'train',case_index(g,0));assert c.problem.coefficient==COEFFICIENTS[p]
        assert c.problem.physical_time==(None if p=='poisson' else (0.,1.))
        assert all(b.status.value=='known' for b in c.problem.symbolic.boundaries)
        c.problem.symbolic.validate()
        assert c.problem.symbolic.parameter_blocks and not any(b.known_values for b in c.problem.symbolic.parameter_blocks)


def test_balanced_task_cycles_fair_streams(arithmetic_dataset):
    c=config(arithmetic_dataset);s=BalancedSampler(c,arithmetic_dataset)
    keys=[k for i in range(60) for k in s.keys(i)];counts=Counter(k[:2] for k in keys)
    assert len(counts)==24 and set(counts.values())=={10}
    assert all(len(set(s.keys(i)))==4 for i in range(60))
    c2=replace(c,architecture='C_GEO',model_key=MODELS[1]);s2=BalancedSampler(c2,arithmetic_dataset)
    assert all(s.keys(i)==s2.keys(i) for i in range(20))
    a,ai=s.batch(0);b,bi=s2.batch(0)
    for token in ('observations','queries'):
        for k,v in vars(getattr(a,token)).items():
            if v is not None:assert torch.equal(v,getattr(getattr(b,token),k))
    assert ai==bi and s.state_dict(9)['cursor']==9
    EVIDENCE['balanced_task_exposures_240_cases']=dict((str(k),v) for k,v in counts.items())


def test_continuous_encounters_noise_counts_masks(arithmetic_dataset):
    provider=Provider(arithmetic_dataset);c=Case('annulus','wave','train',2)
    counts=set()
    for i in range(50):
        b,_=provider.batch((c,),cursor=i,count=32);counts.add(int(b.observations.valid.sum()))
        m=b.observations;assert m.valid.all() and m.time_present.all() and m.sigma.min()>0
        q=m.quantity_id[0].numpy();assert (q==0).sum()==(q==1).sum()
        assert c.problem.mesh.features(m.position[0].double().numpy())['inside'].all()
        assert np.all((m.sigma[0,:,0].numpy()/np.array([provider.norms.scale('wave',int(k)) for k in q]))>=.01999999)
    assert counts=={8,16,32,64,128}
    a,ai=provider.batch((c,),cursor=0,count=32);b,bi=provider.batch((c,),cursor=1,count=32)
    assert not torch.equal(a.observations.position,b.observations.position) and c.case_id==a.case_ids[0]==b.case_ids[0]
    a,ai=provider.batch((c,),training=False,count=32);b,bi=provider.batch((c,),training=False,count=32)
    assert ai==bi and torch.equal(a.observations.value,b.observations.value)
    p,_=provider.batch((Case('square','poisson','train',0),),count=8)
    assert not p.observations.time_present.any() and not p.queries.time_present.any()


def test_preview_no_model_optimizer_solver_or_lookup(arithmetic_dataset,tmp_path,monkeypatch):
    c=config(arithmetic_dataset)
    def fail(*a,**k):pytest.fail('preview performed an expensive operation')
    monkeypatch.setattr(torch.optim.AdamW,'__init__',fail)
    monkeypatch.setattr('pdeep.unified.training.Engine',fail)
    monkeypatch.setattr('pdeep.unified.reference.FixedReference',fail)
    r=train_models((c,),None,tmp_path);assert r['optimizer_constructors']==0 and r['checkpoints_searched']==0
    assert preview(arithmetic_dataset.config,(c,),tmp_path)['reference_solves']==0
    with pytest.raises(ValidationError):train_models((c,),None,tmp_path,lookup='search')
    with pytest.raises(ValidationError):train_models((c,),None,tmp_path,run_mode='train')


def test_holdout_family_excluded(arithmetic_dataset):
    d=replace(arithmetic_dataset.config,holdout_family='annulus');arithmetic_dataset.config=d
    c=config(arithmetic_dataset);s=BalancedSampler(c,arithmetic_dataset)
    assert all(k[1]!='annulus' for i in range(50) for k in s.keys(i))
    assert len(d.tasks)==24 and len(d.training_tasks)==21


def test_neural_seed_does_not_change_physical_encounters(arithmetic_dataset):
    a=config(arithmetic_dataset);b=replace(a,root_seed=7193)
    sa=BalancedSampler(a,arithmetic_dataset);sb=BalancedSampler(b,arithmetic_dataset)
    assert a.root_seed!=b.root_seed and a.data_seed==b.data_seed
    for i in range(10):assert sa.keys(i)==sb.keys(i)
    aa,ai=sa.batch(0);bb,bi=sb.batch(0);assert ai==bi and torch.equal(aa.observations.value,bb.observations.value)
