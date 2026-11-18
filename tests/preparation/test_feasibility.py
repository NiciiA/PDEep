from dataclasses import replace
from collections import Counter
from types import SimpleNamespace
import numpy as np
import pytest,torch
from pdeep.unified.config import case_index,DatasetConfig
from pdeep.unified.dataset import inventory
from pdeep.unified_modes.data import Sampler
from pdeep.unified.streams import coordinates_id
from pdeep.preparation.feasibility import configuration,parent_configuration,make_plan,save_plan
from pdeep.preparation.feasibility_training import configurations,train_models
from pdeep.preparation.planning import encounters
from pdeep.preparation.provider import FeasibilityProvider
from pdeep.preparation.reuse import reuse_case,ReuseBlocked
from pdeep.smooth.data import Case,Normalizers
from pdeep.smooth.reference import Reference
from pdeep.learning_diagnostics.reference_cache import FixedReference
from pdeep.artifact_io import read_payload,write_artifact

def test_all_24_profile_budgets_prefix_and_split_isolation(monkeypatch):
    c=configuration();parent=parent_configuration(c)
    assert c.totals=={'train':192,'validation':48,'test':48} and len(c.tasks)==24
    assert c.stream_updates==3000 and c.stream_queries==4096
    pools={}
    for split in ('train','validation','test'):
        assert set(c.keys(split))<=set(parent.keys(split))
        pools[split]={Case(g,p,s,i).case_id for p,g,s,i in c.keys(split)}
    assert not pools['train']&pools['validation'] and not pools['train']&pools['test']
    def forbidden(*a,**k):pytest.fail('preview constructs optimizer')
    monkeypatch.setattr(torch.optim.AdamW,'__init__',forbidden)
    cs=configurations(c,device='cpu');r=train_models(cs,None,'unused',run_mode='preview',stop_after=3000)
    assert r['optimizer_steps']==0 and r['session_stop']==r['planned_horizon']==3000
    assert all(x.capacity=='large' and x.main_target==3000 for x in cs)

def test_parent_draws_modes_and_provider_no_recycling(certificate,tmp_path):
    c=replace(configuration(),stream_updates=60,stream_queries=16,stream_grid=4,qc_points=8)
    parent=replace(c,stream_updates=2000);rows=inventory(c);plan=make_plan(c,rows,parent,encounters(parent,inventory(parent)))
    out=tmp_path/'reference_cache';out.mkdir()
    write_artifact(out,'feasibility-plan',{'artifact_type':'diagnostic_bundle','architecture_id':'REFERENCE_ONLY','run_id':'independent-draw-control'},plan,required=tuple(plan))
    d=SimpleNamespace(root=tmp_path,config=c,norms=Normalizers.fromdict(certificate['normalizers']),index={'feasibility':{'mapping_identity':plan['mapping_identity']}},require_case=lambda c:None,cache=SimpleNamespace(evaluate=lambda c,x,t,q:np.zeros(len(x),np.float64)))
    cfg=SimpleNamespace(data_seed=c.seed,device='cpu',query_count=16,mode_mixture=(.3,.4,.3),hybrid_zero_probability=.25)
    sampler=Sampler(cfg,d,provider=FeasibilityProvider(d));exposures=Counter();used=set()
    for cursor in range(60):
        batch,identities=sampler.batch(cursor)
        for view,ident in zip(batch.cases,identities):
            exposures[(view.pde,view.family,view.mode)]+=1
            key=(view.case_id,ident['reference_cursor'],ident['reference_slot']);assert key not in used;used.add(key)
            if view.mode=='FORWARD':assert ident['valid_observations']==0
            assert not view.condition.values[~view.condition.available].any()
    assert len(exposures)==72 and all(n==(4 if mode=='INVERSE' else 3) for (_,_,mode),n in exposures.items())

def test_completed_reference_reuse_is_byte_exact_and_missing_design_blocks(prepared,certificate,tmp_path,monkeypatch):
    c=prepared.cases('train')[0];_,p=read_payload(prepared.root/prepared.index['records'][c.case_id]);events=p['stream_events'][:1]
    target=replace(prepared.config,stream_updates=2)
    def forbidden(*a,**k):pytest.fail('completed donor numerically regenerated')
    monkeypatch.setattr(Reference,'__init__',forbidden);monkeypatch.setattr(FixedReference,'__init__',forbidden)
    result=reuse_case(prepared.path,c,target,events,tmp_path/'cases/new.json',tmp_path,certificate,finer=True)
    assert result is not None
    for key,v in result[1]['labels'].items():torch.testing.assert_close(v,p['labels'][key],atol=0,rtol=0)
    with pytest.raises(ReuseBlocked):reuse_case(prepared.path,c,replace(target,stream_queries=17),events,tmp_path/'cases/missing.json',tmp_path,certificate)
