from pathlib import Path
from types import SimpleNamespace
import json,os
import numpy as np
import pytest,torch
from pdeep.artifact_io import write_artifact,read_payload
from pdeep.artifact_rng import capture_rng
from pdeep.artifact_tensors import tree_identity
from pdeep.core import digest
from pdeep.smooth.data import Case,Normalizers
from pdeep.smooth.bank import physical_design
from pdeep.unified.data import BalancedSampler
from pdeep.unified.config import SCHEMA
from pdeep.unified.streams import coordinates_id
from pdeep.unified_modes.data import Sampler
from pdeep.unified_modes import preflight as research
from pdeep.preparation.feasibility import configuration
from pdeep.preparation.feasibility_training import configurations
from pdeep.preparation.provider import FeasibilityProvider
from pdeep.preparation.preflight import TrainingDraws,run_preflight
from pdeep.preparation.loading import load_dataset,MissingPreparedDraw
from pdeep.preparation.portable import Dataset
from pdeep.preparation.transfer import sync_folder

@pytest.fixture(scope='module')
def virtual_training(tmp_path_factory,certificate):
    # Verified schedule metadata + synthetic labels for adapter UNIT tests only.
    # Real27M-model/4096-query evidence is collected by the bounded script.
    root=tmp_path_factory.mktemp('preflight-draw-unit');c=configuration()
    cases={key:Case(g,p,s,i,c.conditioning_h) for key in c.keys('train') for p,g,s,i in (key,)}
    ids={k:v.case_id for k,v in cases.items()};dataset=SimpleNamespace(config=c)
    sampler=BalancedSampler(SimpleNamespace(data_seed=c.seed),dataset,provider=object());mapping=[];case_ids=[]
    for cursor in range(3000):
        keys=sampler.keys(cursor);case_ids.append([ids[k] for k in keys]);mapping.append([(10000+cursor,slot,64) for slot in range(4)])
    plan={'mapping':torch.tensor(mapping),'case_ids':case_ids,'mapping_identity':'adapter-unit-only'}
    write_artifact(root/'reference_cache','feasibility-plan',{'artifact_type':'diagnostic_bundle','architecture_id':'REFERENCE_ONLY','run_id':'unit-no-numerical-reference'},plan,required=tuple(plan))
    accessed=[];ready=[]
    def evaluate(c,x,t,q):accessed.append((c.case_id,c.split,coordinates_id(x,t,q)));return x[:,0]+t
    def require_case(c):assert c.split=='train'
    return SimpleNamespace(config=c,root=root,norms=Normalizers.fromdict(certificate['normalizers']),index={'feasibility':{'mapping_identity':'adapter-unit-only'}},cache=SimpleNamespace(evaluate=evaluate),require_case=require_case,require_ready=lambda s:ready.append(s),cases=lambda split:tuple(cases.values()) if split=='train' else pytest.fail('non-TRAIN optimizer access'),accessed=accessed,ready=ready)

def test_exact_feasibility_provider_draw_and_prefix(virtual_training):
    d=virtual_training;draws=TrainingDraws(d);case=d.cases('train')[0];cursor,slot=draws.visits[case.case_id][0]
    cohort=tuple(Case(g,p,s,i,d.config.conditioning_h) for p,g,s,i in draws.sampler.keys(cursor))
    expected,ids=draws.provider.batch(cohort,cursor=cursor,count=4096,seed=42)
    batch=draws.preflight_batch(d,case,'INVERSE',4096,'cpu',42)
    torch.testing.assert_close(batch.queries.position,expected.queries.position[slot:slot+1],atol=0,rtol=0)
    torch.testing.assert_close(batch.observations.value,expected.observations.value[slot:slot+1],atol=0,rtol=0)
    small=draws.preflight_batch(d,case,'FORWARD',32,'cpu',42)
    torch.testing.assert_close(small.queries.value,batch.queries.value[:,:32],atol=0,rtol=0)
    assert not small.observations.valid.any() and set(v[1] for v in d.accessed)=={'train'}

def test_shared_preflight_logic_is_bound_without_mutating_research_or_streams(virtual_training,tmp_path,monkeypatch):
    d=virtual_training;configs=configurations(d.config,device='cpu');train=Sampler(configs[0],d,provider=FeasibilityProvider(d));before,ids=train.batch(0);state=train.state_dict(0);rng=digest(tree_identity(capture_rng()));batches=[]
    class Tiny(torch.nn.Module):
        def __init__(self):super().__init__();self.known_field_encoder=torch.nn.Linear(1,1);self.observation=torch.nn.Linear(1,1)
    class Adapter:
        def __init__(self,c):self.cache=SimpleNamespace(clear_queries=lambda:None);self.last={}
        def loss(self,model,batch,**kw):return SimpleNamespace(total=sum(p.square().sum() for p in model.parameters()))
    def accumulated(model,c,cases,make_batch):
        assert len({v.case_id for v in cases})==4
        for case,mode in zip(cases,('FORWARD','INVERSE','HYBRID','INVERSE')):
            b=make_batch(case,mode);assert b.queries.valid.sum()==4096 and case.split=='train';batches.append(b)
        return {'status':'fits','effective_cases':4,'optimizer_steps':0}
    monkeypatch.setattr(research,'create',lambda *a,**k:Tiny());monkeypatch.setattr(research,'Adapter',Adapter)
    monkeypatch.setattr(research,'model_inventory',lambda m:{'trainable_parameters':4,'model_config':{}})
    monkeypatch.setattr(research,'profile',lambda m,b,a,**k:{'status':'fits','optimizer_steps':0})
    monkeypatch.setattr(research,'accumulated_backward',accumulated)
    def old_progress(*a,**k):pytest.fail('preflight updated a prior cell monitor')
    monkeypatch.setattr(d,'progress',old_progress,raising=False)
    ready=d.require_ready
    def readiness(split):
        ready(split);d.progress('Dataset readiness',split=split,seconds=.1026)
    monkeypatch.setattr(d,'require_ready',readiness)
    original_provider=research.Provider
    path,r=run_preflight(configs,d,tmp_path)
    assert d.progress is old_progress
    assert research.Provider is original_provider and r['provider']=='FeasibilityProvider' and r['split']=='train'
    assert len(batches)==24 and set(d.ready)=={'train'} and digest(tree_identity(capture_rng()))==rng
    assert len(r['representative_accumulation_reports'])==3
    assert {c.family for b in batches for c in b.cases}>={'annulus','rectangle','hole','square'}
    after,new_ids=train.batch(0);assert new_ids==ids and train.state_dict(0)==state
    torch.testing.assert_close(before.queries.value,after.queries.value,atol=0,rtol=0)
    assert read_payload(path)[1]['preflight']['mapping_identity']==d.index['feasibility']['mapping_identity']

def test_research_preflight_retains_research_recipe(virtual_training):
    d=virtual_training;case=d.cases('train')[0];b=research.preflight_batch(d,case,'INVERSE',4096,'cpu',42)
    design=physical_design(case,4096,set_id=(SCHEMA,'supervised',42,0,0))
    assert d.accessed[-1][2]==coordinates_id(*design[:3]) and b.observations.valid.sum()==128

def test_warm_local_load_and_integrity_survive_cached_metadata(prepared,certificate,tmp_path,monkeypatch):
    source=tmp_path/'copy';sync_folder(prepared.path,source)
    def forbidden(*a,**k):pytest.fail('ordinary load rebuilt numerical reference')
    monkeypatch.setattr('pdeep.smooth.reference.Reference.__init__',forbidden)
    d=load_dataset(source,certificate);d.require_ready('train');assert not d.verified
    c=d.cases('train')[0];d.require_case(c);loads=d.load_profile['case_payload_loads'];d.require_case(c);assert d.load_profile['case_payload_loads']==loads
    again=load_dataset(source,certificate);assert again is d and again.load_profile['dataset_object_reused']
    rel=next(k for k in d.index['files'] if k.endswith('.pt'));file=source/rel;old=file.stat();raw=bytearray(file.read_bytes());raw[len(raw)//2]^=1;file.write_bytes(raw);os.utime(file,ns=(old.st_atime_ns,old.st_mtime_ns))
    with pytest.raises(ValueError,match='corrupt'):load_dataset(source,certificate)

def test_missing_label_is_exact_explicit_and_never_solved(prepared,monkeypatch):
    d=prepared;c=d.cases('train')[0]
    monkeypatch.setattr('pdeep.smooth.reference.Reference.__init__',lambda *a,**k:pytest.fail('online solve'))
    x=np.array([[.333,.444]]);t=np.zeros(1);q=np.zeros(1,np.int64)
    with pytest.raises(MissingPreparedDraw) as caught:d.cache.evaluate(c,x,t,q)
    r=caught.value.request;assert r['coordinates_id']==coordinates_id(x,t,q) and r['case_key'][2]=='train' and r['case_id']==c.case_id
    assert 'points=1' in str(caught.value) and 'coordinates_id=' in str(caught.value)

def test_explicit_offline_supplement_preserves_original_artifacts(prepared,certificate,tmp_path,monkeypatch):
    from pdeep.preparation import supplements
    source=tmp_path/'supplement';sync_folder(prepared.path,source);d=Dataset(source,certificate);c=d.cases('train')[0]
    x=np.array([[.333,.444]]);t=np.zeros(1);q=np.zeros(1,np.int64)
    with pytest.raises(MissingPreparedDraw) as caught:d.cache.evaluate(c,x,t,q)
    request=tmp_path/'request.json';request.write_text(json.dumps(caught.value.request));before=(d.root/d.index['records'][c.case_id]).read_bytes()
    monkeypatch.setattr(supplements,'FastReference',lambda c:SimpleNamespace(evaluate=lambda x,t,q:np.full(len(x),1.234,np.float64)))
    monkeypatch.setattr(supplements,'quality_check_full',lambda *a,**k:{'passed':True,'scope':'unit mock; not numerical evidence'})
    r=supplements.prepare_label(d.path,request,certificate);assert r['status']=='prepared' and not r['original_case_artifacts_modified']
    restored=Dataset(source,certificate);np.testing.assert_array_equal(restored.cache.evaluate(c,x,t,q),[1.234])
    assert (d.root/d.index['records'][c.case_id]).read_bytes()==before
    assert supplements.prepare_label(d.path,request,certificate)['reference_solves']==0

def test_repeat_restore_uses_verified_local_snapshot_not_source_payloads(prepared,certificate,tmp_path,monkeypatch):
    from pdeep.preparation import workflow
    from pdeep.preparation.transfer import archive_dataset
    root=tmp_path/'local';first=workflow.prepare(prepared.config,root,certificate,action='restore',manifest=prepared.path)
    def forbidden(*a,**k):pytest.fail('matching restored snapshot was recopied or extracted')
    monkeypatch.setattr(workflow,'sync_folder',forbidden)
    again=workflow.prepare(prepared.config,root,certificate,action='restore',manifest=prepared.path)
    assert again is first and again.load_profile['restore']['reused_local_snapshot']
    archive=tmp_path/'transfer.tar.gz';archive_dataset(prepared.path,archive);monkeypatch.setattr(workflow,'extract_archive',forbidden)
    same=workflow.prepare(prepared.config,root,certificate,action='restore',manifest=archive)
    assert same is first and same.load_profile['restore']['reused_local_snapshot']

def test_exact_reported_cross_cpu_hash_is_replayed_not_approximately_matched(certificate,monkeypatch):
    from pdeep.preparation import coordinates
    from pdeep.smooth.data import tokens
    c=Case('annulus','poisson','train',2);sid=(SCHEMA,'encounter',42,339,0)
    native=physical_design(c,64,observation=True,set_id=sid)
    variants=dict(coordinates.arithmetic_variants(c,64,observation=True,set_id=sid,design=native))
    producer=variants['fma-210'];consumer=variants['separate-012']
    expected='077a8e086ef4aae26e775a408b14a9751da821caeb4d115356ab7930fd43e395'
    reported='3bd26d89f5f0bde563c3aef1d8ffb3540726f3feee21124e888bf1e746de0a04'
    assert coordinates_id(*producer[:3])==expected and coordinates_id(*consumer[:3])==reported
    assert np.max(abs(producer[0]-consumer[0]))==np.finfo(float).eps/2
    clean=torch.arange(64,dtype=torch.float64);seen=[]
    def evaluate(c,x,t,q):
        key=coordinates_id(x,t,q);seen.append(key);assert key==expected;return clean.numpy()
    d=SimpleNamespace(config=configuration(),norms=Normalizers.fromdict(certificate['normalizers']),index={},load_profile={},require_case=lambda c:None,cache=SimpleNamespace(_payload=lambda c:{'labels':{expected:clean}},evaluate=evaluate))
    monkeypatch.setattr(coordinates,'physical_design',lambda *a,**kw:consumer)
    p=object.__new__(FeasibilityProvider);p.dataset=d;p.norms=d.norms;p.cache=d.cache
    got,meta=p.token_set(c,64,observation=True,set_id=sid)
    want,wi=tokens(c,64,d.norms,observation=True,set_id=sid,coordinates=producer,clean=clean.numpy())
    for name in vars(got):
        if getattr(got,name) is not None:torch.testing.assert_close(getattr(got,name),getattr(want,name),atol=0,rtol=0)
    assert meta['identity']==wi['identity'] and seen==[expected]

def test_smaller_diagnostic_slices_verified_values_without_new_lookup(virtual_training,monkeypatch):
    d=virtual_training;draws=TrainingDraws(d);case=d.cases('train')[0]
    full=draws.preflight_batch(d,case,'INVERSE',4096,'cpu',42)
    monkeypatch.setattr(d.cache,'evaluate',lambda *a,**k:pytest.fail('a prefix caused another reference lookup'))
    for n in (32,64):
        small=draws.preflight_batch(d,case,'INVERSE',n,'cpu',42)
        for name in vars(full.queries):
            if getattr(full.queries,name) is not None:torch.testing.assert_close(getattr(small.queries,name),getattr(full.queries,name)[:,:n],atol=0,rtol=0)

def test_audit_collects_all_missing_draws_before_any_model(virtual_training,tmp_path,monkeypatch):
    d=virtual_training
    monkeypatch.setitem(d.index,'manifest_identity','unit-only');monkeypatch.setitem(d.index,'certificate_id','unit-only')
    monkeypatch.setattr(d.cache,'_payload',lambda c:{'labels':{}},raising=False)
    chosen=d.cases('train')[:2];draws=TrainingDraws(d)
    monkeypatch.setattr(research,'create',lambda *a,**k:pytest.fail('model allocated before audit'))
    with pytest.raises(ValueError,match='ALL 4 missing'):draws.audit(chosen,tmp_path)
    report=json.loads(next((tmp_path/'preflight-draw-audits').glob('*.json')).read_text())
    assert len(report['required'])==len(report['missing'])==4 and report['model_allocations']==0
    assert report['independent_designs_required']==report['reference_solves']==0 and not draws.prepared

def test_monitor_separates_old_lifetime_from_readiness_and_preceding_work(monkeypatch):
    from pdeep.preparation import monitor
    clock=[0.];monkeypatch.setattr(monitor.time,'monotonic',lambda:clock[0])
    old=monitor.Dashboard(notebook=False);clock[0]=658.33;old.phase('readiness',seconds=.1026)
    assert old.latest['cell_elapsed_seconds']==658.33 and old.latest['reported_stage_seconds']==.1026
    fresh=monitor.Dashboard(notebook=False)
    with fresh.operation('Architecture inventory'):clock[0]+=4
    with fresh.operation('TRAIN readiness'):
        clock[0]+=.1026;fresh.phase('Dataset readiness',seconds=.1026)
    records=fresh.latest['completed_operation_timings']
    assert records[0]['seconds']==4 and records[1]['seconds']==pytest.approx(.1026)
    assert fresh.latest['cell_elapsed_seconds']==pytest.approx(4.1026)

def test_batch_offline_repair_reuses_one_reference_per_case(prepared,certificate,tmp_path,monkeypatch):
    from pdeep.preparation import supplements
    root=tmp_path/'batch';sync_folder(prepared.path,root);d=Dataset(root,certificate);c=d.cases('train')[0];requests=[]
    for point in (.313,.317):
        with pytest.raises(MissingPreparedDraw) as caught:d.cache.evaluate(c,np.array([[point,.449]]),np.zeros(1),np.zeros(1,np.int64))
        requests.append(caught.value.request)
    audit=tmp_path/'audit.json';audit.write_text(json.dumps({'schema':'G1-feasibility-preflight-dependencies-v1','dataset_identity':d.config.identity,'missing':requests}))
    calls=[]
    def reference(case):calls.append(case.case_id);return SimpleNamespace(evaluate=lambda x,t,q:np.full(len(x),2.,np.float64))
    monkeypatch.setattr(supplements,'FastReference',reference)
    monkeypatch.setattr(supplements,'quality_check_full',lambda *a,**k:{'passed':True,'scope':'unit mock only'})
    result=supplements.prepare_label_batch(d.path,audit,certificate)
    assert result['requested_designs']==2 and result['new_full_reference_cases']==1 and len(calls)==1
    assert [r['physical_cases_solved'] for r in result['results']]==[1,0]
    assert supplements.prepare_label_batch(d.path,audit,certificate)['new_full_reference_cases']==0 and len(calls)==1
