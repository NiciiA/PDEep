"""Short CPU arithmetic controls; no new reference certification or research run."""
from pathlib import Path
from types import SimpleNamespace
from dataclasses import asdict,replace
import json
import numpy as np
import pytest,torch
from pdeep.core import digest
from pdeep.artifact_io import write_artifact,read_payload
from pdeep.artifact_tensors import state_identity,tree_identity
from pdeep.artifact_rng import capture_rng
from pdeep.unified.data import BalancedSampler
from pdeep.unified.dataset import inventory
from pdeep.preparation import readiness,checkpointing,validation
from pdeep.preparation.feasibility import parent_configuration
from pdeep.preparation.feasibility_training import TrainConfig,Engine,resume
from pdeep.preparation.loading import MissingPreparedDraw
from pdeep.unified_modes.dashboard import Dashboard
from tests.unified.conftest import ArithmeticDataset
from tests.unified_modes.conftest import config as base_config

@pytest.fixture
def fixture_data(certificate,tmp_path,monkeypatch):
    d=ArithmeticDataset(certificate);d.config=replace(d.config,stream_updates=3,stream_queries=8,stream_validation_queries=8)
    d.root=tmp_path/'data';d.path=d.root/'manifest.json';d.root.mkdir()
    rows=inventory(d.config);ids={tuple(r['key']):r['case_id'] for r in rows}
    sampler=BalancedSampler(SimpleNamespace(data_seed=42),d);mapping=[];case_ids=[]
    for cursor in range(3):
        mapping.append([(100+cursor,slot,64) for slot in range(4)]);case_ids.append([ids[k] for k in sampler.keys(cursor)])
    plan={'mapping':torch.tensor(mapping),'case_ids':case_ids,'mapping_identity':'arithmetic-boundary-control'}
    write_artifact(d.root/'reference_cache','feasibility-plan',{'artifact_type':'diagnostic_bundle','architecture_id':'REFERENCE_ONLY','run_id':'arithmetic-unit-control'},plan,required=tuple(plan))
    d.index={'certificate_id':certificate['certification_id'],'manifest_identity':'arithmetic-unit-control','cases':rows,'feasibility':{'mapping_identity':plan['mapping_identity'],'parent_dataset_identity':parent_configuration(d.config).identity}}
    monkeypatch.setattr(readiness,'verified_folder',lambda *a,**k:(d.path,d.index))
    return d

def config(d,model='C_GEO/FIELD_ONLY'):
    c=base_config(d,model,validation_every=2,checkpoint_every=2)
    return TrainConfig(**asdict(c),reference_parent_identity=parent_configuration(d.config).identity)

def monitor(d):return Dashboard(d.config,notebook=False)

def test_complete_gate_and_strict_missing_batch_before_optimizer(fixture_data,tmp_path,monkeypatch):
    d=fixture_data;c=config(d);seen=[];native=readiness.prepared_design
    def select(data,case,*a,**kw):
        assert case.split!='test';seen.append((case.split,case.pde,case.family,kw['observation']));return native(data,case,*a,**kw)
    monkeypatch.setattr(readiness,'prepared_design',select)
    before=tree_identity(capture_rng());r=readiness.ensure_ready((c,),d,tmp_path/'ok')
    assert r['checked_designs']==24+96 and r['validation_designs']==96 and r['training_designs']==24 and not r['missing']
    assert len({(p,g) for s,p,g,o in seen if s=='validation'})==24 and tree_identity(capture_rng())==before
    seen.clear();readiness.ensure_ready((c,),d,tmp_path/'ok');assert not seen
    def absent(data,case,n,**kw):
        x=np.zeros((n,2));raise MissingPreparedDraw(data,case,x,np.zeros(n),np.zeros(n,dtype=np.int64))
    monkeypatch.setattr(readiness,'prepared_design',absent)
    monkeypatch.setattr(torch.optim.AdamW,'__init__',lambda *a,**k:pytest.fail('optimizer before complete readiness'))
    with pytest.raises(ValueError,match='ALL 120 missing'):Engine(c,d,tmp_path/'missing',monitor=monitor(d))
    report=json.loads(next((tmp_path/'missing/training-readiness').glob('*/*.json')).read_text())
    assert len(report['missing'])==120 and {v['split'] for v in report['missing']}=={'train','validation'}
    assert all((tmp_path/'missing/training-readiness'/d.config.identity/v['request_file']).exists() for v in report['missing'])
    # A receipt is never accepted after content tampering.
    path=next((tmp_path/'ok/training-readiness').glob('*/*.json'));body=json.loads(path.read_text());body['checked_designs']=0;path.write_text(json.dumps(body))
    with pytest.raises(ValueError,match='receipt identity'):readiness.ensure_ready((c,),d,tmp_path/'ok')

@pytest.mark.parametrize('model',('A_GEO/KIN_001','C_GEO/FIELD_ONLY','C_GEO/ALL_001'))
def test_successful_boundary_all_models(fixture_data,tmp_path,model,monkeypatch):
    d=fixture_data;c=config(d,model);calls=[]
    # Keep real validation traversal/labels/metrics; only predictions are reduced
    # in this boundary UNIT test, whose target is persistence rather than accuracy.
    def predict(net,view,obs,q,**kw):calls.append((view.case_id,view.mode));return np.zeros(q.valid.shape[1]),0.
    monkeypatch.setattr(validation.original,'predict',predict)
    e=Engine(c,d,tmp_path/model.split('/')[0],monitor=monitor(d));e.run_until(2,launch=True)
    r,p=read_payload(e.state.last_checkpoint)
    assert p['training_state']['successful']==p['scheduler']['completed']==2 and r['checkpoint_reason']=='validated'
    assert len(calls)==144 and len(set(calls))==144 and {m for _,m in calls}=={'FORWARD','INVERSE','HYBRID'}
    assert e.state.best_checkpoint and e.state.last_checkpoint and len(e.state.validation_history)==1
    assert any('before-validation' in v['path'] for v in e.state.checkpoint_history)
    assert p['scaler'] is None and all(k in p for k in ('optimizer','rng','sampler','monitor_state','history_refs'))

def test_failure_preserves_exact_resume_and_no_repeated_updates(fixture_data,tmp_path,monkeypatch):
    d=fixture_data;c=config(d);native=checkpointing.evaluate
    def fail(*a,**k):raise RuntimeError('intentional validation boundary failure')
    monkeypatch.setattr(checkpointing,'evaluate',fail)
    e=Engine(c,d,tmp_path/'original',monitor=monitor(d))
    with pytest.raises(RuntimeError,match='intentional'):e.run_until(2,launch=True)
    path=e.state.last_checkpoint;r,p=read_payload(path)
    assert r['validation_pending'] and p['training_state']['successful']==2 and e.state.last_checkpoint and e.scheduler.completed==2
    rng_before=tree_identity(capture_rng());optimizer_before=tree_identity(e.optimizer.state_dict())
    monkeypatch.setattr(checkpointing,'evaluate',native)
    monkeypatch.setattr(validation.original,'predict',lambda net,view,obs,q,**kw:(np.zeros(q.valid.shape[1]),0.))
    restored=resume(path,c,d,tmp_path/'resumed',monitor=monitor(d))
    assert isinstance(restored,Engine) and state_identity(restored.model)==state_identity(e.model)
    assert tree_identity(restored.optimizer.state_dict())==optimizer_before and tree_identity(capture_rng())==rng_before
    # Stop exactly at interrupted update: validation still finishes, no update2 repeated.
    restored.run_until(2,launch=True);assert restored.state.successful==restored.state.sampler_cursor==restored.scheduler.completed==2
    assert restored.state.validation_history[-1]['step']==2
    assert read_payload(restored.state.last_checkpoint)[0]['checkpoint_reason']=='validated'
    again=resume(restored.state.last_checkpoint,c,d,tmp_path/'post-validated-resume',monitor=monitor(d))
    assert again.state.best_checkpoint==again.state.last_checkpoint and again.state.best_validation_step==2
    # Same next update, same Adam moments and RNG as uninterrupted continuation.
    e.run_until(2,launch=True);e.session_stop=3;restored.session_stop=3;e.update();restored.update()
    assert state_identity(e.model)==state_identity(restored.model)
    assert tree_identity(e.optimizer.state_dict())==tree_identity(restored.optimizer.state_dict())
    assert e.state.successful==restored.state.successful==3


def test_drive_verification_and_failure_keep_local(fixture_data,tmp_path,monkeypatch):
    from pdeep.artifact_drive import PersistencePolicy
    d=fixture_data;c=config(d);mount=tmp_path/'drive';mount.mkdir()
    policy=PersistencePolicy(drive_enabled=True,drive_mount=str(mount),drive_explicitly_mounted=True,backend='filesystem_test',drive_root=str(mount/'PDEInverseResearch'))
    e=Engine(c,d,tmp_path/'runs',monitor=monitor(d),persistence=policy)
    assert e.state.persistence['Drive_enabled'] and e.state.persistence['Drive_status']=='not_started' and e.state.persistence['storage_probe']['Drive_verified']
    e.session_stop=3;e.update();path=e.save_checkpoint();status=e.state.persistence['last_sync']
    assert status['Drive_verified'] and read_payload(status['Drive_manifest'])[1]['training_state']['successful']==1
    assert Path(path).exists()
    monkeypatch.setattr(checkpointing,'sync',lambda *a:{'Drive_status':'failed','error':'simulated disconnect'})
    second=e.save_checkpoint();assert Path(second).exists() and e.state.persistence['local_status']=='verified' and e.state.persistence['Drive_status']=='failed'
    assert str(second) in e.state.persistence['backlog']


def test_approved_adapter_resume_keeps_original_experiment_identity(fixture_data,tmp_path):
    from pdeep.preparation.compatibility import resume_configurations,approved_adapter
    d=fixture_data;current=config(d);legacy=replace(current,feasibility_source='d58219e5a8eea2f299259f6dbe8fffc64a9aa951209d564bef54d7e94eb42181')
    assert approved_adapter(legacy.feasibility_source) and not approved_adapter('unreviewed-source')
    e=Engine(legacy,d,tmp_path/'old',monitor=monitor(d));e.session_stop=3;e.update();path=e.save_checkpoint()
    selected=resume_configurations((current,),{current.model_key:path})[0]
    assert selected.identity==legacy.identity and selected.run_id==legacy.run_id
    restored=resume(path,selected,d,tmp_path/'restored',monitor=monitor(d))
    assert restored.state.successful==1 and state_identity(restored.model)==state_identity(e.model)


def test_retained_exception_can_be_saved_before_kernel_restart(fixture_data,tmp_path,monkeypatch):
    d=fixture_data;e=Engine(config(d),d,tmp_path/'live',monitor=monitor(d))
    monkeypatch.setattr(checkpointing,'evaluate',lambda *a,**k:(_ for _ in ()).throw(RuntimeError('retained failure')))
    with pytest.raises(RuntimeError) as caught:e.run_until(2,launch=True)
    recovered=checkpointing.recover_live_failure(model_key=e.config.model_key,exception=caught.value)
    assert recovered['successful']==2 and read_payload(recovered['checkpoint'])[1]['scheduler']['completed']==2


def test_reported_validation_hash_replays_existing_exact_design(certificate,monkeypatch):
    from pdeep.preparation import coordinates
    from pdeep.smooth.data import Case,Normalizers
    from pdeep.smooth.bank import physical_design
    from pdeep.unified.streams import coordinates_id
    case=Case('square','poisson','validation',0);sid=('unified-validation',42)
    native=physical_design(case,512,set_id=sid)
    variants=dict(coordinates.arithmetic_variants(case,512,observation=False,set_id=sid,design=native))
    producer=variants['fma-210'];consumer=variants['separate-012']
    expected='49d8c18265ccc6076265ed9fb3a996cd4afb3521180cf8b814337595b37e0f2f'
    assert coordinates_id(*producer[:3])==expected
    assert coordinates_id(*consumer[:3])=='fbdad2015f317547a38533ce51c1985c3a24e6cb1d249c418f0bb3cce38f0e4b'
    # Synthetic values solely test coordinate/value pairing; no reference claim.
    clean=np.arange(512,dtype=np.float64);seen=[]
    def fetch(c,x,t,q):
        seen.append((c.split,coordinates_id(x,t,q)));assert seen[-1]==('validation',expected);return clean
    d=SimpleNamespace(norms=Normalizers.fromdict(certificate['normalizers']),index={},require_case=lambda c:None,cache=SimpleNamespace(_payload=lambda c:{'labels':{expected:clean}},evaluate=fetch))
    monkeypatch.setattr(coordinates,'physical_design',lambda *a,**k:consumer)
    token,_=validation.Provider(d).token_set(case,512,set_id=sid)
    assert len(seen)==1 and torch.equal(token.value[0,:,0],torch.from_numpy(clean).float())
    with pytest.raises(ValueError,match='validation cases'):validation.Provider(d).token_set(Case('square','poisson','test',0),512,set_id=sid)


def test_complete_offline_batch_manifest_accepts_validation_requests_and_checks_shards(tmp_path,monkeypatch):
    from pdeep.preparation import supplements
    from pdeep.artifact_io import atomic_json,sha256
    root=tmp_path/'audit';root.mkdir();request=root/'validation.json'
    row={'case_id':'existing-validation-case','coordinates_id':'exact-key','case_key':['poisson','square','validation',0]}
    atomic_json(request,row)
    body={'signature':{'schema':'G1-complete-training-readiness-v1','dataset':'verified-dataset'},'missing':[{'case_id':row['case_id'],'coordinates_id':row['coordinates_id'],'request_file':request.name,'sha256':sha256(request)}]}
    body['content_digest']=digest(body);audit=root/'audit.json';atomic_json(audit,body)
    monkeypatch.setattr(supplements,'read_index',lambda p:(Path(p),{'dataset_identity':'verified-dataset'}));seen=[]
    def offline(manifest,path,cert,**kw):
        r=json.loads(Path(path).read_text());seen.append(r);assert r['case_key'][2]=='validation';return {'status':'offline-interface-control-only'}
    monkeypatch.setattr(supplements,'prepare_label',offline)
    result=supplements.prepare_label_batch(tmp_path/'dataset.json',audit,{})
    assert result['requested_designs']==1 and seen==[row] and result['new_full_reference_cases']==0
    request.write_text('{}')
    with pytest.raises(ValueError,match='request checksum'):supplements.prepare_label_batch(tmp_path/'dataset.json',audit,{})
