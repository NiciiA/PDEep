"""Wave derivative/loss/orchestration controls. Zero optimizer updates allowed."""
from dataclasses import replace,asdict
from pathlib import Path
import json
import numpy as np
import pytest
import torch
from pdeep.core import ValidationError,digest
from pdeep.learning_diagnostics.wave_verification import analytic_controls,model_controls,reference_controls,verify_wave
from pdeep.learning_diagnostics.wave_study import prepare_wave_study,run_wave_study,recover_wave_checkpoints
from pdeep.learning_diagnostics.kinematic import coordinates,physical_residual
from pdeep.learning_diagnostics.study_controls import configured_adapter,physics_weights,STAGED
from pdeep.learning_diagnostics.engine import DiagnosticConfig
from pdeep.geometric.models import create
from pdeep.learning_diagnostics.metrics import gradient_terms
from pdeep.artifact_tensors import state_identity
from test_optimization_diagnostics import certificate,unit_data,cfg
from test_diagnostic_suite import synthetic_runtime

@pytest.fixture(autouse=True)
def no_training(monkeypatch):
    def fail(*a,**kw):pytest.fail('optimizer updates / numerical certification forbidden')
    for cls in (torch.optim.AdamW,torch.optim.Adam,torch.optim.SGD):monkeypatch.setattr(cls,'step',fail)
    monkeypatch.setattr('pdeep.smooth.qualification.ensure_compatible_certificate',fail)


def test_independent_manufactured_square_rectangle():
    rows=analytic_controls();assert all(r['passed'] for r in rows),[r for r in rows if not r['passed']]
    assert len(rows)>=22 and any('rectangle physical spatial gradient'==r['test'] for r in rows)


def test_actual_model_jacobians_physical_heads_fd_and_chunking(unit_data):
    model=create('C_GEO',unit_data.normalizers,'verification',seed=42)
    before=state_identity(model);rows=model_controls(model,unit_data)
    assert all(r['passed'] for r in rows),[r for r in rows if not r['passed']]
    assert before==state_identity(model)


def test_reference_floor_representation_not_plot_image(unit_data):
    rows,floor=reference_controls(unit_data,count=5)
    assert all(r['passed'] for r in rows),[r for r in rows if not r['passed']]
    assert floor['kinematic_sv_normalized_MSE']<1e-17
    assert 'element interiors' in floor['strong_scope'] and floor['weak']['weak']>=0


def test_kinematic_loss_exact_weight_parameter_paths_and_field_unchanged(unit_data):
    fo=cfg(unit_data,objective_variant='FIELD_ONLY',physics_weights=(0.,0.,0.))
    kin=replace(fo,objective_variant='KIN_001',kinematic_weight=.001,kinematic_queries=17)
    m=create('C_GEO',unit_data.normalizers,'verification',seed=42);before=state_identity(m)
    expected=configured_adapter(fo).loss(m,unit_data.batch())
    adapter=configured_adapter(kin);r=adapter.loss(m,unit_data.batch())
    assert set(r.unweighted)=={'field','auxiliary','kinematic'}
    assert torch.allclose(r.unweighted['field'],expected.unweighted['field'])
    assert torch.allclose(r.total,expected.total+.001*r.unweighted['kinematic'])
    b=unit_data.batch();cs=m.prepare_context(b.specs,b.observations);x,t=coordinates(unit_data.case,17,42,0)
    xx=torch.tensor(x,dtype=torch.float32);tt=torch.tensor(t[:,None],dtype=torch.float32,requires_grad=True)
    exact=((physical_residual(m,cs[0],xx,tt)/unit_data.normalizers.scale('wave','v'))**2).mean()
    assert torch.allclose(r.unweighted['kinematic'],exact,rtol=5e-6,atol=1e-6)
    gs=torch.autograd.grad(r.unweighted['kinematic'],(m.u_head.weight,m.v_head.weight,m.decoder.input.weight))
    assert all(g.isfinite().all() and g.abs().max()>0 for g in gs)
    report=gradient_terms(m,b,adapter)
    assert report['raw_gradient_norms']['kinematic']>0
    assert np.isclose(report['weighted_gradient_norms']['kinematic'],.001*report['raw_gradient_norms']['kinematic'],rtol=1e-5)
    assert report['optimizer_updates']==0 and state_identity(m)==before


def test_kinematic_stream_separate_counter_resume_and_chunk_gradients(unit_data):
    from pdeep.learning_diagnostics.queries import coordinates as supervised
    c=unit_data.case;x,t=coordinates(c,512,42,7)
    assert (t==0).any() and (t==1).any() and ((t>0)&(t<.1)).any() and ((t>.1)&(t<.5)).any() and ((t>.5)&(t<1)).any()
    assert np.array_equal(x,coordinates(c,512,42,7)[0]) and not np.array_equal(x,coordinates(c,512,42,8)[0])
    assert not np.array_equal(x,supervised(c,512,42,7,revision='continuous-queries-v3')[0])
    xx=unit_data.diagnostic.position[0].numpy();assert not set(a.tobytes() for a in x.astype(np.float32))&set(a.tobytes() for a in xx)
    cfg0=cfg(unit_data,objective_variant='KIN_001',physics_weights=(0.,0.,0.),kinematic_weight=.001,kinematic_queries=17)
    restored=DiagnosticConfig(**json.loads(json.dumps(asdict(cfg0))));assert restored.identity==cfg0.identity
    a=create('C_GEO',unit_data.normalizers,'verification',seed=42);b=create('C_GEO',unit_data.normalizers,'verification',seed=42)
    aa=configured_adapter(cfg0,successful=7);bb=configured_adapter(replace(restored,query_chunk=7),successful=7)
    ra=aa.loss(a,unit_data.batch());rb=bb.loss(b,unit_data.batch());ra.total.backward();rb.total.backward()
    assert aa.last['kinematic'][0]['identity']==bb.last['kinematic'][0]['identity']
    assert torch.allclose(ra.total,rb.total,rtol=1e-5,atol=1e-6)
    for p,q in zip(a.parameters(),b.parameters()):
        if p.grad is not None:assert torch.allclose(p.grad,q.grad,rtol=2e-4,atol=4e-6)

@pytest.mark.parametrize('count',[8192,16384,32768,65536])
def test_preview_counts_gate_schedule_and_no_optimizer(certificate,tmp_path,monkeypatch,count):
    monkeypatch.setattr(torch.optim.AdamW,'__init__',lambda *a,**kw:pytest.fail('preview constructs optimizer'))
    p=prepare_wave_study(certificate=certificate,root=tmp_path,device='cpu',query_count=count)
    assert tuple(p.settings.variant_settings)==('FIELD_ONLY','KIN_001','STAGED_001')
    assert p.settings.variant_settings['KIN_001']['kinematic_queries']==512
    assert p.settings.variant_settings['STAGED_001']['physics_schedule']=='staged_300_700'
    assert run_wave_study(p,None)['optimizer_updates']==0
    with pytest.raises(ValidationError,match='verification'):run_wave_study(p,{},run_mode='train')


def test_staged_old_default_and_explicit_other_budget(unit_data,certificate,tmp_path):
    c=cfg(unit_data,objective_variant='STAGED_001',**STAGED)
    assert physics_weights(c,300)==(0.,)*3 and physics_weights(c,500)==(.0005,)*3 and physics_weights(c,700)==(.001,)*3
    p=prepare_wave_study(certificate=certificate,root=tmp_path,device='cpu',updates=500)
    cc=replace(c,capacity='medium',preset='bounded',updates=500,physics_schedule=p.settings.variant_settings['STAGED_001']['physics_schedule'])
    assert physics_weights(cc,150)==(0.,)*3 and physics_weights(cc,250)==(.0005,)*3 and physics_weights(cc,350)==(.001,)*3


def test_complete_verification_binds_case_config_source_no_optimizer(unit_data,certificate,tmp_path,monkeypatch):
    monkeypatch.setattr(torch.optim.AdamW,'__init__',lambda *a,**kw:pytest.fail('derivative verification constructs optimizer'))
    p=prepare_wave_study(certificate=certificate,root=tmp_path,device='cpu',capacity='verification',query_count=32,query_chunk=17,updates=2)
    m=create('C_GEO',unit_data.normalizers,'verification',seed=42)
    r=verify_wave(p,data=unit_data,model=m,reference_count=5)
    assert r['passed'] and r['optimizer_updates']==0 and not r['qualification_executed']
    assert r['plan_identity']==p.identity and r['observation_identity']==unit_data.identities['observation']
    inventory=recover_wave_checkpoints(p,r);assert inventory['missing']==['FIELD_ONLY','KIN_001','STAGED_001']
    with pytest.raises(ValidationError):recover_wave_checkpoints(p,{**r,'plan_identity':'stale'})


def test_four_cells_safe_defaults_and_evaluation_no_training():
    import nbformat,ast
    nb=nbformat.read(Path(__file__).resolve().parents[1]/'notebooks/02_train.ipynb',4)
    cells=[c for c in nb.cells if 'wave_kinematic' in c.metadata.get('tags',[])]
    assert len(cells)==4 and all(c.cell_type=='code' for c in cells)
    assert 'WAVE_RUN_MODE = "preview"' in cells[0].source
    assert 'verify_wave(' in cells[1].source and 'run_wave_study(' in cells[2].source and 'compare_wave(' in cells[3].source
    for c in cells:
        ast.parse(c.source)
        for term in ('Button(','input(','Thread(','weights_only=False'):assert term not in c.source
    for term in ('run_wave_study(','Adam','launch_diagnostic(','Engine('):assert term not in cells[3].source


def test_mocked_all_variants_once_one_dashboard_resume(certificate,tmp_path,synthetic_runtime,monkeypatch):
    from pdeep.learning_diagnostics import wave_study
    # Synthetic events only: optimizer.step and certificate qualification forbidden.
    monkeypatch.setattr(wave_study,'recover_wave_checkpoints',lambda *a,**kw:{'selected':{},'missing':list(wave_study.VARIANTS)})
    p=prepare_wave_study(certificate=certificate,root=tmp_path,device='cpu',capacity='verification',updates=2,query_count=32,query_chunk=17,amp_mode='off')
    verification={'passed':True,'plan_identity':p.identity,'source':p.source}
    r=run_wave_study(p,verification,run_mode='train',notebook=False)
    runs=r['entries']['c_physics_ablation']['runs'];assert list(runs)==list(wave_study.VARIANTS)
    assert all(run['last_successful_update']==run['session_optimizer_updates']==2 for run in runs.values())
    before=list(synthetic_runtime.updates)
    pp=prepare_wave_study(certificate=certificate,root=tmp_path,device='cpu',capacity='verification',updates=2,query_count=32,query_chunk=17,amp_mode='off')
    rr=run_wave_study(pp,verification,run_mode='train',notebook=False)
    assert synthetic_runtime.updates==before
    assert all(run['session_optimizer_updates']==0 and run['result_origin']=='restored from completed checkpoint' for run in rr['entries']['c_physics_ablation']['runs'].values())
    # Reject a corrupt newest immutable snapshot, recover an earlier verified
    # revision, without touching any real/previous checkpoint.
    from pdeep.artifact_io import read_manifest
    latest=Path(rr['snapshot']);record=read_manifest(latest)
    pt=latest.parent/next(iter(record['files']));pt.write_bytes(pt.read_bytes()+b'explicit synthetic corruption')
    with pytest.warns(RuntimeWarning,match='latest fully verified earlier snapshot'):
        recovered=prepare_wave_study(certificate=certificate,root=tmp_path,device='cpu',capacity='verification',updates=2,query_count=32,query_chunk=17,amp_mode='off')
    assert recovered.action=='resume' and recovered.snapshot_recovery_failures
    assert synthetic_runtime.updates==before


def test_recovery_checks_actual_payload_data_case_schedule_and_initialization(unit_data,certificate,tmp_path):
    from pdeep.learning_diagnostics.wave_study import member_config
    from pdeep.learning_diagnostics.engine import SCHEMA
    from pdeep.smooth.training import implementation_identity
    from pdeep.artifact_io import write_artifact,read_manifest
    p=prepare_wave_study(certificate=certificate,root=tmp_path,device='cpu',capacity='verification',updates=2,query_count=32,query_chunk=17,amp_mode='off')
    m=create('C_GEO',unit_data.normalizers,'verification',seed=42)
    v={'passed':True,'plan_identity':p.identity,'source':p.source,'physical_case':unit_data.case.case_id,'data_identity':unit_data.identity,
       'observation_identity':unit_data.identities['observation'],'model_config':asdict(m.config),'initialization_identity':state_identity(m)}
    c=member_config(p,unit_data.identity,'FIELD_ONLY','explicit-synthetic-recovery-fixture')
    # Synthetic verified artifact for inventory control only; never represents a
    # trained-model result or feeds a real optimizer/resume.
    meta={'artifact_type':'training_checkpoint','architecture_id':'C_GEO','run_id':c.run_id,'benchmark':'G1_smooth_v1','scientific_schema':SCHEMA,
          'config':asdict(c),'diagnostic_source_id':p.source,'implementation_identity':implementation_identity(),
          'execution_qualification':certificate,'fixed_case':unit_data.case.manifest,'fixed_data_identity':unit_data.identity,
          'runtime':{'dependencies':{'torch':torch.__version__}},'successful_update':2}
    payload={'model':m.state_dict(),'model_config':asdict(m.config),'optimizer':{},'scheduler':{},'rng':{},'sampler':{'observations':unit_data.identities['observation']},
             'training_state':{'successful':2},'normalizers':unit_data.normalizers.asdict()}
    path=write_artifact(tmp_path/'learning_diagnostics'/'synthetic-candidate','diagnostic-update-00000002',meta,payload)
    r=recover_wave_checkpoints(p,v);assert r['selected']['FIELD_ONLY']['successful']==2
    p.action='resume';p.snapshot={'entries':{'c_physics_ablation':{'runs':p.reused_runs}}}
    resumed=recover_wave_checkpoints(p,v)
    assert resumed['selected']['FIELD_ONLY']['origin']=='restored from completed checkpoint'
    assert 'FIELD_ONLY' not in resumed['missing'] and resumed['optimizer_updates']==0
    p.action='fresh';p.snapshot=None
    # Distinct case/observations/config/source are never matched by filename.
    for key,value in [('physical_case','wrong'),('observation_identity','wrong'),('data_identity','wrong'),('initialization_identity','wrong')]:
        if key=='initialization_identity':continue # raw checkpoints derive deterministic initialization from guarded source/seed
        rr=recover_wave_checkpoints(p,{**v,key:value});assert not rr['selected'] and rr['rejected']
    c2=replace(c,query_chunk=16)
    bad=write_artifact(tmp_path/'learning_diagnostics'/'wrong-chunk','diagnostic-update-00000002',{**meta,'config':asdict(c2),'run_id':c2.run_id},payload)
    assert any('configuration mismatch' in row['reason'] for row in recover_wave_checkpoints(p,v)['rejected'])
    record=read_manifest(path);pt=path.parent/next(iter(record['files']));pt.unlink()
    rr=recover_wave_checkpoints(p,v);assert not rr['selected'] and rr['rejected']


def test_evaluation_only_comparison_metrics_and_compact_figures(unit_data,certificate,tmp_path,monkeypatch):
    from pdeep.learning_diagnostics import wave_comparison as wc
    from pdeep.learning_diagnostics import wave_study as ws
    from pdeep.learning_diagnostics.consistency import summarize,TIMES
    from pdeep.learning_diagnostics.suite import _checkpoint_path
    from pdeep.artifact_io import write_artifact,read_manifest
    from pdeep.artifact_drive import PersistencePolicy
    # Explicitly mocked numerical/model metrics; exercise real report reduction,
    # plotting and verified filesystem Drive copier with NO optimizer constructor.
    monkeypatch.setattr(torch.optim.AdamW,'__init__',lambda *a,**kw:pytest.fail('evaluation constructs optimizer'))
    drive=tmp_path/'drive';drive.mkdir();policy=PersistencePolicy(True,str(drive),True,backend='filesystem_test')
    p=prepare_wave_study(certificate=certificate,root=tmp_path/'local',persistence=policy,device='cpu',capacity='verification',updates=2,query_count=32,query_chunk=17)
    runs={};models={};coords=np.array([[.17,.19],[.73,.21],[.31,.84],[.77,.69]])
    x=np.tile(coords,(len(TIMES),1));t=np.repeat(TIMES,len(coords));truth=np.sin(np.pi*x[:,0])*np.sin(np.pi*x[:,1])*np.cos(t)
    for j,label in enumerate(ws.VARIANTS):
        path=write_artifact(tmp_path/'synthetic-reports'/label,'model',{'artifact_type':'training_checkpoint','architecture_id':'C_GEO','run_id':label},
             {'training_state':{'elapsed_seconds':float(j+1)}})
        r=read_manifest(path);runs[label]={'checkpoint':str(path),'checkpoint_id':r['artifact_id'],'status':'completed','last_successful_update':2}
    monkeypatch.setattr(ws,'recover_wave_checkpoints',lambda *a,**kw:{'selected':{},'missing':[]})
    physical_case=unit_data.case.case_id;obs=unit_data.identities['observation']
    def numerical(paths,cert,root,**kw):
        root.mkdir(parents=True,exist_ok=True)
        for j,label in enumerate(paths):
            values={'u':truth+.001*(j+1),'v':2*truth+.002*(j+1),'kinematic':np.full(len(t),.003),'dynamic':np.full(len(t),.04)}
            exact={**values,'u':truth,'v':2*truth,'kinematic':np.zeros(len(t)),'dynamic':np.zeros(len(t))}
            report=summarize(values,exact,t,unit_data.case,unit_data.normalizers)
            floor=summarize(exact,exact,t,unit_data.case,unit_data.normalizers)
            np.savez(root/(digest(label)[:12]+'.npz'),position=x,time=t,**values,truth_u=truth,truth_v=2*truth)
            models[label]={'updates':2,'physical_case':physical_case,'observation_identity':obs,'pointwise':report,'reference_floor':floor,
                'weak':{'weak':.01,'boundary':.02,'initial':.03},'reference_weak':{'weak':.0006},
                'reconstruction':{'sensors':{'u':{'standardized_MSE':.1},'v':{'standardized_MSE':.2}}}}
        return {'models':models,'missing':{},'comparison':'paired','optimizer_updates':0}
    monkeypatch.setattr(wc,'evaluate_wave_checkpoints',numerical)
    report=wc.compare_wave(p,{'passed':True,'plan_identity':p.identity,'physical_case':physical_case,'observation_identity':obs},
         result={'entries':{'c_physics_ablation':{'runs':runs}}},display=False,count=4)
    assert len(report['rows'])==6 and len(report['figures'])==4 and report['sync']['Drive_verified']
    assert 'Matched trained checkpoints evaluated' in report['decision']
    assert np.isclose(report['ratios_to_FIELD_ONLY']['KIN_001']['u_MSE_ratio'],4.)
    assert report['sync']['backend']=='filesystem_test' and report['optimizer_updates']==0 and not report['optimizer_constructed']
    assert all(row['session_optimizer_updates']==0 for row in report['rows'])
    assert np.isclose(report['rows'][0]['kinematic_sv_MSE'],(.003/unit_data.normalizers.scale('wave','v'))**2)


def test_real_three_arm_initialization_observation_and_supervision_pairing(unit_data,certificate,tmp_path,monkeypatch):
    from pdeep.learning_diagnostics.workflows import prepare_diagnostic
    from pdeep.learning_diagnostics.queries import QuerySampler
    from pdeep.learning_diagnostics.data import token_id
    monkeypatch.setattr('pdeep.learning_diagnostics.workflows.prepare_data',lambda *a,**kw:unit_data)
    spec=prepare_wave_study(certificate=certificate,root=tmp_path,capacity='verification',device='cpu',updates=2,query_count=32,query_chunk=17)
    plan=prepare_diagnostic(model='C_GEO',experiment='c_physics_ablation',certificate=certificate,root=tmp_path,capacity='verification',updates=2,device='cpu',seed=42,
        query_mode='dense_resampled',query_count=32,query_chunk=17,query_revision='continuous-queries-v3',variant_settings=spec.settings.variant_settings)
    assert tuple(c.objective_variant for c in plan.configs)==('FIELD_ONLY','KIN_001','STAGED_001')
    assert len(set(plan.initial_model_ids))==1 and len({m.u_head.weight.data_ptr() for m in plan.models})==3
    ids=[]
    for c in plan.configs:
        sampler=QuerySampler(unit_data,c);batch,_=sampler.batch(7)
        ids.append(token_id(batch.queries));assert token_id(batch.observations)==unit_data.identities['observation']
    assert len(set(ids))==1
