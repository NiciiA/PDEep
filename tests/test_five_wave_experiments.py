"""A/C matched wave extension: nonoptimizing verification and synthetic events."""
from dataclasses import asdict,replace
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
import torch
from pdeep.core import ValidationError,digest
from pdeep.artifact_tensors import state_identity
from pdeep.artifact_io import write_artifact,read_manifest,read_payload
from pdeep.learning_diagnostics.wave_study import (prepare_wave_experiments,verify_wave_experiments,recover_wave_experiments,
    run_wave_experiments,prepare_wave_study,member_config,recover_wave_checkpoints,EXPERIMENTS)
from pdeep.learning_diagnostics.study_controls import configured_adapter
from pdeep.learning_diagnostics.engine import approved_diagnostic,SCHEMA,DiagnosticConfig
from pdeep.learning_diagnostics.suite import _member_label
from pdeep.learning_diagnostics.queries import QuerySampler
from pdeep.learning_diagnostics.data import token_id
from pdeep.geometric.models import create
from pdeep.geometric.physics import Adapter
from pdeep.study_monitoring import WaveDashboard
from test_optimization_diagnostics import certificate,unit_data
from test_diagnostic_suite import synthetic_runtime

@pytest.fixture(autouse=True)
def no_training(monkeypatch):
    def fail(*a,**kw):pytest.fail('no optimizer steps or recertification allowed')
    for cls in (torch.optim.Adam,torch.optim.AdamW,torch.optim.SGD):monkeypatch.setattr(cls,'step',fail)
    monkeypatch.setattr('pdeep.smooth.qualification.ensure_compatible_certificate',fail)


def plans(certificate,tmp_path,**kw):
    return prepare_wave_experiments(certificate=certificate,root=tmp_path,capacity='verification',device='cpu',updates=2,query_count=32,query_chunk=17,amp_mode='off',**kw)


def test_five_preview_same_case_budget_no_optimizer(certificate,tmp_path,monkeypatch):
    monkeypatch.setattr(torch.optim.AdamW,'__init__',lambda *a,**kw:pytest.fail('preview optimizer constructor'))
    ps=plans(certificate,tmp_path)
    assert tuple(p.model+'/'+v for p in ps for v in p.settings.variant_settings)==EXPERIMENTS
    assert len({p.recipes[0]['physical_case'] for p in ps})==1
    assert all((p.settings.updates,p.settings.query_count,p.settings.query_chunk,p.settings.seed)==(2,32,17,42) for p in ps)
    assert run_wave_experiments(ps,None)['optimizer_updates']==0
    assert recover_wave_experiments(ps,{p.model:{'passed':True,'plan_identity':p.identity} for p in ps})['A_GEO']['missing']==['BASELINE','KIN_001']
    with pytest.raises(ValidationError):run_wave_experiments(ps,{},run_mode='train')
    with pytest.raises(ValidationError):plans(certificate,tmp_path,experiments=['A_GEO/BASELINE','C_GEO/FIELD_ONLY','A_GEO/KIN_001'])


def test_real_architecture_pairing_original_A_loss_and_kinematic_gradient(unit_data,certificate,tmp_path,monkeypatch):
    from pdeep.learning_diagnostics.workflows import prepare_diagnostic
    monkeypatch.setattr('pdeep.learning_diagnostics.workflows.prepare_data',lambda *a,**kw:unit_data)
    ids=[];observations=[];modelids={}
    for plan in plans(certificate,tmp_path):
        prepared=prepare_diagnostic(model=plan.model,experiment=plan.experiments[0],certificate=certificate,root=tmp_path,capacity='verification',updates=2,device='cpu',seed=42,
            query_mode='dense_resampled',query_count=32,query_chunk=17,query_revision='continuous-queries-v3',variant_settings=plan.settings.variant_settings)
        modelids[plan.model]=prepared.initial_model_ids
        assert len(set(prepared.initial_model_ids))==1
        assert len({m.u_head.weight.data_ptr() for m in prepared.models})==len(prepared.models)
        for c,m in zip(prepared.configs,prepared.models):
            sampler=QuerySampler(unit_data,c);batch,_=sampler.batch(7)
            ids.append(token_id(batch.queries));observations.append(token_id(batch.observations))
            if plan.model=='A_GEO':
                r=configured_adapter(c).loss(m,unit_data.batch())
                original=Adapter('A_GEO').loss(m,unit_data.batch())
                assert torch.allclose(r.unweighted['field'],original.unweighted['field'],rtol=4e-6,atol=1e-6)
                assert torch.allclose(r.unweighted['auxiliary'],original.unweighted['auxiliary'],rtol=4e-6,atol=1e-6)
                if c.objective_variant=='BASELINE':assert torch.allclose(r.total,original.total,rtol=4e-6,atol=1e-6)
                else:
                    assert set(r.unweighted)=={'field','auxiliary','kinematic'}
                    assert torch.allclose(r.total,original.total+.001*r.unweighted['kinematic'],rtol=4e-6,atol=1e-6)
                    gs=torch.autograd.grad(r.unweighted['kinematic'],(m.u_head.weight,m.v_head.weight,m.decoder.input.weight))
                    assert all(g.isfinite().all() and g.abs().max()>0 for g in gs)
    assert len(set(ids))==len(set(observations))==1
    assert modelids['A_GEO'][0]!=modelids['C_GEO'][0]


def test_derivative_gates_both_architectures_no_optimizer(unit_data,certificate,tmp_path,monkeypatch):
    monkeypatch.setattr(torch.optim.AdamW,'__init__',lambda *a,**kw:pytest.fail('derivative optimizer constructor'))
    ps=plans(certificate,tmp_path);reports=verify_wave_experiments(ps,data=unit_data,reference_count=4)
    assert set(reports)=={'A_GEO','C_GEO'} and all(r['passed'] for r in reports.values()),reports
    assert len({r['evaluation_identity'] for r in reports.values()})==1
    assert all(r['optimizer_updates']==0 and not r['qualification_executed'] for r in reports.values())
    assert any(row['test']=='A decoder time Jacobian off-diagonal' for row in reports['A_GEO']['rows'])
    assert all(r['gradient_scale_diagnostics']['raw_gradient_norms']['kinematic']>0 for r in reports.values())


def test_five_sequential_order_one_handle_and_completed_resume(certificate,tmp_path,synthetic_runtime):
    ps=plans(certificate,tmp_path)
    reports={p.model:{'passed':True,'plan_identity':p.identity,'source':p.source} for p in ps}
    # Scheduling/persistence control only: no real model updates; fake event fixture.
    from unittest.mock import patch
    handles=[];labels=[];handle=object()
    def render(text,previous):
        handles.append(previous)
        try:label=json.loads(text.split('\nRecent events:')[0])['experiment']
        except (ValueError,KeyError):label=None
        if label and label.startswith('Variant '):labels.append(label)
        return handle
    monitor=WaveDashboard(ps,notebook=False,interval=0,renderer=render)
    with patch('pdeep.learning_diagnostics.wave_study.recover_wave_checkpoints',return_value={'selected':{},'missing':[]}):
        r=run_wave_experiments(ps,reports,run_mode='train',notebook=False,monitor=monitor)
    assert r['status']=='completed' and r['session_optimizer_updates']==10
    observed=[]
    for label in labels:
        if label not in observed:observed.append(label)
    assert observed==[f'Variant {i}/5: {name}' for i,name in enumerate(EXPERIMENTS,1)]
    assert handles[0] is None and all(item is handle for item in handles[1:]) and not monitor.errors
    assert len(synthetic_runtime.calls)==5
    before=list(synthetic_runtime.updates);again=plans(certificate,tmp_path,checkpoint_lookup='explicit',action='resume',resume_paths={a:r['architectures'][a]['snapshot'] for a in ('A_GEO','C_GEO')})
    with patch('pdeep.learning_diagnostics.wave_study.recover_wave_checkpoints',return_value={'selected':{},'missing':[]}):
        rr=run_wave_experiments(again,reports,run_mode='train',notebook=False)
    assert rr['session_optimizer_updates']==0 and synthetic_runtime.updates==before


def test_trusted_C_kin_checkpoint_source_continues_unchanged(unit_data,certificate,tmp_path):
    ps=plans(certificate,tmp_path);c=member_config(ps[1],unit_data.identity,'KIN_001','previous-C-kin')
    c=replace(c,diagnostic_source_id='2fe9e8e84dacbfe8a25d7e6158eecd5442d94ae5200fad0be0e74e5c9547eff7')
    assert approved_diagnostic(c)
    assert not approved_diagnostic(replace(c,diagnostic_source_id='unapproved'))


def test_verified_A_FULL_alias_preserves_original_config_and_payload(unit_data,certificate,tmp_path):
    from pdeep.smooth.training import implementation_identity
    p=plans(certificate,tmp_path)[0]
    model=create('A_GEO',unit_data.normalizers,'verification',seed=42)
    c=replace(member_config(p,unit_data.identity,'BASELINE','existing-A-wave'),objective_variant='FULL')
    meta={'artifact_type':'training_checkpoint','architecture_id':'A_GEO','run_id':c.run_id,'benchmark':'G1_smooth_v1','scientific_schema':SCHEMA,
          'config':asdict(c),'diagnostic_source_id':c.diagnostic_source_id,'implementation_identity':implementation_identity(),
          'execution_qualification':certificate,'fixed_case':unit_data.case.manifest,'fixed_data_identity':unit_data.identity,
          'runtime':{'dependencies':{'torch':torch.__version__}},'successful_update':2}
    # Explicit synthetic serialization control, not a claim of trained weights.
    payload={'model':model.state_dict(),'model_config':asdict(model.config),'normalizers':unit_data.normalizers.asdict(),
             'optimizer':{'sentinel':'preserve'},'scheduler':{'sentinel':7},'rng':{'sentinel':42},'sampler':{'observations':unit_data.identities['observation']},'training_state':{'successful':2}}
    from wave_checkpoint_controls import states
    payload.update(states(model,c,2,unit_data.identities['observation']))
    path=write_artifact(tmp_path/'learning_diagnostics'/'old-A','diagnostic-update-00000002',meta,payload)
    original=path.read_bytes();v={'passed':True,'plan_identity':p.identity,'physical_case':unit_data.case.case_id,'data_identity':unit_data.identity,
        'observation_identity':unit_data.identities['observation'],'model_config':asdict(model.config),'initialization_identity':state_identity(model)}
    p.wave_lookup='explicit';p.wave_resume_paths={'BASELINE':str(path)}
    recovered=recover_wave_checkpoints(p,v)
    assert recovered['selected']['BASELINE']['successful']==2
    run=p.reused_runs['BASELINE'];assert run['objective_alias']=='A_FULL_is_BASELINE'
    assert _member_label(c,p.reused_runs)=='BASELINE' and c.objective_variant=='FULL' and c.identity==run['config_identity']
    assert path.read_bytes()==original
    from pdeep.artifact_tensors import tree_identity
    _,saved=read_payload(path)
    assert digest(tree_identity(saved['optimizer']))==digest(tree_identity(payload['optimizer']))
    assert digest(tree_identity(saved['rng']))==digest(tree_identity(payload['rng']))
    with pytest.raises(ValidationError):_member_label(replace(c,architecture='C_GEO'),{'BASELINE':{**run,'config_identity':replace(c,architecture='C_GEO').identity}})


def test_combined_evaluation_five_models_clipping_cost_and_own_baselines(unit_data,certificate,tmp_path,monkeypatch):
    from pdeep.learning_diagnostics import wave_comparison as wc,wave_study as ws
    from pdeep.learning_diagnostics.consistency import summarize,TIMES
    monkeypatch.setattr(torch.optim.AdamW,'__init__',lambda *a,**kw:pytest.fail('evaluation constructs optimizer'))
    ps=plans(certificate,tmp_path);outputs={'architectures':{}};models={}
    for p in ps:
        rr={}
        for label in p.settings.variant_settings:
            full=p.model+'/'+label;i=EXPERIMENTS.index(full)
            path=write_artifact(tmp_path/'synthetic-evaluation'/p.model/label,'checkpoint',
                {'artifact_type':'training_checkpoint','architecture_id':p.model,'run_id':full},
                {'training_state':{'elapsed_seconds':100*(i+1)}})
            meta=read_manifest(path)
            rr[label]={'checkpoint':str(path),'checkpoint_id':meta['artifact_id'],'report':{'stability':{'clipping_fraction':.2,'clipping_records':2,'step_median_seconds':.3,'step_p95_seconds':.4}},'session_optimizer_updates':0}
        outputs['architectures'][p.model]={'entries':{p.experiments[0]:{'runs':rr}}}
    monkeypatch.setattr(ws,'recover_wave_experiments',lambda *a,**kw:pytest.fail('evaluation must never trigger recovery'))
    xx=np.array([[.17,.19],[.73,.21],[.31,.84],[.77,.69]])
    x=np.tile(xx,(len(TIMES),1));t=np.repeat(TIMES,len(xx));truth=np.sin(np.pi*x[:,0])*np.sin(np.pi*x[:,1])*np.cos(t)
    def numeric(paths,cert,root,**kw):
        assert tuple(paths)==EXPERIMENTS
        root.mkdir(parents=True,exist_ok=True)
        for i,label in enumerate(paths):
            val={'u':truth+.001*(i+1),'v':2*truth+.002*(i+1),'kinematic':np.full(len(t),.003),'dynamic':np.full(len(t),.04)}
            exact={**val,'u':truth,'v':2*truth,'kinematic':np.zeros(len(t)),'dynamic':np.zeros(len(t))}
            np.savez(root/(digest(label)[:12]+'.npz'),position=x,time=t,**val,truth_u=truth,truth_v=2*truth)
            models[label]={'updates':2,'physical_case':unit_data.case.case_id,'observation_identity':unit_data.identities['observation'],'coordinates_identity':'identical-queries',
                'pointwise':summarize(val,exact,t,unit_data.case,unit_data.normalizers),'reference_floor':summarize(exact,exact,t,unit_data.case,unit_data.normalizers),
                'weak':{'weak':.01,'initial':.03,'boundary':.02},'reference_weak':{'weak':.0006},'reconstruction':{'sensors':{'u':{'standardized_MSE':.1},'v':{'standardized_MSE':.2}}}}
        return {'models':models,'missing':{},'comparison':'paired'}
    monkeypatch.setattr(wc,'evaluate_wave_checkpoints',numeric)
    v={p.model:{'passed':True,'plan_identity':p.identity,'physical_case':unit_data.case.case_id,'observation_identity':unit_data.identities['observation']} for p in ps}
    r=wc.compare_wave_experiments(ps,v,result=outputs,display=False,count=4)
    assert len(r['rows'])==10 and len(r['figures'])==4 and not r['optimizer_constructed'] and r['optimizer_updates']==0
    assert all(row['clipping_fraction']==.2 and row['step_median_seconds']==.3 and row['late_standardized_MSE']>0 for row in r['rows'])
    ratios=r['ratios_to_baseline']
    assert ratios['A_GEO/KIN_001']['baseline']=='A_GEO/BASELINE' and np.isclose(ratios['A_GEO/KIN_001']['u_MSE_ratio'],4.)
    assert ratios['C_GEO/KIN_001']['baseline']=='C_GEO/FIELD_ONLY' and np.isclose(ratios['C_GEO/KIN_001']['u_MSE_ratio'],16/9)
    assert all(Path(path).is_file() for path in r['figures'])


def test_partial_A_alias_uses_original_resumable_config_in_existing_runner(certificate,tmp_path,synthetic_runtime,monkeypatch):
    from pdeep.learning_diagnostics import suite
    from pdeep.smooth.training import implementation_identity
    p=plans(certificate,tmp_path)[0]
    # Existing runner's synthetic-event fixture only; NEVER constructs an optimizer.
    prepared=suite.prepare_diagnostic(model='A_GEO',experiment='wave_single',capacity='verification',updates=2,device='cpu',seed=42,
        certificate=certificate,root=tmp_path,persistence=p.persistence,run_name='older-A-wave',advanced={},query_mode='dense_resampled',query_count=32,
        query_time_policy='stratified',query_sampler_seed=42,query_chunk=17,query_checkpointing=False,amp_mode='off',memory_target_gb=11.,
        query_revision='continuous-queries-v3',variant_settings=p.settings.variant_settings)
    c=replace(prepared.configs[0],objective_variant='FULL')
    meta={'artifact_type':'training_checkpoint','architecture_id':'A_GEO','run_id':c.run_id,'config':asdict(c),'scientific_schema':SCHEMA,'benchmark':'G1_smooth_v1',
        'implementation_identity':implementation_identity(),'diagnostic_source_id':c.diagnostic_source_id,'execution_qualification':certificate,
        'fixed_data_identity':prepared.data.identity,'fixed_case':prepared.data.case.manifest,'runtime':{'dependencies':{'torch':torch.__version__}},'successful_update':1}
    payload={'model':prepared.models[0].state_dict(),'model_config':{},'optimizer':{'sentinel':8},'scheduler':{'sentinel':9},'rng':{'sentinel':10},'sampler':{},
        'normalizers':{'identity':certificate['normalizers']['identity']},'training_state':{'successful':1}}
    path=write_artifact(tmp_path/'old-synthetic-A','diagnostic-update-00000001',meta,payload)
    original=path.read_bytes();record=read_manifest(path)
    p.reused_runs={'BASELINE':{'status':'pending','checkpoint':str(path),'checkpoint_id':record['artifact_id'],'last_successful_update':1,
        'config_identity':c.identity,'run_id':c.run_id,'data_identity':prepared.data.identity,'case_id':prepared.data.case.case_id,
        'objective_alias':'A_FULL_is_BASELINE','reused_from':str(path),'initialization_identity':state_identity(prepared.models[0])}}
    p.wave_lookup='explicit';p.wave_resume_paths={'BASELINE':str(path)}
    monkeypatch.setattr('pdeep.learning_diagnostics.wave_study.recover_wave_checkpoints',lambda *a,**kw:{'selected':{},'missing':[]})
    report={'A_GEO':{'passed':True,'plan_identity':p.identity,'source':p.source}}
    result=run_wave_experiments((p,),report,run_mode='train',notebook=False)
    runs=result['architectures']['A_GEO']['entries']['wave_single']['runs']
    assert list(runs)==['BASELINE','KIN_001'] and result['session_optimizer_updates']==3
    assert runs['BASELINE']['result_origin']=='resumed' and runs['BASELINE']['session_optimizer_updates']==1
    latest,_=read_payload(runs['BASELINE']['checkpoint'])
    assert latest['config']['objective_variant']=='FULL' and latest['run_id']==c.run_id
    assert path.read_bytes()==original and not synthetic_runtime.failure['name']
