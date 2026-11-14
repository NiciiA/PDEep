from dataclasses import replace,asdict
from pathlib import Path
import json,time
import numpy as np
import pytest
import torch
from pdeep.core import ValidationError,digest
from pdeep.artifact_io import read_payload,read_manifest
from pdeep.artifact_tensors import tree_identity,state_identity
from pdeep.artifact_drive import PersistencePolicy
from pdeep.unified.training import Engine,resume,inspect_checkpoint
from pdeep.unified.evaluation import aggregate,statistics,paired_intervals,evaluate_model
from pdeep.unified.monitor import Dashboard
from .conftest import config,EVIDENCE


def test_two_disposable_real_optimizer_updates_exact_resume(arithmetic_dataset,tmp_path):
    # Real existing verification network/AdamW/four-case accumulation, ZERO target
    # arithmetic fixture. NOT a trained scientific result/reference qualification.
    c=config(arithmetic_dataset,model='C_GEO/FIELD_ONLY');e=Engine(c,arithmetic_dataset,tmp_path/'local')
    first=e.update();assert first.successful==1 and first.counts['cases']==4
    path=e.save_checkpoint();record,p=read_payload(path);assert record['scientific_schema']=='G1-unified-multicase-v1'
    assert p['optimizer']['state'] and p['scheduler']['completed']==1 and p['scaler'] is None
    # Strict default PyTorch loader succeeds, without scoped NumPy allowlists.
    torch.load(Path(path).with_suffix('.pt'),weights_only=True)
    second=e.update();weights=state_identity(e.model);opt=digest(tree_identity(e.optimizer.state_dict()));order=e.state.rng_sampler_state
    r=resume(path,c,arithmetic_dataset,tmp_path/'resumed');again=r.update()
    assert state_identity(r.model)==weights and digest(tree_identity(r.optimizer.state_dict()))==opt
    assert r.state.rng_sampler_state==order and second.case_ids==again.case_ids and second.total==again.total
    assert r.state.successful==2 and r.state.cases_processed==8
    with pytest.raises(ValidationError,match='configuration'):
        inspect_checkpoint(path,replace(c,learning_rate=1e-4),arithmetic_dataset)
    with pytest.raises(ValidationError):e.save_checkpoint();e.save_checkpoint() # no clobber even identical update
    EVIDENCE['real_disposable_optimizer_updates']=3 # first + uninterrupted second + replayed second
    EVIDENCE['exact_resume_model_discrepancy']=0.;EVIDENCE['exact_resume_Adam_state_discrepancy']=0.


def test_planned_boundary_retains_full_schedule(arithmetic_dataset,tmp_path,monkeypatch):
    c=config(arithmetic_dataset);e=Engine(c,arithmetic_dataset,tmp_path)
    # No further optimizer work: emulate an already reached session boundary.
    e.state.successful=2;e.session_stop=2
    r=e.run_until(2,launch=True);assert r['status']=='already_at_requested_boundary'
    assert e.scheduler.config.updates==3 and e.config.updates==3
    with pytest.raises(ValidationError):e.run_until(3,launch=False)


def test_validation_aggregate_wave_not_double_weighted():
    rows=[{'pde':p,'quantity':q,'standardized_MSE':v} for p,q,v in [('poisson','u',1.),('heat','u',3.),('wave','u',5.),('wave','v',7.)]]
    result=aggregate(rows);assert result['selection_standardized_MSE']==pytest.approx(10/3)
    rows=[dict(r,case_id='case',family='square',model='A') for r in rows]
    more=[dict(r,model='C',standardized_MSE=r['standardized_MSE']+1) for r in rows]
    assert len(statistics(rows+more))==8
    assert all(r['mean_difference']==-1 and r['paired_bootstrap_95_CI'] is None for r in paired_intervals(rows+more))


def test_unseen_evaluation_no_update_restores_mode(arithmetic_dataset):
    from pdeep.geometric.models import create
    model=create('A_GEO',arithmetic_dataset.norms,'verification',seed=42);before=state_identity(model)
    r=evaluate_model(model,arithmetic_dataset,grid_size=None,count=8,diagnostics=False,max_cases=2)
    assert r['split']=='validation' and all(v['split']=='validation' for v in r['rows'])
    assert model.training and state_identity(model)==before
    with pytest.raises(ValidationError):evaluate_model(model,arithmetic_dataset,split='train')


def test_heartbeat_has_no_rng_or_mode_effects(capsys):
    rng=torch.random.get_rng_state();numpy=np.random.get_state();seen=[]
    m=Dashboard(notebook=False,renderer=lambda text,handle:seen.append(json.loads(text)),heartbeat_seconds=.01)
    with m.live():m.phase('Reference preparation',pde='wave');time.sleep(.04)
    assert len(seen)>=3 and torch.equal(rng,torch.random.get_rng_state()) and np.array_equal(numpy[1],np.random.get_state()[1])
    assert seen[-1]['stage']=='Reference preparation' and seen[-1]['whole_study_ETA_seconds'] is None
    def fail(*a):raise RuntimeError('display')
    with Dashboard(renderer=fail).live():pass
    assert 'UNIFIED MULTI-PDE' in capsys.readouterr().out


def test_real_zero_update_preflight_runs_all_objectives(arithmetic_dataset,tmp_path,monkeypatch):
    from pdeep.unified.preflight import run_preflight,require_preflight
    from pdeep.unified.config import MODELS
    def forbidden(*a,**kw):pytest.fail('preflight constructed an optimizer')
    monkeypatch.setattr(torch.optim.AdamW,'__init__',forbidden)
    configs=tuple(config(arithmetic_dataset,k) for k in MODELS)
    path,result=run_preflight(configs,arithmetic_dataset,tmp_path,progress=lambda *a,**k:None)
    assert result['optimizer_steps']==0 and len(result['reports'])==9
    assert all(r['performance']['optimizer_steps']==0 and r['geometry']=='annulus' for r in result['reports'])
    require_preflight(path,configs,arithmetic_dataset)
    with pytest.raises(ValidationError):require_preflight(path,(replace(configs[0],query_chunk=3),),arithmetic_dataset)
    EVIDENCE['preflight_zero_optimizer_updates']=0
    EVIDENCE['preflight_checked_model_PDE_pairs']=9
    EVIDENCE['preflight_cpu_elapsed_seconds']=result['seconds']


def test_training_driver_three_models_sequential_and_completed_skip(arithmetic_dataset,tmp_path,monkeypatch):
    import pdeep.unified.training as mod
    from pdeep.unified.config import MODELS
    configs=tuple(config(arithmetic_dataset,k) for k in MODELS);calls=[]
    monkeypatch.setattr('pdeep.unified.preflight.require_preflight',lambda *a:None) # orchestration fixture only
    class Stub:
        def __init__(self,c,*a,**kw):self.c=c;calls.append(c.model_key)
        def run_until(self,n,launch):return {'model':self.c.model_key,'successful':n,'latest':None,'persistence':{}}
    monkeypatch.setattr(mod,'Engine',Stub)
    m=Dashboard(notebook=False)
    r=mod.train_models(configs,arithmetic_dataset,tmp_path,run_mode='train',stop_after=2,monitor=m)
    assert list(r)==list(MODELS) and calls==list(MODELS) and all(v['successful']==2 for v in r.values())
    monkeypatch.setattr(mod,'inspect_checkpoint',lambda *a:({}, {'training_state':{'successful':3,'persistence':{},'best_checkpoint':None,'elapsed_seconds':0.}}))
    calls.clear();r=mod.train_models(configs,arithmetic_dataset,tmp_path,run_mode='train',stop_after=2,lookup='explicit',action='resume',resume_paths={k:'known-path.json' for k in MODELS},monitor=m)
    assert not calls and all(v['successful']==3 for v in r.values())


def test_family_holdout_excluded_from_checkpoint_selection(arithmetic_dataset):
    from pdeep.geometric.models import create
    arithmetic_dataset.config=replace(arithmetic_dataset.config,holdout_family='square')
    model=create('A_GEO',arithmetic_dataset.norms,'verification',seed=42)
    r=evaluate_model(model,arithmetic_dataset,grid_size=None,count=8,diagnostics=False,max_cases=2,familiar_only=True)
    assert all(v['family']!='square' for v in r['rows'])


def test_gaussian_geometry_coefficient_conditioning_against_covariance_form():
    from pdeep.unified.evaluation import gaussian_posterior
    rng=np.random.default_rng(7193);H=rng.normal(size=(7,12));y=rng.normal(size=7);sigma=rng.uniform(.02,.08,7)
    mean,cov,d=gaussian_posterior(H,y,sigma)
    sensor_cov=H@H.T+np.diag(sigma**2)
    independent_mean=H.T@np.linalg.solve(sensor_cov,y);independent_cov=np.eye(12)-H.T@np.linalg.solve(sensor_cov,H)
    err=max(float(np.max(abs(mean-independent_mean))),float(np.max(abs(cov-independent_cov))))
    assert err<1e-10 and d['relative_precision_residual']<1e-12
    m,C,_=gaussian_posterior(np.empty((0,12)),np.empty(0),np.empty(0));np.testing.assert_array_equal(m,0.);np.testing.assert_array_equal(C,np.eye(12))
    with pytest.raises(ValidationError,match='strictly positive'):gaussian_posterior(H,y,np.zeros(7))
    EVIDENCE['cached_Gaussian_information_covariance_max_discrepancy']=err


def test_3000_boundary_keeps_declared_30000_cosine_schedule(arithmetic_dataset):
    from types import SimpleNamespace
    from pdeep.training import SuccessfulScheduler
    import math
    c=replace(config(arithmetic_dataset),capacity='medium',preset='research',updates=30000,warmup_updates=250,scheduler='warmup_then_cosine')
    optimizer=SimpleNamespace(param_groups=[{'lr':0.}]);scheduler=SuccessfulScheduler(optimizer,c)
    scheduler.completed=3000;scheduler._set();snapshot=scheduler.state_dict()
    expected=c.learning_rate*(.1+.9*.5*(1+math.cos(math.pi*(3001-250)/(30000-250))))
    assert snapshot['next_lr']==pytest.approx(expected) and snapshot['next_lr']>c.learning_rate*.9
    restored=SuccessfulScheduler(SimpleNamespace(param_groups=[{'lr':0.}]),c);restored.load_state_dict(snapshot)
    assert restored.config.updates==30000 and restored.completed==3000
    EVIDENCE['pilot_boundary_full_schedule_next_LR']=snapshot['next_lr']
