from dataclasses import replace
from pathlib import Path
from collections import Counter
import numpy as np
import pytest,torch
from pdeep.smooth.data import Case
from pdeep.unified.data import Provider
from pdeep.unified_modes.model import create
from pdeep.unified_modes.preflight import preflight_batch,run_preflight
from pdeep.unified_modes.conditioning import MODES,CaseView,FieldCondition
from pdeep.unified_modes.data import Sampler
from pdeep.unified_modes.loss import Adapter
from pdeep.unified_modes.training import Engine,resume,inspect_checkpoint,train_models,finetune_inverse
from pdeep.unified_modes.dashboard import Dashboard
from pdeep.artifact_tensors import state_identity
from .conftest import config,EVIDENCE

@pytest.mark.parametrize('model',('A_GEO/KIN_001','C_GEO/FIELD_ONLY','C_GEO/ALL_001'))
@pytest.mark.parametrize('mode',MODES)
@pytest.mark.parametrize('task',[(p,g) for p in ('poisson','heat','wave') for g in ('square','rectangle','lshape','disk','ellipse','hole','annulus','polygon')])
def test_72_categories_all_models_backward(dataset,model,mode,task):
    c=config(dataset,model);net=create(c.architecture,dataset.norms,'verification');case=Case(task[1],task[0],'train',0,.24);batch=preflight_batch(dataset,case,mode,4,'cpu');adapter=Adapter(c);r=adapter.loss(net,batch);r.total.backward()
    assert torch.isfinite(r.total) and any(p.grad is not None and p.grad.abs().max()>0 for p in net.known_field_encoder.parameters())
    if mode!='FORWARD':assert any(p.grad is not None and p.grad.abs().max()>0 for p in net.observation.parameters())
    assert ('kinematic' in r.weighted)==(model=='A_GEO/KIN_001' and task[0]=='wave')
    assert ('weak' in r.weighted)==(model=='C_GEO/ALL_001');assert ('initial' in r.weighted)==(model=='C_GEO/ALL_001' and task[0]!='poisson')
    EVIDENCE.setdefault('backward_categories',{}).setdefault(model,[]).append('/'.join((*task,mode)))


def test_sampling_balanced_and_no_hidden_values(dataset):
    c=config(dataset);s=Sampler(c,dataset);exposure=Counter()
    for cursor in range(60):
        for slot,key in enumerate(s.keys(cursor)):
            mode,mask,pattern,empty=s.plan(cursor,slot,key[0],key[1]);exposure[key[0],key[1],mode]+=1
            if mode=='FORWARD':assert mask.all() and empty
            elif mode=='INVERSE':assert not mask.any() and not empty
            else:assert 0<mask.mean()<1
    assert len(exposure)==72 and all(n==(4 if mode=='INVERSE' else 3) for (p,g,mode),n in exposure.items())
    b,ids=s.batch(0)
    for row,view in enumerate(b.cases):
        assert not view.condition.values[~view.condition.available].any()
        if view.mode=='FORWARD':assert not b.observations.valid[row].any() and not b.observations.value[row].any()
    assert [v.case_id for v in b.cases]==b.case_ids or tuple(v.case_id for v in b.cases)==b.case_ids
    EVIDENCE['mode_sampling']={'updates_window':60,'cases':240,'categories':72,'per_task_modes':{'FORWARD':3,'INVERSE':4,'HYBRID':3},'global_RNG':'counter keyed; no stochastic global state needed'}


def test_exact_resume_and_milestone_scheduler(dataset,tmp_path):
    c=config(dataset);m=Dashboard(dataset.config,notebook=False);e=Engine(c,dataset,tmp_path/'run',monitor=m);e.update();path=e.save_checkpoint();r=resume(path,c,dataset,tmp_path/'resume',monitor=Dashboard(dataset.config,notebook=False));assert state_identity(e.model)==state_identity(r.model)
    assert e.sampler.state_dict(1)==r.sampler.state_dict(1)
    e.update();r.update();assert state_identity(e.model)==state_identity(r.model);assert e.sampler.plan(2,0,'wave','hole')[0]==r.sampler.plan(2,0,'wave','hole')[0]
    with pytest.raises(ValueError,match='horizon'):inspect_checkpoint(path,replace(c,updates=50),dataset)
    # Learning rate is fixed by the declared100k horizon; a50k session boundary never mutates it.
    from pdeep.training import SuccessfulScheduler
    long=config(dataset,preset='research',capacity='medium',updates=100000,main_target=50000,warmup_updates=250,scheduler='warmup_then_cosine');p=torch.nn.Parameter(torch.zeros(()));opt=torch.optim.AdamW([p],lr=long.learning_rate);scheduler=SuccessfulScheduler(opt,long);scheduler.completed=50000;scheduler._set();lr50=opt.param_groups[0]['lr'];state=scheduler.state_dict();clone=SuccessfulScheduler(opt,long);clone.load_state_dict(state);assert clone.state_dict()==state;scheduler.completed=100000;scheduler._set();assert abs(opt.param_groups[0]['lr']-3e-5)<1e-15
    EVIDENCE['resume']={'successful_updates':2,'bitwise_resumed_model':True,'mode_order_preserved':True,'LR_at_50k_in_100k_schedule':lr50,'LR_at_100k':opt.param_groups[0]['lr'],'schedule_stretch_rejected':True}


def test_preview_preflight_no_optimizer(dataset,tmp_path,monkeypatch):
    configs=tuple(config(dataset,k) for k in ('A_GEO/KIN_001','C_GEO/FIELD_ONLY','C_GEO/ALL_001'))
    def forbidden(*a,**kw):pytest.fail('unexpected optimizer in preview/preflight')
    monkeypatch.setattr(torch.optim.AdamW,'__init__',forbidden);r=train_models(configs,None,tmp_path,run_mode='preview');assert r['optimizer_steps']==0
    path,p=run_preflight(configs,dataset,tmp_path,monitor=Dashboard(dataset.config,notebook=False));assert p['optimizer_steps']==0 and len(p['reports'])==27
    EVIDENCE['preflight']={'model_PDE_mode_combinations':27,'optimizer_updates':0,'seconds':p['seconds'],'device':'CPU','T4':'not available; actual configured CUDA preflight remains required'}
