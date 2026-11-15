from dataclasses import replace
from pathlib import Path
import json,time
import numpy as np
import pytest,torch
from pdeep.artifact_io import read_payload
from pdeep.artifact_tensors import state_identity
from pdeep.unified.training import Engine,resume,train_models
from pdeep.unified_bayes.compatibility import approved_source,resume_configurations
from pdeep.unified_bayes.dashboard import Dashboard
from tests.unified.conftest import ArithmeticDataset,config
from .conftest import EVIDENCE


def test_real_updates_resume_monitor_and_no_rng(certificate,tmp_path):
    d=ArithmeticDataset(certificate);c=config(d,model='C_GEO/FIELD_ONLY');views=[];m=Dashboard(d.config,notebook=False,renderer=lambda s,h:views.append(json.loads(s)),warmup=1)
    e=Engine(c,d,tmp_path/'first',monitor=m);e.update();assert e.state.successful==1
    assert m.latest['successful']==1 and m.latest['unique_physical_cases']==4 and m.latest['cases_processed']==4
    assert m.latest['losses']['kinematic'].startswith('N/A') and m.latest['AMP_loss_scale'].startswith('N/A')
    assert m.latest['median_update_seconds']>0 and m.latest['current_model_ETA_seconds']>0
    assert m.latest['recent_PDE_weighted_objectives'] and sum(m.latest['recent_PDE_cases'].values())==4
    path=e.save_checkpoint();before=state_identity(e.model);r=resume(path,c,d,tmp_path/'resumed',monitor=Dashboard(d.config,notebook=False,warmup=1));assert state_identity(r.model)==before
    assert r.monitor.state_dict()['seen_case_ids']==m.state_dict()['seen_case_ids']
    assert r.state.successful==1 and r.scheduler.state_dict()==e.scheduler.state_dict()
    e.update();r.update();assert state_identity(r.model)==state_identity(e.model) and r.state.successful==2
    # Monitoring itself consumes no global training RNG.
    tr=torch.random.get_rng_state().clone();nr=np.random.get_state();m.phase('Preparing references');m.render();assert torch.equal(tr,torch.random.get_rng_state()) and np.array_equal(nr[1],np.random.get_state()[1])
    EVIDENCE['dashboard_real_steps']={'successful':r.state.successful,'exact_resume_model_identity':state_identity(r.model),'seen_case_ids':len(r.monitor.seen),'median_update_seconds':m.latest['median_update_seconds'],'AMP':'FP32, no scaler'}


def test_actual_predecessor_checkpoint(certificate,tmp_path):
    pointer=Path('/tmp/g1-bayes-legacy-manifest.txt')
    if not pointer.exists():pytest.skip('actual pre-edit checkpoint unavailable outside implementation session; generated exact-resume test remains portable')
    path=pointer.read_text().strip();record,payload=read_payload(path);d=ArithmeticDataset(certificate);c=config(d,'C_GEO/FIELD_ONLY');c=resume_configurations((c,),{c.model_key:path})[0]
    assert c.source_id==record['implementation_identity'] and approved_source(c.source_id)
    e=resume(path,c,d,tmp_path,monitor=Dashboard(notebook=False));assert all(torch.equal(e.model.state_dict()[k],v) for k,v in payload['model'].items())
    assert e.scheduler.state_dict()==payload['scheduler'] and e.state.successful==1
    for k,v in payload['optimizer']['state'].items():
        for name,a in v.items():assert torch.equal(a,e.optimizer.state_dict()['state'][k][name])
    assert len(e.monitor.seen)==4;e.update();assert e.state.successful==2
    assert not approved_source('unreviewed-source')
    with pytest.raises(ValueError,match='settings differ'):resume_configurations((replace(c,learning_rate=.01),),{c.model_key:path})
    EVIDENCE['actual_predecessor_resume']={'original_source':record['implementation_identity'],'model_and_Adam_restored_exactly':True,'scheduler_restored_exactly':True,'next_successful_update':2,'existing_checkpoint_modified':False}


def test_measured_eta_objectives_fallback_and_heartbeat(capsys):
    def failure(*a):raise RuntimeError('display fails')
    d=Dashboard(renderer=failure,heartbeat_seconds=.01)
    with pytest.raises(RuntimeError,match='training fixture'):
        with d.live():d.phase('Syncing Drive',model='C_GEO/ALL_001');time.sleep(.025);raise RuntimeError('training fixture')
    assert d.latest['failed_stage']=='Syncing Drive' and d.render_count>=3
    assert 'Syncing Drive' in capsys.readouterr().out
    d.phase('Completed',models=3);assert d.latest['completed_models']==3
    d.latest['median_update_seconds']=123.;d.phase('Model construction',model='C_GEO/ALL_001');assert 'median_update_seconds' not in d.latest
    d.objectives({'update':10,'microbatches':[{'per_case':[{'pde':'wave','weighted':{'field':2,'kinematic':.001}}]}]});assert d.latest['recent_PDE_weighted_objectives']=={'wave':2.001}
