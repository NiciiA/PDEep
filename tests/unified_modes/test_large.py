"""Large actual networks, bounded CPU arithmetic; NOT accuracy/T4 certification."""
from dataclasses import asdict,replace
import json,time
from pathlib import Path
import numpy as np
import pytest,torch
from pdeep.smooth.data import Case
from pdeep.geometric.models import ModelConfig
from pdeep.unified_modes.architecture import model_config,inventory,verify_definition
from pdeep.unified_modes.model import create
from pdeep.unified_modes.preflight import preflight_batch
from pdeep.unified_modes.performance import accumulated_backward
from pdeep.unified_modes.training import Engine,inspect_checkpoint,train_models
from pdeep.unified_modes.inference import load,infer
from pdeep.unified_modes.dashboard import Dashboard
from pdeep.unified_modes.conditioning import FieldCondition,CaseView,MODES
from pdeep.unified_modes.evaluation import empty_observations
from pdeep.unified.data import Provider
from pdeep.unified.evaluation import predict
from pdeep.artifact_tensors import state_identity
from .conftest import config

RESULTS={}
@pytest.fixture(scope='module',autouse=True)
def report():
    start=time.perf_counter();yield
    p=Path('reports/g1_large');p.mkdir(parents=True,exist_ok=True)
    (p/'bounded_checks.json').write_text(json.dumps({'scope':'actual large CPU FP32 arithmetic checks; no neural training, reference solves or recertification','CUDA_available':torch.cuda.is_available(),'T4_peak_GB':None,'T4_step_seconds':None,'seconds':time.perf_counter()-start,'checks':RESULTS},indent=2)+'\n')

def test_actual_inventory_and_preserved_medium(dataset):
    rows=[];cs=[]
    for key in ('A_GEO/KIN_001','C_GEO/FIELD_ONLY','C_GEO/ALL_001'):
        m=create(key.split('/')[0],dataset.norms,'large');i=inventory(m);rows.append({'model':key,**i})
        assert i['total_parameters']==i['trainable_parameters']>=20_000_000
        assert model_config(m.config.architecture,'medium')==ModelConfig.preset(m.config.architecture,'medium')
        assert sum(i['components'].values())==i['total_parameters']
        if m.config.architecture=='C_GEO':cs.append((asdict(m.config),i['trainable_parameters'],state_identity(m)))
        assert m.config.width==384 and m.config.heads==8 and (m.config.coarse,m.config.fine)==(64,256)
    assert cs[0]==cs[1]
    # Same seeded initialization is an independent instance, never shared tensors.
    a=create('C_GEO',dataset.norms,'large');b=create('C_GEO',dataset.norms,'large')
    assert all(p.data_ptr()!=q.data_ptr() for p,q in zip(a.parameters(),b.parameters()))
    with pytest.raises(ValueError,match='20M'):inventory(__import__('pdeep.unified_modes.model',fromlist=['ConditionedModel']).ConditionedModel(replace(ModelConfig.preset('A_GEO','verification'),capacity='large'),dataset.norms))
    RESULTS['inventory']=rows

@pytest.mark.parametrize('arch',('A_GEO','C_GEO'))
def test_large_all_tasks_modes_and_chunk_masks(dataset,arch):
    model=create(arch,dataset.norms,'large');records=[];maximum=0.
    with torch.no_grad():
        for pde in ('poisson','heat','wave'):
            for family in ('square','rectangle','lshape','disk','ellipse','hole','annulus','polygon'):
                case=Case(family,pde,'train',0,.24)
                for mode in MODES:
                    b=preflight_batch(dataset,case,mode,4,'cpu');ctx=model.prepare_context(b.specs,b.observations);y=model.predict(ctx,b.queries)
                    assert y.shape==(1,4,1) and torch.isfinite(y).all()
                    assert b.cases[0].condition.values[~b.cases[0].condition.available].sum()==0
                    assert bool(b.queries.time_present.any())==(pde!='poisson')
                    records.append((pde,family,mode))
        case=Case('hole','wave','train',0,.12)
        for mode in MODES:
            b=preflight_batch(dataset,case,mode,8,'cpu')
            whole,_=predict(model,b.cases[0],b.observations,b.queries,chunk=8)
            chunked,_=predict(model,b.cases[0],b.observations,b.queries,chunk=3)
            error=float(abs(whole-chunked).max());maximum=max(error,maximum)
            assert error<2e-6*max(1.,float(abs(whole).max()))
            changed=preflight_batch(dataset,case,mode,8,'cpu')
            if mode!='FORWARD':
                changed.observations.value.add_(.1*dataset.norms.scale('wave',0))
                yo,_=predict(model,changed.cases[0],changed.observations,changed.queries,chunk=8)
                assert float(abs(whole-yo).max())>1e-7
            if mode!='INVERSE':
                f=b.cases[0].condition;new=replace(f,values=f.values+.1*f.available)
                ctx=model.prepare_context((CaseView(case,new,mode).problem,),b.observations);yk=model.predict(ctx,b.queries).numpy()[0,:,0]
                assert float(abs(whole-yk).max())>1e-7
    RESULTS[arch+'_categories']={'count':len(records),'scope':'all24 tasks ×3 modes; coarse conditioning mesh.24 arithmetic labels, no reference-accuracy claim','query_chunk_max_absolute_discrepancy':maximum,'hybrid_both_inputs_sensitive':True}

@pytest.mark.parametrize('key',('A_GEO/KIN_001','C_GEO/FIELD_ONLY','C_GEO/ALL_001'))
def test_large_real_four_case_backward(dataset,key,monkeypatch):
    def forbidden(*a,**kw):pytest.fail('optimizer construction in large preflight')
    monkeypatch.setattr(torch.optim.AdamW,'__init__',forbidden)
    cfg=config(dataset,key,capacity='large',query_count=8,query_chunk=5)
    model=create(cfg.architecture,dataset.norms,'large');before=state_identity(model)
    cases=[Case(g,p,'train',i,.12) for i,(g,p) in enumerate((('hole','wave'),('lshape','heat'),('polygon','poisson'),('annulus','wave')))]
    result=accumulated_backward(model,cfg,cases,lambda case,mode:preflight_batch(dataset,case,mode,cfg.query_count,'cpu'))
    assert result['status']=='fits' and result['optimizer_steps']==0 and result['effective_cases']==4
    assert result['inactive_submodules']==[] and state_identity(model)==before
    assert result['physics_forward_CPU_wall_enqueue_seconds']>0 if key!='C_GEO/FIELD_ONLY' else result['physics_forward_CPU_wall_enqueue_seconds']==0
    assert train_models((cfg,),None,'unused',run_mode='preview')['optimizer_steps']==0
    RESULT={k:v for k,v in result.items() if k not in ('active_parameter_names','inventory')};RESULT['active_parameter_tensors']=len(result['active_parameter_names']);RESULT['total_parameter_tensors']=len(tuple(model.parameters()))
    assert RESULT['active_parameter_tensors']==RESULT['total_parameter_tensors']
    RESULTS[key+'_accumulated_backward']=RESULT

def test_large_checkpoint_playground_and_wrong_capacity(dataset,certificate,tmp_path,monkeypatch):
    cfg=config(dataset,'C_GEO/ALL_001',capacity='large');engine=Engine(cfg,dataset,tmp_path,monitor=Dashboard(notebook=False))
    checkpoint=engine.save_checkpoint();r,_=inspect_checkpoint(checkpoint,cfg,dataset)
    assert r['parameter_counts']['trainable_parameters']==engine.model.parameter_count
    assert engine.monitor.latest['model_capacity']=='large'
    def forbidden(*a,**kw):pytest.fail('optimizer construction in playground')
    monkeypatch.setattr(torch.optim.AdamW,'__init__',forbidden)
    with pytest.raises(ValueError,match='capacity differs'):load(checkpoint,certificate,selection='latest',expected_capacity='medium')
    model,metadata=load(checkpoint,certificate,selection='latest',expected_capacity='large')
    assert state_identity(model)==state_identity(engine.model)
    assert metadata['architecture_inventory']['trainable_parameters']==model.parameter_count
    with pytest.raises(ValueError,match='definition differs'):verify_definition({**r,'model_config':asdict(model_config('C_GEO','medium'))},model)
    for mode in MODES:
        available=np.ones(64,bool) if mode=='FORWARD' else np.zeros(64,bool)
        if mode=='HYBRID':available[:32]=True
        result=infer(model,'wave','annulus',np.ones(64),available,mode=mode,grid_size=3,query_chunk=5)
        assert np.isfinite(result['prediction']).all()
    RESULTS['checkpoint']={'strict_large_reload':True,'trainable_parameters':model.parameter_count,'optimizer_updates':0,'wrong_capacity_rejected':True,'all_playground_modes':True,'exact_resume_state_checks':'also exercised on bounded verification-capacity engine; no research updates'}
