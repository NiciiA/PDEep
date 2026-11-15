from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest,torch
from pdeep.smooth.data import Case
from pdeep.unified.data import Provider
from pdeep.unified.config import DatasetConfig
from pdeep.geometric.data import Batch
from pdeep.unified_modes.model import create
from pdeep.unified_modes.training import Engine,resume,finetune_inverse
from pdeep.unified_modes.evaluation import evaluate,evaluation_views,information_view
from pdeep.unified_modes.workflow import summaries
from pdeep.unified_modes.inference import load,infer,plot
from pdeep.unified_modes.loss import Adapter
from pdeep.unified_modes.preflight import preflight_batch,throughput_batch
from pdeep.unified_modes.dashboard import Dashboard
from pdeep.unified_modes.conditioning import FieldCondition,CaseView
from pdeep.artifact_tensors import state_identity
from tests.unified.conftest import ArithmeticDataset,config as old_config
from .conftest import config,EVIDENCE


def test_gradient_accumulation_case_and_quantity_weighting(dataset):
    cfg=config(dataset);a=create('C_GEO',dataset.norms,'verification');b=create('C_GEO',dataset.norms,'verification');sampler=__import__('pdeep.unified_modes.data',fromlist=['Sampler']).Sampler(cfg,dataset);batch,ids=sampler.batch(0)
    Adapter(cfg).loss(a,batch).total.backward()
    for i in range(4):(Adapter(cfg).loss(b,batch.slice(i,i+1)).total/4).backward()
    errors=[float((p.grad-q.grad).abs().max()) for p,q in zip(a.parameters(),b.parameters()) if p.grad is not None];maximum=max(errors);scale=max(float(p.grad.abs().max()) for p in a.parameters() if p.grad is not None);assert maximum<5e-6*max(1,scale)
    EVIDENCE['accumulation']={'maximum_gradient_discrepancy':maximum,'gradient_scale':scale,'effective_cases':4,'one_step_per_effective_batch':'inherited shared successful-update loop'}


def test_evaluation_playground_and_finetune(certificate,tmp_path,monkeypatch):
    d=ArithmeticDataset(certificate,DatasetConfig(preset='smoke',pdes=('poisson',),families=('hole',)));cfg=config(d);e=Engine(cfg,d,tmp_path/'new',monitor=Dashboard(notebook=False));path=e.save_checkpoint();model,meta=load(path,certificate,selection='latest');before=state_identity(model)
    def forbidden(*a,**kw):pytest.fail('evaluation constructed optimizer')
    with monkeypatch.context() as m:
        m.setattr(torch.optim.AdamW,'__init__',forbidden)
        result=evaluate(model,d,split='test',full=True,grid_size=4,max_cases_per_task=1,model_key=cfg.model_key,root=tmp_path/'evaluation',progress=lambda *a,**k:None)
        assert len(result['rows'])==42 and set(r['mode'] for r in result['rows'])=={'FORWARD','INVERSE','HYBRID'} and result['optimizer_updates']==0
        report=summaries(result['rows'],(cfg,));assert len(report['main'])==3
        f=infer(model,'poisson','hole',np.ones(32),np.ones(32,bool),mode='FORWARD',grid_size=4);assert f['status']=='fully specified';fig=plot(f);fig.savefig(tmp_path/'playground.png')
        h=infer(model,'poisson','hole',np.full(32,np.nan),np.zeros(32,bool),mode='INVERSE',grid_size=4);assert h['field_condition'].known_fraction==0
        assert state_identity(model)==before
    from pdeep.unified.training import Engine as OldEngine
    old=OldEngine(old_config(d,'C_GEO/FIELD_ONLY'),d,tmp_path/'old',monitor=Dashboard(notebook=False));oldpath=old.save_checkpoint();fine=Engine(replace(cfg,experiment_id='explicit-finetune',initialization='finetune_inverse'),d,tmp_path/'fine',monitor=Dashboard(notebook=False));transfer=finetune_inverse(oldpath,fine)
    assert transfer['new_parameter_names'] and not fine.optimizer.state and fine.state.successful==0
    with pytest.raises(ValueError,match='inverse-only'):load(oldpath,certificate,selection='latest')
    EVIDENCE['workflow']={'mode_views_tested':len(result['rows']),'optimizer_updates_during_evaluation':0,'known_field_playground':True,'inverse_only_rejected':True,'weights_only_finetune':True}


def test_field_token_padding(dataset):
    c=Case('disk','heat','train',0);f=FieldCondition.from_case(c,np.ones(32,bool));extra=7
    kwargs={}
    for name in ('quantity','component','centers','radius','values','applicable','available','valid'):
        a=getattr(f,name);pad=np.repeat(a[:1],extra,axis=0)
        if name in ('applicable','available','valid'):pad[:]=False
        if name=='values':pad[:]=np.nan
        kwargs[name]=np.concatenate((a,pad))
    padded=replace(f,**kwargs);model=create('A_GEO',dataset.norms,'verification');ot=__import__('pdeep.unified_modes.evaluation',fromlist=['empty_observations']).empty_observations(c,dataset.norms);qt=Provider(dataset).token_set(c,8,set_id=('unified-validation',42))[0]
    run=lambda x:model.predict(model.prepare_context((CaseView(c,x,'FORWARD').problem,),ot),qt)
    torch.testing.assert_close(run(f),run(padded),atol=2e-8,rtol=2e-5)


def test_real_small_forward_reference(certificate,tmp_path):
    from tests.unified_bayes.test_operators import control
    from pdeep.unified_bayes.operators import ForwardOperators,Coordinates
    from pdeep.unified_modes.bayes import conditional
    errors={}
    for pde in ('poisson','heat','wave'):
        c=Case('hole',pde,'validation',0);x=c.problem.mesh.sample(12,np.random.default_rng(3));t=np.tile([0.,.0137,.199,1.],3);q=np.arange(12)%2 if pde=='wave' else np.zeros(12,int);coords=Coordinates(x,None if pde=='poisson' else t,q)
        op=ForwardOperators(tmp_path/pde,certificate,_reference_factory=control,_control_identity='NONPRODUCTION-three-mode-reference-control-P2-h.16');op.prepare(pde,c.family,{'query':coords});B,offset,_=op.get(pde,c.family,coords);a=np.concatenate([c.fields[r].coefficients for r in ('u0','v0')]) if pde=='wave' else next(iter(c.fields.values())).coefficients
        post=conditional(np.empty((0,len(a))),[],[],a,np.ones(len(a),bool));pred=post.predict(B,offset=offset)
        fields={k:[v] for k,v in c.fields.items()};truth,_=op._forward(control(c.family),pde,fields,coords);err=float(abs(pred['mean']-truth[:,0]).max());assert err<1e-10;assert not pred['latent_variance'].any();errors[pde]=err
    EVIDENCE['fully_known_numerical_forward']={'geometry':'hole','small_P2_control':True,'physical_max_errors':errors,'uncertainty_exact_zero':True}


def test_sequential_shared_engine_segments_and_completed_skip(certificate,tmp_path,monkeypatch):
    from pdeep.unified_modes.preflight import run_preflight
    from pdeep.unified_modes.training import train_models
    d=ArithmeticDataset(certificate,DatasetConfig(preset='smoke',pdes=('poisson',),families=('square',),counts=(4,2,2)))
    configs=tuple(config(d,k,validation_every=1,checkpoint_every=1) for k in ('A_GEO/KIN_001','C_GEO/FIELD_ONLY','C_GEO/ALL_001'));m=Dashboard(d.config,notebook=False)
    pre,p=run_preflight(configs,d,tmp_path,monitor=m)
    r=train_models(configs,d,tmp_path/'runs',run_mode='train',stop_after=2,preflight_path=pre,monitor=m)
    assert list(r)==[c.model_key for c in configs] and all(v['successful']==2 for v in r.values()) and m.latest['completed_models']==3
    for v in r.values():
        assert len(list((Path(v['run_dir'])/'history-log-shards').glob('*steps.jsonl')))==2
        from pdeep.unified_modes.training import inspect_checkpoint
        c=next(c for c in configs if c.model_key==v['model']);record,p=inspect_checkpoint(v['latest'],c,d);assert len(p['history_refs'])==2
    def forbidden(*a,**kw):pytest.fail('completed model constructed optimizer')
    monkeypatch.setattr(torch.optim.AdamW,'__init__',forbidden)
    skip=train_models(configs,d,tmp_path/'runs',run_mode='train',action='resume',lookup='explicit',resume_paths={k:v['latest'] for k,v in r.items()},stop_after=2,preflight_path=pre,monitor=m)
    assert all(v['status']=='already_at_requested_boundary' for v in skip.values())
    EVIDENCE['sequential']={'models':list(r),'successful_arithmetic_updates_per_model':2,'completed_resume_optimizer_steps':0,'raw_log_rotation':'retained2 windows/model; verified incremental shards complete','bounded_report_segments':'existing run loop at validation/milestone boundaries'}


def test_production_poisson_conditional_evaluation(certificate,tmp_path):
    from pdeep.unified_bayes.operators import ForwardOperators,Coordinates
    from pdeep.unified_bayes.evaluation import observation_design,query_design
    from pdeep.unified_modes.classical import evaluate as evaluate_classical
    from pdeep.smooth.reference import reference
    d=ArithmeticDataset(certificate,DatasetConfig(preset='smoke',pdes=('poisson',),families=('square',)));c=d.cases('test')[0];op=ForwardOperators(tmp_path/'operators',certificate)
    designs={'query':Coordinates.design(query_design(c,4))}
    for n in (8,16,32,64,128):designs[str(n)]=Coordinates.design(observation_design(c,n,d.config.seed))
    prepared=op.prepare(c.pde,c.family,designs)
    class ActualCache:
        def __init__(self):self.memo={}
        def evaluate(self,c,x,t,q,**kw):
            from pdeep.unified.streams import coordinates_id
            key=coordinates_id(x,t,q)
            if key not in self.memo:self.memo[key]=reference(c.family,.02,4).poisson([c.fields['f']],x,8)[0][0]
            return self.memo[key]
    d.cache=ActualCache();result=evaluate_classical(d,op,tmp_path/'classical',grid_size=4,max_cases_per_task=1)
    assert len(result['rows'])==78 and len(result['operator_batches'])==1
    forward=next(r for r in result['rows'] if r['mode']=='FORWARD');assert forward['latent_variance_max']==0 and forward['maximum_physical_reference_discrepancy']<1e-12 and forward['calibration'] is None
    again=evaluate_classical(d,op,tmp_path/'classical',grid_size=4,max_cases_per_task=1);assert result['rows']==again['rows']
    EVIDENCE['production_baseline_control']={'PDE':'poisson','geometry':'square','physical_cases':1,'grid_points':len(designs['query'].position),'information_views':42,'rows_including_conditional_prior':78,'full_reference':'P4 h.02 quadrature8 CPU64','maximum_forward_error':forward['maximum_physical_reference_discrepancy'],'zero_forward_variance':True,'preprocessing_seconds':prepared['end_to_end_seconds'],'evaluation_seconds':result['evaluation_wall_seconds'],'scope':'bounded numerical control, not a full test-set/generalization result'}


def test_snapshot_scoped_reports_and_case_resume(certificate,tmp_path,monkeypatch):
    from pdeep.unified_modes.workflow import evaluate_checkpoints,combined
    from pdeep.artifact_io import read_payload
    d=ArithmeticDataset(certificate,DatasetConfig(preset='smoke',pdes=('poisson',),families=('hole',)));cfg=config(d);e=Engine(cfg,d,tmp_path/'trained-fixture',monitor=Dashboard(notebook=False));checkpoint=e.save_checkpoint()
    result=evaluate_checkpoints({cfg.model_key:str(checkpoint)},(cfg,),d,tmp_path/'evaluation',selection='latest',grid_size=4,max_cases_per_task=1,diagnostics=False)
    assert len(result['rows'])==42 and len(result['indices'])==1
    record,payload=read_payload(result['comparison_manifest'],kind='diagnostic_bundle');assert 'rows' not in payload['summary'] and payload['figure_png']
    index=__import__('json').loads(Path(result['indices'][0]).read_text());assert len(index['case_results'])==1 # one bundle, not42 small Drive copies.
    import pdeep.unified_modes.evaluation as evaluation_module
    def forbidden(*a,**k):pytest.fail('recomputed completed case')
    monkeypatch.setattr(evaluation_module,'predict',forbidden)
    again=evaluate_checkpoints({cfg.model_key:str(checkpoint)},(cfg,),d,tmp_path/'evaluation',selection='latest',grid_size=4,max_cases_per_task=1,diagnostics=False);assert result['rows']==again['rows']
    final=combined(result,None,(cfg,),tmp_path/'combined');assert read_payload(final['comparison_manifest'])[1]['summary']['main']
    assert final['report']['forward_hybrid_baseline_comparison']
    EVIDENCE['snapshot_reports']={'neural_information_views':42,'case_artifacts':1,'cached_prediction_calls':0,'summary_and_PNGs_verified':True,'scope':'untrained/arithmetic plotting and persistence control'}
